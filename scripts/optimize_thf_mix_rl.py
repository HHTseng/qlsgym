#!/usr/bin/env python
"""Two-stage Optuna search for RL agents trained with the fixed ThF+ mix FNO.

The broad study uses only FNO validation trajectories.  The best configurations
are then retrained for one million transitions with separate promotion seeds.
The selected configuration for each agent is finally retrained with seeds 0--4
and evaluated on both the FNO and exact dynamics with the locked holdout seed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import torch

from qlsgym.rl.off_policy import (
    DDQNAgent,
    DDQNConfig,
    DiscreteSACAgent,
    SACConfig,
    ddqn_policy,
    sac_policy,
)
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, git_sha, json_safe, rollout_metrics, write_json


AGENTS = ("ppo", "sac_discrete", "ddqn")
DEFAULT_OUTPUT = Path("results/thf_rl_optuna_mix")
DEFAULT_MANIFEST = Path(os.environ.get("QLSGYM_WORK", "~/qlsgym_work")).expanduser() / "checkpoints/thf/mix.json"
BROAD_EVAL_SEED = 31_001
PROMOTION_EVAL_SEED = 32_001
FINAL_EVAL_SEED = 20_001


def compact_metrics(metrics: dict) -> dict:
    drop = {"lengths", "actual_lengths", "successes"}
    return {key: value for key, value in metrics.items() if key not in drop}


def objective_score(metrics: dict) -> float:
    """Success is primary; the small second term breaks ties by pulse cost."""
    success = float(metrics["success_fraction"])
    efficiency = 1.0 - float(metrics["average_actions"]) / float(metrics["max_pulses"])
    return success + 0.02 * efficiency


def storage(output: Path) -> optuna.storages.RDBStorage:
    path = (output / "optuna.sqlite3").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        f"sqlite:///{path}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def study_name(agent: str) -> str:
    return f"thf_mix_{agent}"


def load_study(output: Path, agent: str, worker_id: int = 0, create: bool = True):
    sampler = optuna.samplers.TPESampler(
        seed=73_000 + worker_id,
        n_startup_trials=20,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    if create:
        return optuna.create_study(
            study_name=study_name(agent),
            storage=storage(output),
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    return optuna.load_study(study_name=study_name(agent), storage=storage(output))


def env_namespace(args, device: str):
    return argparse.Namespace(
        device=device,
        p_target=args.p_target,
        max_pulses=args.max_pulses,
        rho=args.rho,
        progress=False,
        fno_tag="mix",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )


def suggest_config(trial: optuna.Trial, agent: str, steps: int, snapshot_eval: int, horizon: int):
    seed = 10_000 + trial.number
    if agent == "ppo":
        n_envs = 128
        n_steps = trial.suggest_categorical("n_steps", [16, 32, 64])
        updates = max(1, math.ceil(steps / (n_envs * n_steps)))
        reward_factor = trial.suggest_categorical("reward_scale_factor", [0.5, 1.0, 2.0])
        return PPOConfig(
            n_envs=n_envs,
            n_steps=n_steps,
            total_steps=steps,
            epochs=trial.suggest_categorical("epochs", [2, 4, 8]),
            minibatches=trial.suggest_categorical("minibatches", [4, 8, 16]),
            lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            gamma=trial.suggest_categorical("gamma", [0.97, 0.99, 0.995, 1.0]),
            gae_lambda=trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 0.995]),
            clip=trial.suggest_categorical("clip", [0.1, 0.2, 0.3]),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
            vf_coef=trial.suggest_categorical("vf_coef", [0.25, 0.5, 1.0]),
            max_grad_norm=trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0]),
            hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
            n_hidden_layers=trial.suggest_int("n_hidden_layers", 1, 3),
            obs=trial.suggest_categorical("obs", ["p", "sqrt"]),
            value_target=trial.suggest_categorical("value_target", ["gae", "qmdp", "qmdp_gae"]),
            reward_scale=reward_factor / horizon,
            eval_every=max(1, updates // 4),
            eval_rollouts=snapshot_eval,
            eval_greedy=False,
            seed=seed,
        )
    if agent == "ddqn":
        return DDQNConfig(
            n_envs=128,
            total_steps=steps,
            lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            gamma=trial.suggest_categorical("gamma", [0.97, 0.99, 0.995, 1.0]),
            tau=trial.suggest_float("tau", 1e-4, 5e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            buffer_size=max(50_000, steps // 2),
            learning_starts=trial.suggest_categorical(
                "learning_starts", [2_048, 5_120, 10_240, 20_480]
            ),
            train_freq=trial.suggest_categorical("train_freq", [1, 2]),
            gradient_steps=trial.suggest_categorical("gradient_steps", [1, 2, 4]),
            eps_start=1.0,
            eps_end=trial.suggest_float("eps_end", 0.005, 0.15, log=True),
            eps_fraction=trial.suggest_float("eps_fraction", 0.2, 1.0),
            hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
            depth=trial.suggest_int("depth", 1, 3),
            seed=seed,
        )
    return SACConfig(
        n_envs=128,
        total_steps=steps,
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        gamma=trial.suggest_categorical("gamma", [0.95, 0.97, 0.99, 0.995]),
        tau=trial.suggest_float("tau", 1e-4, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        buffer_size=max(50_000, steps // 2),
        learning_starts=trial.suggest_categorical(
            "learning_starts", [2_048, 5_120, 10_240, 20_480]
        ),
        train_freq=trial.suggest_categorical("train_freq", [1, 2]),
        gradient_steps=trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        target_entropy_ratio=trial.suggest_float("target_entropy_ratio", 0.05, 0.95),
        alpha=trial.suggest_float("alpha_raw", 0.002, 0.30, log=True) / horizon,
        autotune_alpha=trial.suggest_categorical("autotune_alpha", [True, False]),
        hidden=trial.suggest_categorical("hidden", [128, 256, 512]),
        depth=trial.suggest_int("depth", 1, 3),
        seed=seed,
    )


def broad_objective(trial, agent_name, fno, args):
    started = time.perf_counter()
    config = suggest_config(
        trial, agent_name, args.broad_steps, args.snapshot_eval, args.max_pulses
    )
    trial.set_user_attr("config", asdict(config))
    trial.set_user_attr("train_seed", config.seed)
    report_index = 0

    def report(policy, metrics=None):
        nonlocal report_index
        if metrics is None:
            metrics = rollout_metrics(
                fno, policy, args.snapshot_eval, 42_420 + report_index, args.eval_batch
            )
        report_index += 1
        score = objective_score(metrics)
        trial.report(score, report_index)
        trial.set_user_attr("last_snapshot", compact_metrics(metrics))
        if trial.should_prune():
            raise optuna.TrialPruned(f"pruned after fidelity rung {report_index}")

    try:
        if agent_name == "ppo":
            def on_snapshot(snapshot, net):
                metrics = {
                    "success_fraction": snapshot["success"],
                    "average_actions": snapshot["mean_pulses"],
                    "max_pulses": args.max_pulses,
                }
                report(None, metrics)

            result = train_ppo(fno, config, env_eval=fno, on_snapshot=on_snapshot)
            state = result.best_state_dict or result.final_state_dict
            policy = policy_from_state_dict(
                state, result.n_in, result.n_actions, config,
                device=args.device, greedy=False,
            )
        elif agent_name == "ddqn":
            trained = DDQNAgent(fno, config)
            trained.train(
                log_points=4,
                on_log=lambda model, _: report(ddqn_policy(model)),
            )
            policy = ddqn_policy(trained)
        else:
            trained = DiscreteSACAgent(fno, config)
            trained.train(
                log_points=4,
                on_log=lambda model, _: report(sac_policy(model, stochastic=True)),
            )
            policy = sac_policy(trained, stochastic=True)

        metrics = rollout_metrics(
            fno, policy, args.broad_eval, BROAD_EVAL_SEED, args.eval_batch
        )
        score = objective_score(metrics)
        trial.set_user_attr("fno_validation", compact_metrics(metrics))
        trial.set_user_attr("wall_clock_s", time.perf_counter() - started)
        return score
    finally:
        torch.cuda.empty_cache()


def optimize(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = args.device
    _, _, fno, _, _, contract = build_environments(env_namespace(args, device), 128)
    write_json(output / "contract.json", contract)
    agents = tuple(item.strip() for item in args.agents.split(",") if item.strip())
    unknown = set(agents) - set(AGENTS)
    if unknown:
        raise ValueError(f"unknown agents: {sorted(unknown)}")
    for agent_name in agents:
        study = load_study(output, agent_name, args.worker_id)
        print(
            f"worker {args.worker_id}: {agent_name}, {args.trials_per_agent} trials, "
            f"device={device}", flush=True,
        )
        study.optimize(
            lambda trial: broad_objective(trial, agent_name, fno, args),
            n_trials=args.trials_per_agent,
            gc_after_trial=True,
            catch=(RuntimeError,),
        )


def init_studies(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for agent_name in AGENTS:
        load_study(output, agent_name, worker_id=0, create=True)
    print(f"initialized {len(AGENTS)} studies under {output}")


def completed_trials(study):
    return [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def prepare(args):
    output = Path(args.output)
    promotion_seeds = [int(value) for value in args.promotion_seeds.split(",")]
    jobs = []
    selected = {}
    for agent_name in AGENTS:
        study = load_study(output, agent_name, create=False)
        ranked = sorted(completed_trials(study), key=lambda item: item.value, reverse=True)
        if len(ranked) < args.top_k:
            raise RuntimeError(
                f"{agent_name} has only {len(ranked)} complete trials; need {args.top_k}"
            )
        selected[agent_name] = []
        for rank, trial in enumerate(ranked[:args.top_k], start=1):
            config_id = f"{agent_name}_t{trial.number}"
            entry = {
                "agent": agent_name,
                "config_id": config_id,
                "broad_rank": rank,
                "trial_number": trial.number,
                "broad_score": trial.value,
                "config": trial.user_attrs["config"],
            }
            selected[agent_name].append(entry)
            for seed in promotion_seeds:
                jobs.append({**entry, "stage": "promotion", "train_seed": seed})
    write_json(output / "broad_finalists.json", selected)
    write_json(output / "promotion_jobs.json", jobs)
    print(f"prepared {len(jobs)} promotion jobs")


def full_config(agent_name: str, raw: dict, seed: int, steps: int, eval_rollouts: int):
    config = dict(raw)
    config["seed"] = seed
    config["total_steps"] = steps
    if agent_name == "ppo":
        updates = max(1, math.ceil(steps / (config["n_envs"] * config["n_steps"])))
        config["eval_every"] = updates
        config["eval_rollouts"] = eval_rollouts
        config["eval_greedy"] = False
        return PPOConfig(**config)
    config["buffer_size"] = max(50_000, steps // 2)
    if agent_name == "ddqn":
        return DDQNConfig(**config)
    return SACConfig(**config)


def save_model(path: Path, agent_name: str, config, trained, state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if agent_name == "ppo":
        payload = {"agent": agent_name, "config": asdict(config), "state_dict": state}
    elif agent_name == "ddqn":
        payload = {
            "agent": agent_name,
            "config": asdict(config),
            "state_dict": trained.online.state_dict(),
        }
    else:
        payload = {
            "agent": agent_name,
            "config": asdict(config),
            "actor_state_dict": trained.network.actor.state_dict(),
            "critic1_state_dict": trained.network.q1.state_dict(),
            "critic2_state_dict": trained.network.q2.state_dict(),
            "alpha": float(trained.network.log_alpha.detach().exp()),
        }
    torch.save(payload, path)


def run_full_job(job, args, fno, exact, engine, contract):
    started = time.time()
    agent_name = job["agent"]
    stage = job["stage"]
    episodes = args.promotion_eval if stage == "promotion" else args.final_eval
    eval_seed = PROMOTION_EVAL_SEED if stage == "promotion" else FINAL_EVAL_SEED
    config = full_config(
        agent_name, job["config"], job["train_seed"], args.full_steps,
        min(args.snapshot_eval, episodes),
    )
    if agent_name == "ppo":
        result = train_ppo(fno, config, env_eval=fno, log=print)
        state = result.best_state_dict or result.final_state_dict
        policy = policy_from_state_dict(
            state, result.n_in, result.n_actions, config,
            device=args.device, greedy=False,
        )
        training = result.as_dict()
        trained = None
    elif agent_name == "ddqn":
        trained = DDQNAgent(fno, config)
        stats = trained.train()
        policy = ddqn_policy(trained)
        training = {**stats.as_dict(), "config": asdict(config)}
        state = None
    else:
        trained = DiscreteSACAgent(fno, config)
        stats = trained.train()
        policy = sac_policy(trained, stochastic=True)
        training = {**stats.as_dict(), "config": asdict(config)}
        state = None

    exact_metrics = rollout_metrics(exact, policy, episodes, eval_seed, args.eval_batch)
    fno_metrics = rollout_metrics(fno, policy, episodes, eval_seed, args.eval_batch)
    model_path = None
    if stage == "final":
        model_path = Path(args.output) / "models" / (
            f"{agent_name}_{job['config_id']}_s{job['train_seed']}.pt"
        )
        save_model(model_path, agent_name, config, trained, state)
    return {
        "schema_version": 1,
        "job": job,
        "contract": contract,
        "training": training,
        "evaluation": {"exact": exact_metrics, "fno": fno_metrics},
        "selection_score_fno": objective_score(fno_metrics),
        "surrogate_calls": dict(engine.calls),
        "surrogate_fraction": engine.surrogate_fraction(),
        "model_path": None if model_path is None else str(model_path),
        "wall_clock_s": time.time() - started,
        "provenance": {
            "git_sha": git_sha(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "optuna": optuna.__version__,
            "host": socket.gethostname(),
            "device": args.device,
        },
    }


def result_path(output: Path, job: dict) -> Path:
    return output / job["stage"] / (
        f"{job['agent']}_{job['config_id']}_s{job['train_seed']}.json"
    )


def run_queue(args):
    output = Path(args.output)
    jobs_path = output / f"{args.stage}_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    _, _, fno, exact, engine, contract = build_environments(env_namespace(args, args.device), 128)
    failures = 0
    for job in jobs:
        destination = result_path(output, job)
        if destination.exists():
            continue
        claim = destination.with_suffix(destination.suffix + ".claim")
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"worker={args.worker_id} pid={os.getpid()} time={time.time()}\n")
        print(f"worker {args.worker_id}: {job}", flush=True)
        try:
            record = run_full_job(job, args, fno, exact, engine, contract)
            write_json(destination, record)
        except Exception as error:
            failures += 1
            write_json(
                destination.with_suffix(".error.json"),
                {"job": job, "error": repr(error), "traceback": traceback.format_exc()},
            )
        finally:
            claim.unlink(missing_ok=True)
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"worker {args.worker_id} had {failures} failed jobs")


def load_stage(output: Path, stage: str):
    records = []
    for path in sorted((output / stage).glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        records.append(json.loads(path.read_text()))
    return records


def select(args):
    output = Path(args.output)
    records = load_stage(output, "promotion")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["job"]["agent"], record["job"]["config_id"])].append(record)
    seeds = [int(value) for value in args.final_seeds.split(",")]
    choices = {}
    jobs = []
    for agent_name in AGENTS:
        candidates = []
        for (candidate_agent, config_id), group in grouped.items():
            if candidate_agent != agent_name:
                continue
            score = float(np.mean([item["selection_score_fno"] for item in group]))
            candidates.append((score, config_id, group))
        if not candidates:
            raise RuntimeError(f"no promotion records for {agent_name}")
        score, config_id, group = max(candidates, key=lambda item: item[0])
        template = group[0]["job"]
        choices[agent_name] = {
            "config_id": config_id,
            "promotion_score_fno": score,
            "promotion_runs": len(group),
            "trial_number": template["trial_number"],
            "config": template["config"],
        }
        for seed in seeds:
            jobs.append({
                "agent": agent_name,
                "config_id": config_id,
                "trial_number": template["trial_number"],
                "config": template["config"],
                "stage": "final",
                "train_seed": seed,
            })
    write_json(output / "selected_configs.json", choices)
    write_json(output / "final_jobs.json", jobs)
    print(f"selected one configuration per agent; prepared {len(jobs)} final jobs")


def aggregate_metrics(records, dynamics):
    metrics = [record["evaluation"][dynamics] for record in records]
    result = {}
    for key in ("success_fraction", "unfinished_fraction", "average_actions",
                "mean_pulses_successful", "p85_actions", "cvar10_actions"):
        values = [item[key] for item in metrics if item.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def baseline_rows(path: Path):
    data = json.loads(path.read_text())
    return {row["agent"]: row for row in data["rows"] if row["agent"] in AGENTS}


def make_figures(output: Path, studies, rows, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for axis, agent_name in zip(axes, AGENTS):
        trials = completed_trials(studies[agent_name])
        x = [trial.number for trial in trials]
        y = [trial.value for trial in trials]
        order = np.argsort(x)
        x = np.asarray(x)[order]
        y = np.asarray(y)[order]
        axis.scatter(x, y, s=14, alpha=0.45, color="#4c78a8", label="completed trial")
        if len(y):
            axis.plot(x, np.maximum.accumulate(y), color="#e45756", lw=2, label="best so far")
        axis.set_title(agent_name)
        axis.set_xlabel("trial")
        axis.set_ylabel("FNO selection score")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output / "optuna_history.png", dpi=180)
    plt.close(fig)

    labels = ["PPO", "SAC", "DDQN"]
    x = np.arange(len(AGENTS))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for offset, source, dynamics, color, hatch in (
        (-1.5, "baseline", "exact", "#b8c4ce", ""),
        (-0.5, "optimized", "exact", "#4c78a8", ""),
        (0.5, "baseline", "fno", "#f2c39b", "//"),
        (1.5, "optimized", "fno", "#f58518", "//"),
    ):
        if source == "baseline":
            failure = [100 * baseline[a][dynamics]["unfinished_fraction"] for a in AGENTS]
            actions = [baseline[a][dynamics]["average_actions"] for a in AGENTS]
        else:
            failure = [100 * rows[a][dynamics]["unfinished_fraction"] for a in AGENTS]
            actions = [rows[a][dynamics]["average_actions"] for a in AGENTS]
        name = f"{source} {dynamics}"
        axes[0].bar(x + offset * width, failure, width, label=name, color=color, hatch=hatch)
        axes[1].bar(x + offset * width, actions, width, label=name, color=color, hatch=hatch)
    axes[0].set_ylabel("unfinished episodes (%)")
    axes[1].set_ylabel("penalized average actions")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "optimized_vs_baseline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, agent_name in zip(axes, AGENTS):
        try:
            importance = optuna.importance.get_param_importances(studies[agent_name])
        except Exception:
            importance = {}
        items = list(importance.items())[:10][::-1]
        if items:
            axis.barh([item[0] for item in items], [item[1] for item in items], color="#72b7b2")
        axis.set_title(agent_name)
        axis.set_xlabel("fANOVA importance")
        axis.grid(axis="x", alpha=0.25)
    fig.savefig(output / "parameter_importance.png", dpi=180)
    plt.close(fig)


def summarize(args):
    output = Path(args.output)
    final = load_stage(output, "final")
    grouped = defaultdict(list)
    for record in final:
        grouped[record["job"]["agent"]].append(record)
    missing = [agent for agent in AGENTS if len(grouped[agent]) < args.expected_final_seeds]
    if missing:
        raise RuntimeError(f"incomplete final records for: {missing}")
    baseline = baseline_rows(Path(args.baseline))
    studies = {agent: load_study(output, agent, create=False) for agent in AGENTS}
    choices = json.loads((output / "selected_configs.json").read_text())
    rows = {}
    for agent_name in AGENTS:
        exact = aggregate_metrics(grouped[agent_name], "exact")
        fno = aggregate_metrics(grouped[agent_name], "fno")
        base = baseline[agent_name]
        rows[agent_name] = {
            "config_id": choices[agent_name]["config_id"],
            "trial_number": choices[agent_name]["trial_number"],
            "training_seeds": len(grouped[agent_name]),
            "exact": exact,
            "fno": fno,
            "improvement_vs_locked_baseline": {
                "exact_failure_percentage_points": 100 * (
                    base["exact"]["unfinished_fraction"] - exact["unfinished_fraction"]
                ),
                "exact_average_actions": (
                    base["exact"]["average_actions"] - exact["average_actions"]
                ),
                "fno_failure_percentage_points": 100 * (
                    base["fno"]["unfinished_fraction"] - fno["unfinished_fraction"]
                ),
                "fno_average_actions": (
                    base["fno"]["average_actions"] - fno["average_actions"]
                ),
            },
            "config": choices[agent_name]["config"],
        }
    broad = {}
    for agent_name, study in studies.items():
        counts = Counter(trial.state.name.lower() for trial in study.trials)
        broad[agent_name] = {
            "trials": len(study.trials),
            "states": dict(counts),
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
        }
    report = {
        "status": "complete",
        "selection_rule": (
            "Optuna and promotion use FNO success plus 0.02 times normalized pulse efficiency; "
            "exact dynamics are an audit and the final exact holdout is not used for selection."
        ),
        "budget": {
            "broad_steps": args.broad_steps,
            "full_steps": args.full_steps,
            "broad_eval_seed": BROAD_EVAL_SEED,
            "promotion_eval_seed": PROMOTION_EVAL_SEED,
            "final_eval_seed": FINAL_EVAL_SEED,
        },
        "broad_studies": broad,
        "rows": rows,
    }
    write_json(output / "summary.json", report)

    lines = [
        "# ThF+ downloaded-mix FNO RL optimization",
        "",
        "The FNO checkpoints and action library were fixed. Optuna used only FNO validation "
        "during the broad search. Three configurations per agent were promoted, each with two "
        "new training seeds. The selected configuration was retrained with five seeds and "
        "evaluated on 5,000 FNO and 5,000 exact holdout episodes per seed.",
        "",
        "| Agent | Exact failure baseline | Exact failure optimized | Change | "
        "Exact actions baseline | Exact actions optimized | Change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_name in AGENTS:
        row = rows[agent_name]
        base = baseline[agent_name]["exact"]
        delta = row["improvement_vs_locked_baseline"]
        lines.append(
            f"| {agent_name} | {100 * base['unfinished_fraction']:.2f}% | "
            f"{100 * row['exact']['unfinished_fraction']:.2f}% | "
            f"{delta['exact_failure_percentage_points']:+.2f} pp better | "
            f"{base['average_actions']:.2f} | {row['exact']['average_actions']:.2f} | "
            f"{delta['exact_average_actions']:+.2f} better |"
        )
    lines += [
        "",
        "| Agent | FNO failure baseline | FNO failure optimized | Change | "
        "FNO actions baseline | FNO actions optimized | Change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_name in AGENTS:
        row = rows[agent_name]
        base = baseline[agent_name]["fno"]
        delta = row["improvement_vs_locked_baseline"]
        lines.append(
            f"| {agent_name} | {100 * base['unfinished_fraction']:.2f}% | "
            f"{100 * row['fno']['unfinished_fraction']:.2f}% | "
            f"{delta['fno_failure_percentage_points']:+.2f} pp better | "
            f"{base['average_actions']:.2f} | {row['fno']['average_actions']:.2f} | "
            f"{delta['fno_average_actions']:+.2f} better |"
        )
    lines += [
        "",
        "## Selected configurations",
        "",
    ]
    for agent_name in AGENTS:
        lines += [
            f"### {agent_name}",
            "",
            f"Broad trial `{rows[agent_name]['trial_number']}` / "
            f"`{rows[agent_name]['config_id']}`:",
            "",
            "```json",
            json.dumps(rows[agent_name]["config"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    lines += [
        "## Figures",
        "",
        "- `optuna_history.png`: every completed broad trial and the best-so-far curve.",
        "- `parameter_importance.png`: fANOVA importance for the ten most influential settings.",
        "- `optimized_vs_baseline.png`: locked baseline versus optimized final evaluation.",
        "",
        "Positive changes in the tables mean improvement. Failed episodes are charged the "
        "full 80-action horizon, so average actions jointly reflects success and speed.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    make_figures(output, studies, rows, baseline)
    print(f"wrote {output / 'summary.md'}")


def status(args):
    output = Path(args.output)
    result = {"output": str(output.resolve())}
    for agent_name in AGENTS:
        try:
            study = load_study(output, agent_name, create=False)
        except Exception:
            continue
        result[agent_name] = {
            "states": dict(Counter(trial.state.name.lower() for trial in study.trials)),
            "best_value": study.best_value if completed_trials(study) else None,
        }
    for stage in ("promotion", "final"):
        jobs = output / f"{stage}_jobs.json"
        if jobs.exists():
            planned = len(json.loads(jobs.read_text()))
            complete = len(load_stage(output, stage))
            errors = len(list((output / stage).glob("*.error.json")))
            result[stage] = {"planned": planned, "complete": complete, "errors": errors}
    result["summary_complete"] = (output / "summary.json").exists()
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


def add_environment(parser):
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p-target", type=float, default=0.98)
    parser.add_argument("--max-pulses", type=int, default=80)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--snapshot-eval", type=int, default=256)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.set_defaults(func=init_studies)

    command = commands.add_parser("optimize")
    add_environment(command)
    command.add_argument("--agents", default=",".join(AGENTS))
    command.add_argument("--worker-id", type=int, required=True)
    command.add_argument("--trials-per-agent", type=int, default=40)
    command.add_argument("--broad-steps", type=int, default=250_000)
    command.add_argument("--broad-eval", type=int, default=768)
    command.set_defaults(func=optimize)

    command = commands.add_parser("prepare")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.add_argument("--top-k", type=int, default=3)
    command.add_argument("--promotion-seeds", default="100,101")
    command.set_defaults(func=prepare)

    command = commands.add_parser("run-queue")
    add_environment(command)
    command.add_argument("--stage", choices=("promotion", "final"), required=True)
    command.add_argument("--worker-id", type=int, required=True)
    command.add_argument("--full-steps", type=int, default=1_000_000)
    command.add_argument("--promotion-eval", type=int, default=2_000)
    command.add_argument("--final-eval", type=int, default=5_000)
    command.set_defaults(func=run_queue)

    command = commands.add_parser("select")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.add_argument("--final-seeds", default="0,1,2,3,4")
    command.set_defaults(func=select)

    command = commands.add_parser("summarize")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.add_argument("--baseline", default="results/thf_rl_mix_tara/summary.json")
    command.add_argument("--expected-final-seeds", type=int, default=5)
    command.add_argument("--broad-steps", type=int, default=250_000)
    command.add_argument("--full-steps", type=int, default=1_000_000)
    command.set_defaults(func=summarize)

    command = commands.add_parser("status")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.set_defaults(func=status)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
