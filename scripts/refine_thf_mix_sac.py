#!/usr/bin/env python
"""Focused SAC refinement for the fixed downloaded ThF+ mix FNO.

The search transfers only scale and optimization choices from the successful
H3O+ S17 SAC screen.  ThF+ dynamics, action set, purity threshold, and the S18
two-branch target remain fixed.  Each Optuna trial trains two paired seeds;
promotion and final evaluation use fresh seeds.
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

from qlsgym.rl.off_policy import DiscreteSACAgent, SACConfig, sac_policy
from thf_rl_agents import build_environments, git_sha, json_safe, rollout_metrics, write_json


DEFAULT_OUTPUT = Path("results/thf_rl_optuna_mix_sac_refine")
DEFAULT_MANIFEST = (
    Path(os.environ.get("QLSGYM_WORK", "~/qlsgym_work")).expanduser()
    / "checkpoints/thf/mix.json"
)
STUDY_NAME = "thf_mix_sac_refine_v1"
BROAD_TRAIN_SEEDS = (41_001, 41_002)
BROAD_EVAL_SEEDS = (51_001, 51_002)
PROMOTION_EVAL_SEED = 52_001
FINAL_EVAL_SEED = 20_001


def compact_metrics(metrics: dict) -> dict:
    drop = {"lengths", "actual_lengths", "successes"}
    return {key: value for key, value in metrics.items() if key not in drop}


def objective_score(metrics: dict) -> float:
    success = float(metrics["success_fraction"])
    efficiency = 1.0 - float(metrics["average_actions"]) / float(metrics["max_pulses"])
    return success + 0.02 * efficiency


def robust_score(values: list[float]) -> float:
    """Reward mean performance while mildly penalizing training-seed variance."""
    return float(np.mean(values) - 0.25 * np.std(values, ddof=0))


def storage(output: Path) -> optuna.storages.RDBStorage:
    path = (output / "optuna.sqlite3").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        f"sqlite:///{path}", engine_kwargs={"connect_args": {"timeout": 120}}
    )


def load_study(output: Path, worker_id: int = 0, create: bool = True):
    sampler = optuna.samplers.TPESampler(
        seed=91_000 + worker_id,
        n_startup_trials=8,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    if create:
        return optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage(output),
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )
    return optuna.load_study(study_name=STUDY_NAME, storage=storage(output))


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


def seed_profiles() -> list[tuple[str, dict]]:
    """Controls and H3O+ transfers used to initialize the focused TPE study."""
    return [
        (
            "locked_baseline_control",
            {
                "n_envs": 128,
                "update_ratio": 1 / 32,
                "return_scale": 80.0,
                "alpha_reward_ratio": 0.05,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 10_240,
                "target_entropy_ratio": 0.5,
                "obs": "sqrt",
                "hidden": 256,
                "depth": 2,
            },
        ),
        (
            "h3o_direct_scale_transfer",
            {
                "n_envs": 16,
                "update_ratio": 1 / 8,
                "return_scale": 5.0,
                "alpha_reward_ratio": 0.02699874256614982 * 5.0,
                "lr": 0.0002823134500820829,
                "gamma": 0.99,
                "tau": 0.0029696656967673853,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "target_entropy_ratio": 0.2145690742875123,
                "obs": "p",
                "hidden": 128,
                "depth": 3,
            },
        ),
        (
            "h3o_dimensionless_transfer",
            {
                "n_envs": 16,
                "update_ratio": 1 / 8,
                "return_scale": 80.0,
                "alpha_reward_ratio": 0.02699874256614982 * 5.0,
                "lr": 0.0002823134500820829,
                "gamma": 0.99,
                "tau": 0.0029696656967673853,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "target_entropy_ratio": 0.2145690742875123,
                "obs": "p",
                "hidden": 128,
                "depth": 3,
            },
        ),
        (
            "h3o_s18_moderate_updates",
            {
                "n_envs": 32,
                "update_ratio": 1 / 16,
                "return_scale": 20.0,
                "alpha_reward_ratio": 0.10,
                "lr": 2.8e-4,
                "gamma": 0.99,
                "tau": 0.003,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "target_entropy_ratio": 0.22,
                "obs": "p",
                "hidden": 128,
                "depth": 3,
            },
        ),
    ]


def init_study(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    study = load_study(output)
    if not study.trials:
        for label, params in seed_profiles():
            study.enqueue_trial(params, user_attrs={"seed_profile": label})
        print(f"initialized {STUDY_NAME} with {len(seed_profiles())} seeded profiles")
    else:
        print(f"study already contains {len(study.trials)} trials; leaving it unchanged")


def suggest_config(trial: optuna.Trial, steps: int) -> SACConfig:
    n_envs = trial.suggest_categorical("n_envs", [16, 32, 64, 128])
    update_ratio = trial.suggest_categorical(
        "update_ratio", [1 / 64, 1 / 32, 1 / 16, 1 / 8]
    )
    return_scale = trial.suggest_categorical("return_scale", [5.0, 10.0, 20.0, 40.0, 80.0])
    alpha_reward_ratio = trial.suggest_float("alpha_reward_ratio", 0.02, 0.5, log=True)
    gradient_steps = max(1, int(round(n_envs * update_ratio)))
    return SACConfig(
        n_envs=n_envs,
        total_steps=steps,
        lr=trial.suggest_float("lr", 3e-5, 1e-3, log=True),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995]),
        tau=trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        buffer_size=trial.suggest_categorical(
            "buffer_size", [100_000, 200_000, 400_000]
        ),
        learning_starts=trial.suggest_categorical(
            "learning_starts", [1_000, 2_000, 5_000, 10_240]
        ),
        train_freq=1,
        gradient_steps=gradient_steps,
        target_entropy_ratio=trial.suggest_float("target_entropy_ratio", 0.10, 0.50),
        alpha=alpha_reward_ratio / return_scale,
        autotune_alpha=True,
        return_scale=return_scale,
        obs=trial.suggest_categorical("obs", ["p", "sqrt"]),
        hidden=trial.suggest_categorical("hidden", [128, 256]),
        depth=trial.suggest_int("depth", 1, 3),
        seed=BROAD_TRAIN_SEEDS[0],
    )


def broad_objective(trial, fno, args):
    started = time.perf_counter()
    template = suggest_config(trial, args.broad_steps)
    per_seed = []
    scores = []
    for index, (train_seed, eval_seed) in enumerate(
        zip(BROAD_TRAIN_SEEDS, BROAD_EVAL_SEEDS), start=1
    ):
        raw = asdict(template)
        raw["seed"] = train_seed
        config = SACConfig(**raw)
        agent = DiscreteSACAgent(fno, config)
        stats = agent.train(log_points=4)
        metrics = rollout_metrics(
            fno, sac_policy(agent, stochastic=True), args.broad_eval,
            eval_seed, args.eval_batch,
        )
        score = objective_score(metrics)
        scores.append(score)
        per_seed.append(
            {
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "score": score,
                "metrics": compact_metrics(metrics),
                "wall_clock_s": stats.wall_clock_s,
                "gradient_steps": stats.gradient_steps,
                "final_log": stats.history[-1] if stats.history else {},
            }
        )
        trial.report(robust_score(scores), index)
        del agent
        torch.cuda.empty_cache()

    config_record = asdict(template)
    config_record["seed"] = None
    trial.set_user_attr("config", config_record)
    trial.set_user_attr("per_seed", per_seed)
    trial.set_user_attr("mean_score", float(np.mean(scores)))
    trial.set_user_attr("score_sd", float(np.std(scores, ddof=0)))
    trial.set_user_attr("wall_clock_s", time.perf_counter() - started)
    return robust_score(scores)


def finished_trials(study):
    states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    return [trial for trial in study.trials if trial.state in states]


def completed_trials(study):
    return [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def optimize(args):
    output = Path(args.output)
    _, _, fno, _, _, contract = build_environments(env_namespace(args, args.device), 128)
    write_json(output / "contract.json", contract)
    while True:
        study = load_study(output, args.worker_id)
        if len(finished_trials(study)) >= args.target_trials:
            break
        study.optimize(
            lambda trial: broad_objective(trial, fno, args),
            n_trials=1,
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
    print(f"worker {args.worker_id}: target of {args.target_trials} finished trials reached")


def prepare(args):
    output = Path(args.output)
    study = load_study(output, create=False)
    ranked = sorted(completed_trials(study), key=lambda trial: trial.value, reverse=True)
    if len(ranked) < args.top_k:
        raise RuntimeError(f"only {len(ranked)} complete trials; need {args.top_k}")
    seeds = [int(value) for value in args.promotion_seeds.split(",")]
    finalists = []
    jobs = []
    for rank, trial in enumerate(ranked[:args.top_k], start=1):
        config_id = f"sac_t{trial.number}"
        entry = {
            "config_id": config_id,
            "broad_rank": rank,
            "trial_number": trial.number,
            "broad_score": trial.value,
            "seed_profile": trial.user_attrs.get("seed_profile"),
            "config": trial.user_attrs["config"],
        }
        finalists.append(entry)
        for seed in seeds:
            jobs.append({**entry, "stage": "promotion", "train_seed": seed})
    write_json(output / "broad_finalists.json", finalists)
    write_json(output / "promotion_jobs.json", jobs)
    print(f"prepared {len(jobs)} promotion jobs from {len(finalists)} finalists")


def full_config(raw: dict, seed: int, steps: int) -> SACConfig:
    config = dict(raw)
    config["seed"] = seed
    config["total_steps"] = steps
    return SACConfig(**config)


def save_model(path: Path, config: SACConfig, agent: DiscreteSACAgent):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": "sac_discrete",
            "config": asdict(config),
            "actor_state_dict": agent.network.actor.state_dict(),
            "critic1_state_dict": agent.network.q1.state_dict(),
            "critic2_state_dict": agent.network.q2.state_dict(),
            "alpha": float(agent.network.log_alpha.detach().exp()),
        },
        path,
    )


def run_full_job(job, args, fno, exact, engine, contract):
    started = time.time()
    stage = job["stage"]
    episodes = args.promotion_eval if stage == "promotion" else args.final_eval
    eval_seed = PROMOTION_EVAL_SEED if stage == "promotion" else FINAL_EVAL_SEED
    config = full_config(job["config"], job["train_seed"], args.full_steps)
    agent = DiscreteSACAgent(fno, config)
    stats = agent.train()
    policy = sac_policy(agent, stochastic=True)
    fno_metrics = rollout_metrics(fno, policy, episodes, eval_seed, args.eval_batch)
    exact_metrics = rollout_metrics(exact, policy, episodes, eval_seed, args.eval_batch)
    model_path = None
    if stage == "final":
        model_path = Path(args.output) / "models" / (
            f"{job['config_id']}_s{job['train_seed']}.pt"
        )
        save_model(model_path, config, agent)
    return {
        "schema_version": 1,
        "job": job,
        "contract": contract,
        "training": {
            **stats.as_dict(),
            "config": asdict(config),
            "final_alpha": float(agent.network.log_alpha.detach().exp()),
        },
        "evaluation": {"fno": fno_metrics, "exact": exact_metrics},
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
    return output / job["stage"] / f"{job['config_id']}_s{job['train_seed']}.json"


def run_queue(args):
    output = Path(args.output)
    jobs = json.loads((output / f"{args.stage}_jobs.json").read_text())
    _, _, fno, exact, engine, contract = build_environments(
        env_namespace(args, args.device), 128
    )
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
        print(f"worker {args.worker_id}: {job['stage']} {job['config_id']} seed {job['train_seed']}")
        try:
            write_json(
                destination,
                run_full_job(job, args, fno, exact, engine, contract),
            )
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
        if not path.name.endswith(".error.json"):
            records.append(json.loads(path.read_text()))
    return records


def select(args):
    output = Path(args.output)
    grouped = defaultdict(list)
    for record in load_stage(output, "promotion"):
        grouped[record["job"]["config_id"]].append(record)
    candidates = []
    for config_id, records in grouped.items():
        scores = [float(record["selection_score_fno"]) for record in records]
        candidates.append((robust_score(scores), config_id, records))
    if not candidates:
        raise RuntimeError("no promotion records")
    score, config_id, records = max(candidates, key=lambda item: item[0])
    template = records[0]["job"]
    choice = {
        "config_id": config_id,
        "trial_number": template["trial_number"],
        "promotion_robust_score_fno": score,
        "promotion_runs": len(records),
        "config": template["config"],
    }
    seeds = [int(value) for value in args.final_seeds.split(",")]
    jobs = [
        {
            "config_id": config_id,
            "trial_number": template["trial_number"],
            "config": template["config"],
            "stage": "final",
            "train_seed": seed,
        }
        for seed in seeds
    ]
    write_json(output / "selected_config.json", choice)
    write_json(output / "final_jobs.json", jobs)
    print(f"selected {config_id}; prepared {len(jobs)} final jobs")


def aggregate_metrics(records, dynamics: str) -> dict:
    metrics = [record["evaluation"][dynamics] for record in records]
    result = {}
    for key in (
        "success_fraction", "unfinished_fraction", "average_actions",
        "mean_pulses_successful", "p85_actions", "cvar10_actions",
    ):
        values = [item[key] for item in metrics if item.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def locked_baseline(path: Path) -> dict:
    data = json.loads(path.read_text())
    return next(row for row in data["rows"] if row["agent"] == "sac_discrete")


def previous_optuna(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data["rows"]["sac_discrete"]


def make_figures(output: Path, study, final_row, baseline, previous, importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = sorted(completed_trials(study), key=lambda trial: trial.number)
    x = np.asarray([trial.number for trial in trials])
    y = np.asarray([trial.value for trial in trials])
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.scatter(x, y, s=28, alpha=0.65, color="#4c78a8", label="two-seed trial")
    if len(y):
        axis.plot(x, np.maximum.accumulate(y), color="#e45756", lw=2, label="best so far")
    axis.set(xlabel="trial", ylabel="robust FNO score", title="Focused ThF+ SAC refinement")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output / "sac_refine_history.png", dpi=180)
    plt.close(fig)

    names = ["Locked baseline", "Previous Optuna", "SAC refinement"]
    sources = [baseline, previous, final_row]
    x = np.arange(len(names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, key, ylabel in (
        (axes[0], "unfinished_fraction", "unfinished episodes (%)"),
        (axes[1], "average_actions", "penalized average actions"),
    ):
        exact = [source["exact"][key] for source in sources]
        fno = [source["fno"][key] for source in sources]
        if key == "unfinished_fraction":
            exact = [100 * value for value in exact]
            fno = [100 * value for value in fno]
        axis.bar(x - width / 2, exact, width, label="exact", color="#4c78a8")
        axis.bar(x + width / 2, fno, width, label="FNO", color="#f58518", hatch="//")
        axis.set_xticks(x, names, rotation=12, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output / "sac_refined_vs_prior.png", dpi=180)
    plt.close(fig)

    items = list(importance.items())[:12][::-1]
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    if items:
        axis.barh([item[0] for item in items], [item[1] for item in items], color="#72b7b2")
    axis.set(xlabel="PED-ANOVA importance", title="Focused SAC parameter importance")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(output / "sac_refine_importance.png", dpi=180)
    plt.close(fig)


def summarize(args):
    output = Path(args.output)
    final = load_stage(output, "final")
    if len(final) < args.expected_final_seeds:
        raise RuntimeError(
            f"only {len(final)} final records; need {args.expected_final_seeds}"
        )
    exact = aggregate_metrics(final, "exact")
    fno = aggregate_metrics(final, "fno")
    baseline = locked_baseline(Path(args.baseline))
    previous = previous_optuna(Path(args.previous_optuna))
    choice = json.loads((output / "selected_config.json").read_text())
    study = load_study(output, create=False)
    evaluator = optuna.importance.PedAnovaImportanceEvaluator()
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
    row = {
        "agent": "sac_discrete",
        "config_id": choice["config_id"],
        "trial_number": choice["trial_number"],
        "training_seeds": len(final),
        "exact": exact,
        "fno": fno,
        "config": choice["config"],
    }
    row["improvement_vs_locked_baseline"] = {
        "exact_failure_percentage_points": 100 * (
            baseline["exact"]["unfinished_fraction"] - exact["unfinished_fraction"]
        ),
        "exact_average_actions": baseline["exact"]["average_actions"] - exact["average_actions"],
        "fno_failure_percentage_points": 100 * (
            baseline["fno"]["unfinished_fraction"] - fno["unfinished_fraction"]
        ),
        "fno_average_actions": baseline["fno"]["average_actions"] - fno["average_actions"],
    }
    report = {
        "status": "complete",
        "source_study_caveat": (
            "Transferred optimization choices from the H3O+ S17 validation screen; "
            "ThF+ retains the downloaded mix FNO and exact S18 branch expectation."
        ),
        "selection_rule": (
            "FNO success plus 0.02 normalized pulse efficiency, averaged across seeds "
            "with a 0.25 standard-deviation penalty. Exact dynamics are audit-only."
        ),
        "budget": {
            "broad_steps_per_seed": args.broad_steps,
            "broad_train_seeds": list(BROAD_TRAIN_SEEDS),
            "broad_eval_seeds": list(BROAD_EVAL_SEEDS),
            "full_steps": args.full_steps,
            "promotion_eval_seed": PROMOTION_EVAL_SEED,
            "final_eval_seed": FINAL_EVAL_SEED,
        },
        "study": {
            "trials": len(study.trials),
            "states": dict(Counter(trial.state.name.lower() for trial in study.trials)),
            "best_trial": study.best_trial.number,
            "best_score": study.best_value,
            "best_params": study.best_params,
        },
        "parameter_importance_ped_anova": importance,
        "row": row,
    }
    write_json(output / "summary.json", report)
    trial_records = [
        {
            "number": trial.number,
            "state": trial.state.name.lower(),
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    write_json(output / "broad_trials.json", trial_records)
    make_figures(output, study, row, baseline, previous, importance)

    lines = [
        "# ThF+ SAC refinement from the successful H3O+ Optuna screen",
        "",
        "The H3O+ result is a validation-only S17 screen on exact tables, so its score is not "
        "directly comparable to ThF+. This refinement transfers its optimizer settings while "
        "keeping the downloaded ThF+ `mix` FNO and the S18 two-branch target fixed.",
        "",
        "| Run | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
        f"| Locked SAC baseline | {100*baseline['exact']['unfinished_fraction']:.2f}% | "
        f"{baseline['exact']['average_actions']:.2f} | {100*baseline['fno']['unfinished_fraction']:.2f}% | "
        f"{baseline['fno']['average_actions']:.2f} |",
        f"| Previous Optuna SAC | {100*previous['exact']['unfinished_fraction']:.2f}% | "
        f"{previous['exact']['average_actions']:.2f} | {100*previous['fno']['unfinished_fraction']:.2f}% | "
        f"{previous['fno']['average_actions']:.2f} |",
        f"| Refined SAC | {100*exact['unfinished_fraction']:.2f}% | {exact['average_actions']:.2f} | "
        f"{100*fno['unfinished_fraction']:.2f}% | {fno['average_actions']:.2f} |",
        "",
        "Positive deltas below mean improvement over the locked baseline:",
        "",
        f"- exact failure: {row['improvement_vs_locked_baseline']['exact_failure_percentage_points']:+.2f} pp",
        f"- exact actions: {row['improvement_vs_locked_baseline']['exact_average_actions']:+.2f}",
        f"- FNO failure: {row['improvement_vs_locked_baseline']['fno_failure_percentage_points']:+.2f} pp",
        f"- FNO actions: {row['improvement_vs_locked_baseline']['fno_average_actions']:+.2f}",
        "",
        "## Selected configuration",
        "",
        "```json",
        json.dumps(choice["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Figures",
        "",
        "- `sac_refine_history.png`: paired-seed Optuna history.",
        "- `sac_refined_vs_prior.png`: locked baseline, previous Optuna, and refined SAC.",
        "- `sac_refine_importance.png`: focused-study parameter importance.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {output / 'summary.md'}")


def status(args):
    output = Path(args.output)
    result = {"output": str(output.resolve())}
    try:
        study = load_study(output, create=False)
        complete = completed_trials(study)
        result["study"] = {
            "states": dict(Counter(trial.state.name.lower() for trial in study.trials)),
            "best_value": max((trial.value for trial in complete), default=None),
        }
    except Exception:
        pass
    for stage in ("promotion", "final"):
        jobs_path = output / f"{stage}_jobs.json"
        if jobs_path.exists():
            result[stage] = {
                "planned": len(json.loads(jobs_path.read_text())),
                "complete": len(load_stage(output, stage)),
                "errors": len(list((output / stage).glob("*.error.json"))),
            }
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


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.set_defaults(func=init_study)

    command = commands.add_parser("optimize")
    add_environment(command)
    command.add_argument("--worker-id", type=int, required=True)
    command.add_argument("--target-trials", type=int, default=24)
    command.add_argument("--broad-steps", type=int, default=300_000)
    command.add_argument("--broad-eval", type=int, default=1_024)
    command.set_defaults(func=optimize)

    command = commands.add_parser("prepare")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.add_argument("--top-k", type=int, default=4)
    command.add_argument("--promotion-seeds", default="42001,42002")
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
    command.add_argument("--previous-optuna", default="results/thf_rl_optuna_mix/summary.json")
    command.add_argument("--expected-final-seeds", type=int, default=5)
    command.add_argument("--broad-steps", type=int, default=300_000)
    command.add_argument("--full-steps", type=int, default=1_000_000)
    command.set_defaults(func=summarize)

    command = commands.add_parser("status")
    command.add_argument("--output", default=str(DEFAULT_OUTPUT))
    command.set_defaults(func=status)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
