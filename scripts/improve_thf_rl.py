#!/usr/bin/env python
"""Gated exact-RL diagnosis for the ThF+ FNO improvement study.

The script reuses the strongest fixed-FNO PPO and SAC hyperparameters from
``FNO_RL_optuna``.  It changes only the training dynamics (cached exact tables)
and the optional remaining-budget observation.  All reported comparisons use
the exact simulator; the downloaded ``mix`` FNO is retained as a transfer
diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from qlsgym.rl.off_policy import DiscreteSACAgent, SACConfig, sac_policy
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, git_sha, rollout_metrics, write_json


DEFAULT_OUTPUT = Path("results/thf_fno_rl_improvement")
DEFAULT_MANIFEST = Path(
    os.environ.get("QLSGYM_WORK", "~/qlsgym_work")
).expanduser() / "checkpoints/thf/mix.json"
DIAGNOSE_SEED = 61_001
DIAGNOSE_EVAL_SEED = 62_001
FINAL_EVAL_SEED = 63_001
FINAL_SEEDS = (0, 1, 2, 3, 4)


def env_namespace(args):
    return argparse.Namespace(
        device=args.device,
        p_target=0.98,
        max_pulses=80,
        rho=0.0,
        progress=False,
        fno_tag="mix",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )


def compact_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"lengths", "actual_lengths", "successes"}
    }


def score(metrics: dict) -> float:
    efficiency = 1.0 - metrics["average_actions"] / metrics["max_pulses"]
    return float(metrics["success_fraction"] + 0.02 * efficiency)


def sac_config(steps: int, seed: int, include_budget: bool) -> SACConfig:
    """Focused ``sac_t24`` configuration selected on FNO validation."""
    return SACConfig(
        n_envs=16,
        total_steps=steps,
        lr=0.00015047792717528454,
        gamma=0.995,
        tau=0.003048864797383012,
        batch_size=512,
        buffer_size=100_000,
        learning_starts=1_000,
        train_freq=1,
        gradient_steps=1,
        target_entropy_ratio=0.49969706052497553,
        alpha=0.013268918812005082,
        autotune_alpha=True,
        return_scale=20.0,
        obs="sqrt",
        include_budget=include_budget,
        hidden=128,
        depth=1,
        seed=seed,
    )


def ppo_config(steps: int, seed: int, include_budget: bool) -> PPOConfig:
    """Best PPO profile from the completed general Optuna study."""
    return PPOConfig(
        n_envs=128,
        n_steps=32,
        total_steps=steps,
        epochs=8,
        minibatches=16,
        lr=0.001346363908197788,
        gamma=1.0,
        gae_lambda=0.98,
        clip=0.1,
        ent_coef=0.0005417800781171913,
        vf_coef=0.5,
        max_grad_norm=1.0,
        hidden=512,
        n_hidden_layers=1,
        obs="sqrt",
        include_budget=include_budget,
        value_target="qmdp",
        reward_scale=0.025,
        eval_every=max(1, math.ceil(steps / (128 * 32))),
        eval_rollouts=256,
        eval_greedy=False,
        seed=seed,
    )


def profile_name(agent: str, include_budget: bool) -> str:
    return f"{agent}_{'budget' if include_budget else 'belief'}"


def save_model(path: Path, agent: str, config, learner, state_dict=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if agent == "sac":
        payload = {
            "agent": "sac_discrete",
            "config": asdict(config),
            "actor_state_dict": learner.network.actor.state_dict(),
            "critic1_state_dict": learner.network.q1.state_dict(),
            "critic2_state_dict": learner.network.q2.state_dict(),
            "alpha": float(learner.network.log_alpha.detach().exp()),
        }
    else:
        payload = {
            "agent": "ppo",
            "config": asdict(config),
            "state_dict": state_dict,
            "n_in": learner.n_in,
            "n_actions": learner.n_actions,
        }
    torch.save(payload, path)


def run_one(
    args,
    exact,
    fno,
    contract: dict,
    agent: str,
    include_budget: bool,
    steps: int,
    train_seed: int,
    eval_seed: int,
    eval_episodes: int,
    stage: str,
) -> dict:
    started = time.time()
    name = profile_name(agent, include_budget)
    model_path = Path(args.output) / "models" / f"{stage}_{name}_s{train_seed}.pt"
    if agent == "sac":
        config = sac_config(steps, train_seed, include_budget)
        learner = DiscreteSACAgent(exact, config)
        training = learner.train().as_dict()
        policy = sac_policy(learner, stochastic=True)
        save_model(model_path, agent, config, learner)
    elif agent == "ppo":
        config = ppo_config(steps, train_seed, include_budget)
        result = train_ppo(exact, config, env_eval=exact, log=print)
        state = result.best_state_dict or result.final_state_dict
        policy = policy_from_state_dict(
            state,
            result.n_in,
            result.n_actions,
            config,
            device=args.device,
            greedy=False,
            max_pulses=exact.cfg.max_pulses,
        )
        training = result.as_dict()
        save_model(model_path, agent, config, result, state)
    else:
        raise ValueError(agent)

    exact_metrics = rollout_metrics(
        exact, policy, eval_episodes, eval_seed, args.eval_batch
    )
    fno_metrics = rollout_metrics(
        fno, policy, eval_episodes, eval_seed, args.eval_batch
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "profile": name,
        "agent": agent,
        "include_budget": include_budget,
        "train_engine": "exact cached action tables",
        "contract": contract,
        "config": asdict(config),
        "training": training,
        "evaluation": {"exact": exact_metrics, "fno": fno_metrics},
        "selection_score_exact": score(exact_metrics),
        "model_path": str(model_path),
        "wall_clock_s": time.time() - started,
        "provenance": {
            "git_sha": git_sha(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": args.device,
        },
    }


def environments(args):
    _, _, fno, exact, _, contract = build_environments(env_namespace(args), 128)
    contract = dict(contract)
    contract["diagnostic_train_dynamics"] = "exact cached qlsgym action tables"
    return exact, fno, contract


def diagnose(args) -> None:
    output = Path(args.output)
    destination = output / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    exact, fno, contract = environments(args)
    write_json(output / "contract.json", contract)
    for agent in ("sac", "ppo"):
        for include_budget in (False, True):
            path = destination / f"{profile_name(agent, include_budget)}.json"
            if path.exists():
                continue
            print(f"diagnosis: {agent}, include_budget={include_budget}", flush=True)
            write_json(
                path,
                run_one(
                    args,
                    exact,
                    fno,
                    contract,
                    agent,
                    include_budget,
                    args.diagnose_steps,
                    DIAGNOSE_SEED,
                    DIAGNOSE_EVAL_SEED,
                    args.diagnose_eval,
                    "diagnosis",
                ),
            )
            torch.cuda.empty_cache()


def load_records(path: Path) -> list[dict]:
    return [json.loads(item.read_text()) for item in sorted(path.glob("*.json"))]


def select(args) -> None:
    output = Path(args.output)
    records = load_records(output / "diagnosis")
    if len(records) != 4:
        raise RuntimeError(f"expected four diagnosis records, found {len(records)}")
    winner = max(records, key=lambda item: item["selection_score_exact"])
    selection = {
        "profile": winner["profile"],
        "agent": winner["agent"],
        "include_budget": winner["include_budget"],
        "diagnosis_score_exact": winner["selection_score_exact"],
        "diagnosis_exact": compact_metrics(winner["evaluation"]["exact"]),
        "diagnosis_fno": compact_metrics(winner["evaluation"]["fno"]),
        "selection_rule": "exact success plus 0.02 normalized pulse efficiency",
    }
    write_json(output / "selected_exact_profile.json", selection)
    print(json.dumps(selection, indent=2))


def confirm(args) -> None:
    output = Path(args.output)
    selection = json.loads((output / "selected_exact_profile.json").read_text())
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    exact, fno, contract = environments(args)
    for seed in FINAL_SEEDS:
        path = destination / f"{selection['profile']}_s{seed}.json"
        if path.exists():
            continue
        print(f"confirmation: {selection['profile']}, seed={seed}", flush=True)
        write_json(
            path,
            run_one(
                args,
                exact,
                fno,
                contract,
                selection["agent"],
                bool(selection["include_budget"]),
                args.confirm_steps,
                seed,
                FINAL_EVAL_SEED,
                args.final_eval,
                "confirmation",
            ),
        )
        torch.cuda.empty_cache()


def aggregate(records: list[dict], dynamics: str) -> dict:
    keys = (
        "success_fraction",
        "unfinished_fraction",
        "average_actions",
        "mean_pulses_successful",
        "p85_actions",
        "cvar10_actions",
    )
    out = {}
    for key in keys:
        values = [record["evaluation"][dynamics].get(key) for record in records]
        values = [float(value) for value in values if value is not None]
        out[key] = float(np.mean(values)) if values else None
        out[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def status(args) -> None:
    output = Path(args.output)
    diagnosis = load_records(output / "diagnosis") if (output / "diagnosis").exists() else []
    confirmation = load_records(output / "confirmation") if (output / "confirmation").exists() else []
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "diagnosis": {"complete": len(diagnosis), "planned": 4},
                "selected": (output / "selected_exact_profile.json").exists(),
                "confirmation": {"complete": len(confirmation), "planned": 5},
                "summary_complete": (output / "summary.json").exists(),
            },
            indent=2,
        )
    )


def load_fno_reference(output: Path, agent: str) -> tuple[str, dict]:
    if agent == "ppo":
        data = json.loads(
            (output.parent / "thf_rl_optuna_mix" / "summary.json").read_text()
        )
        return "FNO-trained optimized PPO", data["rows"]["ppo"]
    data = json.loads(
        (output.parent / "thf_rl_optuna_mix_sac_refine" / "summary.json").read_text()
    )
    return "FNO-trained refined SAC", data["row"]


def summarize(args) -> None:
    output = Path(args.output)
    selection = json.loads((output / "selected_exact_profile.json").read_text())
    records = load_records(output / "confirmation")
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(
            f"expected {len(FINAL_SEEDS)} confirmation records, found {len(records)}"
        )
    exact = aggregate(records, "exact")
    fno = aggregate(records, "fno")
    reference_name, reference = load_fno_reference(output, selection["agent"])
    result = {
        "status": "complete",
        "selected_profile": selection,
        "training_seeds": len(records),
        "train_engine": "exact cached action tables",
        "exact": exact,
        "fno": fno,
        "reference": {"name": reference_name, **reference},
        "improvement_vs_same_agent_fno_training": {
            "exact_failure_percentage_points": 100
            * (reference["exact"]["unfinished_fraction"] - exact["unfinished_fraction"]),
            "exact_average_actions": reference["exact"]["average_actions"]
            - exact["average_actions"],
        },
        "interpretation": (
            "Exact training isolates RL and finite-horizon representation from surrogate "
            "bias. FNO evaluation of the exact-trained policy is a transfer diagnostic only."
        ),
    }
    write_json(output / "summary.json", result)

    names = [reference_name, f"Exact-trained {selection['profile']}"]
    exact_failure = [
        100 * reference["exact"]["unfinished_fraction"],
        100 * exact["unfinished_fraction"],
    ]
    exact_actions = [reference["exact"]["average_actions"], exact["average_actions"]]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(2)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(x, exact_failure, color=("#f58518", "#4c78a8"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, exact_actions, color=("#f58518", "#4c78a8"))
    axes[1].set_ylabel("Exact penalized average actions")
    for axis in axes:
        axis.set_xticks(x, names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("ThF+ exact training diagnosis")
    figure.savefig(output / "exact_training_vs_fno.png", dpi=180)
    plt.close(figure)

    delta = result["improvement_vs_same_agent_fno_training"]
    lines = [
        "# ThF+ exact-training diagnosis",
        "",
        "Five fresh seeds were trained for one million transitions with cached exact "
        "action tables. Each seed was evaluated on 5,000 exact and 5,000 downloaded-"
        "`mix` FNO episodes.",
        "",
        "| Run | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {reference_name} | {100*reference['exact']['unfinished_fraction']:.2f}% | "
            f"{reference['exact']['average_actions']:.2f} | "
            f"{100*reference['fno']['unfinished_fraction']:.2f}% | "
            f"{reference['fno']['average_actions']:.2f} |"
        ),
        (
            f"| Exact-trained {selection['profile']} | {100*exact['unfinished_fraction']:.2f}% "
            f"± {100*exact['unfinished_fraction_sd']:.2f} | {exact['average_actions']:.2f} "
            f"± {exact['average_actions_sd']:.2f} | {100*fno['unfinished_fraction']:.2f}% "
            f"± {100*fno['unfinished_fraction_sd']:.2f} | {fno['average_actions']:.2f} "
            f"± {fno['average_actions_sd']:.2f} |"
        ),
        "",
        f"Exact failure improvement: {delta['exact_failure_percentage_points']:+.2f} points.",
        f"Exact action improvement: {delta['exact_average_actions']:+.2f}.",
        "",
        "The exact-trained policy's FNO score measures whether the surrogate preserves a "
        "stronger policy's visited distribution; it is not used to rank the controller.",
        "",
        "![Exact training comparison](exact_training_vs_fno.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "select", "confirm", "summarize", "status"):
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        item.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--eval-batch", type=int, default=256)
        item.add_argument("--diagnose-steps", type=int, default=100_000)
        item.add_argument("--diagnose-eval", type=int, default=1_000)
        item.add_argument("--confirm-steps", type=int, default=1_000_000)
        item.add_argument("--final-eval", type=int, default=5_000)
        item.set_defaults(func=globals()[name])
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
