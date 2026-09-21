#!/usr/bin/env python
"""Fine-tune FNO-pretrained PPO policies on cached exact ThF+ dynamics."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, rollout_metrics, write_json


LRS = (1e-4, 3e-4, 6e-4)
FINAL_SEEDS = tuple(range(5))
DIAGNOSIS_SEED = 71_001
DIAGNOSIS_EVAL_SEED = 71_101
FINAL_EVAL_SEED = 72_101


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


def environments(args):
    _, _, fno, exact, _, contract = build_environments(env_namespace(args), 128)
    return exact, fno, contract


def config(steps: int, seed: int, lr: float) -> PPOConfig:
    return PPOConfig(
        n_envs=128,
        n_steps=32,
        total_steps=steps,
        epochs=8,
        minibatches=16,
        lr=lr,
        gamma=1.0,
        gae_lambda=0.98,
        clip=0.1,
        ent_coef=0.0005417800781171913,
        vf_coef=0.5,
        max_grad_norm=1.0,
        hidden=512,
        n_hidden_layers=1,
        obs="sqrt",
        include_budget=False,
        value_target="qmdp",
        reward_scale=0.025,
        eval_every=max(1, math.ceil(steps / (128 * 32))),
        eval_rollouts=256,
        eval_greedy=False,
        seed=seed,
    )


def source_model(args, seed: int) -> Path:
    return (
        Path(args.source_models)
        / f"ppo_ppo_t71_s{seed}.pt"
    )


def score(metrics: dict) -> float:
    return float(
        metrics["success_fraction"]
        + 0.02 * (1.0 - metrics["average_actions"] / metrics["max_pulses"])
    )


def run_one(args, exact, fno, source_seed, train_seed, lr, steps, episodes, eval_seed, stage):
    source = source_model(args, source_seed)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    cfg = config(steps, train_seed, lr)
    started = time.time()
    trained = train_ppo(
        exact,
        cfg,
        env_eval=exact,
        log=print,
        initial_state_dict=checkpoint["state_dict"],
    )
    state = trained.best_state_dict or trained.final_state_dict
    policy = policy_from_state_dict(
        state,
        trained.n_in,
        trained.n_actions,
        cfg,
        device=args.device,
        greedy=False,
        max_pulses=exact.cfg.max_pulses,
    )
    exact_metrics = rollout_metrics(exact, policy, episodes, eval_seed, args.eval_batch)
    fno_metrics = rollout_metrics(fno, policy, episodes, eval_seed, args.eval_batch)
    model_path = Path(args.output) / "models" / f"{stage}_source_s{source_seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": "ppo",
            "config": asdict(cfg),
            "state_dict": state,
            "source_model": str(source),
            "source_seed": source_seed,
        },
        model_path,
    )
    return {
        "stage": stage,
        "source_model": str(source),
        "source_seed": source_seed,
        "train_seed": train_seed,
        "fine_tune_steps": steps,
        "learning_rate": lr,
        "config": asdict(cfg),
        "training": trained.as_dict(),
        "evaluation": {"exact": exact_metrics, "fno": fno_metrics},
        "selection_score_exact": score(exact_metrics),
        "model_path": str(model_path),
        "seconds": time.time() - started,
    }


def diagnose(args):
    output = Path(args.output)
    destination = output / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    exact, fno, contract = environments(args)
    write_json(output / "contract.json", contract)
    for index, lr in enumerate(LRS):
        path = destination / f"lr_{lr:.0e}.json"
        if path.exists():
            continue
        write_json(
            path,
            run_one(
                args,
                exact,
                fno,
                source_seed=0,
                train_seed=DIAGNOSIS_SEED,
                lr=lr,
                steps=args.diagnose_steps,
                episodes=args.diagnose_eval,
                eval_seed=DIAGNOSIS_EVAL_SEED,
                stage=f"diagnosis_lr{index}",
            ),
        )
        torch.cuda.empty_cache()


def select(args):
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "diagnosis").glob("*.json"))]
    if len(records) != len(LRS):
        raise RuntimeError(f"expected {len(LRS)} diagnosis records, found {len(records)}")
    winner = max(records, key=lambda row: row["selection_score_exact"])
    selection = {
        "learning_rate": winner["learning_rate"],
        "diagnosis_exact": {
            key: winner["evaluation"]["exact"][key]
            for key in ("success_fraction", "average_actions", "mean_pulses_successful")
        },
        "selection_score_exact": winner["selection_score_exact"],
        "rule": "exact success plus 0.02 normalized action efficiency",
    }
    write_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def confirm(args):
    output = Path(args.output)
    selection = json.loads((output / "selection.json").read_text())
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    exact, fno, _ = environments(args)
    for source_seed in FINAL_SEEDS:
        path = destination / f"source_s{source_seed}.json"
        if path.exists():
            continue
        write_json(
            path,
            run_one(
                args,
                exact,
                fno,
                source_seed=source_seed,
                train_seed=72_000 + source_seed,
                lr=float(selection["learning_rate"]),
                steps=args.confirm_steps,
                episodes=args.final_eval,
                eval_seed=FINAL_EVAL_SEED,
                stage="confirmation",
            ),
        )
        torch.cuda.empty_cache()


def aggregate(records, engine):
    keys = ("success_fraction", "unfinished_fraction", "average_actions", "mean_pulses_successful")
    result = {}
    for key in keys:
        values = [float(row["evaluation"][engine][key]) for row in records]
        result[key] = float(np.mean(values))
        result[key + "_sd"] = float(np.std(values, ddof=1))
    return result


def summarize(args):
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "confirmation").glob("*.json"))]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    fno_reference = json.loads((output.parent / "thf_rl_optuna_mix" / "summary.json").read_text())["rows"]["ppo"]
    exact_reference = json.loads((output.parent / "thf_fno_rl_improvement" / "summary.json").read_text())
    exact, fno = aggregate(records, "exact"), aggregate(records, "fno")
    summary = {
        "status": "complete",
        "method": "downloaded-mix FNO pretraining followed by cached-exact PPO fine-tuning",
        "selection": json.loads((output / "selection.json").read_text()),
        "fine_tune_steps": args.confirm_steps,
        "training_seeds": len(records),
        "hybrid": {"exact": exact, "fno": fno},
        "fno_trained_reference": fno_reference,
        "exact_trained_reference": {
            "exact": exact_reference["exact"],
            "fno": exact_reference["fno"],
        },
    }
    write_json(output / "summary.json", summary)

    names = ["FNO trained", "FNO→exact fine-tuned", "Exact trained"]
    exact_rows = [fno_reference["exact"], exact, exact_reference["exact"]]
    failure = [100 * row["unfinished_fraction"] for row in exact_rows]
    actions = [row["average_actions"] for row in exact_rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(x, failure, color=("#f58518", "#54a24b", "#4c78a8"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, actions, color=("#f58518", "#54a24b", "#4c78a8"))
    axes[1].set_ylabel("Exact penalized average actions")
    for axis in axes:
        axis.set_xticks(x, names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ThF+ PPO: surrogate pretraining and exact fine-tuning")
    fig.savefig(output / "hybrid_ppo_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# ThF+ PPO hybrid exact fine-tuning",
        "",
        "| Training | Exact failure | Exact actions |",
        "|---|---:|---:|",
    ]
    for name, row in zip(names, exact_rows):
        lines.append(f"| {name} | {100*row['unfinished_fraction']:.2f}% | {row['average_actions']:.2f} |")
    lines += ["", "![Hybrid PPO comparison](hybrid_ppo_comparison.png)"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "diagnosis": len(list((output / "diagnosis").glob("*.json"))) if (output / "diagnosis").exists() else 0,
        "selected": (output / "selection.json").exists(),
        "confirmation": len(list((output / "confirmation").glob("*.json"))) if (output / "confirmation").exists() else 0,
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("command", choices=("diagnose", "select", "confirm", "summarize", "run", "status"))
    main.add_argument("--output", default="results/thf_mix_ppo_exact_finetune")
    main.add_argument("--source-models", default="results/thf_rl_optuna_mix/models")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--diagnose-steps", type=int, default=100_000)
    main.add_argument("--diagnose-eval", type=int, default=1_000)
    main.add_argument("--confirm-steps", type=int, default=250_000)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


def main():
    args = parser().parse_args()
    commands = ("diagnose", "select", "confirm", "summarize") if args.command == "run" else (args.command,)
    for command in commands:
        globals()[command](args)


if __name__ == "__main__":
    main()
