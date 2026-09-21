#!/usr/bin/env python
"""Continue Optuna PPO under the promoted column-v2 FNO and audit it exactly."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, rollout_metrics, write_json


FINAL_SEEDS = tuple(range(5))
FINAL_EVAL_SEED = 84_101


def parse_seeds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(FINAL_SEEDS)
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or min(seeds) < 0 or max(seeds) >= len(FINAL_SEEDS):
        raise ValueError("seeds must lie in [0, 5)")
    return seeds


def environments(args):
    namespace = argparse.Namespace(
        device=args.device,
        p_target=0.98,
        max_pulses=80,
        rho=0.0,
        progress=False,
        fno_tag="column_v2",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )
    return build_environments(namespace, 128)


def run(args):
    output = Path(args.output)
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    _, _, fno, exact, _, contract = environments(args)
    write_json(output / "contract.json", contract)
    for seed in parse_seeds(args.seeds):
        result_path = destination / f"seed_{seed}.json"
        if result_path.exists():
            continue
        source_path = Path(args.source_models) / f"ppo_ppo_t71_s{seed}.pt"
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        source_config = PPOConfig(**checkpoint["config"])
        config = replace(
            source_config,
            total_steps=args.steps,
            lr=args.lr,
            failure_penalty=args.failure_penalty,
            eval_every=max(1, math.ceil(args.steps / (source_config.n_envs * source_config.n_steps))),
            seed=94_000 + seed,
        )
        print(f"[column-v2 PPO] seed {seed}", flush=True)
        trained = train_ppo(
            fno,
            config,
            env_eval=fno,
            log=print,
            initial_state_dict=checkpoint["state_dict"],
        )
        state = trained.best_state_dict or trained.final_state_dict
        policy = policy_from_state_dict(
            state,
            trained.n_in,
            trained.n_actions,
            config,
            device=args.device,
            greedy=False,
            max_pulses=exact.cfg.max_pulses,
        )
        evaluations = {
            "exact": rollout_metrics(exact, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch),
            "fno": rollout_metrics(fno, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch),
        }
        model_path = output / "models" / f"column_v2_ppo_s{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "agent": "ppo",
            "config": asdict(config),
            "state_dict": state,
            "source_model": str(source_path),
            "manifest": contract["manifest"],
            "manifest_fingerprint": contract["manifest_fingerprint"],
        }, model_path)
        write_json(result_path, {
            "seed": seed,
            "source_model": str(source_path),
            "config": asdict(config),
            "training": trained.as_dict(),
            "evaluation": evaluations,
            "model_path": str(model_path),
        })
        torch.cuda.empty_cache()


def aggregate(records: list[dict], dynamics: str) -> dict:
    result = {}
    for key in ("unfinished_fraction", "average_actions", "mean_pulses_successful"):
        values = np.asarray([row["evaluation"][dynamics][key] for row in records], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    failures = np.asarray(
        [row["evaluation"][dynamics]["unfinished_fraction"] for row in records], dtype=float,
    )
    half_width = 2.776 * failures.std(ddof=1) / math.sqrt(len(failures))
    result["unfinished_fraction_seed_ci95"] = [
        float(max(0.0, failures.mean() - half_width)),
        float(min(1.0, failures.mean() + half_width)),
    ]
    return result


def summarize(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "confirmation").glob("seed_*.json"))
    ]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    source = json.loads((Path(args.mix_summary)).read_text())
    mix_ppo = next(row for row in source["rows"] if row["name"] == "FNO-trained PPO")
    risk = json.loads(Path(args.safe_summary).read_text())["rows"]["risk_actor"]
    improved = {dynamics: aggregate(records, dynamics) for dynamics in ("exact", "fno")}
    improved["transfer_failure_gap_pp"] = 100 * (
        improved["exact"]["unfinished_fraction"] - improved["fno"]["unfinished_fraction"]
    )
    improved["transfer_action_gap"] = (
        improved["exact"]["average_actions"] - improved["fno"]["average_actions"]
    )
    summary = {
        "status": "complete",
        "method": "Optuna mix-FNO PPO continued for 500k steps under column-v2 FNO with timeout cost 20",
        "training_seeds": len(records),
        "evaluation_episodes_per_seed": args.final_eval,
        "mix_fno_ppo": mix_ppo,
        "failure_sensitive_exact_ppo": risk,
        "column_v2_fno_ppo": improved,
    }
    write_json(output / "summary.json", summary)

    labels = ("Downloaded mix PPO", "Exact risk PPO", "Column-v2 FNO PPO")
    rows = (mix_ppo, risk, improved)
    exact_rows = [row["exact"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].bar(x, [100*r["unfinished_fraction"] for r in exact_rows],
                color=("#4c78a8", "#f58518", "#54a24b"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, [r["average_actions"] for r in exact_rows],
                color=("#4c78a8", "#f58518", "#54a24b"))
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("PPO after structurally promoted FNO training")
    fig.savefig(output / "column_v2_ppo_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# PPO under the promoted column-v2 FNO",
        "",
        "| Controller | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in zip(labels, rows):
        lines.append(
            f"| {label} | {100*row['exact']['unfinished_fraction']:.2f}% | "
            f"{row['exact']['average_actions']:.2f} | {100*row['fno']['unfinished_fraction']:.2f}% | "
            f"{row['fno']['average_actions']:.2f} |"
        )
    lines += ["", "![Column-v2 PPO comparison](column_v2_ppo_comparison.png)"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "confirmation": len(list((output / "confirmation").glob("seed_*.json"))),
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("command", choices=("run", "summarize", "status"))
    main.add_argument("--output", default="results/thf_column_fno_v2_ppo")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/column_v2.json"))
    main.add_argument("--source-models", default="results/thf_rl_optuna_mix/models")
    main.add_argument("--mix-summary", default="results/thf_fno_rl_refinement/summary.json")
    main.add_argument("--safe-summary", default="results/thf_safe_hybrid/summary.json")
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--seeds", default="all")
    main.add_argument("--steps", type=int, default=500_000)
    main.add_argument("--lr", type=float, default=6e-4)
    main.add_argument("--failure-penalty", type=float, default=20.0)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[args.command](args)
