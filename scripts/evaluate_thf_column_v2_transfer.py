#!/usr/bin/env python
"""Audit closed-loop transfer under the unpromoted column-v2 FNO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improve_thf_descending_hybrid import FINAL_EVAL_SEED, FINAL_SEEDS, load_actor, parse_indices
from qlsgym.policies import DescendingPopulationPolicy
from thf_rl_agents import build_environments, rollout_metrics, write_json


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
    molecule, library, fno, exact, _, contract = build_environments(namespace, 128)
    return molecule, library, fno, exact, contract


def run(args):
    output = Path(args.output)
    destination = output / "seeds"
    destination.mkdir(parents=True, exist_ok=True)
    _, _, fno, exact, contract = environments(args)
    write_json(output / "contract.json", contract)
    for seed in parse_indices(args.seeds, len(FINAL_SEEDS)):
        path = destination / f"seed_{seed}.json"
        if path.exists():
            continue
        print(f"[column-v2 transfer] actor seed {seed}", flush=True)
        actor = load_actor(args, seed, exact.cfg.max_pulses)
        metrics = rollout_metrics(fno, actor, args.final_eval, FINAL_EVAL_SEED, args.eval_batch)
        write_json(path, {"seed": seed, "column_v2": metrics})
        torch.cuda.empty_cache()


def baseline(args):
    output = Path(args.output)
    path = output / "descending_population.json"
    if path.exists():
        return
    _, library, fno, exact, contract = environments(args)
    write_json(output / "contract.json", contract)
    policy = DescendingPopulationPolicy(library, exact.tables)
    print("[column-v2 transfer] descending population", flush=True)
    metrics = rollout_metrics(fno, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch)
    write_json(path, {"column_v2": metrics})


def aggregate(records):
    result = {}
    for key in ("unfinished_fraction", "average_actions", "mean_pulses_successful"):
        values = np.asarray([row["column_v2"][key] for row in records], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    failures = np.asarray([row["column_v2"]["unfinished_fraction"] for row in records])
    half_width = 2.776 * failures.std(ddof=1) / math.sqrt(len(failures))
    result["unfinished_fraction_seed_ci95"] = [
        float(max(0.0, failures.mean() - half_width)),
        float(min(1.0, failures.mean() + half_width)),
    ]
    return result


def summarize(args):
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "seeds").glob("seed_*.json"))]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five actor records, found {len(records)}")
    baseline_v2 = json.loads((output / "descending_population.json").read_text())["column_v2"]
    safe = json.loads(Path(args.safe_summary).read_text())["rows"]["risk_actor"]
    nonml = json.loads(Path(args.nonml_summary).read_text())["rows"]["descending_population"]
    actor_v2 = aggregate(records)
    summary = {
        "status": "complete",
        "purpose": "diagnostic only; column-v2 failed the 24-pair promotion gate",
        "training_seeds": len(records),
        "evaluation_episodes_per_seed": args.final_eval,
        "failure_sensitive_ppo": {
            "exact": safe["exact"],
            "downloaded_mix": safe["fno"],
            "column_v2": actor_v2,
        },
        "descending_population": {
            "exact": nonml["exact"],
            "downloaded_mix": nonml["fno"],
            "column_v2": baseline_v2,
        },
    }
    write_json(output / "summary.json", summary)

    labels = ("Exact", "Downloaded mix", "Column-v2")
    colors = ("#4c78a8", "#f58518")
    controllers = ("failure_sensitive_ppo", "descending_population")
    names = ("Failure-sensitive PPO", "Descending population")
    keys = ("exact", "downloaded_mix", "column_v2")
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for index, (controller, name, color) in enumerate(zip(controllers, names, colors)):
        row = summary[controller]
        offset = (index - 0.5) * width
        axes[0].bar(x + offset, [100 * row[key]["unfinished_fraction"] for key in keys],
                    width, label=name, color=color)
        axes[1].bar(x + offset, [row[key]["average_actions"] for key in keys],
                    width, label=name, color=color)
    axes[0].set_ylabel("Unfinished episodes (%)")
    axes[1].set_ylabel("Failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Closed-loop audit of the unpromoted column-v2 FNO")
    fig.savefig(output / "column_v2_transfer.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Closed-loop column-v2 FNO audit",
        "",
        "`column_v2` failed the 24-pair structural gate; these results are diagnostic and were not used for training.",
        "",
        "| Controller | Exact failure | Exact actions | Mix failure | Mix actions | Column-v2 failure | Column-v2 actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for controller, name in zip(controllers, names):
        row = summary[controller]
        lines.append(
            f"| {name} | {100*row['exact']['unfinished_fraction']:.2f}% | {row['exact']['average_actions']:.2f} | "
            f"{100*row['downloaded_mix']['unfinished_fraction']:.2f}% | {row['downloaded_mix']['average_actions']:.2f} | "
            f"{100*row['column_v2']['unfinished_fraction']:.2f}% | {row['column_v2']['average_actions']:.2f} |"
        )
    lines += ["", "![Column-v2 closed-loop transfer](column_v2_transfer.png)"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "actor_seeds": len(list((output / "seeds").glob("seed_*.json"))),
        "baseline": (output / "descending_population.json").exists(),
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("command", choices=("run", "baseline", "summarize", "status"))
    main.add_argument("--output", default="results/thf_column_fno_v2_transfer")
    main.add_argument("--source", default="results/thf_safe_hybrid")
    main.add_argument("--safe-summary", default="results/thf_safe_hybrid/summary.json")
    main.add_argument("--nonml-summary", default="results/thf_nonml_controls/summary.json")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/column_v2.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--seeds", default="all")
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[args.command](args)
