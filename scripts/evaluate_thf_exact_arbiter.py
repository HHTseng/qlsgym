#!/usr/bin/env python
"""Evaluate a conservative exact gate between PPO and descending population."""

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

from improve_thf_descending_hybrid import (
    DIAGNOSIS_SEED,
    FINAL_EVAL_SEED,
    FINAL_SEEDS,
    aggregate,
    environments,
    exact_environment,
    load_actor,
    parse_indices,
)
from qlsgym.policies import DescendingPopulationPolicy, ExactCandidateArbiterPolicy
from thf_rl_agents import rollout_metrics, write_json


PURITY_MARGINS = (0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2)


def diagnose(args):
    output = Path(args.output)
    destination = output / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    library, exact = exact_environment(args.device)
    actor = load_actor(args, 0, exact.cfg.max_pulses)
    baseline = DescendingPopulationPolicy(library, exact.tables)
    for index in parse_indices(args.indices, len(PURITY_MARGINS)):
        margin = PURITY_MARGINS[index]
        path = destination / f"candidate_{index:02d}.json"
        if path.exists():
            continue
        print(f"[arbiter] purity margin {margin:g}", flush=True)
        policy = ExactCandidateArbiterPolicy(
            actor, baseline, exact, success_margin=0.0, purity_margin=margin,
        )
        metrics = rollout_metrics(exact, policy, args.diagnosis_eval, DIAGNOSIS_SEED, args.eval_batch)
        write_json(path, {
            "index": index,
            "success_margin": 0.0,
            "purity_margin": margin,
            "evaluation": {"exact": metrics},
        })


def select(args):
    output = Path(args.output)
    paths = [output / "diagnosis" / f"candidate_{index:02d}.json" for index in range(len(PURITY_MARGINS))]
    records = [json.loads(path.read_text()) for path in paths if path.exists()]
    if len(records) != len(PURITY_MARGINS):
        raise RuntimeError(f"expected {len(PURITY_MARGINS)} diagnosis records, found {len(records)}")
    winner = max(
        records,
        key=lambda row: (
            row["evaluation"]["exact"]["success_fraction"],
            -row["evaluation"]["exact"]["average_actions"],
        ),
    )
    selection = {
        "index": winner["index"],
        "success_margin": winner["success_margin"],
        "purity_margin": winner["purity_margin"],
        "validation_exact": {
            key: winner["evaluation"]["exact"][key]
            for key in ("success_fraction", "unfinished_fraction", "average_actions")
        },
        "rule": "minimize exact validation failure, then failure-penalized actions",
    }
    write_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def confirm(args):
    output = Path(args.output)
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    _, library, exact, fno, contract = environments(args)
    write_json(output / "contract.json", contract)
    selection = json.loads((output / "selection.json").read_text())
    baseline = DescendingPopulationPolicy(library, exact.tables)
    for seed in parse_indices(args.seeds, len(FINAL_SEEDS)):
        path = destination / f"seed_{seed}.json"
        if path.exists():
            continue
        actor = load_actor(args, seed, exact.cfg.max_pulses)
        evaluations = {}
        for dynamics, environment in (("exact", exact), ("fno", fno)):
            print(f"[arbiter s{seed}] {dynamics}", flush=True)
            policy = ExactCandidateArbiterPolicy(
                actor,
                baseline,
                exact,
                success_margin=float(selection["success_margin"]),
                purity_margin=float(selection["purity_margin"]),
            )
            evaluations[dynamics] = rollout_metrics(
                environment, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch,
            )
        write_json(path, {"seed": seed, "selection": selection, "evaluation": evaluations})
        torch.cuda.empty_cache()


def summarize(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "confirmation").glob("seed_*.json"))
    ]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    nonml = json.loads((output.parent / "thf_nonml_controls" / "summary.json").read_text())
    baseline = nonml["rows"]["descending_population"]
    baseline_rollout = json.loads(
        (output.parent / "thf_nonml_controls" / "descending_population.json").read_text()
    )["exact"]
    source = json.loads(Path(args.source, "summary.json").read_text())["rows"]["risk_actor"]
    arbiter = {dynamics: aggregate(records, dynamics) for dynamics in ("exact", "fno")}
    failure_differences = np.asarray([
        row["evaluation"]["exact"]["unfinished_fraction"]
        - baseline["exact"]["unfinished_fraction"]
        for row in records
    ])
    action_differences = np.asarray([
        row["evaluation"]["exact"]["average_actions"]
        - baseline["exact"]["average_actions"]
        for row in records
    ])
    failure_half_width = 2.776 * failure_differences.std(ddof=1) / math.sqrt(len(records))
    base_success = np.asarray(baseline_rollout["successes"], dtype=bool)
    rescued = lost = 0
    for row in records:
        success = np.asarray(row["evaluation"]["exact"]["successes"], dtype=bool)
        rescued += int((~base_success & success).sum())
        lost += int((base_success & ~success).sum())
    paired = {
        "failure_difference": float(failure_differences.mean()),
        "failure_difference_seed_ci95": [
            float(failure_differences.mean() - failure_half_width),
            float(failure_differences.mean() + failure_half_width),
        ],
        "action_difference": float(action_differences.mean()),
        "rescued": rescued,
        "lost": lost,
        "episodes": int(len(records) * len(base_success)),
        "net_rescue_fraction": float((rescued - lost) / (len(records) * len(base_success))),
    }
    arbiter["mean_dominates_descending_population"] = (
        arbiter["exact"]["unfinished_fraction"] < baseline["exact"]["unfinished_fraction"]
        and arbiter["exact"]["average_actions"] < baseline["exact"]["average_actions"]
    )
    arbiter["superiority_claim_supported"] = (
        paired["failure_difference_seed_ci95"][1] < 0.0
        and paired["action_difference"] < 0.0
    )
    arbiter["paired_vs_descending_population"] = paired
    summary = {
        "status": "complete",
        "selection": json.loads((output / "selection.json").read_text()),
        "training_seeds": len(records),
        "evaluation_episodes_per_seed": args.final_eval,
        "source_failure_sensitive_ppo": source,
        "descending_population": baseline,
        "exact_candidate_arbiter": arbiter,
    }
    write_json(output / "summary.json", summary)

    labels = ("Failure-sensitive PPO", "Descending population", "Exact candidate arbiter")
    rows = (source["exact"], baseline["exact"], arbiter["exact"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].bar(x, [100*r["unfinished_fraction"] for r in rows],
                color=("#f58518", "#4c78a8", "#54a24b"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, [r["average_actions"] for r in rows],
                color=("#f58518", "#4c78a8", "#54a24b"))
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Conservative exact gate for learned deviations")
    fig.savefig(output / "exact_arbiter_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Conservative exact candidate arbiter",
        "",
        "| Controller | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
        f"| Failure-sensitive PPO | {100*source['exact']['unfinished_fraction']:.2f}% | "
        f"{source['exact']['average_actions']:.2f} | {100*source['fno']['unfinished_fraction']:.2f}% | "
        f"{source['fno']['average_actions']:.2f} |",
        f"| Descending population | {100*baseline['exact']['unfinished_fraction']:.2f}% | "
        f"{baseline['exact']['average_actions']:.2f} | {100*baseline['fno']['unfinished_fraction']:.2f}% | "
        f"{baseline['fno']['average_actions']:.2f} |",
        f"| Exact candidate arbiter | {100*arbiter['exact']['unfinished_fraction']:.2f}% | "
        f"{arbiter['exact']['average_actions']:.2f} | {100*arbiter['fno']['unfinished_fraction']:.2f}% | "
        f"{arbiter['fno']['average_actions']:.2f} |",
        "",
        f"Mean dominates descending population on both exact metrics: "
        f"{arbiter['mean_dominates_descending_population']}.",
        f"Paired superiority rule is supported: {arbiter['superiority_claim_supported']}.",
        f"Failure difference (arbiter - descending): {100*paired['failure_difference']:.2f} pp "
        f"(95% seed CI {100*paired['failure_difference_seed_ci95'][0]:.2f} to "
        f"{100*paired['failure_difference_seed_ci95'][1]:.2f}); action difference "
        f"{paired['action_difference']:.2f}.",
        f"Across {paired['episodes']:,} paired rollouts: {paired['rescued']:,} rescued and "
        f"{paired['lost']:,} lost.",
        "",
        "![Exact arbiter comparison](exact_arbiter_comparison.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "diagnosis": len(list((output / "diagnosis").glob("candidate_*.json"))),
        "selected": (output / "selection.json").exists(),
        "confirmation": len(list((output / "confirmation").glob("seed_*.json"))),
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("command", choices=("diagnose", "select", "confirm", "summarize", "status"))
    main.add_argument("--output", default="results/thf_exact_candidate_arbiter")
    main.add_argument("--source", default="results/thf_safe_hybrid")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--indices", default="all")
    main.add_argument("--seeds", default="all")
    main.add_argument("--diagnosis-eval", type=int, default=2_000)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[args.command](args)
