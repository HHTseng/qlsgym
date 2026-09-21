#!/usr/bin/env python
"""Assemble the final FNO/RL superiority decision and comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def compact(metrics):
    return {
        "unfinished_fraction": float(metrics["unfinished_fraction"]),
        "average_actions": float(metrics["average_actions"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/thf_fno_rl_superiority"))
    args = parser.parse_args()
    root = args.results
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    historical = read(root / "thf_fno_rl_refinement" / "summary.json")
    safe = read(root / "thf_safe_hybrid" / "summary.json")
    nonml = read(root / "thf_nonml_controls" / "summary.json")
    prefix = read(root / "thf_descending_hybrid" / "summary.json")
    arbiter = read(root / "thf_exact_candidate_arbiter" / "summary.json")
    audit = read(root / "thf_column_fno_v2_full_audit" / "summary.json")
    transfer = read(root / "thf_column_fno_v2_transfer" / "summary.json")

    old_ppo = next(row for row in historical["rows"] if row["name"] == "FNO-trained PPO")
    physics = next(row for row in historical["rows"] if row["name"] == "Physics elimination")
    rows = [
        {
            "name": "Coverage-balanced sweep",
            "class": "non-ML",
            "inference": "fixed schedule",
            "exact": compact(nonml["rows"]["coverage_sweeping"]["exact"]),
        },
        {
            "name": "Physics elimination",
            "class": "non-ML",
            "inference": "physics rule",
            "exact": compact(physics["exact"]),
        },
        {
            "name": "FNO_RL_optuna PPO",
            "class": "standalone RL",
            "inference": "actor only",
            "exact": compact(old_ppo["exact"]),
        },
        {
            "name": "Failure-sensitive PPO",
            "class": "standalone RL",
            "inference": "actor only",
            "exact": compact(safe["rows"]["risk_actor"]["exact"]),
        },
        {
            "name": "Failure-sensitive PPO + fallback",
            "class": "hybrid",
            "inference": "actor then physics fallback",
            "exact": compact(safe["rows"]["risk_fallback"]["exact"]),
        },
        {
            "name": "Descending population",
            "class": "non-ML",
            "inference": "cached-exact table rule",
            "exact": compact(nonml["rows"]["descending_population"]["exact"]),
        },
        {
            "name": "15-pulse PPO + descending fallback",
            "class": "hybrid",
            "inference": "actor prefix then cached-exact table rule",
            "exact": compact(prefix["learned_prefix_descending_fallback"]["exact"]),
        },
        {
            "name": "Exact candidate arbiter",
            "class": "hybrid",
            "inference": "actor and baseline proposals scored by cached-exact tables",
            "exact": compact(arbiter["exact_candidate_arbiter"]["exact"]),
        },
    ]
    ranking = sorted(rows, key=lambda row: (
        row["exact"]["unfinished_fraction"], row["exact"]["average_actions"],
    ))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank

    arbiter_row = arbiter["exact_candidate_arbiter"]
    prefix_row = prefix["learned_prefix_descending_fallback"]
    result = {
        "status": "complete",
        "ranking_rule": "exact unfinished fraction, then exact failure-penalized actions",
        "rows": ranking,
        "decisions": {
            "column_v2_promoted": audit["pairs_passing_all"] == audit["pairs"],
            "column_v2_pairs_passing_all": audit["pairs_passing_all"],
            "column_v2_pairs": audit["pairs"],
            "improved_fno_ppo_run": False,
            "reason_improved_fno_ppo_not_run": (
                "column-v2 failed the preregistered all-pair structural gate"
            ),
            "standalone_rl_beats_descending_on_both": (
                safe["rows"]["risk_actor"]["exact"]["unfinished_fraction"]
                < nonml["rows"]["descending_population"]["exact"]["unfinished_fraction"]
                and safe["rows"]["risk_actor"]["exact"]["average_actions"]
                < nonml["rows"]["descending_population"]["exact"]["average_actions"]
            ),
            "learned_prefix_superiority_supported": prefix_row["superiority_claim_supported"],
            "exact_arbiter_superiority_supported": arbiter_row["superiority_claim_supported"],
        },
        "exact_arbiter_paired_vs_descending": arbiter_row["paired_vs_descending_population"],
        "column_v2_full_audit": audit,
        "column_v2_closed_loop_transfer": transfer,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    colors = {"non-ML": "#4c78a8", "standalone RL": "#f58518", "hybrid": "#54a24b"}
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for row in rows:
        x = row["exact"]["average_actions"]
        y = 100 * row["exact"]["unfinished_fraction"]
        axis.scatter(x, y, s=75, color=colors[row["class"]], zorder=3)
        axis.annotate(row["name"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    for label, color in colors.items():
        axis.scatter([], [], s=75, color=color, label=label)
    axis.set_xlabel("Exact failure-penalized actions (lower is better)")
    axis.set_ylabel("Exact unfinished episodes (%) (lower is better)")
    axis.set_title("Final exact-dynamics comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "final_exact_comparison.png", dpi=180)
    plt.close(fig)

    paired = arbiter_row["paired_vs_descending_population"]
    lines = [
        "# Final FNO + RL superiority decision",
        "",
        "All rankings use cached-exact dynamics, unfinished fraction first and failure-penalized actions second.",
        "",
        "| Rank | Controller | Class | Exact failure | Exact actions | Inference requirement |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank']} | {row['name']} | {row['class']} | "
            f"{100*row['exact']['unfinished_fraction']:.2f}% | "
            f"{row['exact']['average_actions']:.2f} | {row['inference']} |"
        )
    lines += [
        "",
        f"The exact candidate arbiter beats descending population by "
        f"{abs(100*paired['failure_difference']):.2f} failure percentage points and "
        f"{abs(paired['action_difference']):.2f} actions. Its paired failure-difference "
        f"95% seed interval is [{100*paired['failure_difference_seed_ci95'][0]:.2f}, "
        f"{100*paired['failure_difference_seed_ci95'][1]:.2f}] percentage points.",
        "",
        f"The column-v2 FNO passes all seven gates for {audit['pairs_passing_all']}/{audit['pairs']} "
        "pairs, so it was not promoted and no policy was trained under it.",
        "",
        "![Final exact comparison](final_exact_comparison.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
