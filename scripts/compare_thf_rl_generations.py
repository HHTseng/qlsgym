#!/usr/bin/env python
"""Compare the original and external-mix five-seed ThF RL generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGENTS = ("sweeping", "random", "physics_elimination", "ppo", "sac_discrete", "ddqn")
LABELS = {
    "sweeping": "Sweeping",
    "random": "Random",
    "physics_elimination": "Physics elimination",
    "ppo": "PPO",
    "sac_discrete": "Discrete SAC",
    "ddqn": "Double DQN",
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def rows_by_agent(summary: dict) -> dict:
    return {row["agent"]: row for row in summary["rows"]}


def mean_curves(directory: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {agent: [] for agent in AGENTS}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        agent = record["job"]["agent"]
        result = record["evaluation"]["exact"]
        length = np.asarray(result["lengths"])
        success = np.asarray(result["successes"], dtype=bool)
        grouped[agent].append(
            np.asarray([100 * ((length <= h) & success).mean() for h in range(81)])
        )
    return {agent: np.mean(curves, axis=0) for agent, curves in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("results/thf_rl_final"))
    parser.add_argument("--mix", type=Path, default=Path("results/thf_rl_mix_tara"))
    args = parser.parse_args()

    original = load_summary(args.original / "summary.json")
    mix = load_summary(args.mix / "summary.json")
    old_rows, new_rows = rows_by_agent(original), rows_by_agent(mix)
    comparison = []
    for agent in AGENTS:
        old, new = old_rows[agent], new_rows[agent]
        comparison.append(
            {
                "agent": agent,
                "original_exact_average_actions": old["exact"]["average_actions"],
                "mix_exact_average_actions": new["exact"]["average_actions"],
                "actions_delta_mix_minus_original": (
                    new["exact"]["average_actions"] - old["exact"]["average_actions"]
                ),
                "original_exact_failure": old["exact"]["unfinished_fraction"],
                "mix_exact_failure": new["exact"]["unfinished_fraction"],
                "failure_delta_mix_minus_original": (
                    new["exact"]["unfinished_fraction"] - old["exact"]["unfinished_fraction"]
                ),
                "original_transfer_failure_gap": old["sim_to_exact_failure_gap"],
                "mix_transfer_failure_gap": new["sim_to_exact_failure_gap"],
            }
        )
    payload = {
        "original_manifest": original["contract"]["manifest_tag"],
        "mix_manifest": mix["contract"]["manifest_tag"],
        "rows": comparison,
    }
    (args.mix / "generation_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "## Original production FNO versus downloaded `mix` FNO",
        "",
        "All entries use the same five-seed, one-million-transition, 5000-exact-rollout contract. Deltas are `mix - original`; negative is better for both exact metrics.",
        "",
        "| Controller | Original exact actions | `mix` exact actions | Delta | Original exact failure | `mix` exact failure | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {LABELS[row['agent']]} | {row['original_exact_average_actions']:.2f} | "
            f"{row['mix_exact_average_actions']:.2f} | {row['actions_delta_mix_minus_original']:+.2f} | "
            f"{100*row['original_exact_failure']:.2f}% | {100*row['mix_exact_failure']:.2f}% | "
            f"{100*row['failure_delta_mix_minus_original']:+.2f} pp |"
        )
    lines += [
        "",
        "Transfer failure gap is exact failure minus FNO failure. Negative values mean the FNO environment is pessimistic.",
        "",
        "| Controller | Original transfer gap | `mix` transfer gap |",
        "|---|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {LABELS[row['agent']]} | {100*row['original_transfer_failure_gap']:+.2f} pp | "
            f"{100*row['mix_transfer_failure_gap']:+.2f} pp |"
        )
    (args.mix / "generation_comparison.md").write_text("\n".join(lines) + "\n")

    x = np.arange(len(AGENTS))
    width = 0.38
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    old_actions = [old_rows[a]["exact"]["average_actions"] for a in AGENTS]
    new_actions = [new_rows[a]["exact"]["average_actions"] for a in AGENTS]
    old_failure = [100 * old_rows[a]["exact"]["unfinished_fraction"] for a in AGENTS]
    new_failure = [100 * new_rows[a]["exact"]["unfinished_fraction"] for a in AGENTS]
    for axis, old_value, new_value, ylabel in (
        (axes[0, 0], old_actions, new_actions, "Exact average actions\nLower better"),
        (axes[0, 1], old_failure, new_failure, "Exact failure rate (%)\nLower better"),
    ):
        axis.bar(x - width / 2, old_value, width, label="original rlprod120v2", color="#999999")
        axis.bar(x + width / 2, new_value, width, label="downloaded mix", color="#0072B2")
        axis.set_xticks(x, [LABELS[a] for a in AGENTS], rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)

    old_curves = mean_curves(args.original / "runs")
    new_curves = mean_curves(args.mix / "runs")
    colors = plt.cm.tab10(np.linspace(0, 1, len(AGENTS)))
    for axis, curves, title in (
        (axes[1, 0], old_curves, "Original production FNO"),
        (axes[1, 1], new_curves, "Downloaded `mix` FNO"),
    ):
        for color, agent in zip(colors, AGENTS):
            axis.plot(range(81), curves[agent], label=LABELS[agent], color=color)
        axis.set(xlabel="Pulses applied", ylabel="Finished exact episodes (%)", title=title,
                 xlim=(0, 80), ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("ThF+ RL generations: identical training and exact-holdout contract")
    figure.tight_layout()
    figure.savefig(args.mix / "thf_mix_vs_rlprod120v2.png", dpi=180)
    figure.savefig(args.mix / "thf_mix_vs_rlprod120v2.pdf")
    plt.close(figure)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
