#!/usr/bin/env python
"""Summarize the matched Jose-PPO comparison under the ThF+ H=80 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ORDER = (
    "branch",
    "jose_matched_fno",
    "jose_main_fno",
    "jose_main_exact",
    "jose_tuned_fno",
)

EXPECTED_SEEDS = {
    "branch": set(range(5)),
    "jose_matched_fno": set(range(5)),
    "jose_main_fno": set(range(5)),
    "jose_main_exact": set(range(3)),
    "jose_tuned_fno": set(range(5)),
}

DISPLAY = {
    "branch": "Branch PPO",
    "jose_matched_fno": "Jose GAE\nmatched",
    "jose_main_fno": "Jose PPO\nFNO-selected",
    "jose_main_exact": "Jose PPO\nexact-selected",
    "jose_tuned_fno": "Jose-tuned PPO\nFNO-selected",
}

PLOT_LABELS = {
    "branch": "Branch\nPPO",
    "jose_matched_fno": "GAE-only\nmatched",
    "jose_main_fno": "Jose PPO\nFNO selection",
    "jose_main_exact": "Jose PPO\nexact selection",
    "jose_tuned_fno": "Jose-tuned\nPPO",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def key(record: dict) -> str:
    job = record["job"]
    profile = job.get("ppo_profile") or "branch"
    if profile == "branch":
        return "branch"
    return f"{profile}_{job['ppo_selection']}"


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def record_row(name: str, records: list[dict], baseline: dict[int, dict]) -> dict:
    records = sorted(records, key=lambda item: item["job"]["train_seed"])
    exact = [item["evaluation"]["exact"] for item in records]
    fno = [item["evaluation"]["fno"] for item in records]
    exact_failure = mean_std([item["unfinished_fraction"] for item in exact])
    exact_actions = mean_std([item["average_actions"] for item in exact])
    fno_failure = mean_std([item["unfinished_fraction"] for item in fno])
    fno_actions = mean_std([item["average_actions"] for item in fno])

    paired = []
    for item in records:
        seed = item["job"]["train_seed"]
        if seed in baseline:
            reference = baseline[seed]["evaluation"]["exact"]
            candidate = item["evaluation"]["exact"]
            paired.append((
                candidate["unfinished_fraction"] - reference["unfinished_fraction"],
                candidate["average_actions"] - reference["average_actions"],
            ))
    delta_failure = mean_std([item[0] for item in paired]) if paired else (0.0, 0.0)
    delta_actions = mean_std([item[1] for item in paired]) if paired else (0.0, 0.0)
    config = dict(records[0]["training"]["config"])
    config.pop("seed", None)
    return {
        "id": name,
        "label": DISPLAY[name].replace("\n", " "),
        "seeds": [item["job"]["train_seed"] for item in records],
        "runs": len(records),
        "selection_dynamics": records[0]["job"].get("ppo_selection", "exact-final-only"),
        "config": config,
        "exact_failure_mean": exact_failure[0],
        "exact_failure_sd": exact_failure[1],
        "exact_actions_mean": exact_actions[0],
        "exact_actions_sd": exact_actions[1],
        "fno_failure_mean": fno_failure[0],
        "fno_failure_sd": fno_failure[1],
        "fno_actions_mean": fno_actions[0],
        "fno_actions_sd": fno_actions[1],
        "exact_minus_fno_failure": exact_failure[0] - fno_failure[0],
        "paired_exact_failure_delta_vs_branch_mean": delta_failure[0],
        "paired_exact_failure_delta_vs_branch_sd": delta_failure[1],
        "paired_exact_actions_delta_vs_branch_mean": delta_actions[0],
        "paired_exact_actions_delta_vs_branch_sd": delta_actions[1],
        "paired_seed_wins": sum(a < 0 and b < 0 for a, b in paired),
        "paired_seed_count": len(paired),
        "wall_clock_hours": sum(item["wall_clock_s"] for item in records) / 3600.0,
    }


def paired_comparison(candidate: list[dict], reference: list[dict]) -> dict:
    """Compare exact-holdout metrics on shared training seeds."""
    candidate = {item["job"]["train_seed"]: item for item in candidate}
    reference = {item["job"]["train_seed"]: item for item in reference}
    seeds = sorted(candidate.keys() & reference.keys())
    failure = []
    actions = []
    for seed in seeds:
        c = candidate[seed]["evaluation"]["exact"]
        r = reference[seed]["evaluation"]["exact"]
        failure.append(c["unfinished_fraction"] - r["unfinished_fraction"])
        actions.append(c["average_actions"] - r["average_actions"])
    failure_mean, failure_sd = mean_std(failure)
    actions_mean, actions_sd = mean_std(actions)
    return {
        "seeds": seeds,
        "failure_delta_mean": failure_mean,
        "failure_delta_sd": failure_sd,
        "actions_delta_mean": actions_mean,
        "actions_delta_sd": actions_sd,
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Jose PPO comparison under the locked ThF+ contract",
        "",
        "All rows use the downloaded `mix` FNO for training, `H=80`, `rho=0`,",
        "5,000 FNO episodes, and 5,000 exact holdout episodes per seed.",
        "Lower unfinished fraction and failure-penalized actions are better.",
        "",
        "| PPO | seeds | selection | exact failure | exact actions | FNO failure | paired failure delta vs branch |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['runs']} | {row['selection_dynamics']} | "
            f"{100 * row['exact_failure_mean']:.2f}% +/- {100 * row['exact_failure_sd']:.2f}% | "
            f"{row['exact_actions_mean']:.2f} +/- {row['exact_actions_sd']:.2f} | "
            f"{100 * row['fno_failure_mean']:.2f}% | "
            f"{100 * row['paired_exact_failure_delta_vs_branch_mean']:+.2f} pp |"
        )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    labels = [PLOT_LABELS[row["id"]] for row in rows]
    colors = ["#777777", "#56B4E9", "#0072B2", "#CC79A7", "#009E73"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes[0].bar(
        x,
        [100 * row["exact_failure_mean"] for row in rows],
        yerr=[100 * row["exact_failure_sd"] for row in rows],
        color=colors,
        capsize=4,
    )
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(
        x,
        [row["exact_actions_mean"] for row in rows],
        yerr=[row["exact_actions_sd"] for row in rows],
        color=colors,
        capsize=4,
    )
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("PPO comparison on the locked ThF+ FNO task")
    figure.tight_layout()
    figure.savefig(output / "jose_ppo_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--baseline-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_records = [load(path) for path in sorted(args.baseline_runs.glob("final_ppo_s*.json"))]
    comparison_records = [load(path) for path in sorted(args.runs.glob("*.json"))]
    if len(baseline_records) != 5:
        raise ValueError(f"expected five branch PPO records, found {len(baseline_records)}")
    all_records = baseline_records + comparison_records
    contracts = {
        (
            item["contract"]["molecule_fingerprint"],
            item["contract"]["library_tag"],
            item["contract"]["manifest_sha256"],
            item["contract"]["p_target"],
            item["contract"]["max_pulses"],
            item["contract"]["rho"],
        )
        for item in all_records
    }
    if len(contracts) != 1:
        raise ValueError("comparison records do not share one physical contract")

    groups: dict[str, list[dict]] = {"branch": baseline_records}
    for item in comparison_records:
        groups.setdefault(key(item), []).append(item)
    missing = [name for name in ORDER if name not in groups]
    if missing:
        raise ValueError(f"missing comparison groups: {missing}")
    for name, expected in EXPECTED_SEEDS.items():
        observed = {item["job"]["train_seed"] for item in groups[name]}
        if observed != expected:
            raise ValueError(
                f"{name} has seeds {sorted(observed)}, expected {sorted(expected)}"
            )
    baseline = {item["job"]["train_seed"]: item for item in baseline_records}
    rows = [record_row(name, groups[name], baseline) for name in ORDER]

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "physical_contract": {
            "molecule": "ThF+",
            "states": 192,
            "actions": 312,
            "p_target": 0.98,
            "horizon": 80,
            "rho": 0.0,
            "manifest_fingerprint": all_records[0]["contract"]["manifest_fingerprint"],
            "manifest_sha256": all_records[0]["contract"]["manifest_sha256"],
            "evaluation_episodes_per_dynamics_per_seed": 5000,
        },
        "published_main_reference": {
            "comparable": False,
            "reason": "main used H=800 and rho=2, whereas this comparison fixes H=80 and rho=0",
            "success": 0.99,
            "mean_pulses": 124.05,
            "training_steps": 2_000_000,
        },
        "paired_comparisons": {
            "standard_gae_only_vs_branch": paired_comparison(
                groups["jose_matched_fno"], groups["branch"]
            ),
            "full_jose_vs_gae_only": paired_comparison(
                groups["jose_main_fno"], groups["jose_matched_fno"]
            ),
            "exact_vs_fno_snapshot_selection": paired_comparison(
                groups["jose_main_exact"], groups["jose_main_fno"]
            ),
            "tuned_vs_full_jose": paired_comparison(
                groups["jose_tuned_fno"], groups["jose_main_fno"]
            ),
        },
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(rows))
    plot(rows, args.output)
    print(markdown(rows))


if __name__ == "__main__":
    main()
