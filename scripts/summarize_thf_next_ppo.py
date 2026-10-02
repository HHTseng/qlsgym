#!/usr/bin/env python
"""Select and summarize the staged PPO study without exact-holdout leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCREEN_PROFILES = (
    "jose_qmdp",
    "jose_aux_01",
    "jose_aux_03",
    "qmdp_tuned_250k",
    "qmdp_tuned_1m",
    "qmdp_tuned_2m",
    "qmdp_lr10_1m",
    "qmdp_lr17_1m",
)
SCREEN_SEEDS = (100, 101)
FINAL_SEEDS = tuple(range(5))

LABELS = {
    "jose_main": "Jose PPO",
    "jose_qmdp": "Jose + qMDP",
    "jose_aux_01": "Jose + auxiliary critic (0.1)",
    "jose_aux_03": "Jose + auxiliary critic (0.3)",
    "qmdp_tuned_250k": "qMDP tuned, 0.25M",
    "qmdp_tuned_1m": "qMDP tuned, 1M",
    "qmdp_tuned_2m": "qMDP tuned, 2M",
    "qmdp_lr10_1m": "qMDP tuned, lr=0.001",
    "qmdp_lr17_1m": "qMDP tuned, lr=0.0017",
}


def load_records(directory: Path) -> list[dict]:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"no JSON records under {directory}")
    return records


def profile(record: dict) -> str:
    return record["job"]["ppo_profile"]


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def validate_contract(records: list[dict]) -> None:
    contracts = {
        (
            item["contract"]["molecule_fingerprint"],
            item["contract"]["library_tag"],
            item["contract"]["manifest_sha256"],
            item["contract"]["p_target"],
            item["contract"]["max_pulses"],
            item["contract"]["rho"],
        )
        for item in records
    }
    if len(contracts) != 1:
        raise ValueError("records do not share one molecular, action, FNO, and task contract")


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(profile(record), []).append(record)
    return groups


def metric_row(name: str, records: list[dict], baseline: dict[int, dict] | None = None) -> dict:
    records = sorted(records, key=lambda item: item["job"]["train_seed"])
    exact = [item["evaluation"]["exact"] for item in records if "exact" in item["evaluation"]]
    fno = [item["evaluation"]["fno"] for item in records]
    fno_failure = mean_sd([item["unfinished_fraction"] for item in fno])
    fno_actions = mean_sd([item["average_actions"] for item in fno])
    row = {
        "profile": name,
        "label": LABELS[name],
        "seeds": [item["job"]["train_seed"] for item in records],
        "runs": len(records),
        "config": {key: value for key, value in records[0]["training"]["config"].items()
                   if key != "seed"},
        "fno_failure_mean": fno_failure[0],
        "fno_failure_sd": fno_failure[1],
        "fno_actions_mean": fno_actions[0],
        "fno_actions_sd": fno_actions[1],
        "wall_clock_hours": sum(item["wall_clock_s"] for item in records) / 3600,
    }
    if not exact:
        return row

    exact_failure = mean_sd([item["unfinished_fraction"] for item in exact])
    exact_actions = mean_sd([item["average_actions"] for item in exact])
    row.update(
        exact_failure_mean=exact_failure[0],
        exact_failure_sd=exact_failure[1],
        exact_actions_mean=exact_actions[0],
        exact_actions_sd=exact_actions[1],
        exact_minus_fno_failure=exact_failure[0] - fno_failure[0],
    )
    if baseline is not None:
        paired = []
        for item in records:
            seed = item["job"]["train_seed"]
            candidate = item["evaluation"]["exact"]
            reference = baseline[seed]["evaluation"]["exact"]
            paired.append((
                candidate["unfinished_fraction"] - reference["unfinished_fraction"],
                candidate["average_actions"] - reference["average_actions"],
            ))
        failure = mean_sd([item[0] for item in paired])
        actions = mean_sd([item[1] for item in paired])
        row.update(
            paired_exact_failure_delta_mean=failure[0],
            paired_exact_failure_delta_sd=failure[1],
            paired_exact_actions_delta_mean=actions[0],
            paired_exact_actions_delta_sd=actions[1],
            paired_seed_wins=sum(a < 0 and b < 0 for a, b in paired),
        )
    return row


def select(args) -> None:
    records = load_records(args.runs)
    validate_contract(records)
    groups = group_records(records)
    if set(groups) != set(SCREEN_PROFILES):
        raise ValueError(f"screen profiles are {sorted(groups)}, expected {sorted(SCREEN_PROFILES)}")
    rows = []
    for name in SCREEN_PROFILES:
        observed = {item["job"]["train_seed"] for item in groups[name]}
        if observed != set(SCREEN_SEEDS):
            raise ValueError(f"{name} has screen seeds {sorted(observed)}")
        for item in groups[name]:
            if set(item["evaluation"]) != {"fno"}:
                raise ValueError("screening records must contain FNO evaluation only")
        rows.append(metric_row(name, groups[name]))
    ranked = sorted(rows, key=lambda row: (row["fno_failure_mean"], row["fno_actions_mean"]))
    result = {
        "status": "selected",
        "selection_dynamics": "downloaded mix FNO only",
        "selection_rule": "two lowest mean FNO failure rates; FNO actions break ties",
        "screen_seeds": list(SCREEN_SEEDS),
        "promoted_profiles": [row["profile"] for row in ranked[:2]],
        "rows": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def markdown(rows: list[dict]) -> str:
    lines = [
        "# PPO beyond Jose under the locked ThF+ contract",
        "",
        "All final rows use five training seeds, 5,000 FNO episodes, and 5,000 exact",
        "holdout episodes per seed. Lower failure and failure-penalized actions are better.",
        "",
        "| PPO | exact failure | exact actions | FNO failure | paired failure vs Jose | paired actions vs Jose | wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["profile"] == "jose_main":
            delta_failure = delta_actions = "--"
            wins = "--"
        else:
            delta_failure = f"{100 * row['paired_exact_failure_delta_mean']:+.2f} pp"
            delta_actions = f"{row['paired_exact_actions_delta_mean']:+.2f}"
            wins = f"{row['paired_seed_wins']}/5"
        lines.append(
            f"| {row['label']} | "
            f"{100 * row['exact_failure_mean']:.2f}% +/- {100 * row['exact_failure_sd']:.2f}% | "
            f"{row['exact_actions_mean']:.2f} +/- {row['exact_actions_sd']:.2f} | "
            f"{100 * row['fno_failure_mean']:.2f}% | {delta_failure} | {delta_actions} | {wins} |"
        )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    labels = [row["label"].replace(" + ", "\n+") for row in rows]
    colors = ["#777777", "#0072B2", "#009E73"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
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
    figure.suptitle("Promoted PPOs versus Jose PPO on exact ThF+ dynamics")
    figure.tight_layout()
    figure.savefig(output / "next_ppo_comparison.png", dpi=180)
    plt.close(figure)


def summarize(args) -> None:
    selection = json.loads(args.selection.read_text())
    promoted = selection["promoted_profiles"]
    if len(promoted) != 2:
        raise ValueError("selection must promote exactly two profiles")
    candidate_records = load_records(args.runs)
    baseline_records = [
        json.loads(path.read_text())
        for path in sorted(args.baseline_runs.glob("final_ppo_jose_main_fno_s*.json"))
    ]
    all_records = baseline_records + candidate_records
    validate_contract(all_records)
    if len(baseline_records) != 5:
        raise ValueError("expected five Jose PPO baseline records")
    baseline = {item["job"]["train_seed"]: item for item in baseline_records}
    groups = group_records(candidate_records)
    if set(groups) != set(promoted):
        raise ValueError(f"final profiles are {sorted(groups)}, expected {sorted(promoted)}")
    for name in promoted:
        observed = {item["job"]["train_seed"] for item in groups[name]}
        if observed != set(FINAL_SEEDS):
            raise ValueError(f"{name} has final seeds {sorted(observed)}")
        if any(set(item["evaluation"]) != {"exact", "fno"} for item in groups[name]):
            raise ValueError("final records must contain both exact and FNO evaluations")

    baseline_row = metric_row("jose_main", baseline_records)
    candidate_rows = [metric_row(name, groups[name], baseline) for name in promoted]
    rows = [baseline_row] + candidate_rows
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "selection": selection,
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
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(rows))
    plot(rows, output)
    print(markdown(rows))


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--runs", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.set_defaults(func=select)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--runs", type=Path, required=True)
    summary_parser.add_argument("--baseline-runs", type=Path, required=True)
    summary_parser.add_argument("--selection", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.set_defaults(func=summarize)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
