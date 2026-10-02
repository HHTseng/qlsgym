#!/usr/bin/env python
"""Select and summarize scale-aware SAC/DDQN transfer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCREEN = {
    "sac_discrete": ("sac_refined_1m", "sac_refined_2m"),
    "ddqn": (
        "ddqn_optuna_1m",
        "ddqn_optuna_2m",
        "ddqn_scaled20_1m",
        "ddqn_raw20_1m",
    ),
}
SCREEN_SEEDS = {100, 101}
FINAL_SEEDS = set(range(5))
LABELS = {
    "sac_refined_1m": "SAC refined, 1M",
    "sac_refined_2m": "SAC refined, 2M",
    "ddqn_optuna_1m": "DDQN Optuna, 1M",
    "ddqn_optuna_2m": "DDQN Optuna, 2M",
    "ddqn_scaled20_1m": "DDQN scaled, 1M",
    "ddqn_raw20_1m": "DDQN raw-belief scaled, 1M",
}


def load_records(directory: Path) -> list[dict]:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"no JSON records under {directory}")
    return records


def key(record: dict) -> tuple[str, str]:
    return record["job"]["agent"], record["job"]["offpolicy_profile"]


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def validate_contract(records: list[dict]) -> None:
    contracts = {
        (
            row["contract"]["molecule_fingerprint"],
            row["contract"]["library_tag"],
            row["contract"]["manifest_sha256"],
            row["contract"]["p_target"],
            row["contract"]["max_pulses"],
            row["contract"]["rho"],
        )
        for row in records
    }
    if len(contracts) != 1:
        raise ValueError("records mix physical, action, FNO, or task contracts")


def group(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        result.setdefault(key(record), []).append(record)
    return result


def row(agent: str, profile: str, records: list[dict], baseline=None) -> dict:
    records = sorted(records, key=lambda item: item["job"]["train_seed"])
    fno = [item["evaluation"]["fno"] for item in records]
    result = {
        "agent": agent,
        "profile": profile,
        "label": LABELS[profile],
        "seeds": [item["job"]["train_seed"] for item in records],
        "config": {
            name: value
            for name, value in records[0]["training"]["config"].items()
            if name != "seed"
        },
        "fno_failure_mean": mean_sd([item["unfinished_fraction"] for item in fno])[0],
        "fno_failure_sd": mean_sd([item["unfinished_fraction"] for item in fno])[1],
        "fno_actions_mean": mean_sd([item["average_actions"] for item in fno])[0],
        "fno_actions_sd": mean_sd([item["average_actions"] for item in fno])[1],
        "wall_clock_hours": sum(item["wall_clock_s"] for item in records) / 3600,
    }
    exact = [item["evaluation"]["exact"] for item in records if "exact" in item["evaluation"]]
    if not exact:
        return result
    result.update(
        exact_failure_mean=mean_sd([item["unfinished_fraction"] for item in exact])[0],
        exact_failure_sd=mean_sd([item["unfinished_fraction"] for item in exact])[1],
        exact_actions_mean=mean_sd([item["average_actions"] for item in exact])[0],
        exact_actions_sd=mean_sd([item["average_actions"] for item in exact])[1],
    )
    result["exact_minus_fno_failure"] = (
        result["exact_failure_mean"] - result["fno_failure_mean"]
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
        result.update(
            paired_exact_failure_delta_mean=mean_sd([item[0] for item in paired])[0],
            paired_exact_failure_delta_sd=mean_sd([item[0] for item in paired])[1],
            paired_exact_actions_delta_mean=mean_sd([item[1] for item in paired])[0],
            paired_exact_actions_delta_sd=mean_sd([item[1] for item in paired])[1],
            paired_seed_wins=sum(a < 0 and b < 0 for a, b in paired),
        )
    return result


def select(args) -> None:
    records = load_records(args.runs)
    validate_contract(records)
    groups = group(records)
    expected = {(agent, profile) for agent, profiles in SCREEN.items() for profile in profiles}
    if set(groups) != expected:
        raise ValueError(f"screen groups are {sorted(groups)}, expected {sorted(expected)}")
    rows = []
    selected = {}
    for agent, profiles in SCREEN.items():
        candidates = []
        for profile in profiles:
            members = groups[(agent, profile)]
            if {item["job"]["train_seed"] for item in members} != SCREEN_SEEDS:
                raise ValueError(f"{profile} does not have both screen seeds")
            if any(set(item["evaluation"]) != {"fno"} for item in members):
                raise ValueError("screening records must contain FNO evaluation only")
            candidate = row(agent, profile, members)
            rows.append(candidate)
            candidates.append(candidate)
        winner = min(
            candidates,
            key=lambda item: (item["fno_failure_mean"], item["fno_actions_mean"]),
        )
        selected[agent] = winner["profile"]
    report = {
        "status": "selected",
        "selection_dynamics": "downloaded mix FNO only",
        "selection_rule": "lowest paired-seed FNO failure; actions break ties",
        "screen_seeds": sorted(SCREEN_SEEDS),
        "selected_profiles": selected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Branch-aware off-policy transfer under the locked ThF+ contract",
        "",
        "Each row uses five training seeds and 5,000 episodes per dynamics and seed.",
        "Lower failure and failure-penalized actions are better.",
        "",
        "| Agent | Exact failure | Exact actions | FNO failure | failure vs branch | actions vs branch | wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['label']} | "
            f"{100 * item['exact_failure_mean']:.2f}% +/- {100 * item['exact_failure_sd']:.2f}% | "
            f"{item['exact_actions_mean']:.2f} +/- {item['exact_actions_sd']:.2f} | "
            f"{100 * item['fno_failure_mean']:.2f}% | "
            f"{100 * item['paired_exact_failure_delta_mean']:+.2f} pp | "
            f"{item['paired_exact_actions_delta_mean']:+.2f} | "
            f"{item['paired_seed_wins']}/5 |"
        )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["label"].replace(", ", "\n") for item in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    axes[0].bar(
        x, [100 * item["exact_failure_mean"] for item in rows],
        yerr=[100 * item["exact_failure_sd"] for item in rows], capsize=4,
        color=["#D55E00", "#CC79A7"],
    )
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(
        x, [item["exact_actions_mean"] for item in rows],
        yerr=[item["exact_actions_sd"] for item in rows], capsize=4,
        color=["#D55E00", "#CC79A7"],
    )
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Transferred SAC and DDQN profiles on exact ThF+ dynamics")
    figure.tight_layout()
    figure.savefig(output / "offpolicy_transfer_comparison.png", dpi=180)
    plt.close(figure)


def summarize(args) -> None:
    selection = json.loads(args.selection.read_text())
    records = load_records(args.runs)
    baselines = load_records(args.baseline_runs)
    validate_contract(records + baselines)
    groups = group(records)
    rows = []
    for agent, profile in selection["selected_profiles"].items():
        members = groups.get((agent, profile), [])
        if {item["job"]["train_seed"] for item in members} != FINAL_SEEDS:
            raise ValueError(f"{profile} does not have five final seeds")
        if any(set(item["evaluation"]) != {"exact", "fno"} for item in members):
            raise ValueError("final records must contain exact and FNO evaluations")
        references = {
            item["job"]["train_seed"]: item
            for item in baselines if item["job"]["agent"] == agent
        }
        if set(references) != FINAL_SEEDS:
            raise ValueError(f"missing paired branch baseline for {agent}")
        rows.append(row(agent, profile, members, references))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "complete",
        "selection": selection,
        "physical_contract": {
            "molecule": "ThF+", "states": 192, "actions": 312,
            "p_target": 0.98, "horizon": 80, "rho": 0.0,
            "manifest_fingerprint": records[0]["contract"]["manifest_fingerprint"],
            "manifest_sha256": records[0]["contract"]["manifest_sha256"],
            "evaluation_episodes_per_dynamics_per_seed": 5000,
        },
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(rows))
    plot(rows, output)
    print(markdown(rows))


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    command = sub.add_parser("select")
    command.add_argument("--runs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=select)
    command = sub.add_parser("summarize")
    command.add_argument("--runs", type=Path, required=True)
    command.add_argument("--baseline-runs", type=Path, required=True)
    command.add_argument("--selection", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=summarize)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
