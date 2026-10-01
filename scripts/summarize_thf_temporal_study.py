#!/usr/bin/env python
"""Compare the six-pair temporal-FNO pilot and decide whether to scale it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_thf_fno_manifest import GATES


PILOT_PAIRS = ((0, "+"), (0, "-"), (1, "+"), (1, "-"), (9, "+"), (11, "+"))


def pair_path(root: Path, block: int, sigma: str) -> Path:
    label = "sp" if sigma == "+" else "sm"
    return root / "pairs" / f"block{block}_{label}.json"


def load_study(specification: str) -> tuple[str, Path, list[dict]]:
    name, raw_path = specification.split("=", 1)
    root = Path(raw_path)
    paths = [pair_path(root, block, sigma) for block, sigma in PILOT_PAIRS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{name} is missing pilot audits:\n" + "\n".join(missing))
    return name, root, [json.loads(path.read_text()) for path in paths]


def aggregate(name: str, root: Path, pairs: list[dict]) -> dict:
    worst = {}
    for metric, threshold in GATES.items():
        pair = max(pairs, key=lambda item: item["gate_metrics"][metric])
        value = pair["gate_metrics"][metric]
        worst[metric] = {
            "value": value,
            "threshold": threshold,
            "block": pair["block"],
            "sigma": pair["sigma"],
            "passes": value <= threshold,
        }
    derivatives = [
        pair.get("temporal_metrics", {}).get("derivative_l1", {}).get("median")
        for pair in pairs
    ]
    spectra = [
        pair.get("temporal_metrics", {}).get("spectral_relative", {}).get("median")
        for pair in pairs
    ]
    return {
        "name": name,
        "audit": str(root.resolve()),
        "pairs": len(pairs),
        "pairs_passing_all": sum(item["passes_all"] for item in pairs),
        "all_pass": all(item["passes_all"] for item in pairs),
        "worst_by_gate": worst,
        "median_derivative_l1": float(np.median([x for x in derivatives if x is not None]))
        if any(x is not None for x in derivatives)
        else None,
        "median_spectral_relative": float(np.median([x for x in spectra if x is not None]))
        if any(x is not None for x in spectra)
        else None,
    }


def plot_gate_comparison(studies: list[dict], destination: Path) -> None:
    metrics = list(GATES)
    values = np.asarray(
        [
            [study["worst_by_gate"][metric]["value"] / GATES[metric] for metric in metrics]
            for study in studies
        ]
    )
    values = np.maximum(values, 1e-8)
    x = np.arange(len(metrics))
    width = 0.8 / len(studies)
    figure, axis = plt.subplots(figsize=(13, 5.5))
    for index, study in enumerate(studies):
        axis.bar(
            x + (index - (len(studies) - 1) / 2) * width,
            values[index],
            width,
            label=study["name"],
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="gate")
    axis.set_yscale("log")
    axis.set_ylabel("worst pilot value / acceptance threshold")
    axis.set_xticks(x, [metric.replace("_", "\n") for metric in metrics], fontsize=8)
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_hard_pair(studies_with_pairs, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, _root, pairs in studies_with_pairs:
        pair = next(item for item in pairs if (item["block"], item["sigma"]) == (11, "+"))
        values = pair.get("temporal_metrics", {}).get("column_tv_by_tau_mean")
        if values:
            axis.plot(np.linspace(0.0, 1.0, len(values)), values, label=name)
    axis.set_xlabel(r"normalized pulse time $\tau/\tau_{\max}$")
    axis.set_ylabel("mean transfer-column TV")
    axis.set_title(r"hard pilot pair $(f,\sigma)=(11,+)$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    loaded = [load_study(item) for item in args.study]
    studies = [aggregate(*item) for item in loaded]
    by_name = {study["name"]: study for study in studies}
    candidates = [by_name[name] for name in args.candidate]
    promoted = [study for study in candidates if study["all_pass"]]
    selected = None
    if promoted:
        selected = min(
            promoted,
            key=lambda study: sum(
                study["worst_by_gate"][metric]["value"] / threshold
                for metric, threshold in GATES.items()
            ),
        )["name"]
    decision = {
        "status": "complete",
        "pilot_pairs": [f"{block},{sigma}" for block, sigma in PILOT_PAIRS],
        "studies": studies,
        "candidate_names": args.candidate,
        "promotion_rule": "all six hard-pilot pairs pass every preregistered column_v2 gate",
        "promoted": selected is not None,
        "selected_candidate": selected,
    }
    (args.output / "summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    plot_gate_comparison(studies, args.output / "pilot_gate_comparison.png")
    plot_hard_pair(loaded, args.output / "hard_pair_temporal_error.png")

    lines = [
        "# Temporal FNO six-pair pilot",
        "",
        "| model | pairs passing all gates | worst branch-mass P95 | worst conditional TV (mass >= 1e-2) | median derivative error | median spectral error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def number(value):
        return "n/a" if value is None else f"{value:.6g}"

    for study in studies:
        derivative = study["median_derivative_l1"]
        spectral = study["median_spectral_relative"]
        lines.append(
            f"| {study['name']} | {study['pairs_passing_all']}/6 | "
            f"{study['worst_by_gate']['branch_mass_abs_error_p95']['value']:.6g} | "
            f"{study['worst_by_gate']['conditional_tv_mass_ge_1e-2_p95']['value']:.6g} | "
            f"{number(derivative)} | {number(spectral)} |"
        )
    lines.extend(
        [
            "",
            f"Promotion: **{'yes' if selected else 'no'}**"
            + (f" (`{selected}`)" if selected else ""),
            "",
            "The full 24-pair study is permitted only after this pilot gate passes.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
