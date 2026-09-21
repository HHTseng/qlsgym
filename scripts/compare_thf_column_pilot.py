#!/usr/bin/env python
"""Decide whether the four-pair transfer-column pilot merits a full retrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_thf_fno_manifest import GATES, pair_path


PAIRS = ((0, "+"), (0, "-"), (1, "+"), (1, "-"))
ACCURACY_GATES = (
    "off_joint_tv_p95",
    "branch_mass_abs_error_p95",
    "conditional_tv_mass_ge_1e-2_p95",
    "conditional_tv_mass_ge_1e-3_p95",
    "block_local_termination_error",
)


def load_pairs(directory: Path) -> dict:
    rows = {}
    for block, sigma in PAIRS:
        path = pair_path(directory, block, sigma)
        if not path.exists():
            raise FileNotFoundError(path)
        rows[(block, sigma)] = json.loads(path.read_text())
    return rows


def safe_ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def compare(current: dict, candidate: dict) -> dict:
    rows = []
    ratios_by_gate = {gate: [] for gate in ACCURACY_GATES}
    stable = True
    for pair in PAIRS:
        before = current[pair]["gate_metrics"]
        after = candidate[pair]["gate_metrics"]
        ratios = {}
        for gate in GATES:
            ratios[gate] = safe_ratio(after[gate], before[gate])
        for gate in ACCURACY_GATES:
            ratios_by_gate[gate].append(ratios[gate])
            # A candidate may vary inside an already acceptable region.  Above
            # the engineering gate, however, a >25% regression blocks scaling.
            stable &= after[gate] <= max(GATES[gate], 1.25 * before[gate])
        rows.append(
            {
                "block": pair[0],
                "sigma": pair[1],
                "current": before,
                "candidate": after,
                "candidate_over_current": ratios,
            }
        )

    median_ratios = {
        gate: float(np.median(values)) for gate, values in ratios_by_gate.items()
    }
    finite_accuracy_ratios = [
        ratio
        for values in ratios_by_gate.values()
        for ratio in values
        if np.isfinite(ratio)
    ]
    overall_ratio = float(np.median(finite_accuracy_ratios))
    structural = all(
        candidate[pair]["gate_metrics"][gate] <= GATES[gate]
        for pair in PAIRS
        for gate in ("tau0_identity_tv_max", "input_linearity_tv_p95")
    )
    conditional = all(
        median_ratios[gate] <= 0.75
        for gate in (
            "conditional_tv_mass_ge_1e-2_p95",
            "conditional_tv_mass_ge_1e-3_p95",
        )
    )
    metrics_improved = sum(
        ratio <= 0.75 for ratio in median_ratios.values()
    )
    criteria = {
        "identity_and_linearity_gates_pass_all_four_pairs": structural,
        "median_accuracy_error_reduction_ge_30pct": overall_ratio <= 0.70,
        "both_conditional_tv_medians_reduce_ge_25pct": conditional,
        "at_least_three_of_five_accuracy_metrics_reduce_ge_25pct": metrics_improved >= 3,
        "no_material_pair_metric_regression": bool(stable),
    }
    return {
        "decision": "promote" if all(criteria.values()) else "stop",
        "passes": all(criteria.values()),
        "criteria": criteria,
        "median_candidate_over_current": median_ratios,
        "overall_median_accuracy_ratio": overall_ratio,
        "pairs": rows,
        "rule": (
            "Promote only when identity and input linearity pass by construction, "
            "the median accuracy error falls at least 30%, both conditional-TV "
            "metrics fall at least 25%, at least 3/5 accuracy metrics fall at least "
            "25%, and no pair/metric regresses materially."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load_pairs(args.current), load_pairs(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passes"] else 2)


if __name__ == "__main__":
    main()
