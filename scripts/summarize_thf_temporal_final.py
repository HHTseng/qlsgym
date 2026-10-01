#!/usr/bin/env python
"""Assemble the final temporal-FNO experiment report from persisted artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


TAGS = {
    "column_retrained": "seqfno_column_pilot",
    "temporal_1layer": "seqfno_temporal1_pilot",
    "temporal_2layer": "seqfno_temporal2_pilot",
    "temporal_2layer_d1": "seqfno_temporal2_d1_pilot",
    "temporal_2layer_d1spec": "seqfno_temporal2_d1spec_pilot",
    "shuffled_time": "seqfno_shuffled_pilot",
    "pure_transformer": "seqfno_transformer_pilot",
    "temporal_2layer_long": "seqfno_temporal2_long_pilot",
}
PAIRS = ((0, "+"), (0, "-"), (1, "+"), (1, "-"), (9, "+"), (11, "+"))
KEY_METRICS = (
    "branch_mass_abs_error_p95",
    "conditional_tv_mass_ge_1e-2_p95",
    "conditional_tv_mass_ge_1e-3_p95",
    "block_local_termination_error",
)


def run_dir(work: Path, tag: str, block: int, sigma: str) -> Path:
    return work / "runs" / f"{tag}_{'sp' if sigma == '+' else 'sm'}_block{block}"


def training_summary(work: Path, label: str, tag: str) -> dict:
    rows = []
    gates = []
    for block, sigma in PAIRS:
        directory = run_dir(work, tag, block, sigma)
        summary = json.loads((directory / "summary.json").read_text())
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        raw_gate = checkpoint["state_dict"].get("attention_gate_raw")
        if raw_gate is not None:
            gates.append(float(torch.sigmoid(raw_gate)))
        rows.append(summary)
    return {
        "label": label,
        "tag": tag,
        "models": len(rows),
        "architecture": sorted({row["architecture"] for row in rows}),
        "parameters_median": int(np.median([row["n_parameters"] for row in rows])),
        "best_validation_loss_median": float(np.median([row["best_val_loss"] for row in rows])),
        "best_validation_infidelity_median": float(
            np.median([row["best_onres_infidelity"] for row in rows])
        ),
        "epoch_seconds_median": float(np.median([row["epoch_time_median_s"] for row in rows])),
        "epochs": sorted({row["epochs_completed"] for row in rows}),
        "estimated_gpu_hours": float(
            sum(row["epoch_time_median_s"] * row["epochs_completed"] for row in rows) / 3600.0
        ),
        "attention_gate_median": float(np.median(gates)) if gates else None,
        "attention_gate_range": [float(min(gates)), float(max(gates))] if gates else None,
    }


def pair_file(root: Path, block: int, sigma: str) -> Path:
    return root / "pairs" / f"block{block}_{'sp' if sigma == '+' else 'sm'}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    initial = json.loads(
        (repo / "results/thf_temporal_fno_pilot/comparison/summary.json").read_text()
    )
    ablations = json.loads(
        (repo / "results/thf_temporal_fno_refinement/comparison/summary.json").read_text()
    )
    final = json.loads(
        (repo / "results/thf_temporal_baseline_audit/comparison/summary.json").read_text()
    )
    studies = {item["name"]: item for item in final["studies"]}

    baseline_root = repo / "results/thf_temporal_baseline_audit/column_v2"
    temporal_root = repo / "results/thf_temporal_fno_long_pilot/seqfno_temporal2_long_pilot"
    pair_comparison = []
    for block, sigma in PAIRS:
        baseline = json.loads(pair_file(baseline_root, block, sigma).read_text())
        temporal = json.loads(pair_file(temporal_root, block, sigma).read_text())
        pair_comparison.append(
            {
                "block": block,
                "sigma": sigma,
                "column_v2_passes_all": baseline["passes_all"],
                "temporal_passes_all": temporal["passes_all"],
                "column_v2": {key: baseline["gate_metrics"][key] for key in KEY_METRICS},
                "temporal": {key: temporal["gate_metrics"][key] for key in KEY_METRICS},
                "temporal_minus_column_v2": {
                    key: temporal["gate_metrics"][key] - baseline["gate_metrics"][key]
                    for key in KEY_METRICS
                },
            }
        )

    training = [training_summary(args.work, label, tag) for label, tag in TAGS.items()]
    result = {
        "status": "complete",
        "branch": "seqFNO_RL",
        "base_branch": "FNO_RL_optuna",
        "gpu_indices": [0, 2],
        "pilot_pairs": [f"{block},{sigma}" for block, sigma in PAIRS],
        "models_trained": sum(item["models"] for item in training),
        "estimated_training_gpu_hours": sum(item["estimated_gpu_hours"] for item in training),
        "training": training,
        "initial_comparison": initial,
        "ablation_comparison": ablations,
        "matched_budget_comparison": final,
        "pair_comparison": pair_comparison,
        "promotion": {
            "passed": final["promoted"],
            "decision": "stop_before_full_24_pair_and_rl_training",
            "reason": (
                "The matched-budget temporal model passed 4/6 pilot pairs, but block 9/+ "
                "failed the termination gate and block 11/+ failed branch-aware gates."
            ),
        },
        "rl_reference": {
            "optimized_ppo_exact_failure": 0.4178,
            "optimized_ppo_exact_average_actions": 51.37,
            "refined_sac_exact_failure": 0.5707,
            "refined_sac_exact_average_actions": 59.85,
            "physics_elimination_exact_failure": 0.2936,
            "physics_elimination_exact_average_actions": 51.49,
            "new_temporal_fno_rl_runs": 0,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    final_rows = studies
    lines = [
        "# Temporal FNO final result",
        "",
        f"Trained models: **{result['models_trained']}** over six block/polarization pairs; "
        f"estimated training compute: **{result['estimated_training_gpu_hours']:.2f} GPU-hours**.",
        "",
        "| model | pairs passing all gates | branch-mass P95, worst | conditional TV P95, worst | on-resonance derivative error | on-resonance spectral error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("downloaded_mix", "column_v2", "temporal_2layer_long"):
        row = final_rows[name]
        lines.append(
            f"| {name} | {row['pairs_passing_all']}/6 | "
            f"{row['worst_by_gate']['branch_mass_abs_error_p95']['value']:.6g} | "
            f"{row['worst_by_gate']['conditional_tv_mass_ge_1e-2_p95']['value']:.6g} | "
            f"{row['median_derivative_l1']:.6g} | {row['median_spectral_relative']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The matched-budget temporal model improves the easy pairs 0 and 1, raising "
            "complete gate passage from 2/6 to 4/6. It does not repair block 9/+ "
            "termination or the block 11/+ branch errors.",
            "",
            "Full 24-pair training and new PPO/SAC training were not run because the "
            "preregistered pilot gate failed. Existing optimized PPO and refined SAC "
            "remain the valid RL references on the downloaded `mix` FNO.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["promotion"], indent=2))


if __name__ == "__main__":
    main()
