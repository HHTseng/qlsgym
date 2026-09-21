#!/usr/bin/env python
"""Assemble the final FNO-structure and exact-RL refinement comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("results")
OUTPUT = ROOT / "thf_fno_rl_refinement"


def read(path):
    return json.loads(Path(path).read_text())


def row(name, method, exact, fno=None, seeds=5):
    return {
        "name": name,
        "method": method,
        "training_seeds": seeds,
        "exact": exact,
        "fno": fno,
    }


def main():
    locked = read(ROOT / "thf_rl_mix_tara/summary.json")
    physics = next(item for item in locked["rows"] if item["agent"] == "physics_elimination")
    optuna = read(ROOT / "thf_rl_optuna_mix/summary.json")["rows"]["ppo"]
    refined_sac = read(ROOT / "thf_rl_optuna_mix_sac_refine/summary.json")["row"]
    exact_ppo = read(ROOT / "thf_fno_rl_improvement/summary.json")
    hybrid_ppo = read(ROOT / "thf_mix_ppo_exact_finetune/summary.json")["hybrid"]
    hybrid_sac = read(ROOT / "thf_mix_sac_exact_finetune/summary.json")["hybrid"]
    audit = read(ROOT / "thf_mix_structural_audit/summary.json")
    pilot = read(ROOT / "thf_column_fno_pilot_comparison.json")

    rows = [
        row("Physics elimination", "exact model-based baseline", physics["exact"], physics["fno"], 0),
        row("Hybrid PPO", "mix FNO pretraining + 250k exact", hybrid_ppo["exact"], hybrid_ppo["fno"]),
        row("Exact-trained PPO", "1M cached-exact", exact_ppo["exact"], exact_ppo["fno"]),
        row("Hybrid SAC", "mix FNO pretraining + 250k exact", hybrid_sac["exact"], hybrid_sac["fno"]),
        row("FNO-trained PPO", "1M downloaded mix", optuna["exact"], optuna["fno"]),
        row("FNO-trained SAC", "1M downloaded mix", refined_sac["exact"], refined_sac["fno"]),
    ]
    rows.sort(key=lambda item: (item["exact"]["unfinished_fraction"], item["exact"]["average_actions"]))
    for rank, item in enumerate(rows, start=1):
        item["exact_rank"] = rank
        if item["fno"] is not None:
            item["transfer_failure_gap_pp"] = 100 * (
                item["exact"]["unfinished_fraction"] - item["fno"]["unfinished_fraction"]
            )
            item["transfer_action_gap"] = (
                item["exact"]["average_actions"] - item["fno"]["average_actions"]
            )

    result = {
        "status": "complete",
        "ranking_rule": "exact failure rate, then exact failure-penalized average actions",
        "rows": rows,
        "downloaded_mix_structural_audit": audit,
        "transfer_column_pilot": {
            "decision": pilot["decision"],
            "criteria": pilot["criteria"],
            "median_candidate_over_current": pilot["median_candidate_over_current"],
            "interpretation": (
                "The pilot fixed identity, input linearity, and off-resonance behavior, "
                "but worsened branch mass and conditional state accuracy; full training stopped."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    names = [item["name"] for item in rows]
    failure = [100 * item["exact"]["unfinished_fraction"] for item in rows]
    failure_sd = [100 * item["exact"].get("unfinished_fraction_sd", 0.0) for item in rows]
    actions = [item["exact"]["average_actions"] for item in rows]
    action_sd = [item["exact"].get("average_actions_sd", 0.0) for item in rows]
    colors = ["#9c755f" if "Physics" in name else "#54a24b" if "Hybrid" in name else "#4c78a8" for name in names]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].bar(x, failure, yerr=failure_sd, color=colors, capsize=3)
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, actions, yerr=action_sd, color=colors, capsize=3)
    axes[1].set_ylabel("Exact failure-penalized average actions")
    for axis in axes:
        axis.set_xticks(x, names, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ThF+ FNO/RL refinement: final exact-dynamics ranking")
    fig.savefig(OUTPUT / "final_exact_ranking.png", dpi=180)
    plt.close(fig)

    lines = [
        "# ThF+ FNO/RL refinement",
        "",
        "Policies are ranked by exact failure, then exact failure-penalized actions.",
        "",
        "| Rank | Controller | Training | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in rows:
        exact = item["exact"]
        fno = item["fno"]
        fno_failure = "—" if fno is None else f"{100*fno['unfinished_fraction']:.2f}%"
        fno_actions = "—" if fno is None else f"{fno['average_actions']:.2f}"
        lines.append(
            f"| {item['exact_rank']} | {item['name']} | {item['method']} | "
            f"{100*exact['unfinished_fraction']:.2f}% | {exact['average_actions']:.2f} | "
            f"{fno_failure} | {fno_actions} |"
        )
    lines += [
        "",
        f"Downloaded `mix` pairs passing every structural gate: **{audit['pairs_passing_all']}/24**.",
        f"Transfer-column pilot decision: **{pilot['decision']}**; the 24-pair retrain was not run.",
        "",
        "![Final exact ranking](final_exact_ranking.png)",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
