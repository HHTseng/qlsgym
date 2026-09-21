#!/usr/bin/env python
"""Evaluate stronger finite-budget non-ML ThF+ reference controllers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from qlsgym.policies import CoverageSweepingPolicy, DescendingPopulationPolicy
from thf_rl_agents import build_environments, rollout_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/thf_nonml_controls")
    parser.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=82_101)
    args = parser.parse_args()
    env_args = argparse.Namespace(
        device=args.device,
        p_target=0.98,
        max_pulses=80,
        rho=0.0,
        progress=False,
        fno_tag="mix",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )
    molecule, library, fno, exact, _, contract = build_environments(env_args, 128)
    output = Path(args.output)
    policies = {
        "coverage_sweeping": (
            CoverageSweepingPolicy(library.n_actions, molecule.task.max_pulses),
            "80 evenly spaced controls spanning the complete 312-action library",
        ),
        "descending_population": (
            DescendingPopulationPolicy(library, exact.tables),
            "address the most populated state with its maximum-yield exact-table action",
        ),
    }
    results = {}
    for name, (policy, description) in policies.items():
        record = {
            "status": "complete",
            "policy": name.replace("_", " "),
            "description": description,
            "contract": contract,
            "exact": rollout_metrics(exact, policy, args.episodes, args.seed, args.eval_batch),
            "fno": rollout_metrics(fno, policy, args.episodes, args.seed, args.eval_batch),
        }
        write_json(output / f"{name}.json", record)
        results[name] = record

    compact = {
        "status": "complete",
        "evaluation_episodes": args.episodes,
        "evaluation_seed": args.seed,
        "rows": {
            name: {
                dynamics: {
                    key: record[dynamics][key]
                    for key in ("unfinished_fraction", "average_actions")
                }
                for dynamics in ("exact", "fno")
            }
            for name, record in results.items()
        },
    }
    write_json(output / "summary.json", compact)
    lines = [
        "# Stronger non-ML controls",
        "",
        "Coverage sweeping spans the full action library. Descending population is an adaptive exact-table heuristic.",
        "",
        "| Controller | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in results.items():
        label = name.replace("_", " ").title()
        lines.append(
            f"| {label} | {100*record['exact']['unfinished_fraction']:.2f}% | "
            f"{record['exact']['average_actions']:.2f} | "
            f"{100*record['fno']['unfinished_fraction']:.2f}% | "
            f"{record['fno']['average_actions']:.2f} |"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
