#!/usr/bin/env python
"""Evaluate stronger finite-budget non-ML ThF+ reference controllers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from qlsgym.policies import CoverageSweepingPolicy
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
    policy = CoverageSweepingPolicy(library.n_actions, molecule.task.max_pulses)
    result = {
        "status": "complete",
        "policy": "coverage sweeping",
        "description": "80 evenly spaced controls spanning the complete 312-action library",
        "contract": contract,
        "exact": rollout_metrics(exact, policy, args.episodes, args.seed, args.eval_batch),
        "fno": rollout_metrics(fno, policy, args.episodes, args.seed, args.eval_batch),
    }
    output = Path(args.output)
    write_json(output / "coverage_sweeping.json", result)
    lines = [
        "# Coverage-balanced sweeping",
        "",
        "This finite-budget reference visits 80 evenly spaced controls across the full 312-action library.",
        "",
        "| Dynamics | Failure | Penalized actions |",
        "|---|---:|---:|",
        f"| Exact | {100*result['exact']['unfinished_fraction']:.2f}% | {result['exact']['average_actions']:.2f} |",
        f"| Downloaded FNO | {100*result['fno']['unfinished_fraction']:.2f}% | {result['fno']['average_actions']:.2f} |",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
