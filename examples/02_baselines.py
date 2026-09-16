"""python examples/02_baselines.py thf 200 150 # n_rollouts, max_pulses"""
from __future__ import annotations

import sys

from qlsgym import load_molecule
from qlsgym.env.actions import ActionLibrary
from qlsgym.policies import (PhysicsEliminationPolicy, RandomPolicy, ScorePlannerPolicy,
                             SweepingPolicy, rollout)
from qlsgym.policies.score import ScoreConfig


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    n_rollouts = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    max_pulses = int(sys.argv[3]) if len(sys.argv) > 3 else None

    from qlsgym.physics.engines import ExactEngine

    mol = load_molecule(name)
    max_pulses = mol.task.max_pulses if max_pulses is None else max_pulses
    lib = ActionLibrary.physics_subset(mol, clip_tau=(name == "synthetic"))
    engine = ExactEngine(mol)
    print(f"{mol.name}: {mol.n_states} states, library {lib.n_actions} actions "
          f"({lib.n_grid} grid + {lib.n_primitives} primitives), "
          f"n_rollouts={n_rollouts}, max_pulses={max_pulses}, p_target={mol.task.p_target}\n")

    policies = [
        ("sweeping", SweepingPolicy(lib.n_actions)),
        ("random", RandomPolicy(lib.n_actions)),
        ("score", ScorePlannerPolicy(engine, lib, ScoreConfig().for_molecule(mol))),
    ]
    if mol.name == "thf":
        policies.append(("physics", PhysicsEliminationPolicy(mol, lib)))

    print(f"{'policy':>10s}  {'success':>8s}  {'mean pulses':>12s}  {'successful':>10s}  {'engine calls':>12s}")
    for label, policy in policies:
        res = rollout(engine, policy, n_rollouts=n_rollouts, seed=0, max_pulses=max_pulses, library=lib)
        print(f"{label:>10s}  {res.success_fraction:7.3f}  {res.mean_pulses:12.2f}  "
              f"{res.mean_pulses_successful:10.2f}  {res.engine_calls:12d}  ({res.seconds:.1f}s)")
    print("\ntarget states of the last policy run (label, count):", res.target_states[:4])


if __name__ == "__main__":
    main()
