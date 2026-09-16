"""Swap the exact engine for the FNO surrogate and compare a rollout."""
from __future__ import annotations

import argparse
import time

import numpy as np

from qlsgym import load_molecule
from qlsgym.env.actions import ActionLibrary
from qlsgym.physics.engines import ExactEngine
from qlsgym.policies.baselines import ScorePlannerPolicy
from qlsgym.policies.rollout import rollout
from qlsgym.surrogate.manifest import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--molecule", default="thf", choices=["thf", "h3o"])
    ap.add_argument("--tag", default="mix", help="checkpoint manifest tag "
                    "(scripts/make_manifest.py); e.g. 'mix' or 'a1.0' for thf")
    ap.add_argument("--n-rollouts", type=int, default=200)
    ap.add_argument("--max-pulses", type=int, default=None, help="default: molecule.task.max_pulses")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="torch device for the FNO forward pass")
    args = ap.parse_args()

    mol = load_molecule(args.molecule)
    library = ActionLibrary.physics_subset(mol)

    print(f"{mol.name}: {mol.n_states} states, {len(mol.blocks)} blocks, "
          f"tag={args.tag!r}, n_rollouts={args.n_rollouts}")

    print("\n-- exact engine --")
    t0 = time.time()
    exact = ExactEngine(mol)
    r_exact = rollout(exact, ScorePlannerPolicy(exact, library), n_rollouts=args.n_rollouts, seed=args.seed,
                      max_pulses=args.max_pulses, library=library)
    print(f"success {r_exact.success_fraction:.3f}  mean_pulses {r_exact.mean_pulses:.1f}  "
          f"({time.time() - t0:.1f}s)")

    print("\n-- FNO surrogate (exact fallback where untrained) --")
    t0 = time.time()
    eng = load_manifest(mol, args.tag, device=args.device)
    r_fno = rollout(eng, ScorePlannerPolicy(eng, library), n_rollouts=args.n_rollouts, seed=args.seed,
                    max_pulses=args.max_pulses, library=library)
    print(f"success {r_fno.success_fraction:.3f}  mean_pulses {r_fno.mean_pulses:.1f}  "
          f"({time.time() - t0:.1f}s)")
    print(f"surrogate fraction of in-loop calls: {eng.surrogate_fraction():.3f}  calls={eng.calls}")

    print(f"\nsuccess rate: exact {r_exact.success_fraction:.3f} vs FNO {r_fno.success_fraction:.3f} "
          f"(delta {r_fno.success_fraction - r_exact.success_fraction:+.3f})")


if __name__ == "__main__":
    main()
