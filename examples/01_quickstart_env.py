from __future__ import annotations

import numpy as np

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, PurificationEnv, build_action_tables

MOLECULE = "synthetic"


def main() -> None:
    mol = load_molecule(MOLECULE)
    print(f"molecule {mol.name!r}: {mol.n_states} states, {mol.system.n_blocks} blocks, "
          f"{len(mol.primitives)} primitives, p_target={mol.task.p_target}, "
          f"max_pulses={mol.task.max_pulses}")

    lib = ActionLibrary.physics_subset(mol, clip_tau=(mol.name == "synthetic"))
    print(f"action library: {lib.n_grid} grid actions + {lib.n_primitives} primitives "
          f"= {lib.n_actions} total (tag {lib.tag()})")

    tables = build_action_tables(mol, lib)
    print(f"tables: {tables.nbytes() / 1e6:.2f} MB, builder={tables.manifest.get('builder')}")

    env = PurificationEnv(mol, lib, tables, batch=1)
    print(env)

    belief = env.reset(seed=0)
    print(f"t=0   max(p) = {float(belief.max()):.4f}")

    rng = np.random.default_rng(0)
    for t in range(8):
        a = int(rng.integers(env.n_actions))
        act = lib.decode(a)
        kind = f"primitive[{act.primitive}]" if act.is_primitive else "grid"
        tr = env.step(np.array([a]))
        print(f"t={t + 1:2d}  action {a:4d} ({kind}, sigma{act.sigma}, "
              f"omega/2pi={act.omega / (2 * np.pi):9.3f} kHz)  -> nu={int(tr.outcome[0])}  "
              f"max(p) = {float(tr.belief[0].max()):.4f}  reward={float(tr.reward[0]):+.2f}  "
              f"done={bool(tr.done[0])}")
        if bool(tr.done[0]) or bool(tr.truncated[0]):
            break

    P0, P1 = env.apply(belief, np.array([0]))
    print(f"\napply() (tree-search hook): action 0 from the initial belief splits into "
          f"Pi(nu=0)={float(P0.sum()):.4f}, Pi(nu>=1)={float(P1.sum()):.4f}")


if __name__ == "__main__":
    main()
