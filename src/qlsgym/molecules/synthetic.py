"""A tiny random molecule for unit tests: 2-3 small blocks, a few couplings, one primitive."""
from __future__ import annotations
import numpy as np
from ..spec import Block, Molecule, Primitive, System, Task, Trap, Window, TWO_PI


def build(seed: int = 0, block_sizes=(4, 3, 3), n_nu: int = 2, n_tau: int = 16) -> Molecule:
    rng = np.random.default_rng(seed)
    nu_f = TWO_PI * 1000.0                      # rad/ms
    n = int(sum(block_sizes))
    energies = np.cumsum(rng.uniform(TWO_PI * 5, TWO_PI * 40, n))   # spread ladder, rad/ms
    levels, blocks, i_idx, f_idx, om, bos = [], [], [], [], [], np.full(n, -1)
    off = 0
    for bi, M in enumerate(block_sizes):
        states = np.arange(off, off + M)
        il, fl = np.arange(M - 1), np.arange(1, M)              # nearest-neighbour chain
        w = rng.uniform(TWO_PI * 1.0, TWO_PI * 3.0, M - 1) * np.exp(1j * rng.uniform(0, TWO_PI, M - 1))
        blocks.append(Block(bi, states, (bi, "+"), il, fl, w))
        levels += [(bi, k) for k in range(M)]
        i_idx += list(states[il]); f_idx += list(states[fl]); om += list(w)
        bos[states] = bi
        off += M
    system = System(tuple(levels), energies, np.array(i_idx), np.array(f_idx),
                    np.array(om, dtype=complex), tuple(blocks), bos)
    window = Window(omega_min=nu_f - TWO_PI * 100, omega_max=nu_f + TWO_PI * 100,
                    tau_max_ms=2.0, n_tau=n_tau, rwa_cutoff=1e4, omega_min_coupling=1.0)
    prim = Primitive("thz-like", "+", TWO_PI * 5.0e6, 0.5, (0, 1), source=0, target=int(block_sizes[0]))
    return Molecule("synthetic", system, Trap(nu_f, 0.08, n_nu), window,
                    Task(temperature_k=1.0, p_target=0.98, max_pulses=20), (prim,),
                    {"note": "random test molecule"})
