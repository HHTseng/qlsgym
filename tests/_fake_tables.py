"""Fake dynamics for env/policy unit tests -- no physics package required."""
from __future__ import annotations

import hashlib

import numpy as np

from qlsgym.env.actions import ActionLibrary
from qlsgym.env.cache import ActionTables, PrimitiveTable, primitive_sector, tables_from_engine
from qlsgym.spec import Molecule


def _seed_of(*parts) -> int:
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def column_stochastic(rng, rows: int, cols: int) -> np.ndarray:
    a = rng.random((rows, cols))
    return a / a.sum(0, keepdims=True)


class FakeEngine:
    """Deterministic pseudo-dynamics satisfying TauBatchedEngine."""

    def __init__(self, molecule: Molecule, tau_indices=None, seed: int = 0, leak: float = 0.03):
        self.molecule = molecule
        self.tau_indices = np.arange(molecule.window.n_tau) if tau_indices is None else np.asarray(tau_indices)
        self.seed, self.leak = int(seed), float(leak)
        self.calls = 0
        self._cache: dict = {}

    def _table(self, sigma: str, omega: float, key, S: int) -> np.ndarray:
        k = (sigma, round(float(omega), 6), key)
        t = self._cache.get(k)
        if t is None:
            rng = np.random.default_rng(_seed_of(self.seed, *k))
            taus = self.molecule.tau_grid()[self.tau_indices]
            j = int(rng.integers(S))
            jt = (j + 1) % S if S > 1 else j
            rabi = rng.uniform(1.0, 4.0)
            go = np.sin(0.5 * rabi * taus) ** 2                         # (n_tau,)
            t = np.zeros((taus.size, 2 * S, S))
            t[:, :S, :] = np.eye(S)[None]
            t[:, j, j] = 1.0 - go
            t[:, S + jt, j] = go
            # a small tau-independent leak from every column into nu >= 1
            L = column_stochastic(rng, S, S) * self.leak                # (S, S)
            t[:, :S, :] *= (1.0 - self.leak)
            t[:, S:, :] = t[:, S:, :] * (1.0 - self.leak) + L[None]
            self._cache[k] = t
        return t

    def _sectors(self, sigma: str, omega: float):
        mol = self.molecule
        if mol.window.contains(omega):
            return [(b.states, ("block", b.index)) for b in mol.blocks]
        for k, p in enumerate(mol.primitives):
            if p.sigma == sigma and abs(p.omega - omega) < 1e-6:
                return [(primitive_sector(mol, k), ("prim", k))]
        raise ValueError(f"omega={omega} sigma={sigma} is neither in-window nor a primitive")

    def branches_all_tau(self, p_in, omega, sigma):
        self.calls += 1
        p_in = np.asarray(p_in, dtype=np.float64)
        nt = self.tau_indices.size
        out0 = np.tile(p_in, (nt, 1))
        out1 = np.zeros((nt, p_in.size))
        for states, key in self._sectors(sigma, float(omega)):
            S = states.size
            t = self._table(sigma, omega, key, S)
            res = t @ p_in[states]
            out0[:, states] = res[:, :S]
            out1[:, states] = res[:, S:]
        return out0, out1


def fake_tables(molecule: Molecule, library: ActionLibrary, engine: FakeEngine | None = None,
                dtype=np.float64) -> ActionTables:
    engine = engine or FakeEngine(molecule)
    return tables_from_engine(molecule, library, engine, dtype=dtype)


def random_tables(molecule: Molecule, library: ActionLibrary, seed: int = 0, dtype=np.float32) -> ActionTables:
    """Unrelated random column-stochastic tables of the right shapes."""
    rng = np.random.default_rng(seed)
    blocks = [np.stack([column_stochastic(rng, 2 * b.n_states, b.n_states) for _ in range(library.n_grid)]).astype(dtype)
              for b in molecule.blocks]
    prims = []
    for k in range(library.n_primitives):
        st = primitive_sector(molecule, k)
        prims.append(PrimitiveTable(st, column_stochastic(rng, 2 * st.size, st.size).astype(dtype),
                                    library.primitive_tau_index(k), tuple(molecule.primitives[k].blocks)))
    return ActionTables(molecule.fingerprint(), library.tag(), blocks, prims, {"builder": "random"})
