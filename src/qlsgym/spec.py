"""qlsgym.spec -- the contract every other module codes against."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

TWO_PI = 2.0 * np.pi
SIGMAS: tuple[str, str] = ("+", "-")


# Molecular structure


@dataclass(frozen=True)
class Block:
    """One closed subspace of the in-window sigma+ Raman coupling graph."""

    index: int
    states: np.ndarray        # (M,) global molecular indices, sorted
    key: tuple
    i_local: np.ndarray       # (n_tr,) sigma+ source, *local* index into states
    f_local: np.ndarray       # (n_tr,) sigma+ target, local
    omega: np.ndarray         # (n_tr,) complex two-photon Rabi rate, rad/ms

    @property
    def n_states(self) -> int:
        return int(self.states.size)

    def dim(self, n_nu: int) -> int:
        """Hilbert-space dimension when n_nu motional levels are kept."""
        return n_nu * self.n_states


@dataclass(frozen=True)
class System:
    """Levels, energies and the sigma+ coupling list; molecule-agnostic."""

    levels: tuple                 # per-state label tuples (for humans and tests)
    energies: np.ndarray          # (n,) rad/ms
    i_idx: np.ndarray             # (n_tr,) sigma+ source, global index (all blocks, incl. cross-block)
    f_idx: np.ndarray             # (n_tr,) sigma+ target, global
    omega_c: np.ndarray           # (n_tr,) complex Rabi, rad/ms
    blocks: tuple                 # tuple[Block, ...]
    block_of_state: np.ndarray    # (n,) block index per state (-1 if in no block)

    @property
    def n_states(self) -> int:
        return len(self.levels)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


# Per-molecule control settings


@dataclass(frozen=True)
class Trap:
    nu_f: float          # rad/ms, secular frequency of the shared mode addressed by the sidebands
    eta: float           # Lamb-Dicke parameter of the molecule for that mode
    n_nu: int            # motional levels propagated (2 = paper's nu<=1; ThF+ needs 7)


@dataclass(frozen=True)
class Window:
    """The in-window Raman drive: what the FNO is trained on and what the grid action set spans."""

    omega_min: float                          # rad/ms
    omega_max: float                          # rad/ms
    tau_max_ms: float                         # pulse durations are the grid tau_grid()
    n_tau: int
    rwa_cutoff: float                         # rad/ms; couplings detuned by more than this anywhere in the window are dropped
    omega_min_coupling: float                 # rad/ms; weaker sigma+ couplings are dropped from the block graph
    carrier_bands: tuple = ()                 # ((lo, hi), ...) rad/ms, masked out of the grid (ThF+ F5)
    # physics-informed embedding constants (paper Eqs. 11-16); surrogate only
    s_emb: float = 0.05
    beta_emb: float = 0.01

    def tau_grid(self) -> np.ndarray:
        """n_tau pulse durations over [0, tau_max] **inclusive**, i.e. linspace(0, tau_max, n_tau)."""
        return np.linspace(0.0, self.tau_max_ms, self.n_tau)

    def contains(self, omega: float) -> bool:
        return bool(self.omega_min <= omega <= self.omega_max)


@dataclass(frozen=True)
class Primitive:
    """A fixed off-window pulse (THz rotational drive for H3O+, cross-J Raman sideband for ThF+)."""

    label: str
    sigma: str
    omega: float                  # rad/ms, outside the window
    tau_ms: float                 # the pulse's own duration (its pi-time)
    blocks: tuple                 # block indices coupled by this drive
    source: int = -1              # global state it was designed to empty (-1 = n/a)
    target: int = -1


@dataclass(frozen=True)
class Task:
    temperature_k: float
    p_target: float = 0.98
    max_pulses: int = 80


@dataclass(frozen=True)
class Molecule:
    """Everything a gym, planner or surrogate needs to know about one system."""

    name: str
    system: System
    trap: Trap
    window: Window
    task: Task
    primitives: tuple = ()
    provenance: dict = field(default_factory=dict)   # table files, B field, J_max, coupling model, git rev ...

    # convenience
    @property
    def n_states(self) -> int:
        return self.system.n_states

    @property
    def blocks(self) -> tuple:
        return self.system.blocks

    def tau_grid(self) -> np.ndarray:
        return self.window.tau_grid()

    def sideband_mirror(self, omega: float) -> float:
        """The sigma- drive that addresses the same molecular transition as a sigma+ drive at
        omega: 2 nu_f - omega.
        """
        return 2.0 * self.trap.nu_f - omega

    def fingerprint(self) -> str:
        """Short stable hash of the physics: tables, trap, window, n_nu."""
        h = hashlib.sha256()
        s = self.system
        for a in (s.energies, s.i_idx, s.f_idx, s.omega_c):
            h.update(np.ascontiguousarray(a).tobytes())
        for b in s.blocks:
            h.update(np.ascontiguousarray(b.states).tobytes())
        meta = {
            "name": self.name,
            "trap": [self.trap.nu_f, self.trap.eta, self.trap.n_nu],
            "window": [self.window.omega_min, self.window.omega_max,
                       self.window.tau_max_ms, self.window.n_tau,
                       self.window.rwa_cutoff, self.window.omega_min_coupling,
                       list(map(list, self.window.carrier_bands))],
            "primitives": [(p.sigma, p.omega, p.tau_ms, list(p.blocks)) for p in self.primitives],
        }
        h.update(json.dumps(meta, sort_keys=True).encode())
        return h.hexdigest()[:12]


# Dynamics


@runtime_checkable
class TauBatchedEngine(Protocol):
    """Anything that can answer: *if I apply this pulse to this belief, what are the two
    measurement branches, for every duration on my tau grid?*.
    """

    molecule: Molecule
    # indices into molecule.tau_grid() that this engine evaluates
    tau_indices: np.ndarray

    def branches_all_tau(
        self, p_in: np.ndarray, omega: float, sigma: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """(P0, P1), each (len(tau_indices), n_states) float64, unnormalised (their sums are the
        two outcome probabilities).
        """
        ...


class BatchedEngine(TauBatchedEngine, Protocol):
    """Optional fast path: many frequencies at once (surrogate / GPU)."""

    def branches_batch(
        self, p_in: np.ndarray, omegas: np.ndarray, sigma: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """(P0, P1), each (n_omega, len(tau_indices), n_states)."""
        ...


def branches_batch_fallback(
    engine: TauBatchedEngine, p_in: np.ndarray, omegas: np.ndarray, sigma: str
) -> tuple[np.ndarray, np.ndarray]:
    """Loop implementation of BatchedEngine.branches_batch for engines that do not provide one."""
    fn = getattr(engine, "branches_batch", None)
    if fn is not None:
        return fn(p_in, omegas, sigma)
    outs = [engine.branches_all_tau(p_in, float(w), sigma) for w in np.asarray(omegas, dtype=np.float64)]
    return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])


# Actions (the discrete interface between physics and learning)


@dataclass(frozen=True)
class Action:
    """One pulse."""

    sigma: str
    omega: float
    tau_index: int = -1
    primitive: int = -1          # index into molecule.primitives, or -1

    @property
    def is_primitive(self) -> bool:
        return self.primitive >= 0


def check_branches(p0: np.ndarray, p1: np.ndarray, n_states: int, atol: float = 1e-6) -> None:
    """Sanity checks shared by tests and engines: shapes, non-negativity, probability conservation."""
    assert p0.shape == p1.shape and p0.shape[-1] == n_states, (p0.shape, p1.shape)
    assert np.all(p0 >= -atol) and np.all(p1 >= -atol)
    tot = p0.sum(-1) + p1.sum(-1)
    assert np.allclose(tot, 1.0, atol=atol), (tot.min(), tot.max())
