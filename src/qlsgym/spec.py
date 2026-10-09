from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

TWO_PI = 2.0 * np.pi
SIGMAS: tuple[str, str] = ("+", "-")


@dataclass(frozen=True)
class Block:
    index: int
    states: np.ndarray
    key: tuple
    i_local: np.ndarray
    f_local: np.ndarray
    omega: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.states.size)

    def dim(self, n_nu: int) -> int:
        return n_nu * self.n_states


@dataclass(frozen=True)
class System:
    levels: tuple
    energies: np.ndarray
    i_idx: np.ndarray
    f_idx: np.ndarray
    omega_c: np.ndarray
    blocks: tuple
    block_of_state: np.ndarray

    @property
    def n_states(self) -> int:
        return len(self.levels)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class Trap:
    nu_f: float
    eta: float
    n_nu: int


@dataclass(frozen=True)
class Window:
    omega_min: float
    omega_max: float
    tau_max_ms: float
    n_tau: int
    rwa_cutoff: float
    omega_min_coupling: float
    carrier_bands: tuple = ()
    s_emb: float = 0.05
    beta_emb: float = 0.01

    def tau_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.tau_max_ms, self.n_tau)

    def contains(self, omega: float) -> bool:
        return bool(self.omega_min <= omega <= self.omega_max)


@dataclass(frozen=True)
class Primitive:
    label: str
    sigma: str
    omega: float
    tau_ms: float
    blocks: tuple
    source: int = -1
    target: int = -1


@dataclass(frozen=True)
class Task:
    temperature_k: float
    p_target: float = 0.98
    max_pulses: int = 80


@dataclass(frozen=True)
class Molecule:
    name: str
    system: System
    trap: Trap
    window: Window
    task: Task
    primitives: tuple = ()
    provenance: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.system.n_states

    @property
    def blocks(self) -> tuple:
        return self.system.blocks

    def tau_grid(self) -> np.ndarray:
        return self.window.tau_grid()

    def sideband_mirror(self, omega: float) -> float:
        return 2.0 * self.trap.nu_f - omega

    def fingerprint(self) -> str:
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


@runtime_checkable
class TauBatchedEngine(Protocol):
    molecule: Molecule
    tau_indices: np.ndarray

    def branches_all_tau(
        self, p_in: np.ndarray, omega: float, sigma: str
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class BatchedEngine(TauBatchedEngine, Protocol):
    def branches_batch(
        self, p_in: np.ndarray, omegas: np.ndarray, sigma: str
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


def branches_batch_fallback(
    engine: TauBatchedEngine, p_in: np.ndarray, omegas: np.ndarray, sigma: str
) -> tuple[np.ndarray, np.ndarray]:
    fn = getattr(engine, "branches_batch", None)
    if fn is not None:
        return fn(p_in, omegas, sigma)
    outs = [engine.branches_all_tau(p_in, float(w), sigma) for w in np.asarray(omegas, dtype=np.float64)]
    return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])


@dataclass(frozen=True)
class Action:
    sigma: str
    omega: float
    tau_index: int = -1
    primitive: int = -1

    @property
    def is_primitive(self) -> bool:
        return self.primitive >= 0


def check_branches(p0: np.ndarray, p1: np.ndarray, n_states: int, atol: float = 1e-6) -> None:
    assert p0.shape == p1.shape and p0.shape[-1] == n_states, (p0.shape, p1.shape)
    assert np.all(p0 >= -atol) and np.all(p1 >= -atol)
    tot = p0.sum(-1) + p1.sum(-1)
    assert np.allclose(tot, 1.0, atol=atol), (tot.min(), tot.max())
