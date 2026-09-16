"""CUDA-Q Dynamics engine: the lumped transfer columns of ExactEngine, from an ODE solve."""

from __future__ import annotations

import importlib.util
import itertools
from dataclasses import dataclass

import numpy as np

from ..spec import Molecule
from .engines import _EngineBase
from .hamiltonian import lump, make_operator, operator

__all__ = ["CudaqEngine", "dynamics_available", "DEFAULT_MAX_STEP_SIZE", "MAX_PHASE_PER_STEP"]

DEFAULT_MAX_STEP_SIZE = 3.0e-5
MAX_PHASE_PER_STEP = 0.05
UNITARITY_ATOL = 1e-6

_UID = itertools.count()
_STATUS: tuple[bool, str] | None = None


def dynamics_available(recheck: bool = False) -> tuple[bool, str]:
    """(can this process run the engine, why)."""
    global _STATUS
    if _STATUS is not None and not recheck:
        return _STATUS
    _STATUS = _probe()
    return _STATUS


def _probe() -> tuple[bool, str]:
    # importing cudaq costs ~20 s, so the cheap disqualifiers come first
    if importlib.util.find_spec("cudaq") is None:
        return False, "cudaq is not installed"
    import torch

    if torch.cuda.device_count() < 1:
        return False, "no CUDA device visible"
    try:
        import cudaq
    except Exception as exc:
        return False, f"cudaq not importable: {type(exc).__name__}: {exc}"
    n_gpu = int(cudaq.num_available_gpus())
    if n_gpu < 1:
        return False, "cudaq sees no GPU"
    try:
        # set_target is the only honest probe: it is what brings up cuDensityMat.
        cudaq.set_target("dynamics")
    except Exception as exc:
        return False, f"cudaq target 'dynamics' unavailable: {exc}"
    return True, f"{cudaq.__version__}, dynamics target, {n_gpu} GPU(s)"


# H(omega) for one sector


@dataclass(frozen=True)
class _Drive:
    """H(omega) = h_ref + (omega - w_ref) grad, exactly (BlockOperator.build is affine in omega)."""

    n_states: int
    w_ref: float
    h_ref: np.ndarray
    grad: np.ndarray
    h_id: str
    grad_id: str

    @property
    def dim(self) -> int:
        return int(self.h_ref.shape[0])

    def dense(self, omega: float) -> np.ndarray:
        return self.h_ref + (float(omega) - self.w_ref) * self.grad

    def symbolic(self, omega: float):
        from cudaq import operators

        return (operators.instantiate(self.h_id, 0)
                + complex(float(omega) - self.w_ref) * operators.instantiate(self.grad_id, 0))


def _block_operator(molecule: Molecule, sector, sigma: str, omega: float):
    """The BlockOperator physics.propagate.transfer_matrix would use for this drive."""
    w = None if molecule.window.contains(float(omega)) else float(omega)
    if sector.index < 0:
        return make_operator(molecule, sector, sigma, omega_lo=w, omega_hi=w)
    return operator(molecule, int(sector.index), sigma, w)


def make_drive(molecule: Molecule, sector, sigma: str, omega: float) -> _Drive:
    op = _block_operator(molecule, sector, sigma, omega)
    w = molecule.window
    in_window = w.contains(float(omega))
    # reference the split at the drive itself when off-window, so that reconstructing
    # H(omega) never differences two numbers larger than the window is wide
    w_ref = 0.5 * (w.omega_min + w.omega_max) if in_window else float(omega)
    step = max(1.0, w.omega_max - w.omega_min)
    h_ref = op.build(w_ref)
    grad = (op.build(w_ref + step) - h_ref) / step
    uid = next(_UID)
    return _Drive(n_states=op.n_states, w_ref=float(w_ref),
                  h_ref=np.ascontiguousarray(h_ref, dtype=np.complex128),
                  grad=np.ascontiguousarray(grad, dtype=np.complex128),
                  h_id=f"qlsgym_h_{uid}", grad_id=f"qlsgym_dh_{uid}")


# schedule, step size, columns


def schedule_and_rows(taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integration schedule (sorted, unique, starting at tau = 0) and each tau's row in it."""
    taus = np.asarray(taus, dtype=np.float64)
    sched = np.unique(np.concatenate([[0.0], taus]))
    return sched, np.searchsorted(sched, taus)


def step_size(h: np.ndarray, max_step_size: float, max_phase: float) -> float:
    norm = float(np.abs(h).sum(1).max())        # bounds the spectral radius
    return min(float(max_step_size), float(max_phase) / norm) if norm > 0.0 else float(max_step_size)


def rk_steps(sched: np.ndarray, dt: float) -> int:
    return int(np.ceil(np.diff(sched) / dt).sum())


def columns_from_amplitudes(psi: np.ndarray, n_states: int, atol: float = UNITARITY_ATOL) -> np.ndarray:
    """(n_basis, n_sched, dim) amplitudes -> (n_sched, 2 M, M) lumped transfer columns."""
    t = lump(np.abs(np.moveaxis(psi, 0, -1)) ** 2, n_states)
    err = float(np.abs(t.sum(-2) - 1.0).max())
    if not err <= atol:                         # not "err > atol": a diverged run gives NaN
        raise RuntimeError(f"CUDA-Q integration lost {err:.3g} of the norm (tolerance {atol:.3g}): "
                           "the RK4 step is too large for this Hamiltonian")
    return t


class CudaqEngine(_EngineBase):
    """ExactEngine's contract, with each drive's propagator integrated by cudaq.evolve.

    One evolve call per (sector, sigma, omega) covers every tau: the schedule is the tau grid
    and the intermediate states are kept.
    """

    def __init__(self, molecule: Molecule, tau_indices=None,
                 max_step_size: float = DEFAULT_MAX_STEP_SIZE,
                 max_phase_per_step: float = MAX_PHASE_PER_STEP,
                 unitarity_atol: float = UNITARITY_ATOL):
        super().__init__(molecule, tau_indices)
        ok, reason = dynamics_available()
        if not ok:
            raise RuntimeError(f"CudaqEngine cannot run here: {reason}")
        self.max_step_size = float(max_step_size)
        self.max_phase_per_step = float(max_phase_per_step)
        self.unitarity_atol = float(unitarity_atol)
        self.settings = {"target": "dynamics", "integrator": "RungeKutta(order=4)",
                         "max_step_size": self.max_step_size,
                         "max_phase_per_step": self.max_phase_per_step,
                         "unitarity_atol": self.unitarity_atol, "probe": reason}
        self.stats = {"solves": 0, "rk_steps": 0}
        self._drives: dict = {}
        self._defined: set = set()

    def branches_all_tau(self, p_in, omega, sigma):
        return self.exact_branches_all_tau(p_in, omega, sigma)

    def drive(self, sector, sigma: str, omega: float) -> _Drive:
        key = (int(sector.index), tuple(sector.key), sigma)
        d = self._drives.get(key)
        if d is None:
            d = make_drive(self.molecule, sector, sigma, float(omega))
            self._drives[key] = d
        return d

    def _compute_columns(self, sector, sigma: str, omega: float) -> np.ndarray:
        drive = self.drive(sector, sigma, omega)
        dt = step_size(drive.dense(omega), self.max_step_size, self.max_phase_per_step)
        sched, rows = schedule_and_rows(self.taus)
        psi = self._evolve(drive, float(omega), sched, dt)
        self.stats["solves"] += 1
        self.stats["rk_steps"] += rk_steps(sched, dt)
        return columns_from_amplitudes(psi, drive.n_states, self.unitarity_atol)[rows]

    def _evolve(self, drive: _Drive, omega: float, sched: np.ndarray, dt: float) -> np.ndarray:
        """(M, len(sched), dim): the nu = 0 basis states propagated to every schedule point."""
        import cudaq
        from cudaq.dynamics.helpers import IntermediateResultSave

        self._define(drive)
        eye = np.eye(drive.dim, dtype=np.complex128)
        res = cudaq.evolve(
            drive.symbolic(omega),
            {0: drive.dim},
            cudaq.Schedule([float(t) for t in sched], ["t"]),
            [cudaq.State.from_data(eye[:, j].copy()) for j in range(drive.n_states)],
            store_intermediate_results=IntermediateResultSave.ALL,
            integrator=cudaq.RungeKuttaIntegrator(order=4, max_step_size=float(dt)),
        )
        out = res if isinstance(res, (list, tuple)) else [res]
        return np.stack([np.stack([np.array(s) for s in r.intermediate_states()]) for r in out])

    def _define(self, drive: _Drive) -> None:
        from cudaq import operators

        if drive.h_id in self._defined:
            return
        operators.define(drive.h_id, [drive.dim], lambda: drive.h_ref, override=True)
        operators.define(drive.grad_id, [drive.dim], lambda: drive.grad, override=True)
        self._defined.add(drive.h_id)
