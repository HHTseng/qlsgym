from __future__ import annotations

import importlib.util
import itertools
from dataclasses import dataclass

import numpy as np

from ..spec import Molecule
from .engines import _EngineBase
from .hamiltonian import lump, make_operator, operator

__all__ = ["CudaqEngine", "dynamics_available", "DEFAULT_ATOL", "DEFAULT_MAX_STEP_SIZE"]

DEFAULT_ATOL = 1.0e-6
DEFAULT_MAX_STEP_SIZE = 3.0e-5
GUARD = 2.0

_UID = itertools.count()
_STATUS: tuple[bool, str] | None = None


def dynamics_available(recheck: bool = False) -> tuple[bool, str]:
    global _STATUS
    if _STATUS is not None and not recheck:
        return _STATUS
    _STATUS = _probe()
    return _STATUS


def _probe() -> tuple[bool, str]:
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
        cudaq.set_target("dynamics")
    except Exception as exc:
        return False, f"cudaq target 'dynamics' unavailable: {exc}"
    return True, f"{cudaq.__version__}, dynamics target, {n_gpu} GPU(s)"


@dataclass(frozen=True)
class _Drive:
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
    w = None if molecule.window.contains(float(omega)) else float(omega)
    if sector.index < 0:
        return make_operator(molecule, sector, sigma, omega_lo=w, omega_hi=w)
    return operator(molecule, int(sector.index), sigma, w)


def make_drive(molecule: Molecule, sector, sigma: str, omega: float) -> _Drive:
    op = _block_operator(molecule, sector, sigma, omega)
    w = molecule.window
    in_window = w.contains(float(omega))
    w_ref = 0.5 * (w.omega_min + w.omega_max) if in_window else float(omega)
    step = max(1.0, w.omega_max - w.omega_min)
    h_ref = op.build(w_ref)
    grad = (op.build(w_ref + step) - h_ref) / step
    uid = next(_UID)
    return _Drive(n_states=op.n_states, w_ref=float(w_ref),
                  h_ref=np.ascontiguousarray(h_ref, dtype=np.complex128),
                  grad=np.ascontiguousarray(grad, dtype=np.complex128),
                  h_id=f"qlsgym_h_{uid}", grad_id=f"qlsgym_dh_{uid}")


def _basis_vector(dim: int, j: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.complex128)
    v[j] = 1.0
    return v


def schedule_and_rows(taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    taus = np.asarray(taus, dtype=np.float64)
    sched = np.unique(np.concatenate([[0.0], taus]))
    return sched, np.searchsorted(sched, taus)


def step_size(h: np.ndarray, t_max: float, atol: float, max_step_size: float) -> float:
    norm = float(np.abs(h).sum(1).max())
    if norm <= 0.0 or t_max <= 0.0:
        return float(max_step_size)
    return min(float(max_step_size), (72.0 * float(atol) / (float(t_max) * norm)) ** 0.2 / norm)


def rk_steps(sched: np.ndarray, dt: float) -> int:
    return int(np.ceil(np.diff(sched) / dt).sum())


def columns_from_amplitudes(psi: np.ndarray, n_states: int) -> np.ndarray:
    return lump(np.abs(np.moveaxis(psi, 0, -1)) ** 2, n_states)


def norm_loss(t: np.ndarray) -> float:
    return float(np.abs(t.sum(-2) - 1.0).max())


class CudaqEngine(_EngineBase):
    def __init__(self, molecule: Molecule, tau_indices=None, atol: float = DEFAULT_ATOL,
                 max_step_size: float = DEFAULT_MAX_STEP_SIZE):
        super().__init__(molecule, tau_indices)
        ok, reason = dynamics_available()
        if not ok:
            raise RuntimeError(f"CudaqEngine cannot run here: {reason}")
        self.atol = float(atol)
        self.max_step_size = float(max_step_size)
        self.settings = {"target": "dynamics", "integrator": "RungeKutta(order=4)",
                         "atol": self.atol, "max_step_size": self.max_step_size, "probe": reason}
        self.stats = {"solves": 0, "rk_steps": 0, "max_norm_loss": 0.0, "min_step_size": None}
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
        sched, rows = schedule_and_rows(self.taus)
        dt = step_size(drive.dense(omega), float(sched[-1]), self.atol, self.max_step_size)
        t = columns_from_amplitudes(self._evolve(drive, float(omega), sched, dt), drive.n_states)
        lost = norm_loss(t)
        if not lost <= GUARD * self.atol:
            raise RuntimeError(
                f"CUDA-Q lost {lost:.3g} of the norm at omega = {omega:.6g}, sigma{sigma}, "
                f"dt = {dt:.3g} ms, more than the {GUARD * self.atol:.3g} the step was chosen for")
        self.stats["solves"] += 1
        self.stats["rk_steps"] += rk_steps(sched, dt)
        self.stats["max_norm_loss"] = max(self.stats["max_norm_loss"], lost)
        self.stats["min_step_size"] = min(self.stats["min_step_size"] or dt, dt)
        return t[rows]

    def _evolve(self, drive: _Drive, omega: float, sched: np.ndarray, dt: float) -> np.ndarray:
        import cudaq
        from cudaq.dynamics.helpers import IntermediateResultSave

        self._define(drive)
        res = cudaq.evolve(
            drive.symbolic(omega),
            {0: drive.dim},
            cudaq.Schedule([float(t) for t in sched], ["t"]),
            [cudaq.State.from_data(_basis_vector(drive.dim, j)) for j in range(drive.n_states)],
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
