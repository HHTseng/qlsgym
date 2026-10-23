from __future__ import annotations

import numpy as np

from ..spec import Block, Molecule
from .hamiltonian import BlockOperator, lump, make_operator, operator

__all__ = ["propagator_series", "propagator_column_series", "transfer_matrix", "transfer_matrix_for_operator",
           "population_trajectories"]


def propagator_series(h: np.ndarray, taus: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h)
    phases = np.exp(-1j * np.outer(np.asarray(taus, dtype=np.float64), evals))
    return (evecs[None, :, :] * phases[:, None, :]) @ evecs.conj().T


def propagator_column_series(h: np.ndarray, taus: np.ndarray, n_cols: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h)
    phases = np.exp(-1j * np.outer(np.asarray(taus, dtype=np.float64), evals))
    right = evecs[:n_cols, :].conj().T
    return evecs[None, :, :] @ (phases[:, :, None] * right[None, :, :])


def transfer_matrix_for_operator(op: BlockOperator, omega: float, taus: np.ndarray) -> np.ndarray:
    m = op.n_states
    u = propagator_column_series(op.build(float(omega)), taus, m)
    return lump(np.abs(u) ** 2, m)


def transfer_matrix(
    molecule: Molecule,
    block_index,
    omega: float,
    sigma: str = "+",
    tau_indices=None,
) -> np.ndarray:
    taus = molecule.tau_grid()
    if tau_indices is not None:
        taus = taus[np.asarray(tau_indices, dtype=np.int64)]
    omega = float(omega)
    in_window = molecule.window.contains(omega)
    if isinstance(block_index, Block):
        if in_window:
            op = make_operator(molecule, block_index, sigma)
        else:
            op = make_operator(molecule, block_index, sigma, omega_lo=omega, omega_hi=omega)
    else:
        op = operator(molecule, int(block_index), sigma, None if in_window else omega)
    return transfer_matrix_for_operator(op, omega, taus)


def population_trajectories(t_mat: np.ndarray, p0: np.ndarray) -> np.ndarray:
    single = p0.ndim == 1
    p0 = np.atleast_2d(p0)
    out = np.einsum("tab,nb->nta", t_mat, p0)
    return out[0] if single else out
