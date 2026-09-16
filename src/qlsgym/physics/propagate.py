"""Exact reference propagation of the single-pulse population dynamics."""

from __future__ import annotations

import numpy as np

from ..spec import Block, Molecule
from .hamiltonian import BlockOperator, lump, make_operator, operator

__all__ = ["propagator_series", "transfer_matrix", "transfer_matrix_for_operator",
           "population_trajectories"]


def propagator_series(h: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """U(tau) for every tau in taus; shape (n_tau, dim, dim)."""
    evals, evecs = np.linalg.eigh(h)
    phases = np.exp(-1j * np.outer(np.asarray(taus, dtype=np.float64), evals))   # (n_tau, dim)
    return (evecs[None, :, :] * phases[:, None, :]) @ evecs.conj().T


def transfer_matrix_for_operator(op: BlockOperator, omega: float, taus: np.ndarray) -> np.ndarray:
    """(len(taus), 2 M, M) lumped transfer columns of one operator at omega: propagates
    nu = 0 .. n_nu - 1 exactly, keeps the nu = 0 columns, lumps rows to nu = 0 / nu >= 1.
    """
    m = op.n_states
    u = propagator_series(op.build(float(omega)), taus)
    return lump(np.abs(u[:, :, :m]) ** 2, m)


def transfer_matrix(
    molecule: Molecule,
    block_index,
    omega: float,
    sigma: str = "+",
    tau_indices=None,
) -> np.ndarray:
    """Lumped population transfer columns T(tau): (n_tau, 2 M, M) float64, n_tau = len(tau_indices)
    (default the whole molecule.tau_grid).
    """
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
    """p(tau) = T(tau) p0 for lumped columns t_mat (n_tau, 2M, M) and p0 (M,) or (n_batch, M);
    returns (n_tau, 2M) or (n_batch, n_tau, 2M).
    """
    single = p0.ndim == 1
    p0 = np.atleast_2d(p0)
    out = np.einsum("tab,nb->nta", t_mat, p0)
    return out[0] if single else out
