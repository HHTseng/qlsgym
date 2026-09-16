"""Sideband resonances, embedding transitions and carrier bands."""

from __future__ import annotations

import numpy as np

from ..spec import Block, Molecule, TWO_PI


def _block(molecule: Molecule, block) -> Block:
    return molecule.system.blocks[int(block)] if not isinstance(block, Block) else block


def resonant_frequencies(molecule: Molecule, block, sigma: str = "+") -> np.ndarray:
    """Sideband resonance omega^(+/-) of each coupling in the block (Eq. 3)."""
    b = _block(molecule, block)
    e = molecule.system.energies
    d_omega = e[b.states[b.f_local]] - e[b.states[b.i_local]]
    if sigma == "+":
        return molecule.trap.nu_f + d_omega
    if sigma == "-":
        return molecule.trap.nu_f - d_omega
    raise ValueError(f"sigma must be '+' or '-', got {sigma!r}")


def embedding_transitions(
    molecule: Molecule,
    block_index,
    sigma: str = "+",
    delta_max: float | None = None,
    omega_min_coupling: float | None = None,
    omega_lo: float | None = None,
    omega_hi: float | None = None,
) -> np.ndarray:
    """Indices (into the block's coupling list) retained by the embedding."""
    w = molecule.window
    delta_max = float(molecule.provenance.get("delta_max", 1.0e4)) if delta_max is None else delta_max
    omega_min_coupling = w.omega_min_coupling if omega_min_coupling is None else omega_min_coupling
    lo = w.omega_min if omega_lo is None else omega_lo
    hi = w.omega_max if omega_hi is None else omega_hi
    b = _block(molecule, block_index)
    w_res = resonant_frequencies(molecule, b, sigma)
    dist = np.maximum(0.0, np.maximum(lo - w_res, w_res - hi))
    near = dist <= delta_max
    strong = np.abs(b.omega) >= omega_min_coupling
    return np.where(near & strong)[0]


def resonance_geometry(molecule: Molecule, block_index, sigma: str = "+"):
    """(omega_res, hwhm) of the retained (embedding) transitions of one block; the half-linewidth
    is eta |Omega_q| in rad/ms.
    """
    b = _block(molecule, block_index)
    idx = embedding_transitions(molecule, b, sigma)
    return resonant_frequencies(molecule, b, sigma)[idx], molecule.trap.eta * np.abs(b.omega[idx])


def resonance_distance(molecule: Molecule, omegas, block_index, sigma: str = "+") -> np.ndarray:
    """min_q |omega_res,q - omega| / (eta |Omega_q|) -- distance in HWHM."""
    w_res, lw = resonance_geometry(molecule, block_index, sigma)
    om = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
    if w_res.size == 0:
        return np.full(om.shape, np.inf)
    return (np.abs(om[:, None] - w_res[None, :]) / lw[None, :]).min(axis=1)


def is_on_resonance(molecule: Molecule, omegas, block_index, sigma: str = "+",
                    n_linewidths: float = 1.0) -> np.ndarray:
    """Boolean mask: within n_linewidths HWHM of a retained transition."""
    return resonance_distance(molecule, omegas, block_index, sigma) <= n_linewidths


def carrier_bands(
    molecule: Molecule,
    sigma: str = "+",
    half_width_khz: float = 1.0,
    min_coupling: float = 0.02 * TWO_PI,
    omega_lo: float | None = None,
    omega_hi: float | None = None,
) -> np.ndarray:
    """Drive bands where a *carrier* is resonant -- (n, 2) of (lo, hi) rad/ms."""
    s = molecule.system
    w = molecule.window
    lo = w.omega_min if omega_lo is None else omega_lo
    hi = w.omega_max if omega_hi is None else omega_hi
    d = s.energies[s.f_idx] - s.energies[s.i_idx]
    wc = d if sigma == "+" else -d
    keep = (np.abs(s.omega_c) >= min_coupling) & (wc > lo) & (wc < hi)
    c = np.unique(np.round(wc[keep], 6))
    if c.size == 0:
        return np.zeros((0, 2))
    hw = half_width_khz * TWO_PI
    return np.stack([c - hw, c + hw], axis=1)


def mask_carriers(omegas, bands) -> np.ndarray:
    """Boolean mask, True where omegas is clear of every (lo, hi) band."""
    om = np.asarray(omegas, dtype=np.float64)
    ok = np.ones(om.shape, dtype=bool)
    for lo, hi in bands:
        ok &= ~((om >= lo) & (om <= hi))
    return ok
