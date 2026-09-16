"""Accuracy metrics for the surrogate (Eqs. 27-30) and the trivial p(tau) = p(0) baseline."""

from __future__ import annotations

import numpy as np
import torch

from ..spec import Molecule
from .embedding import Embedding

Tensor = torch.Tensor


# Population fidelity (Eqs. 27-30)


def population_fidelity(p_pred: Tensor, p_true: Tensor, dim: int = -1) -> Tensor:
    """Eq. 27, F_p = (sum_b sqrt(p_pred_b p_true_b))^2."""
    root = torch.sqrt(p_pred.clamp_min(0) * p_true.clamp_min(0))
    return root.sum(dim=dim).square()


def population_infidelity(p_pred: Tensor, p_true: Tensor, dim: int = -1) -> Tensor:
    """Eq. 28, I_p(tau) = 1 - F_p(tau), clamped at 0."""
    return (1.0 - population_fidelity(p_pred, p_true, dim=dim)).clamp_min(0.0)


def infidelity_curve(p_pred: Tensor, p_true: Tensor) -> Tensor:
    """I_p(tau) for trajectories shaped (..., P_tau, 2 M_f)."""
    return population_infidelity(p_pred, p_true, dim=-1)


def static_baseline(p_true: Tensor) -> Tensor:
    """Infidelity curve of the trivial predictor p(tau) = p(tau^(1))."""
    return infidelity_curve(p_true[..., :1, :].expand_as(p_true), p_true)


def mean_relative_error(p_pred: Tensor, p_true: Tensor, active_threshold: float = 1e-4) -> Tensor:
    """Mean relative error over the channels that move by more than active_threshold."""
    swing = (p_true - p_true[..., :1, :]).abs().amax(dim=-2, keepdim=True)
    active = (swing > active_threshold).expand_as(p_true)
    rel = (p_pred - p_true).abs() / p_true.clamp_min(1e-12)
    n = active.sum(dim=-1).clamp_min(1)
    return (rel * active).sum(dim=-1) / n


# Resonance geometry: what "on resonance" means
#
# A drive frequency is "on resonance" for a block when it lies within
# n_linewidths half-widths eta |Omega_q| of one of the transitions the
# embedding retains (Eq. 11):
#
#     min_q  |omega_res,q - omega| / (eta |Omega_q|)  <=  n_linewidths .
#
# n_linewidths = 1 therefore means "at least 50 % of the maximum
# achievable population transfer is available at this frequency".


def resonance_geometry(molecule: Molecule, block_index: int, sigma: str = "+") -> tuple[np.ndarray, np.ndarray]:
    """(omega_res, linewidth) of one block's retained transitions; linewidth is the HWHM."""
    emb = Embedding(molecule, block_index, sigma)
    return emb.w_res, emb.linewidths


def resonance_distance(molecule: Molecule, omegas: np.ndarray, block_index: int, sigma: str = "+") -> np.ndarray:
    """min_q |omega_res,q - omega| / (eta |Omega_q|) -- distance in HWHM."""
    return Embedding(molecule, block_index, sigma).resonance_distance(omegas)


def is_on_resonance(
    molecule: Molecule, omegas: np.ndarray, block_index: int, sigma: str = "+", n_linewidths: float = 1.0
) -> np.ndarray:
    """Boolean mask: within n_linewidths HWHM of a retained transition."""
    return resonance_distance(molecule, omegas, block_index, sigma) <= n_linewidths


def active_fraction(
    molecule: Molecule, block_index: int, sigma: str = "+", n_linewidths: float = 1.0,
    n_samples: int = 200_000, seed: int = 0,
) -> float:
    """Fraction of the drive window that is on resonance (Monte-Carlo)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(molecule.window.omega_min, molecule.window.omega_max, size=n_samples)
    return float(is_on_resonance(molecule, w, block_index, sigma, n_linewidths).mean())


# Stratified summary


def stratified_summary(time_avg: np.ndarray, static_time_avg: np.ndarray, on_mask: np.ndarray, prefix: str = "") -> dict:
    """Median / p95 / max of <Ibar_p>_tau and its static baseline per stratum, plus their ratio."""
    out: dict = {}
    for name, sel in (("on", on_mask), ("off", ~on_mask), ("all", np.ones_like(on_mask))):
        n = int(sel.sum())
        out[f"{prefix}{name}_n"] = n
        if n == 0:
            continue
        out[f"{prefix}{name}_median"] = float(np.median(time_avg[sel]))
        out[f"{prefix}{name}_p95"] = float(np.percentile(time_avg[sel], 95))
        out[f"{prefix}{name}_max"] = float(np.max(time_avg[sel]))
        out[f"{prefix}{name}_static_median"] = float(np.median(static_time_avg[sel]))
        ratio = time_avg[sel] / np.maximum(static_time_avg[sel], 1e-30)
        out[f"{prefix}{name}_ratio_to_static_median"] = float(np.median(ratio))
        out[f"{prefix}{name}_ratio_to_static_p95"] = float(np.percentile(ratio, 95))
    return out


def percentile_bands(curves: np.ndarray, percentiles: tuple = (5, 25, 50, 75, 95)) -> dict:
    """Percentiles across the frequency axis of (n_freq, P_tau) curves."""
    return {int(q): np.percentile(curves, q, axis=0) for q in percentiles}
