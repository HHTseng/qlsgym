from __future__ import annotations

import numpy as np
import torch

from ..spec import Molecule
from .embedding import Embedding

Tensor = torch.Tensor


def population_fidelity(p_pred: Tensor, p_true: Tensor, dim: int = -1) -> Tensor:
    root = torch.sqrt(p_pred.clamp_min(0) * p_true.clamp_min(0))
    return root.sum(dim=dim).square()


def population_infidelity(p_pred: Tensor, p_true: Tensor, dim: int = -1) -> Tensor:
    return (1.0 - population_fidelity(p_pred, p_true, dim=dim)).clamp_min(0.0)


def infidelity_curve(p_pred: Tensor, p_true: Tensor) -> Tensor:
    return population_infidelity(p_pred, p_true, dim=-1)


def static_baseline(p_true: Tensor) -> Tensor:
    return infidelity_curve(p_true[..., :1, :].expand_as(p_true), p_true)


def mean_relative_error(p_pred: Tensor, p_true: Tensor, active_threshold: float = 1e-4) -> Tensor:
    swing = (p_true - p_true[..., :1, :]).abs().amax(dim=-2, keepdim=True)
    active = (swing > active_threshold).expand_as(p_true)
    rel = (p_pred - p_true).abs() / p_true.clamp_min(1e-12)
    n = active.sum(dim=-1).clamp_min(1)
    return (rel * active).sum(dim=-1) / n


def resonance_geometry(molecule: Molecule, block_index: int, sigma: str = "+") -> tuple[np.ndarray, np.ndarray]:
    emb = Embedding(molecule, block_index, sigma)
    return emb.w_res, emb.linewidths


def resonance_distance(molecule: Molecule, omegas: np.ndarray, block_index: int, sigma: str = "+") -> np.ndarray:
    return Embedding(molecule, block_index, sigma).resonance_distance(omegas)


def is_on_resonance(
    molecule: Molecule, omegas: np.ndarray, block_index: int, sigma: str = "+", n_linewidths: float = 1.0
) -> np.ndarray:
    return resonance_distance(molecule, omegas, block_index, sigma) <= n_linewidths


def active_fraction(
    molecule: Molecule, block_index: int, sigma: str = "+", n_linewidths: float = 1.0,
    n_samples: int = 200_000, seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    w = rng.uniform(molecule.window.omega_min, molecule.window.omega_max, size=n_samples)
    return float(is_on_resonance(molecule, w, block_index, sigma, n_linewidths).mean())


def stratified_summary(time_avg: np.ndarray, static_time_avg: np.ndarray, on_mask: np.ndarray, prefix: str = "") -> dict:
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
    return {int(q): np.percentile(curves, q, axis=0) for q in percentiles}
