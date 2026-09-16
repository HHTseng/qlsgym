"""qlsgym.policies.score -- the one-step pulse score of arXiv:2410.11839 Eq. 25."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreConfig:
    """Weights and functional choices for Eq. 25 (module docstring)."""

    w_tr: float = 0.5
    w_br: float = 1.5
    w1_scale: float = 1.0
    br_mode: str = "expected"       # 'expected' | 'best' | 'success' | 'progress'
    success_width: float = 0.005    # width of the soft threshold ('success', 'progress')
    p_target: float = 0.98
    eps: float = 1e-12

    def for_molecule(self, molecule) -> "ScoreConfig":
        from dataclasses import replace
        return replace(self, p_target=molecule.task.p_target)


def _clears(q, cfg: ScoreConfig):
    return 1.0 / (1.0 + np.exp(-(q - cfg.p_target) / cfg.success_width))


def score_terms(p_in: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                cfg: ScoreConfig = ScoreConfig()) -> tuple[float, float, float]:
    """(S_tr, S_br, Pi_1) for one candidate pulse (scalar version)."""
    pi1 = float(p1.sum())
    pi0 = float(p0.sum())
    j_star = int(np.argmax(p_in))
    moved = p_in - p0
    s_tr = float(moved[j_star] - (moved.sum() - moved[j_star]))

    def purity(branch, mass):
        return float(branch.max() / mass) if mass > cfg.eps else 0.0

    q0, q1 = purity(p0, pi0), purity(p1, pi1)
    if cfg.br_mode == "expected":
        s_br = pi0 * q0 + pi1 * q1
    elif cfg.br_mode == "best":
        s_br = max(q0, q1)
    elif cfg.br_mode in ("success", "progress"):
        def _c(q):
            return 1.0 / (1.0 + math.exp(-(q - cfg.p_target) / cfg.success_width))
        if cfg.br_mode == "success":
            s_br = pi0 * _c(q0) + pi1 * _c(q1)
        else:
            q_in = float(p_in.max())
            s_br = (q0 - q_in) / max(1.0 - q_in, cfg.eps) + pi1 * _c(q1)
    else:
        raise ValueError(f"unknown br_mode {cfg.br_mode!r}")
    return s_tr, s_br, pi1


def score(p_in: np.ndarray, p0: np.ndarray, p1: np.ndarray, cfg: ScoreConfig = ScoreConfig()) -> float:
    """Eq. 25 for one candidate."""
    s_tr, s_br, pi1 = score_terms(p_in, p0, p1, cfg)
    w1 = cfg.w1_scale * (1.0 - float(p_in.max()))
    return cfg.w_tr * s_tr + cfg.w_br * s_br + w1 * pi1


def score_batch(p_in: np.ndarray, p0: np.ndarray, p1: np.ndarray, cfg: ScoreConfig = ScoreConfig()) -> np.ndarray:
    """Vectorised Eq. 25 over any leading candidate axes: p0, p1 of shape (..., n) for one p_in of
    shape (n,) -> scores (...).
    """
    p_in = np.asarray(p_in, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    pi0 = p0.sum(-1)
    pi1 = p1.sum(-1)
    j_star = int(np.argmax(p_in))
    moved = p_in - p0
    s_tr = 2.0 * moved[..., j_star] - moved.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = np.where(pi0 > cfg.eps, p0.max(-1) / np.maximum(pi0, cfg.eps), 0.0)
        q1 = np.where(pi1 > cfg.eps, p1.max(-1) / np.maximum(pi1, cfg.eps), 0.0)
    if cfg.br_mode == "expected":
        s_br = pi0 * q0 + pi1 * q1
    elif cfg.br_mode == "best":
        s_br = np.maximum(q0, q1)
    elif cfg.br_mode == "success":
        s_br = pi0 * _clears(q0, cfg) + pi1 * _clears(q1, cfg)
    elif cfg.br_mode == "progress":
        q_in = float(p_in.max())
        gain = (q0 - q_in) / max(1.0 - q_in, cfg.eps)
        s_br = gain + pi1 * _clears(q1, cfg)
    else:
        raise ValueError(f"unknown br_mode {cfg.br_mode!r}")
    w1 = cfg.w1_scale * (1.0 - float(p_in.max()))
    return cfg.w_tr * s_tr + cfg.w_br * s_br + w1 * pi1
