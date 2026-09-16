"""The finished-episodes curve of arXiv:2410.11839 Fig. 4c / Fig. S7 / Table 1."""

from __future__ import annotations

import math

import numpy as np

from ..policies.rollout import RolloutResult


def finished_curve(lengths, successes, max_pulses: int) -> np.ndarray:
    """(max_pulses + 1,) fraction of episodes finished within k pulses."""
    lengths = np.asarray(lengths, dtype=np.int64)
    successes = np.asarray(successes, dtype=bool)
    if lengths.shape != successes.shape:
        raise ValueError("lengths and successes must have the same length")
    if lengths.size == 0:
        raise ValueError("no episodes")
    if (lengths < 0).any() or (lengths[successes] > max_pulses).any():
        raise ValueError("a successful episode is longer than max_pulses")
    counts = np.bincount(lengths[successes], minlength=max_pulses + 1)[: max_pulses + 1]
    return np.cumsum(counts) / lengths.size


def pulses_to(curve, frac: float = 0.85) -> int | None:
    """First pulse count at which curve reaches frac; None if never."""
    curve = np.asarray(curve, dtype=np.float64)
    hit = np.where(curve >= frac - 1e-12)[0]
    return int(hit[0]) if hit.size else None


def from_rollout(res: RolloutResult) -> dict:
    """Curve, P85 and the headline statistics of one evaluation."""
    curve = finished_curve(res.lengths, res.successes, res.max_pulses)
    mps = res.mean_pulses_successful
    return {
        "curve": curve.tolist(),
        "p85": pulses_to(curve, 0.85),
        "success": float(res.success_fraction),
        "success_err": float(res.success_err),
        "mean_pulses": float(res.mean_pulses),
        "mean_pulses_successful": None if not math.isfinite(mps) else float(mps),
        "n_episodes": int(res.n_rollouts),
        "max_pulses": int(res.max_pulses),
        "p_target": float(res.p_target),
        "lengths": [int(x) for x in res.lengths],
        "successes": [bool(x) for x in res.successes],
        "outcomes": dict(res.outcomes),
        "target_states": list(res.target_states[:8]),
    }
