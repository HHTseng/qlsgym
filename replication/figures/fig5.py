"""Figure 5 -- the planner's three in-loop engines, from the stage-5 payload."""

from __future__ import annotations

import numpy as np

from .style import PAPER_KW, save

# arm -> (tick label, colour, marker), matplotlib's default cycle.
ARMS = {"exact": ("exact", "C0", "o"),
        "trivial": ("trivial\n$p(\\tau){=}p(0)$", "C1", "s"),
        "fno": ("FNO", "C2", "D")}


def render(s05: dict, paper: dict, outdir, name: str = "fig5") -> list:
    import matplotlib.pyplot as plt

    arms = s05["arms"]
    order = [a for a in s05["protocol"]["arms"] if a in arms]
    quoted = (paper.get("FIG5A_PAPER") or {}).get("dw1.0") or {}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.0), constrained_layout=True)
    _panel(a, arms, order, quoted, "convergence", "Converged fraction",
           f"exact propagation, {s05['protocol']['n_rollouts_per_seed']} rollouts/seed",
           legend=True)
    _panel(b, arms, order, quoted, "mean_pulses", "Pulses",
           f"budget {s05['protocol']['max_pulses']}", legend=False)
    _successes_only(b, arms, order)
    return save(fig, name, outdir)


def _series(arm: dict, field: str):
    """(per-seed points, mean over seeds, sd over seeds): one population."""
    if field == "convergence":
        v = arm["validated"]
        return v["per_seed"], v["mean"], v.get("sd")
    per = np.asarray(arm["mean_pulses_per_seed"], float)
    sd = float(per.std(ddof=1)) if per.size > 1 else None
    return per, arm.get("mean_pulses"), sd


def _panel(ax, arms: dict, order, quoted: dict, field: str, ylabel: str, title: str,
           legend: bool) -> None:
    for i, key in enumerate(order):
        _label, colour, marker = ARMS[key]
        per, mean, sd = _series(arms[key], field)
        per = np.asarray(per, float)
        x = i + np.linspace(-0.16, 0.16, per.size) if per.size > 1 else np.array([float(i)])
        ax.plot(x, per, marker, ms=4.5, mfc="none", mec=colour, mew=1.0, lw=0,
                label="per seed" if i == 0 else None)
        if mean is not None and np.isfinite(mean):
            ax.errorbar(i, mean, yerr=0.0 if sd is None or not np.isfinite(sd) else sd,
                        fmt=marker, color=colour, ms=9.0, capsize=4.0, lw=0, elinewidth=1.4,
                        label="mean $\\pm$ sd over seeds" if i == 0 else None)
    for role, ls in (("best", "--"), ("refined", ":")):
        v = (quoted.get(role) or {}).get(field)
        if v is not None:
            ax.axhline(v, label=f"paper, {role}", **{**PAPER_KW, "linestyle": ls})
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ARMS[a][0] for a in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(fontsize=8)


def _successes_only(ax, arms: dict, order) -> None:
    """The mean over the converged rollouts alone: a different population, so a tick of its own,
    and the only entry in this panel's legend.
    """
    from matplotlib.lines import Line2D

    drawn = False
    for i, key in enumerate(order):
        v = arms[key].get("mean_pulses_successful")
        if v is None or not np.isfinite(v):
            continue
        ax.plot(i, v, "_", ms=16, mew=2.0, color=ARMS[key][1])
        drawn = True
    if drawn:
        ax.legend(handles=[Line2D([], [], marker="_", ms=10, mew=2.0, ls="none", color="k")],
                  labels=["mean, converged rollouts only"], fontsize=8)
