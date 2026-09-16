"""Figure 6 -- each approximate engine's own view against exact validation."""

from __future__ import annotations

import numpy as np

from .style import save

ARMS = {"trivial": (r"trivial $p(\tau){=}p(0)$", "C1", "s"), "fno": ("FNO", "C2", "D")}


def render(s05: dict, paper: dict, outdir, name: str = "fig6") -> list:
    import matplotlib.pyplot as plt

    arms = s05["arms"]
    order = [a for a in s05["protocol"]["arms"] if a in ARMS and arms.get(a, {}).get("view")]
    fig, ax = plt.subplots(figsize=(5, 4.0), constrained_layout=True)
    for i, key in enumerate(order):
        label, colour, marker = ARMS[key]
        view = np.asarray(arms[key]["view"]["per_seed"], float)
        val = np.asarray(arms[key]["validated"]["per_seed"], float)
        x = np.array([0.0, 1.0]) + 0.05 * (i - 0.5 * (len(order) - 1))
        for v, e in zip(view, val):
            ax.plot(x, [v, e], color=colour, lw=1.0, alpha=0.5, marker=marker, ms=4.0,
                    mfc="none", mew=1.0)
        ax.plot(x, [view.mean(), val.mean()], color=colour, lw=2.4, marker=marker, ms=9.0,
                label=label)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["engine's own view", "exact validation"])
    ax.set_ylabel("Success fraction")
    ax.set_title("one line per planning seed")
    if order:
        ax.legend(fontsize=8)
    return save(fig, name, outdir)
