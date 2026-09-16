"""The finished-episodes figure (arXiv:2410.11839 Fig. 4c / Fig. S7 layout)."""

from __future__ import annotations

import os

# arm -> (colour, linestyle, class)
STYLE = {
    "sweeping": ("C4", ":", "baseline"),
    "random": ("C1", ":", "baseline"),
    "planner-exact": ("C2", "--", "exact dynamics"),
    "planner-fno": ("C2", "-", "surrogate in the loop"),
    "rl-exact": ("C0", "--", "exact dynamics"),
    "rl-fno": ("C0", "-", "surrogate in the loop"),
}
DEFAULT_STYLE = ("k", "-", "unknown")
# guides: thin, light grey, solid -- deliberately unlike every arm
GUIDE_COLOUR = "0.78"
GUIDE_WIDTH = 0.8
BUDGET_COLOUR = "0.55"


def style_legend(results: list | None = None) -> str:
    """The colour/linestyle -> arm mapping, for the terminal table."""
    arms = [r["arm"] for r in results] if results else list(STYLE)
    lines = [f"{'arm':>14s}  {'colour':>7s}  {'style':>5s}  class"]
    for arm in arms:
        c, ls, kind = STYLE.get(arm, DEFAULT_STYLE)
        lines.append(f"{arm:>14s}  {c:>7s}  {ls:>5s}  {kind}")
    lines.append(f"{'85% guide':>14s}  {GUIDE_COLOUR:>7s}  {'-':>5s}  horizontal guide")
    lines.append(f"{'task budget':>14s}  {BUDGET_COLOUR:>7s}  {'-':>5s}  vertical guide")
    lines.append(f"{'P85':>14s}  {'arm':>7s}  {'arm':>5s}  vertical mark at each arm's P85")
    return "\n".join(lines)


def draw(ax, results: list, frac: float = 0.85, budget: int | None = None,
         legend: bool = True) -> None:
    """Draw the curves on ax.

    A legend names the curves and is the one piece of text the figure needs:
    without it the reader cannot tell the arms apart, which is content, not
    decoration.  Everything else stays off the image -- no title, no caption,
    no annotation.  `legend=False` for the bare curves.
    """
    import numpy as np

    # guides first, so every arm is drawn over them
    ax.axhline(100.0 * frac, color=GUIDE_COLOUR, linestyle="-", linewidth=GUIDE_WIDTH, zorder=1)
    if budget is not None and results and budget < max(len(r["curve"]) for r in results) - 1:
        ax.axvline(budget, color=BUDGET_COLOUR, linestyle="-", linewidth=GUIDE_WIDTH, zorder=1)
    for r in results:
        colour, ls, _ = STYLE.get(r["arm"], DEFAULT_STYLE)
        y = 100.0 * np.asarray(r["curve"], dtype=float)
        ax.plot(np.arange(y.size), y, color=colour, linestyle=ls, linewidth=1.6, zorder=3,
                label=r["arm"])
        if r.get("p85") is not None:
            ax.axvline(r["p85"], color=colour, linestyle=ls, linewidth=0.9, alpha=0.55, zorder=2)
    ax.set_xlabel("pulses applied")
    ax.set_ylabel("finished episodes (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    if legend and results:
        ax.legend(loc="lower right", frameon=False, fontsize=8)


def render(results: list, out_stem: str, frac: float = 0.85, budget: int | None = None,
           legend: bool = True) -> list:
    """Write <out_stem>.png and .pdf from run_arm records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    draw(ax, results, frac=frac, budget=budget, legend=legend)
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)), exist_ok=True)
    paths = [out_stem + ".png", out_stem + ".pdf"]
    for p in paths:
        fig.savefig(p, dpi=200)
    plt.close(fig)
    return paths


def table(results: list) -> str:
    """One line per arm: success, mean pulses, P85 -- then the figure's style mapping."""
    lines = [f"{'arm':>14s}  {'success':>8s}  {'pulses':>7s}  {'P85':>4s}  {'n':>5s}"]
    for r in results:
        p85 = "-" if r.get("p85") is None else str(r["p85"])
        lines.append(f"{r['arm']:>14s}  {r['success']:8.3f}  {r['mean_pulses']:7.1f}  {p85:>4s}  "
                     f"{r['n_episodes']:5d}")
    lines.append("")
    lines.append("figure key (the arms are also named by the figure's legend):")
    lines.append(style_legend(results))
    return "\n".join(lines)
