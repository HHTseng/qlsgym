"""Figure 3 -- forward accuracy of the surrogates, from the stage-4 payload."""

from __future__ import annotations

import numpy as np

from .style import PAPER_KW, TRIVIAL_LS, save

# stratum key -> (label, colour, marker), matplotlib's default cycle.
STRATA = (("uniform", "uniform draw", "C0", "o"),
          ("on", "on resonance", "C1", "^"),
          ("off", "off resonance", "C2", "v"))
# The trivial predictor is exact at tau^(1), so its first sample is machine
# zero; dropped on a log axis only, never from the payload.
DROP = 1


def binned_median(x, y, n_bins: int = 25):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    keep = [(0.5 * (edges[b] + edges[b + 1]), float(np.median(y[idx == b])))
            for b in range(n_bins) if (idx == b).any()]
    return np.asarray([c for c, _ in keep]), np.asarray([m for _, m in keep])


def render(s04: dict, paper: dict, outdir, name: str = "fig3") -> list:
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    _populations(a, s04["fig3a"])
    _bands(b, s04["pooled"]["fig3b"], paper)
    _scatter(c, s04["pooled"]["fig3c"], paper)
    return save(fig, name, outdir)


def _populations(ax, f3a: dict) -> None:
    tau = np.asarray(f3a["tau_ms"], float)
    nu = np.asarray(f3a["nu"], int)
    true, pred = np.asarray(f3a["p_true"], float), np.asarray(f3a["p_pred"], float)
    every = max(1, tau.size // 12)
    for manifold, colour, marker in ((0, "C0", "o"), (1, "C1", "s")):
        for n, j in enumerate(np.flatnonzero(nu == manifold)):
            first = n == 0
            ax.plot(tau, pred[j], color=colour, lw=1.1,
                    label=rf"$\nu={manifold}$ FNO" if first else None)
            ax.plot(tau[::every], true[j][::every], marker, ms=3.4, mfc="none", mec=colour,
                    mew=0.9, lw=0, label=rf"$\nu={manifold}$ exact" if first else None)
    ax.set_title(f"block {f3a['block']} sigma{f3a['sigma']}, "
                 f"$\\omega/2\\pi$ = {f3a['omega_khz']:.1f} kHz")
    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel("Population")
    ax.set_xlim(tau[0], tau[-1])
    ax.legend(ncols=2, fontsize=8)


def _bands(ax, f3b: dict, paper: dict) -> None:
    tau = np.asarray(f3b["tau_ms"], float)
    if f3b.get("uniform"):
        q = {k: np.asarray(v, float) for k, v in f3b["uniform"]["bands"].items()}
        for lo, hi, alpha in (("5", "95", 0.15), ("25", "75", 0.3)):
            ax.fill_between(tau, q[lo], q[hi], color="C0", alpha=alpha, lw=0,
                            label=f"uniform {lo}-{hi}%")
    for key, label, colour, _ in STRATA:
        if not f3b.get(key):
            continue
        ax.plot(tau, np.asarray(f3b[key]["bands"]["50"], float), color=colour, label=label)
        ax.plot(tau[DROP:], np.asarray(f3b[key]["trivial_median"], float)[DROP:], color=colour,
                lw=1.1, ls=TRIVIAL_LS, label=f"{label}, trivial" if key == "uniform" else None)
    ax.axhline(paper["FIG3B_MEDIAN"], label="paper median", **PAPER_KW)
    ax.axhline(paper["FIG3B_P95_BOUND"], label="paper p95", **{**PAPER_KW, "linestyle": ":"})
    ax.set_yscale("log")
    ax.set_title("pooled over trained blocks")
    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$\overline{\mathcal{I}}_p(\tau)$")
    ax.set_xlim(tau[0], tau[-1])
    ax.legend(ncols=2, fontsize=8)


def _scatter(ax, f3c: dict, paper: dict) -> None:
    for key, label, colour, marker in STRATA:
        if not f3c.get(key):
            continue
        ax.scatter(f3c[key]["omega_khz"], f3c[key]["time_avg"], s=9, alpha=0.5, lw=0,
                   color=colour, marker=marker, label=label)
    if f3c.get("uniform"):
        uni = f3c["uniform"]
        ax.plot(*binned_median(uni["omega_khz"], uni["time_avg"]), color="C0", lw=2.0,
                label="uniform, binned median")
        ax.plot(*binned_median(uni["omega_khz"], uni["trivial_time_avg"]), color="C0", lw=1.4,
                ls=TRIVIAL_LS, label="uniform, trivial")
    ax.axhline(paper["FIG3C_TYPICAL_BOUND"], label="paper bound", **PAPER_KW)
    ax.set_yscale("log")
    ax.set_title("time-averaged, per test frequency")
    ax.set_xlabel(r"$\omega/2\pi$ (kHz)")
    ax.set_ylabel(r"$\langle\overline{\mathcal{I}}_p\rangle_\tau$")
    ax.legend(ncols=2, fontsize=8, scatterpoints=3)
