"""The two things every replication figure shares."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reserved for the paper's own published values, never for a measurement.
PAPER_KW = dict(color="0.35", linestyle="--", linewidth=1.4, zorder=5)

# p(tau) = p(0), the trivial predictor: its stratum's colour, dotted -- it is our
# measurement, so it is not drawn in the paper style.
TRIVIAL_LS = (0, (1, 1.6))


def save(fig, name: str, outdir) -> list:
    """Write name.pdf and name.png into outdir; return both paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(os.fspath(p), dpi=150, bbox_inches="tight")
        paths.append(os.fspath(p))
    plt.close(fig)
    return paths
