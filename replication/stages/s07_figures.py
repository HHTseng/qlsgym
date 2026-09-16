"""Stage 7 -- every figure, from OUTPUTS/replication/ alone (Figs. 3-6)."""

from __future__ import annotations

import os

from ..config import Config, figures_dir, read_result, result_path

NAME = "s07_figures"
TITLE = "Render Figs. 3-6 from the result tree"
PAPER = "Figs. 3-6"
REQUIRES = ("s01_physics", "s04_accuracy", "s05_planner")
COST = "seconds, CPU"

# The constants of replication.paper the renderers quote.  Named here so a
# missing one fails with the name, not with an AttributeError deep in a
# draw call.  These are quotations, never substitutes.
PAPER_CONSTANTS = ("FIG3B_MEDIAN", "FIG3B_P95_BOUND", "FIG3C_TYPICAL_BOUND", "FIG5A_PAPER")


def paper_constants() -> dict:
    from .. import paper as P

    missing = [k for k in PAPER_CONSTANTS if not hasattr(P, k)]
    if missing:
        raise AttributeError(
            "replication/paper.py is missing the constant(s) " + ", ".join(missing)
            + ": every number the paper states lives there as a named quotation; "
              "nothing here may invent one")
    return {k: getattr(P, k) for k in PAPER_CONSTANTS}


def plan(cfg: Config) -> list:
    lines = [f"render fig3, fig5, fig6 to {figures_dir()} as .pdf and .png"]
    for stage in REQUIRES:
        p = result_path(stage)
        lines.append(f"read {p.name}: {'present' if p.exists() else 'MISSING, run ' + stage}")
    lines += [
        "fig3 <- s04_accuracy: the showcase trajectory, the Ibar_p(tau) band stack per "
        "stratum and <Ibar_p>_tau against drive frequency, each with the trivial "
        "p(tau) = p(0) predictor beside it",
        "fig5 <- s05_planner: converged fraction and pulses of the exact / trivial / FNO "
        "arms, every seed drawn, always under exact propagation",
        "fig6 <- s05_planner: each approximate arm's own view against that validation",
        f"quote {len(PAPER_CONSTANTS)} constants from replication/paper.py, drawn grey and "
        "dashed; compute nothing from them",
        "s01_physics is read only to refuse a stale Hamiltonian vintage",
        "no checkpoint, no engine, no torch, no physics: JSON in, figures out",
    ]
    return lines


def run(cfg: Config) -> dict:
    from ..config import load_molecule

    molecule = load_molecule(cfg)
    results = {s: read_result(s, cfg, molecule=molecule) for s in REQUIRES}
    paper = paper_constants()

    from ..figures import fig3, fig5, fig6

    outdir = figures_dir()
    files = (fig3.render(results["s04_accuracy"], paper, outdir)
             + fig5.render(results["s05_planner"], paper, outdir)
             + fig6.render(results["s05_planner"], paper, outdir))
    return {"figures_dir": os.fspath(outdir), "files": files}
