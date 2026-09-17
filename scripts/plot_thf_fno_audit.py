#!/usr/bin/env python
"""Plot selection-set FNO error against qlsgym's static population predictor."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/qlsgym_fno/fno_blocks/rlpilot.json"))
    parser.add_argument("--output", type=Path, default=Path("results/qlsgym_fno/thf_fno_block_accuracy.png"))
    args = parser.parse_args()
    audit = json.loads(args.input.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for axis, sigma in zip(axes, ("+", "-")):
        rows = sorted(
            (row for row in audit["blocks"] if row["sigma"] == sigma),
            key=lambda row: row["block"],
        )
        x = np.arange(len(rows))
        axis.bar(x - 0.2, [row["summary"]["best_onres"] for row in rows],
                 width=0.4, color="#0072B2", label="Best pilot FNO")
        axis.bar(x + 0.2, [row["onres_static"] for row in rows],
                 width=0.4, color="#D55E00", label="Static/no-change predictor")
        axis.set_xticks(x, [row["block"] for row in rows])
        axis.set_xlabel(f"ThF+ Hamiltonian block (sigma {sigma})")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("On-resonance population infidelity\nlower is better")
    axes[1].legend(fontsize=8)
    figure.suptitle("Pilot surrogate accuracy — selection set, not independent test")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
