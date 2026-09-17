#!/usr/bin/env python
"""Collect completed production checkpoints; optionally require all 24 pairs."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlsgym import load_molecule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tag", default="rlprod120v2")
    parser.add_argument("--output", type=Path, default=Path("results/thf_fno_blocks"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    mol = load_molecule("thf")
    rows, missing = [], []
    for sigma, code in (("+", "sp"), ("-", "sm")):
        for block in range(12):
            directory = args.work / "runs" / f"{args.tag}_{code}_block{block}"
            if not (directory / "summary.json").exists() or not (directory / "heldout_summary.json").exists():
                missing.append([block, sigma])
                continue
            summary = json.loads((directory / "summary.json").read_text())
            heldout = json.loads((directory / "heldout_summary.json").read_text())
            if summary["epochs_completed"] != 120 or heldout["molecule_fingerprint"] != mol.fingerprint():
                raise ValueError(f"wrong training/physics contract: {directory}")
            rows.append({"block": block, "sigma": sigma, "summary": summary, "heldout": heldout})
    if args.require_complete and missing:
        raise ValueError(f"unfinished/missing production checkpoints: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"tag": args.tag, "fingerprint": mol.fingerprint(), "completed_pairs": len(rows),
              "missing_pairs": missing, "blocks": rows}
    (args.output / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for axis, sigma in zip(axes, ("+", "-")):
            group = [r for r in rows if r["sigma"] == sigma]
            x = np.arange(len(group))
            for shift, mode, color, label in ((-0.2, "diffuse_alpha1", "#0072B2", "Diffuse alpha=1"),
                                             (0.2, "control_mix_alpha_minus2", "#D55E00", "Control mixture")):
                axis.bar(x + shift, [r["heldout"][mode]["on_on_median"] for r in group], width=0.4, color=color, label=label)
            axis.set_xticks(x, [r["block"] for r in group])
            axis.set_xlabel(f"Block (sigma {sigma})")
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("On-resonance population infidelity\nIndependent median; lower better")
        axes[1].legend(fontsize=8)
        figure.suptitle(f"ThF+ production FNO: {len(rows)}/24 completed pairs, 120 epochs")
        figure.tight_layout()
        figure.savefig(args.output / "thf_production_accuracy.png", dpi=180)
        plt.close(figure)
    print(f"collected {len(rows)}/24 pairs into {args.output}")


if __name__ == "__main__":
    main()
