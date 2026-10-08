#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import work_dir
from qlsgym.surrogate.manifest import StaleCheckpoint, build_manifest


def block_runs(work: str, tag: str, sigmas: tuple = ("+", "-"), blocks=range(12)) -> dict:
    runs = {}
    for sigma in sigmas:
        p = "sp" if sigma == "+" else "sm"
        for b in blocks:
            d = os.path.join(work, "runs", f"{tag}_{p}_block{b}")
            if os.path.isdir(d):
                runs[(b, sigma)] = d
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", required=True, choices=["thf", "h3o", "synthetic"])
    ap.add_argument("--tag", required=True, help="the --tag the runs were trained with")
    ap.add_argument("--note", default="", help="free text stored in the manifest")
    ap.add_argument("--checkpoint", default="best_onres.pt", help="checkpoint file inside each run")
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse unless every (block, sigma) has a run")
    ap.add_argument("--no-check-load", dest="check_load", action="store_false",
                    help="skip loading every checkpoint through BlockFNO before writing "
                    "(faster; the provenance check still runs)")
    args = ap.parse_args()

    mol = load_molecule(args.molecule)
    work = work_dir()
    runs = block_runs(work, args.tag, ("+", "-"), range(len(mol.blocks)))
    want = 2 * len(mol.blocks)
    if not runs:
        raise SystemExit(f"no runs named {args.tag}_{{sp,sm}}_block<b> under {work}/runs")
    missing = sorted({(b.index, s) for b in mol.blocks for s in ("+", "-")} - set(runs))
    print(f"[{args.molecule}/{args.tag}] {len(runs)}/{want} runs ({args.checkpoint}) from {work}/runs")
    if missing:
        print(f"  missing (exact fallback): {missing}")
        if args.require_complete:
            raise SystemExit("--require-complete: not writing a partial manifest")
    try:
        m = build_manifest(mol, args.tag, runs, checkpoint_name=args.checkpoint, note=args.note,
                           check_load=args.check_load)
    except StaleCheckpoint as exc:
        bar = "!" * 100
        print(f"\n{bar}\nREFUSED [{args.molecule}/{args.tag}]: checkpoints not trained on the current tables; "
              f"no manifest written.\n{exc}\n{bar}\n", file=sys.stderr)
        raise
    path = m.save()
    print(m.table())
    print(f"-> {path}")


if __name__ == "__main__":
    main()
