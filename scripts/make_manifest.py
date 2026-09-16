#!/usr/bin/env python
"""Build checkpoint manifests from the source projects' runs."""
from __future__ import annotations

import argparse
import os
import sys

from qlsgym import load_molecule
from qlsgym.surrogate.manifest import StaleCheckpoint, build_manifest


def thffno_runs(work: str, prefix: str, sigmas: tuple, blocks=range(12)) -> dict:
    runs = {}
    for sigma in sigmas:
        p = "sp" if sigma == "+" else "sm"
        for b in blocks:
            d = os.path.join(work, "runs", f"{prefix}{p}_block{b}")
            if os.path.isdir(d):
                runs[(b, sigma)] = d
    return runs


def thffno_a1_runs(work: str, blocks=range(12)) -> dict:
    """prod_a1.0_block* (sigma+) and prod_a1.0_sm_block* (sigma-)."""
    runs = {}
    for b in blocks:
        for sigma, name in (("+", f"prod_a1.0_block{b}"), ("-", f"prod_a1.0_sm_block{b}")):
            d = os.path.join(work, "runs", name)
            if os.path.isdir(d):
                runs[(b, sigma)] = d
    return runs


# block_index -> best available fnorepl run directory (module docstring)
H3O_UNIQUE_BLOCK_RUNS = {
    0: "v3_b0_128k",
    2: "v3_b2_256k",
    3: "v3_b3_64k",
    5: "v3_b5_16k",
    7: "v3_b7_128k",
    8: "v2_b8_unif_256k",
}


def fnorepl_runs(work: str) -> dict:
    runs = {}
    for b, name in H3O_UNIQUE_BLOCK_RUNS.items():
        d = os.path.join(work, "runs", name)
        if os.path.isdir(d):
            runs[(b, "+")] = d
    return runs


def validate(manifest, molecule_name: str) -> None:
    """Compare each entry against the source package's own FnoEngine, where importable."""
    import numpy as np

    from qlsgym.physics.engines import ExactEngine
    from qlsgym.surrogate.fno_engine import FnoEngine
    from qlsgym.surrogate.manifest import parse_key

    try:
        if molecule_name == "thf":
            from thffno.fno_engine import FnoEngine as SourceFnoEngine
        elif molecule_name == "h3o":
            from fnorepl.fno_engine import FnoEngine as SourceFnoEngine
        else:
            print(f"no source package known for molecule {molecule_name!r}; skipping --validate")
            return
    except ImportError:
        print("source package not importable (source scripts/env.sh); skipping --validate")
        return

    mol = load_molecule(molecule_name)
    fallback = ExactEngine(mol)
    e = mol.system.energies - mol.system.energies.min()
    p = np.exp(-e / e.max()); p = p / p.sum()
    max_seen = 0.0
    for key, entry in manifest.entries.items():
        b, sigma = parse_key(key)
        if sigma != "+":
            continue   # the source FnoEngine never routes sigma- through the surrogate (Q19)
        mine = FnoEngine(mol, {(b, "+"): entry.path}, fallback=fallback, device="cpu", legacy=True)
        theirs = SourceFnoEngine({b: entry.path}, device="cpu")
        w = mol.window
        res = mol.trap.nu_f + (mol.system.energies[mol.blocks[b].states[mol.blocks[b].f_local]]
                               - mol.system.energies[mol.blocks[b].states[mol.blocks[b].i_local]])
        res = np.sort(res[(res > w.omega_min) & (res < w.omega_max)])
        om = float(res[len(res) // 2]) if res.size else 0.5 * (w.omega_min + w.omega_max)
        a0, a1 = mine.branches_all_tau(p, om, "+")
        b0, b1 = theirs.branches_all_tau(p, om, "+")
        resid = max(float(np.abs(a0 - b0).max()), float(np.abs(a1 - b1).max()))
        entry.parity_residual = resid
        max_seen = max(max_seen, resid)
        print(f"  validated block {b} sigma+: residual {resid:.3e}")
    manifest.validated = {"method": "FnoEngine.branches_all_tau vs source FnoEngine, thermal state, "
                          "one mid-window sigma+ omega per block (port check only, not physics vintage)",
                          "max_residual": max_seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", required=True, choices=["thf", "h3o"])
    ap.add_argument("--validate", action="store_true", help="run the real-checkpoint parity "
                    "check against the source package's own FnoEngine and record the residual")
    ap.add_argument("--check-load", dest="check_load", action="store_true", default=True)
    ap.add_argument("--no-check-load", dest="check_load", action="store_false",
                    help="skip loading every checkpoint through qlsgym's BlockFNO before writing "
                    "the manifest (faster, but no structural guarantee; the provenance guard still runs)")
    ap.add_argument("--allow-unprovenanced", action="store_true",
                    help="admit checkpoints whose provenance cannot be checked (no ck['tables'] stamp and "
                    "no stratified_eval.npz) as provenance 'unverified'; never admits one that FAILED a check")
    args = ap.parse_args()

    mol = load_molecule(args.molecule)

    if args.molecule == "thf":
        work = os.environ.get("THFFNO_WORK")
        if not work:
            raise SystemExit("THFFNO_WORK is not set; source scripts/env.sh")
        specs = [
            ("mix", thffno_runs(work, "mix_", ("+", "-")), "thffno",
             "log-uniform concentrated mixture (alpha=-2); recommended for closed-loop control"),
            ("a1.0", thffno_a1_runs(work), "thffno",
             "uniform-simplex training (alpha=1, the paper's implicit choice); kept for comparison"),
        ]
    else:
        work = os.environ.get("FNOREPL_WORK")
        if not work:
            raise SystemExit("FNOREPL_WORK is not set; source scripts/env.sh")
        # H3O+ physics is unchanged since these were trained; they carry no table stamp but every run
        # has stratified_eval.npz and passes the resonance check, so no --allow-unprovenanced is needed.
        specs = [
            ("prod", fnorepl_runs(work), "fnorepl",
             "one checkpoint per unique (K, parity) block [0,2,3,5,7,8]; the four parity "
             "partners are NOT covered (qlsgym does not implement parity relabelling) and fall "
             "back to the exact engine through FnoEngine"),
        ]

    # build (and provenance-check) every tag before writing any, so a refusal leaves nothing behind
    built = []
    for tag, runs, source, note in specs:
        if not runs:
            print(f"[{args.molecule}/{tag}] no run directories found under {work}/runs; skipping")
            continue
        ckpt_name = "best_onres.pt" if source == "thffno" else "best.pt"
        print(f"[{args.molecule}/{tag}] {len(runs)} checkpoints ({ckpt_name}) from {work}/runs")
        try:
            m = build_manifest(mol, tag, runs, source=source, checkpoint_name=ckpt_name, note=note,
                               check_load=args.check_load, allow_unprovenanced=args.allow_unprovenanced)
        except StaleCheckpoint as exc:
            bar = "!" * 100
            print(f"\n{bar}\nREFUSED [{args.molecule}/{tag}]: checkpoints not trained on the current tables; "
                  f"NO manifest written for --molecule {args.molecule}.\n{exc}\n{bar}\n", file=sys.stderr)
            raise
        built.append(m)

    for m in built:
        if args.validate:
            validate(m, args.molecule)
        path = m.save()
        print(m.table())
        print(f"-> {path}\n")


if __name__ == "__main__":
    main()
