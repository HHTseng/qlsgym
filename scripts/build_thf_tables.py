#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import numpy as np

from qlsgym.molecules import thf, thf_tables
from qlsgym.spec import TWO_PI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--j-max", type=int, default=thf.J_MAX)
    ap.add_argument("--b-gauss", type=float, default=thf.B_FIELD_GAUSS)
    ap.add_argument("--e-vcm", type=float, default=thf.E_FIELD_V_PER_CM)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "thf"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    levels_file, rabi_file = thf_tables.table_files(args.outdir, args.j_max, args.b_gauss, args.e_vcm)
    levels, cpl = thf_tables.build_tables(args.j_max, args.b_gauss, args.e_vcm)
    thf_tables.write_tables(levels, cpl, levels_file, rabi_file, args.j_max, args.b_gauss, args.e_vcm)

    e = levels.energies_MHz
    print(f"{len(levels.labels)} levels, E in [{e.min():.3f}, {e.max():.3f}] MHz -> {levels_file}")
    print(f"{cpl.i_idx.size} sigma+ couplings, Jones {cpl.jones} -> {rabi_file}")
    js = np.array([l[0] for l in levels.labels])
    dj = np.abs(js[cpl.f_idx] - js[cpl.i_idx])
    for d in sorted(set(dj.tolist())):
        sel = dj == d
        print(f"  dJ = {d}: {sel.sum():5d} couplings, |Omega|/2pi in "
              f"[{np.abs(cpl.omega[sel]).min() / TWO_PI:.4g}, {np.abs(cpl.omega[sel]).max() / TWO_PI:.4g}] kHz")


if __name__ == "__main__":
    main()
