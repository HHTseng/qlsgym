#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import numpy as np

from qlsgym.molecules import thf229_tables as T
from qlsgym.spec import TWO_PI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--j-max", type=int, default=T.J_MAX)
    ap.add_argument("--b-gauss", type=float, default=T.B_FIELD_GAUSS)
    ap.add_argument("--e-vcm", type=float, default=T.E_FIELD_V_PER_CM)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "thf229"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    levels_file, rabi_file = T.table_files(args.outdir, args.j_max, args.b_gauss, args.e_vcm)
    levels, cpl = T.build_tables(args.j_max, args.b_gauss, args.e_vcm)
    T.write_tables(levels, cpl, levels_file, rabi_file, args.j_max, args.b_gauss, args.e_vcm)

    e = levels.energies_MHz
    print(f"{len(levels.labels)} levels, E in [{e.min():.3f}, {e.max():.3f}] MHz -> {levels_file}")
    print(f"{cpl.i_idx.size} sigma+ couplings -> {rabi_file}")
    lab = levels.labels
    js = np.array([l[0] for l in lab])
    f1 = np.array([l[1] for l in lab])
    for name, a in (("dJ", js), ("dF1", f1)):
        d = np.abs(a[cpl.f_idx] - a[cpl.i_idx])
        for v in sorted(set(d.tolist())):
            sel = d == v
            print(f"  {name} = {v:g}: {sel.sum():5d} couplings, |Omega|/2pi in "
                  f"[{np.abs(cpl.omega[sel]).min() / TWO_PI:.4g}, {np.abs(cpl.omega[sel]).max() / TWO_PI:.4g}] kHz")


if __name__ == "__main__":
    main()
