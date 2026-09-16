"""Acceptance for the surrogate: qlsgym's BlockFNO / FnoEngine load a real source-package
checkpoint bit-for-bit and reproduce that package's own FnoEngine on real molecules.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.physics.engines import ExactEngine
from qlsgym.surrogate.fno_engine import FnoEngine
from qlsgym.surrogate.manifest import StaleCheckpoint, build_manifest, check_checkpoint_provenance

THFFNO_WORK = os.environ.get("THFFNO_WORK", "")
FNOREPL_WORK = os.environ.get("FNOREPL_WORK", "")
TOL = 1e-6


def test_build_manifest_refuses_current_thf_checkpoint():
    """A real ThF+ checkpoint trained on the pre-D14 tables cannot enter a manifest."""
    run = os.path.join(THFFNO_WORK, "runs", "prod_a1.0_block0")
    if not os.path.isdir(run):
        pytest.skip(f"no run at {run} (SKIPPED)")
    mol = load_molecule("thf")
    for flag in (False, True):
        with pytest.raises(StaleCheckpoint, match="resonance check failed") as exc:
            build_manifest(mol, "stale", {(0, "+"): run}, source="thffno", allow_unprovenanced=flag)
        assert "retrain on the current tables" in str(exc.value) and run in str(exc.value)


def test_thf_mix_sp_block0_port_matches_thffno():
    """PORT check only: qlsgym's FnoEngine and thffno's produce the same numbers from the same
    weights on the same (current) tables.
    """
    pytest.importorskip("thffno", reason="thffno not on PYTHONPATH: source scripts/env.sh (SKIPPED)")
    ckpt = os.path.join(THFFNO_WORK, "runs", "mix_sp_block0", "best_onres.pt")
    if not os.path.exists(ckpt):
        pytest.skip(f"no checkpoint at {ckpt} (SKIPPED)")
    import torch

    from thffno.fno_engine import FnoEngine as SourceFnoEngine

    mol = load_molecule("thf")
    with pytest.raises(StaleCheckpoint, match="resonance check failed"):
        check_checkpoint_provenance(torch.load(ckpt, map_location="cpu", weights_only=False),
                                    os.path.dirname(ckpt), mol, 0, "+", allow_unprovenanced=True)
    mine = FnoEngine(mol, {(0, "+"): ckpt}, fallback=ExactEngine(mol), device="cpu", legacy=True)
    try:
        theirs = SourceFnoEngine({0: ckpt}, device="cpu", allow_unprovenanced=True)
    except TypeError:                                   # thffno predating its provenance guard
        theirs = SourceFnoEngine({0: ckpt}, device="cpu")
    except RuntimeError as exc:
        if type(exc).__name__ != "StaleCheckpoint":
            raise
        pytest.skip(f"thffno's own guard refuses {ckpt} (trained on pre-D14 tables): no port comparison "
                    f"possible until it is retrained (SKIPPED) -- {str(exc)[:160]}")

    e = mol.system.energies - mol.system.energies.min()
    p = np.exp(-e / e.max())
    p = p / p.sum()   # a stand-in thermal-shaped belief (test_parity_thf.py owns the exact thffno comparison)
    w = mol.window
    res = mol.trap.nu_f + (mol.system.energies[mol.blocks[0].states[mol.blocks[0].f_local]]
                           - mol.system.energies[mol.blocks[0].states[mol.blocks[0].i_local]])
    res = np.sort(res[(res > w.omega_min) & (res < w.omega_max)])
    omegas = [float(w.omega_min) + 1.0, float(res[len(res) // 2]), float(w.omega_max) - 1.0]

    max_res = 0.0
    for om in omegas:
        a0, a1 = mine.branches_all_tau(p, om, "+")
        b0, b1 = theirs.branches_all_tau(p, om, "+")
        max_res = max(max_res, float(np.abs(a0 - b0).max()), float(np.abs(a1 - b1).max()))
    assert max_res < TOL, f"qlsgym vs thffno FnoEngine residual {max_res:.3e} >= {TOL:.0e}"
    print(f"\nThF mix_sp_block0 vs thffno.fno_engine.FnoEngine: max residual {max_res:.3e}")


def test_h3o_v3_b0_matches_fnorepl():
    pytest.importorskip("fnorepl", reason="fnorepl not on PYTHONPATH: source scripts/env.sh (SKIPPED)")
    ckpt = os.path.join(FNOREPL_WORK, "runs", "v3_b0_128k", "best.pt")
    if not os.path.exists(ckpt):
        pytest.skip(f"no checkpoint at {ckpt} (SKIPPED)")
    from fnorepl.fno_engine import FnoEngine as SourceFnoEngine

    mol = load_molecule("h3o")
    mine = FnoEngine(mol, {(0, "+"): ckpt}, fallback=ExactEngine(mol), device="cpu", legacy=True)
    theirs = SourceFnoEngine({0: ckpt}, device="cpu")

    e = mol.system.energies - mol.system.energies.min()
    p = np.exp(-e / e.max())
    p = p / p.sum()
    w = mol.window
    res = mol.trap.nu_f + (mol.system.energies[mol.blocks[0].states[mol.blocks[0].f_local]]
                           - mol.system.energies[mol.blocks[0].states[mol.blocks[0].i_local]])
    res = np.sort(res[(res > w.omega_min) & (res < w.omega_max)])
    omegas = [float(w.omega_min) + 1.0, float(res[len(res) // 2]), float(w.omega_max) - 1.0]

    max_res = 0.0
    for om in omegas:
        a0, a1 = mine.branches_all_tau(p, om, "+")
        b0, b1 = theirs.branches_all_tau(p, om, "+")
        max_res = max(max_res, float(np.abs(a0 - b0).max()), float(np.abs(a1 - b1).max()))
    assert max_res < TOL, f"qlsgym vs fnorepl FnoEngine residual {max_res:.3e} >= {TOL:.0e}"
    print(f"\nH3O v3_b0_128k vs fnorepl.fno_engine.FnoEngine: max residual {max_res:.3e}")
