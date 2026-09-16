"""Manifest round trip in a temp work dir, qlsgym and legacy sources. The provenance guard itself
is tested in test_surrogate_provenance.py.
"""
import json

import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.manifest import (Manifest, StaleCheckpoint, build_manifest, load_manifest, manifest_path,
                                       parse_key, table_stamp)
from qlsgym.surrogate.train import FingerprintMismatch, TrainConfig, checkpoint_dict

TINY = FNOConfig(n_modes=4, hidden_channels=8, n_layers=1, lifting_channel_ratio=2, projection_channel_ratio=2)


class FakeExact:
    def __init__(self, molecule, tau_indices=None):
        self.molecule = molecule
        self.tau_indices = np.arange(molecule.window.n_tau) if tau_indices is None else np.asarray(tau_indices)

    def branches_all_tau(self, p_in, omega, sigma):
        p0 = np.repeat(np.asarray(p_in)[None], self.tau_indices.size, 0)
        return p0, np.zeros_like(p0)


def make_runs(mol, root, legacy=False, ratio=0.5):
    """qlsgym runs carry the table stamp train writes; legacy thffno runs carry none."""
    runs = {}
    torch.manual_seed(0)
    for b in mol.blocks:
        m = BlockFNO.for_block(mol, b.index, "+", TINY)
        d = root / f"run_block{b.index}"
        d.mkdir()
        ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(n_freq=3, n_init=2, alpha=-2.0),
                             DataConfig(n_freq=1, n_init=1), 7, 0.1, 0.2,
                             tables=None if legacy else table_stamp(mol))
        if legacy:
            for k in ("fingerprint", "molecule", "n_nu"):
                ck["meta"].pop(k)
            ck.pop("source")
        torch.save(ck, d / "best_onres.pt")
        (d / "summary.json").write_text(json.dumps({"block": b.index, "strat_on_all_ratio_to_static_median": ratio,
                                                    "best_onres_epoch": 7, "n_pairs": 3}))
        runs[(b.index, "+")] = str(d)
    return runs


def test_round_trip_qlsgym_source(tmp_path):
    mol = load_molecule("synthetic")
    work = tmp_path / "work"
    runs = make_runs(mol, tmp_path)
    m = build_manifest(mol, "unit", runs, source="qlsgym", note="test")
    assert m.fingerprint == mol.fingerprint() and set(m.entries) == {"0,+", "1,+", "2,+"}
    assert {e.provenance for e in m.entries.values()} == {"fingerprint"}
    e = m.entries["1,+"]
    assert e.onres_ratio_to_static == 0.5 and e.summary["best_onres_epoch"] == 7 and e.train_data["alpha"] == -2.0
    path = m.save(work=str(work))
    assert path == manifest_path("synthetic", "unit", str(work))
    j = json.load(open(path))
    assert j["entries"]["0,+"]["path"].startswith("/") and j["source"] == "qlsgym"
    m2 = Manifest.load(path)
    assert m2.checkpoints() == m.checkpoints() and parse_key("2,+") == (2, "+")
    assert "onres ratio 0.5000" in m2.table()
    eng = load_manifest(mol, "unit", work=str(work), fallback=FakeExact(mol), device="cpu")
    assert eng.trained == {(0, "+"), (1, "+"), (2, "+")} and eng.manifest.tag == "unit"
    p = np.full(mol.n_states, 1 / mol.n_states)
    p0, p1 = eng.branches_all_tau(p, mol.trap.nu_f, "+")
    assert p0.shape == (mol.window.n_tau, mol.n_states)
    # subset + fingerprint guard
    eng = load_manifest(mol, "unit", work=str(work), fallback=FakeExact(mol), device="cpu", blocks={(0, "+")})
    assert eng.trained == {(0, "+")}
    other = load_molecule("synthetic", seed=3)
    with pytest.raises(RuntimeError):
        load_manifest(other, "unit", work=str(work), fallback=FakeExact(other), device="cpu")
    with pytest.raises(FingerprintMismatch):
        build_manifest(other, "bad", runs, source="qlsgym")


def test_legacy_source(tmp_path):
    mol = load_molecule("synthetic")
    runs = make_runs(mol, tmp_path, legacy=True)
    # a source-package checkpoint has no fingerprint: refused as "qlsgym", accepted as legacy
    with pytest.raises(FingerprintMismatch):
        build_manifest(mol, "leg", runs, source="qlsgym")
    # ... and, having no table stamp and no stratified_eval.npz, it is refused by the provenance guard
    # unless explicitly admitted as "unverified"
    with pytest.raises(StaleCheckpoint, match="3 of 3 checkpoints"):
        build_manifest(mol, "leg", runs, source="thffno")
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        m = build_manifest(mol, "leg", runs, source="thffno", allow_unprovenanced=True)
    assert {e.provenance for e in m.entries.values()} == {"unverified"}
    path = m.save(work=str(tmp_path / "work"))
    with pytest.warns(UserWarning, match="unverified"):
        eng = load_manifest(mol, "leg", work=str(tmp_path / "work"), fallback=FakeExact(mol), device="cpu")
    assert eng.trained == {(0, "+"), (1, "+"), (2, "+")}
    assert Manifest.load(path).source == "thffno"
    # missing checkpoint / wrong block registration are errors
    with pytest.raises(FileNotFoundError):
        build_manifest(mol, "x", {(0, "+"): str(tmp_path / "nope")}, source="thffno")
    with pytest.raises(ValueError):
        build_manifest(mol, "x", {(1, "+"): runs[(0, "+")]}, source="thffno", allow_unprovenanced=True)
