import dataclasses
import json
import os
import sys
import warnings

import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.manifest import (StaleCheckpoint, build_manifest, check_checkpoint_provenance,
                                       load_manifest, manifest_path, sha256_file, table_stamp)
from qlsgym.surrogate.train import TrainConfig, checkpoint_dict

TINY = FNOConfig(n_modes=4, hidden_channels=8, n_layers=1, lifting_channel_ratio=2, projection_channel_ratio=2)
RETRAIN = "retrain on the current tables"


class FakeExact:
    def __init__(self, molecule, tau_indices=None):
        self.molecule = molecule
        self.tau_indices = np.arange(molecule.window.n_tau) if tau_indices is None else np.asarray(tau_indices)

    def branches_all_tau(self, p_in, omega, sigma):
        p0 = np.repeat(np.asarray(p_in)[None], self.tau_indices.size, 0)
        return p0, np.zeros_like(p0)


@pytest.fixture
def tabled(tmp_path):
    lf, rf = tmp_path / "levels.txt", tmp_path / "rabi.txt"
    lf.write_text("# levels\n0 0.0\n1 1.0\n")
    rf.write_text("# rabi\n0 1 1.0 0.0 1.0\n")
    base = load_molecule("synthetic")
    mol = dataclasses.replace(base, provenance={**base.provenance, "levels_file": str(lf), "rabi_file": str(rf),
                                                "heff_git_rev": "abc1234", "heff_spec_hash": "f00d"})
    return mol, lf, rf


def _run_dir(tmp_path, name="run"):
    d = tmp_path / name
    d.mkdir()
    return str(d)


def _no_warnings():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("error")
    return ctx


def test_table_stamp(tabled):
    mol, lf, rf = tabled
    s = table_stamp(mol)
    assert s == {"levels_sha256": sha256_file(lf), "rabi_sha256": sha256_file(rf),
                 "heff_git_rev": "abc1234", "heff_spec_hash": "f00d"}
    import hashlib
    assert s["levels_sha256"] == hashlib.sha256(lf.read_bytes()).hexdigest()
    assert table_stamp(load_molecule("synthetic")) == {"heff_git_rev": None, "heff_spec_hash": None}
    m = BlockFNO.for_block(mol, 0, "+", TINY)
    ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(), DataConfig(), 0, 0.1, 0.2, tables=s)
    assert ck["tables"] == s


def test_hash_match_accepts(tabled, tmp_path):
    mol, _, _ = tabled
    ck = {"meta": {}, "tables": table_stamp(mol)}
    ctx = _no_warnings()
    try:
        assert check_checkpoint_provenance(ck, _run_dir(tmp_path), mol, 0, "+") == "tables-sha256"
    finally:
        ctx.__exit__(None, None, None)


def test_hash_mismatch_refuses_even_with_flag(tabled, tmp_path):
    mol, lf, rf = tabled
    ck = {"meta": {}, "tables": table_stamp(mol)}
    rf.write_text("# rabi\n0 1 1.0 0.0 1.0001\n")
    run = _run_dir(tmp_path)
    with pytest.raises(StaleCheckpoint, match="tables-sha256 check failed") as exc:
        check_checkpoint_provenance(ck, run, mol, 0, "+")
    assert run in str(exc.value) and RETRAIN in str(exc.value) and "rabi_sha256" in str(exc.value)
    for bad in ({"heff_git_rev": None}, "garbage"):
        with pytest.raises(StaleCheckpoint, match="tables-sha256 check failed"):
            check_checkpoint_provenance({"meta": {}, "tables": bad}, run, mol, 0, "+")


def test_stamp_on_tableless_molecule_uses_fingerprint(tmp_path):
    mol = load_molecule("synthetic")
    run = _run_dir(tmp_path)
    ok = {"meta": {"fingerprint": mol.fingerprint()}, "tables": table_stamp(mol)}
    assert check_checkpoint_provenance(ok, run, mol, 0, "+") == "fingerprint"
    bad = {"meta": {"fingerprint": "000000000000"}, "tables": table_stamp(mol)}
    with pytest.raises(StaleCheckpoint, match="fingerprint check failed"):
        check_checkpoint_provenance(bad, run, mol, 0, "+")


@pytest.mark.parametrize("molecule", ["tabled", "synthetic"])
def test_no_stamp_is_refused(tabled, tmp_path, molecule):
    mol = tabled[0] if molecule == "tabled" else load_molecule("synthetic")
    run = _run_dir(tmp_path)
    with pytest.raises(StaleCheckpoint, match="no table stamp") as exc:
        check_checkpoint_provenance({"meta": {"fingerprint": mol.fingerprint()}}, run, mol, 0, "+")
    assert run in str(exc.value) and RETRAIN in str(exc.value)


def _save_ckpt(mol, run, block, stamp=None, sigma="+"):
    m = BlockFNO.for_block(mol, block, sigma, TINY)
    ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(alpha=-2.0), DataConfig(), 0, 0.1, 0.2, tables=stamp)
    torch.save(ck, f"{run}/best_onres.pt")


def test_build_manifest_checks_every_entry_and_records_provenance(tabled, tmp_path):
    mol, _, _ = tabled
    torch.manual_seed(0)
    stamped, bare = _run_dir(tmp_path, "stamped"), _run_dir(tmp_path, "bare")
    _save_ckpt(mol, stamped, 0, table_stamp(mol))
    _save_ckpt(mol, bare, 1)
    with pytest.raises(StaleCheckpoint, match="1 of 2 checkpoints") as exc:
        build_manifest(mol, "prov", {(0, "+"): stamped, (1, "+"): bare})
    assert bare in str(exc.value) and stamped not in str(exc.value)

    m = build_manifest(mol, "prov", {(0, "+"): stamped})
    assert {k: e.provenance for k, e in m.entries.items()} == {"0,+": "tables-sha256"}
    work = str(tmp_path / "work")
    m.save(work=work)
    ctx = _no_warnings()
    try:
        eng = load_manifest(mol, "prov", work=work, fallback=FakeExact(mol), device="cpu")
    finally:
        ctx.__exit__(None, None, None)
    assert eng.trained == {(0, "+")}


def test_build_manifest_from_mix_run_dirs_checks_the_stamp(tabled, tmp_path):
    mol, _, rf = tabled
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import make_manifest

    torch.manual_seed(0)
    work = tmp_path / "work"
    (work / "runs").mkdir(parents=True)
    want = {(0, "+"), (1, "-")}
    for b, sg in sorted(want):
        d = work / "runs" / f"mix_{'sp' if sg == '+' else 'sm'}_block{b}"
        d.mkdir()
        _save_ckpt(mol, str(d), b, table_stamp(mol), sigma=sg)
    (work / "runs" / "other_sp_block0").mkdir()
    runs = make_manifest.block_runs(str(work), "mix", ("+", "-"), range(len(mol.blocks)))
    assert set(runs) == want

    ctx = _no_warnings()
    try:
        m = build_manifest(mol, "mix", runs)
    finally:
        ctx.__exit__(None, None, None)
    assert {k: e.provenance for k, e in m.entries.items()} == {"0,+": "tables-sha256", "1,-": "tables-sha256"}
    assert m.entries["0,+"].train_data["alpha"] == -2.0
    m.save(work=str(work))

    rf.write_text("# rabi\n0 1 1.0 0.0 1.0001\n")
    with pytest.raises(StaleCheckpoint, match="tables-sha256 check failed") as exc:
        build_manifest(mol, "mix", runs)
    assert "2 of 2 checkpoints" in str(exc.value) and RETRAIN in str(exc.value)
    eng = load_manifest(mol, "mix", work=str(work), fallback=FakeExact(mol), device="cpu")
    assert eng.trained == want


@pytest.mark.parametrize("value", [None, "", "trust-me"])
def test_load_manifest_refuses_entry_without_provenance(tabled, tmp_path, value):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    _save_ckpt(mol, run, 0, table_stamp(mol))
    work = str(tmp_path / "work")
    path = build_manifest(mol, "old", {(0, "+"): run}).save(work=work)
    j = json.load(open(path))
    if value is None:
        del j["entries"]["0,+"]["provenance"]
    else:
        j["entries"]["0,+"]["provenance"] = value
    json.dump(j, open(path, "w"))
    for kw in ({}, {"allow_mismatch": True}):
        with pytest.raises(StaleCheckpoint, match="no recorded provenance") as exc:
            load_manifest(mol, "old", work=work, fallback=FakeExact(mol), device="cpu", **kw)
        assert "0,+" in str(exc.value)
    assert manifest_path(mol.name, "old", work) == path
