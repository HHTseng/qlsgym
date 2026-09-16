"""The physics-vintage guard check_checkpoint_provenance (contract shared with thffno.train) and its
enforcement in build_manifest / load_manifest.
"""
import dataclasses
import json
import warnings

import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.manifest import (StaleCheckpoint, build_manifest, check_checkpoint_provenance,
                                       current_resonances, load_manifest, manifest_path, reference_hwhm,
                                       sha256_file, table_stamp)
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
    """The synthetic molecule, with provenance pointing at two table files (only their bytes are
    hashed; the physics stays the synthetic one).
    """
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


def _write_eval(run_dir, omegas):
    om = np.asarray(omegas, dtype=np.float64)
    np.savez_compressed(f"{run_dir}/stratified_eval.npz", on_omegas=om, on_on_resonance=np.ones(om.size, bool),
                        on_time_avg=np.zeros(om.size))


def _on_resonance(mol, block, sigma, n, rng, spread=1.0):
    res = current_resonances(mol, block, sigma)
    hw, _ = reference_hwhm(mol, block)
    return res[rng.integers(0, res.size, n)] + rng.uniform(-spread, spread, n) * hw


def _off_resonance(mol, block, sigma, n, min_hwhm=20.0):
    res = current_resonances(mol, block, sigma)
    hw, _ = reference_hwhm(mol, block)
    grid = np.linspace(mol.window.omega_min + 1, mol.window.omega_max - 1, 20000)
    far = grid[np.abs(grid[:, None] - res[None]).min(1) > min_hwhm * hw]
    return far[np.linspace(0, far.size - 1, n).astype(int)]


def _no_warnings():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("error")
    return ctx


# the stamp


def test_table_stamp(tabled):
    mol, lf, rf = tabled
    s = table_stamp(mol)
    assert s == {"levels_sha256": sha256_file(lf), "rabi_sha256": sha256_file(rf),
                 "heff_git_rev": "abc1234", "heff_spec_hash": "f00d"}
    import hashlib
    assert s["levels_sha256"] == hashlib.sha256(lf.read_bytes()).hexdigest()
    # no table files -> no hashes; no heff header -> None
    assert table_stamp(load_molecule("synthetic")) == {"heff_git_rev": None, "heff_spec_hash": None}
    m = BlockFNO.for_block(mol, 0, "+", TINY)
    ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(), DataConfig(), 0, 0.1, 0.2, tables=s)
    assert ck["tables"] == s


def test_reference_hwhm_real_molecules():
    thf, h3o = load_molecule("thf"), load_molecule("h3o")
    assert abs(reference_hwhm(thf, 0)[0] / (2 * np.pi) - 0.2028) < 1e-3       # 3 kHz reference rung
    assert abs(reference_hwhm(h3o, 0)[0] / (2 * np.pi) - 0.1793) < 1e-3       # 2 kHz, paper Eq. 39
    syn = load_molecule("synthetic")
    hw, src = reference_hwhm(syn, 1)
    eta = syn.trap.eta
    assert "max |Omega|" in src and np.isclose(hw, eta * np.exp(-eta ** 2 / 2) * np.abs(syn.blocks[1].omega).max())


# rule 1: stamped checkpoints


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
    rf.write_text("# rabi\n0 1 1.0 0.0 1.0001\n")          # tables regenerated after training
    run = _run_dir(tmp_path)
    _write_eval(run, _on_resonance(mol, 0, "+", 50, np.random.default_rng(0)))   # a passing npz changes nothing
    for flag in (False, True):
        with pytest.raises(StaleCheckpoint, match="tables-sha256 check failed") as exc:
            check_checkpoint_provenance(ck, run, mol, 0, "+", allow_unprovenanced=flag)
        assert run in str(exc.value) and RETRAIN in str(exc.value) and "rabi_sha256" in str(exc.value)
    # a stamp without hashes, or not a dict at all, cannot vouch for a molecule that has table files
    for bad in ({"heff_git_rev": None}, "garbage"):
        with pytest.raises(StaleCheckpoint, match="tables-sha256 check failed"):
            check_checkpoint_provenance({"meta": {}, "tables": bad}, run, mol, 0, "+", True)


def test_stamp_on_tableless_molecule_uses_fingerprint(tmp_path):
    mol = load_molecule("synthetic")
    run = _run_dir(tmp_path)
    ok = {"meta": {"fingerprint": mol.fingerprint()}, "tables": table_stamp(mol)}
    assert check_checkpoint_provenance(ok, run, mol, 0, "+") == "fingerprint"
    bad = {"meta": {"fingerprint": "000000000000"}, "tables": table_stamp(mol)}
    with pytest.raises(StaleCheckpoint, match="fingerprint check failed"):
        check_checkpoint_provenance(bad, run, mol, 0, "+", allow_unprovenanced=True)


# rule 2: unstamped checkpoints


@pytest.mark.parametrize("sigma", ["+", "-"])
def test_no_stamp_resonance_pass_warns_and_accepts(tabled, tmp_path, sigma):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    _write_eval(run, _on_resonance(mol, 0, sigma, 100, np.random.default_rng(1)))
    with pytest.warns(UserWarning, match="resonance check"):
        assert check_checkpoint_provenance({"meta": {}}, run, mol, 0, sigma) == "resonance-check"


def test_no_stamp_resonance_fail_refuses_even_with_flag(tabled, tmp_path):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    _write_eval(run, _off_resonance(mol, 0, "+", 100))
    for flag in (False, True):
        with pytest.raises(StaleCheckpoint, match="resonance check failed") as exc:
            check_checkpoint_provenance({"meta": {}}, run, mol, 0, "+", allow_unprovenanced=flag)
        assert run in str(exc.value) and RETRAIN in str(exc.value) and "0%" in str(exc.value)
    # the resonances of the *other* sigma are not this checkpoint's resonances
    run2 = _run_dir(tmp_path, "run2")
    minus = current_resonances(mol, 0, "-")
    if np.abs(minus[:, None] - current_resonances(mol, 0, "+")[None]).min() > 10 * reference_hwhm(mol, 0)[0]:
        _write_eval(run2, minus)
        with pytest.raises(StaleCheckpoint, match="resonance check failed"):
            check_checkpoint_provenance({"meta": {}}, run2, mol, 0, "+")


def test_resonance_threshold_is_90_percent(tabled, tmp_path):
    mol, _, _ = tabled
    rng = np.random.default_rng(2)
    on, off = _on_resonance(mol, 1, "+", 20, rng), _off_resonance(mol, 1, "+", 20)
    run = _run_dir(tmp_path, "r18")
    _write_eval(run, np.r_[on[:18], off[:2]])
    with pytest.warns(UserWarning):
        assert check_checkpoint_provenance({"meta": {}}, run, mol, 1, "+") == "resonance-check"
    run = _run_dir(tmp_path, "r17")
    _write_eval(run, np.r_[on[:17], off[:3]])
    with pytest.raises(StaleCheckpoint, match="85%"):
        check_checkpoint_provenance({"meta": {}}, run, mol, 1, "+", allow_unprovenanced=True)


def test_no_stamp_no_npz_refuses_then_flag_accepts_unverified(tabled, tmp_path):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    with pytest.raises(StaleCheckpoint, match="allow_unprovenanced") as exc:
        check_checkpoint_provenance({"meta": {}}, run, mol, 0, "+")
    assert run in str(exc.value) and RETRAIN in str(exc.value)
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        assert check_checkpoint_provenance({"meta": {}}, run, mol, 0, "+", allow_unprovenanced=True) == "unverified"
    # an npz without on-resonance frequencies is as good as none
    np.savez_compressed(f"{run}/stratified_eval.npz", on_omegas=np.zeros(3), on_on_resonance=np.zeros(3, bool))
    with pytest.raises(StaleCheckpoint, match="no usable stratified_eval.npz"):
        check_checkpoint_provenance({"meta": {}}, run, mol, 0, "+")


def test_unknown_block_or_sigma_counts_as_nothing_to_check(tabled, tmp_path):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    _write_eval(run, _on_resonance(mol, 0, "+", 100, np.random.default_rng(3)))   # would pass for (0, "+")
    for b, s in ((None, "+"), (0, None), (None, None)):
        with pytest.raises(StaleCheckpoint, match="allow_unprovenanced"):
            check_checkpoint_provenance({"meta": {}}, run, mol, b, s)
        with pytest.warns(UserWarning, match="UNVERIFIED"):
            assert check_checkpoint_provenance({"meta": {}}, run, mol, b, s, allow_unprovenanced=True) == "unverified"
    # a missing-keys npz is "no npz" too
    np.savez_compressed(f"{run}/stratified_eval.npz", omegas=np.zeros(3))
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        assert check_checkpoint_provenance({"meta": {}}, run, mol, 0, "+", allow_unprovenanced=True) == "unverified"


@pytest.mark.parametrize("block", [3, 99, -1])
def test_block_outside_molecule_is_a_failed_check(tabled, tmp_path, block):
    mol, _, _ = tabled                                   # synthetic: blocks 0, 1, 2
    run = _run_dir(tmp_path)
    for npz in (False, True):
        if npz:
            _write_eval(run, _on_resonance(mol, 0, "+", 10, np.random.default_rng(4)))
        with pytest.raises(StaleCheckpoint, match="resonance check failed.*does not exist") as exc:
            check_checkpoint_provenance({"meta": {}}, run, mol, block, "+", allow_unprovenanced=True)
        assert RETRAIN in str(exc.value)


# build_manifest / load_manifest


def _save_ckpt(mol, run, block, stamp=None):
    m = BlockFNO.for_block(mol, block, "+", TINY)
    ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(), DataConfig(), 0, 0.1, 0.2, tables=stamp)
    for k in ("fingerprint", "molecule", "n_nu"):     # a source-package (legacy) checkpoint
        ck["meta"].pop(k)
    ck.pop("source")
    torch.save(ck, f"{run}/best_onres.pt")


def test_build_manifest_checks_every_entry_and_records_provenance(tabled, tmp_path):
    mol, _, _ = tabled
    torch.manual_seed(0)
    stamped, bare = _run_dir(tmp_path, "stamped"), _run_dir(tmp_path, "bare")
    _save_ckpt(mol, stamped, 0, table_stamp(mol))
    _save_ckpt(mol, bare, 1)
    runs = {(0, "+"): stamped, (1, "+"): bare}
    with pytest.raises(StaleCheckpoint, match="1 of 2 checkpoints") as exc:
        build_manifest(mol, "prov", runs, source="thffno")
    assert bare in str(exc.value) and stamped not in str(exc.value)

    with pytest.warns(UserWarning, match="UNVERIFIED"):
        m = build_manifest(mol, "prov", runs, source="thffno", allow_unprovenanced=True)
    assert {k: e.provenance for k, e in m.entries.items()} == {"0,+": "tables-sha256", "1,+": "unverified"}
    work = str(tmp_path / "work")
    m.save(work=work)
    with pytest.warns(UserWarning, match="unverified"):
        eng = load_manifest(mol, "prov", work=work, fallback=FakeExact(mol), device="cpu")
    assert eng.trained == {(0, "+"), (1, "+")}
    assert eng.manifest.entries["0,+"].provenance == "tables-sha256"
    ctx = _no_warnings()          # the stamped entry alone loads silently
    try:
        load_manifest(mol, "prov", work=work, fallback=FakeExact(mol), device="cpu", blocks={(0, "+")})
    finally:
        ctx.__exit__(None, None, None)


@pytest.mark.parametrize("value", [None, "", "trust-me"])
def test_load_manifest_refuses_entry_without_provenance(tabled, tmp_path, value):
    mol, _, _ = tabled
    run = _run_dir(tmp_path)
    _save_ckpt(mol, run, 0, table_stamp(mol))
    work = str(tmp_path / "work")
    path = build_manifest(mol, "old", {(0, "+"): run}, source="thffno").save(work=work)
    j = json.load(open(path))
    if value is None:
        del j["entries"]["0,+"]["provenance"]                # a manifest written before the guard
    else:
        j["entries"]["0,+"]["provenance"] = value
    json.dump(j, open(path, "w"))
    for kw in ({}, {"allow_mismatch": True}):
        with pytest.raises(StaleCheckpoint, match="no recorded provenance") as exc:
            load_manifest(mol, "old", work=work, fallback=FakeExact(mol), device="cpu", **kw)
        assert "0,+" in str(exc.value)
    assert manifest_path(mol.name, "old", work) == path
