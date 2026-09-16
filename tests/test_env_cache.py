"""Table cache provenance: exact arms must never load surrogate-probed tables."""
import json
import os

import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid
from qlsgym.env.cache import build_action_tables, load_action_tables, tables_dir

from _fake_tables import FakeEngine


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=4, n_tau_slots=2, seed=2))
    return mol, lib


def test_cache_refuses_tables_built_by_a_different_builder(setup, tmp_path):
    mol, lib = setup
    eng = FakeEngine(mol, tau_indices=None, seed=11)
    t = build_action_tables(mol, lib, out_dir=str(tmp_path), engine=eng)
    assert t.manifest["builder"] == "engine:FakeEngine"
    d = tables_dir(mol, lib, str(tmp_path))
    with open(os.path.join(d, "manifest.json")) as fh:
        assert json.load(fh)["builder"] == "engine:FakeEngine"
    # the cache path is keyed on (molecule, fingerprint, library tag) only, so without
    # the builder check the next "exact" caller would silently reuse these tables
    with pytest.raises(RuntimeError, match="builder|built by"):
        build_action_tables(mol, lib, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="built by"):
        load_action_tables(mol, lib, out_dir=str(tmp_path), builder_prefix="physics:")
    # asking for what is actually there still works
    assert load_action_tables(mol, lib, out_dir=str(tmp_path), builder_prefix="engine:") is not None


def test_physics_tables_round_trip_through_the_builder_check(setup, tmp_path):
    mol, lib = setup
    t = build_action_tables(mol, lib, out_dir=str(tmp_path))
    assert t.manifest["builder"].startswith("physics:")
    again = build_action_tables(mol, lib, out_dir=str(tmp_path))       # cache hit, no error
    assert again.manifest["builder"] == t.manifest["builder"]
    with pytest.raises(RuntimeError, match="built by"):
        load_action_tables(mol, lib, out_dir=str(tmp_path), builder_prefix="engine:FakeEngine")
