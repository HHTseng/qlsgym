"""Gymnasium wrapper: standard reset/step API, spaces, both branches in info. Skipped entirely if
gymnasium is not installed (env/gym.py degrades gracefully).
"""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid
from qlsgym.env.gym import HAS_GYMNASIUM, QLSGymEnv

from _fake_tables import FakeEngine, fake_tables

pytestmark = pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium is not installed")


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=4, n_tau_slots=2, seed=1))
    tables = fake_tables(mol, lib, FakeEngine(mol, seed=2))
    return mol, lib, tables


def test_spaces_and_reset(setup):
    import gymnasium as gym

    mol, lib, tables = setup
    env = QLSGymEnv(mol, lib, tables, seed=0)
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (mol.n_states,)
    assert env.observation_space.low.min() == 0.0 and env.observation_space.high.max() == 1.0
    assert env.action_space.n == lib.n_actions
    obs, info = env.reset(seed=0)
    assert obs.shape == (mol.n_states,) and obs.dtype == np.float64
    assert np.isclose(obs.sum(), 1.0)
    assert info["pulses"] == 0


def test_step_returns_gymnasium_tuple_and_both_branches(setup):
    mol, lib, tables = setup
    env = QLSGymEnv(mol, lib, tables, seed=3)
    env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (mol.n_states,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    for k in ("s0", "s1", "pi0", "pi1", "done0", "done1", "outcome", "action"):
        assert k in info
    assert np.isclose(info["pi0"] + info["pi1"], 1.0)
    assert np.isclose(info["s0"].sum(), 1.0) and np.isclose(info["s1"].sum(), 1.0)
    with pytest.raises(ValueError):
        env.step(lib.n_actions)


def test_reset_seed_reproducible(setup):
    mol, lib, tables = setup
    env = QLSGymEnv(mol, lib, tables)
    env.reset(seed=7)
    traj1 = [env.step(a % lib.n_actions)[0].copy() for a in range(5)]
    env.reset(seed=7)
    traj2 = [env.step(a % lib.n_actions)[0].copy() for a in range(5)]
    for a, b in zip(traj1, traj2):
        assert np.array_equal(a, b)


def test_missing_gymnasium_raises_import_error(monkeypatch):
    import qlsgym.env.gym as gymmod

    monkeypatch.setattr(gymmod, "HAS_GYMNASIUM", False)
    with pytest.raises(ImportError):
        gymmod.QLSGymEnv(None, None, None)
