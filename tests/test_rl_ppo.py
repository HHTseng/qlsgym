"""PPO on the synthetic molecule with fake tables: the loop, both estimators, checkpoints."""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
from qlsgym.policies import Policy, RandomPolicy, rollout
from qlsgym.rl import (ActorPolicy, PPOConfig, load_policy, policy_from_state_dict, save_policy,
                       train_ppo)

from _fake_tables import FakeEngine, fake_tables


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=6, n_tau_slots=3, seed=6))
    tables = fake_tables(mol, lib, FakeEngine(mol, seed=7))
    return mol, lib, tables


def _env(setup, batch=8):
    mol, lib, tables = setup
    return PurificationEnv(mol, lib, tables, EnvConfig(rho=0.0), batch=batch)


@pytest.mark.parametrize("target", ["gae", "qmdp"])
def test_train_runs_and_records(setup, target):
    env = _env(setup)
    cfg = PPOConfig(n_envs=8, n_steps=6, total_steps=8 * 6 * 3, minibatches=2, epochs=2,
                    eval_every=2, eval_rollouts=6, value_target=target, hidden=16, seed=1)
    lines = []
    res = train_ppo(env, cfg, log=lines.append)
    assert res.n_updates == 3 and res.env_steps == 8 * 6 * 3
    assert len(res.history) == 3 and all(np.isfinite(h["loss_pi"]) for h in res.history)
    assert [s["update"] for s in res.snapshots] == [2, 3]     # every 2 updates plus the last
    assert res.best in res.snapshots and 0.0 <= res.best["success"] <= 1.0
    assert lines and lines[0].startswith("[ppo] update 2/3")


def test_config_validation():
    with pytest.raises(ValueError):
        PPOConfig(obs="log")
    with pytest.raises(ValueError):
        PPOConfig(value_target="td")
    assert PPOConfig(n_envs=4, n_steps=4, total_steps=30).n_updates == 2


def test_actor_policy_protocol_and_reproducibility(setup):
    mol, lib, tables = setup
    env = _env(setup)
    cfg = PPOConfig(n_envs=8, n_steps=4, total_steps=32, minibatches=2, epochs=1, eval_every=1,
                    eval_rollouts=4, hidden=16, seed=2)
    res = train_ppo(env, cfg)
    pol = policy_from_state_dict(res.best_state_dict, res.n_in, res.n_actions, cfg, greedy=True)
    assert isinstance(pol, Policy)
    beliefs = np.random.default_rng(0).dirichlet(np.ones(mol.n_states), size=5)
    a1 = pol.act_batch(beliefs, 0, np.random.default_rng(0))
    a2 = pol.act_batch(beliefs, 0, np.random.default_rng(9))
    assert np.array_equal(a1, a2) and a1.shape == (5,) and a1.max() < lib.n_actions
    assert pol.act(beliefs[0], 0, np.random.default_rng(0)) == int(a1[0])
    stoch = policy_from_state_dict(res.best_state_dict, res.n_in, res.n_actions, cfg, greedy=False)
    s1 = stoch.act_batch(beliefs, 0, np.random.default_rng(5))
    s2 = stoch.act_batch(beliefs, 0, np.random.default_rng(5))
    assert np.array_equal(s1, s2)                         # same rng, same draws
    r = rollout(env, pol, n_rollouts=6, seed=0)
    assert r.n_rollouts == 6


def test_checkpoint_round_trip(setup, tmp_path):
    mol, lib, tables = setup
    env = _env(setup)
    cfg = PPOConfig(n_envs=4, n_steps=4, total_steps=16, minibatches=1, epochs=1, eval_every=1,
                    eval_rollouts=2, hidden=16, seed=3)
    res = train_ppo(env, cfg)
    pol = policy_from_state_dict(res.final_state_dict, res.n_in, res.n_actions, cfg)
    path = save_policy(pol.net, cfg, str(tmp_path / "actor.pt"), meta={"library": lib.tag()})
    back = load_policy(path)
    assert back.meta == {"library": lib.tag()} and back.config == cfg
    beliefs = np.random.default_rng(1).dirichlet(np.ones(mol.n_states), size=7)
    assert np.allclose(back.logits(beliefs), pol.logits(beliefs))


def test_eval_env_must_match(setup):
    mol, lib, tables = setup
    env = _env(setup)
    other_lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=3, n_tau_slots=2, seed=1))
    other = PurificationEnv(mol, other_lib, fake_tables(mol, other_lib, FakeEngine(mol, seed=1)), batch=2)
    with pytest.raises(ValueError):
        train_ppo(env, PPOConfig(n_envs=2, n_steps=2, total_steps=4, minibatches=1), env_eval=other)


@pytest.mark.slow
def test_learns_the_easy_fake_dynamics(setup):
    """The fake dynamics are pi-pulse-like, so a policy that pulses the right states purifies far
    faster than a random one.
    """
    env = _env(setup, batch=64)
    cfg = PPOConfig(n_envs=64, n_steps=16, total_steps=64 * 16 * 150, minibatches=4, epochs=4,
                    eval_every=50, eval_rollouts=100, hidden=64, seed=0, lr=3e-3)
    res = train_ppo(env, cfg)
    pol = policy_from_state_dict(res.best_state_dict, res.n_in, res.n_actions, cfg)
    learned = rollout(env, pol, n_rollouts=300, seed=11)
    rnd = rollout(env, RandomPolicy(env.n_actions), n_rollouts=300, seed=11)
    # measured 2026-09-16: learned 0.95 / 10.0 pulses, random 0.26 / 18.6
    assert learned.success_fraction > rnd.success_fraction + 0.3
    assert learned.mean_pulses < rnd.mean_pulses - 3.0
