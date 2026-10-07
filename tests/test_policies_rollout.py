"""Monte-Carlo rollout driver, end to end on the synthetic molecule."""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
from qlsgym.policies import PhysicsEliminationPolicy, RandomPolicy, SweepingPolicy, rollout

from _fake_tables import FakeEngine, fake_tables


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=6, n_tau_slots=3, seed=6))
    eng = FakeEngine(mol, seed=7)
    tables = fake_tables(mol, lib, eng)
    return mol, lib, eng, tables


def test_rollout_engine_path_no_cache(setup):
    mol, lib, eng, _ = setup
    policy = SweepingPolicy(lib.n_actions)
    res = rollout(eng, policy, n_rollouts=40, seed=0, max_pulses=mol.task.max_pulses, library=lib)
    assert res.n_rollouts == 40
    assert 0.0 <= res.success_fraction <= 1.0
    assert res.engine_calls > 0
    assert len(res.lengths) == 40 and len(res.successes) == 40
    assert res.engine == "FakeEngine"


def test_rollout_env_path_matches_engine_path_statistically(setup):
    mol, lib, eng, tables = setup
    env = PurificationEnv(mol, lib, tables, EnvConfig(), batch=1)
    policy_env = SweepingPolicy(lib.n_actions)
    res_env = rollout(env, policy_env, n_rollouts=60, seed=1)
    policy_eng = SweepingPolicy(lib.n_actions)
    res_eng = rollout(eng, policy_eng, n_rollouts=60, seed=1, library=lib, max_pulses=mol.task.max_pulses)
    # same tables (fake_tables built from the same engine), same seed and
    # policy: the two paths should land close in success rate and pulse count.
    assert abs(res_env.success_fraction - res_eng.success_fraction) < 0.25
    assert abs(res_env.mean_pulses - res_eng.mean_pulses) < 5.0


def test_rollout_engine_requires_library():
    mol = load_molecule("synthetic")
    eng = FakeEngine(mol)
    with pytest.raises(ValueError):
        rollout(eng, RandomPolicy(5), n_rollouts=1)


def test_rollout_records_trajectories_when_asked(setup):
    mol, lib, eng, _ = setup
    policy = RandomPolicy(lib.n_actions)
    res = rollout(eng, policy, n_rollouts=3, seed=0, max_pulses=mol.task.max_pulses,
                  library=lib, record=True)
    assert res.trajectories is not None and len(res.trajectories) == 3
    for traj, length in zip(res.trajectories, res.lengths):
        assert len(traj) == length


def test_target_states_is_latched_at_the_step_the_episode_succeeded(setup):
    """Finished rows keep receiving pulses, so the final argmax can differ from the winner."""
    import collections

    import torch

    mol, lib, eng, tables = setup
    env = PurificationEnv(mol, lib, tables, EnvConfig(), batch=1)
    res = rollout(env, RandomPolicy(lib.n_actions), n_rollouts=48, seed=3, record=True)
    assert any(res.successes), "the fixture must produce at least one success to test"

    replay = collections.Counter()
    for succeeded, moves in zip(res.successes, res.trajectories):
        if not succeeded:
            continue
        p = torch.as_tensor(env.p_init, dtype=torch.float64).reshape(1, -1)
        for action, outcome in moves:
            s0, s1, *_ = env.branch_outcomes(p, [int(action)])
            p = s1 if int(outcome) == 1 else s0
        replay[int(torch.argmax(p[0]))] += 1
    assert sum(replay.values()) == sum(res.successes)
    got = collections.Counter({int(n) if not isinstance(n, list) else tuple(n): c
                               for n, c in res.target_states})
    # target_states stores level labels; compare by count, keyed on the label
    labels = mol.system.levels
    want = collections.Counter()
    for n, c in replay.items():
        want[tuple(labels[n]) if n < len(labels) else n] += c
    assert dict(got) == dict(want.most_common(8))
