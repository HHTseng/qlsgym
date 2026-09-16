"""Monte-Carlo rollout driver, end to end on the synthetic molecule plus ThF+ integration."""
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


# integration: ThF+ physics-designed elimination policy vs the source script

N_ROLLOUTS, SEED, MAX_PULSES = 90, 0, 150


def _source_baseline(n_rollouts: int, seed: int, max_pulses: int):
    """Run ThF_232_19/scripts/baseline_protocol.py --policy physics --engine exact in-process and
    return (success_fraction, mean_pulses, lengths).
    """
    import os
    import sys

    root = os.environ.get("THFFNO_ROOT")
    if not root:
        pytest.skip("THFFNO_ROOT not set (source scripts/env.sh)")
    sys.path.insert(0, os.path.join(root, "scripts"))
    from baseline_protocol import build_actions, choose
    from thffno.planner import ExactTauBatchedEngine, cross_j_primitives, initial_state
    from thffno.system import default_system
    from thffno.units import P_TARGET, tau_grid

    s = default_system()
    taus = tau_grid()
    engine = ExactTauBatchedEngine(s, tau_indices=np.arange(taus.size))
    actions = build_actions(s, taus)
    prims = cross_j_primitives(s, 24, mode="dark_state")
    for q in prims:
        q.tau_index = int(np.argmin(np.abs(taus - q.tau_pi)))
    p_init = initial_state(s, 4.0)

    rng = np.random.default_rng(seed)
    ok, npulse = [], []
    for _ in range(n_rollouts):
        p = p_init.copy()
        last_sigma: dict = {}
        k = 0
        while True:
            if p.max() >= P_TARGET:
                ok.append(True)
                break
            if k >= max_pulses:
                ok.append(False)
                break
            pick = choose(s, engine, p, actions, prims, last_sigma, 1e-4, n_cand=3, tau_rule="pi")
            if pick is None:
                ok.append(False)
                break
            w, sg, ti, _why = pick
            a, c = engine.branches_all_tau(p, float(w), sg)
            p0, p1 = a[ti], c[ti]
            m0, m1 = p0.sum(), p1.sum()
            p = p1 / m1 if rng.random() < m1 / max(m0 + m1, 1e-300) else p0 / m0
            k += 1
        npulse.append(k)
    return float(np.mean(ok)), float(np.mean(npulse)), npulse


@pytest.mark.integration
def test_physics_elimination_policy_on_thf_matches_baseline_script():
    """Acceptance: the gym reproduces the source scripts/baseline_protocol.py --policy physics
    --engine exact *exactly*, at n = 90, seed = 0, max_pulses = 150.
    """
    pytest.importorskip("qlsgym.physics.engines")
    pytest.importorskip("thffno", reason="thffno not on PYTHONPATH: source scripts/env.sh (SKIPPED)")
    from qlsgym.physics.engines import ExactEngine

    mol = load_molecule("thf")
    lib = ActionLibrary.physics_subset(mol)
    engine = ExactEngine(mol)
    policy = PhysicsEliminationPolicy(mol, lib)
    res = rollout(engine, policy, n_rollouts=N_ROLLOUTS, seed=SEED, max_pulses=MAX_PULSES, library=lib)

    want_success, want_pulses, want_lengths = _source_baseline(N_ROLLOUTS, SEED, MAX_PULSES)
    print(f"\nThF+ physics elimination, n={N_ROLLOUTS} seed={SEED}: "
          f"gym success {res.success_fraction:.4f} mean pulses {res.mean_pulses:.2f} | "
          f"source {want_success:.4f} / {want_pulses:.2f}")
    assert res.lengths == want_lengths, "per-rollout pulse counts diverged from the source script"
    assert abs(res.success_fraction - want_success) < 1e-12, (res.success_fraction, want_success)
    assert abs(res.mean_pulses - want_pulses) < 1e-12, (res.mean_pulses, want_pulses)


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
