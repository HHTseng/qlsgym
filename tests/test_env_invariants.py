"""PurificationEnv invariants (ported from the old gym's test_invariants.py) on the synthetic
molecule with FakeEngine-derived tables.
"""
import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv, Transition
from qlsgym.spec import check_branches

from _fake_tables import FakeEngine, fake_tables, random_tables


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=6, n_tau_slots=3, seed=2))
    eng = FakeEngine(mol, seed=1)
    tables = fake_tables(mol, lib, eng)
    return mol, lib, eng, tables


def _env(setup, batch=8, **cfg):
    mol, lib, _, tables = setup
    return PurificationEnv(mol, lib, tables, EnvConfig(**cfg), batch=batch)


def rand_beliefs(n_beliefs, n, seed=0, concentrated=True):
    rng = np.random.default_rng(seed)
    b = rng.dirichlet(np.full(n, 0.3), size=n_beliefs)
    if concentrated:      # a few nearly pure rows too
        b[0] = 0.0; b[0, 1] = 1.0
        b[1] = np.full(n, 0.005); b[1, 2] = 1.0 - 0.005 * (n - 1)
    return b


def test_reset_is_thermal_and_on_simplex(setup):
    env = _env(setup, batch=3)
    s = env.reset(seed=0)
    assert s.shape == (3, env.n_states) and s.dtype == torch.float64
    assert torch.allclose(s.sum(-1), torch.ones(3, dtype=torch.float64))
    assert torch.all(s[0] == s[1])
    assert env.cfg.p_target == setup[0].task.p_target and env.cfg.max_pulses == setup[0].task.max_pulses


def test_branch_probs_are_a_distribution_and_beliefs_stay_on_simplex(setup):
    env = _env(setup, batch=64)
    b = torch.as_tensor(rand_beliefs(64, env.n_states))
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(env.n_actions, size=64)
        s0, s1, pi0, pi1, r0, r1, d0, d1 = env.branch_outcomes(b, a)
        assert torch.allclose(pi0 + pi1, torch.ones_like(pi0))
        assert torch.all(pi0 >= 0) and torch.all(pi1 >= 0)
        assert torch.all(s0 >= 0) and torch.all(s1 >= 0)
        assert torch.allclose(s0.sum(-1), torch.ones_like(pi0)) and torch.allclose(s1.sum(-1), torch.ones_like(pi1))
        P0, P1 = env.apply(b, a)
        check_branches(P0.numpy(), P1.numpy(), env.n_states, atol=1e-9)


def test_apply_matches_engine_including_primitives(setup):
    mol, lib, eng, _ = setup
    env = _env(setup, batch=lib.n_actions)
    b = torch.as_tensor(rand_beliefs(lib.n_actions, mol.n_states, seed=4))
    acts = np.arange(lib.n_actions)
    P0, P1 = env.apply(b, acts)
    for a in acts:
        act = lib.decode(a)
        p0, p1 = eng.branches_all_tau(b[a].numpy(), act.omega, act.sigma)
        ti = act.tau_index
        assert np.allclose(P0[a].numpy(), p0[ti], atol=1e-12)
        assert np.allclose(P1[a].numpy(), p1[ti], atol=1e-12)
    # the primitive really couples two blocks: population leaves block 0 into block 1's states
    e = np.zeros(mol.n_states); e[mol.primitives[0].source] = 1.0
    P0, P1 = env.apply(torch.as_tensor(e), np.array([lib.n_grid]))
    moved_to = np.where(P1.numpy() > 1e-9)[0]
    assert moved_to.size and set(mol.system.block_of_state[moved_to]) <= set(mol.primitives[0].blocks)


def test_apply_is_batch_independent_and_1d_friendly(setup):
    env = _env(setup, batch=4)
    b = torch.as_tensor(rand_beliefs(4, env.n_states, seed=7))
    a = np.array([1, 5, env.n_actions - 1, 3])
    P0, P1 = env.apply(b, a)
    for k in range(4):
        q0, q1 = env.apply(b[k], np.array([a[k]]))
        assert torch.allclose(P0[k], q0) and torch.allclose(P1[k], q1)


def test_step_belief_is_the_sampled_branch(setup):
    env = _env(setup, batch=32)
    env.reset(seed=3)
    rng = np.random.default_rng(1)
    for _ in range(6):
        a = rng.integers(env.n_actions, size=32)
        tr = env.step(a)
        assert isinstance(tr, Transition)
        pick = torch.where(tr.outcome.bool().unsqueeze(-1), tr.s1, tr.s0)
        assert torch.equal(tr.belief, pick) and torch.equal(env.state, tr.belief)
        assert torch.equal(tr.reward, torch.where(tr.outcome.bool(), tr.r1, tr.r0))
        assert torch.equal(tr.done, torch.where(tr.outcome.bool(), tr.done1, tr.done0))


def test_done_iff_purity_and_truncation_at_budget(setup):
    env = _env(setup, batch=16, p_target=0.9, max_pulses=5)
    env.reset(seed=0)
    for t in range(5):
        tr = env.step(np.full(16, t % env.n_actions))
        assert torch.equal(tr.done, tr.belief.max(-1).values >= 0.9)
        assert torch.equal(tr.done0, tr.s0.max(-1).values >= 0.9)
        if t < 4:
            assert not tr.truncated.any()
    assert torch.equal(tr.truncated, ~tr.done)


def test_purity_threshold_is_inclusive(setup):
    mol, lib, _, tables = setup
    env = PurificationEnv(mol, lib, tables, EnvConfig(p_target=0.5), batch=1)
    b = torch.zeros(1, mol.n_states, dtype=torch.float64); b[0, 0] = 0.5; b[0, 1] = 0.5
    assert bool(env.is_done(b)[0])


def test_seed_reproducibility_and_batch_independence(setup):
    env = _env(setup, batch=8)
    rng = np.random.default_rng(5)
    acts = [rng.integers(env.n_actions, size=8) for _ in range(6)]

    def run(seed):
        env.reset(seed=seed)
        return [env.step(a).belief.clone() for a in acts]

    x, y = run(11), run(11)
    assert all(torch.equal(p, q) for p, q in zip(x, y))
    z = run(12)
    assert any(not torch.equal(p, q) for p, q in zip(x, z))
    # trajectory k with batch 8 == trajectory 0 of a batch-1 env fed the same draws?  Not
    # guaranteed by torch.rand; what *is* guaranteed: each row depends only on its own action.
    env.reset(seed=1)
    tr = env.step(acts[0])
    env2 = _env(setup, batch=8)
    env2.reset(seed=1)
    a2 = acts[0].copy(); a2[3] = (a2[3] + 1) % env.n_actions
    tr2 = env2.step(a2)
    rows = [k for k in range(8) if k != 3]
    assert torch.equal(tr.s0[rows], tr2.s0[rows]) and torch.equal(tr.s1[rows], tr2.s1[rows])
    assert torch.equal(tr.belief[rows], tr2.belief[rows])


def test_reward_is_step_reward_minus_overlap_penalty(setup):
    mol, lib, _, tables = setup
    b = torch.as_tensor(rand_beliefs(16, mol.n_states, seed=2))
    a = np.arange(16) % lib.n_actions
    env = PurificationEnv(mol, lib, tables, EnvConfig(rho=2.0, penalty_mode="indicator"), batch=16)
    s0, s1, *_rest, r0, r1, d0, d1 = env.branch_outcomes(b, a)
    cos0 = (b * s0).sum(-1) / (b.norm(dim=-1) * s0.norm(dim=-1))
    exp0 = -1.0 - 2.0 * (cos0 > 1 - 1 / mol.n_states).to(torch.float64)
    assert torch.allclose(r0, exp0)
    assert set(torch.unique(r0).tolist()) <= {-1.0, -3.0}
    env = PurificationEnv(mol, lib, tables, EnvConfig(rho=2.0, penalty_mode="proportional"), batch=16)
    *_r, r0p, r1p, _, _ = env.branch_outcomes(b, a)
    assert torch.allclose(r0p, -1.0 - 2.0 * cos0.clamp_min(0))
    env0 = PurificationEnv(mol, lib, tables, EnvConfig(rho=0.0), batch=16)
    *_r, r00, r10, _, _ = env0.branch_outcomes(b, a)
    assert torch.all(r00 == -1.0) and torch.all(r10 == -1.0)
    with pytest.raises(ValueError):
        PurificationEnv(mol, lib, tables, EnvConfig(penalty_mode="bogus"))


def test_zero_mass_branch_keeps_belief_and_is_not_done(setup):
    mol, lib, _, tables = setup
    env = PurificationEnv(mol, lib, tables, EnvConfig(p_target=0.5), batch=1)
    # a pure state in a block that the chosen drive does not address at all -> nu>=1 branch may vanish
    for a in range(lib.n_grid):
        e = torch.zeros(1, mol.n_states, dtype=torch.float64); e[0, -1] = 1.0
        s0, s1, pi0, pi1, r0, r1, d0, d1 = env.branch_outcomes(e, np.array([a]))
        if float(pi1[0]) <= env.cfg.min_branch_prob:
            assert torch.equal(s1, e) and not bool(d1[0]) and bool(d0[0])
            break
    else:
        pytest.skip("fake tables never produced an unreachable branch")


def test_out_of_range_actions_and_mismatched_tables(setup):
    mol, lib, _, tables = setup
    env = _env(setup, batch=2)
    with pytest.raises(IndexError):
        env.apply(env.reset(), np.array([0, lib.n_actions]))
    with pytest.raises(IndexError):
        env.apply(env.reset(), np.array([-1, 0]))
    other = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=2, n_tau_slots=2))
    with pytest.raises(RuntimeError):
        PurificationEnv(mol, other, tables)
    bad = random_tables(mol, lib)
    bad.fingerprint = "deadbeef0000"
    with pytest.raises(RuntimeError):
        PurificationEnv(mol, lib, bad)


def test_clone_shares_tables_with_new_batch(setup):
    env = _env(setup, batch=2)
    e2 = env.clone(5)
    assert e2.batch == 5 and e2.state.shape == (5, env.n_states) and e2._T[0] is env._T[0]
    assert env.batch == 2


def test_float32_tables_promote_to_float64_beliefs(setup):
    mol, lib, eng, _ = setup
    tables = fake_tables(mol, lib, eng, dtype=np.float32)
    env = PurificationEnv(mol, lib, tables, batch=4)
    tr = env.step(np.array([0, 1, 2, lib.n_grid]))
    assert tr.belief.dtype == torch.float64 and tr.pi1.dtype == torch.float64
    assert torch.allclose(tr.belief.sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-6)


def test_reset_rows_restores_only_the_selected_trajectories():
    from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
    from _fake_tables import FakeEngine, fake_tables
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=4, n_tau_slots=2, seed=1))
    tables = fake_tables(mol, lib, FakeEngine(mol, seed=2))
    env = PurificationEnv(mol, lib, tables, EnvConfig(), batch=4)
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.torch.zeros(4, dtype=env.torch.long))
    before = env.state.clone()
    mask = env.torch.tensor([True, False, True, False])
    env.reset_rows(mask)
    p0 = env.torch.as_tensor(env.p_init, dtype=env.torch.float64)
    assert env.torch.allclose(env.state[0], p0) and env.torch.allclose(env.state[2], p0)
    assert env.torch.equal(env.state[1], before[1]) and env.torch.equal(env.state[3], before[3])
    assert env.steps.tolist() == [0, 3, 0, 3]
    with pytest.raises(ValueError):
        env.reset_rows(env.torch.tensor([True]))
