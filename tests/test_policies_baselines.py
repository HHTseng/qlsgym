"""Reference policies on the synthetic molecule with fake tables/engine; PhysicsEliminationPolicy's
molecule guard; ScorePlannerPolicy closed-loop re-planning against FakeEngine.
"""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid
from qlsgym.policies import (DescendingPopulationPolicy, PhysicsEliminationPolicy, RandomPolicy,
                             ScorePlannerPolicy, SweepingPolicy)
from qlsgym.policies.score import ScoreConfig

from _fake_tables import FakeEngine, fake_tables


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2, seed=4))
    eng = FakeEngine(mol, seed=5)
    tables = fake_tables(mol, lib, eng)
    return mol, lib, eng, tables


def test_sweeping_cycles_actions(setup):
    _, lib, _, _ = setup
    pol = SweepingPolicy(lib.n_actions)
    rng = np.random.default_rng(0)
    belief = np.full(4, 0.25)
    picks = [pol.act(belief, t, rng) for t in range(2 * lib.n_actions)]
    assert picks == [t % lib.n_actions for t in range(2 * lib.n_actions)]
    assert np.array_equal(pol.act_batch(np.zeros((3, 4)), 1, rng), np.full(3, 1))


def test_random_policy_is_uniform_over_library(setup):
    _, lib, _, _ = setup
    pol = RandomPolicy(lib.n_actions)
    rng = np.random.default_rng(0)
    draws = [pol.act(None, 0, rng) for _ in range(500)]
    assert min(draws) >= 0 and max(draws) < lib.n_actions
    assert len(set(draws)) > 1


def test_descending_population_addresses_most_populated_state(setup):
    mol, lib, _, tables = setup
    pol = DescendingPopulationPolicy(lib, tables)
    rng = np.random.default_rng(0)
    belief = np.zeros(mol.n_states)
    belief[3] = 1.0
    a = pol.act(belief, 0, rng)
    assert a == pol.best_action[3]
    beliefs = np.zeros((2, mol.n_states)); beliefs[0, 3] = 1.0; beliefs[1, 0] = 1.0
    acts = pol.act_batch(beliefs, 0, rng)
    assert acts[0] == pol.best_action[3] and acts[1] == pol.best_action[0]


def test_physics_elimination_policy_requires_thf_molecule(setup):
    mol, lib, _, _ = setup
    assert mol.name != "thf"
    with pytest.raises(ValueError, match="ThF"):
        PhysicsEliminationPolicy(mol, lib)


def test_score_planner_policy_closed_loop_on_synthetic(setup):
    mol, lib, eng, _ = setup
    pol = ScorePlannerPolicy(eng, lib, ScoreConfig().for_molecule(mol), tau_mode="library")
    rng = np.random.default_rng(0)
    from qlsgym.env.env import boltzmann_belief
    belief = boltzmann_belief(mol)
    a = pol.act(belief, 0, rng)
    assert a is not None and 0 <= int(a) < lib.n_actions
    assert pol.calls > 0


def test_score_planner_policy_free_tau_returns_action(setup):
    mol, lib, eng, _ = setup
    pol = ScorePlannerPolicy(eng, lib, ScoreConfig().for_molecule(mol), tau_mode="free")
    rng = np.random.default_rng(0)
    from qlsgym.env.env import boltzmann_belief
    belief = boltzmann_belief(mol)
    pick = pol.act(belief, 0, rng)
    assert pick is not None


def test_score_planner_pool_sampling(setup):
    """n_pool=None is the greedy first maximum; a pool draws uniformly from the top scores (plus
    everything within delta_s of the best) and nothing else.
    """
    mol, lib, eng, _ = setup
    from qlsgym.env.env import boltzmann_belief
    belief = boltzmann_belief(mol)
    greedy = ScorePlannerPolicy(eng, lib, ScoreConfig().for_molecule(mol))
    a_greedy = greedy.act(belief, 0, np.random.default_rng(0))
    one = ScorePlannerPolicy(eng, lib, ScoreConfig().for_molecule(mol), n_pool=1, delta_s=0.0)
    assert one.act(belief, 0, np.random.default_rng(3)) == a_greedy
    with pytest.raises(ValueError):
        ScorePlannerPolicy(eng, lib, n_pool=0)
    pooled = ScorePlannerPolicy(eng, lib, ScoreConfig().for_molecule(mol), n_pool=3, delta_s=0.0)
    rng = np.random.default_rng(1)
    draws = {pooled.act(belief, 0, rng) for _ in range(80)}
    assert a_greedy in draws and 1 < len(draws) <= 3
    # exactly the three best-scoring drives are reachable
    cands = [(v, k) for k, v in enumerate([3.0, 1.0, 2.5, 2.9, -1.0])]
    picks = {pooled._choose(cands, np.random.default_rng(s)) for s in range(40)}
    assert picks == {0, 2, 3}
    wide = ScorePlannerPolicy(eng, lib, n_pool=1, delta_s=0.15)   # 2.9 is within 0.15 of 3.0
    picks = {wide._choose(cands, np.random.default_rng(s)) for s in range(40)}
    assert picks == {0, 3}
