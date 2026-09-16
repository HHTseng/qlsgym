"""qlsgym.surrogate.fno_env.FnoEnv: the surrogate replaces exactly the blocks it covers and nothing
else, batched rows each get their own belief and drive, and the guards fire.
"""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
from qlsgym.surrogate.fno_env import FnoEnv

from _fake_tables import FakeEngine, fake_tables


class StubFno:
    """Minimal FnoEngine stand-in: answers covered blocks by moving all of the block's population
    into the nu >= 1 branch of its first state.
    """

    def __init__(self, molecule, covered, tau_indices):
        self.molecule = molecule
        self.trained = set(covered)
        self.tau_indices = np.asarray(tau_indices, dtype=np.int64)
        self.calls = {"fno": 0}
        self.seen = []

    def surrogate_fraction(self):
        return 1.0

    def block_branches(self, block_index, sigma, sub, omegas):
        sub = np.atleast_2d(np.asarray(sub, dtype=np.float64))
        omegas = np.asarray(omegas, dtype=np.float64).ravel()
        self.calls["fno"] += omegas.size
        self.seen.append((block_index, sigma, sub.copy(), omegas.copy()))
        n, m = sub.shape[0], sub.shape[1]
        out = np.zeros((n, self.tau_indices.size, 2 * m))
        out[:, :, m] = sub.sum(1)[:, None]          # all mass -> nu >= 1, state 0
        return out


@pytest.fixture(scope="module")
def setup():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2, seed=4))
    tables = fake_tables(mol, lib, FakeEngine(mol, seed=5))
    return mol, lib, tables


def _taus(lib):
    prim = [lib.primitive_tau_index(k) for k in range(lib.n_primitives)]
    return np.unique(np.concatenate([lib.tau_indices, np.asarray(prim, dtype=np.int64)]))


def test_only_covered_blocks_are_replaced(setup):
    mol, lib, tables = setup
    covered = [(0, "+")]
    stub = StubFno(mol, covered, _taus(lib))
    exact = PurificationEnv(mol, lib, tables, EnvConfig(), batch=1)
    env = FnoEnv(mol, lib, tables, stub, EnvConfig(), batch=1)
    p = np.asarray(exact.p_init)
    a = next(i for i in range(lib.n_grid) if lib.decode(i).sigma == "+")
    e0, e1 = exact.apply(p, [a])
    f0, f1 = env.apply(p, [a])
    states = mol.blocks[0].states
    other = np.setdiff1d(np.arange(mol.n_states), states)
    assert np.allclose(e0[other].numpy(), f0[other].numpy())          # untouched
    assert np.allclose(e1[other].numpy(), f1[other].numpy())
    assert f0[states].numpy().sum() == pytest.approx(0.0)             # the stub's answer
    assert f1[states[0]].item() == pytest.approx(p[states].sum())
    # a sigma- action is not covered, so nothing changes
    am = next(i for i in range(lib.n_grid) if lib.decode(i).sigma == "-")
    assert np.allclose(exact.apply(p, [am])[0].numpy(), env.apply(p, [am])[0].numpy())


def test_rows_get_their_own_belief_and_drive(setup):
    mol, lib, tables = setup
    stub = StubFno(mol, [(0, "+")], _taus(lib))
    env = FnoEnv(mol, lib, tables, stub, EnvConfig(), batch=3)
    plus = [i for i in range(lib.n_grid) if lib.decode(i).sigma == "+"][:3]
    rng = np.random.default_rng(0)
    beliefs = rng.dirichlet(np.ones(mol.n_states), size=3)
    env.apply(beliefs, plus)
    b_index, sigma, sub, omegas = stub.seen[-1]
    assert b_index == 0 and sigma == "+" and sub.shape == (3, mol.blocks[0].states.size)
    assert np.allclose(sub, beliefs[:, mol.blocks[0].states])
    assert np.allclose(omegas, [lib.decode(a).omega for a in plus])


def test_primitives_stay_exact(setup):
    mol, lib, tables = setup
    if lib.n_primitives == 0:
        pytest.skip("this molecule has no primitives")
    stub = StubFno(mol, [(0, "+")], _taus(lib))
    exact = PurificationEnv(mol, lib, tables, EnvConfig(), batch=1)
    env = FnoEnv(mol, lib, tables, stub, EnvConfig(), batch=1)
    p = np.asarray(exact.p_init)
    a = lib.n_grid
    assert np.allclose(exact.apply(p, [a])[0].numpy(), env.apply(p, [a])[0].numpy())
    assert stub.calls["fno"] == 0


def test_guards(setup):
    mol, lib, tables = setup
    with pytest.raises(ValueError):                    # covers nothing
        FnoEnv(mol, lib, tables, StubFno(mol, [], _taus(lib)))
    with pytest.raises(ValueError):                    # engine lacks a duration
        FnoEnv(mol, lib, tables, StubFno(mol, [(0, "+")], _taus(lib)[:1]))


def test_steps_and_rollout_run(setup):
    from qlsgym.policies import RandomPolicy, rollout
    mol, lib, tables = setup
    stub = StubFno(mol, [(0, "+")], _taus(lib))
    env = FnoEnv(mol, lib, tables, stub, EnvConfig(), batch=4)
    env.reset(seed=0)
    tr = env.step(env.torch.zeros(4, dtype=env.torch.long))
    assert tr.belief.shape == (4, mol.n_states)
    assert env.substituted_fraction() == 1.0
    res = rollout(env, RandomPolicy(lib.n_actions), n_rollouts=8, seed=1)
    assert res.n_rollouts == 8
