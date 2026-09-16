"""FnoEngine on the synthetic molecule with randomly initialised tiny models and a fake fallback:
protocol conformance, call counters, batched == looped.
"""
import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.spec import BatchedEngine, TauBatchedEngine, check_branches
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.fno_engine import FnoEngine
from qlsgym.surrogate.train import TrainConfig, checkpoint_dict

TINY = FNOConfig(n_modes=4, hidden_channels=8, n_layers=1, lifting_channel_ratio=2, projection_channel_ratio=2)


class FakeExact:
    """Stands in for physics.engines.ExactEngine: p(tau) = p(0) in the nu=0 branch, except that a
    primitive moves its source's population to nu>=1.
    """

    def __init__(self, molecule, tau_indices=None):
        self.molecule = molecule
        self.tau_indices = np.arange(molecule.window.n_tau) if tau_indices is None else np.asarray(tau_indices)
        self.n_calls = 0

    def branches_all_tau(self, p_in, omega, sigma):
        self.n_calls += 1
        nt = self.tau_indices.size
        p0 = np.repeat(np.asarray(p_in, dtype=np.float64)[None], nt, 0)
        p1 = np.zeros_like(p0)
        for prim in self.molecule.primitives:
            if np.isclose(omega, prim.omega) and prim.sigma == sigma and prim.source >= 0:
                p1[:, prim.source] = p0[:, prim.source]
                p0[:, prim.source] = 0.0
        return p0, p1


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic")


@pytest.fixture(scope="module")
def ckpts(mol, tmp_path_factory):
    d = tmp_path_factory.mktemp("ck")
    out = {}
    torch.manual_seed(0)
    for b in mol.blocks:
        for sigma in ("+", "-"):
            if sigma == "-" and b.index != 1:
                continue                       # one sigma- surrogate only
            m = BlockFNO.for_block(mol, b.index, sigma, TINY)
            p = d / f"b{b.index}{'p' if sigma == '+' else 'm'}.pt"
            torch.save(checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(n_freq=1, n_init=1),
                                       DataConfig(n_freq=1, n_init=1), 0, 0.0, 0.0), p)
            out[(b.index, sigma)] = str(p)
    return out


def thermal(mol):
    e = mol.system.energies - mol.system.energies.min()
    w = np.exp(-e / e.max())
    return w / w.sum()


def test_protocol_and_counters(mol, ckpts):
    fb = FakeExact(mol)
    eng = FnoEngine(mol, ckpts, fallback=fb, device="cpu")
    assert isinstance(eng, TauBatchedEngine) and isinstance(eng, BatchedEngine)
    assert eng.trained == set(ckpts)
    p = thermal(mol)
    nt, n = mol.window.n_tau, mol.n_states
    w_in = 0.5 * (mol.window.omega_min + mol.window.omega_max)
    # sigma+ in-window: every block covered -> no fallback call
    p0, p1 = eng.branches_all_tau(p, w_in, "+")
    assert p0.shape == p1.shape == (nt, n)
    check_branches(p0, p1, n)
    assert eng.calls == {"fno": 3, "exact_sigma_minus": 0, "exact_untrained": 0, "exact_primitive": 0}
    assert fb.n_calls == 0
    # per-block mass is conserved by the surrogate path
    for b in mol.blocks:
        np.testing.assert_allclose(p0[:, b.states].sum(1) + p1[:, b.states].sum(1), p[b.states].sum(), atol=1e-6)
    # sigma- in-window: block 1 surrogate, blocks 0 and 2 through the fallback (one call, masked state)
    p0, p1 = eng.branches_all_tau(p, w_in, "-")
    check_branches(p0, p1, n)
    assert eng.calls["fno"] == 4 and eng.calls["exact_sigma_minus"] == 2 and fb.n_calls == 1
    np.testing.assert_allclose(p0[:, mol.blocks[0].states],
                               np.broadcast_to(p[mol.blocks[0].states], p0[:, mol.blocks[0].states].shape))  # fake: identity
    # primitive (off-window): everything exact through the fallback
    prim = mol.primitives[0]
    p0, p1 = eng.branches_all_tau(p, prim.omega, prim.sigma)
    check_branches(p0, p1, n)
    assert eng.calls["exact_primitive"] == 1 and fb.n_calls == 2
    assert p0[:, prim.source].max() == 0.0 and p1[:, prim.source].min() > 0
    assert 0 < eng.surrogate_fraction() < 1
    # zero-mass blocks are skipped (no surrogate call for them)
    q = np.zeros(n); q[mol.blocks[2].states] = p[mol.blocks[2].states] / p[mol.blocks[2].states].sum()
    before = eng.calls["fno"]
    eng.branches_all_tau(q, w_in, "+")
    assert eng.calls["fno"] == before + 1


def test_batch_equals_stacked_loop(mol, ckpts):
    eng = FnoEngine(mol, ckpts, fallback=FakeExact(mol), device="cpu")
    p = thermal(mol)
    rng = np.random.default_rng(0)
    ws = np.concatenate([rng.uniform(mol.window.omega_min, mol.window.omega_max, 4), [mol.primitives[0].omega]])
    for sigma in ("+", "-"):
        a, c = eng.branches_batch(p, ws, sigma)
        assert a.shape == (ws.size, mol.window.n_tau, mol.n_states)
        check_branches(a, c, mol.n_states)
        loop = [eng.branches_all_tau(p, w, sigma) for w in ws]
        np.testing.assert_allclose(a, np.stack([o[0] for o in loop]), atol=1e-7)
        np.testing.assert_allclose(c, np.stack([o[1] for o in loop]), atol=1e-7)
    assert eng.branches_batch(p, np.zeros(0), "+")[0].shape == (0, mol.window.n_tau, mol.n_states)


def test_partial_coverage_and_tau_subset(mol, ckpts):
    ti = np.array([0, 5, 15])
    fb = FakeExact(mol, ti)
    only0 = {k: v for k, v in ckpts.items() if k == (0, "+")}
    eng = FnoEngine(mol, only0, tau_indices=ti, fallback=fb, device="cpu")
    p = thermal(mol)
    w_in = mol.window.omega_min + 1.0
    p0, p1 = eng.branches_all_tau(p, w_in, "+")
    assert p0.shape == (3, mol.n_states)
    check_branches(p0, p1, mol.n_states)
    assert eng.calls["fno"] == 1 and eng.calls["exact_untrained"] == 2 and fb.n_calls == 1
    full = FnoEngine(mol, only0, fallback=FakeExact(mol), device="cpu")
    f0, f1 = full.branches_all_tau(p, w_in, "+")
    np.testing.assert_allclose(p0, f0[ti], atol=1e-7)
    np.testing.assert_allclose(p1, f1[ti], atol=1e-7)
    # branches(): one duration row
    a, c = eng.branches(p, w_in, "+", 5)
    np.testing.assert_allclose(a, p0[1]); np.testing.assert_allclose(c, p1[1])
    with pytest.raises(ValueError):
        eng.branches(p, w_in, "+", 7)
    # mismatched fallback grid is refused; string keys are accepted
    with pytest.raises(ValueError):
        FnoEngine(mol, only0, tau_indices=ti, fallback=FakeExact(mol), device="cpu")
    eng2 = FnoEngine(mol, {"0,+": only0[(0, "+")]}, fallback=FakeExact(mol), device="cpu")
    assert eng2.trained == {(0, "+")}
    with pytest.raises(ValueError):
        FnoEngine(mol, {(1, "+"): only0[(0, "+")]}, fallback=FakeExact(mol), device="cpu")
