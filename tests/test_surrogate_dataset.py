"""Initial-state and frequency sampling (pure numpy); split generation needs agent A's
physics.torch_ops and is skipped without it.
"""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.spec import Window
from qlsgym.surrogate.dataset import (DataConfig, activity_weights, mask_carriers, off_resonant_frequency_sample,
                                      pair_indices, random_mixed_populations, resonant_frequency_sample,
                                      sample_frequencies, sample_training_frequencies)
from qlsgym.surrogate.metrics import is_on_resonance, static_baseline, stratified_summary


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic")


def test_random_mixed_populations():
    rng = np.random.default_rng(0)
    p = random_mixed_populations(26, 500, rng, 1.0)
    assert p.shape == (500, 26) and np.all(p >= 0)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert np.median(p.max(1)) < 0.25                     # uniform simplex: spread states
    q = random_mixed_populations(26, 500, np.random.default_rng(0), -2.0)
    np.testing.assert_allclose(q.sum(1), 1.0)
    assert np.percentile(q.max(1), 90) > 0.85              # log-uniform mixture reaches concentrated states
    assert np.median(q.max(1)) > np.median(p.max(1))


def test_frequency_sampling(mol):
    rng = np.random.default_rng(0)
    w = sample_frequencies(mol, 200, rng)
    assert np.all(np.diff(w) >= 0) and w.min() >= mol.window.omega_min and w.max() <= mol.window.omega_max
    on = resonant_frequency_sample(mol, 0, 50, rng, "+", n_linewidths=1.0)
    assert on.size == 50 and is_on_resonance(mol, on, 0, "+", 1.0).all()
    off = off_resonant_frequency_sample(mol, 0, 50, rng, "+", n_linewidths=1.0)
    assert off.size == 50 and not is_on_resonance(mol, off, 0, "+", 1.0).any()
    cfg = DataConfig(n_freq=40, freq_sampling="mixture", resonant_frac=0.5, resonant_spread=1.0)
    m = sample_training_frequencies(mol, 0, cfg, np.random.default_rng(1), "+")
    assert m.size == 40 and is_on_resonance(mol, m, 0, "+", 1.0).sum() >= 20
    assert cfg.key() == DataConfig(n_freq=40, freq_sampling="mixture", resonant_frac=0.5, resonant_spread=1.0).key()
    assert cfg.key() != DataConfig(n_freq=41, freq_sampling="mixture").key()


def test_carrier_mask(mol):
    import dataclasses
    lo, hi = mol.trap.nu_f - 1.0, mol.trap.nu_f + 1.0
    w = dataclasses.replace(mol.window, carrier_bands=((lo, hi),))
    m2 = dataclasses.replace(mol, window=w)
    assert not mask_carriers(m2, np.array([mol.trap.nu_f]))[0]
    x = sample_frequencies(m2, 2000, np.random.default_rng(0))
    assert mask_carriers(m2, x).all()


def test_pairing_and_weights():
    kf, kl = pair_indices(DataConfig(n_freq=3, n_init=2, pairing="product"), np.random.default_rng(0))
    assert kf.size == 6 and set(zip(kf, kl)) == {(i, j) for i in range(3) for j in range(2)}
    kf, kl = pair_indices(DataConfig(n_freq=4, n_init=4, pairing="zip"), np.random.default_rng(0))
    np.testing.assert_array_equal(kf, kl)
    kf, _ = pair_indices(DataConfig(n_freq=4, n_init=9, n_pairs=7), np.random.default_rng(0))
    assert kf.size == 7
    with pytest.raises(ValueError):
        pair_indices(DataConfig(n_freq=3, n_init=2, pairing="zip"), np.random.default_rng(0))
    import torch
    p = torch.zeros(2, 5, 4)
    p[0, :, 0] = torch.linspace(0, 1, 5); p[0, :, 1] = 1 - p[0, :, 0]
    p[1, :, 2] = 1.0
    w = activity_weights(p, lam=1.0)
    torch.testing.assert_close(w[0], torch.tensor([2.0, 2.0, 1.0, 1.0]))
    torch.testing.assert_close(w[1], torch.ones(4))
    assert float(static_baseline(p)[1].max()) == 0.0
    s = stratified_summary(np.array([1e-3, 2e-3, 3e-3]), np.array([1e-2, 1e-3, 3e-3]), np.array([True, False, False]), "u_")
    assert s["u_on_n"] == 1 and s["u_off_n"] == 2 and s["u_all_ratio_to_static_median"] == pytest.approx(1.0)


def test_build_split_if_physics_present(mol, tmp_path):
    pytest.importorskip("qlsgym.physics.torch_ops", reason="agent A's physics.torch_ops not present")
    from qlsgym.surrogate.dataset import build_or_load_split, split_path
    cfg = DataConfig(n_freq=3, n_init=2, n_pairs=4, seed=0)
    sp = build_or_load_split(mol, 0, cfg, "+", device="cpu", work=str(tmp_path))
    e_n = mol.blocks[0].n_states
    assert sp.x.shape == (4, e_n + mol.blocks[0].omega.size + 1, mol.window.n_tau)
    assert sp.p_true.shape == (4, mol.window.n_tau, 2 * e_n) and sp.weights.shape == (4, 2 * e_n)
    np.testing.assert_allclose(sp.p_true.sum(-1).numpy(), 1.0, atol=1e-5)
    assert split_path(mol, 0, "+", cfg, str(tmp_path)).startswith(str(tmp_path)) and mol.fingerprint() in split_path(mol, 0, "+", cfg, str(tmp_path))
    sp2 = build_or_load_split(mol, 0, cfg, "+", device="cpu", work=str(tmp_path))   # from disk
    np.testing.assert_allclose(sp2.x.numpy(), sp.x.numpy())
