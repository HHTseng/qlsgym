"""Embedding shape / normalisation on the synthetic molecule."""
import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.embedding import (Embedding, TorchEmbedding, block_shapes, resonant_frequencies,
                                        retained_transitions)


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic")


def test_geometry(mol):
    for b in mol.blocks:
        for sigma in ("+", "-"):
            e = Embedding(mol, b.index, sigma)
            # synthetic couplings are 2pi..6pi rad/ms >= omega_min_coupling and all resonances in-window
            assert e.n_transitions == b.omega.size
            assert e.n_channels == b.n_states + b.omega.size + 1
            assert block_shapes(mol, b.index, sigma) == (b.n_states, b.omega.size, e.n_channels)
            assert e.taus.shape == (mol.window.n_tau,)
            w = resonant_frequencies(mol, b.index, sigma)
            d = mol.system.energies[b.states[b.f_local]] - mol.system.energies[b.states[b.i_local]]
            np.testing.assert_allclose(w, mol.trap.nu_f + (d if sigma == "+" else -d))
            assert e.fingerprint == mol.fingerprint()


def test_retention_cutoffs(mol):
    b = mol.blocks[0]
    assert retained_transitions(mol, 0, "+", delta_max=1e9).size == b.omega.size
    # a cutoff of zero keeps only resonances inside the window (all of them here)
    assert retained_transitions(mol, 0, "+", delta_max=0.0).size == b.omega.size


def test_build_shape_and_channels(mol):
    e = Embedding(mol, 0, "+")
    rng = np.random.default_rng(0)
    p0 = rng.dirichlet(np.ones(e.n_states), size=5)
    omega = rng.uniform(mol.window.omega_min, mol.window.omega_max, 5)
    x = e.build(p0, omega)
    assert x.shape == (5, e.n_channels, mol.window.n_tau)
    # state channels are the initial populations broadcast in time
    np.testing.assert_allclose(x[:, : e.n_states, :], np.repeat(p0[:, :, None], mol.window.n_tau, 2))
    # control channels are bounded by s_emb
    assert np.abs(x[:, e.n_states:, :]).max() <= mol.window.s_emb + 1e-12
    # Eq. 15 at omega_min: w~ = -1
    xw = e.build(p0[:1], mol.window.omega_min)[0, -1]
    np.testing.assert_allclose(xw, mol.window.s_emb * np.sin(-2 * np.pi * e.taus / mol.window.tau_max_ms), atol=1e-12)
    # Eq. 13 for one transition
    delta = e.w_res[0] - omega[0]
    np.testing.assert_allclose(x[0, e.n_states], mol.window.s_emb * np.sin(delta * e.taus) / (1 + mol.window.beta_emb * abs(delta)), atol=1e-12)
    # scalar omega broadcasts across the batch
    assert e.build(p0, omega[0]).shape == x.shape
    with pytest.raises(ValueError):
        e.build(p0, omega[:2])


def test_torch_twin_matches_numpy(mol):
    e = Embedding(mol, 1, "-")
    te = TorchEmbedding(mol, 1, "-")
    te2 = TorchEmbedding.from_numpy(e)
    rng = np.random.default_rng(1)
    p0 = rng.dirichlet(np.ones(e.n_states), size=4)
    omega = rng.uniform(mol.window.omega_min, mol.window.omega_max, 4)
    x = e.build(p0, omega)
    for t in (te, te2):
        xt = t.build(torch.as_tensor(p0), torch.as_tensor(omega), out_dtype=torch.float64).numpy()
        np.testing.assert_allclose(xt, x, atol=1e-12)
        assert t.n_channels == e.n_channels
    x32 = te.build(torch.as_tensor(p0), torch.as_tensor(omega)).numpy()
    assert x32.dtype == np.float32
    np.testing.assert_allclose(x32, x, atol=1e-6)


def test_agrees_with_physics_spectrum_if_present(mol):
    spectrum = pytest.importorskip("qlsgym.physics.spectrum", reason="agent A's physics.spectrum not present")
    for b in mol.blocks:
        for sigma in ("+", "-"):
            mine = retained_transitions(mol, b.index, sigma)
            theirs = np.asarray(spectrum.embedding_transitions(mol, b.index, sigma))
            np.testing.assert_array_equal(np.sort(mine), np.sort(theirs))
