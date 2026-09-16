"""BlockFNO forward shape, softmax normalisation, checkpoint meta and load_model checks."""
import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.embedding import Embedding
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.train import FingerprintMismatch, TrainConfig, checkpoint_dict, load_model

TINY = FNOConfig(n_modes=4, hidden_channels=8, n_layers=1, lifting_channel_ratio=2, projection_channel_ratio=2)


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic")


def test_forward_shape_and_softmax(mol):
    torch.manual_seed(0)
    e = Embedding(mol, 0, "+")
    m = BlockFNO.for_block(mol, 0, "+", TINY)
    assert (m.in_channels, m.out_channels) == (e.n_channels, 2 * e.n_states)
    x = torch.as_tensor(e.build(np.full((3, e.n_states), 1 / e.n_states),
                                np.full(3, mol.trap.nu_f)), dtype=torch.float32)
    y = m(x)
    assert y.shape == (3, 2 * e.n_states, mol.window.n_tau)
    assert torch.all(y >= 0)
    torch.testing.assert_close(y.sum(1), torch.ones(3, mol.window.n_tau))
    assert m.logits(x).shape == y.shape
    meta = m.metadata()
    for k in ("in_channels", "out_channels", "block_index", "sigma", "config", "molecule", "fingerprint", "n_nu"):
        assert k in meta
    assert meta["molecule"] == "synthetic" and meta["fingerprint"] == mol.fingerprint()
    assert meta["n_nu"] == mol.trap.n_nu and meta["block_index"] == 0 and meta["sigma"] == "+"


def test_checkpoint_roundtrip_and_fingerprint_checks(mol, tmp_path):
    torch.manual_seed(1)
    m = BlockFNO.for_block(mol, 1, "-", TINY)
    ck = checkpoint_dict(m, TrainConfig(fno=TINY), DataConfig(n_freq=2, n_init=2), DataConfig(n_freq=2, n_init=2), 0, 0.1, 0.2)
    p = tmp_path / "best_onres.pt"
    torch.save(ck, p)
    m2 = load_model(str(p), "cpu", molecule=mol)
    e = Embedding(mol, 1, "-")
    x = torch.randn(2, e.n_channels, mol.window.n_tau)
    torch.testing.assert_close(m(x), m2(x))
    assert m2.fingerprint == mol.fingerprint() and m2.sigma == "-" and m2.block_index == 1
    # different physics -> refused
    other = load_molecule("synthetic", seed=1)
    assert other.fingerprint() != mol.fingerprint()
    with pytest.raises(FingerprintMismatch):
        load_model(str(p), "cpu", molecule=other)
    load_model(str(p), "cpu", molecule=other, allow_mismatch=True)   # same geometry, explicit override
    # legacy (no fingerprint in meta) -> refused unless legacy=True
    ck["meta"].pop("fingerprint"); ck["meta"].pop("molecule"); ck["meta"].pop("n_nu")
    torch.save(ck, p)
    with pytest.raises(FingerprintMismatch):
        load_model(str(p), "cpu", molecule=mol)
    m3 = load_model(str(p), "cpu", molecule=mol, legacy=True)
    torch.testing.assert_close(m(x), m3(x))
    # wrong block geometry is always refused, legacy or not
    ck["meta"]["block_index"] = 0
    torch.save(ck, p)
    with pytest.raises(FingerprintMismatch):
        load_model(str(p), "cpu", molecule=mol, legacy=True)
