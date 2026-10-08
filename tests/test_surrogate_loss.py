import pytest
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import FNOConfig
from qlsgym.surrogate.train import TrainConfig, train, weighted_kl_loss, weighted_loss

F64 = torch.float64


def _problem(seed=0, b=3, p=5, k=8):
    g = torch.Generator().manual_seed(seed)
    p_true = torch.softmax(3 * torch.randn(b, p, k, generator=g, dtype=F64), -1)
    w = 1 + 5 * torch.rand(b, k, generator=g, dtype=F64)
    return p_true, w


def _logit_grad(loss_fn, logits):
    z = logits.clone().requires_grad_(True)
    loss_fn(z).backward()
    return z.grad


def test_wkl_vanishes_at_the_truth_and_is_positive_elsewhere():
    p_true, w = _problem()
    assert float(weighted_kl_loss(p_true.log(), p_true, w)) == pytest.approx(0.0, abs=1e-12)
    other = torch.softmax(torch.randn(p_true.shape, dtype=F64), -1).log()
    assert float(weighted_kl_loss(other, p_true, w, "none").min()) > 0


def test_wkl_minimum_is_the_truth_for_any_weights_but_weighted_ce_is_biased():
    p_true, w = _problem(1)
    g = _logit_grad(lambda z: weighted_kl_loss(z, p_true, w), p_true.log())
    assert g.abs().max() < 1e-12

    def naive(z):
        return -(w[:, None, :] * p_true * torch.log_softmax(z, -1)).sum()
    assert _logit_grad(naive, p_true.log()).abs().max() > 1e-2
    assert _logit_grad(naive, (w[:, None, :] * p_true).log()).abs().max() < 1e-12


def test_wkl_with_unit_weights_is_cross_entropy_minus_entropy():
    p_true, _ = _problem(2)
    z = torch.randn(p_true.shape, dtype=F64)
    ones = torch.ones(p_true.shape[0], p_true.shape[-1], dtype=F64)
    ce = -(p_true * torch.log_softmax(z, -1)).sum(-1).mean(-1)
    h = -(torch.xlogy(p_true, p_true)).sum(-1).mean(-1)
    torch.testing.assert_close(weighted_kl_loss(z, p_true, ones, "none"), ce - h)


def test_wkl_logit_gradient_has_the_cross_entropy_form():
    p_true, w = _problem(3)
    z = torch.randn(p_true.shape, dtype=F64)
    q = torch.softmax(z, -1)
    wb = w[:, None, :]
    want = (wb * (q - p_true) - q * (wb * (q - p_true)).sum(-1, keepdim=True)) / p_true.shape[1]
    torch.testing.assert_close(_logit_grad(lambda x: weighted_kl_loss(x, p_true, w), z), want)


def test_underestimated_population_is_corrected_by_wkl_not_by_eq21():
    p_true = torch.tensor([[[0.5, 0.5]]], dtype=F64)
    z = torch.tensor([[[-9.21, 0.0]]], dtype=F64)
    w = torch.ones(1, 2, dtype=F64)
    g_paper = _logit_grad(lambda x: weighted_loss(torch.softmax(x, -1), p_true, w), z)[0, 0, 0]
    g_wkl = _logit_grad(lambda x: weighted_kl_loss(x, p_true, w), z)[0, 0, 0]
    assert g_wkl == pytest.approx(-0.5, abs=1e-3)
    assert abs(float(g_paper)) < 1e-3 * abs(float(g_wkl))


def test_train_runs_with_wkl():
    mol = load_molecule("synthetic")
    cfg = TrainConfig(epochs=2, batch_size=8, loss="wkl", sel_n_freq=4, sel_n_init=4,
                      fno=FNOConfig(n_modes=4, hidden_channels=8, n_layers=1,
                                    lifting_channel_ratio=2, projection_channel_ratio=2))
    data = DataConfig(n_freq=8, n_init=8, n_pairs=32, seed=0)
    out = train(mol, 0, "+", cfg, data, DataConfig(n_freq=4, n_init=4, n_pairs=8, seed=1),
                device="cpu", verbose=False)
    h = out["history"]
    assert len(h["train"]) == 2 and all(x >= 0 and x == x for x in h["train"] + h["val"])
