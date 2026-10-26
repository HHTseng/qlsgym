import numpy as np
import torch
from torch import nn

from qlsgym import load_molecule
from qlsgym.surrogate.fno import BlockFNO, FNOConfig
from qlsgym.surrogate.train import evaluate_basis_p1


class Static(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        p0 = x[:, :self.m, :]
        return torch.cat([p0, torch.zeros_like(p0)], dim=1)


def test_static_model_scores_exactly_the_static_baseline():
    mol = load_molecule("synthetic")
    m = mol.blocks[0].n_states
    out = evaluate_basis_p1(Static(m), mol, 0, "+", device="cpu")
    s = out["summary"]
    assert s["n_lines"] > 0 and out["err"].shape == (s["n_lines"], m)
    np.testing.assert_allclose(out["err"], out["static"], atol=1e-6)
    assert s["worst"] == s["static_worst"] and s["static_worst"] > 0.5


def test_runs_on_a_real_fno():
    torch.manual_seed(0)
    mol = load_molecule("synthetic")
    model = BlockFNO.for_block(mol, 0, "+", FNOConfig(n_modes=4, hidden_channels=8, n_layers=1,
                                                      lifting_channel_ratio=2, projection_channel_ratio=2))
    s = evaluate_basis_p1(model, mol, 0, "+", device="cpu")["summary"]
    assert 0 <= s["median"] <= s["p95"] <= s["worst"] <= 1
