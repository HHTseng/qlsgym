import numpy as np
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.column_train import (
    ColumnTrainConfig,
    complete_loss,
    sample_states,
    temporal_losses,
)
from qlsgym.surrogate.embedding import TorchEmbedding
from qlsgym.surrogate.temporal_fno import (
    ContinuousTimeEmbedding,
    TemporalColumnFNO,
    TemporalColumnFNOConfig,
)
from qlsgym.surrogate.train import load_model


def tiny_config(**changes):
    values = dict(
        n_modes=4,
        fno_hidden=8,
        fno_layers=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        factorization="tucker",
        rank=0.5,
        domain_padding=0.0,
        positional_embedding="grid",
        d_model=8,
        attention_heads=2,
        attention_layers=1,
        ff_dim=16,
        dropout=0.0,
        time_frequencies=2,
        attention_gate_init=0.05,
        off_resonance_linewidths=3.0,
        identity_logit_bias=3.0,
    )
    values.update(changes)
    return TemporalColumnFNOConfig(**values)


def test_continuous_time_embedding_shape_and_grid_independence():
    embedding = ContinuousTimeEmbedding(d_model=8, n_frequencies=3)
    coarse = embedding(torch.linspace(0.0, 2.0, 5))
    fine = embedding(torch.linspace(0.0, 2.0, 9))
    assert coarse.shape == (5, 8)
    assert torch.allclose(coarse[[0, 2, 4]], fine[[0, 4, 8]], atol=1e-6)


def test_temporal_column_fno_structural_contract_and_determinism():
    molecule = load_molecule("synthetic")
    embedding = TorchEmbedding(molecule, 0, "+", "cpu")
    model = TemporalColumnFNO.for_block(molecule, 0, "+", tiny_config()).eval()
    omega = torch.tensor([embedding.numpy.w_res[0]], dtype=torch.float64)
    controls = embedding.control_channels(omega, out_dtype=torch.float32)
    first = model.columns(controls)
    second = model.columns(controls)
    m = molecule.blocks[0].n_states
    assert first.shape == (1, molecule.window.n_tau, 2 * m, m)
    assert torch.equal(first, second)
    assert torch.all(first >= 0.0)
    assert torch.allclose(first.sum(2), torch.ones_like(first.sum(2)))
    identity = torch.zeros((2 * m, m))
    identity[:m] = torch.eye(m)
    assert torch.equal(first[0, 0], identity)

    populations = torch.rand((2, m), dtype=torch.float64)
    populations /= populations.sum(1, keepdim=True)
    mixed = populations.mean(0, keepdim=True)
    separate = model.propagate(populations, embedding, omega.expand(2)).mean(0)
    together = model.propagate(mixed, embedding, omega)[0]
    assert torch.allclose(together, separate, atol=1e-10)


def test_temporal_checkpoint_round_trip(tmp_path):
    molecule = load_molecule("synthetic")
    model = TemporalColumnFNO.for_block(molecule, 0, "+", tiny_config())
    path = tmp_path / "temporal.pt"
    torch.save(
        {"state_dict": model.state_dict(), "meta": model.metadata(), "source": "qlsgym"},
        path,
    )
    loaded = load_model(str(path), "cpu", molecule=molecule)
    assert isinstance(loaded, TemporalColumnFNO)
    assert loaded.metadata() == model.metadata()


def test_pure_transformer_ablation_round_trip(tmp_path):
    molecule = load_molecule("synthetic")
    model = TemporalColumnFNO.for_block(
        molecule, 0, "+", tiny_config(spectral_trunk=False)
    )
    assert model.metadata()["architecture"] == "temporal_transformer_columns_v1"
    path = tmp_path / "transformer.pt"
    torch.save(
        {"state_dict": model.state_dict(), "meta": model.metadata(), "source": "qlsgym"},
        path,
    )
    loaded = load_model(str(path), "cpu", molecule=molecule)
    assert isinstance(loaded, TemporalColumnFNO)
    assert loaded.metadata() == model.metadata()


def test_temporal_losses_and_gradients_are_finite():
    molecule = load_molecule("synthetic")
    embedding = TorchEmbedding(molecule, 0, "+", "cpu")
    model = TemporalColumnFNO.for_block(molecule, 0, "+", tiny_config())
    omega = torch.tensor([embedding.numpy.w_res[0]], dtype=torch.float64)
    predicted = model.columns(embedding.control_channels(omega, out_dtype=torch.float32))
    truth = predicted.detach().clone()
    d1, spectral = temporal_losses(predicted, truth, model.taus)
    assert d1.item() == 0.0
    assert spectral.item() == 0.0

    states = sample_states(model.n_states, 4, np.random.default_rng(7), "cpu")
    config = ColumnTrainConfig(derivative_weight=0.1, spectral_weight=0.02)
    loss, metrics = complete_loss(predicted, truth, states, model.taus, config)
    loss.backward()
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
