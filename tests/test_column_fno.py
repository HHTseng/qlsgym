import numpy as np
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.column_train import ColumnTrainConfig, branch_aware_loss, sample_states
from qlsgym.surrogate.embedding import TorchEmbedding
from qlsgym.surrogate.fno import ColumnFNO, ColumnFNOConfig
from qlsgym.surrogate.train import load_model


def tiny_config(**changes):
    values = dict(
        n_modes=4,
        hidden_channels=8,
        n_layers=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        factorization="tucker",
        rank=0.5,
        domain_padding=0.0,
        positional_embedding="grid",
        off_resonance_linewidths=3.0,
    )
    values.update(changes)
    return ColumnFNOConfig(**values)


def test_column_fno_is_stochastic_identity_and_linear():
    molecule = load_molecule("synthetic")
    embedding = TorchEmbedding(molecule, 0, "+", "cpu")
    model = ColumnFNO.for_block(molecule, 0, "+", tiny_config())
    omega = torch.tensor([embedding.numpy.w_res[0]], dtype=torch.float64)
    controls = embedding.control_channels(omega, out_dtype=torch.float32)
    columns = model.columns(controls)
    m = molecule.blocks[0].n_states
    assert columns.shape == (1, molecule.window.n_tau, 2 * m, m)
    assert torch.allclose(columns.sum(2), torch.ones_like(columns.sum(2)))
    want = torch.zeros((2 * m, m))
    want[:m] = torch.eye(m)
    assert torch.equal(columns[0, 0], want)

    p = torch.rand((2, m), dtype=torch.float64)
    p /= p.sum(1, keepdim=True)
    mixed = p.mean(0, keepdim=True)
    separate = model.propagate(p, embedding, omega.expand(2)).mean(0)
    together = model.propagate(mixed, embedding, omega)[0]
    assert torch.allclose(together, separate, atol=1e-10)


def test_column_fno_checkpoint_round_trip(tmp_path):
    molecule = load_molecule("synthetic")
    model = ColumnFNO.for_block(molecule, 0, "+", tiny_config())
    path = tmp_path / "column.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": model.metadata(),
            "source": "qlsgym",
        },
        path,
    )
    loaded = load_model(str(path), "cpu", molecule=molecule)
    assert isinstance(loaded, ColumnFNO)
    assert loaded.metadata() == model.metadata()


def test_column_loss_has_finite_gradients_with_structural_zeros():
    molecule = load_molecule("synthetic")
    embedding = TorchEmbedding(molecule, 0, "+", "cpu")
    model = ColumnFNO.for_block(molecule, 0, "+", tiny_config())
    omega = torch.tensor([embedding.numpy.w_res[0]], dtype=torch.float64)
    predicted = model.columns(
        embedding.control_channels(omega, out_dtype=torch.float32)
    )
    truth = predicted.detach().clone()
    states = sample_states(
        model.n_states, 4, np.random.default_rng(7), predicted.device
    )
    loss, _ = branch_aware_loss(predicted, truth, states, ColumnTrainConfig())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
