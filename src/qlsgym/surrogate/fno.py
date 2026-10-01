"""The per-block FNO surrogate (paper Sec. II.3, Eqs. 16-17; Appendix B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..spec import Molecule


@dataclass(frozen=True)
class FNOConfig:
    """Paper Table 2, architecture rows."""

    n_modes: int = 60
    hidden_channels: int = 256
    n_layers: int = 4
    lifting_channel_ratio: int = 5
    projection_channel_ratio: int = 5
    factorization: str = "tucker"
    rank: float = 0.8
    domain_padding: float = 0.1
    # Not specified in the paper; neuralop's default is a concatenated
    # coordinate grid, which is what an out-of-the-box FNO uses.
    positional_embedding: str | None = "grid"


@dataclass(frozen=True)
class ColumnFNOConfig(FNOConfig):
    """FNO that predicts a stochastic transfer column for every input state."""

    off_resonance_linewidths: float = 3.0
    # Add this logit to the unchanged-state entry of every transfer column.
    # Zero preserves v1 checkpoints; a positive value gives correction models
    # a physically useful near-identity initialization.
    identity_logit_bias: float = 0.0


class BlockFNO(nn.Module):
    """State-map FNO 𝒢_{θ,k} for one Hamiltonian block and one σ."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: FNOConfig | None = None,
        block_index: int | None = None,
        sigma: str = "+",
        molecule_name: str | None = None,
        fingerprint: str | None = None,
        n_nu: int | None = None,
    ) -> None:
        super().__init__()
        from neuralop.models import FNO   # heavy import; only needed here

        cfg = config or FNOConfig()
        self.config = cfg
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block_index = block_index
        self.sigma = sigma
        self.molecule_name = molecule_name
        self.fingerprint = fingerprint
        self.n_nu = n_nu
        self.net = FNO(
            n_modes=(cfg.n_modes,),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            lifting_channel_ratio=cfg.lifting_channel_ratio,
            projection_channel_ratio=cfg.projection_channel_ratio,
            factorization=cfg.factorization,
            rank=cfg.rank,
            domain_padding=cfg.domain_padding,
            positional_embedding=cfg.positional_embedding,
        )

    @classmethod
    def for_block(
        cls,
        molecule: Molecule,
        block_index: int,
        sigma: str = "+",
        config: FNOConfig | None = None,
    ) -> "BlockFNO":
        """Instantiate with the channel counts implied by the block geometry."""
        from .embedding import block_shapes

        m, _q, n_in = block_shapes(molecule, block_index, sigma)
        return cls(n_in, 2 * m, config, block_index=int(block_index), sigma=sigma,
                   molecule_name=molecule.name, fingerprint=molecule.fingerprint(),
                   n_nu=int(molecule.trap.n_nu))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (B, 2 M_f, P_tau) network output, before normalisation."""
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the stacked branch populations (v̂_{α,0}, v̂_{α,1})."""
        return torch.softmax(self.net(x), dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def metadata(self) -> dict:
        """What a checkpoint records about this model."""
        return {
            "architecture": "state_map_v1",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "block_index": self.block_index,
            "sigma": self.sigma,
            "config": asdict(self.config),
            "molecule": self.molecule_name,
            "fingerprint": self.fingerprint,
            "n_nu": self.n_nu,
        }


class ColumnFNO(nn.Module):
    """Map α to (B̂_{α,0}, B̂_{α,1}), linearly acting on s_t.

    The construction enforces nonnegative stochastic columns and
    B̂_{(ω,0,σ)} = (I, 0)ᵀ.
    """

    def __init__(
        self,
        control_channels: int,
        n_states: int,
        config: ColumnFNOConfig | None = None,
        block_index: int | None = None,
        sigma: str = "+",
        molecule_name: str | None = None,
        fingerprint: str | None = None,
        n_nu: int | None = None,
    ) -> None:
        super().__init__()
        from neuralop.models import FNO

        cfg = config or ColumnFNOConfig()
        self.config = cfg
        self.control_channels = int(control_channels)
        self.n_states = int(n_states)
        self.in_channels = self.control_channels
        self.out_channels = 2 * self.n_states * self.n_states
        self.block_index = block_index
        self.sigma = sigma
        self.molecule_name = molecule_name
        self.fingerprint = fingerprint
        self.n_nu = n_nu
        identity_bias = torch.zeros(2 * self.n_states, self.n_states)
        identity_bias[torch.arange(self.n_states), torch.arange(self.n_states)] = (
            cfg.identity_logit_bias
        )
        self.register_buffer("identity_bias", identity_bias, persistent=False)
        self.net = FNO(
            n_modes=(cfg.n_modes,),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_channels=cfg.hidden_channels,
            n_layers=cfg.n_layers,
            lifting_channel_ratio=cfg.lifting_channel_ratio,
            projection_channel_ratio=cfg.projection_channel_ratio,
            factorization=cfg.factorization,
            rank=cfg.rank,
            domain_padding=cfg.domain_padding,
            positional_embedding=cfg.positional_embedding,
        )

    @classmethod
    def for_block(
        cls,
        molecule: Molecule,
        block_index: int,
        sigma: str = "+",
        config: ColumnFNOConfig | None = None,
    ) -> "ColumnFNO":
        from .embedding import block_shapes

        m, q, _ = block_shapes(molecule, block_index, sigma)
        return cls(
            q + 1,
            m,
            config,
            block_index=int(block_index),
            sigma=sigma,
            molecule_name=molecule.name,
            fingerprint=molecule.fingerprint(),
            n_nu=int(molecule.trap.n_nu),
        )

    def columns(self, controls: torch.Tensor) -> torch.Tensor:
        """Return B̂_α=[B̂_{α,0}; B̂_{α,1}] with shape (B, P_τ, 2M, M)."""
        logits = self.net(controls)
        batch, _, n_tau = logits.shape
        columns = logits.reshape(
            batch, self.n_states, 2 * self.n_states, n_tau
        ).permute(0, 3, 2, 1)
        columns = torch.softmax(columns + self.identity_bias[None, None], dim=2)
        static = torch.zeros(
            (2 * self.n_states, self.n_states),
            dtype=columns.dtype,
            device=columns.device,
        )
        static[: self.n_states] = torch.eye(
            self.n_states, dtype=columns.dtype, device=columns.device
        )
        if n_tau:
            mask = torch.zeros(n_tau, dtype=columns.dtype, device=columns.device)
            mask[0] = 1.0
            columns = columns * (1.0 - mask[None, :, None, None])
            columns = columns + static[None, None] * mask[None, :, None, None]
        return columns

    def propagate(self, population, embedding, omegas) -> torch.Tensor:
        """Evaluate v̂_{α,k}=B̂_{α,k}s_t; off-resonant rows use (I, 0)ᵀ."""
        population = torch.as_tensor(
            population, dtype=torch.float64, device=embedding.device
        )
        omegas = torch.as_tensor(omegas, dtype=torch.float64, device=embedding.device)
        controls = embedding.control_channels(omegas, out_dtype=torch.float32)
        columns = self.columns(controls).double()
        trajectory = torch.einsum("bpom,bm->bpo", columns, population)
        distance = embedding.numpy.resonance_distance(
            omegas.detach().cpu().numpy()
        )
        off = torch.as_tensor(
            distance > self.config.off_resonance_linewidths,
            dtype=torch.bool,
            device=trajectory.device,
        )
        if bool(off.any()):
            static = torch.zeros_like(trajectory[off])
            static[:, :, : self.n_states] = population[off, None, :]
            trajectory[off] = static
        return trajectory

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        return self.columns(controls)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict:
        return {
            "architecture": "transfer_columns_v1",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "control_channels": self.control_channels,
            "n_states": self.n_states,
            "block_index": self.block_index,
            "sigma": self.sigma,
            "config": asdict(self.config),
            "molecule": self.molecule_name,
            "fingerprint": self.fingerprint,
            "n_nu": self.n_nu,
        }
