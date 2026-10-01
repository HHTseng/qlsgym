"""Transfer-column FNO with attention along intra-pulse time ``tau``.

The model maps control features ``C(omega, sigma)`` to stochastic transfer
columns ``B_(alpha,k)``.  It never consumes RL-step history: attention acts on
the pulse-time grid, where the coherent Hamiltonian trajectory is continuous.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..spec import Molecule


@dataclass(frozen=True)
class TemporalColumnFNOConfig:
    n_modes: int = 40
    fno_hidden: int = 96
    fno_layers: int = 3
    lifting_channel_ratio: int = 3
    projection_channel_ratio: int = 3
    factorization: str = "tucker"
    rank: float = 0.6
    domain_padding: float = 0.1
    positional_embedding: str | None = "grid"
    d_model: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0
    time_frequencies: int = 16
    attention_gate_init: float = 0.075
    off_resonance_linewidths: float = 3.0
    identity_logit_bias: float = 6.0
    spectral_trunk: bool = True
    # A fixed permutation of E_tau is a negative control.  None is physical.
    shuffled_time_seed: int | None = None


class ContinuousTimeEmbedding(nn.Module):
    """Map physical ``tau/tau_max`` to a Fourier feature vector in R^d."""

    def __init__(self, d_model: int, n_frequencies: int = 16) -> None:
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        self.projection = nn.Linear(1 + 2 * self.n_frequencies, d_model)

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        if taus.ndim != 1:
            raise ValueError("taus must be a one-dimensional physical-time grid")
        scale = taus[-1].clamp_min(torch.finfo(taus.dtype).eps)
        normalized = (taus / scale).unsqueeze(-1)
        frequencies = torch.arange(
            1,
            self.n_frequencies + 1,
            dtype=taus.dtype,
            device=taus.device,
        )
        phase = 2.0 * math.pi * normalized * frequencies
        features = torch.cat((normalized, phase.sin(), phase.cos()), dim=-1)
        return self.projection(features)


class TemporalAttentionBlock(nn.Module):
    """Pre-LayerNorm bidirectional attention on the pulse-time tokens."""

    def __init__(self, config: TemporalColumnFNOConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.attention_layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class TemporalColumnFNO(nn.Module):
    """Map ``C(omega,sigma)`` to ``B_(alpha,k)`` over the full tau grid.

    ``columns`` has shape ``(B, P_tau, 2 M, M)`` and is stochastic along the
    output-state axis.  Consequently ``propagate`` is exactly linear in the
    input population.  The first time row is replaced by ``(I, 0)^T``.
    """

    def __init__(
        self,
        control_channels: int,
        n_states: int,
        taus: torch.Tensor,
        config: TemporalColumnFNOConfig | None = None,
        block_index: int | None = None,
        sigma: str = "+",
        molecule_name: str | None = None,
        fingerprint: str | None = None,
        n_nu: int | None = None,
    ) -> None:
        super().__init__()
        from neuralop.models import FNO

        cfg = config or TemporalColumnFNOConfig()
        if cfg.attention_layers < 1:
            raise ValueError("attention_layers must be positive")
        if not 0.0 < cfg.attention_gate_init < 1.0:
            raise ValueError("attention_gate_init must lie strictly between zero and one")
        if cfg.d_model % cfg.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")

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
        self.register_buffer("taus", torch.as_tensor(taus, dtype=torch.float32).clone())

        if cfg.spectral_trunk:
            self.fno = FNO(
                n_modes=(cfg.n_modes,),
                in_channels=self.control_channels,
                out_channels=cfg.d_model,
                hidden_channels=cfg.fno_hidden,
                n_layers=cfg.fno_layers,
                lifting_channel_ratio=cfg.lifting_channel_ratio,
                projection_channel_ratio=cfg.projection_channel_ratio,
                factorization=cfg.factorization,
                rank=cfg.rank,
                domain_padding=cfg.domain_padding,
                positional_embedding=cfg.positional_embedding,
            )
        else:
            # Pure-Transformer ablation: pointwise control lifting, so all
            # cross-time mixing must come from attention rather than Fourier modes.
            self.fno = nn.Sequential(
                nn.Conv1d(self.control_channels, cfg.d_model, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1),
            )
        self.time_embedding = ContinuousTimeEmbedding(
            cfg.d_model, cfg.time_frequencies
        )
        self.temporal = TemporalAttentionBlock(cfg)
        self.head = nn.Linear(cfg.d_model, self.out_channels)
        gate_logit = math.log(cfg.attention_gate_init / (1.0 - cfg.attention_gate_init))
        self.attention_gate_raw = nn.Parameter(torch.tensor(gate_logit))

        identity_bias = torch.zeros(2 * self.n_states, self.n_states)
        index = torch.arange(self.n_states)
        identity_bias[index, index] = cfg.identity_logit_bias
        self.register_buffer("identity_bias", identity_bias, persistent=False)
        permutation = torch.arange(len(self.taus))
        if cfg.shuffled_time_seed is not None:
            generator = torch.Generator().manual_seed(cfg.shuffled_time_seed)
            permutation = torch.randperm(len(self.taus), generator=generator)
        self.register_buffer("time_permutation", permutation, persistent=False)

    @classmethod
    def for_block(
        cls,
        molecule: Molecule,
        block_index: int,
        sigma: str = "+",
        config: TemporalColumnFNOConfig | None = None,
    ) -> "TemporalColumnFNO":
        from .embedding import block_shapes

        m, q, _ = block_shapes(molecule, block_index, sigma)
        return cls(
            q + 1,
            m,
            torch.as_tensor(molecule.tau_grid(), dtype=torch.float32),
            config,
            block_index=int(block_index),
            sigma=sigma,
            molecule_name=molecule.name,
            fingerprint=molecule.fingerprint(),
            n_nu=int(molecule.trap.n_nu),
        )

    def columns(self, controls: torch.Tensor) -> torch.Tensor:
        """Return stochastic columns with shape ``(B,P_tau,2M,M)``."""
        latent = self.fno(controls).transpose(1, 2)
        time = self.time_embedding(self.taus.to(latent.dtype))
        tokens = latent + time[self.time_permutation].unsqueeze(0)
        attended = self.temporal(tokens)
        gate = torch.sigmoid(self.attention_gate_raw)
        logits = self.head(tokens + gate * (attended - tokens))
        batch, n_tau, _ = logits.shape
        columns = logits.reshape(
            batch, n_tau, self.n_states, 2 * self.n_states
        ).permute(0, 1, 3, 2)
        columns = torch.softmax(columns + self.identity_bias[None, None], dim=2)

        static = torch.zeros(
            (2 * self.n_states, self.n_states),
            dtype=columns.dtype,
            device=columns.device,
        )
        static[: self.n_states] = torch.eye(
            self.n_states, dtype=columns.dtype, device=columns.device
        )
        mask = torch.zeros(n_tau, dtype=columns.dtype, device=columns.device)
        if n_tau:
            mask[0] = 1.0
        return columns * (1.0 - mask[None, :, None, None]) + static[None, None] * mask[
            None, :, None, None
        ]

    def propagate(self, population, embedding, omegas) -> torch.Tensor:
        population = torch.as_tensor(
            population, dtype=torch.float64, device=embedding.device
        )
        omegas = torch.as_tensor(omegas, dtype=torch.float64, device=embedding.device)
        controls = embedding.control_channels(omegas, out_dtype=torch.float32)
        trajectory = torch.einsum(
            "bpom,bm->bpo", self.columns(controls).double(), population
        )
        distance = embedding.numpy.resonance_distance(omegas.detach().cpu().numpy())
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
            "architecture": (
                "temporal_column_fno_v1"
                if self.config.spectral_trunk
                else "temporal_transformer_columns_v1"
            ),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "control_channels": self.control_channels,
            "n_states": self.n_states,
            "taus": self.taus.detach().cpu().tolist(),
            "block_index": self.block_index,
            "sigma": self.sigma,
            "config": asdict(self.config),
            "molecule": self.molecule_name,
            "fingerprint": self.fingerprint,
            "n_nu": self.n_nu,
        }
