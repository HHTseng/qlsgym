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


class BlockFNO(nn.Module):
    """FNO surrogate G_theta^(f) for one Hamiltonian block and one sigma."""

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
        """Predicted populations, normalised at each time step (Sec. II.3)."""
        return torch.softmax(self.net(x), dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def metadata(self) -> dict:
        """What a checkpoint records about this model."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "block_index": self.block_index,
            "sigma": self.sigma,
            "config": asdict(self.config),
            "molecule": self.molecule_name,
            "fingerprint": self.fingerprint,
            "n_nu": self.n_nu,
        }
