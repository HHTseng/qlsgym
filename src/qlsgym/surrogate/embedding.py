"""Physics-informed input embedding for the FNO surrogate (paper Eqs. 11-16)."""

from __future__ import annotations

import numpy as np
import torch

from ..spec import Molecule

# Paper Eq. 11 / Table 2 detuning cutoff, rad/ms.
PAPER_DELTA_MAX = 1.0e4


def embedding_delta_max(molecule: Molecule) -> float:
    """Eq. 11 Delta_max for this molecule (see the module note)."""
    return float(molecule.provenance.get("delta_max", PAPER_DELTA_MAX))


def resonant_frequencies(molecule: Molecule, block_index: int, sigma: str = "+") -> np.ndarray:
    """Sideband resonance omega^(+/-) of each coupling in the block (Eq. 3), shape (n_tr,)."""
    block = molecule.blocks[block_index]
    e = molecule.system.energies
    d_omega = e[block.states[block.f_local]] - e[block.states[block.i_local]]
    if sigma == "+":
        return molecule.trap.nu_f + d_omega
    if sigma == "-":
        return molecule.trap.nu_f - d_omega
    raise ValueError(f"sigma must be '+' or '-', got {sigma!r}")


def retained_transitions(
    molecule: Molecule, block_index: int, sigma: str = "+", delta_max: float | None = None
) -> np.ndarray:
    """Indices (into the block's coupling list) retained by Eq. 11."""
    delta_max = embedding_delta_max(molecule) if delta_max is None else float(delta_max)
    block = molecule.blocks[block_index]
    w = molecule.window
    w_res = resonant_frequencies(molecule, block_index, sigma)
    dist = np.maximum(0.0, np.maximum(w.omega_min - w_res, w_res - w.omega_max))
    near = dist <= delta_max
    strong = np.abs(block.omega) >= w.omega_min_coupling
    return np.where(near & strong)[0]


class Embedding:
    """Per-(molecule, block, sigma) embedding geometry, numpy version."""

    def __init__(
        self, molecule: Molecule, block_index: int, sigma: str = "+", delta_max: float | None = None
    ) -> None:
        block = molecule.blocks[block_index]
        w = molecule.window
        self.molecule_name = molecule.name
        self.fingerprint = molecule.fingerprint()
        self.block_index = int(block_index)
        self.sigma = sigma
        self.n_states = int(block.n_states)
        self.delta_max = embedding_delta_max(molecule) if delta_max is None else float(delta_max)
        self.transitions = retained_transitions(molecule, block_index, sigma, self.delta_max)
        self.w_res = resonant_frequencies(molecule, block_index, sigma)[self.transitions]
        # HWHM eta |Omega_q| of each retained transition, rad/ms
        self.linewidths = molecule.trap.eta * np.abs(block.omega[self.transitions])
        self.taus = molecule.tau_grid().astype(np.float64)
        self.omega_min = float(w.omega_min)
        self.omega_max = float(w.omega_max)
        self.tau_max = float(w.tau_max_ms)
        self.s_emb = float(w.s_emb)
        self.beta_emb = float(w.beta_emb)

    # geometry
    @property
    def n_transitions(self) -> int:
        return int(self.w_res.size)

    @property
    def n_channels(self) -> int:
        """M_f + Q + 1 -- the FNO input width."""
        return self.n_states + self.n_transitions + 1

    @property
    def n_tau(self) -> int:
        return int(self.taus.size)

    def resonance_distance(self, omega: np.ndarray | float) -> np.ndarray:
        """min_q |omega_res,q - omega| / (eta |Omega_q|) -- distance in HWHM."""
        omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        return (np.abs(omega[:, None] - self.w_res[None, :]) / self.linewidths[None, :]).min(axis=1)

    # Eqs. 12-16
    def detunings(self, omega: np.ndarray | float) -> np.ndarray:
        """Delta_q(omega, sigma) (Eq. 12) -- shape (Q,) or (B, Q)."""
        omega = np.asarray(omega, dtype=np.float64)
        return self.w_res - omega[..., None]

    def control_channels(self, omega: np.ndarray | float) -> np.ndarray:
        """The Q + 1 control channels, shape (B, Q + 1, P_tau)."""
        omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        delta = self.detunings(omega)                                       # (B, Q)
        phi = (self.s_emb * np.sin(delta[..., None] * self.taus)
               / (1.0 + self.beta_emb * np.abs(delta)[..., None]))          # (B, Q, P)
        w_tilde = 2.0 * (omega - self.omega_min) / (self.omega_max - self.omega_min) - 1.0
        phi_w = self.s_emb * np.sin(2.0 * np.pi * w_tilde[:, None] * (self.taus / self.tau_max))
        return np.concatenate([phi, phi_w[:, None, :]], axis=1)

    def build(self, p0: np.ndarray, omega: np.ndarray | float) -> np.ndarray:
        """Full FNO input (Eq. 16), shape (B, M_f + Q + 1, P_tau)."""
        p0 = np.atleast_2d(np.asarray(p0, dtype=np.float64))
        omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        if omega.size == 1 and p0.shape[0] > 1:
            omega = np.repeat(omega, p0.shape[0])
        if p0.shape[0] != omega.shape[0]:
            raise ValueError("p0 and omega batch sizes disagree")
        if p0.shape[1] != self.n_states:
            raise ValueError(f"p0 has {p0.shape[1]} states, block has {self.n_states}")
        state = np.repeat(p0[:, :, None], self.taus.size, axis=2)
        return np.concatenate([state, self.control_channels(omega)], axis=1)


class TorchEmbedding:
    """Torch twin of Embedding (ported from thffno.torch_ops)."""

    def __init__(
        self,
        molecule: Molecule,
        block_index: int,
        sigma: str = "+",
        device: torch.device | str = "cpu",
        delta_max: float | None = None,
    ) -> None:
        self._init_from(Embedding(molecule, block_index, sigma, delta_max), device)

    @classmethod
    def from_numpy(cls, emb: Embedding, device: torch.device | str = "cpu") -> "TorchEmbedding":
        obj = cls.__new__(cls)
        obj._init_from(emb, device)
        return obj

    def _init_from(self, emb: Embedding, device) -> None:
        self.numpy = emb
        self.molecule_name, self.fingerprint = emb.molecule_name, emb.fingerprint
        self.block_index, self.sigma, self.n_states = emb.block_index, emb.sigma, emb.n_states
        self.omega_min, self.omega_max, self.tau_max = emb.omega_min, emb.omega_max, emb.tau_max
        self.s_emb, self.beta_emb = emb.s_emb, emb.beta_emb
        self.w_res = torch.as_tensor(emb.w_res, dtype=torch.float64, device=device)
        self.taus = torch.as_tensor(emb.taus, dtype=torch.float64, device=device)

    @property
    def n_transitions(self) -> int:
        return int(self.w_res.numel())

    @property
    def n_channels(self) -> int:
        return self.n_states + self.n_transitions + 1

    @property
    def device(self) -> torch.device:
        return self.w_res.device

    def detunings(self, omega: torch.Tensor) -> torch.Tensor:
        return self.w_res - omega[..., None]

    def control_channels(self, omega: torch.Tensor, out_dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """The Q + 1 control channels (Eqs. 13, 15), (B, Q + 1, P_tau)."""
        omega = torch.atleast_1d(torch.as_tensor(omega, dtype=torch.float64, device=self.device))
        delta = self.detunings(omega)
        phi = (self.s_emb * torch.sin(delta[..., None] * self.taus)
               / (1.0 + self.beta_emb * delta.abs()[..., None]))
        w_tilde = 2.0 * (omega - self.omega_min) / (self.omega_max - self.omega_min) - 1.0
        phi_w = self.s_emb * torch.sin(2.0 * np.pi * w_tilde[:, None] * (self.taus / self.tau_max))
        return torch.cat([phi, phi_w[:, None, :]], dim=1).to(out_dtype)

    def build(self, p0: torch.Tensor, omega: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Full FNO input (Eq. 16), shape (B, M_f + Q + 1, P_tau)."""
        p0 = torch.atleast_2d(torch.as_tensor(p0, device=self.device))
        omega = torch.atleast_1d(torch.as_tensor(omega, dtype=torch.float64, device=self.device))
        if omega.numel() == 1 and p0.shape[0] > 1:
            omega = omega.expand(p0.shape[0])
        if p0.shape[0] != omega.shape[0]:
            raise ValueError("p0 and omega batch sizes disagree")
        if p0.shape[1] != self.n_states:
            raise ValueError(f"p0 has {p0.shape[1]} states, block has {self.n_states}")
        state = p0.to(out_dtype)[:, :, None].expand(p0.shape[0], self.n_states, self.taus.numel())
        return torch.cat([state, self.control_channels(omega, out_dtype)], dim=1)


def resonance_geometry_of(molecule: Molecule, block_index: int, sigma: str = "+") -> tuple[np.ndarray, np.ndarray]:
    """(omega_res, linewidth) of the retained transitions; linewidth is the HWHM in rad/ms."""
    emb = Embedding(molecule, block_index, sigma)
    return emb.w_res, emb.linewidths


def block_shapes(molecule: Molecule, block_index: int, sigma: str = "+") -> tuple[int, int, int]:
    """(M_f, Q, in_channels) for one block."""
    emb = Embedding(molecule, block_index, sigma)
    return emb.n_states, emb.n_transitions, emb.n_channels
