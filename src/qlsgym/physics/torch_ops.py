"""Torch mirror of the exact propagation, batched over drive frequency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..spec import Block, Molecule
from .hamiltonian import BlockOperator, make_operator, operator

__all__ = ["TorchBlockOperator", "propagator_columns", "transfer_columns",
           "trajectories_from_columns"]


@dataclass(frozen=True)
class TorchBlockOperator:
    """Device-resident copy of a BlockOperator."""

    block_index: int
    n_states: int
    sigma: str
    nu_f: float
    n_nu: int
    e_shifted: torch.Tensor       # (M,) float64
    src: torch.Tensor             # (n_keep (n_nu-1),) int64, already nu-offset
    dst: torch.Tensor             # same
    gval: torch.Tensor            # (n_keep (n_nu-1),) complex128, rung factors applied
    component: torch.Tensor       # (n_nu M,) int64
    n_component: int

    @property
    def dim(self) -> int:
        return self.n_nu * self.n_states

    @property
    def device(self) -> torch.device:
        return self.e_shifted.device

    @classmethod
    def from_numpy(cls, op: BlockOperator, device="cpu") -> "TorchBlockOperator":
        src, dst, gval = op._graph(op.n_nu)
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        return cls(
            block_index=op.block_index, n_states=op.n_states, sigma=op.sigma,
            nu_f=float(op.nu_f), n_nu=int(op.n_nu),
            e_shifted=t(op.e_shifted, torch.float64),
            src=t(src, torch.int64), dst=t(dst, torch.int64),
            gval=t(gval, torch.complex128),
            component=t(op.component_n, torch.int64), n_component=int(op.n_component_n),
        )

    def build(self, omegas: torch.Tensor) -> torch.Tensor:
        """Batched (B, n_nu M, n_nu M) Hamiltonians; the torch twin of BlockOperator.build."""
        dev = self.e_shifted.device
        omegas = torch.atleast_1d(omegas).to(device=dev, dtype=torch.float64)
        b, m, n_nu = omegas.shape[0], self.n_states, self.n_nu
        dim = n_nu * m
        diag = torch.cat(
            [self.e_shifted[None, :] + k * (self.nu_f - omegas)[:, None] for k in range(n_nu)], dim=1
        )                                                                    # (B, dim)
        counts = torch.bincount(self.component, minlength=self.n_component).to(torch.float64)
        sums = torch.zeros((b, self.n_component), dtype=torch.float64, device=dev).index_add_(
            1, self.component, diag)
        diag = diag - (sums / counts)[:, self.component]

        h = torch.zeros((b, dim, dim), dtype=torch.complex128, device=dev)
        idx = torch.arange(dim, device=dev)
        h[:, idx, idx] = diag.to(torch.complex128)
        if self.src.numel():
            batch = torch.arange(b, device=dev)[:, None].expand(b, self.src.numel())
            rows = self.dst[None, :].expand_as(batch)
            cols = self.src[None, :].expand_as(batch)
            gv = self.gval[None, :].expand_as(batch)
            h.index_put_((batch, rows, cols), gv, accumulate=True)
            h.index_put_((batch, cols, rows), gv.conj(), accumulate=True)
        return h


def propagator_columns(h: torch.Tensor, taus: torch.Tensor, n_cols: int | None = None) -> torch.Tensor:
    """U(tau)[:,:n_cols] for a batch of Hamiltonians (B, D, D); returns (B, n_tau, D, n_cols)."""
    evals, evecs = torch.linalg.eigh(h)                                    # (B,D), (B,D,D)
    phases = torch.exp(-1j * taus.to(evals.dtype)[None, :, None] * evals[:, None, :])   # (B,P,D)
    n_cols = h.shape[-1] if n_cols is None else n_cols
    w = evecs[:, None, :, :] * phases[:, :, None, :]                        # (B,P,D,D)
    return w @ evecs[:, None, :n_cols, :].conj().transpose(-1, -2)


_TORCH_OPS: dict = {}


def torch_operator(molecule: Molecule, block, sigma: str, omega: float | None, device) -> TorchBlockOperator:
    """Cached TorchBlockOperator; omega=None is the in-window operator, otherwise the RWA is taken
    at omega (a primitive).
    """
    dev = torch.device(device)
    if isinstance(block, Block):
        if omega is None:
            op = make_operator(molecule, block, sigma)
        else:
            op = make_operator(molecule, block, sigma, omega_lo=omega, omega_hi=omega)
        return TorchBlockOperator.from_numpy(op, dev)
    key = (molecule.fingerprint(), int(block), sigma, None if omega is None else float(omega), str(dev))
    top = _TORCH_OPS.get(key)
    if top is None:
        top = TorchBlockOperator.from_numpy(operator(molecule, int(block), sigma, omega), dev)
        _TORCH_OPS[key] = top
    return top


def transfer_columns(
    molecule: Molecule,
    block_index,
    omegas,
    sigma: str = "+",
    tau_indices=None,
    device="cpu",
    out_dtype: torch.dtype = torch.float32,
    chunk_bytes: float = 2.0e9,
    op: TorchBlockOperator | None = None,
) -> torch.Tensor:
    """Lumped transfer columns for many drive frequencies at once: (n_omega, n_tau, 2 M, M) on
    device, n_tau = len(tau_indices) (default the whole tau grid).
    """
    dev = torch.device(device)
    om = torch.as_tensor(np.asarray(omegas, dtype=np.float64).reshape(-1), dtype=torch.float64, device=dev) \
        if not isinstance(omegas, torch.Tensor) else omegas.reshape(-1).to(device=dev, dtype=torch.float64)
    if op is None:
        w = molecule.window
        inside = (om >= w.omega_min) & (om <= w.omega_max)
        if bool(inside.all()):
            op = torch_operator(molecule, block_index, sigma, None, dev)
        elif bool((~inside).all()) and om.numel() and bool((om == om[0]).all()):
            op = torch_operator(molecule, block_index, sigma, float(om[0].item()), dev)
        else:
            raise ValueError("omegas must be all in-window or all one off-window primitive frequency")
    taus_np = molecule.tau_grid()
    if tau_indices is not None:
        taus_np = taus_np[np.asarray(tau_indices, dtype=np.int64)]
    taus = torch.as_tensor(taus_np, dtype=torch.float64, device=dev)
    n_tau, m, n_nu = taus.numel(), op.n_states, op.n_nu
    per_freq = max(1, n_tau * (n_nu * m) ** 2 * 16)
    chunk = max(1, min(256, int(chunk_bytes // per_freq)))
    out = torch.empty((om.numel(), n_tau, 2 * m, m), dtype=out_dtype, device=dev)
    for lo in range(0, om.numel(), chunk):
        hi = min(lo + chunk, om.numel())
        u = propagator_columns(op.build(om[lo:hi]), taus, n_cols=m)         # (b, P, n_nu M, M)
        t = u.real.square() + u.imag.square()
        out[lo:hi] = torch.cat(
            [t[..., :m, :], t[..., m:, :].unflatten(-2, (n_nu - 1, m)).sum(-3)], dim=-2
        ).to(out_dtype)
    return out


def trajectories_from_columns(t0: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    """p(tau) = T0(tau) p0: t0 (..., n_tau, 2M, M), p0 (..., M) (leading axes broadcast); returns
    (..., n_tau, 2M).
    """
    return (t0 * p0[..., None, None, :]).sum(-1)
