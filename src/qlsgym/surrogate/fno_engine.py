"""The trained surrogate inside the loop: per-(block, sigma) FNOs with an exact fallback."""

from __future__ import annotations

import os

import numpy as np
import torch

from ..spec import Molecule, TauBatchedEngine, branches_batch_fallback
from .embedding import TorchEmbedding
from .train import load_model


class FnoEngine:
    """Evaluate v̂_{α,k}=𝒢_{θ,k}(s_t,ω,σ)(τ), with exact fallback."""

    def __init__(
        self,
        molecule: Molecule,
        checkpoints: dict,
        tau_indices: np.ndarray | None = None,
        device: torch.device | str | None = None,
        fallback: TauBatchedEngine | None = None,
        legacy: bool | set = False,
        allow_mismatch: bool = False,
    ) -> None:
        self.molecule = molecule
        n_tau = molecule.window.n_tau
        self.tau_indices = (np.arange(n_tau) if tau_indices is None
                            else np.asarray(tau_indices, dtype=np.int64))
        if self.tau_indices.size and (self.tau_indices.min() < 0 or self.tau_indices.max() >= n_tau):
            raise ValueError("tau_indices out of range")
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints = {self._key(k): v for k, v in checkpoints.items()}
        self._fallback = fallback
        if fallback is not None:
            self._check_fallback(fallback)
        self._models: dict = {}
        self._embs: dict = {}
        self.calls = {"fno": 0, "exact_sigma_minus": 0, "exact_untrained": 0, "exact_primitive": 0}
        for key, path in self.checkpoints.items():
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            leg = legacy if isinstance(legacy, bool) else (key in legacy)
            model = load_model(path, self.device, molecule=molecule, allow_mismatch=allow_mismatch, legacy=leg)
            if (int(model.block_index), model.sigma) != key:
                raise ValueError(f"{path} is for block {model.block_index} sigma{model.sigma}, registered as {key}")
            self._models[key] = model
            self._embs[key] = TorchEmbedding(molecule, key[0], key[1], self.device)
        self._tau_t = torch.as_tensor(self.tau_indices, device=self.device)

    # helpers
    @staticmethod
    def _key(k) -> tuple[int, str]:
        if isinstance(k, str):
            b, s = k.split(",")
            return int(b), s.strip()
        return int(k[0]), str(k[1])

    def _check_fallback(self, fb) -> None:
        fti = getattr(fb, "tau_indices", None)
        if fti is None or not np.array_equal(np.asarray(fti), self.tau_indices):
            raise ValueError("fallback.tau_indices must equal FnoEngine.tau_indices "
                             "(rows of branches_all_tau are zipped positionally)")

    @property
    def fallback(self) -> TauBatchedEngine:
        if self._fallback is None:
            from ..physics.engines import ExactEngine   # lazy: only when needed

            self._fallback = ExactEngine(self.molecule, tau_indices=self.tau_indices)
            self._check_fallback(self._fallback)
        return self._fallback

    @property
    def trained(self) -> set:
        """{(block_index, sigma), ...} covered by a surrogate."""
        return set(self._models)

    def surrogate_fraction(self) -> float:
        tot = sum(self.calls.values())
        return self.calls["fno"] / tot if tot else float("nan")

    def in_window(self, omega: float) -> bool:
        return self.molecule.window.contains(float(omega))

    # surrogate evaluation
    @torch.no_grad()
    def _fno_block_batch(self, key: tuple, sub: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """Return stacked (v̂_{α,0}, v̂_{α,1}) for one molecular block."""
        sub = np.atleast_2d(np.asarray(sub, dtype=np.float64))      # (1, M) or (n, M)
        omegas = np.asarray(omegas, dtype=np.float64).ravel()
        n = omegas.size
        if sub.shape[0] == 1 and n > 1:
            sub = np.broadcast_to(sub, (n, sub.shape[1]))
        if sub.shape[0] != n:
            raise ValueError(f"sub has {sub.shape[0]} rows, omegas has {n}")
        mass = sub.sum(1, keepdims=True)                            # (n, 1)
        live = mass[:, 0] > 0.0
        emb = self._embs[key]
        s_block = torch.as_tensor(
            sub / np.where(mass > 0.0, mass, 1.0),
            dtype=torch.float64,
            device=self.device,
        )
        w = torch.as_tensor(omegas, device=self.device)
        model = self._models[key]
        if hasattr(model, "propagate"):
            traj = model.propagate(s_block, emb, w)
        else:
            x = emb.build(s_block, w, out_dtype=torch.float32)      # (n, C, P_tau)
            y = model(x)                                            # (n, 2M, P_tau)
            traj = y.permute(0, 2, 1).double()                      # (n, P_tau, 2M)
        out = traj[:, self._tau_t].cpu().numpy() * mass[:, None]
        out[~live] = 0.0
        return out

    def block_branches(self, block_index: int, sigma: str, sub: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """Return stacked (v̂_{α,0}, v̂_{α,1}) for one block and many ω."""
        key = (int(block_index), str(sigma))
        if key not in self._models:
            raise KeyError(f"no surrogate for block {block_index} sigma{sigma}; "
                           f"trained: {sorted(self.trained)}")
        omegas = np.asarray(omegas, dtype=np.float64).ravel()
        self.calls["fno"] += int(omegas.size)
        return self._fno_block_batch(key, sub, omegas)

    # protocol
    def branches_batch(self, p_in: np.ndarray, omegas: np.ndarray, sigma: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (v̂_{α,0}, v̂_{α,1}) with shape (n_omega, n_tau, N)."""
        p_in = np.asarray(p_in, dtype=np.float64)
        omegas = np.asarray(omegas, dtype=np.float64).ravel()
        n, ns, nt = omegas.size, self.molecule.n_states, self.tau_indices.size
        v0 = np.zeros((n, nt, ns))
        v1 = np.zeros((n, nt, ns))
        if n == 0:
            return v0, v1
        in_win = np.array([self.in_window(w) for w in omegas])
        idx_in = np.where(in_win)[0]
        p_rest = p_in.copy()
        if idx_in.size:
            for b in self.molecule.blocks:
                key = (b.index, sigma)
                sub = p_in[b.states]
                mass = float(sub.sum())
                if mass <= 0.0:
                    continue
                if key in self._models:
                    self.calls["fno"] += int(idx_in.size)
                    res = self._fno_block_batch(key, sub, omegas[idx_in])
                    m = b.n_states
                    v0[np.ix_(idx_in, np.arange(nt), b.states)] = res[..., :m]
                    v1[np.ix_(idx_in, np.arange(nt), b.states)] = res[..., m:]
                    p_rest[b.states] = 0.0
                else:
                    self.calls["exact_sigma_minus" if sigma == "-" else "exact_untrained"] += int(idx_in.size)
            if p_rest.sum() > 0.0:
                exact0, exact1 = branches_batch_fallback(
                    self.fallback, p_rest, omegas[idx_in], sigma
                )
                v0[idx_in] += exact0
                v1[idx_in] += exact1
        for row in np.where(~in_win)[0]:
            self.calls["exact_primitive"] += 1
            v0[row], v1[row] = self.fallback.branches_all_tau(
                p_in, float(omegas[row]), sigma
            )
        self._check_conservation(v0, v1, p_in)
        return v0, v1

    def branches_all_tau(self, p_in: np.ndarray, omega: float, sigma: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (v̂_{α,0}, v̂_{α,1}) over τ for one (s_t, ω, σ)."""
        v0, v1 = self.branches_batch(p_in, np.array([float(omega)]), sigma)
        return v0[0], v1[0]

    def branches(self, p_in: np.ndarray, omega: float, sigma: str, tau_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (v̂_{α,0}, v̂_{α,1}) for α=(ω, τ[tau_index], σ)."""
        rows = np.where(self.tau_indices == int(tau_index))[0]
        if rows.size == 0:
            raise ValueError(f"tau_index {tau_index} is not in this engine's tau_indices")
        v0, v1 = self.branches_all_tau(p_in, omega, sigma)
        return v0[rows[0]], v1[rows[0]]

    def _check_conservation(self, v0, v1, p_in) -> None:
        """Check Σ_k 1ᵀv̂_{α,k}=1ᵀs_t for every evaluated α."""
        total = v0.sum(-1) + v1.sum(-1)
        if not np.allclose(total, float(p_in.sum()), atol=1e-4):
            raise RuntimeError(
                f"FnoEngine broke probability conservation: sum(p_out) in "
                f"[{total.min():.4f}, {total.max():.4f}] vs sum(p_in) = {p_in.sum():.4f}")
