"""Implementations of qlsgym.spec.TauBatchedEngine."""

from __future__ import annotations

import numpy as np

from ..spec import Block, Molecule, Primitive
from .hamiltonian import rwa_sectors
from .propagate import transfer_matrix
from .spectrum import is_on_resonance

__all__ = ["ExactEngine", "TorchExactEngine", "TrivialEngine", "NoisyExactEngine",
           "TableEngine", "BlockSubstituteEngine"]


class _EngineBase:
    """Shared bookkeeping: tau grid, sector dispatch, per-drive column cache."""

    def __init__(self, molecule: Molecule, tau_indices=None):
        self.molecule = molecule
        n_tau = molecule.window.n_tau
        self.tau_indices = (np.arange(n_tau) if tau_indices is None
                            else np.asarray(tau_indices, dtype=np.int64).reshape(-1))
        if self.tau_indices.size and (self.tau_indices.min() < 0 or self.tau_indices.max() >= n_tau):
            raise ValueError(f"tau_indices must lie in [0, {n_tau})")
        self.taus = molecule.tau_grid()[self.tau_indices]
        # maps a global tau index to its row in the subsampled outputs
        self._row = {int(t): k for k, t in enumerate(self.tau_indices)}
        self._cache: dict = {}
        self._sectors_cache: dict = {}

    # dispatch
    @property
    def n_states(self) -> int:
        return self.molecule.n_states

    @property
    def cache_bytes(self) -> int:
        return sum(v.nbytes for v in self._cache.values())

    def in_window(self, omega: float) -> bool:
        return self.molecule.window.contains(float(omega))

    def primitive_for(self, omega: float, sigma: str | None = None) -> Primitive | None:
        """The molecule primitive whose omega matches (relative 1e-9); sigma narrows the match when
        several share a frequency.
        """
        best = None
        for p in self.molecule.primitives:
            if abs(p.omega - omega) <= 1e-9 * max(1.0, abs(p.omega)):
                if sigma is None or p.sigma == sigma:
                    return p
                best = best or p
        return best

    def sectors(self, sigma: str, omega: float) -> tuple:
        """Independently propagated sectors at this drive: the blocks in-window, the RWA sectors at
        a primitive frequency.
        """
        omega = float(omega)
        if self.in_window(omega):
            return self.molecule.blocks
        if self.primitive_for(omega) is None:
            w = self.molecule.window
            raise ValueError(
                f"omega = {omega:.6g} rad/ms is neither in the window "
                f"[{w.omega_min:.6g}, {w.omega_max:.6g}] nor a primitive of {self.molecule.name!r}")
        key = (sigma, omega)
        sec = self._sectors_cache.get(key)
        if sec is None:
            sec = rwa_sectors(self.molecule, sigma, omega)
            self._sectors_cache[key] = sec
        return sec

    # columns
    def _compute_columns(self, sector: Block, sigma: str, omega: float) -> np.ndarray:
        return transfer_matrix(self.molecule, sector if sector.index < 0 else int(sector.index),
                               omega, sigma, self.tau_indices)

    def columns(self, sector: Block, sigma: str, omega: float) -> np.ndarray:
        """(len(tau_indices), 2 M, M) lumped transfer columns, cached per (sector, sigma, omega)."""
        key = (int(sector.index), tuple(sector.key), sigma, float(omega))
        t = self._cache.get(key)
        if t is None:
            t = np.asarray(self._compute_columns(sector, sigma, float(omega)), dtype=np.float64)
            self._cache[key] = t
        return t

    def exact_branches_all_tau(self, p_in, omega, sigma):
        """Exact (P0, P1) over tau_indices through the sectors of this drive."""
        p_in = np.asarray(p_in, dtype=np.float64)
        nt = self.tau_indices.size
        out0 = np.tile(p_in, (nt, 1))
        out1 = np.zeros((nt, p_in.size))
        for sec in self.sectors(sigma, omega):
            sub = p_in[sec.states]
            if sub.sum() <= 0.0:
                continue
            t = self.columns(sec, sigma, omega)                 # (nt, 2M, M)
            out = t @ sub
            m = sec.n_states
            out0[:, sec.states] = out[:, :m]
            out1[:, sec.states] = out[:, m:]
        return out0, out1

    def branches(self, p_in, omega, sigma, tau_index: int):
        """Single-duration convenience: (P0, P1) at global tau_index."""
        a, c = self.branches_all_tau(p_in, omega, sigma)
        r = self._row[int(tau_index)]
        return a[r], c[r]


class ExactEngine(_EngineBase):
    """Ground truth: exact block (or sector) propagators in numpy."""

    def branches_all_tau(self, p_in, omega, sigma):
        return self.exact_branches_all_tau(p_in, omega, sigma)


class TorchExactEngine(_EngineBase):
    """Exact propagation through qlsgym.physics.torch_ops."""

    def __init__(self, molecule: Molecule, device="cpu", tau_indices=None, chunk_bytes: float = 2.0e9):
        super().__init__(molecule, tau_indices)
        import torch
        self.device = torch.device(device)
        self.chunk_bytes = float(chunk_bytes)

    def _compute_columns(self, sector: Block, sigma: str, omega: float) -> np.ndarray:
        import torch
        from .torch_ops import transfer_columns
        t = transfer_columns(self.molecule, sector if sector.index < 0 else int(sector.index),
                             np.array([omega]), sigma, self.tau_indices, self.device,
                             torch.float64, self.chunk_bytes)
        return t[0].cpu().numpy()

    def branches_all_tau(self, p_in, omega, sigma):
        return self.exact_branches_all_tau(p_in, omega, sigma)

    def branches_batch(self, p_in, omegas, sigma):
        """(P0, P1), each (n_omega, len(tau_indices), n_states)."""
        import torch
        from .torch_ops import transfer_columns
        omegas = np.asarray(omegas, dtype=np.float64).reshape(-1)
        p_in = np.asarray(p_in, dtype=np.float64)
        if not all(self.in_window(w) for w in omegas):
            outs = [self.branches_all_tau(p_in, float(w), sigma) for w in omegas]
            return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])
        nt, n = self.tau_indices.size, p_in.size
        out0 = np.tile(p_in, (omegas.size, nt, 1))
        out1 = np.zeros((omegas.size, nt, n))
        for b in self.molecule.blocks:
            sub = p_in[b.states]
            if sub.sum() <= 0.0:
                continue
            t = transfer_columns(self.molecule, b.index, omegas, sigma, self.tau_indices,
                                 self.device, torch.float64, self.chunk_bytes)     # (W, nt, 2M, M)
            out = (t @ torch.as_tensor(sub, dtype=torch.float64, device=self.device)).cpu().numpy()
            m = b.n_states
            out0[:, :, b.states] = out[:, :, :m]
            out1[:, :, b.states] = out[:, :, m:]
        return out0, out1


# Counterfactual engines (ThF control_engines, referee response Sec. VI)


class BlockSubstituteEngine(ExactEngine):
    """Exact engine with in-window pulses replaced on selected blocks."""

    kind = "substitute"

    def __init__(self, molecule: Molecule, tau_indices=None, blocks=None, sigmas=("+",)):
        super().__init__(molecule, tau_indices)
        self.sub_blocks = ({b.index for b in molecule.blocks} if blocks is None
                           else {int(b) for b in blocks})
        self.sub_sigmas = tuple(sigmas)
        self.calls = {"substituted": 0, "exact_sigma_minus": 0, "exact_untrained": 0, "exact_thz": 0}

    def surrogate_fraction(self) -> float:
        tot = sum(self.calls.values())
        return self.calls["substituted"] / tot if tot else float("nan")

    def _substitute_block(self, b: int, sub: np.ndarray, omega: float) -> np.ndarray:
        """(len(tau_indices), 2 M) populations, summing to sub.sum."""
        raise NotImplementedError

    def branches_all_tau(self, p_in, omega, sigma):
        p_in = np.asarray(p_in, dtype=np.float64)
        omega = float(omega)
        if not self.in_window(omega):
            self.calls["exact_thz"] += 1
            return self.exact_branches_all_tau(p_in, omega, sigma)
        nt, n = self.tau_indices.size, p_in.size
        out0 = np.zeros((nt, n))
        out1 = np.zeros((nt, n))
        for b in self.molecule.blocks:
            sub = p_in[b.states]
            if float(sub.sum()) <= 0.0:
                continue
            m = b.n_states
            if sigma in self.sub_sigmas and b.index in self.sub_blocks:
                self.calls["substituted"] += 1
                out = self._substitute_block(b.index, sub, omega)
            else:
                self.calls["exact_sigma_minus" if sigma == "-" else "exact_untrained"] += 1
                out = self.columns(b, sigma, omega) @ sub
            out0[:, b.states] = out[:, :m]
            out1[:, b.states] = out[:, m:]
        total = out0.sum(1) + out1.sum(1)
        if not np.allclose(total, float(p_in.sum()), atol=1e-4):
            raise RuntimeError(f"{type(self).__name__} broke probability conservation: "
                               f"sum(p_out) in [{total.min():.4f}, {total.max():.4f}] vs "
                               f"sum(p_in) = {p_in.sum():.4f}")
        return out0, out1


class TrivialEngine(BlockSubstituteEngine):
    """p(tau) = p(0): the static baseline, in the planning loop."""

    kind = "trivial"

    def _substitute_block(self, b: int, sub: np.ndarray, omega: float) -> np.ndarray:
        m = sub.size
        out = np.zeros((self.tau_indices.size, 2 * m), dtype=np.float64)
        out[:, :m] = sub[None, :]
        return out


class NoisyExactEngine(BlockSubstituteEngine):
    """Exact propagation on the substituted blocks, plus calibrated error."""

    kind = "noisy"

    def __init__(self, molecule: Molecule, tau_indices=None, blocks=None, sigmas=("+",),
                 rel_l1: float = 0.0, purity_gamma: float = 0.0, seed: int = 0,
                 rel_l1_on=None, rel_l1_off=None, n_linewidths: float = 1.0,
                 halluc_nu1=None, branch_purity=None, defect_samples=None):
        super().__init__(molecule, tau_indices, blocks, sigmas)
        self.rel_l1 = float(rel_l1)
        self.rel_l1_on = self.rel_l1 if rel_l1_on is None else float(rel_l1_on)
        self.rel_l1_off = self.rel_l1 if rel_l1_off is None else float(rel_l1_off)
        self.purity_gamma = float(purity_gamma)
        self.seed = int(seed)
        self.n_linewidths = float(n_linewidths)
        self._field: dict = {}
        self._onres: dict = {}
        if halluc_nu1 is None:
            self.halluc_nu1 = {}
        elif isinstance(halluc_nu1, dict):
            self.halluc_nu1 = {int(k): float(v) for k, v in halluc_nu1.items()}
        else:
            self.halluc_nu1 = {int(b): float(halluc_nu1) for b in self.sub_blocks}
        for b, h in self.halluc_nu1.items():
            if not 0.0 <= h <= 1.0:
                raise ValueError(f"halluc_nu1[{b}] = {h} is not a fraction")
        self.defect_samples = ({int(k): np.asarray(v, dtype=np.float64) for k, v in defect_samples.items()}
                               if defect_samples else {})
        for b, a in self.defect_samples.items():
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"defect_samples[{b}] must be (N, 3) = (pi1, q0, q1); got {a.shape}")
        self.branch_purity = ({int(k): (float(v[0]), float(v[1])) for k, v in branch_purity.items()}
                              if branch_purity else {})
        for b, (q0, q1) in self.branch_purity.items():
            if not (0.0 <= q0 <= 1.0 and 0.0 <= q1 <= 1.0):
                raise ValueError(f"branch_purity[{b}] = {(q0, q1)} not in [0,1]")
        self._current_sigma = "+"

    def branches_all_tau(self, p_in, omega, sigma):
        self._current_sigma = sigma
        return super().branches_all_tau(p_in, omega, sigma)

    def rel_l1_at(self, b: int, omega: float, sigma: str = "+") -> float:
        key = (b, sigma, int(round(float(omega) * 1e6)))
        on = self._onres.get(key)
        if on is None:
            on = bool(is_on_resonance(self.molecule, np.array([float(omega)]), b, sigma, self.n_linewidths)[0])
            self._onres[key] = on
        return self.rel_l1_on if on else self.rel_l1_off

    def _noise_field(self, b: int, omega: float, shape, kind: str = "n") -> np.ndarray:
        """Deterministic field for (block, omega), keyed on the frequency rounded to 1e-6 rad/ms."""
        key = (b, int(round(float(omega) * 1e6)), kind)
        f = self._field.get(key)
        if f is None or f.shape != tuple(shape):
            rng = np.random.default_rng([self.seed, b, key[1], ord(kind)])
            f = rng.random(shape) if kind == "u" else rng.standard_normal(shape)
            self._field[key] = f
        return f

    def _substitute_block(self, b: int, sub: np.ndarray, omega: float) -> np.ndarray:
        m = sub.size
        mass = float(sub.sum())
        sigma = self._current_sigma
        out = self.columns(self.molecule.blocks[b], sigma, omega) @ sub      # (nt, 2M)

        h = self.halluc_nu1.get(b, 0.0)
        if h > 0.0:
            moved = h * out[:, :m]
            out[:, :m] -= moved
            out[:, m:] += moved

        if self.purity_gamma > 0.0:
            for lo, hi in ((0, m), (m, 2 * m)):
                half = out[:, lo:hi]
                s = half.sum(1, keepdims=True)
                sharp = half ** (1.0 + self.purity_gamma)
                z = sharp.sum(1, keepdims=True)
                out[:, lo:hi] = np.where(z > 0.0, sharp * (s / np.maximum(z, 1e-300)), half)

        samp = self.defect_samples.get(b)
        if samp is not None:
            u = self._noise_field(b, omega, (self.tau_indices.size,), kind="u")
            idx = np.clip((u * samp.shape[0]).astype(np.int64), 0, samp.shape[0] - 1)
            drawn = samp[idx]
            moved = drawn[:, :1] * out[:, :m]
            out[:, :m] -= moved
            out[:, m:] += moved
            qt = (drawn[:, 1], drawn[:, 2])
        else:
            qt = self.branch_purity.get(b)
        if qt is not None:
            for half, q_t in ((slice(0, m), qt[0]), (slice(m, 2 * m), qt[1])):
                blk = out[:, half]
                sm = blk.sum(1, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    q_old = np.where(sm[:, 0] > 0, blk.max(1) / np.maximum(sm[:, 0], 1e-300), 0.0)
                q_t = np.broadcast_to(np.asarray(q_t, dtype=np.float64), q_old.shape)
                t = np.clip((q_t - q_old) / np.maximum(1.0 - q_old, 1e-12), 0.0, 1.0)
                top = np.zeros_like(blk)
                top[np.arange(blk.shape[0]), blk.argmax(1)] = 1.0
                out[:, half] = (1.0 - t)[:, None] * blk + t[:, None] * top * sm

        eps = self.rel_l1_at(b, omega, sigma)
        if eps > 0.0:
            sd = eps * mass / (2 * m * np.sqrt(2.0 / np.pi))
            out = out + sd * self._noise_field(b, omega, out.shape)
            np.clip(out, 0.0, None, out=out)

        s = out.sum(1, keepdims=True)
        bad = (s <= 0.0).ravel()
        if bad.any():
            out[bad, :m] = sub[None, :]
            out[bad, m:] = 0.0
            s = out.sum(1, keepdims=True)
        return out * (mass / s)


# Dense tables


class TableEngine(_EngineBase):
    """Engine backed by pre-computed lumped transfer tables."""

    def __init__(self, molecule: Molecule, tables: dict, tau_indices=None):
        super().__init__(molecule, tau_indices)
        self.tables = {}
        for (sigma, omega), entry in tables.items():
            self.tables[(sigma, float(omega))] = self._normalise(entry)

    def _normalise(self, entry):
        out = []
        items = entry.items() if isinstance(entry, dict) else entry
        for k, t in items:
            states = self.molecule.blocks[int(k)].states if np.isscalar(k) else np.asarray(k, dtype=np.int64)
            t = np.asarray(t, dtype=np.float64)
            if t.shape != (self.tau_indices.size, 2 * states.size, states.size):
                raise ValueError(f"table for {k!r} has shape {t.shape}, expected "
                                 f"{(self.tau_indices.size, 2 * states.size, states.size)}")
            out.append((states, t))
        return out

    def lookup(self, sigma: str, omega: float):
        key = (sigma, float(omega))
        if key in self.tables:
            return self.tables[key]
        for (s, w), v in self.tables.items():
            if s == sigma and abs(w - omega) <= 1e-9 * max(1.0, abs(w)):
                return v
        raise KeyError(f"no table for sigma={sigma!r}, omega={omega!r}")

    def branches_all_tau(self, p_in, omega, sigma):
        p_in = np.asarray(p_in, dtype=np.float64)
        nt = self.tau_indices.size
        out0 = np.tile(p_in, (nt, 1))
        out1 = np.zeros((nt, p_in.size))
        for states, t in self.lookup(sigma, float(omega)):
            sub = p_in[states]
            if sub.sum() <= 0.0:
                continue
            out = t @ sub
            m = states.size
            out0[:, states] = out[:, :m]
            out1[:, states] = out[:, m:]
        return out0, out1

    @classmethod
    def from_engine(cls, engine: _EngineBase, pulses) -> "TableEngine":
        """Tabulate pulses (iterable of (sigma, omega)) from an exact engine's sectors and columns."""
        tables = {}
        for sigma, omega in pulses:
            tables[(sigma, float(omega))] = [(sec.states, engine.columns(sec, sigma, float(omega)))
                                             for sec in engine.sectors(sigma, float(omega))]
        return cls(engine.molecule, tables, engine.tau_indices)
