"""qlsgym.env.env -- the batched belief-MDP purification environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..spec import Molecule
from .actions import ActionLibrary
from .cache import ActionTables

KB_OVER_HBAR = 2.0 * np.pi * 2.0836619123e7   # (rad/ms) / K, as in both source packages


@dataclass(frozen=True)
class EnvConfig:
    """Stopping rule, budget and reward."""

    p_target: float | None = None
    max_pulses: int | None = None
    temperature_k: float | None = None
    # overlap-penalty weight rho (r_o of Ref. [29]); 0 disables
    rho: float = 1.0
    # penalise when cos(p_t, p_{t+1}) > threshold; None -> 1 - 1/n_states
    overlap_threshold: float | None = None
    # 'indicator' (-rho once crossed) or 'proportional' (-rho * cos)
    penalty_mode: str = "indicator"
    step_reward: float = -1.0
    # branches below this probability are treated as unreachable (float32 noise)
    min_branch_prob: float = 1e-9

    def resolve(self, molecule: Molecule) -> "EnvConfig":
        t = molecule.task
        return replace(
            self,
            p_target=t.p_target if self.p_target is None else self.p_target,
            max_pulses=t.max_pulses if self.max_pulses is None else self.max_pulses,
            temperature_k=t.temperature_k if self.temperature_k is None else self.temperature_k,
        )


@dataclass
class Transition:
    """One environment step for B trajectories (all tensors on the env device)."""

    belief: "torch.Tensor"       # (B, n) float64
    reward: "torch.Tensor"       # (B,)   float64
    done: "torch.Tensor"         # (B,)   bool
    truncated: "torch.Tensor"    # (B,)   bool
    outcome: "torch.Tensor"      # (B,)   long, 0 or 1
    s0: "torch.Tensor"           # (B, n) posterior given nu = 0
    s1: "torch.Tensor"           # (B, n) posterior given nu >= 1
    pi0: "torch.Tensor"          # (B,)   probability of nu = 0
    pi1: "torch.Tensor"          # (B,)   probability of nu >= 1
    r0: "torch.Tensor"
    r1: "torch.Tensor"
    done0: "torch.Tensor"
    done1: "torch.Tensor"

    @property
    def terminated(self):
        return self.done


def boltzmann_belief(molecule: Molecule, temperature_k: float | None = None) -> np.ndarray:
    """Thermal initial belief (paper Eq. 5)."""
    T = molecule.task.temperature_k if temperature_k is None else float(temperature_k)
    try:
        from ..physics.thermal import boltzmann
        return np.asarray(boltzmann(molecule, temperature_k=T), dtype=np.float64)
    except ImportError:
        e = molecule.system.energies - molecule.system.energies.min()
        w = np.exp(-e / (KB_OVER_HBAR * T))
        return w / w.sum()


class PurificationEnv:
    """Batched, table-driven purification cycle (module docstring)."""

    def __init__(
        self,
        molecule: Molecule,
        library: ActionLibrary,
        tables: ActionTables,
        cfg: EnvConfig | None = None,
        device: str = "cpu",
        batch: int = 1,
        p_init: np.ndarray | None = None,
    ):
        import torch

        self.torch = torch
        self.molecule = molecule
        self.library = library
        self.cfg = (cfg or EnvConfig()).resolve(molecule)
        self.device = torch.device(device)
        self.batch = int(batch)
        self.n_states = molecule.n_states
        self.n_actions = library.n_actions
        self.n_grid = library.n_grid
        tables.check(molecule, library)
        self.tables = tables

        tdt = torch.float64 if np.dtype(tables.dtype) == np.float64 else torch.float32
        self._tdt = tdt
        # grid tables: per block (n_grid, 2M, M); index_select on the action axis
        self._T = [torch.as_tensor(np.ascontiguousarray(b), dtype=tdt, device=self.device) for b in tables.blocks]
        self._idx = [torch.as_tensor(b.states, dtype=torch.long, device=self.device) for b in molecule.blocks]
        self._M = [b.n_states for b in molecule.blocks]
        unblocked = np.where(molecule.system.block_of_state < 0)[0]
        self._unblocked = torch.as_tensor(unblocked, dtype=torch.long, device=self.device)
        # primitive tables: per primitive (2S, S) over its union sector
        self._P = [(torch.as_tensor(p.states, dtype=torch.long, device=self.device),
                    torch.as_tensor(np.ascontiguousarray(p.table), dtype=tdt, device=self.device))
                   for p in tables.primitives]

        if p_init is None:
            p_init = boltzmann_belief(molecule, self.cfg.temperature_k)
        p_init = np.asarray(p_init, dtype=np.float64)
        if p_init.shape != (self.n_states,) or abs(p_init.sum() - 1.0) > 1e-8 or (p_init < 0).any():
            raise ValueError("p_init must be a probability vector of length n_states")
        self._p_init = torch.as_tensor(p_init, dtype=torch.float64, device=self.device)
        self.thresh = (self.cfg.overlap_threshold if self.cfg.overlap_threshold is not None
                       else 1.0 - 1.0 / self.n_states)
        if self.cfg.penalty_mode not in ("indicator", "proportional"):
            raise ValueError(f"unknown penalty_mode {self.cfg.penalty_mode!r}")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0)
        self.state = self._p_init.expand(self.batch, -1).clone()
        self.steps = torch.zeros(self.batch, dtype=torch.long, device=self.device)

    # construction helpers

    @property
    def p_init(self) -> np.ndarray:
        return self._p_init.cpu().numpy().copy()

    def clone(self, batch: int) -> "PurificationEnv":
        """A second view on the same tables with a different batch size."""
        import copy

        other = copy.copy(self)
        other.batch = int(batch)
        other.generator = self.torch.Generator(device=self.device)
        other.generator.manual_seed(0)
        other.reset()
        return other

    # physics

    def apply(self, belief, actions):
        """Unnormalised branches (P0, P1), each (B, n) float64, for belief (B, n) and integer
        actions (B,).
        """
        torch = self.torch
        belief = torch.as_tensor(belief, dtype=torch.float64, device=self.device)
        squeeze = belief.dim() == 1
        if squeeze:
            belief = belief.unsqueeze(0)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        if actions.numel() == 1 and belief.shape[0] > 1:
            actions = actions.expand(belief.shape[0])
        if actions.shape[0] != belief.shape[0]:
            raise ValueError("actions must have one entry per belief row")
        if bool((actions < 0).any()) or bool((actions >= self.n_actions).any()):
            raise IndexError("action index out of range")
        B = belief.shape[0]
        is_prim = actions >= self.n_grid
        a_grid = torch.where(is_prim, torch.zeros_like(actions), actions)
        out0 = torch.zeros_like(belief)
        out1 = torch.zeros_like(belief)
        st = belief.to(self._tdt)
        for k in range(len(self._T)):
            idx, m = self._idx[k], self._M[k]
            mat = self._T[k].index_select(0, a_grid)                    # (B, 2M, M)
            res = torch.bmm(mat, st[:, idx].unsqueeze(-1)).squeeze(-1).to(torch.float64)
            out0[:, idx] = res[:, :m]
            out1[:, idx] = res[:, m:]
        if self._unblocked.numel():
            out0[:, self._unblocked] = belief[:, self._unblocked]
        if bool(is_prim.any()):
            # primitive rows: identity outside the sector, (2S, S) table inside
            out0 = torch.where(is_prim.unsqueeze(-1), belief, out0)
            out1 = torch.where(is_prim.unsqueeze(-1), torch.zeros_like(out1), out1)
            for k, (states, T) in enumerate(self._P):
                rows = torch.where(actions == self.n_grid + k)[0]
                if rows.numel() == 0:
                    continue
                S = states.numel()
                sub = st[rows][:, states]                                 # (r, S)
                res = (sub @ T.transpose(0, 1)).to(torch.float64)          # (r, 2S)
                r0 = out0[rows]; r0[:, states] = res[:, :S]; out0[rows] = r0
                r1 = out1[rows]; r1[:, states] = res[:, S:]; out1[rows] = r1
        out0 = out0.clamp_min_(0.0)
        out1 = out1.clamp_min_(0.0)
        if squeeze:
            return out0[0], out1[0]
        return out0, out1

    def branch_outcomes(self, belief, actions):
        """Everything one qMDP transition needs, for both outcomes (Eqs. 8-10)."""
        torch = self.torch
        belief = torch.as_tensor(belief, dtype=torch.float64, device=self.device)
        p0, p1 = self.apply(belief, actions)
        m0, m1 = p0.sum(-1), p1.sum(-1)
        tot = (m0 + m1).clamp_min(1e-300)
        pi0, pi1 = m0 / tot, m1 / tot
        s0 = p0 / m0.clamp_min(self.cfg.min_branch_prob).unsqueeze(-1)
        s1 = p1 / m1.clamp_min(self.cfg.min_branch_prob).unsqueeze(-1)
        ok0 = pi0 > self.cfg.min_branch_prob
        ok1 = pi1 > self.cfg.min_branch_prob
        s0 = torch.where(ok0.unsqueeze(-1), s0, belief)
        s1 = torch.where(ok1.unsqueeze(-1), s1, belief)
        r0 = self._reward(belief, s0)
        r1 = self._reward(belief, s1)
        done0 = (s0.max(-1).values >= self.cfg.p_target) & ok0
        done1 = (s1.max(-1).values >= self.cfg.p_target) & ok1
        return s0, s1, pi0, pi1, r0, r1, done0, done1

    def _reward(self, s_prev, s_next):
        num = (s_prev * s_next).sum(-1)
        den = s_prev.norm(dim=-1) * s_next.norm(dim=-1)
        cos = num / den.clamp_min(1e-300)
        if self.cfg.penalty_mode == "indicator":
            pen = (cos > self.thresh).to(s_prev.dtype) * self.cfg.rho
        else:
            pen = cos.clamp_min(0.0) * self.cfg.rho
        return self.cfg.step_reward - pen

    def is_done(self, belief) -> "torch.Tensor":
        belief = self.torch.as_tensor(belief, dtype=self.torch.float64, device=self.device)
        return belief.max(-1).values >= self.cfg.p_target

    # driver

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    def reset(self, seed: int | None = None, batch: int | None = None):
        """Return the (B, n) initial belief; reseed the sampler if asked."""
        if batch is not None:
            self.batch = int(batch)
        if seed is not None:
            self.seed(seed)
        self.state = self._p_init.expand(self.batch, -1).clone()
        self.steps = self.torch.zeros(self.batch, dtype=self.torch.long, device=self.device)
        return self.state

    def reset_rows(self, mask) -> None:
        """Put the trajectories selected by the boolean mask (B,) back in the initial belief with a
        zero pulse count, leaving the others alone.
        """
        torch = self.torch
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        if mask.shape[0] != self.batch:
            raise ValueError("mask must have one entry per trajectory")
        if bool(mask.any()):
            self.state = torch.where(mask.unsqueeze(-1), self._p_init.expand(self.batch, -1), self.state)
            self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)

    def step(self, actions) -> Transition:
        """Sample one motional outcome per trajectory and advance."""
        torch = self.torch
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        if actions.numel() == 1 and self.batch > 1:
            actions = actions.expand(self.batch)
        s0, s1, pi0, pi1, r0, r1, d0, d1 = self.branch_outcomes(self.state, actions)
        u = torch.rand(self.batch, device=self.device, dtype=torch.float64, generator=self.generator)
        take1 = u < pi1
        self.state = torch.where(take1.unsqueeze(-1), s1, s0)
        self.steps = self.steps + 1
        reward = torch.where(take1, r1, r0)
        done = torch.where(take1, d1, d0)
        trunc = (self.steps >= self.cfg.max_pulses) & ~done
        return Transition(self.state, reward, done, trunc, take1.to(torch.long),
                          s0, s1, pi0, pi1, r0, r1, d0, d1)

    def __repr__(self) -> str:
        return (f"PurificationEnv({self.molecule.name!r}, n_states={self.n_states}, "
                f"n_actions={self.n_actions}, batch={self.batch}, device={self.device}, "
                f"p_target={self.cfg.p_target}, max_pulses={self.cfg.max_pulses})")
