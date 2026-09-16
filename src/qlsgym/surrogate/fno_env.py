"""An env whose covered blocks are propagated by the FNO on the belief the agent holds."""

from __future__ import annotations

import numpy as np

from ..env.actions import ActionLibrary
from ..env.cache import ActionTables
from ..env.env import EnvConfig, PurificationEnv
from ..spec import Molecule
from .fno_engine import FnoEngine


class FnoEnv(PurificationEnv):
    """PurificationEnv with the surrogate in the step."""

    def __init__(self, molecule: Molecule, library: ActionLibrary, tables: ActionTables,
                 engine: FnoEngine, cfg: EnvConfig | None = None, device: str = "cpu",
                 batch: int = 1, p_init: np.ndarray | None = None):
        super().__init__(molecule, library, tables, cfg=cfg, device=device, batch=batch, p_init=p_init)
        if engine.molecule.fingerprint() != molecule.fingerprint():
            raise ValueError("engine and env are not the same molecule")
        eng_tau = np.asarray(engine.tau_indices, dtype=np.int64)
        missing = sorted(set(library.tau_indices.tolist()) - set(eng_tau.tolist()))
        if missing:
            raise ValueError(f"the engine lacks tau indices {missing[:5]} the library uses; build it with "
                             f"qlsgym.benchmark.protocol.library_taus(library)")
        self.engine = engine
        pos = {int(t): i for i, t in enumerate(eng_tau)}
        self._row_tau = np.asarray([pos[int(t)] for t in library.tau_indices], dtype=np.int64)
        self._row_sigma = np.asarray(library.sigmas, dtype=object)
        self._row_omega = np.asarray(library.omegas, dtype=np.float64)
        # (block, sigma) the surrogate answers for, in block order
        self._covered = sorted(engine.trained)
        if not self._covered:
            raise ValueError("this manifest covers no (block, sigma); there is nothing to substitute")

    def substituted_fraction(self) -> float:
        """Share of this env's block propagations that went through the network."""
        return self.engine.surrogate_fraction()

    def apply(self, belief, actions):
        """Exact everywhere, then the covered blocks replaced by the surrogate."""
        torch = self.torch
        out0, out1 = super().apply(belief, actions)
        belief_t = torch.as_tensor(belief, dtype=torch.float64, device=self.device)
        squeeze = belief_t.dim() == 1
        if squeeze:
            belief_t, out0, out1 = belief_t.unsqueeze(0), out0.unsqueeze(0), out1.unsqueeze(0)
        acts = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        if acts.numel() == 1 and belief_t.shape[0] > 1:
            acts = acts.expand(belief_t.shape[0])
        a = acts.cpu().numpy()
        is_grid = a < self.n_grid
        if is_grid.any():
            p = belief_t.detach().cpu().numpy()
            sig = np.where(is_grid, self._row_sigma[np.clip(a, 0, self.n_grid - 1)], "")
            for b_index, sigma in self._covered:
                rows = np.where(is_grid & (sig == sigma))[0]
                if rows.size == 0:
                    continue
                states = self.molecule.blocks[b_index].states
                m = states.size
                res = self.engine.block_branches(b_index, sigma, p[np.ix_(rows, states)],
                                                 self._row_omega[a[rows]])
                take = res[np.arange(rows.size), self._row_tau[a[rows]]]        # (r, 2M)
                idx_r = torch.as_tensor(rows, dtype=torch.long, device=self.device)[:, None]
                idx_s = torch.as_tensor(states, dtype=torch.long, device=self.device)[None, :]
                out0[idx_r, idx_s] = torch.as_tensor(np.ascontiguousarray(take[:, :m]),
                                                     dtype=torch.float64, device=self.device)
                out1[idx_r, idx_s] = torch.as_tensor(np.ascontiguousarray(take[:, m:]),
                                                     dtype=torch.float64, device=self.device)
            out0.clamp_min_(0.0)
            out1.clamp_min_(0.0)
        if squeeze:
            return out0[0], out1[0]
        return out0, out1

    def __repr__(self) -> str:
        return (f"FnoEnv({self.molecule.name}, {self.n_states} states, {self.n_actions} actions, "
                f"batch={self.batch}, covered={self._covered})")
