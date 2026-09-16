"""qlsgym.env.gym -- a thin single-trajectory Gymnasium view of PurificationEnv."""

from __future__ import annotations

import numpy as np

from ..spec import Molecule
from .actions import ActionLibrary
from .cache import ActionTables
from .env import EnvConfig, PurificationEnv

try:
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    HAS_GYMNASIUM = True
    _Base = _gym.Env
except ImportError:  # pragma: no cover - depends on the install
    _gym = None
    _spaces = None
    HAS_GYMNASIUM = False
    _Base = object


class QLSGymEnv(_Base):
    """Gymnasium wrapper; one trajectory; float64 observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        molecule: Molecule,
        library: ActionLibrary,
        tables: ActionTables,
        cfg: EnvConfig | None = None,
        seed: int | None = None,
        device: str = "cpu",
        p_init: np.ndarray | None = None,
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is not installed; `pip install gymnasium` (or qlsgym[gym]) "
                              "or use qlsgym.env.PurificationEnv directly")
        super().__init__()
        self.core = PurificationEnv(molecule, library, tables, cfg, device=device, batch=1, p_init=p_init)
        self.molecule = molecule
        self.library = library
        n = molecule.n_states
        self.observation_space = _spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float64)
        self.action_space = _spaces.Discrete(library.n_actions)
        self._seed = seed
        self._t = 0
        if seed is not None:
            self.core.seed(seed)

    # Gymnasium API

    def _obs(self):
        return self.core.state[0].detach().cpu().numpy().astype(np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.core.reset(seed=seed, batch=1)
        self._t = 0
        return self._obs(), {"pulses": 0, "p_max": float(self.core.state[0].max())}

    def step(self, action):
        a = int(action)
        if not 0 <= a < self.library.n_actions:
            raise ValueError(f"action {a} outside Discrete({self.library.n_actions})")
        tr = self.core.step(np.array([a]))
        self._t += 1
        obs = self._obs()
        cpu = lambda x: x[0].detach().cpu().numpy()  # noqa: E731
        info = {
            "outcome": int(tr.outcome[0]),
            "s0": cpu(tr.s0), "s1": cpu(tr.s1),
            "pi0": float(tr.pi0[0]), "pi1": float(tr.pi1[0]),
            "r0": float(tr.r0[0]), "r1": float(tr.r1[0]),
            "done0": bool(tr.done0[0]), "done1": bool(tr.done1[0]),
            "pulses": self._t, "p_max": float(obs.max()),
            "action": self.library.decode(a),
        }
        return obs, float(tr.reward[0]), bool(tr.done[0]), bool(tr.truncated[0]), info

    def render(self):  # pragma: no cover - nothing to draw
        return None

    def close(self):
        return None
