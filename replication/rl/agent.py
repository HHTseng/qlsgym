"""The qMDP-DQN agent of arXiv:2608.03702 Appendix C.1, on the qlsgym env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentConfig:

    hidden: int = 128               # "Q-network had hidden dimension 128"
    n_hidden_layers: int = 2        # Three-layer net, 128 nodes/layer
    batch_size: int = 256           # "replay batches of size 256"
    memory: int = 250_000           # "replay memory size 2.5e5"
    gamma: float = 1.0              # "discount factor gamma = 1"
    eps_start: float = 1.0
    eps_end: float = 0.025
    eps_decay_steps: float = 7.2e4  # "decayed ... over 7.2e4 environment steps"
    eps_schedule: str = "linear"
    tau_rl: float = 1.0e-4          # Table 3 config 3: target-network soft update
    eta_rl: float = 5.0e-4          # Table 3 config 3: Adam learning rate
    grad_clip: float = 100.0
    snapshot_every: int = 100       # "snapshots every 100 episodes"
    snapshot_rollouts: int = 200    # "each evaluated with 200 Monte-Carlo rollouts"
    updates_per_step: int = 1
    bootstrap_on_truncation: bool = False
    double_q: bool = False
    seed: int = 0

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def make_qnet(n_in: int, n_out: int, cfg: AgentConfig):
    """Fully connected Q-network, n_in -> 128 -> 128 -> n_out"""
    import torch.nn as nn

    layers: list = [nn.Linear(n_in, cfg.hidden), nn.ReLU()]
    for _ in range(cfg.n_hidden_layers - 1):
        layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU()]
    layers += [nn.Linear(cfg.hidden, n_out)]
    return nn.Sequential(*layers)


def epsilon(step: int, cfg: AgentConfig) -> float:
    if cfg.eps_schedule == "linear":
        f = min(1.0, step / cfg.eps_decay_steps)
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * f
    if cfg.eps_schedule == "exp":
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-step / cfg.eps_decay_steps)
    if cfg.eps_schedule == "exp_reach":
        f = min(1.0, step / cfg.eps_decay_steps)
        return cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** f
    raise ValueError(f"unknown eps_schedule {cfg.eps_schedule!r}")


def untrained_action_rows(out_weight, init_weight, rtol: float = 1e-3):
    """Mask of Q-head output rows that never received a gradient."""
    num = (out_weight * init_weight).sum(-1)
    den = (init_weight * init_weight).sum(-1)
    c = num / den.clamp_min(1e-30)
    resid = (out_weight - c.unsqueeze(-1) * init_weight).norm(dim=-1)
    return resid <= rtol * init_weight.norm(dim=-1)


class QNetPolicy:
    stateful = False

    def __init__(self, net, n_actions: int, device: str = "cpu", eps: float = 0.0):
        self.net = net
        self.n_actions = int(n_actions)
        self.device = device
        self.eps = float(eps)
        net.eval()

    def q_values(self, beliefs) -> np.ndarray:
        import torch

        x = torch.as_tensor(np.asarray(beliefs, dtype=np.float32), device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.net(x).cpu().numpy()

    def act_batch(self, beliefs, t, rng):
        a = self.q_values(beliefs).argmax(-1).astype(np.int64)
        if self.eps > 0.0:
            explore = rng.random(a.shape[0]) < self.eps
            if explore.any():
                a[explore] = rng.integers(self.n_actions, size=int(explore.sum()))
        return a

    def act(self, belief, t, rng):
        return int(self.act_batch(np.asarray(belief)[None, :], t, rng)[0])

    def __repr__(self) -> str:
        return f"QNetPolicy(n_actions={self.n_actions}, eps={self.eps}, device={self.device})"
