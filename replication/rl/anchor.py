from __future__ import annotations

import numpy as np


def one_step_yield(env, belief=None, chunk: int = 2048):
    """(Pi_1, purity_1) for every action applied to belief."""
    torch = env.torch
    s = torch.as_tensor(np.asarray(env.p_init if belief is None else belief, dtype=np.float64),
                        device=env.device)
    n = env.n_actions
    pi1 = torch.zeros(n, dtype=torch.float64, device=env.device)
    pur1 = torch.zeros(n, dtype=torch.float64, device=env.device)
    for lo in range(0, n, chunk):
        a = torch.arange(lo, min(lo + chunk, n), device=env.device)
        rep = s.reshape(1, -1).expand(a.numel(), -1)
        _, p1 = env.apply(rep, a)
        m = p1.sum(-1)
        pi1[lo:lo + a.numel()] = m
        pur1[lo:lo + a.numel()] = p1.max(-1).values / m.clamp_min(1e-300)
    return pi1.cpu().numpy(), pur1.cpu().numpy()


def best_fixed_action(env, belief=None) -> dict:
    pi1, pur1 = one_step_yield(env, belief)
    finishing = pi1 * (pur1 >= env.cfg.p_target)
    best = int(np.argmax(finishing)) if finishing.max() > 0 else int(np.argmax(pi1))
    act = env.library.decode(best)
    return {
        "action": best,
        "pi1": float(pi1[best]),
        "purity_nu1": float(pur1[best]),
        "selected_by": "pi1*[purity>=p_target]" if finishing.max() > 0 else "pi1 (none can finish in one shot)",
        "n_actions_finishing_in_one_shot": int((finishing > 0).sum()),
        "is_primitive": bool(env.library.is_primitive(best)),
        "decoded": {"sigma": act.sigma, "omega_over_2pi_khz": float(act.omega / (2.0 * np.pi)),
                    "tau_index": int(act.tau_index), "primitive": int(act.primitive)},
    }


class FixedActionPolicy:

    stateful = False

    def __init__(self, action: int, n_actions: int):
        if not 0 <= int(action) < int(n_actions):
            raise IndexError(f"action {action} out of range [0, {n_actions})")
        self.action = int(action)
        self.n_actions = int(n_actions)

    def act(self, belief, t, rng):
        return self.action

    def act_batch(self, beliefs, t, rng):
        return np.full(np.shape(beliefs)[0], self.action, dtype=np.int64)

    def __repr__(self) -> str:
        return f"FixedActionPolicy(action={self.action})"
