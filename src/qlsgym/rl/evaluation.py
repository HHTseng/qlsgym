"""Memory-bounded evaluation with private state for each physics-policy row."""

import copy
import time
from collections import Counter

import numpy as np


def evaluate_policy(environment, policy, episodes, seed, batch_size=128):
    """Failed episodes score H, including policies that stop with no action.

    Chunk seeds depend only on the evaluation seed and chunk index, never the
    training seed/agent. Keep batch_size fixed across a comparison. Stateful
    policies are shallow-copied and reset per row; their immutable physics
    tables are shared, while trajectory memory remains private.
    """
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    started = time.perf_counter()
    H = environment.cfg.max_pulses
    lengths, actual_lengths, successes = [], [], []
    reasons, targets = Counter(), Counter()
    for index, lo in enumerate(range(0, episodes, batch_size)):
        B = min(batch_size, episodes - lo)
        chunk_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        rng = np.random.default_rng(chunk_seed)
        env = environment.clone(B)
        state = env.reset(seed=chunk_seed, batch=B)
        row_policies = None
        if getattr(policy, "stateful", False):
            row_policies = [copy.copy(policy) for _ in range(B)]
            for child in row_policies:
                child.reset()
        elif hasattr(policy, "reset"):
            policy.reset()
        ok = env.is_done(state).cpu().numpy()
        alive = ~ok
        used = np.full(B, H, dtype=np.int64)
        used[ok] = 0
        no_action = np.zeros(B, dtype=bool)
        target = np.full(B, -1, dtype=np.int64)
        target[ok] = state.argmax(-1).cpu().numpy()[ok]
        for t in range(H):
            if not alive.any():
                break
            beliefs = state.detach().cpu().numpy()
            if row_policies is None and hasattr(policy, "act_batch"):
                actions = np.asarray(policy.act_batch(beliefs, t, rng), dtype=np.int64)
            else:
                actions = np.zeros(B, dtype=np.int64)
                for row in np.flatnonzero(alive):
                    controller = policy if row_policies is None else row_policies[row]
                    pick = controller.act(beliefs[row], t, rng)
                    if pick is None:
                        no_action[row] = True
                        alive[row] = False
                        used[row] = t
                    else:
                        actions[row] = int(pick) if isinstance(pick, (int, np.integer)) else env.library.encode(pick)
            if not alive.any():
                break
            transition = env.step(actions)
            state = transition.belief
            just = transition.done.cpu().numpy() & alive
            used[just] = t + 1
            target[just] = state.argmax(-1).cpu().numpy()[just]
            ok |= just
            alive &= ~just
        actual_lengths.extend(used.tolist())
        lengths.extend(np.where(ok, used, H).tolist())
        successes.extend(ok.tolist())
        reasons.update(success=int(ok.sum()), no_action=int(no_action.sum()),
                       max_pulses=int((~ok & ~no_action).sum()))
        targets.update(target[ok].tolist())
    length = np.asarray(lengths, dtype=float)
    success = np.asarray(successes, dtype=bool)
    rate = float(success.mean())
    tail = max(1, int(np.ceil(0.1 * episodes)))
    return {
        "n_rollouts": episodes, "max_pulses": H,
        "p_target": environment.cfg.p_target,
        "success_fraction": rate,
        "success_err": float(np.sqrt(rate * (1 - rate) / episodes)),
        "unfinished_fraction": 1 - rate,
        "mean_pulses": float(length.mean()),
        "average_actions": float(length.mean()),
        "median_actions": float(np.median(length)),
        "p85_actions": float(np.quantile(length, 0.85)),
        "cvar10_actions": float(np.sort(length)[-tail:].mean()),
        "mean_pulses_successful": float(length[success].mean()) if success.any() else None,
        "lengths": lengths, "actual_lengths": actual_lengths, "successes": successes,
        "outcomes": {key: count / episodes for key, count in reasons.items()},
        "target_states": targets.most_common(8),
        "evaluation_batch": batch_size, "seed": seed,
        "seconds": time.perf_counter() - started,
    }
