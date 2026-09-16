"""qlsgym.policies.rollout -- Monte-Carlo evaluation of a policy."""

from __future__ import annotations

import collections
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..env.actions import ActionLibrary
from ..env.env import PurificationEnv, boltzmann_belief
from ..spec import Action, TauBatchedEngine


@dataclass
class RolloutResult:
    n_rollouts: int
    max_pulses: int
    p_target: float
    success_fraction: float
    success_err: float
    mean_pulses: float
    mean_pulses_successful: float
    lengths: list                      # pulses used per rollout (max_pulses if failed)
    successes: list                    # bool per rollout
    outcomes: dict                     # {"success": .., "max_pulses": .., "no_action": ..} fractions
    target_states: list                # [[level label, count], ...] most common first
    engine_calls: int = 0              # branches_all_tau calls made to advance the state
    policy_engine_calls: int | None = None   # engine drives evaluated by the policy, if it counts
    seconds: float = 0.0
    trajectories: list | None = None   # per rollout [(action, outcome), ...] when record=True
    policy: str = ""
    engine: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("trajectories")
        if not math.isfinite(d["mean_pulses_successful"]):
            d["mean_pulses_successful"] = None
        return d

    def summary(self) -> str:
        return (f"{self.policy:>22s}: success {self.success_fraction:.3f} +- {self.success_err:.3f}, "
                f"mean pulses {self.mean_pulses:.1f} (successful {self.mean_pulses_successful:.1f}), "
                f"n={self.n_rollouts}, {self.seconds:.1f}s")


def _finish(lengths, ok, reasons, targets, molecule, max_pulses, p_target, t0, calls, policy, engine,
            traj, n_rollouts):
    ok = np.asarray(ok, dtype=bool)
    ln = np.asarray(lengths, dtype=float)
    conv = float(ok.mean()) if ok.size else 0.0
    labels = molecule.system.levels
    return RolloutResult(
        n_rollouts=int(n_rollouts), max_pulses=int(max_pulses), p_target=float(p_target),
        success_fraction=conv,
        success_err=float(math.sqrt(max(conv * (1 - conv), 0.0) / max(n_rollouts, 1))),
        mean_pulses=float(ln.mean()) if ln.size else float("nan"),
        mean_pulses_successful=float(ln[ok].mean()) if ok.any() else float("nan"),
        lengths=[int(x) for x in lengths], successes=[bool(x) for x in ok],
        outcomes={k: v / n_rollouts for k, v in reasons.items()},
        target_states=[[list(labels[n]) if n < len(labels) else n, c] for n, c in targets.most_common(8)],
        engine_calls=int(calls),
        policy_engine_calls=getattr(policy, "calls", None),
        seconds=time.time() - t0, trajectories=traj,
        policy=type(policy).__name__, engine=engine,
    )


def _as_action(pick, library: ActionLibrary) -> Action:
    if isinstance(pick, Action):
        return pick
    return library.decode(int(pick))


def rollout(env_or_engine, policy, n_rollouts: int = 100, seed: int = 0, max_pulses: int | None = None,
            p_target: float | None = None, library: ActionLibrary | None = None,
            p_init: np.ndarray | None = None, record: bool = False, progress: bool = False) -> RolloutResult:
    """Monte-Carlo evaluation (module docstring)."""
    if isinstance(env_or_engine, PurificationEnv):
        return _rollout_env(env_or_engine, policy, n_rollouts, seed, max_pulses, p_target, record, progress)
    if library is None:
        raise ValueError("rollout against an engine needs the ActionLibrary the policy indexes into")
    return _rollout_engine(env_or_engine, policy, library, n_rollouts, seed, max_pulses, p_target, p_init,
                           record, progress)


# engine path


def _rollout_engine(engine: TauBatchedEngine, policy, library: ActionLibrary, n_rollouts, seed, max_pulses,
                    p_target, p_init, record, progress) -> RolloutResult:
    mol = library.molecule
    max_pulses = mol.task.max_pulses if max_pulses is None else int(max_pulses)
    p_target = mol.task.p_target if p_target is None else float(p_target)
    p_init = boltzmann_belief(mol) if p_init is None else np.asarray(p_init, dtype=np.float64)
    rows = {int(t): i for i, t in enumerate(np.asarray(engine.tau_indices))}
    rng = np.random.default_rng(seed)
    t0 = time.time()
    ok, lengths, reasons, targets, calls = [], [], collections.Counter(), collections.Counter(), 0
    traj = [] if record else None
    for r in range(n_rollouts):
        p = p_init.copy()
        if hasattr(policy, "reset"):
            policy.reset()
        k = 0
        steps = []
        while True:
            if p.max() >= p_target:
                ok.append(True); reasons["success"] += 1; targets[int(np.argmax(p))] += 1
                break
            if k >= max_pulses:
                ok.append(False); reasons["max_pulses"] += 1
                break
            pick = policy.act(p, k, rng)
            if pick is None:
                ok.append(False); reasons["no_action"] += 1
                break
            act = _as_action(pick, library)
            ti = act.tau_index if act.tau_index >= 0 else library.primitive_tau_index(act.primitive)
            if ti not in rows:
                raise ValueError(f"engine.tau_indices lacks tau index {ti} chosen by the policy")
            a, c = engine.branches_all_tau(p, float(act.omega), act.sigma)
            calls += 1
            p0, p1 = np.asarray(a)[rows[ti]], np.asarray(c)[rows[ti]]
            m0, m1 = float(p0.sum()), float(p1.sum())
            if rng.random() < m1 / max(m0 + m1, 1e-300):
                p = p1 / m1
                out = 1
            else:
                p = p0 / m0
                out = 0
            if record:
                steps.append((act, out))
            k += 1
        lengths.append(k)
        if record:
            traj.append(steps)
        if progress and (r + 1) % 25 == 0:
            print(f"  {r + 1}/{n_rollouts}: success {np.mean(ok):.3f}, mean pulses {np.mean(lengths):.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return _finish(lengths, ok, reasons, targets, mol, max_pulses, p_target, t0, calls, policy,
                   type(engine).__name__, traj, n_rollouts)


# environment path


def _rollout_env(env: PurificationEnv, policy, n_rollouts, seed, max_pulses, p_target, record, progress):
    max_pulses = env.cfg.max_pulses if max_pulses is None else int(max_pulses)
    p_target = env.cfg.p_target if p_target is None else float(p_target)
    if p_target != env.cfg.p_target or max_pulses != env.cfg.max_pulses:
        from dataclasses import replace
        env = env.clone(env.batch)
        env.cfg = replace(env.cfg, p_target=p_target, max_pulses=max_pulses)
    stateful = bool(getattr(policy, "stateful", False))
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if stateful:
        # one trajectory at a time so per-trajectory policy memory is private
        env.reset(seed=seed, batch=1)
        chunks = [1] * n_rollouts
    else:
        env.reset(seed=seed, batch=n_rollouts)
        chunks = [n_rollouts]
    ok, lengths, reasons, targets = [], [], collections.Counter(), collections.Counter()
    traj = [] if record else None
    for B in chunks:
        if stateful:
            env.reset(batch=1)
            policy.reset()
        else:
            env.reset(batch=B)
            if hasattr(policy, "reset"):
                policy.reset()
        state = env.state
        alive = np.ones(B, dtype=bool)
        length = np.full(B, max_pulses, dtype=np.int64)
        succ = np.zeros(B, dtype=bool)
        no_act = np.zeros(B, dtype=bool)
        # argmax of the belief AT the step the episode succeeded.  Finished rows
        # stay in the batch and keep receiving pulses, so the belief at the end
        # of the loop is the post-success one and its argmax can differ.
        target = np.full(B, -1, dtype=np.int64)
        steps = [[] for _ in range(B)] if record else None
        # success may already hold at t = 0
        done0 = env.is_done(state).cpu().numpy()
        if done0.any():
            target[done0] = state.argmax(-1).cpu().numpy()[done0]
        succ |= done0; length[done0] = 0; alive &= ~done0
        for t in range(max_pulses):
            if not alive.any():
                break
            beliefs = state.detach().cpu().numpy()
            if hasattr(policy, "act_batch"):
                acts = np.asarray(policy.act_batch(beliefs, t, rng), dtype=np.int64)
            else:
                picks = [policy.act(beliefs[b], t, rng) if alive[b] else 0 for b in range(B)]
                acts = np.zeros(B, dtype=np.int64)
                for b, pk in enumerate(picks):
                    if pk is None:
                        no_act[b] = True; alive[b] = False; length[b] = t
                    elif isinstance(pk, Action):
                        acts[b] = env.library.encode(pk)
                        if env.library.decode(acts[b]).tau_index != pk.tau_index:
                            raise ValueError("policy returned an Action whose duration is not in the "
                                             "library; use the engine path for free-tau policies")
                    else:
                        acts[b] = int(pk)
            tr = env.step(acts)
            state = tr.belief
            done = tr.done.cpu().numpy()
            out = tr.outcome.cpu().numpy()
            just = done & alive
            length[just] = t + 1
            if just.any():
                target[just] = tr.belief.argmax(-1).cpu().numpy()[just]
            succ |= just
            if record:
                for b in np.where(alive)[0]:
                    steps[b].append((int(acts[b]), int(out[b])))
            alive &= ~done
        for b in range(B):
            ok.append(bool(succ[b]))
            lengths.append(int(length[b]))
            if succ[b]:
                reasons["success"] += 1; targets[int(target[b])] += 1
            elif no_act[b]:
                reasons["no_action"] += 1
            else:
                reasons["max_pulses"] += 1
        if record:
            traj.extend(steps)
        if progress:
            print(f"  {len(ok)}/{n_rollouts}: success {np.mean(ok):.3f}, mean pulses {np.mean(lengths):.1f}", flush=True)
    return _finish(lengths, ok, reasons, targets, env.molecule, max_pulses, p_target, t0, 0, policy,
                   f"PurificationEnv[{env.device}]", traj, n_rollouts)
