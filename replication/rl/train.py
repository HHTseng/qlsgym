"""Training loop, snapshot protocol and snapshot selection for the qMDP-DQN baseline of
arXiv:2608.03702 Appendix C.1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np

from .agent import AgentConfig, QNetPolicy, epsilon, make_qnet, untrained_action_rows
from .replay import Replay

# Appendix C.1: the selected policy is re-evaluated on 1000 Monte-Carlo
# trajectories at seed 777.  Fixed so the final number is never tuned.
EVAL_SEED = 777
# seed of the snapshot evaluations, offset per episode as in the source
SNAPSHOT_SEED = 4242


def run_tag(library, agent_cfg: AgentConfig, env_cfg, episodes: int) -> str:
    """Content hash of everything that makes two runs incomparable."""
    h = hashlib.sha256()
    h.update(library.molecule.fingerprint().encode())
    h.update(library.tag().encode())
    h.update(json.dumps(asdict(agent_cfg), sort_keys=True).encode())
    h.update(json.dumps(asdict(env_cfg), sort_keys=True).encode())
    h.update(str(int(episodes)).encode())
    return h.hexdigest()[:12]


def stamp_snapshot_dir(snapshot_dir: str, tag: str, meta: dict) -> str:
    """Create and stamp snapshot_dir, or refuse if another run owns it."""
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, "run.json")
    if os.path.exists(path):
        with open(path) as fh:
            have = json.load(fh)
        if have.get("run_tag") != tag:
            raise RuntimeError(
                f"{snapshot_dir} holds checkpoints of run {have.get('run_tag')} but this run is "
                f"{tag}; writing here would overwrite them")
        return tag
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"run_tag": tag, **meta}, fh, indent=1)
    os.replace(tmp, path)
    return tag


def snapshot_path(snapshot_dir: str, episode: int) -> str:
    return os.path.join(snapshot_dir, f"ep{episode:05d}.pt")


def load_policy(path: str, n_states: int, n_actions: int, agent_cfg: AgentConfig,
                device: str = "cpu") -> QNetPolicy:
    """Rebuild a snapshot as a greedy QNetPolicy."""
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    net = make_qnet(n_states, n_actions, agent_cfg).to(device)
    net.load_state_dict(ck["state_dict"])
    return QNetPolicy(net, n_actions, device=device)


def select_best(snapshots: list, rule: str = "convergence") -> dict:
    """The strongest snapshot (paper Sec. III.3, "the strongest policy")."""
    if not snapshots:
        raise ValueError("no snapshots to select from: train for at least snapshot_every episodes")
    if rule == "convergence":
        return max(snapshots, key=lambda s: (s["success_fraction"], -s["mean_pulses"]))
    if rule == "mean_length":
        return min(snapshots, key=lambda s: (s["mean_pulses"], -s["success_fraction"]))
    raise ValueError(f"unknown selection rule {rule!r}")


def train(molecule, library, tables, agent_cfg: AgentConfig, env_cfg, episodes: int,
          device: str = "cpu", snapshot_dir: str | None = None, moving_window: int = 100,
          log_every: int = 100, verbose: bool = True) -> dict:
    """Train one qMDP-DQN configuration for episodes episodes."""
    import torch
    import torch.nn as nn

    from qlsgym.env.env import PurificationEnv
    from qlsgym.policies.rollout import rollout

    torch.manual_seed(agent_cfg.seed)
    tag = run_tag(library, agent_cfg, env_cfg, episodes)
    if snapshot_dir:
        stamp_snapshot_dir(snapshot_dir, tag, {
            "molecule": molecule.name, "fingerprint": molecule.fingerprint(),
            "library_tag": library.tag(), "n_actions": library.n_actions,
            "episodes": int(episodes), "agent_cfg": asdict(agent_cfg),
            "env_cfg": asdict(env_cfg)})

    env = PurificationEnv(molecule, library, tables, env_cfg, device=device, batch=1)
    env.seed(agent_cfg.seed + 2)
    eval_env = env.clone(agent_cfg.snapshot_rollouts)
    max_pulses = env.cfg.max_pulses

    n_in, n_out = molecule.n_states, library.n_actions
    policy = make_qnet(n_in, n_out, agent_cfg).to(device)
    target = make_qnet(n_in, n_out, agent_cfg).to(device)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=agent_cfg.eta_rl, amsgrad=True)
    loss_fn = nn.SmoothL1Loss()
    head = [m for m in policy.modules() if isinstance(m, nn.Linear)][-1]
    w_init = head.weight.detach().clone()

    replay = Replay(agent_cfg.memory, n_in, env.device, torch)
    gen = torch.Generator(device=env.device)
    gen.manual_seed(agent_cfg.seed + 1)

    env_steps = 0
    solved_hist: list = []
    curve: list = []
    snapshots: list = []
    t_train = 0.0
    t_eval = 0.0
    t0_all = time.perf_counter()

    for ep in range(1, int(episodes) + 1):
        t0 = time.perf_counter()
        env.reset(batch=1)
        ep_reward, ep_len, solved = 0.0, 0, False
        for t in range(max_pulses):
            eps = epsilon(env_steps, agent_cfg)
            if float(torch.rand(1, generator=gen, device=env.device)) < eps:
                a = torch.randint(0, n_out, (1,), device=env.device, generator=gen)
            else:
                with torch.no_grad():
                    a = policy(env.state.to(torch.float32)).argmax(-1)
            s_prev = env.state.clone()
            tr = env.step(a)
            d0, d1 = tr.done0, tr.done1
            if not agent_cfg.bootstrap_on_truncation and bool(tr.truncated[0]):
                d0 = d0 | tr.truncated
                d1 = d1 | tr.truncated
            replay.push(s_prev, a, tr.s0, tr.s1, tr.pi0, tr.pi1, tr.r0, tr.r1, d0, d1)
            env_steps += 1
            ep_reward += float(tr.reward[0])
            ep_len = t + 1

            if replay.n >= agent_cfg.batch_size:
                for _ in range(agent_cfg.updates_per_step):
                    bs, ba, bs0, bs1, bp0, bp1, br0, br1, bd0, bd1 = replay.sample(
                        agent_cfg.batch_size, generator=gen)
                    q = policy(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        if agent_cfg.double_q:
                            q0 = target(bs0).gather(1, policy(bs0).argmax(1, keepdim=True)).squeeze(1)
                            q1 = target(bs1).gather(1, policy(bs1).argmax(1, keepdim=True)).squeeze(1)
                        else:
                            q0 = target(bs0).max(1).values
                            q1 = target(bs1).max(1).values
                        q0 = q0 * (~bd0).to(bs.dtype)
                        q1 = q1 * (~bd1).to(bs.dtype)
                        y = bp0 * (br0 + agent_cfg.gamma * q0) + bp1 * (br1 + agent_cfg.gamma * q1)
                    loss = loss_fn(q, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_value_(policy.parameters(), agent_cfg.grad_clip)
                    opt.step()
                    with torch.no_grad():
                        for pt, pp in zip(target.parameters(), policy.parameters()):
                            pt.mul_(1 - agent_cfg.tau_rl).add_(pp, alpha=agent_cfg.tau_rl)
                        for bt, bp in zip(target.buffers(), policy.buffers()):
                            bt.copy_(bp)
            if bool(tr.done[0]):
                solved = True
                break
        t_train += time.perf_counter() - t0
        solved_hist.append(bool(solved))
        window = solved_hist[-moving_window:]
        curve.append({"episode": ep, "length": ep_len, "reward": ep_reward,
                      "solved": bool(solved),
                      "moving_success": float(np.mean(window)),
                      "epsilon": epsilon(env_steps, agent_cfg),
                      "env_steps": env_steps, "t_train_s": t_train})

        if ep % agent_cfg.snapshot_every == 0:
            t0 = time.perf_counter()
            policy.eval()
            res = rollout(eval_env, QNetPolicy(policy, n_out, device=device),
                          n_rollouts=agent_cfg.snapshot_rollouts, seed=SNAPSHOT_SEED + ep)
            policy.train()
            t_eval += time.perf_counter() - t0
            untrained = untrained_action_rows(head.weight.detach(), w_init)
            with torch.no_grad():
                p0 = torch.as_tensor(env.p_init, dtype=torch.float32, device=env.device)
                a_greedy = int(policy(p0.reshape(1, -1)).argmax())
            path = None
            if snapshot_dir:
                path = snapshot_path(snapshot_dir, ep)
                torch.save({"state_dict": policy.state_dict(), "episode": ep,
                            "agent_cfg": asdict(agent_cfg), "run_tag": tag,
                            "library_tag": library.tag(),
                            "fingerprint": molecule.fingerprint()}, path)
            snapshots.append({
                "episode": ep, "t_train_s": t_train, "env_steps": env_steps,
                "success_fraction": res.success_fraction, "success_err": res.success_err,
                "mean_pulses": res.mean_pulses,
                "mean_pulses_successful": (float(res.mean_pulses_successful)
                                           if np.isfinite(res.mean_pulses_successful) else None),
                "n_rollouts": res.n_rollouts,
                "frac_untrained_action_rows": float(untrained.to(torch.float32).mean()),
                "greedy_action_is_untrained": bool(untrained[a_greedy]),
                "path": path,
            })
            if verbose:
                print(f"[ep {ep:5d}] len={ep_len:3d} eps={curve[-1]['epsilon']:.4f} "
                      f"snap conv={res.success_fraction:.3f} mean={res.mean_pulses:.2f} "
                      f"t_train={t_train / 60:.1f}m t_eval={t_eval / 60:.1f}m", flush=True)
        elif verbose and log_every and ep % log_every == 0:
            print(f"[ep {ep:5d}] len={ep_len:3d} moving_success={curve[-1]['moving_success']:.3f}",
                  flush=True)

    return {"curve": curve, "snapshots": snapshots, "net": policy, "env": env,
            "eval_env": eval_env, "run_tag": tag, "snapshot_dir": snapshot_dir,
            "seconds_train": t_train, "seconds_snapshot_eval": t_eval,
            "seconds_total": time.perf_counter() - t0_all, "env_steps": env_steps,
            "max_pulses": int(max_pulses)}
