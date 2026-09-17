"""Proximal policy optimisation on the batched belief MDP."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..env.env import PurificationEnv, Transition
from ..policies.rollout import RolloutResult, rollout

# seed of the in-training snapshot evaluations; the caller's final evaluation
# must use a different one so the reported number is not the selection number
SNAPSHOT_SEED = 4242

OBS_TRANSFORMS = ("p", "sqrt")
VALUE_TARGETS = ("gae", "qmdp", "qmdp_gae")


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 256               # parallel trajectories per update
    n_steps: int = 32               # env steps per trajectory per update
    total_steps: int = 2_000_000    # environment steps in total (n_envs * n_steps * updates)
    epochs: int = 4                 # optimisation passes per update
    minibatches: int = 8
    lr: float = 1.0e-3
    gamma: float = 1.0              # finite horizon, undiscounted like the source papers
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    n_hidden_layers: int = 2
    obs: str = "sqrt"               # OBS_TRANSFORMS
    value_target: str = "gae"       # VALUE_TARGETS
    # rewards are multiplied by this before learning; None -> 1 / max_pulses
    reward_scale: float | None = None
    eval_every: int = 25            # updates between snapshot evaluations
    eval_rollouts: int = 200
    # How the trained actor acts when evaluated.  A PPO actor IS a stochastic
    # policy; taking its argmax is a different policy, and on a large library it
    # is a much worse one -- it can lock onto one drive and repeat it forever.
    # With 312 ThF+ actions the sampled policy finishes every episode while its
    # argmax finishes 0.5 %, so the eval mode has to be explicit, not assumed.
    eval_greedy: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.obs not in OBS_TRANSFORMS:
            raise ValueError(f"obs must be one of {OBS_TRANSFORMS}, got {self.obs!r}")
        if self.value_target not in VALUE_TARGETS:
            raise ValueError(f"value_target must be one of {VALUE_TARGETS}, got {self.value_target!r}")
        if self.n_envs < 1 or self.n_steps < 1 or self.minibatches < 1 or self.epochs < 1:
            raise ValueError("n_envs, n_steps, minibatches and epochs must be >= 1")

    @property
    def n_updates(self) -> int:
        return max(1, int(math.ceil(self.total_steps / (self.n_envs * self.n_steps))))

    def as_dict(self) -> dict:
        return asdict(self)


# network and policy


def _torch():
    import torch
    return torch


def transform_obs(belief, kind: str):
    """(B, n) belief tensor -> network input (same shape, float32)."""
    x = belief.to(_torch().float32)
    if kind == "sqrt":
        return x.clamp_min(0.0).sqrt()
    return x


def _build_actor_critic():
    torch = _torch()
    nn = torch.nn

    class ActorCritic(nn.Module):
        """Shared trunk n_in -> hidden -> ... -> hidden, a policy head of n_actions logits and a
        value head of one number.
        """

        def __init__(self, n_in: int, n_actions: int, hidden: int = 256, n_hidden_layers: int = 2):
            super().__init__()
            layers: list = [nn.Linear(n_in, hidden), nn.Tanh()]
            for _ in range(n_hidden_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            self.trunk = nn.Sequential(*layers)
            self.pi = nn.Linear(hidden, n_actions)
            self.v = nn.Linear(hidden, 1)
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.zeros_(self.pi.bias)
            nn.init.orthogonal_(self.v.weight, gain=1.0)
            nn.init.zeros_(self.v.bias)
            self.n_in, self.n_actions = int(n_in), int(n_actions)

        def forward(self, x):
            h = self.trunk(x)
            return self.pi(h), self.v(h).squeeze(-1)

    return ActorCritic


ActorCritic = _build_actor_critic()


class ActorPolicy:
    """A trained ActorCritic as a Policy."""

    stateful = False

    def __init__(self, net, obs: str = "sqrt", device: str = "cpu", greedy: bool = True):
        torch = _torch()
        self.net = net.to(device).eval()
        self.obs, self.device, self.greedy = obs, torch.device(device), bool(greedy)
        self.n_actions = int(net.n_actions)

    def logits(self, beliefs) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            x = transform_obs(torch.as_tensor(np.asarray(beliefs, dtype=np.float64), device=self.device), self.obs)
            lg, _ = self.net(x)
        return lg.detach().cpu().numpy()

    def act_batch(self, beliefs, t, rng):
        beliefs = np.asarray(beliefs, dtype=np.float64)
        squeeze = beliefs.ndim == 1
        if squeeze:
            beliefs = beliefs[None]
        lg = self.logits(beliefs)
        if self.greedy:
            a = np.argmax(lg, axis=1)
        else:
            z = lg - lg.max(1, keepdims=True)
            pr = np.exp(z)
            pr /= pr.sum(1, keepdims=True)
            u = rng.random(pr.shape[0])
            a = (pr.cumsum(1) < u[:, None]).sum(1)
            a = np.minimum(a, pr.shape[1] - 1)
        a = a.astype(np.int64)
        return int(a[0]) if squeeze else a

    def act(self, belief, t, rng):
        return int(self.act_batch(belief, t, rng))

    def __repr__(self) -> str:
        return f"ActorPolicy(n_actions={self.n_actions}, obs={self.obs!r}, greedy={self.greedy})"


def save_policy(net, cfg: PPOConfig, path: str, meta: dict | None = None) -> str:
    torch = _torch()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "n_in": net.n_in, "n_actions": net.n_actions, "ppo_config": cfg.as_dict(),
                "meta": meta or {}}, path)
    return path


def load_policy(path: str, device: str = "cpu", greedy: bool = True) -> ActorPolicy:
    """The saved actor as an ActorPolicy; .meta carries what save_policy was given (library tag,
    fingerprint, ...).
    """
    torch = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PPOConfig(**ck["ppo_config"])
    net = ActorCritic(ck["n_in"], ck["n_actions"], cfg.hidden, cfg.n_hidden_layers)
    net.load_state_dict(ck["state_dict"])
    pol = ActorPolicy(net, cfg.obs, device, greedy)
    pol.meta = ck.get("meta", {})
    pol.config = cfg
    return pol


# training


@dataclass
class TrainResult:
    config: dict
    n_updates: int
    env_steps: int
    seconds: float
    history: list                       # per update: {update, env_steps, loss_pi, loss_v, entropy, ...}
    snapshots: list                     # per evaluation: {update, env_steps, success, mean_pulses, ...}
    best: dict                          # the selected snapshot's record
    best_state_dict: dict
    final_state_dict: dict
    n_in: int
    n_actions: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("best_state_dict"); d.pop("final_state_dict")
        return d


def _evaluate(net, cfg: PPOConfig, env_eval: PurificationEnv, device) -> RolloutResult:
    # a private view: the rollout resizes and reseeds the env it is given, and
    # env_eval may be the training env itself
    pol = ActorPolicy(net, cfg.obs, device, greedy=cfg.eval_greedy)
    res = rollout(env_eval.clone(cfg.eval_rollouts), pol, n_rollouts=cfg.eval_rollouts, seed=SNAPSHOT_SEED)
    net.train()
    return res


def _better(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    if a["success"] != b["success"]:
        return a["success"] > b["success"]
    return a["mean_pulses"] < b["mean_pulses"]


def compute_advantages(reward, value, terminal, last_value, branch_target,
                       gamma, gae_lambda, value_target):
    """Sampled or branch-expected GAE, with buffers shaped [time, batch].

    Legacy ``qmdp`` is one-step actor-critic. ``qmdp_gae`` accumulates
    expected TD residuals along sampled paths, stopping at episode boundaries.
    Branch targets already contain separate branch terminal/budget masks.
    """
    torch = _torch()
    if value_target == "qmdp":
        return branch_target - value, branch_target
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    for t in reversed(range(len(reward))):
        continue_mask = 1.0 - terminal[t]
        if value_target == "qmdp_gae":
            residual = branch_target[t] - value[t]
        elif value_target == "gae":
            next_value = last_value if t == len(reward) - 1 else value[t + 1]
            residual = reward[t] + gamma * continue_mask * next_value - value[t]
        else:
            raise ValueError(f"unknown value target: {value_target}")
        running = residual + gamma * gae_lambda * continue_mask * running
        advantage[t] = running
    return advantage, advantage + value


def train_ppo(env: PurificationEnv, cfg: PPOConfig, env_eval: PurificationEnv | None = None,
              log=None, on_snapshot=None) -> TrainResult:
    """Train on env (its batch is set to cfg.n_envs); evaluate greedy snapshots on env_eval
    (default: env itself, which is only right when env is exact).
    """
    torch = _torch()
    device = env.device
    env_eval = env if env_eval is None else env_eval
    if env_eval.n_actions != env.n_actions or env_eval.n_states != env.n_states:
        raise ValueError("env_eval must share the molecule and the action library with env")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed + 1)
    net = ActorCritic(env.n_states, env.n_actions, cfg.hidden, cfg.n_hidden_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    r_scale = (1.0 / env.cfg.max_pulses) if cfg.reward_scale is None else float(cfg.reward_scale)

    B, T = cfg.n_envs, cfg.n_steps
    state = env.reset(seed=cfg.seed, batch=B)
    obs_buf = torch.zeros((T, B, env.n_states), dtype=torch.float32, device=device)
    act_buf = torch.zeros((T, B), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, B), dtype=torch.float32, device=device)
    val_buf = torch.zeros((T, B), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((T, B), dtype=torch.float32, device=device)
    term_buf = torch.zeros((T, B), dtype=torch.float32, device=device)   # 1 where the episode ended
    q_buf = torch.zeros((T, B), dtype=torch.float32, device=device)      # qmdp one-step target

    history, snapshots = [], []
    best, best_sd = None, None
    t0 = time.time()
    env_steps = 0
    ep_len_sum, ep_n, ep_succ = 0.0, 0, 0
    for update in range(cfg.n_updates):
        net.eval()
        with torch.no_grad():
            for t in range(T):
                x = transform_obs(state, cfg.obs)
                logits, value = net(x)
                probs = torch.softmax(logits, -1)
                a = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(-1)
                tr: Transition = env.step(a)
                ended = tr.done | tr.truncated
                obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = x, a, logp, value
                rew_buf[t] = tr.reward.to(torch.float32) * r_scale
                term_buf[t] = ended.to(torch.float32)
                if cfg.value_target in ("qmdp", "qmdp_gae"):
                    _, v0 = net(transform_obs(tr.s0, cfg.obs))
                    _, v1 = net(transform_obs(tr.s1, cfg.obs))
                    trunc_next = (env.steps >= env.cfg.max_pulses).to(torch.float32)
                    c0 = (1.0 - tr.done0.to(torch.float32)) * (1.0 - trunc_next)
                    c1 = (1.0 - tr.done1.to(torch.float32)) * (1.0 - trunc_next)
                    q_buf[t] = (tr.pi0.to(torch.float32) * (tr.r0.to(torch.float32) * r_scale + cfg.gamma * c0 * v0)
                                + tr.pi1.to(torch.float32) * (tr.r1.to(torch.float32) * r_scale + cfg.gamma * c1 * v1))
                # episode bookkeeping, then reset the finished rows
                n_end = int(ended.sum())
                if n_end:
                    ep_len_sum += float(env.steps[ended].sum()); ep_n += n_end; ep_succ += int(tr.done.sum())
                    env.reset_rows(ended)
                state = env.state
            _, last_value = net(transform_obs(state, cfg.obs))
        env_steps += B * T
        net.train()

        # advantages
        adv, ret = compute_advantages(rew_buf, val_buf, term_buf, last_value,
                                      q_buf, cfg.gamma, cfg.gae_lambda, cfg.value_target)

        # PPO update
        n = B * T
        flat = lambda z: z.reshape(n, *z.shape[2:])
        f_obs, f_act, f_logp, f_adv, f_ret = flat(obs_buf), flat(act_buf), flat(logp_buf), flat(adv), flat(ret)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        mb = max(1, n // cfg.minibatches)
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "clip_frac": 0.0, "approx_kl": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen, device=device)
            for lo in range(0, n, mb):
                idx = perm[lo:lo + mb]
                logits, value = net(f_obs[idx])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, f_act[idx][:, None]).squeeze(-1)
                ratio = torch.exp(logp - f_logp[idx])
                a_ = f_adv[idx]
                loss_pi = -torch.min(ratio * a_, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * a_).mean()
                loss_v = 0.5 * (value - f_ret[idx]).pow(2).mean()
                ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    stats["loss_pi"] += float(loss_pi); stats["loss_v"] += float(loss_v); stats["entropy"] += float(ent)
                    stats["clip_frac"] += float(((ratio - 1.0).abs() > cfg.clip).float().mean())
                    stats["approx_kl"] += float((f_logp[idx] - logp).mean())
                n_mb += 1
        rec = {k: v / max(n_mb, 1) for k, v in stats.items()}
        rec.update(update=update + 1, env_steps=env_steps, seconds=time.time() - t0,
                   train_episodes=ep_n, train_mean_length=(ep_len_sum / ep_n if ep_n else None),
                   train_success=(ep_succ / ep_n if ep_n else None))
        history.append(rec)
        ep_len_sum, ep_n, ep_succ = 0.0, 0, 0

        last = update + 1 == cfg.n_updates
        if (update + 1) % cfg.eval_every == 0 or last:
            res = _evaluate(net, cfg, env_eval, device)
            snap = {"update": update + 1, "env_steps": env_steps, "success": res.success_fraction,
                    "success_err": res.success_err, "mean_pulses": res.mean_pulses,
                    "mean_pulses_successful": (None if not math.isfinite(res.mean_pulses_successful)
                                               else res.mean_pulses_successful),
                    "n_rollouts": res.n_rollouts, "seed": SNAPSHOT_SEED, "seconds": time.time() - t0}
            snapshots.append(snap)
            if _better(snap, best):
                best = snap
                best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if log is not None:
                mode = "greedy" if cfg.eval_greedy else "sampled"
                log(f"[ppo] update {update + 1}/{cfg.n_updates} steps {env_steps}: {mode} success "
                    f"{res.success_fraction:.3f} mean pulses {res.mean_pulses:.1f} | train success "
                    f"{rec['train_success']} len {rec['train_mean_length']} ent {rec['entropy']:.3f} "
                    f"({time.time() - t0:.0f}s)")
            if on_snapshot is not None:
                on_snapshot(snap, net)
    final_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return TrainResult(cfg.as_dict(), cfg.n_updates, env_steps, time.time() - t0, history, snapshots,
                       best, best_sd, final_sd, env.n_states, env.n_actions)


def policy_from_state_dict(sd: dict, n_in: int, n_actions: int, cfg: PPOConfig, device="cpu",
                           greedy: bool = True) -> ActorPolicy:
    net = ActorCritic(n_in, n_actions, cfg.hidden, cfg.n_hidden_layers)
    net.load_state_dict(sd)
    return ActorPolicy(net, cfg.obs, device, greedy)
