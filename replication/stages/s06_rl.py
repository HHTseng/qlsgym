"""Stage 6 -- the qMDP-DQN reinforcement-learning baseline (paper App. C.1)."""

from __future__ import annotations

import time
from dataclasses import asdict

from .. import paper as P
from .. import rl as rlmod
from ..config import Config, read_result, work_dir

NAME = "s06_rl"
TITLE = "the qMDP-DQN reinforcement-learning baseline"
PAPER = "App. C.1"
REQUIRES = ("s01_physics",)
COST = "~1 h, GPU (cache + 3000 episodes, one configuration)"

# The paper's own numbers, copied out of replication.paper
# into the payload so the quotation travels beside our measurement, never in
# place of it.  The experimental settings App. C.1 states but Fig. 5 does not
# plot (the action count, the episode and snapshot protocol) are stated here.
PAPER_QUOTED = {
    "source": "arXiv:2608.03702, App. C.1 and Sec. III.3",
    "n_actions": 35280,
    "n_actions_sentence": "1764 frequency choices per polarization x 2 polarizations x 10 "
                          "pulse-duration indices, plus 'the additional numerical pulses from "
                          "the THz region'",
    "episodes": 3000,
    "snapshot_every": 100,
    "snapshot_rollouts": 200,
    "final_rollouts": P.N_MC_ROLLOUTS,
    "final_seed": 777,
    "converged_fraction": P.FIG5A_RL_PAPER["convergence"],
    "converged_fraction_err": P.FIG5A_RL_PAPER["convergence_err"],
    "mean_pulses_all_rollouts": P.FIG5A_RL_PAPER["mean_pulses"],
    "mean_pulses_successful_inferred": 7.5,
    "mean_pulses_note": "the paper reports one 'average episode length'; its own arithmetic "
                        "(0.428 L + 0.572 x 80 = 48.974) pins it as the all-rollouts mean",
    "selected_config": P.RL_SELECTED_CONFIG,
    "selected_episode": P.RL_SELECTED_EPISODE,
    "hours_train_12_configs": P.RL_SWEEP_TRAIN_HOURS,
    "hours_snapshot_validation": P.RL_SNAPSHOT_EVAL_HOURS,
}

# The prior replication's measured numbers, for orientation only.  Theirs, not
# ours: a different action library (35700) and a 12-configuration sweep.
FNOREPL_QUOTED = {
    "source": "FNO_REPL/_llm_notes/05_rl_baseline.md, primary run (21-resonance THz library)",
    "n_actions": 35700,
    "converged_fraction": 0.226,
    "converged_fraction_err": 0.013,
    "mean_pulses_all_rollouts": 62.67,
    "mean_pulses_successful": 3.33,
    "selected": "configuration 1, episode 1300",
    "anchor_converged_fraction": 0.086,
    "anchor_mean_pulses": 73.38,
    "minutes_train_per_config": 14.3,
}


# configuration


def _get(cfg: Config, key: str, default):
    return cfg.overrides.get(key, default)


def _control_grid(cfg: Config, molecule):
    """App. C.1's grid: n_freq frequencies x n_tau_slots durations."""
    from qlsgym.env.actions import ControlGrid

    n_freq, seed = int(_get(cfg, "n_freq", 1764)), int(_get(cfg, "grid_seed", 0))
    for slots in range(int(_get(cfg, "n_tau_slots", 10)), 0, -1):
        try:
            return ControlGrid.rl_discrete(molecule, n_freq=n_freq, n_tau_slots=slots, seed=seed)
        except ValueError:
            pass
    raise ValueError(f"{molecule.name}'s tau grid holds no rl_discrete duration slot")


def _library(cfg: Config, molecule):
    """The shared discrete action library (paper Eq. 23 / App. C.1)."""
    from qlsgym.env.actions import ActionLibrary

    include = bool(_get(cfg, "include_primitives", True))
    kind = _get(cfg, "library", "rl_discrete")
    if kind == "physics_subset":
        return ActionLibrary.physics_subset(molecule, include_primitives=include, clip_tau=True)
    if kind != "rl_discrete":
        raise ValueError(f"unknown library {kind!r}: use 'rl_discrete' or 'physics_subset'")
    return ActionLibrary.from_grid(molecule, _control_grid(cfg, molecule),
                                   include_primitives=include)


def _env_cfg(cfg: Config):
    from qlsgym.env.env import EnvConfig

    return EnvConfig(max_pulses=int(cfg.max_pulses),
                     rho=float(_get(cfg, "rho", 2.0)),
                     penalty_mode=str(_get(cfg, "penalty_mode", "indicator")))


def _agent_cfg(cfg: Config):
    return rlmod.AgentConfig(
        hidden=int(_get(cfg, "hidden", 128)),
        batch_size=int(_get(cfg, "batch_size", 256)),
        memory=int(_get(cfg, "memory", 250_000)),
        eps_end=float(_get(cfg, "eps_end", 0.025)),
        eps_decay_steps=float(_get(cfg, "eps_decay_steps", 7.2e4)),
        tau_rl=float(_get(cfg, "tau_rl", 1.0e-4)),
        eta_rl=float(_get(cfg, "eta_rl", 5.0e-4)),
        snapshot_every=int(_get(cfg, "snapshot_every", 100)),
        snapshot_rollouts=int(_get(cfg, "snapshot_rollouts", 200)),
        double_q=bool(_get(cfg, "double_q", False)),
        seed=int(cfg.seed),
    )


def _cache(molecule, library) -> dict:
    """Where the transfer tables live and whether they are already there."""
    import os

    from qlsgym.env.cache import tables_dir

    d = tables_dir(molecule, library)
    per_action = sum(2 * b.n_states ** 2 for b in molecule.blocks)
    return {"dir": d,
            "present": os.path.exists(os.path.join(d, "manifest.json")),
            "est_bytes": int(per_action * library.n_grid * 4),
            "floats_per_action": int(per_action),
            "n_drives": len(library.drives())}


def _snapshot_dir(cfg: Config, tag: str) -> str:
    import os

    return os.path.join(str(work_dir()), "replication", "rl", f"{cfg.tag}_{NAME}_{tag}")


# the stage


def plan(cfg: Config) -> list:
    from ..config import load_molecule

    mol = load_molecule(cfg)
    lib = _library(cfg, mol)
    agent_cfg = _agent_cfg(cfg)
    env_cfg = _env_cfg(cfg)
    cache = _cache(mol, lib)
    tag = rlmod.run_tag(lib, agent_cfg, env_cfg, cfg.rl_episodes)
    n_snap = cfg.rl_episodes // agent_cfg.snapshot_every
    lines = [
        f"molecule {mol.name} ({mol.n_states} states, fingerprint {mol.fingerprint()}); "
        f"needs s01_physics only -- the RL baseline runs on exact tables, not the surrogate",
        f"action library {_get(cfg, 'library', 'rl_discrete')}: {lib.n_grid} grid + "
        f"{lib.n_primitives} primitives = {lib.n_actions} actions (tag {lib.tag()}); "
        f"paper App. C.1 states {PAPER_QUOTED['n_actions']}",
        f"transfer tables: {cache['n_drives']} drives, "
        f"{cache['floats_per_action']} floats/action, ~{cache['est_bytes'] / 1e9:.2f} GB float32 "
        f"at {cache['dir']}",
        ("tables PRESENT (reused)" if cache["present"] else
         "tables ABSENT -- run() will build them with qlsgym.env.cache.build_action_tables "
         f"on device {cfg.resolved_device()}; this is the expensive part of the stage"),
        f"train 1 configuration (paper Table 3 row 3, its own dagger: tau_RL={agent_cfg.tau_rl}, "
        f"eta_RL={agent_cfg.eta_rl}, rho={env_cfg.rho}, eps_end={agent_cfg.eps_end}) for "
        f"{cfg.rl_episodes} episodes of at most {cfg.max_pulses} pulses",
        f"snapshot every {agent_cfg.snapshot_every} episodes -> {n_snap} snapshots x "
        f"{agent_cfg.snapshot_rollouts} greedy rollouts, saved to {_snapshot_dir(cfg, tag)}",
        f"select the best snapshot by {_get(cfg, 'selection_rule', 'convergence')}, re-evaluate "
        f"it on {cfg.n_rollouts} fresh trajectories at seed {rlmod.EVAL_SEED} "
        f"(App. C.1 used {PAPER_QUOTED['final_rollouts']}; set --set n_rollouts= to match)",
        f"measure the non-adaptive anchor (repeat the best single action) on the same library, "
        f"belief, budget and seed",
        f"{len(rlmod.DEVIATIONS)} recorded deviations from App. C.1: "
        + ", ".join(d["id"] for d in rlmod.DEVIATIONS),
    ]
    if n_snap == 0:
        lines.append(f"WARNING: rl_episodes={cfg.rl_episodes} < snapshot_every="
                     f"{agent_cfg.snapshot_every}: no snapshot would be taken")
    return lines


def run(cfg: Config) -> dict:
    from qlsgym.env.cache import build_action_tables
    from qlsgym.policies.rollout import rollout

    from ..config import load_molecule

    t0 = time.time()
    mol = load_molecule(cfg)
    read_result("s01_physics", cfg, mol)          # fails with "run stage ... first"

    lib = _library(cfg, mol)
    agent_cfg = _agent_cfg(cfg)
    env_cfg = _env_cfg(cfg)
    device = cfg.resolved_device()
    verbose = bool(_get(cfg, "verbose", True))

    t_cache = time.time()
    cache = _cache(mol, lib)
    tables = build_action_tables(mol, lib, device=device, progress=verbose)
    t_cache = time.time() - t_cache

    tag = rlmod.run_tag(lib, agent_cfg, env_cfg, cfg.rl_episodes)
    snap_dir = _snapshot_dir(cfg, tag)
    out = rlmod.train(mol, lib, tables, agent_cfg, env_cfg, episodes=cfg.rl_episodes,
                      device=device, snapshot_dir=snap_dir,
                      moving_window=int(_get(cfg, "moving_window", 100)), verbose=verbose)

    rule = str(_get(cfg, "selection_rule", "convergence"))
    best = rlmod.select_best(out["snapshots"], rule)
    best_by_length = rlmod.select_best(out["snapshots"], "mean_length")
    policy = (rlmod.load_policy(best["path"], mol.n_states, lib.n_actions, agent_cfg, device)
              if best["path"] else rlmod.QNetPolicy(out["net"], lib.n_actions, device=device))
    eval_env = out["eval_env"]
    t_eval = time.time()
    trained = rollout(eval_env, policy, n_rollouts=cfg.n_rollouts, seed=rlmod.EVAL_SEED)
    t_eval = time.time() - t_eval

    anchor_pick = rlmod.best_fixed_action(eval_env)
    anchor = rollout(eval_env, rlmod.FixedActionPolicy(anchor_pick["action"], lib.n_actions),
                     n_rollouts=cfg.n_rollouts, seed=rlmod.EVAL_SEED)

    return {
        "molecule": mol.name,
        "device": device,
        "library": {**lib.describe(), "cache_dir": cache["dir"],
                    "cache_est_bytes": cache["est_bytes"],
                    "cache_was_present": cache["present"]},
        "agent_cfg": asdict(agent_cfg),
        "env_cfg": asdict(env_cfg),
        "training": {
            "episodes": int(cfg.rl_episodes),
            "env_steps": out["env_steps"],
            "max_pulses": out["max_pulses"],
            "moving_window": int(_get(cfg, "moving_window", 100)),
            "curve_columns": ["episode", "moving_success", "env_steps"],
            "curve": [[c["episode"], round(c["moving_success"], 4), c["env_steps"]]
                      for c in out["curve"]],
            "snapshots": out["snapshots"],
            "snapshot_dir": snap_dir,
            "run_tag": out["run_tag"],
        },
        "selected_snapshot": {"rule": rule, **{k: best[k] for k in
                                               ("episode", "success_fraction", "mean_pulses", "path")},
                              "by_mean_length_would_be": best_by_length["episode"],
                              "frac_untrained_action_rows": best["frac_untrained_action_rows"],
                              "greedy_action_is_untrained": best["greedy_action_is_untrained"]},
        "trained_policy": {**trained.as_dict(),
                           "converged_fraction": trained.success_fraction,
                           "eval_seed": rlmod.EVAL_SEED,
                           "selection_optimism": best["success_fraction"] - trained.success_fraction},
        "anchor": {**anchor.as_dict(), "converged_fraction": anchor.success_fraction,
                   "eval_seed": rlmod.EVAL_SEED, "pick": anchor_pick,
                   "what": "repeat the best single action for the whole budget; not in the "
                           "paper, it separates 'the agent is weak' from 'the environment is capped'"},
        "seconds": {"cache": t_cache, "train": out["seconds_train"],
                    "snapshot_eval": out["seconds_snapshot_eval"],
                    "final_eval": t_eval, "wall": time.time() - t0},
        "reference": {"paper": PAPER_QUOTED, "fnorepl": FNOREPL_QUOTED,
                      "note": "theirs, quoted beside ours and never substituted for it; "
                              "both used a different action library from "
                              "this one -- see deviations.thz_actions_are_primitives"},
        "deviations": list(rlmod.DEVIATIONS),
    }
