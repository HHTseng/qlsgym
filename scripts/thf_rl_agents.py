#!/usr/bin/env python
"""Train PPO, categorical SAC, or Double DQN in a qlsgym FNO environment.

Training uses the surrogate-backed ``FnoEnv``.  Model selection and the final
reported score use the exact ThF+ table environment with a disjoint outcome
seed. The compact 312-action ThF+ physics library is used throughout.

Examples
--------
Run one smoke job::

    python scripts/thf_rl_agents.py run \
      --agent ppo --preset smoke --manifest /path/to/thf/pilot.json

Aggregate all JSON records and draw the exact-holdout comparison::

    python scripts/thf_rl_agents.py summarize \
      --input results/qlsgym_fno/runs --output results/qlsgym_fno
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from qlsgym.rl.off_policy import (
    DDQNAgent,
    DDQNConfig,
    DiscreteSACAgent,
    SACConfig,
    ddqn_policy,
    sac_policy,
)


ROOT = Path(__file__).resolve().parents[1]
AGENTS = (
    "sweeping", "random", "physics_elimination", "ppo", "sac_discrete", "ddqn"
)
PRESETS = {
    "smoke": {
        "ppo_steps": 8_192,
        "offpolicy_steps": 8_192,
        "n_envs_ppo": 64,
        "n_envs_offpolicy": 32,
        "learning_starts": 1_024,
        "eval_episodes": 200,
    },
    "pilot": {
        "ppo_steps": 200_000,
        "offpolicy_steps": 200_000,
        "n_envs_ppo": 128,
        "n_envs_offpolicy": 32,
        "learning_starts": 4_096,
        "eval_episodes": 1_000,
    },
    "primary": {
        "ppo_steps": 2_000_000,
        "offpolicy_steps": 1_000_000,
        "n_envs_ppo": 256,
        "n_envs_offpolicy": 64,
        "learning_starts": 10_000,
        "eval_episodes": 5_000,
    },
    # Equal transition and vectorization budgets for the locked ranking.
    "final": {
        "ppo_steps": 1_000_000,
        "offpolicy_steps": 1_000_000,
        "n_envs_ppo": 128,
        "n_envs_offpolicy": 128,
        "learning_starts": 10_240,
        "eval_episodes": 5_000,
    },
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def git_sha(directory: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True
        ).strip()
    except Exception:
        return "unknown"


def library_taus(library) -> np.ndarray:
    primitive = [
        library.primitive_tau_index(index) for index in range(library.n_primitives)
    ]
    return np.unique(
        np.concatenate((library.tau_indices, np.asarray(primitive, dtype=np.int64)))
    )


def build_environments(args, batch: int):
    import qlsgym
    from qlsgym.env.actions import ActionLibrary
    from qlsgym.env.cache import build_action_tables
    from qlsgym.env.env import EnvConfig, PurificationEnv
    from qlsgym.physics.engines import ExactEngine
    from qlsgym.surrogate.fno_env import FnoEnv
    from qlsgym.surrogate.manifest import load_manifest, sha256_file

    molecule = qlsgym.load_molecule("thf")
    library = ActionLibrary.physics_subset(molecule)
    config = EnvConfig(
        p_target=args.p_target,
        max_pulses=args.max_pulses,
        rho=args.rho,
        penalty_mode="indicator",
    )
    tables = build_action_tables(
        molecule, library, device=args.device, progress=args.progress
    )
    exact = PurificationEnv(
        molecule, library, tables, config, device=args.device, batch=batch
    )
    fallback = ExactEngine(molecule, tau_indices=library_taus(library))
    engine = load_manifest(
        molecule,
        args.fno_tag,
        tau_indices=library_taus(library),
        device=args.device,
        fallback=fallback,
        path=args.manifest,
    )
    covered = len(engine.trained)
    possible = 2 * molecule.system.n_blocks
    coverage = covered / possible
    if coverage < args.min_manifest_coverage:
        raise RuntimeError(
            f"manifest covers {covered}/{possible} block-polarization pairs "
            f"({coverage:.1%}), below --min-manifest-coverage "
            f"{args.min_manifest_coverage:.1%}"
        )
    fno = FnoEnv(
        molecule, library, tables, engine, config, device=args.device, batch=batch
    )
    contract = {
        "molecule": molecule.name,
        "molecule_display": "ThF+",
        "molecule_fingerprint": molecule.fingerprint(),
        "qlsgym_git_sha": git_sha(Path(qlsgym.__file__).resolve().parents[2]),
        "n_states": molecule.n_states,
        "n_nu": molecule.trap.n_nu,
        "n_actions": library.n_actions,
        "n_grid_actions": library.n_grid,
        "n_primitives": library.n_primitives,
        "library_tag": library.tag(),
        "p_target": fno.cfg.p_target,
        "max_pulses": fno.cfg.max_pulses,
        "rho": fno.cfg.rho,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_tag": engine.manifest.tag,
        "manifest_fingerprint": engine.manifest.fingerprint,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "checkpoint_sha256": {key: sha256_file(entry.path)
                              for key, entry in sorted(engine.manifest.entries.items())},
        "covered_block_polarizations": sorted([list(item) for item in engine.trained]),
        "manifest_pair_coverage": coverage,
        "train_dynamics": "FnoEnv: covered Raman blocks use FNO; primitives and uncovered blocks remain exact",
        "evaluation_dynamics": "exact qlsgym action tables",
    }
    return molecule, library, fno, exact, engine, contract


def rollout_metrics(environment, policy, episodes: int, seed: int, batch: int = 128) -> dict:
    from qlsgym.rl.evaluation import evaluate_policy

    return evaluate_policy(environment, policy, episodes, seed, batch)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_agent(args, fno, exact):
    preset = PRESETS[args.preset]
    model_path = Path(args.output) / "models" / f"{args.preset}_{args.agent}_s{args.seed}.pt"
    if model_path.exists() and args.agent in ("ppo", "sac_discrete", "ddqn"):
        raise FileExistsError(f"model exists: {model_path}; choose a new --output")
    if args.agent == "sweeping":
        from qlsgym.policies.baselines import SweepingPolicy

        return SweepingPolicy(fno.n_actions), {"kind": "non-learning baseline"}, None
    if args.agent == "random":
        from qlsgym.policies.baselines import RandomPolicy

        return RandomPolicy(fno.n_actions), {"kind": "uniform-random baseline"}, None
    if args.agent == "physics_elimination":
        from qlsgym.policies.baselines import PhysicsEliminationPolicy

        policy = PhysicsEliminationPolicy(fno.molecule, fno.library, tau_rule="pi")
        return policy, {"kind": "ThF+ physics-designed non-learning baseline"}, None

    if args.agent == "ppo":
        from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo

        config = PPOConfig(
            n_envs=preset["n_envs_ppo"],
            n_steps=32,
            total_steps=preset["ppo_steps"],
            epochs=4,
            minibatches=8,
            lr=3e-4 if args.preset == "final" else 1e-3,
            gamma=1.0,
            gae_lambda=0.95,
            clip=0.2,
            ent_coef=0.01,
            hidden=256,
            n_hidden_layers=2,
            obs="sqrt",
            value_target="qmdp_gae" if args.preset == "final" else "qmdp",
            eval_every=max(1, math.ceil(preset["ppo_steps"] / (
                preset["n_envs_ppo"] * 32
            ))),
            eval_rollouts=min(200, preset["eval_episodes"]),
            eval_greedy=False,
            seed=args.seed,
        )
        result = train_ppo(fno, config, env_eval=exact, log=print)
        state = result.best_state_dict or result.final_state_dict
        save_state(
            model_path,
            {"agent": args.agent, "config": config.as_dict(), "state_dict": state},
        )
        policy = policy_from_state_dict(
            state, result.n_in, result.n_actions, config, device=args.device, greedy=False
        )
        return policy, result.as_dict(), model_path

    if args.agent == "ddqn":
        config = DDQNConfig(
            n_envs=preset["n_envs_offpolicy"],
            total_steps=preset["offpolicy_steps"],
            learning_starts=preset["learning_starts"],
            buffer_size=max(20_000, preset["offpolicy_steps"] // 2),
            eps_fraction=0.72 if args.preset == "final" else 0.35,
            gradient_steps=4 if args.preset == "final" else 1,
            seed=args.seed,
        )
        agent = DDQNAgent(fno, config)
        stats = agent.train()
        save_state(
            model_path,
            {"agent": args.agent, "config": asdict(config), "state_dict": agent.online.state_dict()},
        )
        return ddqn_policy(agent), {
            **stats.as_dict(), "config": asdict(config),
            "reward_scale": 1.0 / args.max_pulses,
        }, model_path

    config = SACConfig(
        n_envs=preset["n_envs_offpolicy"],
        total_steps=preset["offpolicy_steps"],
        learning_starts=preset["learning_starts"],
        buffer_size=max(20_000, preset["offpolicy_steps"] // 2),
        gradient_steps=4 if args.preset == "final" else 1,
        # Critic rewards are divided by H. Scale temperature in the same units
        # so this normalization does not amplify entropy regularization by H.
        alpha=args.sac_alpha_raw / args.max_pulses,
        seed=args.seed,
    )
    agent = DiscreteSACAgent(fno, config)
    stats = agent.train()
    save_state(
        model_path,
        {
            "agent": args.agent,
            "config": asdict(config),
            "actor_state_dict": agent.network.actor.state_dict(),
            "critic1_state_dict": agent.network.q1.state_dict(),
            "critic2_state_dict": agent.network.q2.state_dict(),
            "alpha": float(agent.network.log_alpha.detach().exp()),
        },
    )
    return sac_policy(agent, stochastic=True), {
        **stats.as_dict(), "config": asdict(config),
        "reward_scale": 1.0 / args.max_pulses,
        "initial_alpha_raw_reward_units": args.sac_alpha_raw,
    }, model_path


def run(args) -> None:
    if args.agent not in AGENTS:
        raise ValueError(f"--agent must be one of {AGENTS}")
    preset = PRESETS[args.preset]
    path = Path(args.output) / "runs" / f"{args.preset}_{args.agent}_s{args.seed}.json"
    if path.exists():
        raise FileExistsError(f"result exists: {path}; choose a new --output")
    _, _, fno, exact, engine, contract = build_environments(
        args, max(preset["n_envs_ppo"], preset["n_envs_offpolicy"])
    )
    started = time.time()
    policy, training, model_path = train_agent(args, fno, exact)
    exact_result = rollout_metrics(
        exact, policy, preset["eval_episodes"], args.eval_seed, args.eval_batch
    )
    fno_result = rollout_metrics(
        fno, policy, preset["eval_episodes"], args.eval_seed, args.eval_batch
    )
    record = {
        "schema_version": 3,
        "job": {
            "agent": args.agent,
            "preset": args.preset,
            "train_seed": args.seed,
            "eval_seed": args.eval_seed,
            "eval_episodes": preset["eval_episodes"],
            "evaluation_batch": args.eval_batch,
        },
        "contract": contract,
        "training": training,
        "evaluation": {"exact": exact_result, "fno": fno_result},
        "surrogate_calls": dict(engine.calls),
        "surrogate_fraction": engine.surrogate_fraction(),
        "surrogate_fraction_scope": "engine calls only; FnoEnv exact primitives bypass this counter",
        "model_path": None if model_path is None else str(model_path),
        "wall_clock_s": time.time() - started,
        "provenance": {
            "git_sha": git_sha(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": args.device,
        },
    }
    write_json(path, record)
    print(f"wrote {path}")
    print(json.dumps(json_safe({key: value for key, value in exact_result.items()
                              if key not in ("lengths", "actual_lengths", "successes")}), indent=2))


def load_records(directory: Path) -> list[dict]:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"no JSON records under {directory}")
    fingerprints = {record["contract"]["molecule_fingerprint"] for record in records}
    libraries = {record["contract"]["library_tag"] for record in records}
    manifests = {
        (record["contract"]["manifest_tag"], record["contract"].get("manifest_sha256"),
         json.dumps(record["contract"].get("checkpoint_sha256"), sort_keys=True))
        for record in records
    }
    tasks = {
        (record["contract"]["p_target"], record["contract"]["max_pulses"],
         record["contract"]["rho"], record["job"]["preset"])
        for record in records
    }
    if any(len(items) != 1 for items in (fingerprints, libraries, manifests, tasks)):
        raise ValueError("records mix molecule, library, FNO manifest, task, or training-budget contracts")
    for agent in ("ppo", "sac_discrete", "ddqn"):
        settings = set()
        for record in records:
            if record["job"].get("agent") == agent:
                config = dict(record["training"].get("config", {}))
                config.pop("seed", None)
                settings.add(json.dumps(config, sort_keys=True))
        if len(settings) > 1:
            raise ValueError(f"records mix {agent} hyperparameter contracts")
    return records


def summarize(args) -> None:
    records = load_records(Path(args.input))
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["job"]["agent"], []).append(record)
    rows = []
    for agent in AGENTS:
        group = groups.get(agent, [])
        if not group:
            continue
        exact = [record["evaluation"]["exact"] for record in group]
        fno = [record["evaluation"]["fno"] for record in group]
        row = {
            "agent": agent,
            "evaluation_runs": len(group),
            "training_seeds": (
                len({record["job"]["train_seed"] for record in group})
                if agent in ("ppo", "sac_discrete", "ddqn") else 0
            ),
            "exact_average_actions": float(np.mean([item["average_actions"] for item in exact])),
            "exact_unfinished_fraction": float(np.mean([item["unfinished_fraction"] for item in exact])),
            "exact_p85_actions": float(np.mean([item["p85_actions"] for item in exact])),
            "exact_cvar10_actions": float(np.mean([item["cvar10_actions"] for item in exact])),
            "fno_average_actions": float(np.mean([item["average_actions"] for item in fno])),
            "fno_unfinished_fraction": float(np.mean([item["unfinished_fraction"] for item in fno])),
        }
        row["sim_to_exact_gap"] = (
            row["exact_average_actions"] - row["fno_average_actions"]
        )
        rows.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", rows)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["agent"].replace("_", " ").upper() for row in rows]
    horizon = records[0]["contract"]["max_pulses"]
    preset = records[0]["job"]["preset"]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, [row["exact_average_actions"] for row in rows], color="#0072B2")
    axes[0].set_ylabel(f"Average actions (failures count as {horizon})\nlower is better")
    axes[1].bar(
        x,
        [100 * row["exact_unfinished_fraction"] for row in rows],
        color="#D55E00",
    )
    axes[1].set_ylabel("Unfinished/failure rate (%)\nlower is better")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_xlabel("Controller (exact ThF+ holdout)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"ThF+ exact holdout — {preset} run (not a final RL ranking)")
    figure.tight_layout()
    figure.savefig(output / "thf_fno_agent_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(rows, indent=2))


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--agent", required=True, choices=AGENTS)
    run_parser.add_argument("--preset", default="smoke", choices=tuple(PRESETS))
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--fno-tag", default="rlpilot")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--eval-seed", type=int, default=20_001)
    run_parser.add_argument("--eval-batch", type=int, default=128)
    run_parser.add_argument("--output", default=str(ROOT / "results" / "qlsgym_fno"))
    run_parser.add_argument("--p-target", type=float, default=0.98)
    run_parser.add_argument("--max-pulses", type=int, default=80)
    run_parser.add_argument("--rho", type=float, default=0.0)
    run_parser.add_argument("--sac-alpha-raw", type=float, default=0.05,
                            help="initial entropy temperature in unnormalized -1 pulse-reward units")
    run_parser.add_argument("--min-manifest-coverage", type=float, default=0.5)
    run_parser.add_argument("--progress", action="store_true")
    run_parser.set_defaults(func=run)

    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument(
        "--input", default=str(ROOT / "results" / "qlsgym_fno" / "runs")
    )
    summary_parser.add_argument(
        "--output", default=str(ROOT / "results" / "qlsgym_fno")
    )
    summary_parser.set_defaults(func=summarize)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
