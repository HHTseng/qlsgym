#!/usr/bin/env python
"""Bounded sequence-architecture study on the fixed ThF+ ``mix`` FNO.

Two independent workers run ``ppo`` and ``sac`` on separate GPUs.  Each worker
screens the same causal encoders with a common seed, then confirms the MLP and
the two best sequence candidates with two common seeds.  Candidate selection
uses FNO validation only; exact cached dynamics are a transfer audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from qlsgym.rl.off_policy import DiscreteSACAgent, SACConfig, sac_policy
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, git_sha, json_safe, rollout_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]
SCREEN_SEED = 23001
CONFIRM_SEEDS = (31001, 31002)
SCREEN_EVAL_SEED = 81001
CONFIRM_FNO_SEED = 82001
CONFIRM_EXACT_SEED = 83001


def candidates() -> list[dict]:
    common = {
        "d_model": 128, "sequence_layers": 2, "n_heads": 4,
        "ff_dim": 256, "dropout": 0.0,
    }
    return [
        {"id": "mlp", "encoder": "mlp", "context_len": 1},
        {"id": "stack_k4", "encoder": "stack", "context_len": 4, **common},
        {"id": "stack_k8", "encoder": "stack", "context_len": 8, **common},
        {"id": "gru_k4", "encoder": "gru", "context_len": 4, **common},
        {"id": "gru_k8", "encoder": "gru", "context_len": 8, **common},
        {"id": "transformer_k1", "encoder": "transformer", "context_len": 1, **common},
        {"id": "transformer_k4", "encoder": "transformer", "context_len": 4, **common},
        {"id": "transformer_k8", "encoder": "transformer", "context_len": 8, **common},
        {
            "id": "transformer_k8_state_only", "encoder": "transformer",
            "context_len": 8, "history_action": False,
            "history_outcome": False, "history_physics": False, **common,
        },
        {
            "id": "transformer_k8_no_position", "encoder": "transformer",
            "context_len": 8, "history_position": False, **common,
        },
        {
            "id": "transformer_ln_k1", "encoder": "transformer", "context_len": 1,
            "history_token_norm": True, **common,
        },
        {
            "id": "transformer_ln_k4", "encoder": "transformer", "context_len": 4,
            "history_token_norm": True, **common,
        },
        {
            "id": "transformer_ln_k8", "encoder": "transformer", "context_len": 8,
            "history_token_norm": True, **common,
        },
        {
            "id": "transformer_ln_k8_state_only", "encoder": "transformer",
            "context_len": 8, "history_action": False,
            "history_outcome": False, "history_physics": False,
            "history_token_norm": True, **common,
        },
        {
            "id": "transformer_ln_k8_no_position", "encoder": "transformer",
            "context_len": 8, "history_position": False,
            "history_token_norm": True, **common,
        },
    ]


PPO_BASE = {
    "clip": 0.1,
    "ent_coef": 0.0005417800781171913,
    "epochs": 8,
    "gae_lambda": 0.98,
    "gamma": 1.0,
    "hidden": 512,
    "lr": 0.001346363908197788,
    "max_grad_norm": 1.0,
    "minibatches": 16,
    "n_envs": 128,
    "n_hidden_layers": 1,
    "n_steps": 32,
    "obs": "sqrt",
    "reward_scale": 0.025,
    "value_target": "qmdp",
    "vf_coef": 0.5,
    "eval_greedy": False,
}

SAC_BASE = {
    "alpha": 0.013268918812005082,
    "autotune_alpha": True,
    "batch_size": 512,
    "depth": 1,
    "gamma": 0.995,
    "gradient_steps": 1,
    "hidden": 128,
    "learning_starts": 1000,
    "lr": 0.00015047792717528454,
    "n_envs": 16,
    "obs": "sqrt",
    "return_scale": 20.0,
    "target_entropy_ratio": 0.49969706052497553,
    "tau": 0.003048864797383012,
    "train_freq": 1,
}


def environment_args(args):
    return argparse.Namespace(
        p_target=0.98, max_pulses=80, rho=1.0,
        device=args.device, progress=False, fno_tag="mix",
        manifest=args.manifest, min_manifest_coverage=1.0,
    )


def score(metrics: dict) -> float:
    """Failure first, with a small failure-penalized action tie breaker."""
    return float(metrics["unfinished_fraction"] + 0.02 * metrics["average_actions"] / 80.0)


def make_config(agent: str, candidate: dict, steps: int, seed: int,
                eval_episodes: int):
    architecture = {key: value for key, value in candidate.items() if key != "id"}
    if agent == "ppo":
        base = dict(PPO_BASE)
        base.update(architecture)
        base.update(
            total_steps=steps, seed=seed, eval_rollouts=eval_episodes,
            eval_every=10**9,
        )
        return PPOConfig(**base)
    base = dict(SAC_BASE)
    base.update(architecture)
    base.update(
        total_steps=steps, seed=seed,
        buffer_size=max(50_000, min(100_000, steps // 2)),
    )
    return SACConfig(**base)


def train_one(agent: str, candidate: dict, steps: int, seed: int,
              eval_episodes: int, fno, exact, stage: str,
              output: Path, contract: dict) -> dict:
    run_id = f"{stage}_{agent}_{candidate['id']}_s{seed}"
    path = output / "runs" / f"{run_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    config = make_config(agent, candidate, steps, seed, min(eval_episodes, 256))
    print(f"[{run_id}] train {steps} steps", flush=True)
    if agent == "ppo":
        result = train_ppo(fno, config, env_eval=fno, log=print)
        state = result.best_state_dict or result.final_state_dict
        policy = policy_from_state_dict(
            state, result.n_in, result.n_actions, config,
            device=fno.device, greedy=False, max_pulses=fno.cfg.max_pulses,
        )
        training = result.as_dict()
        model = {
            "agent": "ppo", "config": asdict(config), "state_dict": state,
            "n_states": result.n_in, "n_actions": result.n_actions,
            "max_pulses": fno.cfg.max_pulses,
        }
        parameters = sum(item.numel() for item in policy.net.parameters())
    else:
        trained = DiscreteSACAgent(fno, config)
        stats = trained.train()
        policy = sac_policy(trained, stochastic=True)
        training = {**stats.as_dict(), "config": asdict(config)}
        model = {
            "agent": "sac", "config": asdict(config),
            "network_state_dict": trained.network.state_dict(),
            "n_states": trained.n_states, "n_actions": trained.n_actions,
            "max_pulses": fno.cfg.max_pulses,
        }
        parameters = sum(item.numel() for item in trained.network.parameters())
    fno_metrics = rollout_metrics(
        fno, policy, eval_episodes,
        SCREEN_EVAL_SEED if stage == "screen" else CONFIRM_FNO_SEED,
        args_eval_batch(eval_episodes),
    )
    exact_metrics = rollout_metrics(
        exact, policy, eval_episodes,
        SCREEN_EVAL_SEED if stage == "screen" else CONFIRM_EXACT_SEED,
        args_eval_batch(eval_episodes),
    )
    model_path = output / "models" / f"{run_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "agent": agent,
        "candidate": candidate,
        "train_seed": seed,
        "steps": steps,
        "parameter_count": parameters,
        "config": asdict(config),
        "training": training,
        "evaluation": {"fno": fno_metrics, "exact": exact_metrics},
        "selection_score_fno": score(fno_metrics),
        "model_path": str(model_path),
        "wall_clock_s": time.time() - started,
        "contract": contract,
        "provenance": {
            "git_sha": git_sha(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(path, record)
    print(
        f"[{run_id}] FNO fail={fno_metrics['unfinished_fraction']:.4f}, "
        f"exact fail={exact_metrics['unfinished_fraction']:.4f}, "
        f"A_exact={exact_metrics['average_actions']:.2f}, "
        f"{record['wall_clock_s'] / 60:.1f} min",
        flush=True,
    )
    del policy, model
    torch.cuda.empty_cache()
    return record


def args_eval_batch(episodes: int) -> int:
    return min(256, episodes)


def selected_candidates(records: list[dict]) -> list[dict]:
    sequence = [record for record in records if record["candidate"]["id"] != "mlp"]
    sequence.sort(key=lambda record: record["selection_score_fno"])
    chosen = [{"id": "mlp", "encoder": "mlp", "context_len": 1}]
    chosen.extend(record["candidate"] for record in sequence[:2])
    return chosen


def run_agent(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, _, fno, exact, _, contract = build_environments(environment_args(args), 128)
    write_json(output / "contract.json", contract)
    deadline = time.monotonic() + args.max_hours * 3600
    screen = []
    for candidate in candidates():
        if time.monotonic() >= deadline:
            print("deadline reached before completing screen", flush=True)
            break
        screen.append(train_one(
            args.agent, candidate, args.screen_steps, SCREEN_SEED,
            args.screen_eval, fno, exact, "screen", output, contract,
        ))
    if len(screen) < len(candidates()):
        return
    chosen = selected_candidates(screen)
    write_json(output / f"selected_{args.agent}.json", chosen)
    for candidate in chosen:
        for seed in CONFIRM_SEEDS:
            if time.monotonic() >= deadline:
                print("deadline reached before completing confirmation", flush=True)
                return
            train_one(
                args.agent, candidate, args.confirm_steps, seed,
                args.confirm_eval, fno, exact, "confirm", output, contract,
            )
    write_json(output / f"complete_{args.agent}.json", {
        "status": "complete", "agent": args.agent, "selected": chosen,
        "finished_at": time.time(),
    })


def aggregate(records: list[dict], dynamics: str) -> dict:
    fields = ("unfinished_fraction", "average_actions", "mean_pulses_successful")
    result = {"training_seeds": len(records)}
    for field in fields:
        values = [record["evaluation"][dynamics][field] for record in records]
        values = [value for value in values if value is not None]
        result[field] = float(np.mean(values))
        result[field + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def compact_metrics(metrics: dict) -> dict:
    """Keep the policy-level quantities needed for architecture comparison."""
    return {
        key: metrics.get(key)
        for key in (
            "n_rollouts", "unfinished_fraction", "average_actions",
            "mean_pulses_successful",
        )
    }


def summarize(args) -> None:
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*.json"))]
    rows = []
    for agent in ("ppo", "sac"):
        groups: dict[str, list[dict]] = {}
        for record in records:
            if record["agent"] == agent and record["stage"] == "confirm":
                groups.setdefault(record["candidate"]["id"], []).append(record)
        for candidate_id, group in sorted(groups.items()):
            rows.append({
                "agent": agent, "candidate": candidate_id,
                "architecture": group[0]["candidate"],
                "parameter_count": group[0]["parameter_count"],
                "wall_clock_s_mean": float(np.mean([item["wall_clock_s"] for item in group])),
                "fno": aggregate(group, "fno"),
                "exact": aggregate(group, "exact"),
            })
    for agent in ("ppo", "sac"):
        control = next(
            (row for row in rows if row["agent"] == agent and row["candidate"] == "mlp"),
            None,
        )
        if control is None:
            continue
        for row in rows:
            if row["agent"] != agent:
                continue
            row["delta_vs_mlp"] = {
                "exact_failure_percentage_points": 100.0 * (
                    row["exact"]["unfinished_fraction"]
                    - control["exact"]["unfinished_fraction"]
                ),
                "exact_average_actions": (
                    row["exact"]["average_actions"]
                    - control["exact"]["average_actions"]
                ),
            }
    screen_rows = []
    for record in records:
        if record["stage"] == "screen":
            screen_rows.append({
                "agent": record["agent"], "candidate": record["candidate"]["id"],
                "steps": record["steps"], "parameter_count": record["parameter_count"],
                "fno": compact_metrics(record["evaluation"]["fno"]),
                "exact": compact_metrics(record["evaluation"]["exact"]),
                "selection_score_fno": record["selection_score_fno"],
            })
    budgets = {}
    for agent in ("ppo", "sac"):
        budgets[agent] = {
            stage: sorted({record["steps"] for record in records
                           if record["agent"] == agent and record["stage"] == stage})
            for stage in ("screen", "confirm")
        }
    base_ppo = json.loads((ROOT / "results/thf_rl_optuna_mix/summary.json").read_text())["rows"]["ppo"]
    base_sac = json.loads((ROOT / "results/thf_rl_optuna_mix_sac_refine/summary.json").read_text())["row"]
    summary = {
        "status": "complete" if all((output / f"complete_{agent}.json").exists()
                                      for agent in ("ppo", "sac")) else "partial",
        "claim": (
            "Architecture comparison under a fixed downloaded mix FNO; exact cached dynamics "
            "are a transfer audit. The full belief is theoretically Markov, so history is "
            "beneficial only empirically under approximation or latent-state effects."
        ),
        "design": {
            "screen_seed": SCREEN_SEED, "confirm_seeds": list(CONFIRM_SEEDS),
            "selection_dynamics": "downloaded mix FNO",
            "transfer_audit": "exact cached tables",
            "budgets_by_agent": budgets,
            "screen_eval": sorted({record["evaluation"]["exact"]["n_rollouts"]
                                   for record in records if record["stage"] == "screen"}),
            "confirm_eval": sorted({record["evaluation"]["exact"]["n_rollouts"]
                                    for record in records if record["stage"] == "confirm"}),
        },
        "historical_five_seed_reference": {
            "ppo": base_ppo, "sac": base_sac,
        },
        "screen_rows": screen_rows,
        "rows": rows,
    }
    write_json(output / "summary.json", summary)
    make_outputs(output, summary)


def make_outputs(output: Path, summary: dict) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        summary["rows"],
        key=lambda row: (
            row["agent"], row["candidate"] != "mlp",
            row["exact"]["unfinished_fraction"],
        ),
    )
    short = {
        "mlp": "MLP", "gru_k8": "GRU K=8", "stack_k4": "stack K=4",
        "transformer_k8_state_only": "Transformer K=8\nstate only",
        "transformer_ln_k8_state_only": "Transformer+LN K=8\nstate only",
    }
    labels = [f"{row['agent'].upper()}\n{short.get(row['candidate'], row['candidate'])}"
              for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.3 * len(rows)), 8), sharex=True)
    axes[0].bar(x - width / 2, [row["fno"]["unfinished_fraction"] for row in rows], width,
                yerr=[row["fno"]["unfinished_fraction_sd"] for row in rows],
                capsize=3, label="mix FNO")
    axes[0].bar(x + width / 2, [row["exact"]["unfinished_fraction"] for row in rows], width,
                yerr=[row["exact"]["unfinished_fraction_sd"] for row in rows],
                capsize=3, label="exact")
    axes[0].set_ylabel("unfinished fraction")
    axes[0].set_title("Two-seed confirmation after fixed-mix-FNO training")
    axes[0].legend()
    axes[1].bar(x - width / 2, [row["fno"]["average_actions"] for row in rows], width)
    axes[1].bar(x + width / 2, [row["exact"]["average_actions"] for row in rows], width)
    axes[1].set_ylabel("failure-penalized average actions")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output / "sequence_vs_mlp.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
    for axis, agent in zip(axes, ("ppo", "sac")):
        screen = sorted(
            (row for row in summary["screen_rows"] if row["agent"] == agent),
            key=lambda row: row["fno"]["unfinished_fraction"],
        )
        y = np.arange(len(screen))
        axis.barh(y + 0.18, [row["exact"]["unfinished_fraction"] for row in screen],
                  0.36, label="exact")
        axis.barh(y - 0.18, [row["fno"]["unfinished_fraction"] for row in screen],
                  0.36, label="mix FNO")
        axis.set_yticks(y, [row["candidate"] for row in screen], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(agent.upper())
        axis.set_xlabel("unfinished fraction")
        axis.set_xlim(0, 1.02)
    axes[0].legend()
    fig.suptitle("Single-seed architecture screen (selection used mix FNO)")
    fig.tight_layout()
    fig.savefig(output / "sequence_screen.png", dpi=180)
    plt.close(fig)

    lines = [
        "# ThF+ sequence-aware PPO/SAC study", "",
        "Training dynamics: fixed downloaded `mix` FNO. Candidate selection uses FNO validation; "
        "exact cached dynamics are a held-out transfer audit.", "",
        "| agent | encoder | exact unfinished | exact actions | delta failure vs MLP | delta actions vs MLP | FNO unfinished | parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["delta_vs_mlp"]
        lines.append(
            f"| {row['agent']} | {row['candidate']} | "
            f"{row['exact']['unfinished_fraction']:.3f} | {row['exact']['average_actions']:.2f} | "
            f"{delta['exact_failure_percentage_points']:+.2f} pp | "
            f"{delta['exact_average_actions']:+.2f} | "
            f"{row['fno']['unfinished_fraction']:.3f} | "
            f"{row['parameter_count']:,} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-agent")
    run.add_argument("agent", choices=("ppo", "sac"))
    run.add_argument("--device", default="cuda")
    run.add_argument("--manifest", default=os.path.expandvars(
        "$QLSGYM_WORK/checkpoints/thf/mix.json"
    ))
    run.add_argument("--output", default="results/thf_sequence_rl_mix")
    run.add_argument("--screen-steps", type=int, default=100_000)
    run.add_argument("--confirm-steps", type=int, default=300_000)
    run.add_argument("--screen-eval", type=int, default=500)
    run.add_argument("--confirm-eval", type=int, default=2_000)
    run.add_argument("--max-hours", type=float, default=5.75)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", default="results/thf_sequence_rl_mix")
    summary.add_argument("--screen-steps", type=int, default=100_000)
    summary.add_argument("--confirm-steps", type=int, default=300_000)
    summary.add_argument("--screen-eval", type=int, default=500)
    summary.add_argument("--confirm-eval", type=int, default=2_000)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "run-agent":
        run_agent(arguments)
    else:
        summarize(arguments)
