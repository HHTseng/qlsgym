#!/usr/bin/env python
"""Diagnose and improve the failure tail of the FNO-pretrained ThF+ PPO policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym.policies import BatchedFallbackPolicy, PhysicsEliminationPolicy
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from thf_rl_agents import build_environments, rollout_metrics, write_json


FAILURE_PENALTIES = (0.0, 20.0, 40.0, 80.0)
RISK_CANDIDATES = tuple(
    (penalty, include_budget)
    for include_budget in (False, True)
    for penalty in FAILURE_PENALTIES
)
FALLBACK_CANDIDATES = (
    (0, 0),
    (10, 0), (20, 0), (30, 0), (40, 0),
    (0, 5), (0, 8),
    (20, 5), (20, 8), (30, 5), (30, 8), (40, 5), (40, 8),
)
FINAL_SEEDS = tuple(range(5))
FALLBACK_EVAL_SEED = 81_101
RISK_EVAL_SEED = 81_201
FINAL_EVAL_SEED = 82_101


def parse_indices(value: str, upper: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(upper))
    indices = [int(item) for item in value.split(",") if item.strip()]
    if not indices or min(indices) < 0 or max(indices) >= upper:
        raise ValueError(f"indices must lie in [0, {upper})")
    return indices


def env_namespace(args):
    return argparse.Namespace(
        device=args.device,
        p_target=0.98,
        max_pulses=80,
        rho=0.0,
        progress=False,
        fno_tag="mix",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )


def environments(args):
    molecule, library, fno, exact, _, contract = build_environments(env_namespace(args), 128)
    return molecule, library, exact, fno, contract


def source_path(args, seed: int) -> Path:
    return Path(args.source_models) / f"confirmation_source_s{seed}.pt"


def checkpoint_policy(path: Path, device: str, max_pulses: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PPOConfig(**checkpoint["config"])
    state = checkpoint["state_dict"]
    n_in = int(state["trunk.0.weight"].shape[1])
    n_actions = int(state["pi.weight"].shape[0])
    policy = policy_from_state_dict(
        state, n_in, n_actions, cfg, device=device, greedy=False, max_pulses=max_pulses,
    )
    return checkpoint, policy


def fallback_policy(actor, molecule, library, max_pulses, switch_remaining, stagnation_steps):
    physics = PhysicsEliminationPolicy(molecule, library, tau_rule="pi")
    return BatchedFallbackPolicy(
        actor,
        physics,
        max_pulses=max_pulses,
        switch_remaining=switch_remaining,
        stagnation_steps=stagnation_steps,
        min_purity_gain=1e-3,
        action_encoder=library.encode,
    )


def candidate_name(switch_remaining: int, stagnation_steps: int) -> str:
    return f"remaining_{switch_remaining:02d}_stagnation_{stagnation_steps:02d}"


def risk_name(penalty: float, include_budget: bool) -> str:
    return f"penalty_{int(penalty):03d}_budget_{int(include_budget)}"


def fallback_diagnose(args):
    output = Path(args.output)
    destination = output / "fallback_diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    molecule, library, exact, _, contract = environments(args)
    write_json(output / "contract.json", contract)
    _, actor = checkpoint_policy(source_path(args, 0), args.device, exact.cfg.max_pulses)
    for switch_remaining, stagnation_steps in FALLBACK_CANDIDATES:
        name = candidate_name(switch_remaining, stagnation_steps)
        path = destination / f"{name}.json"
        if path.exists():
            continue
        print(f"[fallback] {name}", flush=True)
        policy = fallback_policy(
            actor, molecule, library, exact.cfg.max_pulses,
            switch_remaining, stagnation_steps,
        )
        metrics = rollout_metrics(
            exact, policy, args.fallback_eval, FALLBACK_EVAL_SEED, args.eval_batch,
        )
        write_json(path, {
            "name": name,
            "switch_remaining": switch_remaining,
            "stagnation_steps": stagnation_steps,
            "min_purity_gain": 1e-3,
            "source_seed": 0,
            "evaluation": {"exact": metrics},
        })


def fallback_select(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "fallback_diagnosis").glob("*.json"))
    ]
    if len(records) != len(FALLBACK_CANDIDATES):
        raise RuntimeError(
            f"expected {len(FALLBACK_CANDIDATES)} fallback records, found {len(records)}"
        )
    winner = max(
        records,
        key=lambda row: (
            row["evaluation"]["exact"]["success_fraction"],
            -row["evaluation"]["exact"]["average_actions"],
        ),
    )
    selection = {
        key: winner[key]
        for key in ("name", "switch_remaining", "stagnation_steps", "min_purity_gain")
    }
    selection["validation_exact"] = {
        key: winner["evaluation"]["exact"][key]
        for key in ("success_fraction", "unfinished_fraction", "average_actions")
    }
    selection["rule"] = "maximize exact validation success, then minimize exact actions"
    write_json(output / "fallback_selection.json", selection)
    print(json.dumps(selection, indent=2))


def risk_config(steps: int, seed: int, penalty: float, include_budget: bool) -> PPOConfig:
    return PPOConfig(
        n_envs=128,
        n_steps=32,
        total_steps=steps,
        epochs=8,
        minibatches=16,
        lr=6e-4,
        gamma=1.0,
        gae_lambda=0.98,
        clip=0.1,
        ent_coef=0.0005417800781171913,
        vf_coef=0.5,
        max_grad_norm=1.0,
        hidden=512,
        n_hidden_layers=1,
        obs="sqrt",
        include_budget=include_budget,
        value_target="qmdp",
        reward_scale=0.025,
        failure_penalty=penalty,
        eval_every=max(1, math.ceil(steps / (128 * 32))),
        eval_rollouts=256,
        eval_greedy=False,
        seed=seed,
    )


def initial_state(checkpoint: dict, include_budget: bool, n_states: int) -> dict:
    state = {key: value.detach().cpu().clone() for key, value in checkpoint["state_dict"].items()}
    weight = state["trunk.0.weight"]
    if include_budget and weight.shape[1] == n_states:
        state["trunk.0.weight"] = torch.cat(
            (weight, torch.zeros((weight.shape[0], 1), dtype=weight.dtype)), dim=1,
        )
    expected = n_states + int(include_budget)
    if state["trunk.0.weight"].shape[1] != expected:
        raise ValueError("source checkpoint input size is incompatible with requested budget feature")
    return state


def run_risk_training(args, exact, source_seed, penalty, include_budget, steps, episodes, eval_seed, stage):
    checkpoint = torch.load(source_path(args, source_seed), map_location="cpu", weights_only=False)
    cfg = risk_config(steps, 83_000 + source_seed, penalty, include_budget)
    started = time.time()
    trained = train_ppo(
        exact,
        cfg,
        env_eval=exact,
        log=print,
        initial_state_dict=initial_state(checkpoint, include_budget, exact.n_states),
    )
    state = trained.best_state_dict or trained.final_state_dict
    policy = policy_from_state_dict(
        state,
        trained.n_in,
        trained.n_actions,
        cfg,
        device=args.device,
        greedy=False,
        max_pulses=exact.cfg.max_pulses,
    )
    metrics = rollout_metrics(exact, policy, episodes, eval_seed, args.eval_batch)
    model_path = Path(args.output) / "models" / f"{stage}_{risk_name(penalty, include_budget)}_s{source_seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "agent": "ppo",
        "config": asdict(cfg),
        "state_dict": state,
        "source_model": str(source_path(args, source_seed)),
        "source_seed": source_seed,
    }, model_path)
    return {
        "stage": stage,
        "source_seed": source_seed,
        "failure_penalty": penalty,
        "include_budget": include_budget,
        "fine_tune_steps": steps,
        "config": asdict(cfg),
        "training": trained.as_dict(),
        "evaluation": {"exact": metrics},
        "model_path": str(model_path),
        "seconds": time.time() - started,
    }, policy


def risk_diagnose(args):
    output = Path(args.output)
    destination = output / "risk_diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    _, _, exact, _, contract = environments(args)
    write_json(output / "contract.json", contract)
    indices = parse_indices(args.candidate_indices, len(RISK_CANDIDATES))
    for index in indices:
        penalty, include_budget = RISK_CANDIDATES[index]
        name = risk_name(penalty, include_budget)
        path = destination / f"{name}.json"
        if path.exists():
            continue
        print(f"[risk] {name}", flush=True)
        record, _ = run_risk_training(
            args, exact, 0, penalty, include_budget,
            args.risk_diagnose_steps, args.risk_eval, RISK_EVAL_SEED, "diagnosis",
        )
        write_json(path, record)
        torch.cuda.empty_cache()


def risk_select(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "risk_diagnosis").glob("*.json"))
    ]
    if len(records) != len(RISK_CANDIDATES):
        raise RuntimeError(f"expected {len(RISK_CANDIDATES)} risk records, found {len(records)}")
    winner = max(
        records,
        key=lambda row: (
            row["evaluation"]["exact"]["success_fraction"],
            -row["evaluation"]["exact"]["average_actions"],
        ),
    )
    selection = {
        "name": risk_name(winner["failure_penalty"], winner["include_budget"]),
        "failure_penalty": winner["failure_penalty"],
        "include_budget": winner["include_budget"],
        "validation_exact": {
            key: winner["evaluation"]["exact"][key]
            for key in ("success_fraction", "unfinished_fraction", "average_actions")
        },
        "rule": "maximize exact validation success, then minimize exact actions",
    }
    write_json(output / "risk_selection.json", selection)
    print(json.dumps(selection, indent=2))


def evaluate_variants(args, molecule, library, exact, fno, source_seed, risk_policy, risk_record):
    _, source_actor = checkpoint_policy(source_path(args, source_seed), args.device, exact.cfg.max_pulses)
    fallback = json.loads((Path(args.output) / "fallback_selection.json").read_text())

    def with_fallback(actor):
        return fallback_policy(
            actor, molecule, library, exact.cfg.max_pulses,
            int(fallback["switch_remaining"]), int(fallback["stagnation_steps"]),
        )

    variants = {
        "source_actor": source_actor,
        "source_fallback": with_fallback(source_actor),
        "risk_actor": risk_policy,
        "risk_fallback": with_fallback(risk_policy),
    }
    evaluations = {}
    for name, policy in variants.items():
        if name == "risk_actor":
            exact_metrics = risk_record["evaluation"]["exact"]
        else:
            print(f"[confirm s{source_seed}] evaluate {name} exact", flush=True)
            exact_metrics = rollout_metrics(
                exact, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch,
            )
        print(f"[confirm s{source_seed}] evaluate {name} FNO", flush=True)
        fno_metrics = rollout_metrics(
            fno, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch,
        )
        evaluations[name] = {"exact": exact_metrics, "fno": fno_metrics}
    risk_record["evaluation"] = evaluations
    risk_record["fallback"] = fallback
    return risk_record


def confirm(args):
    output = Path(args.output)
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    molecule, library, exact, fno, _ = environments(args)
    selection = json.loads((output / "risk_selection.json").read_text())
    seeds = parse_indices(args.seeds, len(FINAL_SEEDS))
    for source_seed in seeds:
        path = destination / f"source_s{source_seed}.json"
        if path.exists():
            continue
        print(f"[confirm] source seed {source_seed}", flush=True)
        record, policy = run_risk_training(
            args,
            exact,
            source_seed,
            float(selection["failure_penalty"]),
            bool(selection["include_budget"]),
            args.confirm_steps,
            args.final_eval,
            FINAL_EVAL_SEED,
            "confirmation",
        )
        record = evaluate_variants(
            args, molecule, library, exact, fno, source_seed, policy, record,
        )
        write_json(path, record)
        torch.cuda.empty_cache()


def aggregate(records: list[dict], variant: str, engine: str) -> dict:
    keys = ("success_fraction", "unfinished_fraction", "average_actions", "mean_pulses_successful")
    result = {}
    for key in keys:
        values = np.asarray([
            row["evaluation"][variant][engine][key] for row in records
        ], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    failures = np.asarray([
        row["evaluation"][variant][engine]["unfinished_fraction"] for row in records
    ], dtype=float)
    half = 2.776 * failures.std(ddof=1) / math.sqrt(len(failures))
    result["unfinished_fraction_seed_ci95"] = [
        float(max(0.0, failures.mean() - half)),
        float(min(1.0, failures.mean() + half)),
    ]
    return result


def paired_changes(records: list[dict], variant: str) -> dict:
    rescued = lost = retained_success = retained_failure = 0
    for row in records:
        base = np.asarray(row["evaluation"]["source_actor"]["exact"]["successes"], dtype=bool)
        other = np.asarray(row["evaluation"][variant]["exact"]["successes"], dtype=bool)
        rescued += int((~base & other).sum())
        lost += int((base & ~other).sum())
        retained_success += int((base & other).sum())
        retained_failure += int((~base & ~other).sum())
    total = rescued + lost + retained_success + retained_failure
    return {
        "episodes": total,
        "rescued": rescued,
        "lost": lost,
        "retained_success": retained_success,
        "retained_failure": retained_failure,
        "net_rescue_fraction": (rescued - lost) / total,
    }


def summarize(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "confirmation").glob("source_s*.json"))
    ]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    reference = json.loads(
        (output.parent / "thf_fno_rl_refinement" / "summary.json").read_text()
    )
    physics = next(row for row in reference["rows"] if row["name"] == "Physics elimination")
    variants = ("source_actor", "source_fallback", "risk_actor", "risk_fallback")
    labels = {
        "source_actor": "Hybrid PPO",
        "source_fallback": "Hybrid PPO + physics fallback",
        "risk_actor": "Failure-sensitive PPO",
        "risk_fallback": "Failure-sensitive PPO + physics fallback",
    }
    rows = {}
    for variant in variants:
        rows[variant] = {
            "label": labels[variant],
            "exact": aggregate(records, variant, "exact"),
            "fno": aggregate(records, variant, "fno"),
        }
        if variant != "source_actor":
            rows[variant]["paired_vs_source_actor"] = paired_changes(records, variant)
    for row in rows.values():
        exact = row["exact"]
        row["dominates_physics_point_estimate"] = (
            exact["unfinished_fraction"] < physics["exact"]["unfinished_fraction"]
            and exact["average_actions"] < physics["exact"]["average_actions"]
        )
        row["failure_seed_ci_below_physics_point_estimate"] = (
            exact["unfinished_fraction_seed_ci95"][1]
            < physics["exact"]["unfinished_fraction"]
        )
    summary = {
        "status": "complete",
        "method": "FNO-pretrained PPO, cached-exact failure-sensitive fine-tuning, and physics fallback",
        "fallback_selection": json.loads((output / "fallback_selection.json").read_text()),
        "risk_selection": json.loads((output / "risk_selection.json").read_text()),
        "training_seeds": len(records),
        "evaluation_episodes_per_seed": args.final_eval,
        "physics_reference": physics,
        "rows": rows,
        "target": {"unfinished_fraction": 0.25, "average_actions": 48.0},
    }
    write_json(output / "summary.json", summary)

    names = ["Physics elimination"] + [labels[key] for key in variants]
    failures = [100 * physics["exact"]["unfinished_fraction"]] + [
        100 * rows[key]["exact"]["unfinished_fraction"] for key in variants
    ]
    actions = [physics["exact"]["average_actions"]] + [
        rows[key]["exact"]["average_actions"] for key in variants
    ]
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#e45756"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(x, failures, color=colors)
    axes[0].axhline(100 * physics["exact"]["unfinished_fraction"], color="black", ls="--", lw=1)
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, actions, color=colors)
    axes[1].axhline(physics["exact"]["average_actions"], color="black", ls="--", lw=1)
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, names, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ThF+ failure-tail improvement under exact dynamics")
    fig.savefig(output / "safe_hybrid_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# ThF+ FNO-assisted safe-hybrid study",
        "",
        "Primary order: exact unfinished fraction, then exact failure-penalized actions.",
        "",
        "| Controller | Exact unfinished | Exact actions | FNO unfinished | FNO actions |",
        "|---|---:|---:|---:|---:|",
        f"| Physics elimination | {100*physics['exact']['unfinished_fraction']:.2f}% | "
        f"{physics['exact']['average_actions']:.2f} | {100*physics['fno']['unfinished_fraction']:.2f}% | "
        f"{physics['fno']['average_actions']:.2f} |",
    ]
    for key in variants:
        row = rows[key]
        lines.append(
            f"| {row['label']} | {100*row['exact']['unfinished_fraction']:.2f}% | "
            f"{row['exact']['average_actions']:.2f} | {100*row['fno']['unfinished_fraction']:.2f}% | "
            f"{row['fno']['average_actions']:.2f} |"
        )
    lines += [
        "",
        "The summary JSON contains seed standard deviations, seed-level confidence intervals, "
        "paired rescue/loss counts, and both promotion decisions.",
        "",
        "![Safe hybrid comparison](safe_hybrid_comparison.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "fallback_diagnosis": len(list((output / "fallback_diagnosis").glob("*.json"))),
        "fallback_selected": (output / "fallback_selection.json").exists(),
        "risk_diagnosis": len(list((output / "risk_diagnosis").glob("*.json"))),
        "risk_selected": (output / "risk_selection.json").exists(),
        "confirmation": len(list((output / "confirmation").glob("source_s*.json"))),
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument(
        "command",
        choices=(
            "fallback-diagnose", "fallback-select", "risk-diagnose", "risk-select",
            "confirm", "summarize", "status",
        ),
    )
    main.add_argument("--output", default="results/thf_safe_hybrid")
    main.add_argument("--source-models", default="results/thf_mix_ppo_exact_finetune/models")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--candidate-indices", default="all")
    main.add_argument("--seeds", default="all")
    main.add_argument("--fallback-eval", type=int, default=2_000)
    main.add_argument("--risk-diagnose-steps", type=int, default=100_000)
    main.add_argument("--risk-eval", type=int, default=2_000)
    main.add_argument("--confirm-steps", type=int, default=250_000)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


def main():
    args = parser().parse_args()
    command = args.command.replace("-", "_")
    globals()[command](args)


if __name__ == "__main__":
    main()
