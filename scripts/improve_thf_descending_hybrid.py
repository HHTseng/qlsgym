#!/usr/bin/env python
"""Combine the failure-sensitive PPO actor with the strongest exact-table heuristic."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym.policies import BatchedFallbackPolicy, DescendingPopulationPolicy
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict
from thf_rl_agents import build_environments, rollout_metrics, write_json


PREFIX_PULSES = (0, 2, 5, 10, 15, 20, 30, 40, 60, 80)
DIAGNOSIS_SEED = 83_101
FINAL_EVAL_SEED = 82_101
FINAL_SEEDS = tuple(range(5))


def environments(args):
    namespace = argparse.Namespace(
        device=args.device,
        p_target=0.98,
        max_pulses=80,
        rho=0.0,
        progress=False,
        fno_tag="mix",
        manifest=str(Path(args.manifest).expanduser().resolve()),
        min_manifest_coverage=1.0,
    )
    molecule, library, fno, exact, _, contract = build_environments(namespace, 128)
    return molecule, library, exact, fno, contract


def parse_indices(value: str, upper: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(upper))
    indices = [int(item) for item in value.split(",") if item.strip()]
    if not indices or min(indices) < 0 or max(indices) >= upper:
        raise ValueError(f"indices must lie in [0, {upper})")
    return indices


def actor_path(args, seed: int) -> Path:
    return Path(args.source) / "models" / f"confirmation_penalty_020_budget_0_s{seed}.pt"


def load_actor(args, seed: int, max_pulses: int):
    checkpoint = torch.load(actor_path(args, seed), map_location="cpu", weights_only=False)
    cfg = PPOConfig(**checkpoint["config"])
    state = checkpoint["state_dict"]
    return policy_from_state_dict(
        state,
        int(state["trunk.0.weight"].shape[1]),
        int(state["pi.weight"].shape[0]),
        cfg,
        device=args.device,
        greedy=False,
        max_pulses=max_pulses,
    )


def make_hybrid(actor, teacher, library, max_pulses: int, prefix_pulses: int):
    return BatchedFallbackPolicy(
        actor,
        teacher,
        max_pulses=max_pulses,
        switch_remaining=max_pulses - int(prefix_pulses),
        stagnation_steps=0,
        action_encoder=library.encode,
    )


def diagnose(args):
    output = Path(args.output)
    destination = output / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    _, library, exact, _, contract = environments(args)
    write_json(output / "contract.json", contract)
    actor = load_actor(args, 0, exact.cfg.max_pulses)
    teacher = DescendingPopulationPolicy(library, exact.tables)
    for index in parse_indices(args.indices, len(PREFIX_PULSES)):
        prefix = PREFIX_PULSES[index]
        path = destination / f"prefix_{prefix:02d}.json"
        if path.exists():
            continue
        print(f"[diagnose] learned prefix {prefix}", flush=True)
        policy = make_hybrid(actor, teacher, library, exact.cfg.max_pulses, prefix)
        metrics = rollout_metrics(exact, policy, args.diagnosis_eval, DIAGNOSIS_SEED, args.eval_batch)
        write_json(path, {"prefix_pulses": prefix, "evaluation": {"exact": metrics}})


def select(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "diagnosis").glob("prefix_*.json"))
    ]
    if len(records) != len(PREFIX_PULSES):
        raise RuntimeError(f"expected {len(PREFIX_PULSES)} diagnosis records, found {len(records)}")
    winner = max(
        records,
        key=lambda row: (
            row["evaluation"]["exact"]["success_fraction"],
            -row["evaluation"]["exact"]["average_actions"],
        ),
    )
    selection = {
        "prefix_pulses": winner["prefix_pulses"],
        "validation_exact": {
            key: winner["evaluation"]["exact"][key]
            for key in ("success_fraction", "unfinished_fraction", "average_actions")
        },
        "rule": "minimize exact validation failure, then failure-penalized actions",
    }
    write_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def confirm(args):
    output = Path(args.output)
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    _, library, exact, fno, _ = environments(args)
    prefix = int(json.loads((output / "selection.json").read_text())["prefix_pulses"])
    teacher = DescendingPopulationPolicy(library, exact.tables)
    for seed in parse_indices(args.seeds, len(FINAL_SEEDS)):
        path = destination / f"seed_{seed}.json"
        if path.exists():
            continue
        actor = load_actor(args, seed, exact.cfg.max_pulses)
        evaluations = {}
        for dynamics, environment in (("exact", exact), ("fno", fno)):
            print(f"[confirm s{seed}] {dynamics}", flush=True)
            policy = make_hybrid(actor, teacher, library, exact.cfg.max_pulses, prefix)
            evaluations[dynamics] = rollout_metrics(
                environment, policy, args.final_eval, FINAL_EVAL_SEED, args.eval_batch,
            )
        write_json(path, {
            "seed": seed,
            "prefix_pulses": prefix,
            "evaluation": evaluations,
        })
        torch.cuda.empty_cache()


def aggregate(records: list[dict], dynamics: str) -> dict:
    keys = ("unfinished_fraction", "average_actions", "mean_pulses_successful")
    result = {}
    for key in keys:
        values = np.asarray([row["evaluation"][dynamics][key] for row in records], dtype=float)
        result[key] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    failures = np.asarray(
        [row["evaluation"][dynamics]["unfinished_fraction"] for row in records], dtype=float,
    )
    half_width = 2.776 * failures.std(ddof=1) / math.sqrt(len(failures))
    result["unfinished_fraction_seed_ci95"] = [
        float(max(0.0, failures.mean() - half_width)),
        float(min(1.0, failures.mean() + half_width)),
    ]
    return result


def summarize(args):
    output = Path(args.output)
    records = [
        json.loads(path.read_text())
        for path in sorted((output / "confirmation").glob("seed_*.json"))
    ]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    nonml = json.loads((output.parent / "thf_nonml_controls" / "summary.json").read_text())
    teacher = nonml["rows"]["descending_population"]
    source = json.loads((Path(args.source) / "summary.json").read_text())["rows"]["risk_actor"]
    hybrid = {dynamics: aggregate(records, dynamics) for dynamics in ("exact", "fno")}
    hybrid["dominates_descending_population"] = (
        hybrid["exact"]["unfinished_fraction"] < teacher["exact"]["unfinished_fraction"]
        and hybrid["exact"]["average_actions"] < teacher["exact"]["average_actions"]
    )
    summary = {
        "status": "complete",
        "selection": json.loads((output / "selection.json").read_text()),
        "training_seeds": len(records),
        "evaluation_episodes_per_seed": args.final_eval,
        "source_failure_sensitive_ppo": source,
        "descending_population": teacher,
        "learned_prefix_descending_fallback": hybrid,
    }
    write_json(output / "summary.json", summary)

    labels = ("Failure-sensitive PPO", "Descending population", "Learned prefix + fallback")
    exact_rows = (source["exact"], teacher["exact"], hybrid["exact"])
    failures = [100 * row["unfinished_fraction"] for row in exact_rows]
    actions = [row["average_actions"] for row in exact_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].bar(x, failures, color=("#f58518", "#4c78a8", "#54a24b"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, actions, color=("#f58518", "#4c78a8", "#54a24b"))
    axes[1].set_ylabel("Exact failure-penalized actions")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Learned prefix with the strongest adaptive fallback")
    fig.savefig(output / "descending_hybrid_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Learned-prefix and descending-population hybrid",
        "",
        f"Selected learned prefix: {summary['selection']['prefix_pulses']} pulses.",
        "",
        "| Controller | Exact failure | Exact actions | FNO failure | FNO actions |",
        "|---|---:|---:|---:|---:|",
        f"| Failure-sensitive PPO | {100*source['exact']['unfinished_fraction']:.2f}% | "
        f"{source['exact']['average_actions']:.2f} | {100*source['fno']['unfinished_fraction']:.2f}% | "
        f"{source['fno']['average_actions']:.2f} |",
        f"| Descending population | {100*teacher['exact']['unfinished_fraction']:.2f}% | "
        f"{teacher['exact']['average_actions']:.2f} | {100*teacher['fno']['unfinished_fraction']:.2f}% | "
        f"{teacher['fno']['average_actions']:.2f} |",
        f"| Learned prefix + fallback | {100*hybrid['exact']['unfinished_fraction']:.2f}% | "
        f"{hybrid['exact']['average_actions']:.2f} | {100*hybrid['fno']['unfinished_fraction']:.2f}% | "
        f"{hybrid['fno']['average_actions']:.2f} |",
        "",
        f"Dominates descending population on both exact metrics: {hybrid['dominates_descending_population']}.",
        "",
        "![Descending hybrid comparison](descending_hybrid_comparison.png)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "diagnosis": len(list((output / "diagnosis").glob("prefix_*.json"))),
        "selected": (output / "selection.json").exists(),
        "confirmation": len(list((output / "confirmation").glob("seed_*.json"))),
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("command", choices=("diagnose", "select", "confirm", "summarize", "status"))
    main.add_argument("--output", default="results/thf_descending_hybrid")
    main.add_argument("--source", default="results/thf_safe_hybrid")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--indices", default="all")
    main.add_argument("--seeds", default="all")
    main.add_argument("--diagnosis-eval", type=int, default=2_000)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[args.command](args)
