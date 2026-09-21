#!/usr/bin/env python
"""Fine-tune downloaded-FNO SAC checkpoints on cached exact ThF+ dynamics."""

from __future__ import annotations

import argparse
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

from qlsgym.rl.off_policy import DiscreteSACAgent, SACConfig, sac_policy
from thf_rl_agents import build_environments, rollout_metrics, write_json


LRS = (3e-5, 1e-4, 3e-4)
FINAL_SEEDS = tuple(range(5))


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
    _, _, fno, exact, _, contract = build_environments(env_namespace(args), 16)
    return exact, fno, contract


def source_model(args, seed):
    return Path(args.source_models) / f"sac_t24_s{seed}.pt"


def make_agent(exact, checkpoint, steps, train_seed, lr):
    raw = dict(checkpoint["config"])
    raw.update(total_steps=steps, seed=train_seed, lr=lr)
    cfg = SACConfig(**raw)
    agent = DiscreteSACAgent(exact, cfg)
    agent.network.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.network.q1.load_state_dict(checkpoint["critic1_state_dict"])
    agent.network.q2.load_state_dict(checkpoint["critic2_state_dict"])
    agent.target_q1.load_state_dict(checkpoint["critic1_state_dict"])
    agent.target_q2.load_state_dict(checkpoint["critic2_state_dict"])
    with torch.no_grad():
        agent.network.log_alpha.fill_(math.log(float(checkpoint["alpha"])))
    return cfg, agent


def score(metrics):
    return float(metrics["success_fraction"] + 0.02 * (1.0 - metrics["average_actions"] / 80.0))


def run_one(args, exact, fno, source_seed, train_seed, lr, steps, episodes, eval_seed, stage):
    source = source_model(args, source_seed)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    cfg, agent = make_agent(exact, checkpoint, steps, train_seed, lr)
    started = time.time()
    stats = agent.train()
    policy = sac_policy(agent, stochastic=True)
    exact_metrics = rollout_metrics(exact, policy, episodes, eval_seed, args.eval_batch)
    fno_metrics = rollout_metrics(fno, policy, episodes, eval_seed, args.eval_batch)
    model_path = Path(args.output) / "models" / f"{stage}_source_s{source_seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": "sac_discrete",
            "config": asdict(cfg),
            "actor_state_dict": agent.network.actor.state_dict(),
            "critic1_state_dict": agent.network.q1.state_dict(),
            "critic2_state_dict": agent.network.q2.state_dict(),
            "alpha": float(agent.network.log_alpha.detach().exp()),
            "source_model": str(source),
            "source_seed": source_seed,
        },
        model_path,
    )
    return {
        "stage": stage,
        "source_model": str(source),
        "source_seed": source_seed,
        "train_seed": train_seed,
        "fine_tune_steps": steps,
        "learning_rate": lr,
        "config": asdict(cfg),
        "training": stats.as_dict(),
        "evaluation": {"exact": exact_metrics, "fno": fno_metrics},
        "selection_score_exact": score(exact_metrics),
        "model_path": str(model_path),
        "seconds": time.time() - started,
    }


def diagnose_lr(args, index):
    output = Path(args.output)
    destination = output / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    lr = LRS[index]
    path = destination / f"lr_{lr:.0e}.json"
    if path.exists():
        return
    exact, fno, contract = environments(args)
    write_json(output / "contract.json", contract)
    write_json(path, run_one(
        args, exact, fno, 0, 73_001, lr, args.diagnose_steps,
        args.diagnose_eval, 73_101, f"diagnosis_lr{index}",
    ))
    torch.cuda.empty_cache()


def diagnose(args):
    for index, lr in enumerate(LRS):
        diagnose_lr(args, index)


def select(args):
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "diagnosis").glob("*.json"))]
    if len(records) != len(LRS):
        raise RuntimeError(f"expected {len(LRS)} diagnosis records, found {len(records)}")
    winner = max(records, key=lambda row: row["selection_score_exact"])
    result = {
        "learning_rate": winner["learning_rate"],
        "diagnosis_exact": {
            key: winner["evaluation"]["exact"][key]
            for key in ("success_fraction", "average_actions", "mean_pulses_successful")
        },
        "selection_score_exact": winner["selection_score_exact"],
        "rule": "exact success plus 0.02 normalized action efficiency",
    }
    write_json(output / "selection.json", result)
    print(json.dumps(result, indent=2))


def confirm_seed(args, source_seed):
    output = Path(args.output)
    selection = json.loads((output / "selection.json").read_text())
    destination = output / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"source_s{source_seed}.json"
    if path.exists():
        return
    exact, fno, _ = environments(args)
    write_json(path, run_one(
        args, exact, fno, source_seed, 74_000 + source_seed,
        float(selection["learning_rate"]), args.confirm_steps,
        args.final_eval, 74_101, "confirmation",
    ))
    torch.cuda.empty_cache()


def confirm(args):
    for source_seed in FINAL_SEEDS:
        confirm_seed(args, source_seed)


def aggregate(records, engine):
    result = {}
    for key in ("success_fraction", "unfinished_fraction", "average_actions", "mean_pulses_successful"):
        values = [float(row["evaluation"][engine][key]) for row in records]
        result[key] = float(np.mean(values))
        result[key + "_sd"] = float(np.std(values, ddof=1))
    return result


def summarize(args):
    output = Path(args.output)
    records = [json.loads(path.read_text()) for path in sorted((output / "confirmation").glob("*.json"))]
    if len(records) != len(FINAL_SEEDS):
        raise RuntimeError(f"expected five confirmation records, found {len(records)}")
    reference = json.loads((output.parent / "thf_rl_optuna_mix_sac_refine" / "summary.json").read_text())["row"]
    exact, fno = aggregate(records, "exact"), aggregate(records, "fno")
    summary = {
        "status": "complete",
        "method": "downloaded-mix FNO pretraining followed by cached-exact SAC fine-tuning",
        "selection": json.loads((output / "selection.json").read_text()),
        "fine_tune_steps": args.confirm_steps,
        "training_seeds": len(records),
        "hybrid": {"exact": exact, "fno": fno},
        "fno_trained_reference": reference,
    }
    write_json(output / "summary.json", summary)
    names = ["FNO trained", "FNO→exact fine-tuned"]
    rows = [reference["exact"], exact]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    axes[0].bar(x, [100*r["unfinished_fraction"] for r in rows], color=("#f58518", "#54a24b"))
    axes[0].set_ylabel("Exact unfinished episodes (%)")
    axes[1].bar(x, [r["average_actions"] for r in rows], color=("#f58518", "#54a24b"))
    axes[1].set_ylabel("Exact penalized average actions")
    for axis in axes:
        axis.set_xticks(x, names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ThF+ SAC: surrogate pretraining and exact fine-tuning")
    fig.savefig(output / "hybrid_sac_comparison.png", dpi=180)
    plt.close(fig)


def status(args):
    output = Path(args.output)
    print(json.dumps({
        "diagnosis": len(list((output / "diagnosis").glob("*.json"))) if (output / "diagnosis").exists() else 0,
        "selected": (output / "selection.json").exists(),
        "confirmation": len(list((output / "confirmation").glob("*.json"))) if (output / "confirmation").exists() else 0,
        "summary": (output / "summary.json").exists(),
    }, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument(
        "command",
        choices=("diagnose", "diagnose-lr", "select", "confirm", "confirm-seed", "summarize", "run", "status"),
    )
    main.add_argument("--output", default="results/thf_mix_sac_exact_finetune")
    main.add_argument("--source-models", default="results/thf_rl_optuna_mix_sac_refine/models")
    main.add_argument("--manifest", default=os.path.expanduser("~/qlsgym_work/checkpoints/thf/mix.json"))
    main.add_argument("--device", default="cuda:0")
    main.add_argument("--diagnose-steps", type=int, default=100_000)
    main.add_argument("--diagnose-eval", type=int, default=1_000)
    main.add_argument("--confirm-steps", type=int, default=250_000)
    main.add_argument("--final-eval", type=int, default=5_000)
    main.add_argument("--eval-batch", type=int, default=256)
    main.add_argument("--source-seed", type=int, choices=FINAL_SEEDS)
    main.add_argument("--lr-index", type=int, choices=range(len(LRS)))
    return main


def main():
    args = parser().parse_args()
    if args.command == "diagnose-lr":
        if args.lr_index is None:
            raise ValueError("diagnose-lr requires --lr-index")
        diagnose_lr(args, args.lr_index)
        return
    if args.command == "confirm-seed":
        if args.source_seed is None:
            raise ValueError("confirm-seed requires --source-seed")
        confirm_seed(args, args.source_seed)
        return
    commands = ("diagnose", "select", "confirm", "summarize") if args.command == "run" else (args.command,)
    for command in commands:
        globals()[command](args)


if __name__ == "__main__":
    main()
