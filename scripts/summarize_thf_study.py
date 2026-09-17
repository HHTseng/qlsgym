#!/usr/bin/env python
"""Validate the locked full grid, plot exact rankings, and refresh README results.

This script refuses incomplete or mixed-contract grids. Bootstrap intervals for
learned controllers describe five-training-seed variation; baseline failure
intervals are Wilson binomial intervals, not fictitious training-seed intervals.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thf_rl_agents import AGENTS, load_records


LEARNED = ("ppo", "sac_discrete", "ddqn")
LABELS = {"sweeping": "Sweeping", "random": "Random", "physics_elimination": "Physics elimination",
          "ppo": "PPO", "sac_discrete": "Discrete SAC", "ddqn": "Double DQN"}


def validate_grid(records):
    expected = {(agent, seed) for agent in AGENTS for seed in (range(5) if agent in LEARNED else [0])}
    present = [(r["job"]["agent"], r["job"]["train_seed"]) for r in records]
    if len(present) != len(set(present)) or set(present) != expected:
        raise ValueError(f"not the complete final grid: missing={expected - set(present)}, extra={set(present) - expected}")
    for record in records:
        job, contract = record["job"], record["contract"]
        if (job["preset"] != "final" or job["eval_episodes"] != 5000
                or job["eval_seed"] != 20001 or job["evaluation_batch"] != 128
                or contract["manifest_pair_coverage"] != 1.0
                or contract["p_target"] != 0.98 or contract["max_pulses"] != 80
                or contract["rho"] != 0 or contract["n_states"] != 192
                or contract["n_actions"] != 312 or contract["n_nu"] != 7
                or len(contract.get("checkpoint_sha256", {})) != 24):
            raise ValueError(f"not the locked final contract: {job}")
        for engine in ("exact", "fno"):
            result = record["evaluation"][engine]
            length, success = np.asarray(result["lengths"]), np.asarray(result["successes"], dtype=bool)
            if len(length) != 5000 or len(success) != 5000:
                raise ValueError("missing individual rollout results")
            if np.any(length[~success] != 80) or np.any((length < 0) | (length > 80)):
                raise ValueError("failure scoring/length contract mismatch")
            if not np.isclose(length.mean(), result["average_actions"]) or not np.isclose(1 - success.mean(), result["unfinished_fraction"]):
                raise ValueError("reported metrics disagree with individual rollouts")
        if job["agent"] in LEARNED:
            budget = record["training"].get("env_steps")
            cfg = record["training"]["config"]
            if cfg["total_steps"] != 1000000 or cfg["n_envs"] != 128:
                raise ValueError("training budget differs from preregistration")
            if budget is None or not 1000000 <= budget <= 1004096:
                raise ValueError("actual transition budget differs from preregistration")
            required = {"lr": 3e-4, "gamma": 0.99 if job["agent"] == "sac_discrete" else 1.0}
            if job["agent"] == "sac_discrete":
                required.update(alpha=0.05/80, target_entropy_ratio=0.5, gradient_steps=4, learning_starts=10240)
            elif job["agent"] == "ddqn":
                required.update(eps_fraction=0.72, gradient_steps=4, learning_starts=10240)
            else:
                required.update(value_target="qmdp_gae", n_steps=32, epochs=4, minibatches=8,
                                ent_coef=0.01, gae_lambda=0.95, eval_greedy=False)
            if any(cfg.get(key) != value for key, value in required.items()):
                raise ValueError("agent hyperparameters differ from preregistration")


def intervals(group, engine, rng):
    metrics = [r["evaluation"][engine] for r in group]
    actions = np.array([r["average_actions"] for r in metrics])
    failure = np.array([r["unfinished_fraction"] for r in metrics])
    if len(group) > 1:
        indices = rng.integers(len(group), size=(10000, len(group)))
        action_ci = np.quantile(actions[indices].mean(1), [0.025, 0.975])
        failure_ci = np.quantile(failure[indices].mean(1), [0.025, 0.975])
        scope = "training-seed percentile bootstrap (5 seeds)"
    else:
        length = np.asarray(metrics[0]["lengths"])
        samples = [length[rng.integers(len(length), size=len(length))].mean() for _ in range(2000)]
        action_ci = np.quantile(samples, [0.025, 0.975])
        n, p, z = len(length), failure[0], 1.95996398454
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        failure_ci = np.array([center - half, center + half])
        scope = "rollout bootstrap for mean; Wilson binomial for failure"
    return {"average_actions": float(actions.mean()), "actions_ci95": action_ci.tolist(),
            "unfinished_fraction": float(failure.mean()), "failure_ci95": failure_ci.tolist(),
            "uncertainty_scope": scope,
            "p85_actions": float(np.mean([r["p85_actions"] for r in metrics])),
            "cvar10_actions": float(np.mean([r["cvar10_actions"] for r in metrics]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/thf_rl_final/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/thf_rl_final"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    records = load_records(args.input)
    validate_grid(records)
    rng = np.random.default_rng(20260921)
    rows, groups = [], {}
    for agent in AGENTS:
        group = sorted([r for r in records if r["job"]["agent"] == agent], key=lambda r: r["job"]["train_seed"])
        groups[agent] = group
        exact, fno = intervals(group, "exact", rng), intervals(group, "fno", rng)
        rows.append({"agent": agent, "training_seeds": len(group) if agent in LEARNED else 0,
                     "exact": exact, "fno": fno,
                     "sim_to_exact_actions_gap": exact["average_actions"] - fno["average_actions"],
                     "sim_to_exact_failure_gap": exact["unfinished_fraction"] - fno["unfinished_fraction"],
                     "gamma": 0.99 if agent == "sac_discrete" else (1.0 if agent in LEARNED else None)})
    ranking = sorted(rows, key=lambda r: (r["exact"]["unfinished_fraction"], r["exact"]["average_actions"]))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "ranking_rule": "exact failure rate first, then penalized average actions",
               "note": "Operational held-out scores; SAC has a discounted entropy objective. Intervals are exploratory, not dominance tests.",
               "contract": records[0]["contract"], "rows": rows, "operational_order": [r["agent"] for r in ranking]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(rows))
    for axis, key, ci_key, scale, label in (
            (axes[0], "average_actions", "actions_ci95", 1, "Average actions (failure = 80)\nLower better"),
            (axes[1], "unfinished_fraction", "failure_ci95", 100, "Unfinished/failure rate (%)\nLower better")):
        value = np.array([r["exact"][key] for r in rows]) * scale
        bounds = np.array([r["exact"][ci_key] for r in rows]) * scale
        axis.bar(x, value, color="#0072B2")
        axis.errorbar(x, value, yerr=np.maximum(np.stack([value - bounds[:, 0], bounds[:, 1] - value]), 0), fmt="none", color="black", capsize=3)
        for j, r in enumerate(rows):
            if r["training_seeds"]:
                points = [g["evaluation"]["exact"][key] * scale for g in groups[r["agent"]]]
                axis.scatter(np.full(len(points), j), points, color="black", s=10, alpha=0.5)
        axis.set_xticks(x, [LABELS[r["agent"]] for r in rows], rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    for row in rows:
        curves = []
        for record in groups[row["agent"]]:
            result = record["evaluation"]["exact"]
            length, success = np.asarray(result["lengths"]), np.asarray(result["successes"], dtype=bool)
            curves.append([100 * ((length <= h) & success).mean() for h in range(81)])
        axes[2].plot(range(81), np.mean(curves, axis=0), label=LABELS[row["agent"]])
    axes[2].set_xlabel("Pulses applied")
    axes[2].set_ylabel("Finished episodes (%); higher better")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.2)
    figure.suptitle("ThF+ final exact ranking: 5 seeds x ~1M training transitions; 5000 rollouts/model")
    figure.tight_layout()
    figure.savefig(args.output / "thf_final_ranking.png", dpi=180)
    plt.close(figure)

    lines = ["## Final exact-simulator ranking", "", "Complete locked grid: 15 learned policies and three baselines. Each policy has 5000 exact and 5000 surrogate rollouts.", "",
             "| Controller | Training seeds | Exact average actions ↓ (95% CI) | Exact failure ↓ (95% CI) | FNO average actions | FNO failure |", "|---|---:|---:|---:|---:|---:|"]
    for row in ranking:
        e, f = row["exact"], row["fno"]
        lines.append(f"| {LABELS[row['agent']]} | {row['training_seeds']} | {e['average_actions']:.2f} [{e['actions_ci95'][0]:.2f}, {e['actions_ci95'][1]:.2f}] | {100*e['unfinished_fraction']:.2f}% [{100*e['failure_ci95'][0]:.2f}, {100*e['failure_ci95'][1]:.2f}] | {f['average_actions']:.2f} | {100*f['unfinished_fraction']:.2f}% |")
    lines += ["", "![Final ThF+ RL ranking](results/thf_rl_final/thf_final_ranking.png)", "",
              "Order is descriptive: exact failure rate, then average actions. PPO/DDQN use γ=1; SAC uses γ=0.99 with an entropy bonus, so these are operational performance scores, not equal training objectives.", "",
              "Learned-policy intervals bootstrap five training-seed means; baseline mean intervals bootstrap rollouts and failure intervals use Wilson bounds. Five seeds do not establish statistical dominance. Inspect individual JSONs and the FNO-to-exact gap before interpreting a learned advantage."]
    section = "\n".join(lines)
    (args.output / "summary.md").write_text(section + "\n")
    text = args.readme.read_text()
    start, stop = "<!-- FINAL_RL_RESULTS_START -->", "<!-- FINAL_RL_RESULTS_END -->"
    if text.count(start) != 1 or text.count(stop) != 1:
        raise ValueError("README must have exactly one final-results marker pair")
    args.readme.write_text(text.split(start)[0] + start + "\n" + section + "\n" + stop + text.split(stop)[1])
    print(section)


if __name__ == "__main__":
    main()
