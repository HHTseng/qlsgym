#!/usr/bin/env python
"""Create slide-11-style trajectory and slide-12 benchmark figures."""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym import load_molecule
from qlsgym.benchmark.figure import render, table
from qlsgym.benchmark.protocol import BenchmarkConfig, build_library, exact_env
from qlsgym.policies.rollout import rollout
from qlsgym.rl.ppo import load_policy


def main():
    root = os.path.join(os.environ["QLSGYM_OUTPUTS"], "thf")
    with open(os.path.join(root, "rl-fno.json")) as stream:
        record = json.load(stream)
    actor_path = os.path.join(record["rl"]["run_dir"], "actor_best.pt")

    cfg = BenchmarkConfig(
        molecule="thf",
        n_episodes=200,
        seed=777,
        max_pulses=800,
        p_target=0.98,
        fno_tag="mix",
        device="cuda:0",
    )
    molecule = load_molecule("thf")
    library = build_library(cfg, molecule)
    env = exact_env(cfg, molecule, library, batch=200)
    policy = load_policy(actor_path, device="cuda:0", greedy=False)
    result = rollout(env, policy, n_rollouts=200, seed=777, record=True)

    # Replay recorded branches to retain max-belief histories. The initial
    # belief and exact branch tables are the same ones used by the evaluation.
    horizon = int(record["max_pulses"])
    batch = len(result.trajectories)
    beliefs = np.repeat(env.p_init[None, :], batch, axis=0)
    padded = np.empty((batch, horizon + 1))
    padded[:, 0] = beliefs.max(axis=1)
    max_length = max(map(len, result.trajectories))
    for t in range(max_length):
        active = np.asarray([t < len(tr) for tr in result.trajectories])
        actions = np.asarray([tr[t][0] if t < len(tr) else 0 for tr in result.trajectories])
        outcomes = np.asarray([tr[t][1] if t < len(tr) else 0 for tr in result.trajectories])
        u0, u1 = env.apply(beliefs, actions)
        u0, u1 = u0.cpu().numpy(), u1.cpu().numpy()
        selected = np.where(outcomes[:, None] == 1, u1, u0)
        updated = selected / selected.sum(axis=1, keepdims=True)
        beliefs[active] = updated[active]
        padded[:, t + 1] = beliefs.max(axis=1)
    padded[:, max_length + 1:] = padded[:, max_length, None]
    q25, median, q75 = np.percentile(padded, [25, 50, 75], axis=0)
    x = np.arange(horizon + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for values in padded[:30]:
        axes[0].plot(x, values, color="0.75", alpha=0.35, linewidth=0.7)
    axes[0].fill_between(x, q25, q75, color="C0", alpha=0.16)
    axes[0].plot(x, median, color="C0", linewidth=2.2, label="median")
    axes[0].axhline(cfg.p_target, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].text(4, cfg.p_target + 0.015, "target", fontsize=9)
    axes[0].set(xlabel="number of pulses", ylabel="probability of the\nmost likely state",
                xlim=(0, horizon), ylim=(0, 1.02))
    axes[0].legend(frameon=False, loc="center right")

    curve = np.asarray(record["curve"])
    success = float(record["success"])
    successful_lengths = np.asarray(record["lengths"])[np.asarray(record["successes"], dtype=bool)]
    med_success = int(np.median(successful_lengths)) if successful_lengths.size else horizon
    axes[1].step(np.arange(curve.size), curve, where="post", color="C0", linewidth=2.0)
    axes[1].axhline(success, color="0.6", linestyle=":", linewidth=1.0)
    axes[1].axvline(med_success, ymax=(0.5 * success), color="0.6", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.5 * success, xmax=med_success / horizon, color="0.6", linestyle=":", linewidth=1.0)
    axes[1].text(0.66 * horizon, min(success + 0.025, 0.97),
                 f"{success:.3f} of 200 runs", fontsize=9)
    axes[1].text(min(med_success + 12, 0.76 * horizon), 0.5 * success - 0.08,
                 f"median {med_success} pulses", fontsize=9)
    axes[1].set(xlabel="number of pulses", ylabel="fraction of runs prepared",
                xlim=(0, horizon), ylim=(0, 1.02))
    trajectory_path = os.path.join(root, "rl_fno_trajectory_and_completion.png")
    fig.savefig(trajectory_path, dpi=220)
    fig.savefig(trajectory_path.replace(".png", ".pdf"))
    plt.close(fig)

    order = ["sweeping", "random", "planner-exact", "planner-fno", "rl-exact", "rl-fno"]
    results = []
    for arm in order:
        with open(os.path.join(root, arm + ".json")) as stream:
            results.append(json.load(stream))
    render(results, os.path.join(root, "finished"), budget=800)
    with open(os.path.join(root, "summary.txt"), "w") as stream:
        stream.write(table(results) + "\n")
    print(table(results))
    print("wrote", trajectory_path)


if __name__ == "__main__":
    main()
