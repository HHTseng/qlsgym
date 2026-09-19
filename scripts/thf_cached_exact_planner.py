#!/usr/bin/env python
"""Evaluate the fixed-grid exact planner using cached action transfer tables.

For the ThF physics_subset library there is exactly one duration per drive.
Therefore scoring all cached actions is algebraically identical to the
ScorePlannerPolicy(tau_mode="library") path, without rebuilding eigensystems.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np

from qlsgym.benchmark.curve import from_rollout
from qlsgym.benchmark.protocol import (
    BenchmarkConfig,
    build_library,
    exact_engine,
    exact_env,
    library_taus,
    load,
    score_config,
    write_result,
)
from qlsgym.policies.rollout import rollout
from qlsgym.policies.baselines import ScorePlannerPolicy


class CachedExactPlanner:
    stateful = False

    def __init__(self, env, candidate_actions, n_pool=16, delta_s=0.003):
        self.env = env
        self.actions = np.asarray(candidate_actions, dtype=np.int64)
        self.n_pool = int(n_pool)
        self.delta_s = float(delta_s)
        self.calls = 0

    def act_batch(self, beliefs, t, rng):
        torch = self.env.torch
        p = torch.as_tensor(beliefs, dtype=torch.float64, device=self.env.device)
        batch, n_states = p.shape
        n_actions = self.actions.size
        repeated = p[:, None, :].expand(batch, n_actions, n_states).reshape(-1, n_states)
        actions = torch.as_tensor(
            np.tile(self.actions, batch), dtype=torch.long, device=self.env.device
        )
        p0, p1 = self.env.apply(repeated, actions)
        p0 = p0.reshape(batch, n_actions, n_states)
        p1 = p1.reshape(batch, n_actions, n_states)

        # ScoreConfig.for_molecule(thf): w_tr=.5, w_br=1.5,
        # w1_scale=1, br_mode="expected". In expected mode,
        # pi_k * max(p_k/pi_k) = max(p_k).
        pi1 = p1.sum(-1)
        j_star = p.argmax(-1)
        moved = p[:, None, :] - p0
        moved_star = moved.gather(
            2, j_star[:, None, None].expand(batch, n_actions, 1)
        ).squeeze(-1)
        s_tr = 2.0 * moved_star - moved.sum(-1)
        s_br = p0.amax(-1) + p1.amax(-1)
        w1 = 1.0 - p.amax(-1)
        scores = (0.5 * s_tr + 1.5 * s_br + w1[:, None] * pi1).cpu().numpy()

        picks = np.empty(batch, dtype=np.int64)
        for b, vals in enumerate(scores):
            order = np.argsort(-vals, kind="stable")
            pool = set(order[: self.n_pool].tolist())
            pool.update(np.where(vals >= vals[order[0]] - self.delta_s)[0].tolist())
            members = sorted(pool)
            picks[b] = self.actions[members[int(rng.integers(len(members)))]]
        self.calls += batch * n_actions
        return picks


def main():
    cfg = BenchmarkConfig(
        molecule="thf",
        n_episodes=200,
        seed=777,
        max_pulses=800,
        p_target=0.98,
        fno_tag="mix",
        device="cuda:0",
        n_pool=16,
        delta_s=0.003,
    )
    started = time.time()
    molecule = load(cfg)
    library = build_library(cfg, molecule)
    env = exact_env(cfg, molecule, library, batch=cfg.n_episodes, progress=True)

    # Reuse ScorePlannerPolicy's candidate ordering. This preserves its seeded
    # random choice within the active score pool.
    template = ScorePlannerPolicy(
        exact_engine(molecule, library_taus(library)),
        library,
        score_config(cfg, molecule),
        tau_mode="library",
        n_pool=cfg.n_pool,
        delta_s=cfg.delta_s,
    )
    ordered_actions = []
    for _sigma, _omegas, rows_lists in template._groups:
        for rows in rows_lists:
            if len(rows) != 1:
                raise RuntimeError("cached equivalence requires one duration per drive")
            ordered_actions.append(rows[0][1])
    if sorted(ordered_actions) != list(range(library.n_actions)):
        raise RuntimeError("candidate ordering does not cover every action exactly once")

    policy = CachedExactPlanner(env, ordered_actions, cfg.n_pool, cfg.delta_s)
    result = rollout(env, policy, n_rollouts=cfg.n_episodes, seed=cfg.seed, progress=True)
    record = {
        "arm": "planner-exact",
        "policy": "planner-exact (cached fixed-library equivalent)",
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "library": library.describe(),
        "config": cfg.as_dict(),
        "engine": {
            "kind": "CachedActionTables",
            "builder": getattr(env.tables, "metadata", {}).get("builder", "physics transfer columns"),
            "equivalence": "one cached exact transfer table per physics_subset action",
        },
        "rl": None,
        "seconds": time.time() - started,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    record.update(from_rollout(result))
    output = os.path.join(
        os.environ["QLSGYM_OUTPUTS"], "thf", "planner-exact.json"
    )
    write_result(record, output)
    print(
        f"planner-exact: success {record['success']:.3f}, "
        f"mean pulses {record['mean_pulses']:.1f}, P85 {record['p85']}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
