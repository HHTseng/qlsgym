#!/usr/bin/env python
"""Run, merge and plot the policy benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import fields, replace

from qlsgym import load_molecule
from qlsgym.benchmark import ARMS, BenchmarkConfig, merge_results, read_results, run_arm, write_result
from qlsgym.benchmark.figure import render, table
from qlsgym.benchmark.protocol import outputs_dir, result_path, unmerged_parts
from qlsgym.policies.score import ScoreConfig
from qlsgym.rl.ppo import PPOConfig


def _typed(dc_type, kv: list):
    """["a=1", "b=x"] -> a dc_type with those fields set, typed by the defaults."""
    obj = dc_type()
    types = {f.name: type(getattr(obj, f.name)) for f in fields(dc_type)}
    for item in kv or []:
        k, _, v = item.partition("=")
        if k not in types:
            raise SystemExit(f"{dc_type.__name__} has no field {k!r}; fields: {sorted(types)}")
        t = types[k]
        if v.lower() == "none":
            val = None
        elif t is bool:
            val = v.lower() in ("1", "true", "yes")
        elif t is type(None):
            val = float(v) if "." in v or "e" in v.lower() else int(v)
        else:
            val = t(v)
        obj = replace(obj, **{k: val})
    return obj


def cmd_run(a) -> None:
    cfg = BenchmarkConfig(
        molecule=a.molecule, library=a.library, n_episodes=a.n_episodes, seed=a.seed,
        max_pulses=a.max_pulses, p_target=a.p_target, fno_tag=a.fno_tag, device=a.device,
        n_pool=(None if a.n_pool in ("none", "None") else int(a.n_pool)), delta_s=a.delta_s,
        score=_typed(ScoreConfig, a.score), rho=a.rho, penalty_mode=a.penalty_mode,
        ppo=_typed(PPOConfig, a.ppo), retrain=a.retrain, n_freq=a.n_freq, n_tau_slots=a.n_tau_slots,
        grid_seed=a.grid_seed)
    log = lambda s: print(s, flush=True)
    res = run_arm(a.arm, cfg, log=log, progress=a.progress)
    path = a.out or result_path(a.molecule, a.arm, a.part)
    write_result(res, path)
    print(f"wrote {path}")


def cmd_merge(a) -> None:
    parts = sorted(glob.glob(os.path.join(outputs_dir(a.molecule), f"{a.arm}.part*.json")))
    if not parts:
        raise SystemExit(f"no {a.arm}.part*.json under {outputs_dir(a.molecule)}")
    recs = []
    for p in parts:
        with open(p) as fh:
            recs.append(json.load(fh))
    merged = merge_results(recs)
    path = a.out or result_path(a.molecule, a.arm)
    write_result(merged, path)
    print(f"merged {len(parts)} parts, {merged['n_episodes']} episodes -> {path}")
    print(table([merged]))


def _results(a) -> list:
    mol = load_molecule(a.molecule)
    d = outputs_dir(a.molecule)
    orphans = unmerged_parts(d)
    if orphans:
        bar = "!" * 72
        print(bar, file=sys.stderr)
        for arm, parts in orphans.items():
            print(f"!! {arm}: {len(parts)} part file(s) but no {arm}.json -- this arm will be "
                  f"MISSING from the table and the figure.", file=sys.stderr)
            print(f"!!   fix: python scripts/benchmark.py merge {a.molecule} {arm}", file=sys.stderr)
        print(bar, file=sys.stderr)
    res = read_results(mol, warn=lambda s: print("warning:", s, file=sys.stderr))
    if a.arms:
        res = [r for r in res if r["arm"] in a.arms]
    if not res:
        raise SystemExit(f"no current results under {d}")
    return res


def cmd_plot(a) -> None:
    res = _results(a)
    stem = a.out or os.path.join(outputs_dir(a.molecule), "finished")
    budget = load_molecule(a.molecule).task.max_pulses
    for p in render(res, stem, budget=budget):
        print(f"wrote {p}")
    print(table(res))


def cmd_table(a) -> None:
    print(table(_results(a)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate one arm")
    r.add_argument("molecule")
    r.add_argument("arm", choices=ARMS)
    r.add_argument("--library", default="physics_subset", choices=["physics_subset", "rl_discrete"])
    r.add_argument("--n-episodes", type=int, default=1000)
    r.add_argument("--seed", type=int, default=777)
    r.add_argument("--max-pulses", type=int, default=None)
    r.add_argument("--p-target", type=float, default=None)
    r.add_argument("--fno-tag", default="prod")
    r.add_argument("--device", default="cpu")
    r.add_argument("--n-pool", default="16", help="planner pool size, or 'none' for greedy")
    r.add_argument("--delta-s", type=float, default=0.003)
    r.add_argument("--score", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--rho", type=float, default=2.0)
    r.add_argument("--penalty-mode", default="indicator", choices=["indicator", "proportional"])
    r.add_argument("--ppo", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--retrain", action="store_true")
    r.add_argument("--n-freq", type=int, default=1764)
    r.add_argument("--n-tau-slots", type=int, default=10)
    r.add_argument("--grid-seed", type=int, default=0)
    r.add_argument("--part", type=int, default=None, help="chunk index; pair with a distinct --seed")
    r.add_argument("--out", default=None)
    r.add_argument("--progress", action="store_true")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("merge", help="concatenate <arm>.part*.json into <arm>.json")
    m.add_argument("molecule"); m.add_argument("arm", choices=ARMS); m.add_argument("--out", default=None)
    m.set_defaults(fn=cmd_merge)

    for name, fn in (("plot", cmd_plot), ("table", cmd_table)):
        p = sub.add_parser(name)
        p.add_argument("molecule")
        p.add_argument("--arms", nargs="*", default=None)
        p.add_argument("--out", default=None)
        p.set_defaults(fn=fn)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
