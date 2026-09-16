#!/usr/bin/env python
"""The only entry point of the replication."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replication import config as cfgmod            # noqa: E402
from replication.config import Config               # noqa: E402
from replication.stages import base                 # noqa: E402


def _coerce(name: str, value: str):
    current = getattr(Config(), name, None)
    if isinstance(current, dict):
        raise SystemExit(f"--set cannot express {name}={value!r}: {name} is a dict, so it is "
                         f"reachable only from Python, as Config({name}={{...}})")
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    try:
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float):
            return float(value)
    except ValueError:
        raise SystemExit(f"--set {name}={value!r}: {name} expects "
                         f"{type(current).__name__}") from None
    if isinstance(current, tuple):
        return tuple(int(v) if v.strip().lstrip("-").isdigit() else v.strip()
                     for v in value.split(","))
    return value


def build_config(args) -> Config:
    kw = {}
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not hasattr(Config(), key):
            raise SystemExit(f"--set: Config has no field {key!r}")
        kw[key] = _coerce(key, value)
    kw["dry_run"] = bool(args.dry_run)
    if args.device:
        kw["device"] = args.device
    cfg = Config(**kw)
    if kw.get("blocks"):
        try:
            cfgmod.resolve_blocks(cfg)
        except ValueError as exc:
            raise SystemExit(f"--set: {exc}") from None
    return cfg


def cmd_list(cfg: Config) -> int:
    print(f"{'stage':<14} {'state':<9} {'paper':<22} cost")
    print("-" * 78)
    for name in base.ORDER:
        stage = base.load(name)
        state = "done" if cfgmod.stage_is_done(name) else "pending"
        missing = base.check_requirements(stage, cfg)
        if missing and state == "pending":
            state = "blocked"
        print(f"{name:<14} {state:<9} {stage.PAPER:<22} {stage.COST}")
        if missing:
            print(f"{'':<14} needs: {', '.join(missing)}")
    print(f"\noutputs: {cfgmod.outputs_dir()}")
    return 0


def run_stage(name: str, cfg: Config) -> int:
    stage = base.load(name)
    missing = base.check_requirements(stage, cfg)
    print(f"\n=== {name}: {stage.TITLE}  [{stage.PAPER}]  ({stage.COST})")
    if missing:
        print(f"  BLOCKED, missing: {', '.join(missing)}")
        if not cfg.dry_run:
            return 1
    for line in stage.plan(cfg):
        print(f"  - {line}")
    if cfg.dry_run:
        print("  (dry run: nothing executed)")
        return 0
    t0 = time.time()
    payload = stage.run(cfg)
    path = cfgmod.write_result(name, payload, cfg)
    print(f"  wrote {path}  ({time.time() - t0:.1f} s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=base.ORDER, metavar="STAGE",
                    help=f"one of: {', '.join(base.ORDER)}")
    ap.add_argument("--all", action="store_true", help="every stage, in order")
    ap.add_argument("--list", action="store_true", help="the pipeline and its state")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a Config field (repeatable)")
    args = ap.parse_args(argv)
    cfg = build_config(args)

    if args.list or (not args.stage and not args.all):
        return cmd_list(cfg)
    names = list(base.ORDER) if args.all else [args.stage]
    for name in names:
        rc = run_stage(name, cfg)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
