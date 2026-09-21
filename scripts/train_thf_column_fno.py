#!/usr/bin/env python
"""Train and assemble physics-constrained ThF+ transfer-column FNOs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qlsgym import load_molecule
from qlsgym.surrogate.column_train import ColumnTrainConfig, train_column_fno
from qlsgym.surrogate.fno import ColumnFNOConfig
from qlsgym.surrogate.manifest import build_manifest


def run_dir(work: Path, tag: str, block: int, sigma: str) -> Path:
    return work / "runs" / f"{tag}_{'sp' if sigma == '+' else 'sm'}_block{block}"


def train_pair(args):
    molecule = load_molecule("thf")
    destination = run_dir(Path(args.work), args.tag, args.block, args.sigma)
    if (destination / "best.pt").exists() and (destination / "summary.json").exists():
        print(f"reuse completed {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"incomplete output exists: {destination}")
    fno = ColumnFNOConfig(
        n_modes=args.modes,
        hidden_channels=args.hidden,
        n_layers=args.layers,
        lifting_channel_ratio=3,
        projection_channel_ratio=3,
        factorization="tucker",
        rank=0.6,
        domain_padding=0.1,
        positional_embedding="grid",
        off_resonance_linewidths=args.off_resonance_linewidths,
    )
    config = ColumnTrainConfig(
        n_train_freq=args.train_freq,
        n_val_freq=args.val_freq,
        states_per_frequency=args.states,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        val_seed=args.val_seed,
        n_linewidths=args.off_resonance_linewidths,
        fno=fno,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_config.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "block": args.block,
                "sigma": args.sigma,
                "device": args.device,
                "config": config.__dict__,
            },
            indent=2,
            default=lambda value: value.__dict__,
        )
        + "\n"
    )
    summary = train_column_fno(
        molecule,
        args.block,
        args.sigma,
        config,
        device=args.device,
        out_dir=str(destination),
    )
    print(json.dumps(summary, indent=2))


def manifest(args):
    molecule = load_molecule("thf")
    work = Path(args.work).resolve()
    pairs = []
    for token in args.pairs.split(","):
        block, sigma = token.split(":")
        pairs.append((int(block), sigma))
    runs = {pair: str(run_dir(work, args.tag, *pair)) for pair in pairs}
    missing = [path for path in runs.values() if not (Path(path) / "best.pt").exists()]
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    value = build_manifest(
        molecule,
        args.tag,
        runs,
        source="qlsgym",
        checkpoint_name="best.pt",
        note=(
            "physics-constrained transfer-column FNO; exact input linearity and tau=0 "
            f"identity; static beyond {args.off_resonance_linewidths} linewidths"
        ),
        require_summary=True,
        check_load=True,
        device=args.device,
    )
    path = value.save(work=str(work))
    print(value.table())
    print(path)


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-pair")
    train.add_argument("--work", required=True)
    train.add_argument("--tag", default="column_v1")
    train.add_argument("--block", type=int, required=True, choices=range(12))
    train.add_argument("--sigma", required=True, choices=("+", "-"))
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--train-freq", type=int, default=256)
    train.add_argument("--val-freq", type=int, default=48)
    train.add_argument("--states", type=int, default=16)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--val-seed", type=int, default=20260922)
    train.add_argument("--modes", type=int, default=40)
    train.add_argument("--hidden", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--off-resonance-linewidths", type=float, default=3.0)
    train.set_defaults(func=train_pair)
    build = sub.add_parser("manifest")
    build.add_argument("--work", required=True)
    build.add_argument("--tag", default="column_v1")
    build.add_argument("--pairs", default="0:+,0:-,1:+,1:-")
    build.add_argument("--device", default="cpu")
    build.add_argument("--off-resonance-linewidths", type=float, default=3.0)
    build.set_defaults(func=manifest)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
