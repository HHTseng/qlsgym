#!/usr/bin/env python
"""Prepare provenance-stamped ThF+ FNO checkpoints for RL experiments.

The qlsgym repository intentionally does not ship trained weights.  This
script makes that prerequisite explicit and reproducible.  Train independent
block/polarization jobs on available GPUs, then assemble a manifest only after
qlsgym's physics-fingerprint checks pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import qlsgym
from qlsgym.surrogate.dataset import DataConfig
from qlsgym.surrogate.fno import FNOConfig
from qlsgym.surrogate.manifest import build_manifest
from qlsgym.surrogate.train import TrainConfig, evaluate_stratified, load_model, train


PRESETS = {
    # Pipeline check only; not accurate enough for a scientific RL comparison.
    "smoke": {
        "n_freq": 64,
        "n_pairs": 256,
        "val_pairs": 64,
        "epochs": 3,
        "hidden": 32,
        "layers": 2,
        "modes": 20,
        "sel_freq": 8,
        "sel_init": 8,
    },
    # Feasibility pilot.  Treat model error as a reported experimental factor.
    "pilot": {
        "n_freq": 512,
        "n_pairs": 4_096,
        "val_pairs": 512,
        "epochs": 30,
        "hidden": 64,
        "layers": 3,
        "modes": 40,
        "sel_freq": 32,
        "sel_init": 32,
    },
    # Matches qlsgym's production SLURM data scale and default architecture.
    "primary": {
        "n_freq": 24_000,
        "n_pairs": 32_000,
        "val_pairs": 400,
        "epochs": 80,
        "hidden": 256,
        "layers": 4,
        "modes": 60,
        "sel_freq": 64,
        "sel_init": 64,
    },
}
# Longer production run; preserve approximately the primary decay span rather
# than letting an 80-epoch schedule extinguish the learning rate by epoch 120.
PRESETS["long"] = {**PRESETS["primary"], "epochs": 120, "sel_init": 16}
# qlsgym materialises the selection Cartesian product in a single frequency
# chunk. 64 x 64 on a 26-state block needs an 8.25 GiB gather AND an equally
# large multiply temporary. Keep 64 frequencies but 16 initial states for the
# server's 11 GiB cards; independent testing still uses 128 initial states.


def qlsgym_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(qlsgym.__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_directory(work: Path, tag: str, sigma: str, block: int) -> Path:
    polarization = "sp" if sigma == "+" else "sm"
    return work / "runs" / f"{tag}_{polarization}_block{block}"


def train_block(args) -> None:
    values = PRESETS[args.preset]
    molecule = qlsgym.load_molecule("thf")
    output = run_directory(Path(args.work), args.tag, args.sigma, args.block)
    fno = FNOConfig(
        n_modes=values["modes"],
        hidden_channels=values["hidden"],
        n_layers=values["layers"],
        lifting_channel_ratio=5,
        projection_channel_ratio=5,
        factorization="tucker",
        rank=0.8,
        domain_padding=0.1,
    )
    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=values["epochs"],
        lr=5e-4,
        weight_decay=3e-4,
        scheduler_step=12 if args.preset == "long" else (8 if args.preset == "primary" else 5),
        scheduler_gamma=0.75 if args.preset in ("primary", "long") else 0.7,
        train_seed=args.seed,
        val_seed=1,
        sel_n_freq=values["sel_freq"],
        sel_n_init=values["sel_init"],
        fno=fno,
    )
    train_data = DataConfig(
        n_freq=values["n_freq"],
        n_init=1_000,
        n_pairs=values["n_pairs"],
        pairing="random",
        seed=args.seed,
        alpha=-2.0,
        freq_sampling="mixture",
        resonant_frac=0.5,
        resonant_spread=3.0,
    )
    val_data = DataConfig(
        n_freq=min(100, values["n_freq"]),
        n_init=100,
        n_pairs=values["val_pairs"],
        pairing="random",
        seed=1,
        alpha=-2.0,
        freq_sampling="mixture",
        resonant_frac=0.5,
        resonant_spread=3.0,
    )
    if (output / "summary.json").exists() and (output / "best_onres.pt").exists():
        if args.skip_complete:
            previous = json.loads((output / "run_config.json").read_text())
            if (previous["train_config"] != asdict(config)
                    or previous["train_data"] != asdict(train_data)
                    or previous["val_data"] != asdict(val_data)
                    or previous["fingerprint"] != molecule.fingerprint()):
                raise ValueError(f"completed run has a different configuration: {output}")
            print(f"skip completed {output}", flush=True)
            return
        raise FileExistsError(f"completed run exists: {output}; use a new --tag")
    if (output / "run_config.json").exists():
        raise FileExistsError(
            f"incomplete run exists: {output}; qlsgym does not resume optimizer state; "
            "use a new --tag rather than overwriting its checkpoints"
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "molecule": "thf",
        "qlsgym_git_sha": qlsgym_revision(),
        "fingerprint": molecule.fingerprint(),
        "block": args.block,
        "sigma": args.sigma,
        "preset": args.preset,
        "device": args.device,
        "output": str(output),
        "train_config": asdict(config),
        "train_data": asdict(train_data),
        "val_data": asdict(val_data),
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    started = time.time()
    result = train(
        molecule,
        args.block,
        args.sigma,
        config,
        train_data,
        val_data,
        device=args.device,
        out_dir=str(output),
        log_every=1 if args.preset == "smoke" else args.log_every,
        storage_device=args.storage_device,
    )
    summary = {
        "best_val": result["history"]["best_val"],
        "best_epoch": result["history"]["best_epoch"],
        "best_onres": result["history"]["best_onres"],
        "best_onres_infidelity": result["history"]["best_onres"],
        "best_onres_epoch": result["history"]["best_onres_epoch"],
        "n_parameters": result["history"]["n_parameters"],
        "epochs_completed": len(result["history"]["train"]),
        "training_wall_clock_s": time.time() - started,
        "epoch_time_median_s": float(np.median(result["history"]["epoch_time"])),
        "qlsgym_git_sha": metadata["qlsgym_git_sha"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_block(args) -> None:
    """Independent frequency/population seeds; selection data are not a test."""
    molecule = qlsgym.load_molecule("thf")
    directory = run_directory(Path(args.work), args.tag, args.sigma, args.block)
    model = load_model(str(directory / "best_onres.pt"), args.device, molecule=molecule)
    results = {
        "molecule_fingerprint": molecule.fingerprint(),
        "tag": args.tag,
        "block": args.block,
        "sigma": args.sigma,
        "qlsgym_git_sha": qlsgym_revision(),
        "test_seed": args.test_seed,
        "n_freq_per_stratum": args.test_freq,
        "n_init": args.test_init,
        "batch_size": args.batch_size,
        "checkpoint": "best_onres.pt",
    }
    for name, alpha in (("diffuse_alpha1", 1.0), ("control_mix_alpha_minus2", -2.0)):
        summary, raw = evaluate_stratified(
            model, molecule, args.block, args.sigma,
            n_uniform=args.test_freq, n_on=args.test_freq, n_off=args.test_freq,
            n_init=args.test_init, freq_seed=args.test_seed,
            init_seed=args.test_seed + 1, device=args.device,
            alpha=alpha, batch_size=args.batch_size,
        )
        results[name] = summary
        np.savez_compressed(
            directory / f"heldout_{name}.npz",
            **{f"{stratum}_{key}": value for stratum, values in raw.items()
               for key, value in values.items() if isinstance(value, np.ndarray)},
        )
    (directory / "heldout_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


def make_manifest(args) -> None:
    molecule = qlsgym.load_molecule("thf")
    work = Path(args.work).resolve()
    sigmas = ("+", "-") if args.sigmas == "both" else (args.sigmas,)
    runs = {}
    missing = []
    for sigma in sigmas:
        for block in range(molecule.system.n_blocks):
            directory = run_directory(work, args.tag, sigma, block)
            checkpoint = directory / "best_onres.pt"
            if checkpoint.exists():
                runs[(block, sigma)] = str(directory)
            else:
                missing.append(str(checkpoint))
    if missing and not args.allow_partial:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    if not runs:
        raise FileNotFoundError("no checkpoints found")
    manifest = build_manifest(
        molecule,
        args.tag,
        runs,
        source="qlsgym",
        checkpoint_name="best_onres.pt",
        note=(
            f"ThF+ {args.tag} FNO for RL; alpha=-2 mixture; "
            f"{len(runs)}/{2 * molecule.system.n_blocks} block-polarization pairs"
        ),
        require_summary=True,
        check_load=True,
        device=args.device,
    )
    path = manifest.save(work=str(work))
    print(manifest.table())
    print(f"wrote {path}")


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train-block")
    train_parser.add_argument("--work", required=True)
    train_parser.add_argument("--tag", default="rlpilot")
    train_parser.add_argument("--preset", default="pilot", choices=tuple(PRESETS))
    train_parser.add_argument("--block", type=int, required=True, choices=range(12))
    train_parser.add_argument("--sigma", default="+", choices=("+", "-"))
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--storage-device", default="cpu")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=1234)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--skip-complete", action="store_true")
    train_parser.set_defaults(func=train_block)

    evaluate_parser = sub.add_parser("evaluate-block")
    evaluate_parser.add_argument("--work", required=True)
    evaluate_parser.add_argument("--tag", required=True)
    evaluate_parser.add_argument("--block", type=int, required=True, choices=range(12))
    evaluate_parser.add_argument("--sigma", required=True, choices=("+", "-"))
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--test-freq", type=int, default=32)
    evaluate_parser.add_argument("--test-init", type=int, default=128)
    evaluate_parser.add_argument("--test-seed", type=int, default=20260916)
    evaluate_parser.set_defaults(func=evaluate_block)

    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--work", required=True)
    manifest_parser.add_argument("--tag", default="rlpilot")
    manifest_parser.add_argument("--sigmas", default="+", choices=("+", "-", "both"))
    manifest_parser.add_argument("--device", default="cpu")
    manifest_parser.add_argument("--allow-partial", action="store_true")
    manifest_parser.set_defaults(func=make_manifest)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
