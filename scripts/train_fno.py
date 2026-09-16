#!/usr/bin/env python
"""Train one (molecule, block, sigma) FNO surrogate."""
from __future__ import annotations

import argparse
import json
import os

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import DataConfig, work_dir
from qlsgym.surrogate.fno import FNOConfig
from qlsgym.surrogate.train import TrainConfig, evaluate_run, load_model, train


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecule", default="thf", choices=["thf", "h3o", "synthetic"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--sigma", default="+", choices=["+", "-"])
    ap.add_argument("--tag", default="run", help="names the output directory "
                    "(<tag>_<sp|sm>_block<b> under $QLSGYM_WORK/runs, unless --out)")
    ap.add_argument("--out", default=None, help="output directory; overrides --tag's default")

    g = ap.add_argument_group("training data (qlsgym.surrogate.dataset.DataConfig)")
    g.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration; "
                   "negative selects the log-uniform mixture over [10^alpha, 1] "
                   "(the control-search distribution, ThF 'mix_*' runs use -2)")
    g.add_argument("--freq-sampling", default="uniform", choices=["uniform", "mixture"],
                   help="'mixture' oversamples near retained resonances "
                   "(documented deviation from the paper, ThF decision D9)")
    g.add_argument("--resonant-frac", type=float, default=0.5)
    g.add_argument("--resonant-spread", type=float, default=3.0)
    g.add_argument("--n-freq", type=int, default=1000)
    g.add_argument("--n-init", type=int, default=1000)
    g.add_argument("--n-pairs", type=int, default=None)
    g.add_argument("--pairing", default="random", choices=["random", "product", "zip"])
    g.add_argument("--train-seed", type=int, default=1234)
    g.add_argument("--val-n-freq", type=int, default=2000)
    g.add_argument("--val-n-init", type=int, default=1000)
    g.add_argument("--val-n-pairs", type=int, default=2000)
    g.add_argument("--val-seed", type=int, default=1)
    g.add_argument("--sel-n-freq", type=int, default=64, help="on-resonance "
                   "frequencies used to select best_onres.pt (0 disables it)")
    g.add_argument("--sel-n-init", type=int, default=64)

    a = ap.add_argument_group("optimisation (qlsgym.surrogate.train.TrainConfig)")
    a.add_argument("--epochs", type=int, default=200)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--lr", type=float, default=5e-4)
    a.add_argument("--weight-decay", type=float, default=3e-4)
    a.add_argument("--scheduler-step", type=int, default=5)
    a.add_argument("--scheduler-gamma", type=float, default=0.7)
    a.add_argument("--activity-lambda", type=float, default=1.0)
    a.add_argument("--activity-eps", type=float, default=1e-8)
    a.add_argument("--reduction", default="sum", choices=["sum", "mean"])
    a.add_argument("--no-per-sample-sqrt", action="store_true",
                   help="deviation: drop Eq. 21's outer per-sample square root")

    f = ap.add_argument_group("architecture (qlsgym.surrogate.fno.FNOConfig, paper Table 2)")
    f.add_argument("--n-modes", type=int, default=60)
    f.add_argument("--hidden-channels", type=int, default=256)
    f.add_argument("--n-layers", type=int, default=4)
    f.add_argument("--lifting-channel-ratio", type=int, default=5)
    f.add_argument("--projection-channel-ratio", type=int, default=5)
    f.add_argument("--factorization", default="tucker")
    f.add_argument("--rank", type=float, default=0.8)
    f.add_argument("--domain-padding", type=float, default=0.1)

    e = ap.add_argument_group("test-set evaluation (summary.json)")
    e.add_argument("--test-n-freq", type=int, default=100)
    e.add_argument("--test-n-init", type=int, default=500)
    e.add_argument("--test-seed", type=int, default=20260101)
    e.add_argument("--test-n-on", type=int, default=100, help="on-resonance test "
                   "frequencies (0 disables the stratified evaluation)")
    e.add_argument("--test-n-off", type=int, default=100)
    e.add_argument("--n-linewidths", type=float, default=1.0)

    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--storage-device", default=None,
                    help="where the materialised split lives; 'cpu' streams batches to the "
                    "GPU (needed for large blocks at large dataset sizes)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    import torch

    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    molecule = load_molecule(args.molecule)

    out_dir = args.out or os.path.join(
        work_dir(), "runs", f"{args.tag}_{'sp' if args.sigma == '+' else 'sm'}_block{args.block}")
    os.makedirs(out_dir, exist_ok=True)

    fno_cfg = FNOConfig(n_modes=args.n_modes, hidden_channels=args.hidden_channels, n_layers=args.n_layers,
                        lifting_channel_ratio=args.lifting_channel_ratio,
                        projection_channel_ratio=args.projection_channel_ratio,
                        factorization=args.factorization, rank=args.rank, domain_padding=args.domain_padding)
    cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                      scheduler_step=args.scheduler_step, scheduler_gamma=args.scheduler_gamma,
                      activity_lambda=args.activity_lambda, activity_eps=args.activity_eps,
                      reduction=args.reduction, per_sample_sqrt=not args.no_per_sample_sqrt,
                      train_seed=args.train_seed, val_seed=args.val_seed,
                      sel_n_freq=args.sel_n_freq, sel_n_init=args.sel_n_init, fno=fno_cfg)
    train_data = DataConfig(n_freq=args.n_freq, n_init=args.n_init, n_pairs=args.n_pairs, pairing=args.pairing,
                            seed=args.train_seed, alpha=args.alpha, freq_sampling=args.freq_sampling,
                            resonant_frac=args.resonant_frac, resonant_spread=args.resonant_spread)
    val_data = DataConfig(n_freq=args.val_n_freq, n_init=args.val_n_init, n_pairs=args.val_n_pairs,
                          pairing="random", seed=args.val_seed, alpha=args.alpha,
                          freq_sampling=args.freq_sampling, resonant_frac=args.resonant_frac,
                          resonant_spread=args.resonant_spread)

    print(json.dumps({"molecule": molecule.name, "fingerprint": molecule.fingerprint(), "block": args.block,
                      "sigma": args.sigma, "out": out_dir, "device": device,
                      "train_config": {k: v for k, v in vars(cfg).items() if k != "fno"},
                      "fno_config": vars(fno_cfg), "train_data": vars(train_data), "val_data": vars(val_data)},
                     indent=2, default=str), flush=True)

    result = train(molecule, args.block, args.sigma, cfg, train_data, val_data, device=device, out_dir=out_dir,
                   log_every=args.log_every, verbose=not args.quiet, storage_device=args.storage_device)

    # Reload the checkpoint the run itself prefers (on-resonance selection when
    # available, else the paper's validation-loss selection) for the held-out
    # test-set evaluation, exactly as thffno/fnorepl's train_fno.py does.
    ckpt_name = "best_onres.pt" if result["history"].get("best_onres_epoch", -1) >= 0 else "best.pt"
    model = load_model(os.path.join(out_dir, ckpt_name), device, molecule=molecule)
    selection = "on_resonance" if ckpt_name == "best_onres.pt" else "val_loss"

    summary = evaluate_run(model, molecule, args.block, args.sigma, history=result["history"], out_dir=out_dir,
                           n_test_freq=args.test_n_freq, n_test_init=args.test_n_init, n_on=args.test_n_on,
                           n_off=args.test_n_off, test_seed=args.test_seed, n_linewidths=args.n_linewidths,
                           device=device, selection=selection,
                           extra={"tag": args.tag, "checkpoint": ckpt_name})
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_dir}/{{best.pt, best_onres.pt, history.json, summary.json, test_eval.npz}}")


if __name__ == "__main__":
    main()
