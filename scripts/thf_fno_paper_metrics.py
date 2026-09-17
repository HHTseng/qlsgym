#!/usr/bin/env python
"""ThF+ adaptation of arXiv:2608.03702 Fig. 3/4 and Eqs. 27--31.

Evaluate the preselected best_onres checkpoint, not a test-selected winner.
All reference dynamics are exact PyTorch block propagation, NOT CUDA-Q.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import random_mixed_populations, transfer_columns
from qlsgym.surrogate.embedding import TorchEmbedding
from qlsgym.surrogate.metrics import infidelity_curve
from qlsgym.surrogate.train import evaluate_stratified, load_model


@torch.no_grad()
def trajectory_diagnostics(model, molecule, block, sigma, device):
    embedding = TorchEmbedding(molecule, block, sigma, device)
    geometry = embedding.numpy
    retained = (geometry.w_res >= molecule.window.omega_min) & (geometry.w_res <= molecule.window.omega_max)
    candidates = np.flatnonzero(retained)
    if not len(candidates):
        raise ValueError("block has no in-window retained resonances")
    # Physically preselected strongest in-window resonance, not best test error.
    coupling = np.abs(molecule.blocks[block].omega[geometry.transitions[candidates]])
    omega = float(geometry.w_res[candidates[np.argmax(coupling)]])
    m = molecule.blocks[block].n_states
    rng = np.random.default_rng(20260920)
    p0 = torch.as_tensor(rng.dirichlet(np.ones(m))[None], device=device)
    columns = transfer_columns(molecule, block, np.array([omega]), sigma, device=device)[0]
    truth = torch.einsum("pjm,bm->bpj", columns, p0)
    prediction = model(embedding.build(p0, torch.tensor([omega], device=device))).transpose(1, 2).double()
    swing = (truth[0] - truth[0, :1]).abs().amax(0)
    active = swing > 1e-4
    support = active[None] & (truth[0] > 1e-6)
    relative = (prediction[0] - truth[0]).abs() / truth[0].clamp_min(1e-6)
    mre = 100 * (relative * support).sum(1) / support.sum(1).clamp_min(1)
    infidelity = infidelity_curve(prediction, truth)[0]
    # Test input linearity and near-pure beliefs separately from diffuse inputs.
    population = random_mixed_populations(m, 32, rng, -2.0)
    vertices = np.full((m, m), 0.005 / max(m - 1, 1))
    np.fill_diagonal(vertices, 0.995)
    vertex_errors = []
    conditional_tv, branch_probability_error = [], []
    for lo in range(0, m, 8):
        q = torch.as_tensor(vertices[lo:lo + 8], device=device)
        true = torch.einsum("pjm,bm->bpj", columns, q)
        pred = model(embedding.build(q, torch.tensor([omega], device=device))).transpose(1, 2).double()
        vertex_errors.extend(infidelity_curve(pred, true).mean(1).cpu().tolist())
        u, v = true.reshape(len(q), -1, 2, m), pred.reshape(len(q), -1, 2, m)
        pi, pi_hat = u.sum(-1), v.sum(-1)
        tv = 0.5 * (u / pi.clamp_min(1e-15)[..., None] - v / pi_hat.clamp_min(1e-15)[..., None]).abs().sum(-1)
        conditional_tv.extend(tv[pi >= 1e-3].cpu().tolist())
        branch_probability_error.extend((pi - pi_hat).abs().flatten().cpu().tolist())
    q = torch.as_tensor(population[:2], device=device)
    outputs = model(embedding.build(q, torch.tensor([omega], device=device))).double()
    mixed = model(embedding.build(q.mean(0, keepdim=True), torch.tensor([omega], device=device))).double()
    linearity_tv = 0.5 * (mixed[0] - outputs.mean(0)).abs().sum(0).mean()
    summary = {
        "representative_omega_over_2pi_khz": omega / (2 * np.pi),
        "trajectory_time_average_infidelity": float(infidelity.mean()),
        "tau0_identity_tv": float(0.5 * (prediction[0, 0] - truth[0, 0]).abs().sum()),
        "active_channels": int(active.sum()),
        "mre_definition": "active swing>1e-4, true population>1e-6; exclude zero-support channels",
        "mre_max_percent": float(mre.max()),
        "near_pure_vertex_mean_infidelity": float(np.mean(vertex_errors)),
        "near_pure_vertex_p95_infidelity": float(np.quantile(vertex_errors, 0.95)),
        "near_pure_branch_probability_mean_abs_error": float(np.mean(branch_probability_error)),
        "near_pure_conditional_tv_p95_true_mass_ge_1e-3": float(np.quantile(conditional_tv, 0.95)),
        "input_linearity_mean_tv": float(linearity_tv),
    }
    raw = {"tau": molecule.tau_grid(), "truth": truth[0].cpu().numpy(),
           "prediction": prediction[0].cpu().numpy(), "infidelity": infidelity.cpu().numpy(),
           "mre_percent": mre.cpu().numpy(), "active": active.cpu().numpy()}
    return summary, raw


@torch.no_grad()
def benchmark(model, molecule, block, sigma, device, batch_sizes, repeats):
    embedding = TorchEmbedding(molecule, block, sigma, device)
    geometry = embedding.numpy
    omega = float(geometry.w_res[np.argmin(np.abs(geometry.w_res - molecule.trap.nu_f))])
    rng = np.random.default_rng(20260921)
    m = molecule.blocks[block].n_states
    rows = []
    for kind in ("states", "frequencies"):
        for B in batch_sizes:
            q = torch.as_tensor(rng.dirichlet(np.ones(m), B if kind == "states" else 1), device=device)
            ws = np.full(1 if kind == "states" else B, omega)
            if kind == "frequencies":
                ws = rng.uniform(molecule.window.omega_min, molecule.window.omega_max, B)
            w = torch.as_tensor(np.full(B, omega) if kind == "states" else ws, device=device)
            resident_columns = transfer_columns(molecule, block, ws, sigma, device=device)

            def exact(columns):
                return (torch.einsum("pjm,bm->bpj", columns[0], q) if kind == "states"
                        else torch.einsum("bpjm,m->bpj", columns, q[0]))

            operations = {
                "fno": lambda: model(embedding.build(q.expand(B, -1), w)),
                "exact_build_and_apply": lambda: exact(transfer_columns(molecule, block, ws, sigma, device=device)),
                "exact_cached_columns": lambda: exact(resident_columns),
            }
            row = {"kind": kind, "batch_size": B}
            for name, operation in operations.items():
                operation()  # warm-up, operator/module initialization not timed
                samples = []
                for _ in range(repeats):
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    operation()
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    samples.append(time.perf_counter() - started)
                row[name + "_median_s"] = float(np.median(samples))
            row["cold_exact_speedup"] = row["exact_build_and_apply_median_s"] / row["fno_median_s"]
            row["cached_exact_speedup"] = row["exact_cached_columns_median_s"] / row["fno_median_s"]
            rows.append(row)
    return rows


def draw_accuracy(summary, trajectory, raw, path, title):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    tau = trajectory["tau"]
    indices = np.flatnonzero(trajectory["active"])
    swing = np.ptp(trajectory["truth"], axis=0)
    indices = sorted(indices, key=lambda i: swing[i], reverse=True)[:12]
    m = trajectory["truth"].shape[-1] // 2
    for index, channel in enumerate(indices):
        color = plt.cm.tab20(index / max(len(indices), 1))
        label = f"local {channel % m}, " + ("nu=0" if channel < m else "nu>=1")
        axes[0, 0].plot(tau, trajectory["prediction"][:, channel], color=color, label=label)
        axes[0, 0].plot(tau[::10], trajectory["truth"][::10, channel], "o", color=color, ms=2, fillstyle="none")
    axes[0, 0].set_ylabel("Population (line FNO, circle exact)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 0].set_title("Representative resonant trajectory; <=12 active channels")
    axes[1, 0].semilogy(tau, np.maximum(trajectory["infidelity"], 1e-12), color="#0072B2")
    axes[1, 0].set_ylabel("Full-channel population infidelity")
    right = axes[1, 0].twinx()
    right.plot(tau, trajectory["mre_percent"], color="#009E73", alpha=0.7)
    right.set_ylabel("Positive-support active MRE (%)", color="#009E73")
    for color, name in (("#0072B2", "diffuse"), ("#D55E00", "control_mix")):
        curves = raw[name]["uniform"]["curves"]
        bands = np.percentile(curves, [5, 25, 50, 75, 95], axis=0)
        axes[0, 1].fill_between(tau, np.maximum(bands[0], 1e-12), np.maximum(bands[4], 1e-12), color=color, alpha=0.12)
        axes[0, 1].fill_between(tau, np.maximum(bands[1], 1e-12), np.maximum(bands[3], 1e-12), color=color, alpha=0.2)
        axes[0, 1].semilogy(tau, np.maximum(bands[2], 1e-12), color=color, label=name)
        ws = raw[name]["uniform"]["omegas"] / (2 * np.pi)
        err = raw[name]["uniform"]["time_avg"]
        axes[1, 1].scatter(ws, err, color=color, alpha=0.35, s=9)
        bins = np.array_split(np.argsort(ws), min(12, len(ws)))
        axes[1, 1].semilogy([np.median(ws[b]) for b in bins], [np.median(err[b]) for b in bins], color=color, label=name)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_ylabel("Initial-state-averaged infidelity")
    axes[0, 1].set_title("Median, 25-75%, 5-95% across uniform test frequencies")
    axes[1, 1].axhline(3e-3, color="black", ls="--", lw=0.8, label="paper reference 3e-3 (different molecule)")
    axes[1, 1].set_xlabel("Drive omega/(2 pi) (kHz)")
    axes[1, 1].set_ylabel("Initial-state/time-averaged infidelity")
    axes[1, 1].legend(fontsize=6)
    for axis in (axes[0, 0], axes[1, 0], axes[0, 1]):
        axis.set_xlabel("Single-pulse time tau (ms)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--tag", default="rlprod120v2")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--sigma", choices=("+", "-"), default="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-freq", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/thf_fno_validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    molecule = load_molecule("thf")
    sigma_tag = "sp" if args.sigma == "+" else "sm"
    directory = Path(args.work) / "runs" / f"{args.tag}_{sigma_tag}_block{args.block}"
    checkpoint = directory / "best_onres.pt"
    model = load_model(str(checkpoint), args.device, molecule=molecule)
    summary = {"tag": args.tag, "block": args.block, "sigma": args.sigma,
               "molecule_fingerprint": molecule.fingerprint(), "n_nu": molecule.trap.n_nu,
               "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "n_test_freq_per_stratum": args.n_freq, "n_initial_states": args.n_init,
               "frequency_seed": 20260918, "population_seed": 20260919,
               "device": args.device, "reference": "exact PyTorch eigendecomposition, not CUDA-Q"}
    raw = {}
    for name, alpha in (("diffuse", 1.0), ("control_mix", -2.0)):
        metrics, arrays = evaluate_stratified(
            model, molecule, args.block, args.sigma, n_uniform=args.n_freq,
            n_on=args.n_freq, n_off=args.n_freq, n_init=args.n_init,
            freq_seed=20260918, init_seed=20260919, device=args.device,
            alpha=alpha, batch_size=args.batch_size,
        )
        metrics["uniform_fraction_below_3e-3"] = float((arrays["uniform"]["time_avg"] < 3e-3).mean())
        summary[name] = metrics
        raw[name] = arrays
    summary["trajectory"], trajectory = trajectory_diagnostics(model, molecule, args.block, args.sigma, args.device)
    stem = f"{args.tag}_{sigma_tag}_block{args.block}"
    np.savez_compressed(args.output / f"{stem}_test.npz", **{
        f"{mode}_{stratum}_{key}": value for mode, strata in raw.items()
        for stratum, arrays in strata.items() for key, value in arrays.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(args.output / f"{stem}_trajectory.npz", **trajectory)
    draw_accuracy(summary, trajectory, raw, args.output / f"{stem}_accuracy.png",
                  f"ThF+ block {args.block}, sigma {args.sigma}; {args.n_freq} frequencies x {args.n_init} initial states\nPaper-style metrics; best_onres selection, exact PyTorch reference")
    if not args.skip_timing:
        summary["timing"] = benchmark(model, molecule, args.block, args.sigma, args.device, args.batch_sizes, args.repeats)
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for axis, kind in zip(axes, ("states", "frequencies")):
            rows = [row for row in summary["timing"] if row["kind"] == kind]
            for metric, label in (("fno", "FNO + embedding"), ("exact_build_and_apply", "Exact build + apply"),
                                  ("exact_cached_columns", "Exact cached columns")):
                axis.loglog([r["batch_size"] for r in rows], [r[metric + "_median_s"] for r in rows], "o-", label=label)
            axis.set_xlabel(f"Batch size ({kind})")
            axis.set_ylabel("Median wall-clock time (s)")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        figure.suptitle(f"ThF+ block {args.block} forward propagation ({args.device}); not CUDA-Q")
        figure.tight_layout()
        figure.savefig(args.output / f"{stem}_timing.png", dpi=180)
        plt.close(figure)
    (args.output / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
