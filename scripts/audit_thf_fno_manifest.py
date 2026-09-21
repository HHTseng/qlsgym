#!/usr/bin/env python
"""Structural and branch-aware audit of every checkpoint in a ThF+ manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qlsgym import load_molecule
from qlsgym.surrogate.dataset import (
    random_mixed_populations,
    trajectories_from_columns,
    transfer_columns,
)
from qlsgym.surrogate.embedding import TorchEmbedding
from qlsgym.surrogate.manifest import load_manifest
from qlsgym.surrogate.metrics import infidelity_curve
from qlsgym.surrogate.train import stratified_test_frequencies


GATES = {
    "tau0_identity_tv_max": 0.001,
    "input_linearity_tv_p95": 0.001,
    "off_joint_tv_p95": 0.005,
    "branch_mass_abs_error_p95": 0.005,
    "conditional_tv_mass_ge_1e-2_p95": 0.05,
    "conditional_tv_mass_ge_1e-3_p95": 0.10,
    "block_local_termination_error": 0.005,
}


def quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def ensembles(m: int, n: int, rng: np.random.Generator, policy_states=None):
    result = {
        "diffuse": random_mixed_populations(m, n, rng, 1.0),
        "control_mix": random_mixed_populations(m, n, rng, -2.0),
    }
    base = rng.dirichlet(np.ones(m), size=n)
    vertices = np.eye(m)[np.arange(n) % m]
    result["near_pure"] = 0.995 * vertices + 0.005 * base
    if policy_states is not None and len(policy_states):
        pick = rng.integers(len(policy_states), size=n)
        result["policy"] = policy_states[pick]
    return result


def predict(model, embedding, population, omega):
    omega_tensor = torch.full(
        (len(population),), float(omega), dtype=torch.float64, device=population.device
    )
    if hasattr(model, "propagate"):
        return model.propagate(population, embedding, omega_tensor)
    return model(embedding.build(population, omega_tensor)).transpose(1, 2).double()


def load_policy_states(path, states, n_states: int):
    if not path:
        return None
    data = np.load(path)
    full = np.asarray(data["beliefs"], dtype=np.float64)
    if full.ndim != 2 or full.shape[1] != n_states:
        raise ValueError(f"{path}: beliefs must have shape (N,{n_states})")
    sub = full[:, states]
    mass = sub.sum(1)
    live = mass > 1e-12
    return sub[live] / mass[live, None]


@torch.no_grad()
def audit_pair(args) -> dict:
    molecule = load_molecule("thf")
    key = (args.block, args.sigma)
    engine = load_manifest(
        molecule,
        "audit",
        device=args.device,
        path=str(Path(args.manifest).expanduser().resolve()),
        blocks={key},
    )
    model = engine._models[key]
    entry = engine.manifest.entries[f"{args.block},{args.sigma}"]
    checkpoint = Path(entry.path)
    block = molecule.blocks[args.block]
    m = block.n_states
    embedding = TorchEmbedding(molecule, args.block, args.sigma, args.device)
    frequency_sets = stratified_test_frequencies(
        molecule,
        args.block,
        n_uniform=args.n_freq,
        n_on=args.n_freq,
        n_off=args.n_freq,
        seed=args.seed,
        sigma=args.sigma,
        n_linewidths=1.0,
    )
    policy = load_policy_states(args.policy_states, block.states, molecule.n_states)
    rng = np.random.default_rng(args.seed + 1)
    population_sets = ensembles(m, args.n_init, rng, policy)
    result = {
        "block": args.block,
        "sigma": args.sigma,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "manifest_fingerprint": engine.manifest.fingerprint,
        "manifest_source": engine.manifest.source,
        "provenance": entry.provenance,
        "n_freq_per_stratum": args.n_freq,
        "n_init_per_ensemble": args.n_init,
        "ensembles": {},
    }
    pooled = {
        "identity": [],
        "linearity": [],
        "off_joint": [],
        "branch_mass": [],
        "conditional_1e2": [],
        "conditional_1e3": [],
        "termination_error": [],
    }
    for ensemble_name, population_np in population_sets.items():
        population = torch.as_tensor(
            population_np, dtype=torch.float64, device=args.device
        )
        ensemble_result = {}
        for stratum, omegas in frequency_sets.items():
            metrics = {
                "joint_infidelity": [],
                "joint_tv": [],
                "branch_mass_abs_error": [],
                "conditional_tv_mass_ge_1e-2": [],
                "conditional_tv_mass_ge_1e-3": [],
                "block_local_termination_error": [],
                "tau0_identity_tv": [],
                "input_linearity_tv": [],
                "exact_static_tv": [],
            }
            for omega in omegas:
                columns = transfer_columns(
                    molecule,
                    args.block,
                    np.asarray([omega]),
                    args.sigma,
                    device=args.device,
                )[0]
                truth = trajectories_from_columns(columns[None], population)
                prediction = predict(model, embedding, population, omega)
                metrics["joint_infidelity"].extend(
                    infidelity_curve(prediction, truth).cpu().numpy().ravel()
                )
                joint_tv = 0.5 * (prediction - truth).abs().sum(-1)
                metrics["joint_tv"].extend(joint_tv.cpu().numpy().ravel())
                truth_branch = truth.reshape(len(population), -1, 2, m)
                pred_branch = prediction.reshape(len(population), -1, 2, m)
                mass = truth_branch.sum(-1)
                pred_mass = pred_branch.sum(-1)
                mass_error = (pred_mass - mass).abs()
                metrics["branch_mass_abs_error"].extend(
                    mass_error.cpu().numpy().ravel()
                )
                conditional = 0.5 * (
                    truth_branch / mass.clamp_min(1e-15)[..., None]
                    - pred_branch / pred_mass.clamp_min(1e-15)[..., None]
                ).abs().sum(-1)
                for floor, name in (
                    (1e-2, "conditional_tv_mass_ge_1e-2"),
                    (1e-3, "conditional_tv_mass_ge_1e-3"),
                ):
                    selected = conditional[mass >= floor]
                    metrics[name].extend(selected.cpu().numpy().ravel())
                true_done = (
                    truth_branch / mass.clamp_min(1e-15)[..., None]
                ).amax(-1) >= 0.98
                pred_done = (
                    pred_branch / pred_mass.clamp_min(1e-15)[..., None]
                ).amax(-1) >= 0.98
                reachable = mass >= 1e-3
                metrics["block_local_termination_error"].extend(
                    (true_done[reachable] != pred_done[reachable]).cpu().numpy().astype(float)
                )
                static = torch.zeros_like(truth)
                static[:, :, :m] = population[:, None, :]
                metrics["tau0_identity_tv"].extend(
                    (0.5 * (prediction[:, 0] - static[:, 0]).abs().sum(-1)).cpu().numpy()
                )
                if stratum == "off":
                    metrics["exact_static_tv"].extend(
                        (0.5 * (truth - static).abs().sum(-1)).cpu().numpy().ravel()
                    )
                pair_count = min(len(population) // 2, 16)
                if pair_count:
                    a = population[:pair_count]
                    b = population[pair_count:2 * pair_count]
                    mixed = 0.5 * (a + b)
                    pa = predict(model, embedding, a, omega)
                    pb = predict(model, embedding, b, omega)
                    pm = predict(model, embedding, mixed, omega)
                    metrics["input_linearity_tv"].extend(
                        (0.5 * (pm - 0.5 * (pa + pb)).abs().sum(-1)).cpu().numpy().ravel()
                    )
            ensemble_result[stratum] = {
                name: quantiles(values) for name, values in metrics.items()
            }
            pooled["identity"].extend(metrics["tau0_identity_tv"])
            pooled["linearity"].extend(metrics["input_linearity_tv"])
            pooled["branch_mass"].extend(metrics["branch_mass_abs_error"])
            pooled["conditional_1e2"].extend(metrics["conditional_tv_mass_ge_1e-2"])
            pooled["conditional_1e3"].extend(metrics["conditional_tv_mass_ge_1e-3"])
            pooled["termination_error"].extend(metrics["block_local_termination_error"])
            if stratum == "off":
                pooled["off_joint"].extend(metrics["joint_tv"])
        result["ensembles"][ensemble_name] = ensemble_result
    result["gate_metrics"] = {
        "tau0_identity_tv_max": quantiles(pooled["identity"])["max"],
        "input_linearity_tv_p95": quantiles(pooled["linearity"])["p95"],
        "off_joint_tv_p95": quantiles(pooled["off_joint"])["p95"],
        "branch_mass_abs_error_p95": quantiles(pooled["branch_mass"])["p95"],
        "conditional_tv_mass_ge_1e-2_p95": quantiles(pooled["conditional_1e2"])["p95"],
        "conditional_tv_mass_ge_1e-3_p95": quantiles(pooled["conditional_1e3"])["p95"],
        "block_local_termination_error": float(np.mean(pooled["termination_error"])),
    }
    result["passes"] = {
        name: result["gate_metrics"][name] <= threshold
        for name, threshold in GATES.items()
    }
    result["passes_all"] = all(result["passes"].values())
    return result


def pair_path(output: Path, block: int, sigma: str) -> Path:
    return output / "pairs" / f"block{block}_{'sp' if sigma == '+' else 'sm'}.json"


def pair_command(args):
    path = pair_path(Path(args.output), args.block, args.sigma)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = audit_pair(args)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(path), "gate_metrics": result["gate_metrics"], "passes": result["passes"]}, indent=2))


def summarize_command(args):
    output = Path(args.output)
    paths = sorted((output / "pairs").glob("*.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 pair audits, found {len(paths)}")
    pairs = [json.loads(path.read_text()) for path in paths]
    worst = {}
    for gate, threshold in GATES.items():
        item = max(pairs, key=lambda row: row["gate_metrics"][gate])
        worst[gate] = {
            "value": item["gate_metrics"][gate],
            "threshold": threshold,
            "block": item["block"],
            "sigma": item["sigma"],
            "passes": item["gate_metrics"][gate] <= threshold,
        }
    summary = {
        "status": "complete",
        "pairs": len(pairs),
        "pairs_passing_all": sum(row["passes_all"] for row in pairs),
        "gates": GATES,
        "worst_pair_by_gate": worst,
        "manifest": pairs[0]["manifest"],
        "manifest_fingerprint": pairs[0]["manifest_fingerprint"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# FNO manifest structural audit",
        "",
        f"Pairs passing every engineering gate: **{summary['pairs_passing_all']}/24**.",
        "",
        "| Gate | Worst value | Threshold | Worst pair | Pass |",
        "|---|---:|---:|---|:---:|",
    ]
    for gate, item in worst.items():
        lines.append(
            f"| {gate} | {item['value']:.6g} | {item['threshold']:.6g} | "
            f"{item['block']}, {item['sigma']} | {'yes' if item['passes'] else 'no'} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


def status_command(args):
    output = Path(args.output)
    paths = list((output / "pairs").glob("*.json")) if (output / "pairs").exists() else []
    print(json.dumps({"complete": len(paths), "planned": 24, "summary": (output / "summary.json").exists()}, indent=2))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--manifest", required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--block", type=int, required=True, choices=range(12))
    pair.add_argument("--sigma", required=True, choices=("+", "-"))
    pair.add_argument("--device", default="cuda:0")
    pair.add_argument("--n-freq", type=int, default=16)
    pair.add_argument("--n-init", type=int, default=64)
    pair.add_argument("--seed", type=int, default=20260921)
    pair.add_argument("--policy-states")
    pair.set_defaults(func=pair_command)
    for name, function in (("summarize", summarize_command), ("status", status_command)):
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
        item.set_defaults(func=function)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
