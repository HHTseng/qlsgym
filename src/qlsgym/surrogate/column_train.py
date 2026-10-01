"""Training utilities for the physics-constrained transfer-column FNO."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from ..spec import Molecule
from .dataset import random_mixed_populations, resonant_frequency_sample, transfer_columns
from .embedding import TorchEmbedding
from .fno import ColumnFNO, ColumnFNOConfig
from .manifest import table_stamp
from .temporal_fno import TemporalColumnFNO, TemporalColumnFNOConfig


@dataclass(frozen=True)
class ColumnTrainConfig:
    architecture: str = "column"
    n_train_freq: int = 256
    n_val_freq: int = 48
    states_per_frequency: int = 16
    batch_size: int = 2
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 3e-4
    seed: int = 1234
    val_seed: int = 20260922
    n_linewidths: float = 3.0
    column_tv_weight: float = 1.0
    column_ce_weight: float = 0.0
    joint_tv_weight: float = 1.0
    branch_mass_weight: float = 2.0
    conditional_weight: float = 1.0
    infidelity_weight: float = 1.0
    derivative_weight: float = 0.0
    spectral_weight: float = 0.0
    fno: ColumnFNOConfig = field(
        default_factory=lambda: ColumnFNOConfig(
            n_modes=40,
            hidden_channels=64,
            n_layers=3,
            lifting_channel_ratio=3,
            projection_channel_ratio=3,
            factorization="tucker",
            rank=0.6,
            domain_padding=0.1,
            positional_embedding="grid",
            off_resonance_linewidths=3.0,
        )
    )
    temporal: TemporalColumnFNOConfig = field(default_factory=TemporalColumnFNOConfig)


def make_dataset(
    molecule: Molecule,
    block: int,
    sigma: str,
    n_freq: int,
    seed: int,
    n_linewidths: float,
    device,
    chunk: int = 4,
    cache_dir: str | None = None,
) -> dict:
    cache_path = None
    if cache_dir:
        label = "sp" if sigma == "+" else "sm"
        fingerprint = molecule.fingerprint()[:12]
        cache_path = os.path.join(
            cache_dir,
            f"{molecule.name}_{fingerprint}_b{block}_{label}_n{n_freq}_s{seed}_lw{n_linewidths:g}.pt",
        )
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)
    omegas_np = resonant_frequency_sample(
        molecule,
        block,
        n_freq,
        rng,
        sigma,
        n_linewidths=n_linewidths,
    )
    embedding = TorchEmbedding(molecule, block, sigma, device)
    omegas = torch.as_tensor(omegas_np, dtype=torch.float64, device=device)
    controls = embedding.control_channels(omegas, out_dtype=torch.float32).cpu()
    columns = []
    for lo in range(0, n_freq, chunk):
        hi = min(lo + chunk, n_freq)
        columns.append(
            transfer_columns(
                molecule,
                block,
                omegas[lo:hi],
                sigma,
                device=device,
                out_dtype=torch.float32,
            ).cpu()
        )
    data = {
        "omegas": torch.as_tensor(omegas_np, dtype=torch.float64),
        "controls": controls,
        "columns": torch.cat(columns),
        "taus": torch.as_tensor(molecule.tau_grid(), dtype=torch.float32),
        "seed": seed,
        "n_linewidths": n_linewidths,
    }
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = cache_path + f".tmp.{os.getpid()}"
        torch.save(data, temporary)
        os.replace(temporary, cache_path)
    return data


def sample_states(m: int, n: int, rng: np.random.Generator, device) -> torch.Tensor:
    n_vertices = min(m, max(1, n // 2))
    vertices = np.eye(m)[rng.choice(m, size=n_vertices, replace=False)]
    diffuse = random_mixed_populations(m, n - n_vertices, rng, -2.0)
    values = np.concatenate((0.995 * vertices + 0.005 / m, diffuse), axis=0)
    values /= values.sum(1, keepdims=True)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def branch_aware_loss(predicted, truth, states, cfg: ColumnTrainConfig) -> tuple[torch.Tensor, dict]:
    """Compare B̂_{α,k} with B_{α,k} at all instrument levels.

    predicted and truth stack k=0,1 along their 2M output rows. For every
    sampled s_t, the loss compares v_{α,k}, its mass p_k, and the normalized
    posterior F_{α,k}(s_t), in addition to the transfer columns themselves.
    """
    m = states.shape[-1]
    column_tv = 0.5 * (predicted - truth).abs().sum(2).mean()
    column_ce = -(truth * predicted.clamp_min(1e-8).log()).sum(2).mean()
    pred_joint = torch.einsum("bpom,sm->bspo", predicted, states)
    true_joint = torch.einsum("bpom,sm->bspo", truth, states)
    joint_tv = 0.5 * (pred_joint - true_joint).abs().sum(-1).mean()
    pred_v = pred_joint.reshape(*pred_joint.shape[:-1], 2, m)
    true_v = true_joint.reshape(*true_joint.shape[:-1], 2, m)
    pred_p = pred_v.sum(-1)
    true_p = true_v.sum(-1)
    mass_loss = (pred_p - true_p).abs().mean()
    conditional = 0.5 * (
        pred_v / pred_p.clamp_min(1e-8)[..., None]
        - true_v / true_p.clamp_min(1e-8)[..., None]
    ).abs().sum(-1)
    mask = true_p >= 1e-3
    conditional_loss = conditional[mask].mean() if bool(mask.any()) else conditional.mean() * 0.0
    # Exact transfer columns contain many structural zeros (especially at tau=0).
    # sqrt(0) has an infinite derivative, which produces NaN gradients even when
    # the corresponding truth factor is also zero.  A tiny floor keeps the
    # Bhattacharyya overlap differentiable without changing reported accuracy.
    eps = torch.finfo(pred_joint.dtype).eps
    overlap = (
        pred_joint.clamp_min(eps).sqrt() * true_joint.clamp_min(eps).sqrt()
    ).sum(-1)
    infidelity = (1.0 - overlap.square()).clamp_min(0).mean()
    total = (
        cfg.column_tv_weight * column_tv
        + cfg.column_ce_weight * column_ce
        + cfg.joint_tv_weight * joint_tv
        + cfg.branch_mass_weight * mass_loss
        + cfg.conditional_weight * conditional_loss
        + cfg.infidelity_weight * infidelity
    )
    values = {
        "total": float(total.detach()),
        "column_tv": float(column_tv.detach()),
        "column_ce": float(column_ce.detach()),
        "joint_tv": float(joint_tv.detach()),
        "branch_mass": float(mass_loss.detach()),
        "conditional_tv": float(conditional_loss.detach()),
        "infidelity": float(infidelity.detach()),
    }
    return total, values


def temporal_losses(
    predicted: torch.Tensor, truth: torch.Tensor, taus: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match physical-time derivatives and Fourier trajectory content."""
    taus = taus.to(device=predicted.device, dtype=predicted.dtype)
    dt = (taus[1:] - taus[:-1]).clamp_min(torch.finfo(predicted.dtype).eps)
    derivative_pred = (predicted[:, 1:] - predicted[:, :-1]) / dt[None, :, None, None]
    derivative_true = (truth[:, 1:] - truth[:, :-1]) / dt[None, :, None, None]
    derivative = (derivative_pred - derivative_true).abs().mean()

    pred_fft = torch.fft.rfft(predicted, dim=1)
    true_fft = torch.fft.rfft(truth, dim=1)
    frequency = torch.linspace(
        0.0, 1.0, pred_fft.shape[1], device=predicted.device, dtype=predicted.dtype
    )
    weight = 1.0 + 0.25 * frequency.square()
    numerator = (
        weight[None, :, None, None] * (pred_fft - true_fft).abs().square()
    ).sum()
    denominator = (
        weight[None, :, None, None] * true_fft.abs().square()
    ).sum().clamp_min(torch.finfo(predicted.dtype).eps)
    return derivative, numerator / denominator


def complete_loss(predicted, truth, states, taus, cfg: ColumnTrainConfig):
    base, values = branch_aware_loss(predicted, truth, states, cfg)
    if cfg.derivative_weight or cfg.spectral_weight:
        derivative, spectral = temporal_losses(predicted, truth, taus)
    else:
        with torch.no_grad():
            derivative, spectral = temporal_losses(predicted, truth, taus)
    total = base + cfg.derivative_weight * derivative + cfg.spectral_weight * spectral
    values.update(
        base_total=float(base.detach()),
        derivative=float(derivative.detach()),
        spectral=float(spectral.detach()),
        total=float(total.detach()),
    )
    return total, values


@torch.no_grad()
def evaluate(model, data, cfg: ColumnTrainConfig, device, seed: int) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    totals = []
    for lo in range(0, len(data["controls"]), cfg.batch_size):
        hi = min(lo + cfg.batch_size, len(data["controls"]))
        controls = data["controls"][lo:hi].to(device)
        truth = data["columns"][lo:hi].to(device)
        states = sample_states(model.n_states, cfg.states_per_frequency, rng, device)
        _, values = complete_loss(
            model.columns(controls), truth, states, data["taus"], cfg
        )
        totals.append((hi - lo, values))
    count = sum(n for n, _ in totals)
    return {
        key: sum(n * values[key] for n, values in totals) / count
        for key in totals[0][1]
    }


def checkpoint(model, cfg, epoch, validation, molecule):
    return {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": validation["total"],
        "onres_infidelity": validation["infidelity"],
        "meta": model.metadata(),
        "train_config": asdict(cfg),
        "source": "qlsgym",
        "tables": table_stamp(molecule),
    }


def train_column_fno(
    molecule: Molecule,
    block: int,
    sigma: str,
    cfg: ColumnTrainConfig,
    device="cuda",
    out_dir: str | None = None,
    log_every: int = 1,
    cache_dir: str | None = None,
) -> dict:
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_data = make_dataset(
        molecule, block, sigma, cfg.n_train_freq, cfg.seed, cfg.n_linewidths, device,
        cache_dir=cache_dir,
    )
    val_data = make_dataset(
        molecule, block, sigma, cfg.n_val_freq, cfg.val_seed, cfg.n_linewidths, device,
        cache_dir=cache_dir,
    )
    if cfg.architecture == "column":
        model = ColumnFNO.for_block(molecule, block, sigma, cfg.fno).to(device)
    elif cfg.architecture == "temporal":
        model = TemporalColumnFNO.for_block(
            molecule, block, sigma, cfg.temporal
        ).to(device)
    else:
        raise ValueError(f"unknown column architecture {cfg.architecture!r}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    rng = np.random.default_rng(cfg.seed + 17)
    history = []
    best = math.inf
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for epoch in range(cfg.epochs):
        started = time.time()
        model.train()
        order = torch.randperm(len(train_data["controls"]))
        train_values = []
        for lo in range(0, len(order), cfg.batch_size):
            index = order[lo : lo + cfg.batch_size]
            controls = train_data["controls"][index].to(device)
            truth = train_data["columns"][index].to(device)
            states = sample_states(model.n_states, cfg.states_per_frequency, rng, device)
            loss, values = complete_loss(
                model.columns(controls), truth, states, train_data["taus"], cfg
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite column loss at epoch {epoch}, batch {lo // cfg.batch_size}: "
                    f"{values}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_values.append(values)
        scheduler.step()
        validation = evaluate(model, val_data, cfg, device, cfg.val_seed + epoch)
        if not all(np.isfinite(value) for value in validation.values()):
            raise FloatingPointError(
                f"non-finite column validation at epoch {epoch}: {validation}"
            )
        record = {
            "epoch": epoch,
            "train_total": float(np.mean([item["total"] for item in train_values])),
            **{f"val_{key}": value for key, value in validation.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        if validation["total"] < best:
            best = validation["total"]
            if out_dir:
                torch.save(
                    checkpoint(model, cfg, epoch, validation, molecule),
                    os.path.join(out_dir, "best.pt"),
                )
        if epoch % log_every == 0 or epoch + 1 == cfg.epochs:
            print(json.dumps(record), flush=True)
    summary = {
        "architecture": model.metadata()["architecture"],
        "block": block,
        "sigma": sigma,
        "best_val_loss": best,
        "best_onres_infidelity": min(row["val_infidelity"] for row in history),
        "epochs_completed": len(history),
        "n_parameters": model.n_parameters(),
        "epoch_time_median_s": float(np.median([row["seconds"] for row in history])),
        "train_config": asdict(cfg),
    }
    if out_dir:
        with open(os.path.join(out_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
    return summary
