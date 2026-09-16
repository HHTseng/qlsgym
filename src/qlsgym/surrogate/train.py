"""Training loop for the per-block FNO surrogate (paper Sec. II.3, Appendix B)."""

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
from .dataset import (BlockSplit, DataConfig, build_split, random_mixed_populations,
                      resonant_frequency_sample, off_resonant_frequency_sample,
                      sample_frequencies, trajectories_from_columns, transfer_columns)
from .embedding import Embedding, TorchEmbedding
from .fno import BlockFNO, FNOConfig
from .manifest import table_stamp
from .metrics import (active_fraction, infidelity_curve, is_on_resonance, percentile_bands,
                      static_baseline, stratified_summary)

Tensor = torch.Tensor


@dataclass
class TrainConfig:
    """Paper Table 2 (Appendix B.3) plus the two unspecified loss constants."""

    batch_size: int = 32
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 3e-4
    scheduler_step: int = 5
    scheduler_gamma: float = 0.7
    # Eq. 20; NOT given in the paper.
    activity_lambda: float = 1.0
    activity_eps: float = 1e-8
    # Eq. 21 sums over the batch; AdamW is invariant to the rescaling.
    reduction: str = "sum"
    # Eq. 21's per-sample outer square root (True is the paper).
    per_sample_sqrt: bool = True
    train_seed: int = 1234
    val_seed: int = 1
    # on-resonance checkpoint-selection set (0 disables best_onres.pt)
    sel_n_freq: int = 64
    sel_n_init: int = 64
    sel_seed: int = 777
    fno: FNOConfig = field(default_factory=FNOConfig)


# Loss


def weighted_loss(p_pred: Tensor, p_true: Tensor, weights: Tensor,
                  reduction: str = "sum", per_sample_sqrt: bool = True) -> Tensor:
    """Eq. 21."""
    se = (p_pred - p_true).square() * weights[:, None, :]
    # d sqrt(x)/dx -> inf as x -> 0: the additive eps regularises the
    # gradient of a perfect prediction (perturbs a 1e-3 loss by 5e-7 relative).
    per_sample = se.mean(dim=-1).mean(dim=-1).clamp_min(0)
    if per_sample_sqrt:
        per_sample = (per_sample + 1e-12).sqrt()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")


@torch.no_grad()
def evaluate_loss(model: nn.Module, split: BlockSplit, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, len(split)
    for lo in range(0, n, cfg.batch_size):
        idx = torch.arange(lo, min(lo + cfg.batch_size, n))
        xb, yb, wb = split.batch(idx, device)
        pred = model(xb).transpose(1, 2)
        total += float(weighted_loss(pred.float(), yb, wb, "sum"))
    return total / n


@torch.no_grad()
def evaluate_onres_infidelity(model: nn.Module, split: BlockSplit, cfg: TrainConfig, device: torch.device) -> float:
    """Mean <I_p>_tau (Eqs. 28, 30) on a purely on-resonance split."""
    model.eval()
    n, total = len(split), 0.0
    bs = max(cfg.batch_size, 256)
    for lo in range(0, n, bs):
        idx = torch.arange(lo, min(lo + bs, n))
        xb, yb, _ = split.batch(idx, device)
        pred = model(xb).transpose(1, 2).double()
        total += float(infidelity_curve(pred, yb.double()).mean(-1).sum())
    model.train()
    return total / n


# Training


def checkpoint_dict(model: BlockFNO, cfg: TrainConfig, train_data: DataConfig, val_data: DataConfig,
                    epoch: int, val_loss: float, onres: float, tables: dict | None = None) -> dict:
    """What best*.pt contains."""
    ck = {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
          "onres_infidelity": onres, "meta": model.metadata(), "train_config": asdict(cfg),
          "train_data": asdict(train_data), "val_data": asdict(val_data), "source": "qlsgym"}
    if tables is not None:
        ck["tables"] = dict(tables)
    return ck


def train(
    molecule: Molecule,
    block_index: int,
    sigma: str,
    cfg: TrainConfig,
    train_data: DataConfig,
    val_data: DataConfig,
    device: torch.device | str = "cuda",
    out_dir: str | None = None,
    log_every: int = 10,
    verbose: bool = True,
    storage_device: torch.device | str | None = None,
    splits: tuple[BlockSplit, BlockSplit] | None = None,
) -> dict:
    """Train one (molecule, block, sigma) surrogate."""
    device = torch.device(device)
    torch.manual_seed(cfg.train_seed)
    np.random.seed(cfg.train_seed)
    kw = dict(activity_lambda=cfg.activity_lambda, activity_eps=cfg.activity_eps, storage_device=storage_device)
    if splits is None:
        tr = build_split(molecule, block_index, train_data, sigma, device, **kw)
        va = build_split(molecule, block_index, val_data, sigma, device, **kw)
    else:
        tr, va = splits
    sel = None
    if cfg.sel_n_freq and cfg.sel_n_init:
        sel = build_split(molecule, block_index, DataConfig(
            n_freq=cfg.sel_n_freq, n_init=cfg.sel_n_init, n_pairs=cfg.sel_n_freq * cfg.sel_n_init,
            pairing="product", seed=cfg.sel_seed, alpha=train_data.alpha, freq_sampling="mixture",
            resonant_frac=1.0, resonant_spread=1.0), sigma, device, **kw)

    model = BlockFNO.for_block(molecule, block_index, sigma, cfg.fno).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    stamp = table_stamp(molecule)   # hashed once, before training, so the stamp is the tables trained on
    n = len(tr)
    gen = torch.Generator(device="cpu").manual_seed(cfg.train_seed)
    if verbose:
        print(f"{molecule.name} block {block_index} sigma{sigma}: train {n} samples on {tr.storage_device}, "
              f"val {len(va)}; model {model.n_parameters():,} params", flush=True)
    history = {"train": [], "val": [], "lr": [], "epoch_time": [], "onres": []}
    best, best_epoch, best_onres, best_onres_epoch = math.inf, -1, math.inf, -1
    if sel is not None:
        with torch.no_grad():
            history["onres_static"] = float(static_baseline(sel.p_true.to(device).double()).mean())
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        t_start = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo: lo + cfg.batch_size]
            xb, yb, wb = tr.batch(idx, device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).transpose(1, 2)
            loss = weighted_loss(pred.float(), yb, wb, cfg.reduction, cfg.per_sample_sqrt)
            loss.backward()
            opt.step()
            running += float(loss) * (1.0 if cfg.reduction == "sum" else idx.numel())
        sched.step()

        train_loss = running / n
        val_loss = evaluate_loss(model, va, cfg, device)
        onres = evaluate_onres_infidelity(model, sel, cfg, device) if sel is not None else float("nan")
        history["train"].append(train_loss); history["val"].append(val_loss); history["onres"].append(onres)
        history["lr"].append(float(opt.param_groups[0]["lr"])); history["epoch_time"].append(time.time() - t_start)

        def _save(name: str) -> None:
            if out_dir:
                torch.save(checkpoint_dict(model, cfg, train_data, val_data, epoch, val_loss, onres,
                                           tables=stamp),
                           os.path.join(out_dir, name))

        if val_loss < best:
            best, best_epoch = val_loss, epoch
            _save("best.pt")
        if sel is not None and onres < best_onres:
            best_onres, best_onres_epoch = onres, epoch
            _save("best_onres.pt")
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:4d}  train {train_loss:.4e}  val {val_loss:.4e}  onres {onres:.4e}"
                  f"  lr {opt.param_groups[0]['lr']:.2e}  {history['epoch_time'][-1]:.1f}s", flush=True)

    history.update(best_val=best, best_epoch=best_epoch, best_onres=None if best_onres_epoch < 0 else best_onres,
                   best_onres_epoch=best_onres_epoch, n_parameters=model.n_parameters())
    if out_dir:
        with open(os.path.join(out_dir, "history.json"), "w") as fh:
            json.dump(history, fh)
    return {"model": model, "history": history, "train": tr, "val": va}


# Loading


class FingerprintMismatch(RuntimeError):
    """The checkpoint was trained against different physics."""


def load_model(
    path: str,
    device: torch.device | str = "cpu",
    molecule: Molecule | None = None,
    allow_mismatch: bool = False,
    legacy: bool = False,
) -> BlockFNO:
    """Load a best*.pt checkpoint into a BlockFNO."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    fp = meta.get("fingerprint")
    if molecule is not None:
        want = molecule.fingerprint()
        if fp is None:
            if not (legacy or allow_mismatch):
                raise FingerprintMismatch(
                    f"{path}: checkpoint has no qlsgym fingerprint (source-package checkpoint); "
                    f"load it through a manifest with source='thffno'/'fnorepl' or pass legacy=True")
        elif fp != want and not allow_mismatch:
            raise FingerprintMismatch(
                f"{path}: trained against fingerprint {fp} but molecule {molecule.name!r} is {want}")
        if meta.get("molecule") not in (None, molecule.name) and not allow_mismatch:
            raise FingerprintMismatch(f"{path}: trained for molecule {meta['molecule']!r}, not {molecule.name!r}")
        emb = Embedding(molecule, int(meta["block_index"]), meta["sigma"])
        if (meta["in_channels"], meta["out_channels"]) != (emb.n_channels, 2 * emb.n_states):
            raise FingerprintMismatch(
                f"{path}: channel geometry ({meta['in_channels']}, {meta['out_channels']}) does not match "
                f"{molecule.name} block {meta['block_index']} sigma{meta['sigma']} "
                f"({emb.n_channels}, {2 * emb.n_states})")
        if meta.get("n_nu") not in (None, molecule.trap.n_nu) and not allow_mismatch:
            raise FingerprintMismatch(f"{path}: trained with n_nu={meta['n_nu']}, molecule has {molecule.trap.n_nu}")
    model = BlockFNO(meta["in_channels"], meta["out_channels"], FNOConfig(**meta["config"]),
                     block_index=meta["block_index"], sigma=meta["sigma"],
                     molecule_name=meta.get("molecule", molecule.name if molecule else None),
                     fingerprint=fp, n_nu=meta.get("n_nu")).to(device)
    # neuralop injects a non-tensor _metadata entry into state_dict.
    sd = {k: v for k, v in ckpt["state_dict"].items() if k != "_metadata"}
    model.load_state_dict(sd)
    model.eval()
    return model


# Test-set evaluation (Eqs. 27-30, stratified on/off resonance)


@torch.no_grad()
def evaluate_test_frequencies(
    model: nn.Module, molecule: Molecule, block_index: int, omegas: np.ndarray, n_init: int = 500,
    sigma: str = "+", seed: int = 20260101, alpha: float = 1.0, device: torch.device | str = "cuda",
    batch_size: int = 500,
) -> dict:
    """Eqs. 29-30 on unseen frequencies: infidelity averaged over ensemble, then pulse time."""
    device = torch.device(device)
    model = model.to(device).eval()
    m = molecule.blocks[block_index].n_states
    n_tau = molecule.window.n_tau
    rng = np.random.default_rng(seed)
    p0 = torch.as_tensor(random_mixed_populations(m, n_init, rng, alpha), dtype=torch.float64, device=device)
    emb = TorchEmbedding(molecule, block_index, sigma, device)
    omegas_t = torch.as_tensor(np.asarray(omegas, dtype=np.float64), dtype=torch.float64, device=device)
    curves = torch.empty((omegas_t.numel(), n_tau), dtype=torch.float64, device=device)
    stat = torch.empty_like(curves)
    swing = torch.empty(omegas_t.numel(), dtype=torch.float64, device=device)
    for k in range(omegas_t.numel()):
        t0 = transfer_columns(molecule, block_index, omegas_t[k:k + 1], sigma, device=device,
                              out_dtype=torch.float64)[0]                    # (P, 2M, M)
        acc = torch.zeros(n_tau, dtype=torch.float64, device=device)
        acc_s = torch.zeros_like(acc)
        sw = torch.zeros((), dtype=torch.float64, device=device)
        for lo in range(0, n_init, batch_size):
            hi = min(lo + batch_size, n_init)
            pb = p0[lo:hi]
            p_true = trajectories_from_columns(t0[None], pb)                 # (b, P, 2M)
            x = emb.build(pb, omegas_t[k].expand(hi - lo), out_dtype=torch.float32)
            pred = model(x).transpose(1, 2).double()
            acc += infidelity_curve(pred, p_true).sum(0)
            acc_s += static_baseline(p_true).sum(0)
            sw = torch.maximum(sw, (p_true - p_true[:, :1, :]).abs().amax())
        curves[k] = acc / n_init
        stat[k] = acc_s / n_init
        swing[k] = sw
    c, s = curves.cpu().numpy(), stat.cpu().numpy()
    return {"omegas": np.asarray(omegas, dtype=np.float64), "curves": c, "time_avg": c.mean(1),
            "static_curves": s, "static_time_avg": s.mean(1), "swing": swing.cpu().numpy()}


def stratified_test_frequencies(
    molecule: Molecule, block_index: int, n_uniform: int = 100, n_on: int = 100, n_off: int = 100,
    seed: int = 20260101, sigma: str = "+", n_linewidths: float = 1.0,
) -> dict[str, np.ndarray]:
    """Three test-frequency sets: uniform (the paper's sampling), on and off resonance."""
    rng = np.random.default_rng(seed)
    return {"uniform": sample_frequencies(molecule, n_uniform, rng),
            "on": resonant_frequency_sample(molecule, block_index, n_on, rng, sigma, n_linewidths),
            "off": off_resonant_frequency_sample(molecule, block_index, n_off, rng, sigma, n_linewidths)}


@torch.no_grad()
def evaluate_stratified(
    model: nn.Module, molecule: Molecule, block_index: int, sigma: str = "+", n_uniform: int = 100,
    n_on: int = 100, n_off: int = 100, n_init: int = 500, freq_seed: int = 20260101,
    init_seed: int = 20260102, n_linewidths: float = 1.0, device: torch.device | str = "cuda", **kw,
) -> tuple[dict, dict]:
    """Eqs. 27-30 on the uniform, on- and off-resonance test sets."""
    sets = stratified_test_frequencies(molecule, block_index, n_uniform, n_on, n_off, freq_seed, sigma, n_linewidths)
    summary: dict = {"n_linewidths": n_linewidths,
                     "active_fraction": active_fraction(molecule, block_index, sigma, n_linewidths)}
    raw: dict = {}
    for name, ws in sets.items():
        ev = evaluate_test_frequencies(model, molecule, block_index, ws, n_init=n_init, sigma=sigma,
                                       seed=init_seed, device=device, **kw)
        on = is_on_resonance(molecule, ws, block_index, sigma, n_linewidths)
        summary.update(stratified_summary(ev["time_avg"], ev["static_time_avg"], on, prefix=f"{name}_"))
        summary[f"{name}_swing_median"] = float(np.median(ev["swing"]))
        raw[name] = {**ev, "on_resonance": on}
    return summary, raw


def evaluate_run(
    model: nn.Module, molecule: Molecule, block_index: int, sigma: str, history: dict | None = None,
    out_dir: str | None = None, n_test_freq: int = 100, n_test_init: int = 500, n_on: int = 100,
    n_off: int = 100, test_seed: int = 20260101, n_linewidths: float = 1.0,
    device: torch.device | str = "cuda", extra: dict | None = None, selection: str = "on_resonance",
) -> dict:
    """End-of-training evaluation; writes summary.json, test_eval.npz and stratified_eval.npz."""
    ws = sample_frequencies(molecule, n_test_freq, np.random.default_rng(test_seed))
    ev = evaluate_test_frequencies(model, molecule, block_index, ws, n_init=n_test_init, sigma=sigma,
                                   seed=test_seed + 1, device=device)
    bands = percentile_bands(ev["curves"])
    ta = ev["time_avg"]
    h = history or {}
    summary = {
        "molecule": molecule.name, "fingerprint": molecule.fingerprint(), "block": int(block_index),
        "sigma": sigma, "selection": selection,
        "best_val_loss": h.get("best_val"), "best_epoch": h.get("best_epoch"),
        "best_onres_infidelity": h.get("best_onres"), "best_onres_epoch": h.get("best_onres_epoch"),
        "final_val_loss": (h.get("val") or [None])[-1], "n_parameters": h.get("n_parameters"),
        "epoch_time_median_s": float(np.median(h["epoch_time"])) if h.get("epoch_time") else None,
        "median_curve_median_over_tau": float(np.median(bands[50])),
        "median_curve_max_over_tau": float(np.max(bands[50])),
        "p95_curve_max_over_tau": float(np.max(bands[95])),
        "time_avg_median": float(np.median(ta)), "time_avg_mean": float(np.mean(ta)),
        "time_avg_p95": float(np.percentile(ta, 95)), "time_avg_max": float(np.max(ta)),
        "frac_time_avg_below_3e-3": float(np.mean(ta < 3e-3)),
        "static_time_avg_median": float(np.median(ev["static_time_avg"])),
        "frac_freq_fno_beats_static": float(np.mean(ta < ev["static_time_avg"])),
        "swing_median": float(np.median(ev["swing"])), "swing_max": float(np.max(ev["swing"])),
    }
    summary.update(extra or {})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "test_eval.npz"),
                            **{k: v for k, v in ev.items() if isinstance(v, np.ndarray)})
    if n_on > 0:
        strat, raw = evaluate_stratified(model, molecule, block_index, sigma, n_uniform=n_test_freq, n_on=n_on,
                                         n_off=n_off, n_init=n_test_init, freq_seed=test_seed,
                                         init_seed=test_seed + 1, n_linewidths=n_linewidths, device=device)
        summary.update({f"strat_{k}": v for k, v in strat.items()})
        if out_dir:
            np.savez_compressed(os.path.join(out_dir, "stratified_eval.npz"),
                                **{f"{name}_{k}": v for name, e in raw.items() for k, v in e.items()
                                   if isinstance(v, np.ndarray)})
    if out_dir:
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary
