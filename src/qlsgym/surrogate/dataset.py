from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..spec import Molecule
from .embedding import TorchEmbedding, embedding_delta_max, resonance_geometry_of
from .metrics import is_on_resonance

Tensor = torch.Tensor


def work_dir() -> str:
    return os.environ.get("QLSGYM_WORK") or os.path.expanduser("~/.qlsgym")


def random_mixed_populations(
    n_states: int, n_samples: int, rng: np.random.Generator, alpha: float = 1.0
) -> np.ndarray:
    if alpha > 0:
        return rng.dirichlet(np.full(n_states, alpha), size=n_samples)
    lo = float(alpha)
    a = 10.0 ** rng.uniform(lo, 0.0, size=n_samples)
    return np.stack([rng.dirichlet(np.full(n_states, ai)) for ai in a])


def mask_carriers(molecule: Molecule, omegas: np.ndarray) -> np.ndarray:
    ok = np.ones(np.shape(omegas), dtype=bool)
    for lo, hi in molecule.window.carrier_bands:
        ok &= ~((omegas >= lo) & (omegas <= hi))
    return ok


def sample_frequencies(
    molecule: Molecule,
    n: int,
    rng: np.random.Generator,
    lo: float | None = None,
    hi: float | None = None,
    mask_carrier_bands: bool = True,
) -> np.ndarray:
    w = molecule.window
    lo = w.omega_min if lo is None else float(lo)
    hi = w.omega_max if hi is None else float(hi)
    out = np.sort(rng.uniform(lo, hi, size=n))
    if mask_carrier_bands and len(w.carrier_bands):
        for _ in range(64):
            bad = ~mask_carriers(molecule, out)
            if not bad.any():
                break
            out[bad] = rng.uniform(lo, hi, size=int(bad.sum()))
        out = np.sort(out)
    return out


def resonant_frequency_sample(
    molecule: Molecule,
    block_index: int,
    n: int,
    rng: np.random.Generator,
    sigma: str = "+",
    n_linewidths: float = 1.0,
    lo: float | None = None,
    hi: float | None = None,
    max_tries: int = 200,
) -> np.ndarray:
    w = molecule.window
    lo = w.omega_min if lo is None else float(lo)
    hi = w.omega_max if hi is None else float(hi)
    w_res, lw = resonance_geometry_of(molecule, block_index, sigma)
    out: list[np.ndarray] = []
    have = 0
    for _ in range(max_tries):
        q = rng.integers(0, w_res.size, size=4 * max(n - have, 1))
        x = w_res[q] + rng.uniform(-n_linewidths, n_linewidths, size=q.size) * lw[q]
        x = x[(x >= lo) & (x <= hi)]
        x = x[mask_carriers(molecule, x)]
        out.append(x)
        have += x.size
        if have >= n:
            break
    if have < n:
        raise RuntimeError(f"block {block_index}: only {have} of {n} resonant frequencies fell inside the window")
    return np.sort(np.concatenate(out)[:n])


def off_resonant_frequency_sample(
    molecule: Molecule,
    block_index: int,
    n: int,
    rng: np.random.Generator,
    sigma: str = "+",
    n_linewidths: float = 1.0,
    lo: float | None = None,
    hi: float | None = None,
    max_tries: int = 200,
) -> np.ndarray:
    w = molecule.window
    lo = w.omega_min if lo is None else float(lo)
    hi = w.omega_max if hi is None else float(hi)
    out: list[np.ndarray] = []
    have = 0
    for _ in range(max_tries):
        x = rng.uniform(lo, hi, size=4 * max(n - have, 1))
        x = x[~is_on_resonance(molecule, x, block_index, sigma, n_linewidths)]
        x = x[mask_carriers(molecule, x)]
        out.append(x)
        have += x.size
        if have >= n:
            break
    if have < n:
        raise RuntimeError(f"block {block_index}: could not draw {n} off-resonant frequencies")
    return np.sort(np.concatenate(out)[:n])


@dataclass(frozen=True)
class DataConfig:
    n_freq: int = 1000
    n_init: int = 1000
    n_pairs: int | None = None
    pairing: str = "random"
    seed: int = 1234
    alpha: float = 1.0
    freq_sampling: str = "uniform"
    resonant_frac: float = 0.5
    resonant_spread: float = 3.0
    omega_lo: float | None = None
    omega_hi: float | None = None

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def sample_training_frequencies(
    molecule: Molecule, block_index: int, cfg: DataConfig, rng: np.random.Generator, sigma: str = "+"
) -> np.ndarray:
    if cfg.freq_sampling == "uniform":
        return sample_frequencies(molecule, cfg.n_freq, rng, cfg.omega_lo, cfg.omega_hi)
    if cfg.freq_sampling == "mixture":
        n_res = int(round(cfg.resonant_frac * cfg.n_freq))
        parts = []
        if n_res:
            parts.append(resonant_frequency_sample(
                molecule, block_index, n_res, rng, sigma, cfg.resonant_spread, cfg.omega_lo, cfg.omega_hi))
        if cfg.n_freq - n_res:
            parts.append(sample_frequencies(molecule, cfg.n_freq - n_res, rng, cfg.omega_lo, cfg.omega_hi))
        return np.sort(np.concatenate(parts))
    raise ValueError(f"unknown freq_sampling {cfg.freq_sampling!r}")


def pair_indices(cfg: DataConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if cfg.pairing == "zip":
        if cfg.n_freq != cfg.n_init:
            raise ValueError("pairing='zip' requires n_freq == n_init")
        k = np.arange(cfg.n_freq)
        return k, k.copy()
    if cfg.pairing == "product":
        k, l = np.meshgrid(np.arange(cfg.n_freq), np.arange(cfg.n_init), indexing="ij")
        return k.ravel(), l.ravel()
    if cfg.pairing == "random":
        n = cfg.n_pairs if cfg.n_pairs is not None else max(cfg.n_freq, cfg.n_init)
        return rng.integers(0, cfg.n_freq, size=n), rng.integers(0, cfg.n_init, size=n)
    raise ValueError(f"unknown pairing {cfg.pairing!r}")


def transfer_columns(
    molecule: Molecule,
    block_index: int,
    omegas: np.ndarray | Tensor,
    sigma: str = "+",
    device: torch.device | str = "cpu",
    out_dtype: torch.dtype = torch.float64,
    **kw,
) -> Tensor:
    from ..physics.torch_ops import transfer_columns as _tc

    return _tc(molecule, block_index, omegas, sigma, tau_indices=None,
               device=device, out_dtype=out_dtype, **kw)


def trajectories_from_columns(t0: Tensor, p0: Tensor) -> Tensor:
    return (t0 * p0[..., None, None, :]).sum(-1)


def activity_weights(p_true: Tensor, lam: float = 1.0, eps: float = 1e-8) -> Tensor:
    a = (p_true - p_true[..., :1, :]).abs().amax(dim=-2)
    return 1.0 + lam * a / (a.amax(dim=-1, keepdim=True) + eps)


@dataclass
class BlockSplit:
    omegas: Tensor
    p0: Tensor
    pair_freq: Tensor
    pair_init: Tensor
    x: Tensor
    p_true: Tensor
    weights: Tensor
    config: DataConfig
    block_index: int = -1
    sigma: str = "+"
    fingerprint: str = ""

    def __len__(self) -> int:
        return int(self.pair_freq.numel())

    @property
    def storage_device(self) -> torch.device:
        return self.x.device

    def batch(self, idx: Tensor, device: torch.device):
        idx = idx.to(self.x.device)
        return (self.x[idx].to(device, non_blocking=True),
                self.p_true[idx].to(device, non_blocking=True),
                self.weights[idx].to(device, non_blocking=True))

    def to(self, device) -> "BlockSplit":
        d = torch.device(device)
        return BlockSplit(self.omegas.to(d), self.p0.to(d), self.pair_freq.to(d), self.pair_init.to(d),
                          self.x.to(d), self.p_true.to(d), self.weights.to(d), self.config,
                          self.block_index, self.sigma, self.fingerprint)


def build_split(
    molecule: Molecule,
    block_index: int,
    cfg: DataConfig,
    sigma: str = "+",
    device: torch.device | str = "cpu",
    activity_lambda: float = 1.0,
    activity_eps: float = 1e-8,
    freq_chunk: int | None = None,
    storage_device: torch.device | str | None = None,
) -> BlockSplit:
    rng = np.random.default_rng(cfg.seed)
    block = molecule.blocks[block_index]
    m = block.n_states
    n_tau = molecule.window.n_tau
    omegas_np = sample_training_frequencies(molecule, block_index, cfg, rng, sigma)
    p0_np = random_mixed_populations(m, cfg.n_init, rng, cfg.alpha)
    kf, kl = pair_indices(cfg, rng)

    device = torch.device(device)
    store = torch.device(storage_device) if storage_device is not None else device
    omegas = torch.as_tensor(omegas_np, dtype=torch.float64, device=device)
    p0 = torch.as_tensor(p0_np, dtype=torch.float64, device=device)
    pair_freq = torch.as_tensor(kf, dtype=torch.int64, device=device)
    pair_init = torch.as_tensor(kl, dtype=torch.int64, device=device)
    emb = TorchEmbedding(molecule, block_index, sigma, device)

    if freq_chunk is None:
        freq_chunk = max(1, min(256, int(2.0e9 // (n_tau * (2 * m) ** 2 * 16))))
    n_pairs = pair_freq.numel()
    p_true = torch.empty((n_pairs, n_tau, 2 * m), dtype=torch.float32, device=store)
    order = torch.argsort(pair_freq)
    for lo in range(0, cfg.n_freq, freq_chunk):
        hi = min(lo + freq_chunk, cfg.n_freq)
        sel = order[(pair_freq[order] >= lo) & (pair_freq[order] < hi)]
        if sel.numel() == 0:
            continue
        t0 = transfer_columns(molecule, block_index, omegas[lo:hi], sigma, device=device,
                              out_dtype=torch.float64)
        p_true[sel.to(store)] = trajectories_from_columns(
            t0[pair_freq[sel] - lo], p0[pair_init[sel]]).to(dtype=torch.float32, device=store)
        del t0

    x = torch.empty((n_pairs, emb.n_channels, n_tau), dtype=torch.float32, device=store)
    pair_chunk = max(1, int(2.0e8 // max(emb.n_channels * n_tau, 1)))
    for lo in range(0, n_pairs, pair_chunk):
        hi = min(lo + pair_chunk, n_pairs)
        x[lo:hi] = emb.build(p0[pair_init[lo:hi]], omegas[pair_freq[lo:hi]], out_dtype=torch.float32).to(store)
    w = activity_weights(p_true, activity_lambda, activity_eps)
    return BlockSplit(omegas, p0, pair_freq.to(store), pair_init.to(store), x, p_true, w, cfg,
                      int(block_index), sigma, molecule.fingerprint())


def dataset_dir(molecule: Molecule, work: str | None = None) -> str:
    return os.path.join(work or work_dir(), "data", molecule.name, molecule.fingerprint())


def split_path(molecule: Molecule, block_index: int, sigma: str, cfg: DataConfig, work: str | None = None,
               activity_lambda: float = 1.0, activity_eps: float = 1e-8) -> str:
    s = "sp" if sigma == "+" else "sm"
    recipe = {"data": asdict(cfg), "s_emb": molecule.window.s_emb, "beta_emb": molecule.window.beta_emb,
              "delta_max": embedding_delta_max(molecule),
              "activity": [float(activity_lambda), float(activity_eps)]}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(dataset_dir(molecule, work), f"block{int(block_index)}_{s}_{key}.pt")


def save_split(path: str, split: BlockSplit) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu = split.to("cpu")
    torch.save({"omegas": cpu.omegas, "p0": cpu.p0, "pair_freq": cpu.pair_freq, "pair_init": cpu.pair_init,
                "x": cpu.x, "p_true": cpu.p_true, "weights": cpu.weights, "config": asdict(cpu.config),
                "block_index": cpu.block_index, "sigma": cpu.sigma, "fingerprint": cpu.fingerprint}, path)


def load_split(path: str, device: torch.device | str = "cpu") -> BlockSplit:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return BlockSplit(d["omegas"], d["p0"], d["pair_freq"], d["pair_init"], d["x"], d["p_true"], d["weights"],
                      DataConfig(**d["config"]), d["block_index"], d["sigma"], d["fingerprint"]).to(device)


def build_or_load_split(
    molecule: Molecule, block_index: int, cfg: DataConfig, sigma: str = "+", device="cpu",
    storage_device=None, work: str | None = None, cache: bool = True, **kw,
) -> BlockSplit:
    path = split_path(molecule, block_index, sigma, cfg, work,
                      kw.get("activity_lambda", 1.0), kw.get("activity_eps", 1e-8))
    store = storage_device if storage_device is not None else device
    if cache and os.path.exists(path):
        return load_split(path, store)
    split = build_split(molecule, block_index, cfg, sigma, device, storage_device=storage_device, **kw)
    if cache:
        save_split(path, split)
    return split
