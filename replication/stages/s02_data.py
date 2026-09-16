"""Stage 2 -- the exact ground truth the surrogate is trained against."""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import Config, MissingResult, load_molecule, pairs, read_result, result_path

NAME = "s02_data"
TITLE = "exact ground-truth training splits per (block, sigma)"
PAPER = "Sec. II.3"
REQUIRES = ("s01_physics",)
COST = "~1 h, GPU (6 pairs)"

# The paper's implied sampling (module docstring).  Overridable per run
# through cfg.overrides["s02_data"].
SAMPLING: dict = {"freq_sampling": "uniform", "alpha": 1.0, "pairing": "random"}

# Table 2: 100 validation initial states.  The validation split's frequency
# and pair counts are a tenth of the training ones, floored so that a small
# Config still gets a usable split.
VAL_N_INIT = 100
VAL_MIN_FREQ, VAL_MIN_PAIRS = 64, 256

# The checkpoint-selection split belongs to surrogate.train.train, which
# builds it from TrainConfig.sel_*.  It is reported here, so both stages
# must read the same settings: the defaults come from TrainConfig and the
# overrides from the *training* stage's slot in Config.overrides.
# s03_train refuses to run if the two disagree.
SEL_OVERRIDES_FROM = "s03_train"

# On CUDA the materialised split stays in host memory and batches stream to
# the GPU (needed for the 64-state H3O+ blocks; qlsgym README).
STORAGE_DEVICE_ON_CUDA = "cpu"


def sampling(cfg: Config) -> dict:
    """The sampling knobs of this run: SAMPLING plus any override."""
    out = dict(SAMPLING)
    out.update(cfg.overrides.get(NAME, {}))
    return out


def data_configs(cfg: Config):
    """(train, val, selection) DataConfig for one run."""
    from qlsgym.surrogate.dataset import DataConfig
    from qlsgym.surrogate.train import TrainConfig

    s = sampling(cfg)
    tc, over = TrainConfig(), cfg.overrides.get(SEL_OVERRIDES_FROM, {})
    sel_n_freq = int(over.get("sel_n_freq", tc.sel_n_freq))
    sel_n_init = int(over.get("sel_n_init", tc.sel_n_init))
    sel_seed = int(over.get("sel_seed", tc.sel_seed))
    shared = {k: v for k, v in s.items() if k != "pairing"}   # the rest are DataConfig fields
    train = DataConfig(n_freq=cfg.n_freq, n_init=cfg.n_init, n_pairs=cfg.n_pairs,
                       pairing=s["pairing"], seed=cfg.seed, **shared)
    val = DataConfig(n_freq=min(cfg.n_freq, max(VAL_MIN_FREQ, cfg.n_freq // 10)),
                     n_init=min(cfg.n_init, VAL_N_INIT),
                     n_pairs=min(cfg.n_pairs, max(VAL_MIN_PAIRS, cfg.n_pairs // 10)),
                     pairing="random", seed=cfg.seed + 1, **shared)
    selection = DataConfig(n_freq=sel_n_freq, n_init=sel_n_init, n_pairs=sel_n_freq * sel_n_init,
                           pairing="product", seed=sel_seed, alpha=train.alpha,
                           freq_sampling="mixture", resonant_frac=1.0, resonant_spread=1.0)
    return train, val, selection


def estimated_bytes(molecule, block: int, sigma: str, data_cfg) -> int:
    """Float32 size of a materialised split: x + p_true + weights."""
    from qlsgym.surrogate.embedding import block_shapes

    m, _q, in_channels = block_shapes(molecule, block, sigma)
    n_tau = molecule.window.n_tau
    n = data_cfg.n_pairs if data_cfg.n_pairs is not None else max(data_cfg.n_freq, data_cfg.n_init)
    return int(4 * n * (in_channels * n_tau + 2 * m * n_tau + 2 * m))


def _inspect(path: str) -> tuple:
    """(shapes, omegas, fingerprint) of a split file, read through torch.load(mmap=True) so an
    existing multi-GB split is not materialised just to be reported on.
    """
    import torch

    keys = ("omegas", "p0", "pair_freq", "pair_init", "x", "p_true", "weights")
    try:
        d = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        d = torch.load(path, map_location="cpu", weights_only=False)
    shapes = {k: list(d[k].shape) for k in keys}
    return shapes, np.asarray(d["omegas"].clone().numpy(), dtype=np.float64), str(d["fingerprint"])


def _split_record(molecule, block: int, sigma: str, data_cfg, cfg: Config, kind: str) -> dict:
    """Build or load one split and report it (never returns the tensors)."""
    from qlsgym.surrogate.dataset import build_or_load_split, split_path
    from qlsgym.surrogate.metrics import is_on_resonance

    path = split_path(molecule, block, sigma, data_cfg)
    existed = os.path.exists(path)
    t0 = time.time()
    if not existed:
        device = cfg.resolved_device()
        store = STORAGE_DEVICE_ON_CUDA if device == "cuda" else None
        split = build_or_load_split(molecule, block, data_cfg, sigma, device=device, storage_device=store)
        del split
    seconds = time.time() - t0
    shapes, omegas, fingerprint = _inspect(path)
    if fingerprint != molecule.fingerprint():
        raise RuntimeError(f"{path}: split fingerprint {fingerprint} != molecule "
                           f"{molecule.fingerprint()}; delete it and re-run {NAME}")
    on = is_on_resonance(molecule, omegas, block, sigma, 1.0)
    return {"kind": kind, "path": path, "key": data_cfg.key(), "config": vars(data_cfg).copy(),
            "built": not existed, "seconds": round(seconds, 2),
            "bytes": int(os.path.getsize(path)), "shapes": shapes,
            "n_frequencies": int(omegas.size), "n_samples": int(shapes["x"][0]),
            "on_resonance_fraction": float(on.mean()),
            "omega_over_2pi_khz": [float(omegas.min() / (2 * np.pi)), float(omegas.max() / (2 * np.pi))]}


# Stage protocol


def plan(cfg: Config) -> list:
    from qlsgym.surrogate.dataset import dataset_dir, split_path

    molecule = load_molecule(cfg)
    train, val, selection = data_configs(cfg)
    s = sampling(cfg)
    lines = [f"sampling: {s['freq_sampling']} frequencies, Dirichlet alpha = {s['alpha']}, "
             f"pairing {s['pairing']} (paper Sec. II.3 / Table 2; see the module docstring)",
             f"splits under {dataset_dir(molecule)} (keyed on fingerprint {molecule.fingerprint()})"]
    total = 0
    for block, sigma in pairs(cfg, molecule):
        for kind, dc in (("train", train), ("val", val)):
            path = split_path(molecule, block, sigma, dc)
            mb = estimated_bytes(molecule, block, sigma, dc) / 1e6
            total += mb
            state = "exists" if os.path.exists(path) else "BUILD"
            lines.append(f"{state:>6}  block {block} sigma{sigma} {kind:<5} "
                         f"n_freq={dc.n_freq} n_init={dc.n_init} n_pairs={dc.n_pairs} "
                         f"~{mb:.0f} MB  {os.path.basename(path)}")
    lines.append(f"selection split (built inside surrogate.train.train, not cached): "
                 f"n_freq={selection.n_freq} n_init={selection.n_init} pairing=product "
                 f"freq_sampling=mixture resonant_frac=1.0, key {selection.key()}")
    lines.append(f"total materialised size ~{total / 1e3:.1f} GB on {cfg.resolved_device()}")
    return lines


def run(cfg: Config) -> dict:
    molecule = load_molecule(cfg)
    phys = read_result("s01_physics", cfg, molecule=molecule)
    train, val, selection = data_configs(cfg)
    device = cfg.resolved_device()

    from qlsgym.surrogate.dataset import dataset_dir, split_path

    if "block_dims" not in phys:
        raise MissingResult(f"{result_path('s01_physics')} has no 'block_dims': it was written by "
                            f"an older s01_physics; run `python replication/run.py s01_physics` "
                            f"first")

    records = []
    t0 = time.time()
    for block, sigma in pairs(cfg, molecule):
        m_f = int(molecule.blocks[block].n_states)
        if m_f != int(phys["block_dims"][block]):
            raise RuntimeError(f"block {block} has M_f {m_f} but s01_physics measured "
                               f"{phys['block_dims'][block]}: re-run s01_physics")
        entry = {"block": int(block), "sigma": sigma, "m_f": m_f,
                 "estimated_bytes": estimated_bytes(molecule, block, sigma, train)}
        for kind, dc in (("train", train), ("val", val)):
            entry[kind] = _split_record(molecule, block, sigma, dc, cfg, kind)
        entry["selection"] = {
            "kind": "selection", "path": split_path(molecule, block, sigma, selection),
            "key": selection.key(), "config": vars(selection).copy(), "materialised": False,
            "note": "surrogate.train.train builds this split itself (uncached) to select "
                    "best_onres.pt; it is reported, not pre-built",
        }
        records.append(entry)

    built = [r[k] for r in records for k in ("train", "val") if r[k]["built"]]
    return {
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "s01_fingerprint": phys["fingerprint"],
        "device": device,
        "dataset_dir": dataset_dir(molecule),
        "sampling": sampling(cfg),
        "paper_note": "Sec. II.3: batched basis-state evolution; only the nu = 0 columns of the "
                      "population transfer matrix are stored, so initial states are free",
        "pairs": records,
        "totals": {"n_pairs": len(records),
                   "n_splits_built": len(built),
                   "n_splits_reused": 2 * len(records) - len(built),
                   "bytes": int(sum(r[k]["bytes"] for r in records for k in ("train", "val"))),
                   "build_seconds": round(sum(s["seconds"] for s in built), 1),
                   "wall_seconds": round(time.time() - t0, 1)},
    }
