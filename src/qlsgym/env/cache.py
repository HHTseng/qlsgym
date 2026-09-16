"""qlsgym.env.cache -- precomputed transfer matrices for an action library."""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..spec import Molecule, TauBatchedEngine
from .actions import ActionLibrary


@dataclass
class PrimitiveTable:
    states: np.ndarray        # (S,) sorted global indices of the union sector
    table: np.ndarray         # (2S, S) lumped transfer matrix at the primitive's tau
    tau_index: int            # row of the tau grid used
    blocks: tuple = ()        # block indices in the sector


@dataclass
class ActionTables:
    fingerprint: str
    library_tag: str
    blocks: list                    # per block: (n_grid, 2M, M)
    primitives: list                # list[PrimitiveTable]
    manifest: dict = field(default_factory=dict)

    @property
    def dtype(self):
        if self.blocks:
            return self.blocks[0].dtype
        return self.primitives[0].table.dtype if self.primitives else np.float32

    @property
    def n_grid(self) -> int:
        return int(self.blocks[0].shape[0]) if self.blocks else 0

    def nbytes(self) -> int:
        return sum(int(np.asarray(b).nbytes) for b in self.blocks) + sum(int(p.table.nbytes) for p in self.primitives)

    def check(self, molecule: Molecule, library: ActionLibrary) -> None:
        if self.fingerprint != molecule.fingerprint():
            raise RuntimeError(f"tables were built for fingerprint {self.fingerprint}, molecule has "
                               f"{molecule.fingerprint()} -- rebuild")
        if self.library_tag != library.tag():
            raise RuntimeError(f"tables were built for library tag {self.library_tag}, got {library.tag()}")
        if len(self.blocks) != molecule.system.n_blocks:
            raise RuntimeError("block count mismatch")
        for b, t in zip(molecule.blocks, self.blocks):
            m = b.n_states
            if tuple(t.shape) != (library.n_grid, 2 * m, m):
                raise RuntimeError(f"block {b.index}: table shape {t.shape} != {(library.n_grid, 2 * m, m)}")
        if len(self.primitives) != library.n_primitives:
            raise RuntimeError("primitive count mismatch")


# helpers


def work_dir() -> str:
    return os.environ.get("QLSGYM_WORK", os.path.join(os.path.expanduser("~"), "qlsgym_work"))


def tables_dir(molecule: Molecule, library: ActionLibrary, out_dir: str | None = None) -> str:
    """<out_dir or $QLSGYM_WORK/cache>/<name>/<fingerprint>/<library tag>."""
    root = out_dir if out_dir is not None else os.path.join(work_dir(), "cache")
    return os.path.join(root, molecule.name, molecule.fingerprint(), library.tag())


def primitive_sector(molecule: Molecule, k: int) -> np.ndarray:
    """Sorted union of the states of the blocks a primitive couples."""
    p = molecule.primitives[k]
    if not p.blocks:
        raise ValueError(f"primitive {k} ({p.label}) declares no blocks")
    return np.unique(np.concatenate([molecule.blocks[b].states for b in p.blocks]))


def _engine_rows(engine: TauBatchedEngine, needed: np.ndarray) -> np.ndarray:
    """Map global tau indices to rows of the engine's branches_all_tau output."""
    eng = np.asarray(engine.tau_indices, dtype=np.int64)
    pos = {int(t): i for i, t in enumerate(eng)}
    try:
        return np.asarray([pos[int(t)] for t in needed], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"engine.tau_indices lacks tau index {e}; build the engine with the "
                         "library's durations") from None


def _primitive_table_from_engine(molecule: Molecule, library: ActionLibrary, engine: TauBatchedEngine,
                                 k: int, dtype, leak_tol: float = 1e-6) -> PrimitiveTable:
    p = molecule.primitives[k]
    states = primitive_sector(molecule, k)
    n = molecule.n_states
    S = states.size
    ti = library.primitive_tau_index(k)
    row = int(_engine_rows(engine, np.array([ti]))[0])
    outside = np.setdiff1d(np.arange(n), states)
    T = np.zeros((2 * S, S), dtype=np.float64)
    for j in range(S):
        e = np.zeros(n)
        e[states[j]] = 1.0
        p0, p1 = engine.branches_all_tau(e, float(p.omega), p.sigma)
        p0, p1 = np.asarray(p0)[row], np.asarray(p1)[row]
        leak = np.abs(p0[outside]).sum() + np.abs(p1[outside]).sum()
        if leak > leak_tol:
            raise RuntimeError(f"primitive {k} ({p.label}) moves {leak:.2e} of the population outside "
                               f"its declared blocks {p.blocks}; fix Primitive.blocks")
        T[:S, j] = p0[states]
        T[S:, j] = p1[states]
    return PrimitiveTable(states, T.astype(dtype), ti, tuple(int(b) for b in p.blocks))


# generic builder (any engine)


def tables_from_engine(molecule: Molecule, library: ActionLibrary, engine: TauBatchedEngine,
                       dtype=np.float32, progress: bool = False) -> ActionTables:
    """Build tables by probing engine with unit vectors (module docstring)."""
    n = molecule.n_states
    blocks = molecule.blocks
    n_grid = library.n_grid
    out = [np.zeros((n_grid, 2 * b.n_states, b.n_states), dtype=dtype) for b in blocks]
    m_max = max((b.n_states for b in blocks), default=0)
    drives = library.drives()
    for d, (sigma, omega, a_idx, t_idx) in enumerate(drives):
        rows = _engine_rows(engine, t_idx)
        for j in range(m_max):
            e = np.zeros(n)
            for b in blocks:
                if j < b.n_states:
                    e[b.states[j]] = 1.0
            p0, p1 = engine.branches_all_tau(e, float(omega), sigma)
            p0, p1 = np.asarray(p0), np.asarray(p1)
            for b in blocks:
                if j >= b.n_states:
                    continue
                m = b.n_states
                out[b.index][a_idx, :m, j] = p0[rows][:, b.states]
                out[b.index][a_idx, m:, j] = p1[rows][:, b.states]
        if progress and (d % 50 == 0 or d == len(drives) - 1):
            print(f"[qlsgym.cache] drive {d + 1}/{len(drives)}", flush=True)
    prims = [_primitive_table_from_engine(molecule, library, engine, k, dtype)
             for k in range(library.n_primitives)]
    return ActionTables(molecule.fingerprint(), library.tag(), out, prims,
                        _manifest(molecule, library, out, prims, "engine:" + type(engine).__name__))


# production builder (physics package)


def tables_from_physics(molecule: Molecule, library: ActionLibrary, device: str = "cpu",
                        dtype=np.float32, progress: bool = False, chunk: int = 256) -> ActionTables:
    """Grid tables via physics.torch_ops.transfer_columns (batched over omega, chunked so a 35
    280-action grid fits in memory) and primitive tables via the exact engine.
    """
    import torch

    from ..physics.engines import ExactEngine
    from ..physics.torch_ops import transfer_columns

    n_grid = library.n_grid
    blocks = molecule.blocks
    out = [np.zeros((n_grid, 2 * b.n_states, b.n_states), dtype=dtype) for b in blocks]
    tdt = torch.float64 if np.dtype(dtype) == np.float64 else torch.float32
    # group actions by sigma, then unique omega; every action of one drive
    # shares the propagator and differs only in the tau row gathered.
    for sigma in sorted(set(library.sigmas.tolist())):
        sel = np.where(library.sigmas == sigma)[0]
        om_r = np.round(library.omegas[sel], 9)
        uniq, inv = np.unique(om_r, return_inverse=True)
        need_tau = np.unique(library.tau_indices[sel])
        pos = {int(t): i for i, t in enumerate(need_tau)}
        t_rows = np.asarray([pos[int(t)] for t in library.tau_indices[sel]], dtype=np.int64)
        for lo in range(0, uniq.size, chunk):
            omegas = uniq[lo:lo + chunk]
            in_chunk = (inv >= lo) & (inv < lo + chunk)
            for b in blocks:
                cols = transfer_columns(molecule, b.index, omegas, sigma, tau_indices=need_tau,
                                        device=device, out_dtype=tdt)        # (n_w, n_t, 2M, M)
                cols = cols.detach().cpu().numpy()
                out[b.index][sel[in_chunk]] = cols[inv[in_chunk] - lo, t_rows[in_chunk]]
            if progress:
                print(f"[qlsgym.cache] sigma{sigma} omegas {min(lo + chunk, uniq.size)}/{uniq.size}", flush=True)
    prims = []
    if library.n_primitives:
        eng = ExactEngine(molecule, tau_indices=np.unique([library.primitive_tau_index(k)
                                                           for k in range(library.n_primitives)]))
        prims = [_primitive_table_from_engine(molecule, library, eng, k, dtype)
                 for k in range(library.n_primitives)]
    return ActionTables(molecule.fingerprint(), library.tag(), out, prims,
                        _manifest(molecule, library, out, prims, f"physics:transfer_columns[{device}]"))


def _manifest(molecule, library, blocks, prims, builder: str) -> dict:
    return {
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "library_tag": library.tag(),
        "library": library.describe(),
        "builder": builder,
        "dtype": str(np.dtype(blocks[0].dtype) if blocks else prims[0].table.dtype),
        "block_shapes": [list(b.shape) for b in blocks],
        "primitives": [{"index": k, "label": molecule.primitives[k].label, "sigma": molecule.primitives[k].sigma,
                        "omega": float(molecule.primitives[k].omega), "tau_index": int(p.tau_index),
                        "blocks": list(p.blocks), "states": p.states.tolist(), "shape": list(p.table.shape)}
                       for k, p in enumerate(prims)],
        "tau_grid": molecule.tau_grid().tolist(),
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "nbytes": int(sum(b.nbytes for b in blocks) + sum(p.table.nbytes for p in prims)),
    }


# disk


def save_action_tables(tables: ActionTables, d: str) -> str:
    os.makedirs(d, exist_ok=True)
    for i, b in enumerate(tables.blocks):
        _atomic_save(os.path.join(d, f"block_{i:02d}.npy"), np.ascontiguousarray(b))
    for k, p in enumerate(tables.primitives):
        _atomic_save(os.path.join(d, f"prim_{k:02d}.npy"), np.ascontiguousarray(p.table))
    tmp = os.path.join(d, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(tables.manifest, fh, indent=1)
    os.replace(tmp, os.path.join(d, "manifest.json"))
    return d


def _atomic_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def build_action_tables(
    molecule: Molecule,
    library: ActionLibrary,
    device: str = "cpu",
    out_dir: str | None = None,
    progress: bool = False,
    engine: TauBatchedEngine | None = None,
    dtype=np.float32,
    overwrite: bool = False,
    write: bool = True,
) -> ActionTables:
    """Build (or reuse) the tables of library and write them under tables_dir."""
    d = tables_dir(molecule, library, out_dir)
    want = "physics:" if engine is None else f"engine:{type(engine).__name__}"
    if write and not overwrite and os.path.exists(os.path.join(d, "manifest.json")):
        return load_action_tables(molecule, library, out_dir=out_dir, builder_prefix=want)
    if engine is None:
        tables = tables_from_physics(molecule, library, device=device, dtype=dtype, progress=progress)
    else:
        tables = tables_from_engine(molecule, library, engine, dtype=dtype, progress=progress)
    if write:
        save_action_tables(tables, d)
        tables.manifest["path"] = d
    return tables


def load_action_tables(molecule: Molecule, library: ActionLibrary, out_dir: str | None = None,
                       mmap: bool = True, builder_prefix: str | None = None) -> ActionTables:
    """Memory-map the tables of library; fingerprint/tag/builder mismatch is an error."""
    d = tables_dir(molecule, library, out_dir)
    mpath = os.path.join(d, "manifest.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"no action tables at {d}; run build_action_tables()")
    with open(mpath) as fh:
        man = json.load(fh)
    if builder_prefix is not None and not str(man.get("builder", "")).startswith(builder_prefix):
        raise RuntimeError(
            f"cache at {d} was built by {man.get('builder')!r}, this caller needs tables built by "
            f"{builder_prefix!r}* -- rebuild with overwrite=True (tables probed out "
            f"of a surrogate must never stand in for exact dynamics)")
    if man.get("fingerprint") != molecule.fingerprint():
        raise RuntimeError(f"cache at {d} has fingerprint {man.get('fingerprint')}, molecule has "
                           f"{molecule.fingerprint()} -- rebuild")
    if man.get("library_tag") != library.tag():
        raise RuntimeError(f"cache at {d} has library tag {man.get('library_tag')}, library is {library.tag()}")
    mode = "r" if mmap else None
    blocks = [np.load(os.path.join(d, f"block_{i:02d}.npy"), mmap_mode=mode)
              for i in range(molecule.system.n_blocks)]
    prims = [PrimitiveTable(np.asarray(pm["states"], dtype=np.int64),
                            np.load(os.path.join(d, f"prim_{k:02d}.npy"), mmap_mode=mode),
                            int(pm["tau_index"]), tuple(pm["blocks"]))
             for k, pm in enumerate(man.get("primitives", []))]
    man["path"] = d
    t = ActionTables(man["fingerprint"], man["library_tag"], blocks, prims, man)
    t.check(molecule, library)
    return t
