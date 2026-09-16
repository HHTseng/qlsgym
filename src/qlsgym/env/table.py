"""qlsgym.env.table -- the old gym's dense .npz tables as a PurificationEnv."""

from __future__ import annotations

import json
import os

import numpy as np

from ..spec import TWO_PI, Block, Molecule, System, Task, Trap, Window
from .actions import ActionLibrary
from .cache import ActionTables
from .env import EnvConfig, PurificationEnv


def legacy_tables(npz_path: str, json_path: str | None = None, p_target: float = 0.99,
                  max_pulses: int = 1000):
    """(molecule, library, tables, p_init, meta) for an old-gym .npz."""
    z = np.load(npz_path)
    A = np.asarray(z["A"], dtype=np.float64)
    if A.ndim != 4 or A.shape[1] != 2 or A.shape[2] != A.shape[3]:
        raise ValueError(f"expected A[n_actions, 2, n, n], got {A.shape}")
    n_act, _, n, _ = A.shape
    durations = np.asarray(z["durations"], dtype=np.float64) if "durations" in z.files else np.arange(n_act, dtype=float)
    if json_path is None:
        json_path = os.path.splitext(npz_path)[0] + ".json"
    meta = json.load(open(json_path)) if os.path.exists(json_path) else {}
    name = meta.get("name", os.path.splitext(os.path.basename(npz_path))[0])
    labels = tuple(meta.get("labels", [f"s{i}" for i in range(n)]))
    energies = np.asarray(meta.get("energies_hz", np.zeros(n)), dtype=np.float64) * TWO_PI * 1e-3  # Hz -> rad/ms
    prior = meta.get("default_prior", {})
    p_init = np.asarray(prior.get("p", np.full(n, 1.0 / n)), dtype=np.float64)
    T_k = float(prior.get("temperature_K", 300.0))

    states = np.arange(n)
    block = Block(0, states, ("legacy", name), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                  np.zeros(0, dtype=complex))
    system = System(labels, energies, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=complex), (block,), np.zeros(n, dtype=np.int64))
    # a dummy window: omega = action index in [0, n_act - 1], tau grid = one slot per action
    window = Window(omega_min=0.0, omega_max=float(max(n_act - 1, 1)), tau_max_ms=float(n_act), n_tau=n_act,
                    rwa_cutoff=0.0, omega_min_coupling=0.0)
    mol = Molecule(f"legacy:{name}", system, Trap(nu_f=0.0, eta=0.0, n_nu=2), window,
                   Task(temperature_k=T_k, p_target=p_target, max_pulses=max_pulses), (),
                   {"source": os.path.abspath(npz_path), "note": meta.get("molecule", ""),
                    "durations_s": durations.tolist()})
    lib = ActionLibrary.from_actions(mol, [("+", float(a), int(a)) for a in range(n_act)],
                                     include_primitives=False, kind="legacy")
    T = np.concatenate([A[:, 0], A[:, 1]], axis=1)          # (n_act, 2n, n)
    tables = ActionTables(mol.fingerprint(), lib.tag(), [T], [],
                          {"builder": "legacy_npz", "source": npz_path, "dtype": "float64"})
    return mol, lib, tables, p_init, meta


def legacy_table_env(npz_path: str, json_path: str | None = None, p_target: float = 0.99,
                     max_pulses: int = 1000, rho: float = 0.0, device: str = "cpu", batch: int = 1,
                     penalty_mode: str = "indicator") -> PurificationEnv:
    """A PurificationEnv on an old-gym table."""
    mol, lib, tables, p_init, _ = legacy_tables(npz_path, json_path, p_target, max_pulses)
    cfg = EnvConfig(p_target=p_target, max_pulses=max_pulses, rho=rho, penalty_mode=penalty_mode)
    return PurificationEnv(mol, lib, tables, cfg, device=device, batch=batch, p_init=p_init)
