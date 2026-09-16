"""qlsgym.env.actions -- the discrete action space of the belief MDP."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..spec import SIGMAS, TWO_PI, Action, Molecule


# Product control grids (paper Eqs. 23-24, Appendix C.1)


@dataclass(frozen=True)
class ControlGrid:
    """A product grid sigmas x omegas x tau_indices of in-window pulses."""

    omegas: np.ndarray
    tau_indices: np.ndarray
    sigmas: tuple = SIGMAS

    def __post_init__(self):
        object.__setattr__(self, "omegas", np.ascontiguousarray(self.omegas, dtype=np.float64))
        object.__setattr__(self, "tau_indices", np.ascontiguousarray(self.tau_indices, dtype=np.int64))
        object.__setattr__(self, "sigmas", tuple(str(s) for s in self.sigmas))
        if self.omegas.ndim != 1 or self.tau_indices.ndim != 1:
            raise ValueError("omegas and tau_indices must be 1-D")

    @property
    def n_freq(self) -> int:
        return int(self.omegas.size)

    @property
    def n_tau_slots(self) -> int:
        return int(self.tau_indices.size)

    @property
    def n_actions(self) -> int:
        return len(self.sigmas) * self.n_freq * self.n_tau_slots

    # constructors

    @staticmethod
    def _tau_defaults(molecule: Molecule, tau_lo, tau_hi, frac_lo: float):
        n_tau = molecule.window.n_tau
        if tau_lo is None:
            tau_lo = int(round(frac_lo * n_tau))
        if tau_hi is None:
            tau_hi = n_tau
        if not (1 <= tau_lo <= tau_hi <= n_tau):
            raise ValueError(f"need 1 <= tau_lo <= tau_hi <= n_tau={n_tau}, got {tau_lo}, {tau_hi}")
        return int(tau_lo), int(tau_hi)

    @classmethod
    def uniform(
        cls,
        molecule: Molecule,
        d_omega_khz: float,
        tau_lo: int | None = None,
        tau_hi: int | None = None,
        tau_step: int = 1,
        sigmas: tuple = SIGMAS,
        mask_carrier_bands: bool = True,
    ) -> "ControlGrid":
        """Eq. 23 grid: frequencies every d_omega_khz across the window, durations every
        tau_step-th step. Points inside a carrier band are dropped.
        """
        w = molecule.window
        lo_khz, hi_khz = w.omega_min / TWO_PI, w.omega_max / TWO_PI
        n = int(np.floor((hi_khz - lo_khz) / d_omega_khz + 1e-9)) + 1
        omegas = TWO_PI * (lo_khz + d_omega_khz * np.arange(n))
        # Round-tripping the bounds through kHz can put the first or last point
        # one ULP outside the window, which ActionLibrary then refuses; that
        # happens whenever the window width is an exact multiple of d_omega_khz.
        np.clip(omegas, w.omega_min, w.omega_max, out=omegas)
        if mask_carrier_bands:
            omegas = omegas[mask_carriers(molecule, omegas)]
        tau_lo, tau_hi = cls._tau_defaults(molecule, tau_lo, tau_hi, 0.5)
        taus = np.arange(tau_lo, tau_hi + 1, tau_step) - 1
        return cls(omegas, taus, tuple(sigmas))

    @classmethod
    def rl_discrete(
        cls,
        molecule: Molecule,
        n_freq: int = 1764,
        n_tau_slots: int = 10,
        seed: int = 0,
        sigmas: tuple = SIGMAS,
        tau_lo: int | None = None,
        tau_hi: int | None = None,
        mask_carrier_bands: bool = True,
    ) -> "ControlGrid":
        """Appendix C.1: n_freq uniform frequencies and n_tau_slots equally spaced durations."""
        w = molecule.window
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(w.omega_min, w.omega_max, n_freq)
        if mask_carrier_bands and w.carrier_bands:
            bad = ~mask_carriers(molecule, omegas)
            while bad.any():
                omegas[bad] = rng.uniform(w.omega_min, w.omega_max, int(bad.sum()))
                bad = ~mask_carriers(molecule, omegas)
        omegas = np.sort(omegas)
        tau_lo, tau_hi = cls._tau_defaults(molecule, tau_lo, tau_hi, 0.75)
        step = max(1, (tau_hi - tau_lo) // n_tau_slots)
        taus = (tau_lo - 1) + step * np.arange(n_tau_slots)
        if taus[-1] >= w.n_tau:
            raise ValueError("n_tau_slots too large for the tau range")
        return cls(omegas, taus, tuple(sigmas))


def mask_carriers(molecule: Molecule, omegas: np.ndarray) -> np.ndarray:
    """True where omegas is clear of every window.carrier_bands band."""
    ok = np.ones(np.shape(omegas), dtype=bool)
    for lo, hi in molecule.window.carrier_bands:
        ok &= ~((omegas >= lo) & (omegas <= hi))
    return ok


# Resonances (the physics-informed subset needs them; kept local so the action
# library does not depend on the physics package being importable)


def resonant_frequencies(molecule: Molecule, block_index: int, sigma: str = "+") -> np.ndarray:
    """Sideband resonance of every coupling of a block (paper Eq. 3): omega+ = nu_f + (E_f - E_i),
    omega- = nu_f - (E_f - E_i).
    """
    b = molecule.blocks[block_index]
    e = molecule.system.energies
    d = e[b.states[b.f_local]] - e[b.states[b.i_local]]
    if sigma == "+":
        return molecule.trap.nu_f + d
    if sigma == "-":
        return molecule.trap.nu_f - d
    raise ValueError(f"sigma must be '+' or '-', got {sigma!r}")


def pi_time_ms(molecule: Molecule, rabi: float) -> float:
    """Blue-sideband pi-time pi / (eta e^{-eta^2/2} |Omega|) in ms."""
    eta = molecule.trap.eta
    return float(np.pi / (eta * np.exp(-eta ** 2 / 2.0) * abs(rabi)))


def snap_tau(molecule: Molecule, tau_ms: float) -> int:
    """Index of the tau-grid point nearest tau_ms."""
    return int(np.argmin(np.abs(molecule.tau_grid() - tau_ms)))


# Action library


class ActionLibrary:
    """The integer <-> pulse bijection for one molecule."""

    def __init__(
        self,
        molecule: Molecule,
        grid: ControlGrid | None = None,
        *,
        include_primitives: bool = True,
        _actions: tuple | None = None,
        _kind: str | None = None,
        _meta: dict | None = None,
    ):
        self.molecule = molecule
        self.include_primitives = bool(include_primitives)
        if _actions is None:
            grid = grid if grid is not None else ControlGrid.rl_discrete(molecule)
            sig, om, ti = [], [], []
            for s in grid.sigmas:
                for w in grid.omegas:
                    for t in grid.tau_indices:
                        sig.append(s); om.append(float(w)); ti.append(int(t))
            self.grid = grid
            self.kind = _kind or "grid"
        else:
            sig, om, ti = _actions
            self.grid = grid
            self.kind = _kind or "list"
        self.sigmas = np.asarray(sig, dtype="<U1")
        self.omegas = np.asarray(om, dtype=np.float64)
        self.tau_indices = np.asarray(ti, dtype=np.int64)
        self.meta = dict(_meta or {})
        n_tau = molecule.window.n_tau
        if self.tau_indices.size and (self.tau_indices.min() < 0 or self.tau_indices.max() >= n_tau):
            raise ValueError("tau_indices outside the molecule's tau grid")
        for w in self.omegas:
            if not molecule.window.contains(float(w)):
                raise ValueError(f"grid action omega={w} is outside the window; off-window "
                                 "pulses must be molecule.primitives")
        self._lookup = {}
        for a, (s, w, t) in enumerate(zip(self.sigmas, self.omegas, self.tau_indices)):
            key = (str(s), round(float(w), 9), int(t))
            if key in self._lookup:
                raise ValueError(f"duplicate action {key}")
            self._lookup[key] = a
        self._prim_lookup = {}
        if self.include_primitives:
            for k, p in enumerate(molecule.primitives):
                self._prim_lookup.setdefault((p.sigma, round(float(p.omega), 9)), k)

    # constructors

    @classmethod
    def from_grid(cls, molecule: Molecule, grid: ControlGrid, include_primitives: bool = True):
        return cls(molecule, grid, include_primitives=include_primitives)

    @classmethod
    def from_actions(cls, molecule: Molecule, actions, include_primitives: bool = True,
                     kind: str = "list", meta: dict | None = None):
        """Explicit list of (sigma, omega, tau_index) triples, in the given order."""
        sig = [str(a[0]) for a in actions]
        om = [float(a[1]) for a in actions]
        ti = [int(a[2]) for a in actions]
        return cls(molecule, None, include_primitives=include_primitives,
                   _actions=(sig, om, ti), _kind=kind, _meta=meta)

    @classmethod
    def physics_subset(
        cls,
        molecule: Molecule,
        include_primitives: bool = True,
        omega_min_coupling: float | None = None,
        clip_tau: bool = False,
    ) -> "ActionLibrary":
        """Port of scripts/baseline_protocol.py::build_actions (ThF+)."""
        try:  # prefer the physics package's embedding rule when it is there
            from ..physics.spectrum import embedding_transitions as _emb
        except Exception:  # pragma: no cover - exercised only before agent A lands
            _emb = None
        w = molecule.window
        omin = w.omega_min_coupling if omega_min_coupling is None else omega_min_coupling
        taus = molecule.tau_grid()
        e = molecule.system.energies
        triples, src, tgt, cpl, blk = [], [], [], [], []
        seen = set()
        for b in molecule.blocks:
            w_res = resonant_frequencies(molecule, b.index, "+")
            if _emb is not None:
                keep = np.asarray(_emb(molecule, b.index, "+"), dtype=np.int64)
            else:
                keep = np.where((np.abs(b.omega) >= omin))[0]
            for kk in keep:
                ww = float(w_res[kk])
                om = float(abs(b.omega[kk]))
                if om < omin or not w.contains(ww):
                    continue
                t_pi = pi_time_ms(molecule, om)
                if t_pi > taus[-1] and not clip_tau:
                    continue
                ti = int(np.argmin(np.abs(taus - min(t_pi, taus[-1]))))
                gi = int(b.states[b.i_local[kk]])
                gf = int(b.states[b.f_local[kk]])
                for sg, wv, s_, t_ in (("+", ww, gi, gf), ("-", molecule.sideband_mirror(ww), gf, gi)):
                    if not w.contains(wv):
                        continue
                    key = (sg, round(wv, 9), ti)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append((sg, wv, ti))
                    src.append(s_); tgt.append(t_); cpl.append(om); blk.append(int(b.index))
        del e
        meta = {"sources": np.asarray(src, dtype=np.int64), "targets": np.asarray(tgt, dtype=np.int64),
                "couplings": np.asarray(cpl, dtype=np.float64), "blocks": np.asarray(blk, dtype=np.int64)}
        return cls.from_actions(molecule, triples, include_primitives, kind="physics", meta=meta)

    # sizes

    @property
    def n_grid(self) -> int:
        return int(self.omegas.size)

    @property
    def n_primitives(self) -> int:
        return len(self.molecule.primitives) if self.include_primitives else 0

    @property
    def n_actions(self) -> int:
        return self.n_grid + self.n_primitives

    def __len__(self) -> int:
        return self.n_actions

    # the bijection

    def is_primitive(self, a: int) -> bool:
        a = int(a)
        if not 0 <= a < self.n_actions:
            raise IndexError(f"action {a} out of range [0, {self.n_actions})")
        return a >= self.n_grid

    def primitive_tau_index(self, k: int) -> int:
        """Row of the tau grid nearest the primitive's own duration."""
        return snap_tau(self.molecule, self.molecule.primitives[k].tau_ms)

    def decode(self, a: int) -> Action:
        a = int(a)
        if not 0 <= a < self.n_actions:
            raise IndexError(f"action {a} out of range [0, {self.n_actions})")
        if a < self.n_grid:
            return Action(str(self.sigmas[a]), float(self.omegas[a]), int(self.tau_indices[a]), -1)
        k = a - self.n_grid
        p = self.molecule.primitives[k]
        return Action(p.sigma, float(p.omega), self.primitive_tau_index(k), k)

    def encode(self, action: Action) -> int:
        if action.is_primitive:
            if not self.include_primitives:
                raise KeyError("this library carries no primitives")
            return self.n_grid + int(action.primitive)
        key = (str(action.sigma), round(float(action.omega), 9), int(action.tau_index))
        try:
            return self._lookup[key]
        except KeyError:
            # an off-window (sigma, omega) matching a primitive is that primitive
            pk = self._prim_lookup.get((str(action.sigma), round(float(action.omega), 9)))
            if pk is not None:
                return self.n_grid + pk
            raise KeyError(f"action {action} is not in this library") from None

    def __contains__(self, action: Action) -> bool:
        try:
            self.encode(action)
            return True
        except KeyError:
            return False

    @property
    def actions(self) -> Iterator[Action]:
        for a in range(self.n_actions):
            yield self.decode(a)

    def indices_of_tau(self, tau_index: int) -> np.ndarray:
        """Grid action indices at one tau slot (e.g. to restrict a policy)."""
        return np.where(self.tau_indices == int(tau_index))[0]

    # grouping by drive (what an engine call answers)

    def drives(self) -> list:
        """Unique grid (sigma, omega) pairs in first-occurrence order, each with the action indices
        and tau indices it serves: [(sigma, omega, action_idx (k,), tau_idx (k,)), ...].
        """
        order, groups = [], {}
        for a, (s, w) in enumerate(zip(self.sigmas, self.omegas)):
            key = (str(s), round(float(w), 9))
            if key not in groups:
                groups[key] = []
                order.append((key, float(w)))
            groups[key].append(a)
        out = []
        for key, w in order:
            idx = np.asarray(groups[key], dtype=np.int64)
            out.append((key[0], w, idx, self.tau_indices[idx]))
        return out

    # identity

    def tag(self) -> str:
        """Content hash of the action list (and of the molecule's physics), used as the cache
        directory name.
        """
        h = hashlib.sha256()
        h.update(self.molecule.fingerprint().encode())
        h.update("".join(self.sigmas.tolist()).encode())
        h.update(np.ascontiguousarray(np.round(self.omegas, 9)).tobytes())
        h.update(np.ascontiguousarray(self.tau_indices).tobytes())
        h.update(str(self.n_primitives).encode())
        return h.hexdigest()[:16]

    def describe(self) -> dict:
        """JSON-able summary for cache manifests."""
        d = {"kind": self.kind, "n_grid": self.n_grid, "n_primitives": self.n_primitives,
             "n_actions": self.n_actions, "tag": self.tag(),
             "sigmas": sorted(set(self.sigmas.tolist())),
             "n_unique_omegas": int(np.unique(np.round(self.omegas, 9)).size),
             "tau_indices": sorted(set(self.tau_indices.tolist()))}
        if self.grid is not None:
            d["grid"] = {"n_freq": self.grid.n_freq, "n_tau_slots": self.grid.n_tau_slots,
                         "sigmas": list(self.grid.sigmas),
                         "omega_min": float(self.grid.omegas.min()) if self.grid.n_freq else None,
                         "omega_max": float(self.grid.omegas.max()) if self.grid.n_freq else None}
        return d

    def __repr__(self) -> str:
        return (f"ActionLibrary({self.molecule.name!r}, kind={self.kind!r}, n_grid={self.n_grid}, "
                f"n_primitives={self.n_primitives}, tag={self.tag()})")
