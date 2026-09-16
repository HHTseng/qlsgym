"""qlsgym.policies.baselines -- the reference policies."""

from __future__ import annotations

import collections
from typing import Protocol, runtime_checkable

import numpy as np

from ..env.actions import ActionLibrary
from ..env.cache import ActionTables
from ..spec import Action, Molecule, TauBatchedEngine, branches_batch_fallback
from .score import ScoreConfig, score_batch


@runtime_checkable
class Policy(Protocol):
    def act(self, belief: np.ndarray, t: int, rng: np.random.Generator):
        """Action index (int), a Action, or None."""
        ...


# trivial baselines


class SweepingPolicy:
    """Deterministic cycle through the library: action t mod n_actions."""

    stateful = False

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def act(self, belief, t, rng):
        return int(t) % self.n_actions

    def act_batch(self, beliefs, t, rng):
        return np.full(beliefs.shape[0], int(t) % self.n_actions, dtype=np.int64)


class RandomPolicy:
    """Uniform over the library (draws from the rollout's generator)."""

    stateful = False

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)

    def act(self, belief, t, rng):
        return int(rng.integers(self.n_actions))

    def act_batch(self, beliefs, t, rng):
        return rng.integers(self.n_actions, size=beliefs.shape[0])


def yield_table(library: ActionLibrary, tables: ActionTables) -> np.ndarray:
    """q[a, s] = population of state s that action a moves into nu >= 1 (column sums of the nu >= 1
    half of each table).
    """
    n = library.molecule.n_states
    q = np.zeros((library.n_actions, n))
    for b, T in zip(library.molecule.blocks, tables.blocks):
        m = b.n_states
        q[: library.n_grid, b.states] = np.asarray(T)[:, m:, :].sum(1)
    for k, p in enumerate(tables.primitives):
        S = p.states.size
        q[library.n_grid + k, p.states] = np.asarray(p.table)[S:, :].sum(0)
    return q


class DescendingPopulationPolicy:
    """Address the most populated state with the action that best empties it into nu >= 1 (old gym
    descending_population).
    """

    stateful = False

    def __init__(self, library: ActionLibrary, tables: ActionTables):
        q = yield_table(library, tables)
        self.best_action = np.argmax(q, axis=0).astype(np.int64)      # (n_states,)
        self.selectivity = q[self.best_action, np.arange(q.shape[1])]

    def act(self, belief, t, rng):
        return int(self.best_action[int(np.argmax(belief))])

    def act_batch(self, beliefs, t, rng):
        return self.best_action[np.argmax(beliefs, axis=1)]


# ThF+ physics-designed elimination rule


def _expected_purity_best_tau(engine, p, omega, sigma):
    a, c = engine.branches_all_tau(p, float(omega), sigma)
    val = np.asarray(a).max(1) + np.asarray(c).max(1)     # Pi_nu cancels against 1/Pi_nu
    k = int(np.argmax(val))
    return int(np.asarray(engine.tau_indices)[k]), float(val[k])


class PhysicsEliminationPolicy:
    """Port of baseline_protocol.py::choose (module docstring)."""

    stateful = True

    def __init__(self, molecule: Molecule, library: ActionLibrary, eps: float = 1e-4, n_cand: int = 3,
                 tau_rule: str = "pi", engine: TauBatchedEngine | None = None):
        if molecule.name != "thf":
            raise ValueError("PhysicsEliminationPolicy is ThF+-specific: it reads (J, F, parity, m_F) "
                             f"from molecule.system.levels; got molecule {molecule.name!r}")
        if "sources" not in library.meta:
            raise ValueError("PhysicsEliminationPolicy needs ActionLibrary.physics_subset(molecule) "
                             "(per-action source/target/coupling metadata)")
        if tau_rule not in ("pi", "greedy"):
            raise ValueError(f"unknown tau_rule {tau_rule!r}")
        if tau_rule == "greedy" and engine is None:
            raise ValueError("tau_rule='greedy' needs an engine")
        self.molecule, self.library = molecule, library
        self.eps, self.n_cand, self.tau_rule, self.engine = float(eps), int(n_cand), tau_rule, engine
        lv = molecule.system.levels
        self.ladder = [self.ladder_of(l) for l in lv]
        lad_ids = {L: i for i, L in enumerate(sorted(set(self.ladder)))}
        self._lad_index = np.asarray([lad_ids[L] for L in self.ladder])
        self._lad_list = sorted(set(self.ladder))
        # per-ladder candidate actions in library (= script) order
        self.actions: dict = collections.defaultdict(list)
        src, tgt, cpl = library.meta["sources"], library.meta["targets"], library.meta["couplings"]
        for a in range(library.n_grid):
            sg = str(library.sigmas[a])
            gi = int(src[a]) if sg == "+" else int(tgt[a])      # the sigma+ source keys the ladder
            self.actions[self.ladder[gi]].append(
                (a, sg, float(library.omegas[a]), int(library.tau_indices[a]), int(src[a]), int(tgt[a]), float(cpl[a])))
        self.actions = dict(self.actions)
        self.prims = [(k, int(q.source)) for k, q in enumerate(molecule.primitives)] if library.include_primitives else []
        self.last_sigma: dict = {}
        self.why = ""

    @staticmethod
    def ladder_of(level) -> tuple:
        """(J, F, parity) of a level tuple (J, F, parity, m_F, ...)."""
        return (level[0], level[1], level[2])

    def reset(self):
        self.last_sigma = {}
        self.why = ""

    def act(self, p, t, rng):
        p = np.asarray(p, dtype=np.float64)
        eps = self.eps
        star = int(np.argmax(p))
        home = self.ladder[star]
        mass_arr = np.zeros(len(self._lad_list))
        np.add.at(mass_arr, self._lad_index, p)
        mass = {L: float(mass_arr[i]) for i, L in enumerate(self._lad_list)}

        # 1. strip one of the heaviest ladders that is not the one we are keeping
        others = sorted(((m, L) for L, m in mass.items() if L != home and m > eps and L in self.actions),
                        reverse=True)
        best = None
        for m, L in others[: self.n_cand]:
            sg_pref = "-" if self.last_sigma.get(L) == "+" else "+"
            cand = [a for a in self.actions[L] if p[a[4]] > eps]
            if not cand:
                continue
            pref = [a for a in cand if a[1] == sg_pref] or cand
            a = max(pref, key=lambda a: (p[a[4]], a[6]))
            if self.tau_rule == "pi":
                ti, val = a[3], m
            else:
                ti, val = _expected_purity_best_tau(self.engine, p, a[2], a[1])
            if best is None or val > best[0]:
                best = (val, a, ti, L, m)
        if best is not None:
            _v, a, ti, L, m = best
            self.last_sigma[L] = a[1]
            self.why = f"strip {L} (mass {m:.3f}) with sigma{a[1]}"
            return self._emit(a, ti)

        # 2. nothing else left: concentrate inside the home ladder
        cand = [a for a in self.actions.get(home, []) if p[a[4]] > eps and a[4] != star]
        if cand:
            a = max(cand, key=lambda a: (p[a[4]], a[6]))
            ti = a[3] if self.tau_rule == "pi" else _expected_purity_best_tau(self.engine, p, a[2], a[1])[0]
            self.why = f"collapse {home} rung {self.molecule.system.levels[a[4]][3]:+.1f}"
            return self._emit(a, ti)

        # 3. the remaining population is dark to every in-window pulse -> primitive
        dark = sorted(((p[s], k) for k, s in self.prims if p[s] > eps and s != star), reverse=True,
                      key=lambda x: x[0])
        if dark:
            k = dark[0][1]
            self.why = f"primitive from {self.molecule.system.levels[self.prims[k][1]]}"
            return self.library.n_grid + k
        self.why = "no action"
        return None

    def _emit(self, a, ti):
        if ti == a[3]:
            return a[0]
        return Action(a[1], a[2], int(ti), -1)


# Eq. 25 closed-loop planner


class ScorePlannerPolicy:
    """Port of baseline_protocol.py::choose_by_score (module docstring)."""

    stateful = False

    def __init__(self, engine: TauBatchedEngine, library: ActionLibrary, cfg: ScoreConfig | None = None,
                 tau_mode: str = "library", n_pool: int | None = None, delta_s: float = 0.003):
        if tau_mode not in ("library", "free"):
            raise ValueError(f"unknown tau_mode {tau_mode!r}")
        if n_pool is not None and int(n_pool) < 1:
            raise ValueError("n_pool must be >= 1 (or None for the greedy choice)")
        self.engine, self.library, self.tau_mode = engine, library, tau_mode
        self.n_pool = None if n_pool is None else int(n_pool)
        self.delta_s = float(delta_s)
        self.cfg = cfg if cfg is not None else ScoreConfig().for_molecule(library.molecule)
        self.calls = 0
        eng_rows = {int(t): i for i, t in enumerate(np.asarray(engine.tau_indices))}
        # drive -> (omega, sigma, [(engine_row, action_index)])
        drives: dict = {}
        for sg, w, a_idx, t_idx in library.drives():
            key = (round(float(w), 6), sg)
            rows = []
            for a, t in zip(a_idx, t_idx):
                if int(t) not in eng_rows:
                    raise ValueError(f"engine lacks tau index {int(t)} needed by the library")
                rows.append((eng_rows[int(t)], int(a)))
            drives.setdefault(key, (float(w), sg, []))[2].extend(rows)
        for k in range(library.n_primitives):
            q = library.molecule.primitives[k]
            ti = library.primitive_tau_index(k)
            if ti not in eng_rows:
                raise ValueError(f"engine lacks tau index {ti} needed by primitive {k}")
            drives.setdefault((round(float(q.omega), 6), q.sigma), (float(q.omega), q.sigma, []))[2].append(
                (eng_rows[ti], library.n_grid + k))
        by_sigma: dict = collections.OrderedDict()
        for key in sorted(drives):
            w, sg, rows = drives[key]
            by_sigma.setdefault(sg, []).append((w, rows))
        self._groups = [(sg, np.asarray([w for w, _ in lst]), [r for _, r in lst]) for sg, lst in by_sigma.items()]
        self._eng_tau = np.asarray(engine.tau_indices, dtype=np.int64)

    def act(self, p, t, rng):
        p = np.asarray(p, dtype=np.float64)
        cands: list = []                       # (value, pick) per drive, in visiting order
        for sg, ws, rows_list in self._groups:
            P0, P1 = branches_batch_fallback(self.engine, p, ws, sg)
            self.calls += int(ws.size)
            for w, a, c, rows in zip(ws, P0, P1, rows_list):
                sc = score_batch(p, np.asarray(a), np.asarray(c), self.cfg)
                if self.tau_mode == "library":
                    r = np.asarray([r for r, _ in rows])
                    j = int(np.argmax(sc[r]))
                    cands.append((float(sc[r[j]]), rows[j][1]))
                else:
                    k = int(np.argmax(sc))
                    cands.append((float(sc[k]), (sg, float(w), int(self._eng_tau[k]), rows)))
        if not cands:
            return None
        pick = self._choose(cands, rng)
        if isinstance(pick, tuple):
            sg, w, ti, rows = pick
            prim = rows[0][1] - self.library.n_grid if rows and rows[0][1] >= self.library.n_grid else -1
            act = Action(sg, w, ti, prim)
            try:                      # return the index when the library has it
                a = self.library.encode(act)
                if self.library.decode(a).tau_index == ti:
                    return a
            except KeyError:
                pass
            return act
        return int(pick)

    def _choose(self, cands: list, rng):
        vals = np.asarray([v for v, _ in cands])
        if self.n_pool is None:
            return cands[int(np.argmax(vals))][1]          # first maximum, as the script
        order = np.argsort(-vals, kind="stable")
        pool = set(order[:self.n_pool].tolist())
        pool.update(np.where(vals >= vals[order[0]] - self.delta_s)[0].tolist())
        members = sorted(pool)
        return cands[members[int(rng.integers(len(members)))]][1]
