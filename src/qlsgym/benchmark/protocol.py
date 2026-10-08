from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..env.actions import ActionLibrary, ControlGrid
from ..env.cache import build_action_tables, work_dir
from ..env.env import EnvConfig, PurificationEnv
from ..physics.engines import engine_choice
from ..policies.baselines import RandomPolicy, ScorePlannerPolicy, SweepingPolicy
from ..policies.rollout import RolloutResult, rollout
from ..policies.score import ScoreConfig
from ..rl.ppo import PPOConfig, load_policy, policy_from_state_dict, save_policy, train_ppo
from ..spec import Molecule
from .curve import finished_curve, from_rollout, pulses_to

ARMS = ("sweeping", "random", "planner-exact", "planner-fno", "rl-exact", "rl-fno")
FNO_ARMS = ("planner-fno", "rl-fno")
LIBRARIES = ("physics_subset", "rl_discrete")

MOLECULE_DEFAULTS = {"h3o": {"max_pulses": 400, "eval_greedy": True},
                     "thf": {"max_pulses": 800, "eval_greedy": False}}
DEFAULTS_FALLBACK = {"max_pulses": None, "eval_greedy": True}
FNO_TAG_ORDER = ("mix",)


@dataclass(frozen=True)
class BenchmarkConfig:
    molecule: str = "h3o"
    library: str = "physics_subset"
    n_episodes: int = 1000
    seed: int = 777
    max_pulses: int | None = None
    p_target: float | None = None
    fno_tag: str = "mix"
    device: str = "cpu"
    engine: str = "auto"
    n_pool: int | None = 16
    delta_s: float = 0.003
    score: ScoreConfig = ScoreConfig()
    rho: float = 2.0
    penalty_mode: str = "indicator"
    ppo: PPOConfig = PPOConfig()
    retrain: bool = False
    n_freq: int = 1764
    n_tau_slots: int = 10
    grid_seed: int = 0

    def __post_init__(self):
        if self.library not in LIBRARIES:
            raise ValueError(f"library must be one of {LIBRARIES}, got {self.library!r}")
        if self.n_episodes < 1:
            raise ValueError("n_episodes must be >= 1")
        engine_choice(self.engine)

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def molecule_defaults(molecule_name: str) -> dict:
    return dict(MOLECULE_DEFAULTS.get(str(molecule_name), DEFAULTS_FALLBACK))


def default_fno_tag(molecule_name: str, log=None) -> str:
    from ..surrogate.manifest import manifest_path
    say = log if log is not None else (lambda s: print(s, flush=True))
    have = [t for t in FNO_TAG_ORDER if os.path.exists(manifest_path(str(molecule_name), t))]
    if not have:
        d = os.path.dirname(manifest_path(str(molecule_name), FNO_TAG_ORDER[0]))
        present = sorted(f[:-5] for f in os.listdir(d)) if os.path.isdir(d) else []
        raise FileNotFoundError(
            f"no surrogate manifest for {molecule_name} under {d} (tried {list(FNO_TAG_ORDER)}; "
            f"present: {present or 'none'}); pass --fno-tag explicitly")
    if have[0] != FNO_TAG_ORDER[0]:
        say(f"[benchmark] {molecule_name} has no {FNO_TAG_ORDER[0]!r} manifest; "
            f"fno tag defaults to {have[0]!r}")
    return have[0]


def resolve_fno_tag(molecule_name: str, arm: str, tag: str | None = None, log=None) -> str:
    if tag is not None:
        return str(tag)
    try:
        return default_fno_tag(molecule_name, log=log)
    except FileNotFoundError:
        if arm in FNO_ARMS:
            raise
        return BenchmarkConfig.fno_tag


def load(cfg: BenchmarkConfig) -> Molecule:
    from .. import load_molecule
    return load_molecule(cfg.molecule)


def build_library(cfg: BenchmarkConfig, molecule: Molecule) -> ActionLibrary:
    if cfg.library == "physics_subset":
        return ActionLibrary.physics_subset(molecule, clip_tau=(molecule.name == "synthetic"))
    for slots in range(int(cfg.n_tau_slots), 0, -1):
        try:
            grid = ControlGrid.rl_discrete(molecule, n_freq=cfg.n_freq, n_tau_slots=slots, seed=cfg.grid_seed)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"{molecule.name}'s tau grid holds no rl_discrete duration slot")
    return ActionLibrary.from_grid(molecule, grid)


def library_taus(library: ActionLibrary) -> np.ndarray:
    prim = [library.primitive_tau_index(k) for k in range(library.n_primitives)]
    return np.unique(np.concatenate([library.tau_indices, np.asarray(prim, dtype=np.int64)]))


def env_config(cfg: BenchmarkConfig) -> EnvConfig:
    return EnvConfig(p_target=cfg.p_target, max_pulses=cfg.max_pulses, rho=cfg.rho,
                     penalty_mode=cfg.penalty_mode)


def score_config(cfg: BenchmarkConfig, molecule: Molecule) -> ScoreConfig:
    sc = cfg.score.for_molecule(molecule)
    if cfg.p_target is not None:
        sc = replace(sc, p_target=float(cfg.p_target))
    return sc


def exact_env(cfg: BenchmarkConfig, molecule: Molecule, library: ActionLibrary, batch: int = 1,
              progress: bool = False) -> PurificationEnv:
    tables = build_action_tables(molecule, library, device=cfg.device, progress=progress)
    return PurificationEnv(molecule, library, tables, env_config(cfg), device=cfg.device, batch=batch)


def default_engine(cfg: BenchmarkConfig, molecule: Molecule, tau_indices, log=None):
    from ..physics.engines import select_engine
    return select_engine(molecule, tau_indices=tau_indices, prefer=cfg.engine, log=log)


def fno_engine(cfg: BenchmarkConfig, molecule: Molecule, tau_indices, fallback=None, log=None):
    from ..surrogate.manifest import load_manifest
    fb = default_engine(cfg, molecule, tau_indices, log=log) if fallback is None else fallback
    return load_manifest(molecule, cfg.fno_tag, tau_indices=tau_indices, device=cfg.device, fallback=fb)


def fno_env(cfg: BenchmarkConfig, molecule: Molecule, library: ActionLibrary, engine, batch: int = 1,
            progress: bool = False):
    from ..surrogate.fno_env import FnoEnv

    tables = build_action_tables(molecule, library, device=cfg.device, progress=progress)
    return FnoEnv(molecule, library, tables, engine, env_config(cfg), device=cfg.device, batch=batch)


def engine_provenance(engine) -> dict:
    d = {"kind": type(engine).__name__}
    m = getattr(engine, "manifest", None)
    if m is not None:
        d.update(manifest_tag=m.tag, manifest_fingerprint=m.fingerprint, manifest_created=m.created,
                 entries={k: {"path": e.path, "provenance": e.provenance} for k, e in sorted(m.entries.items())})
    for attr in ("selection", "settings", "stats"):
        v = getattr(engine, attr, None)
        if v is not None:
            d[attr] = dict(v)
    fb = getattr(engine, "_fallback", None)
    if fb is not None:
        d["fallback"] = engine_provenance(fb)
    if hasattr(engine, "trained"):
        d["trained"] = sorted([list(k) for k in engine.trained])
    if hasattr(engine, "calls"):
        d["calls"] = dict(engine.calls)
        f = engine.surrogate_fraction()
        d["surrogate_fraction"] = float(f) if math.isfinite(f) else None
    return d


def evaluate(env: PurificationEnv, policy, cfg: BenchmarkConfig, progress: bool = False) -> RolloutResult:
    return rollout(env, policy, n_rollouts=cfg.n_episodes, seed=cfg.seed, progress=progress)


def manifest_digest(engine) -> str | None:
    m = getattr(engine, "manifest", None)
    if m is None:
        return None
    blob = json.dumps(m.to_json().get("entries", {}), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def rl_key(cfg: BenchmarkConfig, library: ActionLibrary, arm: str, molecule: Molecule | None = None,
           engine=None) -> dict:
    ec = env_config(cfg)
    if molecule is not None:
        ec = ec.resolve(molecule)
    return {"arm": arm, "molecule": cfg.molecule, "library_tag": library.tag(),
            "fingerprint": None if molecule is None else molecule.fingerprint(),
            "env_config": asdict(ec), "ppo": cfg.ppo.as_dict(),
            "fno_tag": None if arm == "rl-exact" else cfg.fno_tag,
            "manifest": manifest_digest(engine)}


def rl_run_dir(cfg: BenchmarkConfig, library: ActionLibrary, arm: str, molecule: Molecule | None = None,
               engine=None) -> str:
    src = "exact" if arm == "rl-exact" else cfg.fno_tag
    blob = json.dumps(rl_key(cfg, library, arm, molecule, engine), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:12]
    name = f"{arm}_{src}_{library.tag()[:12]}_{h}_seed{cfg.ppo.seed}"
    return os.path.join(work_dir(), "runs", "ppo", cfg.molecule, name)


def train_or_load_actor(cfg: BenchmarkConfig, molecule: Molecule, library: ActionLibrary,
                        env_train: PurificationEnv, env_eval: PurificationEnv, run_dir: str,
                        tables_builder: str, log=None):
    best_path = os.path.join(run_dir, "actor_best.pt")
    want_env, want_ppo = asdict(env_train.cfg), cfg.ppo.as_dict()
    if os.path.exists(best_path) and not cfg.retrain:
        pol = load_policy(best_path, device=cfg.device, greedy=cfg.ppo.eval_greedy)
        meta = pol.meta
        bad = []
        if meta.get("fingerprint") != molecule.fingerprint():
            bad.append(f"fingerprint {meta.get('fingerprint')} != {molecule.fingerprint()}")
        if meta.get("library_tag") != library.tag():
            bad.append(f"library_tag {meta.get('library_tag')} != {library.tag()}")
        got_env = meta.get("env_config")
        if got_env is not None and got_env != want_env:
            diff = sorted(k for k in set(got_env) | set(want_env) if got_env.get(k) != want_env.get(k))
            bad.append("env_config differs in " + ", ".join(
                f"{k} ({got_env.get(k)!r} != {want_env.get(k)!r})" for k in diff))
        got_ppo = pol.config.as_dict()
        if got_ppo != want_ppo:
            diff = sorted(k for k in set(got_ppo) | set(want_ppo) if got_ppo.get(k) != want_ppo.get(k))
            bad.append("ppo config differs in " + ", ".join(
                f"{k} ({got_ppo.get(k)!r} != {want_ppo.get(k)!r})" for k in diff))
        if bad:
            raise RuntimeError(
                f"{best_path} was not trained for this run, so evaluating it would report one "
                f"configuration while running another: " + "; ".join(bad) + "; use retrain=True")
        tj = os.path.join(run_dir, "train.json")
        train_meta = json.load(open(tj)) if os.path.exists(tj) else dict(meta)
        train_meta["loaded_from"] = best_path
        return pol, train_meta
    os.makedirs(run_dir, exist_ok=True)
    res = train_ppo(env_train, cfg.ppo, env_eval=env_eval, log=log)
    meta = {"fingerprint": molecule.fingerprint(), "library_tag": library.tag(), "molecule": molecule.name,
            "tables_builder": tables_builder, "env_config": want_env, "ppo_config": want_ppo,
            "selected": res.best}
    best = policy_from_state_dict(res.best_state_dict, res.n_in, res.n_actions, cfg.ppo, cfg.device,
                                 greedy=cfg.ppo.eval_greedy)
    final = policy_from_state_dict(res.final_state_dict, res.n_in, res.n_actions, cfg.ppo, cfg.device,
                                  greedy=cfg.ppo.eval_greedy)
    save_policy(best.net, cfg.ppo, best_path, meta)
    save_policy(final.net, cfg.ppo, os.path.join(run_dir, "actor_final.pt"), meta)
    train_meta = {"run_dir": run_dir, "ppo": res.as_dict(), **meta}
    with open(os.path.join(run_dir, "train.json"), "w") as fh:
        json.dump(train_meta, fh, indent=1)
    return best, train_meta


def run_arm(arm: str, cfg: BenchmarkConfig, log=None, progress: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    say = log if log is not None else (lambda s: None)
    t0 = time.time()
    molecule = load(cfg)
    library = build_library(cfg, molecule)
    taus = library_taus(library)
    say(f"[benchmark] {molecule.name}: {molecule.n_states} states, library {cfg.library} "
        f"{library.n_actions} actions ({library.n_grid} grid + {library.n_primitives} primitives)")
    env = exact_env(cfg, molecule, library, batch=cfg.n_episodes, progress=progress)
    say(f"[benchmark] exact tables ready ({time.time() - t0:.0f}s); p_target {env.cfg.p_target}, "
        f"max_pulses {env.cfg.max_pulses}, {cfg.n_episodes} episodes at seed {cfg.seed}")
    engine, engine_info, rl_info, policy_desc = None, None, None, arm

    if arm == "sweeping":
        policy = SweepingPolicy(library.n_actions)
    elif arm == "random":
        policy = RandomPolicy(library.n_actions)
    elif arm.startswith("planner-"):
        ref = default_engine(cfg, molecule, taus, log=say)
        engine = ref if arm == "planner-exact" else fno_engine(cfg, molecule, taus, fallback=ref)
        policy = ScorePlannerPolicy(engine, library, score_config(cfg, molecule), tau_mode="library",
                                    n_pool=cfg.n_pool, delta_s=cfg.delta_s)
        policy_desc = (f"{arm} (n_pool={cfg.n_pool}, delta_s={cfg.delta_s}, "
                       f"dynamics={type(ref).__name__})")
    else:
        if arm == "rl-exact":
            env_train, builder = env.clone(cfg.ppo.n_envs), "physics"
        else:
            engine = fno_engine(cfg, molecule, taus, log=say)
            env_train = fno_env(cfg, molecule, library, engine, batch=cfg.ppo.n_envs, progress=progress)
            builder = "FnoEnv:" + cfg.fno_tag
        run_dir = rl_run_dir(cfg, library, arm, molecule=molecule, engine=engine)
        say(f"[benchmark] {arm}: tables from {builder}; run dir {run_dir}")
        policy, rl_info = train_or_load_actor(cfg, molecule, library, env_train, env, run_dir, builder, log=log)
        say(f"[benchmark] actor ready ({time.time() - t0:.0f}s): selected snapshot {rl_info.get('selected')}")

    res = evaluate(env, policy, cfg, progress=progress)
    if engine is not None:
        engine_info = engine_provenance(engine)
    out = {
        "arm": arm, "policy": policy_desc, "molecule": molecule.name, "fingerprint": molecule.fingerprint(),
        "library": library.describe(), "config": _jsonable(cfg.as_dict()),
        "engine": engine_info, "rl": rl_info, "seconds": time.time() - t0,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out.update(from_rollout(res))
    say(f"[benchmark] {arm}: success {out['success']:.3f}, mean pulses {out['mean_pulses']:.1f}, "
        f"P85 {out['p85']} ({out['seconds']:.0f}s)")
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def outputs_dir(molecule_name: str) -> str:
    base = os.environ.get("QLSGYM_OUTPUTS")
    if base is None:
        root = os.environ.get("QLSGYM_ROOT")
        if root is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        base = os.path.join(root, "OUTPUTS", "benchmark")
    return os.path.join(base, molecule_name)


def result_path(molecule_name: str, arm: str, part: int | None = None) -> str:
    name = f"{arm}.json" if part is None else f"{arm}.part{int(part):02d}.json"
    return os.path.join(outputs_dir(molecule_name), name)


def write_result(result: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = _jsonable(result)
    try:
        blob = json.dumps(payload, indent=1, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"refusing to write {path}: the result contains a non-finite number "
                         f"({exc}); JSON has no NaN or Infinity") from exc
    with open(path, "w") as fh:
        fh.write(blob)
    return path


def merge_results(parts: list) -> dict:
    if not parts:
        raise ValueError("nothing to merge")
    head = parts[0]
    for p in parts[1:]:
        for key in ("arm", "fingerprint", "max_pulses", "p_target"):
            if p[key] != head[key]:
                raise ValueError(f"cannot merge: {key} differs ({p[key]!r} vs {head[key]!r})")
        if p["library"]["tag"] != head["library"]["tag"]:
            raise ValueError("cannot merge: library tag differs")
        if _engine_tag(p) != _engine_tag(head):
            raise ValueError(f"cannot merge: the parts consulted different dynamics "
                             f"({_engine_tag(p)!r} vs {_engine_tag(head)!r})")
    seeds = [p["config"]["seed"] for p in parts]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"cannot merge parts that share an evaluation seed: {seeds}")
    lengths = np.concatenate([p["lengths"] for p in parts]).astype(np.int64)
    succ = np.concatenate([p["successes"] for p in parts]).astype(bool)
    curve = finished_curve(lengths, succ, int(head["max_pulses"]))
    n = lengths.size
    out = dict(head)
    out.update(curve=curve.tolist(), p85=pulses_to(curve, 0.85), success=float(succ.mean()),
               success_err=float(np.sqrt(succ.mean() * (1 - succ.mean()) / n)), mean_pulses=float(lengths.mean()),
               mean_pulses_successful=(float(lengths[succ].mean()) if succ.any() else None), n_episodes=int(n),
               lengths=lengths.tolist(), successes=succ.tolist(), seconds=float(sum(p["seconds"] for p in parts)),
               merged_from=[{"seed": p["config"]["seed"], "n_episodes": p["n_episodes"]} for p in parts])
    out["outcomes"] = {"success": float(succ.mean()), "max_pulses": float(1 - succ.mean())}
    out["config"] = dict(head["config"], n_episodes=int(n), seed=None)
    out["seeds"] = [int(x) for x in seeds]
    out["created"] = _dt.datetime.now().isoformat(timespec="seconds")
    out["engine"] = _merge_engine([p.get("engine") for p in parts])
    return out


def _engine_tag(rec: dict):
    e = rec.get("engine")
    if not isinstance(e, dict):
        return None
    return (e.get("kind"), e.get("manifest_tag"), e.get("manifest_fingerprint"))


def _merge_engine(engines: list):
    present = [e for e in engines if isinstance(e, dict)]
    if not present:
        return None
    out = dict(present[0])
    calls: dict = {}
    for e in present:
        for k, v in (e.get("calls") or {}).items():
            calls[k] = calls.get(k, 0) + int(v)
    if calls:
        out["calls"] = calls
        tot = sum(calls.values())
        out["surrogate_fraction"] = (calls.get("fno", 0) / tot) if tot else None
    out["merged_from_parts"] = len(present)
    return out


def unmerged_parts(directory: str) -> dict:
    out = {}
    for arm in ARMS:
        if os.path.exists(os.path.join(directory, f"{arm}.json")):
            continue
        parts = sorted(glob.glob(os.path.join(directory, f"{arm}.part*.json")))
        if parts:
            out[arm] = parts
    return out


def read_results(molecule: Molecule, directory: str | None = None, warn=None) -> list:
    d = directory or outputs_dir(molecule.name)
    say = warn if warn is not None else (lambda s: None)
    for arm, parts in unmerged_parts(d).items():
        say(f"!! {arm}: {len(parts)} part file(s) under {d} but no {arm}.json, so this arm is "
            f"MISSING from the results; run `benchmark.py merge {molecule.name} {arm}` first "
            f"({', '.join(os.path.basename(x) for x in parts)})")
    out = []
    for arm in ARMS:
        path = os.path.join(d, f"{arm}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            r = json.load(fh)
        if r.get("fingerprint") != molecule.fingerprint():
            say(f"{path}: fingerprint {r.get('fingerprint')} is not the current {molecule.fingerprint()}; skipped")
            continue
        out.append(r)
    return out
