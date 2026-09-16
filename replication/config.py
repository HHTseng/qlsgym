"""Paths, run configuration and result I/O for the replication."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# repository root (.../qlsgym) and the one output tree.
ROOT = Path(__file__).resolve().parents[1]
REPLICATION = ROOT / "replication"


def outputs_dir() -> Path:
    """OUTPUTS/replication/; $QLSGYM_REPL_OUTPUTS overrides it (tests)."""
    env = os.environ.get("QLSGYM_REPL_OUTPUTS")
    return Path(env) if env else ROOT / "OUTPUTS" / "replication"


def figures_dir() -> Path:
    return outputs_dir() / "figures"


def result_path(stage: str) -> Path:
    return outputs_dir() / f"{stage}.json"


# Configuration


@dataclass(frozen=True)
class Config:
    """One replication run. Stages take this and nothing else."""

    molecule: str = "h3o"
    blocks: tuple = ()                # empty = the unique set of resolve_blocks
    sigmas: tuple = ("+",)
    tag: str = "repl"                 # names $QLSGYM_WORK/runs/<tag>_<sp|sm>_block<b>
    device: str = "auto"              # "auto" | "cpu" | "cuda"
    dry_run: bool = False
    seed: int = 20260915
    # s02/s03 sizes (small defaults; the SLURM scripts pass the production ones)
    n_freq: int = 1000
    n_init: int = 1000
    n_pairs: int = 16000
    epochs: int = 80
    # s04/s05/s06 sizes
    n_test_freq: int = 100
    n_test_init: int = 500
    n_rollouts: int = 200
    max_pulses: int = 80
    rl_episodes: int = 3000
    overrides: dict = field(default_factory=dict)   # stage-specific escape hatch

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_molecule(cfg: Config):
    """The molecule every stage works on (import kept local: cheap --list)."""
    from qlsgym import load_molecule as _load
    return _load(cfg.molecule)


# Provenance and result I/O


def _git_rev(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def provenance(cfg: Config, molecule=None) -> dict:
    """Git rev, molecule fingerprint, config, versions, timestamp (R3)."""
    if molecule is None:
        molecule = load_molecule(cfg)
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = None
    return {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qlsgym_git_rev": _git_rev(ROOT),
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "config": cfg.as_dict(),
        "python": platform.python_version(),
        "torch": torch_version,
        "host": platform.node(),
    }


def write_result(stage: str, payload: dict, cfg: Config, molecule=None) -> Path:
    """Write OUTPUTS/replication/<stage>.json atomically, with provenance."""
    path = result_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"stage": stage, "provenance": provenance(cfg, molecule), "result": payload}
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False, default=_jsonable)
    os.replace(tmp, path)
    return path


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} is not JSON-serialisable")


class MissingResult(RuntimeError):
    """A stage's input has not been produced yet, or is not readable."""


class VintageMismatch(RuntimeError):
    """A stored result was computed on a different Hamiltonian (R3)."""


def read_result(stage: str, cfg: Config | None = None, molecule=None,
                check_fingerprint: bool = True) -> dict:
    """The result block of a previous stage, refusing a stale vintage."""
    path = result_path(stage)
    rerun = f"run `python replication/run.py {stage}` first"
    if not path.exists():
        raise MissingResult(f"{path} is missing: {rerun}")
    try:
        with open(path) as fh:
            doc = json.load(fh)
        result = doc["result"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MissingResult(f"{path} is not a {stage} result document ({type(exc).__name__}: "
                            f"{exc}): delete it and {rerun}") from exc
    if check_fingerprint:
        if molecule is None and cfg is not None:
            molecule = load_molecule(cfg)
        if molecule is not None:
            want, got = molecule.fingerprint(), doc.get("provenance", {}).get("fingerprint")
            if want != got:
                raise VintageMismatch(
                    f"{path} was computed on molecule fingerprint {got}, current is "
                    f"{want}: the Hamiltonian changed; re-run stage {stage!r}")
    return result


def stage_is_done(stage: str) -> bool:
    return result_path(stage).exists()


_warned_no_work_dir = False


def work_dir() -> Path:
    """$QLSGYM_WORK — bulk data and checkpoints, never in git (R3)."""
    global _warned_no_work_dir
    from qlsgym.surrogate.dataset import work_dir as _w

    path = Path(_w())
    if not os.environ.get("QLSGYM_WORK") and not _warned_no_work_dir:
        _warned_no_work_dir = True
        print(f"WARNING: $QLSGYM_WORK is unset, so the splits and checkpoints of this run "
              f"(tens of GB) would go to {path}, inside your home quota. "
              f"Run `source scripts/env.sh` first.", file=sys.stderr)
    return path


def run_dir(cfg: Config, block: int, sigma: str) -> Path:
    """Where stage 3 puts one surrogate; matches qlsgym's own layout."""
    sg = "sp" if sigma == "+" else "sm"
    return work_dir() / "runs" / f"{cfg.tag}_{sg}_block{block}"


def resolve_blocks(cfg: Config, molecule=None) -> tuple:
    """The blocks this run trains, checked against the molecule it runs on."""
    from .stages.s01_physics import parity_partners

    if molecule is None:
        molecule = load_molecule(cfg)
    n = len(molecule.blocks)
    if not cfg.blocks:
        return tuple(parity_partners(molecule)["unique_blocks"])
    bad = [b for b in cfg.blocks if not isinstance(b, int) or not 0 <= b < n]
    if bad:
        raise ValueError(f"blocks {bad} are not blocks of {molecule.name}, which has {n} "
                         f"(0-{n - 1}); leave Config.blocks empty for the unique set")
    return tuple(int(b) for b in cfg.blocks)


def pairs(cfg: Config, molecule=None) -> list:
    """The (block, sigma) pairs a run covers, in a deterministic order."""
    return [(b, s) for b in resolve_blocks(cfg, molecule) for s in cfg.sigmas]
