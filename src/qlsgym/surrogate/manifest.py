"""Checkpoint manifests under $QLSGYM_WORK/checkpoints: "<block>,<sigma>" -> checkpoint path."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..spec import Molecule
from .dataset import work_dir

# summary.json keys copied into a manifest entry
_SUMMARY_KEYS = ("strat_on_all_ratio_to_static_median", "strat_on_all_ratio_to_static_p95",
                 "strat_on_on_median", "strat_on_on_static_median", "strat_off_all_ratio_to_static_median",
                 "time_avg_median", "static_time_avg_median", "best_onres_epoch", "best_onres_infidelity",
                 "best_epoch", "n_pairs", "n_freq", "n_init", "n_parameters", "selection", "strat_active_fraction")

# every value check_checkpoint_provenance can return
PROVENANCE_VALUES = ("tables-sha256", "fingerprint")


class StaleCheckpoint(RuntimeError):
    """The checkpoint was not shown to be trained on the molecule's current tables."""


def entry_key(block_index: int, sigma: str) -> str:
    return f"{int(block_index)},{sigma}"


def parse_key(key: str) -> tuple[int, str]:
    b, s = key.split(",")
    return int(b), s.strip()


# Provenance (physics vintage) of a checkpoint


def sha256_file(path) -> str:
    """Hex sha256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_files(molecule: Molecule):
    lf, rf = molecule.provenance.get("levels_file"), molecule.provenance.get("rabi_file")
    return (lf, rf) if lf and rf else None


def table_stamp(molecule: Molecule) -> dict:
    """The ck["tables"] stamp a checkpoint trained on molecule gets."""
    stamp = {"heff_git_rev": molecule.provenance.get("heff_git_rev"),
             "heff_spec_hash": molecule.provenance.get("heff_spec_hash")}
    files = _table_files(molecule)
    if files is not None:
        stamp["levels_sha256"] = sha256_file(files[0])
        stamp["rabi_sha256"] = sha256_file(files[1])
    return stamp


def check_checkpoint_provenance(ck: dict, run_dir: str, molecule: Molecule, block_index: int,
                                sigma: str) -> str:
    """Whether checkpoint ck from run_dir was trained on the molecule's current tables.

    'tables-sha256' when the molecule has table files and the checkpoint's stamp hashes match them,
    'fingerprint' when it has none and the checkpoint's fingerprint matches; StaleCheckpoint otherwise.
    """
    where = f"{run_dir} ({molecule.name} block {block_index} sigma{sigma})"
    retrain = "retrain on the current tables"
    stamp = ck.get("tables") if isinstance(ck, dict) else None
    if stamp is None:
        raise StaleCheckpoint(f"{where}: the checkpoint carries no table stamp (ck['tables']), so nothing "
                              f"shows it was trained on the current tables; {retrain}")
    if not isinstance(stamp, dict):
        stamp = {}                                       # a malformed stamp matches nothing
    files = _table_files(molecule)
    if files is None:
        fp, want = (ck.get("meta") or {}).get("fingerprint"), molecule.fingerprint()
        if fp != want:
            raise StaleCheckpoint(
                f"{where}: fingerprint check failed -- molecule {molecule.name!r} has no table files to "
                f"hash and the checkpoint fingerprint {fp} != molecule fingerprint {want}; {retrain}")
        return "fingerprint"
    for key, path in (("levels_sha256", files[0]), ("rabi_sha256", files[1])):
        got = sha256_file(path)
        if stamp.get(key) != got:
            raise StaleCheckpoint(
                f"{where}: tables-sha256 check failed -- checkpoint {key} {stamp.get(key)} != sha256 "
                f"{got} of {path} (checkpoint heff {stamp.get('heff_git_rev')}, molecule heff "
                f"{molecule.provenance.get('heff_git_rev')}); {retrain}")
    return "tables-sha256"


# Manifest


@dataclass
class ManifestEntry:
    block_index: int
    sigma: str
    path: str                              # absolute checkpoint path
    run_dir: str = ""
    summary: dict = field(default_factory=dict)   # selected summary.json metrics
    train_data: dict = field(default_factory=dict)  # checkpoint's train_data (alpha, n_freq, ...)
    # check_checkpoint_provenance verdict; None = never checked (refused on load)
    provenance: str | None = None

    @property
    def key(self) -> str:
        return entry_key(self.block_index, self.sigma)

    @property
    def onres_ratio_to_static(self) -> float | None:
        return self.summary.get("strat_on_all_ratio_to_static_median")


@dataclass
class Manifest:
    molecule: str
    tag: str
    fingerprint: str                       # qlsgym Molecule.fingerprint() the entries are valid for
    entries: dict = field(default_factory=dict)   # key -> ManifestEntry
    note: str = ""
    created: str = ""

    def checkpoints(self) -> dict:
        """{(block_index, sigma): path} for FnoEngine."""
        return {(e.block_index, e.sigma): e.path for e in self.entries.values()}

    def add(self, entry: ManifestEntry) -> None:
        self.entries[entry.key] = entry

    def to_json(self) -> dict:
        return {"molecule": self.molecule, "tag": self.tag, "fingerprint": self.fingerprint,
                "note": self.note, "created": self.created,
                "entries": {k: asdict(e) for k, e in sorted(self.entries.items(), key=lambda kv: parse_key(kv[0]))}}

    @classmethod
    def from_json(cls, d: dict) -> "Manifest":
        m = cls(d["molecule"], d["tag"], d["fingerprint"], {}, d.get("note", ""), d.get("created", ""))
        for k, e in d.get("entries", {}).items():
            m.entries[k] = ManifestEntry(**e)
        return m

    def save(self, path: str | None = None, work: str | None = None) -> str:
        path = path or manifest_path(self.molecule, self.tag, work)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path) as fh:
            return cls.from_json(json.load(fh))

    def table(self) -> str:
        """Human summary: one line per entry with the on-resonance ratio and provenance."""
        lines = [f"{self.molecule} / {self.tag}  fingerprint={self.fingerprint}  {len(self.entries)} entries"]
        for k, e in sorted(self.entries.items(), key=lambda kv: parse_key(kv[0])):
            r = e.onres_ratio_to_static
            lines.append(f"  {k:>6}  onres ratio {r if r is None else f'{r:.4f}'}  provenance {e.provenance}  "
                         f"{e.path}")
        return "\n".join(lines)


def manifest_path(molecule_name: str, tag: str, work: str | None = None) -> str:
    return os.path.join(work or work_dir(), "checkpoints", molecule_name, f"{tag}.json")


def _read_summary(run_dir: str) -> dict:
    f = os.path.join(run_dir, "summary.json")
    if not os.path.exists(f):
        return {}
    with open(f) as fh:
        j = json.load(fh)
    return {k: j[k] for k in _SUMMARY_KEYS if k in j}


def build_manifest(
    molecule: Molecule,
    tag: str,
    runs: dict,
    checkpoint_name: str = "best_onres.pt",
    note: str = "",
    require_summary: bool = False,
    check_load: bool = True,
    device: str = "cpu",
) -> Manifest:
    """Assemble a manifest from {(block_index, sigma): run_dir}."""
    import torch

    from .train import load_model

    m = Manifest(molecule.name, tag, molecule.fingerprint(), note=note,
                 created=_dt.datetime.now().isoformat(timespec="seconds"))
    refused: list[str] = []
    for (b, s), run_dir in runs.items():
        run_dir = os.path.abspath(run_dir)
        path = os.path.join(run_dir, checkpoint_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        summary = _read_summary(run_dir)
        if require_summary and not summary:
            raise FileNotFoundError(os.path.join(run_dir, "summary.json"))
        if check_load:
            load_model(path, device, molecule=molecule)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        if (int(meta["block_index"]), meta["sigma"]) != (int(b), s):
            raise ValueError(f"{path}: checkpoint is block {meta['block_index']} sigma{meta['sigma']}, "
                             f"registered as ({b}, {s})")
        try:
            prov = check_checkpoint_provenance(ck, run_dir, molecule, int(b), s)
        except StaleCheckpoint as exc:
            refused.append(str(exc))
            continue
        train_data = ck.get("train_data", {}) or {}
        m.add(ManifestEntry(int(b), s, path, run_dir, summary, train_data, provenance=prov))
    if refused:
        raise StaleCheckpoint(f"{len(refused)} of {len(runs)} checkpoints for {molecule.name}/{tag} refused, "
                              f"no manifest built:\n  " + "\n  ".join(refused))
    return m


def load_manifest(
    molecule: Molecule,
    tag: str,
    tau_indices: np.ndarray | None = None,
    device=None,
    fallback=None,
    work: str | None = None,
    path: str | None = None,
    allow_mismatch: bool = False,
    blocks: set | None = None,
):
    """FnoEngine from a manifest."""
    from .fno_engine import FnoEngine

    path = path or manifest_path(molecule.name, tag, work)
    m = Manifest.load(path)
    if m.molecule != molecule.name:
        raise ValueError(f"{path} is for molecule {m.molecule!r}, not {molecule.name!r}")
    if m.fingerprint != molecule.fingerprint() and not allow_mismatch:
        raise RuntimeError(f"{path}: manifest fingerprint {m.fingerprint} != molecule fingerprint "
                           f"{molecule.fingerprint()}; regenerate with scripts/make_manifest.py")
    entries = [e for e in m.entries.values() if blocks is None or (e.block_index, e.sigma) in blocks]
    bad = [f"{e.key} ({e.run_dir or e.path}): provenance {e.provenance!r}"
           for e in entries if e.provenance not in PROVENANCE_VALUES]
    if bad:
        raise StaleCheckpoint(
            f"{path}: {len(bad)} entries have no recorded provenance check (edited by hand?), so nothing "
            f"shows their checkpoints were trained on the current tables; rebuild with "
            f"scripts/make_manifest.py:\n  " + "\n  ".join(bad))
    ck = {(e.block_index, e.sigma): e.path for e in entries}
    eng = FnoEngine(molecule, ck, tau_indices=tau_indices, device=device, fallback=fallback,
                    allow_mismatch=allow_mismatch)
    eng.manifest = m
    return eng
