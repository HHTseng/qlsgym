"""Checkpoint manifests under $QLSGYM_WORK/checkpoints: "<block>,<sigma>" -> checkpoint path."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from ..spec import Molecule
from .dataset import work_dir

# summary.json keys copied into a manifest entry (source names kept)
_SUMMARY_KEYS = ("strat_on_all_ratio_to_static_median", "strat_on_all_ratio_to_static_p95",
                 "strat_on_on_median", "strat_on_on_static_median", "strat_off_all_ratio_to_static_median",
                 "time_avg_median", "static_time_avg_median", "best_onres_epoch", "best_onres_infidelity",
                 "best_epoch", "n_pairs", "n_freq", "n_init", "n_parameters", "selection", "strat_active_fraction")

# resonance check: tolerance, in half-linewidths of the reference transition
RESONANCE_TOL_HWHM = 3.0
# resonance check: minimum fraction of stored on-resonance frequencies within tolerance
RESONANCE_MIN_FRACTION = 0.90
# every value check_checkpoint_provenance can return
PROVENANCE_VALUES = ("tables-sha256", "fingerprint", "resonance-check", "unverified")


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


def reference_hwhm(molecule: Molecule, block_index: int) -> tuple[float, str]:
    """Half-linewidth (rad/ms) the resonance check measures distances in."""
    eta = float(molecule.trap.eta)
    ref = molecule.provenance.get("reference_rabi_khz")
    if ref is not None:
        om, src = 2.0 * np.pi * float(ref), "provenance['reference_rabi_khz']"
    else:
        om, src = float(np.abs(molecule.blocks[int(block_index)].omega).max()), "max |Omega| of the block"
    return eta * float(np.exp(-eta ** 2 / 2.0)) * om, src


def current_resonances(molecule: Molecule, block_index: int, sigma: str) -> np.ndarray:
    """Sideband resonances of the block's couplings that lie inside the drive window."""
    from ..physics.spectrum import resonant_frequencies

    b = molecule.blocks[int(block_index)]
    w = molecule.window
    res = resonant_frequencies(molecule, int(block_index), sigma)
    keep = (res >= w.omega_min) & (res <= w.omega_max) & (np.abs(b.omega) >= w.omega_min_coupling)
    return np.sort(res[keep])


def resonance_fraction(molecule: Molecule, run_dir: str, block_index: int, sigma: str) -> dict | None:
    """The resonance check's measurement, or None without a usable stratified_eval.npz."""
    f = os.path.join(run_dir, "stratified_eval.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        if "on_omegas" not in z.files or "on_on_resonance" not in z.files:
            return None
        om = np.asarray(z["on_omegas"], dtype=np.float64)[np.asarray(z["on_on_resonance"]).astype(bool)]
    if om.size == 0:
        return None
    hwhm, src = reference_hwhm(molecule, block_index)
    res = current_resonances(molecule, block_index, sigma)
    if res.size == 0:
        d = np.full(om.shape, np.inf)
    else:
        d = np.abs(om[:, None] - res[None, :]).min(axis=1) / hwhm
    return {"fraction": float(np.mean(d <= RESONANCE_TOL_HWHM)), "n_on": int(om.size),
            "median_hwhm": float(np.median(d)), "hwhm": hwhm, "hwhm_source": src}


def _provenance(ck: dict, run_dir: str, molecule: Molecule, block_index: int, sigma: str,
                allow_unprovenanced: bool) -> tuple[str, dict]:
    """check_checkpoint_provenance plus the measurement behind it."""
    where = f"{run_dir} ({molecule.name} block {block_index} sigma{sigma})"
    retrain = "retrain on the current tables"
    stamp = ck.get("tables") if isinstance(ck, dict) else None
    if stamp is not None:
        if not isinstance(stamp, dict):
            stamp = {}                                   # a malformed stamp matches nothing
        files = _table_files(molecule)
        if files is None:
            fp, want = (ck.get("meta") or {}).get("fingerprint"), molecule.fingerprint()
            if fp != want:
                raise StaleCheckpoint(
                    f"{where}: fingerprint check failed -- molecule {molecule.name!r} has no table files to "
                    f"hash and the checkpoint fingerprint {fp} != molecule fingerprint {want}; {retrain}")
            return "fingerprint", {}
        for key, path in (("levels_sha256", files[0]), ("rabi_sha256", files[1])):
            got = sha256_file(path)
            if stamp.get(key) != got:
                raise StaleCheckpoint(
                    f"{where}: tables-sha256 check failed -- checkpoint {key} {stamp.get(key)} != sha256 "
                    f"{got} of {path} (checkpoint heff {stamp.get('heff_git_rev')}, molecule heff "
                    f"{molecule.provenance.get('heff_git_rev')}); {retrain}")
        return "tables-sha256", {}

    r = None                      # block_index / sigma unknown counts as "nothing to check" (like no npz)
    if block_index is not None and sigma is not None:
        if not 0 <= int(block_index) < len(molecule.blocks):
            raise StaleCheckpoint(
                f"{where}: resonance check failed -- block {block_index} does not exist in the current "
                f"molecule ({len(molecule.blocks)} blocks); {retrain} (allow_unprovenanced does not "
                f"override a failed check)")
        r = resonance_fraction(molecule, run_dir, int(block_index), sigma)
    if r is None:
        if not allow_unprovenanced:
            raise StaleCheckpoint(
                f"{where}: provenance check failed -- the checkpoint has no table stamp (ck['tables']) and the "
                f"run has no usable stratified_eval.npz for the resonance check, so nothing shows it was "
                f"trained on the current tables; {retrain}, or pass allow_unprovenanced=True "
                f"(make_manifest.py --allow-unprovenanced) if you know the tables are unchanged")
        warnings.warn(
            f"UNVERIFIED CHECKPOINT {where}: no table stamp and no stratified_eval.npz; accepted only because "
            f"allow_unprovenanced=True -- nothing checks that it was trained on the current tables",
            UserWarning, stacklevel=3)
        return "unverified", {}
    detail = {"resonance_fraction": r["fraction"], "resonance_n_on": r["n_on"],
              "resonance_median_hwhm": r["median_hwhm"]}
    if r["fraction"] < RESONANCE_MIN_FRACTION:
        raise StaleCheckpoint(
            f"{where}: resonance check failed -- only {r['fraction']:.0%} of the {r['n_on']} stored "
            f"on-resonance test frequencies lie within {RESONANCE_TOL_HWHM:g} HWHM "
            f"({r['hwhm'] / (2 * np.pi):.3f} kHz) of a current resonance (need "
            f">= {RESONANCE_MIN_FRACTION:.0%}; median distance {r['median_hwhm']:.1f} HWHM): the checkpoint "
            f"was trained on different tables; {retrain} (allow_unprovenanced does not override a failed check)")
    warnings.warn(
        f"{where}: no table stamp; accepted on the resonance check ({r['fraction']:.0%} of {r['n_on']} "
        f"stored on-resonance frequencies within {RESONANCE_TOL_HWHM:g} HWHM of a current resonance)",
        UserWarning, stacklevel=3)
    return "resonance-check", detail


def check_checkpoint_provenance(ck: dict, run_dir: str, molecule: Molecule, block_index: int, sigma: str,
                                allow_unprovenanced: bool = False) -> str:
    """Whether checkpoint ck from run_dir was trained on the molecule's current tables."""
    return _provenance(ck, run_dir, molecule, block_index, sigma, allow_unprovenanced)[0]


# Manifest


@dataclass
class ManifestEntry:
    block_index: int
    sigma: str
    path: str                              # absolute checkpoint path
    run_dir: str = ""
    summary: dict = field(default_factory=dict)   # selected source summary.json metrics
    train_data: dict = field(default_factory=dict)  # checkpoint's train_data (alpha, n_freq, ...)
    parity_residual: float | None = None   # max |qlsgym - source| from --validate, if run
    # check_checkpoint_provenance verdict; None = manifest predates the check (refused)
    provenance: str | None = None
    provenance_detail: dict = field(default_factory=dict)   # e.g. the resonance-check fraction

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
    source: str = "qlsgym"                 # "qlsgym" | "thffno" | "fnorepl"
    entries: dict = field(default_factory=dict)   # key -> ManifestEntry
    note: str = ""
    created: str = ""
    validated: dict = field(default_factory=dict)  # {"method": ..., "max_residual": ...}

    def checkpoints(self) -> dict:
        """{(block_index, sigma): path} for FnoEngine."""
        return {(e.block_index, e.sigma): e.path for e in self.entries.values()}

    def add(self, entry: ManifestEntry) -> None:
        self.entries[entry.key] = entry

    def to_json(self) -> dict:
        return {"molecule": self.molecule, "tag": self.tag, "fingerprint": self.fingerprint,
                "source": self.source, "note": self.note, "created": self.created, "validated": self.validated,
                "entries": {k: asdict(e) for k, e in sorted(self.entries.items(), key=lambda kv: parse_key(kv[0]))}}

    @classmethod
    def from_json(cls, d: dict) -> "Manifest":
        m = cls(d["molecule"], d["tag"], d["fingerprint"], d.get("source", "qlsgym"), {}, d.get("note", ""),
                d.get("created", ""), d.get("validated", {}))
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
        lines = [f"{self.molecule} / {self.tag}  source={self.source}  fingerprint={self.fingerprint}  "
                 f"{len(self.entries)} entries"]
        for k, e in sorted(self.entries.items(), key=lambda kv: parse_key(kv[0])):
            r = e.onres_ratio_to_static
            res = "" if e.parity_residual is None else f"  parity {e.parity_residual:.1e}"
            fr = e.provenance_detail.get("resonance_fraction")
            prov = f"  provenance {e.provenance}" + ("" if fr is None else f" ({fr:.0%})")
            lines.append(f"  {k:>6}  onres ratio {r if r is None else f'{r:.4f}'}{res}{prov}  {e.path}")
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
    source: str = "qlsgym",
    checkpoint_name: str = "best_onres.pt",
    note: str = "",
    require_summary: bool = False,
    check_load: bool = True,
    device: str = "cpu",
    allow_unprovenanced: bool = False,
) -> Manifest:
    """Assemble a manifest from {(block_index, sigma): run_dir}."""
    import torch

    from .train import load_model

    m = Manifest(molecule.name, tag, molecule.fingerprint(), source, note=note,
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
            load_model(path, device, molecule=molecule, legacy=(source != "qlsgym"))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        if (int(meta["block_index"]), meta["sigma"]) != (int(b), s):
            raise ValueError(f"{path}: checkpoint is block {meta['block_index']} sigma{meta['sigma']}, "
                             f"registered as ({b}, {s})")
        try:
            prov, detail = _provenance(ck, run_dir, molecule, int(b), s, allow_unprovenanced)
        except StaleCheckpoint as exc:
            refused.append(str(exc))
            continue
        train_data = ck.get("train_data", {}) or {}
        m.add(ManifestEntry(int(b), s, path, run_dir, summary, train_data, provenance=prov,
                            provenance_detail=detail))
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
            f"{path}: {len(bad)} entries have no recorded provenance check (manifest written before "
            f"check_checkpoint_provenance existed, or edited by hand), so nothing shows their checkpoints "
            f"were trained on the current tables; rebuild with scripts/make_manifest.py, and if that refuses, "
            f"retrain on the current tables:\n  " + "\n  ".join(bad))
    unverified = [e.key for e in entries if e.provenance == "unverified"]
    if unverified:
        warnings.warn(f"{path}: entries {unverified} are provenance 'unverified' (built with "
                      f"allow_unprovenanced=True): nothing checked that they were trained on the current tables",
                      UserWarning, stacklevel=2)
    ck = {(e.block_index, e.sigma): e.path for e in entries}
    eng = FnoEngine(molecule, ck, tau_indices=tau_indices, device=device, fallback=fallback,
                    legacy=(m.source != "qlsgym"), allow_mismatch=allow_mismatch)
    eng.manifest = m
    return eng
