"""Stage 3 -- one FNO surrogate per (block, sigma), trained from scratch."""

from __future__ import annotations

import json
import os
import time

from ..config import Config, load_molecule, pairs, read_result, resolve_blocks, run_dir

NAME = "s03_train"
TITLE = "one FNO surrogate per (block, sigma); writes a checkpoint manifest"
PAPER = "Eqs. 11-21"
REQUIRES = ("s02_data",)
COST = "~4 h, GPU (array of 6)"

# surrogate.train.train writes both: best_onres.pt is selected on a
# held-out on-resonance set and best.pt on the validation loss (Table 2's
# rule, which over a uniform validation split measures the identity-map
# floor).  The manifest is built from the first of these that exists.
CHECKPOINT_PREFERENCE = ("best_onres.pt", "best.pt")

# summary.json keys worth carrying into the replication's own JSON; the
# first is the decisive one: the median over on-resonance test
# frequencies of I_FNO / I_trivial.
SUMMARY_KEYS = ("strat_on_all_ratio_to_static_median", "strat_on_all_ratio_to_static_p95",
                "strat_on_on_median", "strat_on_on_static_median",
                "strat_off_all_ratio_to_static_median", "strat_active_fraction",
                "time_avg_median", "static_time_avg_median", "frac_freq_fno_beats_static",
                "best_val_loss", "best_epoch", "best_onres_infidelity", "best_onres_epoch",
                "n_parameters", "selection", "swing_median")


def train_configs(cfg: Config):
    """(TrainConfig, FNOConfig) for this run."""
    from qlsgym.surrogate.fno import FNOConfig
    from qlsgym.surrogate.train import TrainConfig

    over = dict(cfg.overrides.get(NAME, {}))
    fno = FNOConfig(**over.pop("fno", {}))
    tc = TrainConfig(epochs=cfg.epochs, train_seed=cfg.seed, val_seed=cfg.seed + 1, fno=fno, **over)
    return tc, fno


def selection_config(train_data, tc):
    """The on-resonance split surrogate.train.train builds for itself."""
    from qlsgym.surrogate.dataset import DataConfig

    return DataConfig(n_freq=tc.sel_n_freq, n_init=tc.sel_n_init,
                      n_pairs=tc.sel_n_freq * tc.sel_n_init, pairing="product", seed=tc.sel_seed,
                      alpha=train_data.alpha, freq_sampling="mixture", resonant_frac=1.0,
                      resonant_spread=1.0)


def finished_summary(path: str) -> dict | None:
    """The run's summary.json if it is there and complete, else None."""
    f = os.path.join(path, "summary.json")
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fh:
            summary = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return summary if "fingerprint" in summary else None


def checkpoint_name(path: str) -> str | None:
    for name in CHECKPOINT_PREFERENCE:
        if os.path.exists(os.path.join(path, name)):
            return name
    return None


def _s02_record(data: dict, block: int, sigma: str) -> dict:
    for rec in data["pairs"]:
        if int(rec["block"]) == int(block) and rec["sigma"] == sigma:
            return rec
    raise RuntimeError(f"s02_data has no split for block {block} sigma{sigma}: re-run "
                       f"`python replication/run.py s02_data` with that block in --set blocks=...")


def _train_one(cfg, molecule, block: int, sigma: str, record: dict, tc, out_dir: str) -> dict:
    """scripts/train_fno.py's body for one pair, on cached splits."""
    from qlsgym.surrogate.dataset import DataConfig, build_or_load_split
    from qlsgym.surrogate.train import evaluate_run, load_model, train

    device = cfg.resolved_device()
    store = "cpu" if device == "cuda" else None
    train_data = DataConfig(**record["train"]["config"])
    val_data = DataConfig(**record["val"]["config"])
    kw = dict(activity_lambda=tc.activity_lambda, activity_eps=tc.activity_eps)
    tr = build_or_load_split(molecule, block, train_data, sigma, device=device, storage_device=store, **kw)
    va = build_or_load_split(molecule, block, val_data, sigma, device=device, storage_device=store, **kw)

    result = train(molecule, block, sigma, tc, train_data, val_data, device=device, out_dir=out_dir,
                   verbose=True, storage_device=store, splits=(tr, va))
    history = result["history"]
    name = "best_onres.pt" if history.get("best_onres_epoch", -1) >= 0 else "best.pt"
    model = load_model(os.path.join(out_dir, name), device, molecule=molecule)
    summary = evaluate_run(model, molecule, block, sigma, history=history, out_dir=out_dir,
                           n_test_freq=cfg.n_test_freq, n_test_init=cfg.n_test_init,
                           test_seed=cfg.seed, device=device,
                           selection="on_resonance" if name == "best_onres.pt" else "val_loss",
                           extra={"tag": cfg.tag, "checkpoint": name,
                                  "n_freq": train_data.n_freq, "n_init": train_data.n_init,
                                  "n_pairs": len(tr)})
    return summary


# Stage protocol


def plan(cfg: Config) -> list:
    molecule = load_molecule(cfg)
    tc, fno = train_configs(cfg)
    lines = [f"recipe: {tc.epochs} epochs, batch {tc.batch_size}, AdamW lr {tc.lr:g} "
             f"wd {tc.weight_decay:g}, StepLR({tc.scheduler_step}, {tc.scheduler_gamma}) "
             f"(paper Table 2); FNO {fno.n_modes} modes, {fno.hidden_channels} channels, "
             f"{fno.n_layers} layers",
             f"loss Eqs. 19-21 with lambda = {tc.activity_lambda} and eps = {tc.activity_eps:g} "
             f"(never stated by the paper)"]
    for block, sigma in pairs(cfg, molecule):
        out = run_dir(cfg, block, sigma)
        done = finished_summary(str(out))
        state = "SKIP" if done else "TRAIN"
        detail = ""
        if done:
            ratio = done.get("strat_on_all_ratio_to_static_median")
            detail = f"  (finished, on-resonance I_FNO/I_trivial = {ratio})"
        lines.append(f"{state:>5}  block {block} sigma{sigma} -> {out}{detail}")
    lines.append(f"then build the manifest {cfg.molecule}/{cfg.tag}.json with "
                 f"surrogate.manifest.build_manifest(source='qlsgym') and verify it with "
                 f"load_manifest (provenance: tables-sha256)")
    untrained = sorted(set(range(len(molecule.blocks))) - set(resolve_blocks(cfg, molecule)))
    lines.append(f"device {cfg.resolved_device()}; blocks {untrained} fall back to the exact engine "
                 f"(Eq. 44 relabelling is not implemented)")
    return lines


def run(cfg: Config) -> dict:
    from qlsgym.surrogate.dataset import DataConfig
    from qlsgym.surrogate.manifest import build_manifest, load_manifest, manifest_path

    molecule = load_molecule(cfg)
    data = read_result("s02_data", cfg, molecule=molecule)
    tc, fno = train_configs(cfg)

    runs: dict = {}
    entries = []
    t_stage = time.time()
    for block, sigma in pairs(cfg, molecule):
        record = _s02_record(data, block, sigma)
        want = selection_config(DataConfig(**record["train"]["config"]), tc).key()
        if want != record["selection"]["key"]:
            raise RuntimeError(
                f"block {block} sigma{sigma}: the checkpoint-selection split this stage would build "
                f"({want}) is not the one s02_data reported ({record['selection']['key']}); the two "
                f"stages' TrainConfig sel_* settings have drifted apart")

        out = str(run_dir(cfg, block, sigma))
        done = finished_summary(out)
        t0 = time.time()
        if done is None:
            summary = _train_one(cfg, molecule, block, sigma, record, tc, out)
            skipped = False
        else:
            summary, skipped = done, True
        if summary.get("fingerprint") != molecule.fingerprint():
            raise RuntimeError(f"{out}/summary.json was produced on fingerprint "
                               f"{summary.get('fingerprint')}, molecule is {molecule.fingerprint()}: "
                               f"delete the run directory and retrain")
        runs[(int(block), sigma)] = out
        entries.append({"block": int(block), "sigma": sigma, "run_dir": out, "skipped": skipped,
                        "wall_seconds": round(time.time() - t0, 1),
                        "epochs": int(tc.epochs), "checkpoint": checkpoint_name(out),
                        "train_split": record["train"]["path"], "val_split": record["val"]["path"],
                        "metrics": {k: summary.get(k) for k in SUMMARY_KEYS if k in summary}})

    name = "best_onres.pt" if all(checkpoint_name(r) == "best_onres.pt" for r in runs.values()) else "best.pt"
    manifest = build_manifest(molecule, cfg.tag, runs, source="qlsgym", checkpoint_name=name,
                              note=f"replication of arXiv:2608.03702, stage {NAME}; "
                                   f"{cfg.epochs} epochs, seed {cfg.seed}")
    path = manifest.save()
    engine = load_manifest(molecule, cfg.tag, device=cfg.resolved_device())
    covered = sorted({int(b) for b, _ in manifest.checkpoints()})
    for entry in entries:
        key = f"{entry['block']},{entry['sigma']}"
        entry["provenance"] = manifest.entries[key].provenance
        entry["provenance_detail"] = manifest.entries[key].provenance_detail

    return {
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "device": cfg.resolved_device(),
        "tag": cfg.tag,
        "train_config": {k: v for k, v in vars(tc).items() if k != "fno"},
        "fno_config": vars(fno).copy(),
        "recipe_note": "paper Table 2 (Appendix B.3); Eq. 20's lambda and eps are not stated by "
                       "the paper and default to 1.0 / 1e-8 in qlsgym",
        "runs": entries,
        # manifest is a *path* and checkpoints a {"<block>,<sigma>":
        # path} mapping in manifest.entry_key format: that is the shape the
        # downstream stages look for, so they never have to guess where a
        # checkpoint is.  The detail lives beside them.
        "manifest": path,
        "checkpoints": {f"{b},{s}": p for (b, s), p in sorted(manifest.checkpoints().items())},
        "manifest_info": {
            "path": path,
            "expected_path": manifest_path(molecule.name, cfg.tag),
            "tag": manifest.tag, "source": manifest.source, "fingerprint": manifest.fingerprint,
            "checkpoint_name": name,
            "n_entries": len(manifest.entries),
            "provenance": {k: e.provenance for k, e in sorted(manifest.entries.items())},
            "loads": True,
            "blocks_covered": covered,
            "blocks_falling_back_to_exact": sorted(set(range(len(molecule.blocks))) - set(covered)),
            "fallback_note": "qlsgym does not implement the Eq. 44 parity-partner relabelling, so "
                             "the blocks it does not cover are propagated exactly "
                             "(FnoEngine.calls['exact_untrained'])",
            "engine_checkpoints": len(engine.checkpoints),
        },
        "totals": {"n_runs": len(entries),
                   "n_trained": sum(0 if e["skipped"] else 1 for e in entries),
                   "n_skipped": sum(1 for e in entries if e["skipped"]),
                   "wall_seconds": round(time.time() - t_stage, 1)},
    }
