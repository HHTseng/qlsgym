"""The screening stage must select profiles without exact-simulator metrics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_thf_next_ppo.py"
SPEC = importlib.util.spec_from_file_location("summarize_thf_next_ppo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_screen_selection_uses_fno_only(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    contract = {
        "molecule_fingerprint": "molecule",
        "library_tag": "library",
        "manifest_sha256": "manifest",
        "p_target": 0.98,
        "max_pulses": 80,
        "rho": 0.0,
    }
    for rank, profile in enumerate(MODULE.SCREEN_PROFILES):
        for seed in MODULE.SCREEN_SEEDS:
            failure = 0.20 + 0.01 * rank
            record = {
                "job": {"ppo_profile": profile, "train_seed": seed},
                "contract": contract,
                "training": {"config": {"seed": seed, "total_steps": 100}},
                "evaluation": {
                    "fno": {"unfinished_fraction": failure, "average_actions": 40 + rank}
                },
                "wall_clock_s": 1.0,
            }
            (runs / f"{profile}_{seed}.json").write_text(json.dumps(record))

    output = tmp_path / "selection.json"
    MODULE.select(SimpleNamespace(runs=runs, output=output))
    selected = json.loads(output.read_text())
    assert selected["promoted_profiles"] == list(MODULE.SCREEN_PROFILES[:2])
    assert all("exact_failure_mean" not in row for row in selected["rows"])
