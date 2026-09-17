#!/usr/bin/env python
"""Write the ThF-only results report from numerical artifacts, not log guesses."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    audit_path = root / "results/thf_fno_blocks/rlprod120v2.json"
    audit = json.loads(audit_path.read_text())
    rows = audit["blocks"]
    complete = audit["completed_pairs"] == 24
    fno_lines = ["## Production FNO accuracy", "",
                 f"Completed checkpoints with independent held-out tests: **{len(rows)}/24**. All completed models trained for 120 epochs; checkpoints are preselected `best_onres.pt`, never test-selected.", "",
                 "| Block | σ | On-resonance median infidelity, diffuse ↓ | On-resonance median, control mixture ↓ | Control-mixture P95 ↓ | Static/no-change median |", "|---:|:---:|---:|---:|---:|---:|"]
    for row in rows:
        d, c = row["heldout"]["diffuse_alpha1"], row["heldout"]["control_mix_alpha_minus2"]
        fno_lines.append(f"| {row['block']} | {row['sigma']} | {d['on_on_median']:.5f} | {c['on_on_median']:.5f} | {c['on_on_p95']:.5f} | {c['on_on_static_median']:.5f} |")
    fno_lines += ["", "![Production FNO independent errors](results/thf_fno_blocks/thf_production_accuracy.png)", "",
                  "Each row uses 32 new frequencies per stratum × 128 initial states, seeds 20260916/17, with 200 single-pulse time samples. Values average over initial states and pulse times before computing frequency percentiles. Diffuse and peaked input ensembles are not interchangeable.", "",
                  "Accuracy on resonance improved strongly over the 30-epoch pilot, but a few-percent error floor remains on peaked beliefs. Off-resonance static predictions can outperform FNO; small unconditional error does not guarantee accurate normalized measurement branches or closed-loop purification."]
    paper = []
    for folder in ("thf_fno_preview", "thf_fno_validation"):
        for path in sorted((root / "results" / folder).glob("*_summary.json")):
            item = json.loads(path.read_text())
            stem = path.name.removesuffix("_summary.json")
            t = item["trajectory"]
            paper += ["", f"### {'Preliminary CPU audit' if folder.endswith('preview') else 'Full paper-style audit'}: block {item['block']}, σ={item['sigma']}", "",
                      f"{item['n_test_freq_per_stratum']} new frequencies/stratum × {item['n_initial_states']} initial states; device `{item['device']}`. Reference: exact PyTorch, not CUDA-Q.", "",
                      f"- Representative resonant trajectory: time-average population infidelity {t['trajectory_time_average_infidelity']:.5g}.",
                      f"- Zero-time identity total-variation error: {t['tau0_identity_tv']:.5g} (ideal 0).",
                      f"- Near-pure input mean/P95 infidelity: {t['near_pure_vertex_mean_infidelity']:.5g}/{t['near_pure_vertex_p95_infidelity']:.5g}.",
                      f"- Near-pure conditional-branch P95 TV: {t['near_pure_conditional_tv_p95_true_mass_ge_1e-3']:.5g}, excluding exact branch masses <10⁻³.",
                      f"- Input-mixture linearity TV: {t['input_linearity_mean_tv']:.5g} (ideal 0).", "",
                      f"![Paper-style FNO accuracy](results/{folder}/{stem}_accuracy.png)"]
            branch_plot = root / "results" / folder / f"{stem}_branch_errors.png"
            if branch_plot.exists():
                paper += ["", f"![Near-pure measurement-branch errors](results/{folder}/{stem}_branch_errors.png)"]
            if "timing" in item:
                cold = [r["cold_exact_speedup"] for r in item["timing"]]
                cached = [r["cached_exact_speedup"] for r in item["timing"]]
                paper += ["", f"Exact-build/FNO speedup range: {min(cold):.2f}–{max(cold):.2f}×; cached-exact/FNO: {min(cached):.3g}–{max(cached):.3g}×. These compare different amortization regimes, not the paper's CUDA-Q benchmark.", "",
                          f"![Propagation timings](results/{folder}/{stem}_timing.png)"]
    if paper:
        paper += ["", "The CPU preview finds 5.2% zero-time identity TV, 4.1% input-linearity TV, and conditional-branch P95 TV ≈21% despite near-pure mean joint infidelity ≈0.0042. Large MRE spikes are driven by small positive true populations; infidelity and absolute/TV diagnostics give complementary context. A few-percent joint error is not a closed-loop certification.", "",
                  "For the preview's fixed-frequency state batches, fresh exact propagation can amortize one eigendecomposition over many input states; FNO becomes slower at batch32. For fresh frequency batches FNO reaches ≈8.9× speedup, but cached exact remains ≈90–1000× faster across tested workloads. These CPU timings do not predict GPU timing; the full pipeline logs GPU audits separately. A compact fixed312-action problem can favor exact tables; FNO's stronger motivation is larger or changing/continuous control sets."]
    final_path = root / "results/thf_rl_final/summary.md"
    rl = final_path.read_text() if final_path.exists() else "## Final RL ranking\n\n**Pending, not a completed ranking.** The tmux pipeline waits for all 24 production models, then runs three baselines plus PPO, categorical SAC and DDQN × seeds 0–4 at ~1M transitions each. Every controller has 5000 exact and 5000 FNO rollouts. No pilot/smoke scores are pooled into this ranking."
    # README uses repository-relative images; the docs report needs ../ paths.
    report = ["# ThF+ FNO and RL results", "",
              "This report is regenerated from tracked numerical artifacts. Training weights/caches remain on the server. It adapts metrics from [arXiv:2608.03702](https://arxiv.org/pdf/2608.03702), not its molecule or hardware.", "",
              "## Main interpretation", "",
              "ThF+ has 192 molecular states, 12 Hamiltonian blocks, seven retained motional levels, and 312 controls (288 Raman + 24 exact primitives). Purification target is 0.98, horizon 80, temperature 4 K, and maximum Raman duration 6 ms. FNO replaces Raman propagation; primitives remain exact.", "",
              "The early 8192-transition pilot did not establish a learned advantage: exact physics elimination achieved 51.455 average actions and 28% failure, versus PPO/SAC ~74 actions and 85–86.5% failure; sweeping failed 100%. Its surrogate incorrectly predicted 100% failure for physics elimination. Historical SAC also used an inconsistent entropy scale, corrected in the final grid.", "",
              "Those are feasibility results only. A final learned ranking requires the locked complete grid and exact hold-out evaluation.", ""]
    report += fno_lines + paper + ["", rl, "", "## What to improve next", "",
               "1. If near-pure/branch errors remain large, prioritize FNO fidelity: train on exact collected beliefs and simplex vertices, enforce τ=0 identity and input linearity, and validate measurement-conditioned errors. More epochs alone may not remove softmax leakage.",
               "2. Add a time-to-go feature to all critics/actors for the true finite-horizon task; current belief-only stationary policies do not distinguish identical beliefs with different remaining budgets.",
               "3. Separate representation/control effects from RL optimization: test equal-action sweeping schedules, physics-informed action priors, and exact-trained RL controls. Current fixed-order sweeping uses only the first 80/312 actions before timeout.",
               "4. Only after surrogate validity is checked, use a separate validation-seed Optuna search for PPO entropy/learning rate, DDQN exploration/target timescale, and SAC temperature/target entropy. Do not tune on the final 5000-rollout holdout.",
               "5. Repeat matched-γ comparisons or report performance/objective differences explicitly; SAC's γ=0.99 soft objective is not the γ=1 shortest-path objective.", "",
               "## Provenance and scope", "",
               "Native qlsgym branch starts from main `2a7ee186f09c54b78d5987bcd0a6bb2399749e28`. FNO/RL experiment code and historical numerical records were selectively ported from HHTseng/rl_qls_paper_replication `FNO_RL_agents`, commit `d306d34`; unrelated molecule experiments/history were not merged.", "",
               f"Production status: {'complete' if complete else 'still training'}. Missing pairs: `{audit['missing_pairs']}`. See [metric definitions](FNO_PAPER_METRICS.md) and [locked full study](THF_FINAL_RL_STUDY.md)."]
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/THF_RESULTS.md").write_text(("\n".join(report) + "\n").replace("](results/", "](../results/"))
    readme = root / "README.md"
    text = readme.read_text()
    start, stop = "<!-- FNO_RESULTS_START -->", "<!-- FNO_RESULTS_END -->"
    if text.count(start) != 1 or text.count(stop) != 1:
        raise ValueError("missing unique FNO-results markers")
    readme.write_text(text.split(start)[0] + start + "\n" + "\n".join(fno_lines + paper) + "\n" + stop + text.split(stop)[1])
    print("updated docs/THF_RESULTS.md and README FNO section from artifacts")


if __name__ == "__main__":
    main()
