# ThF+ FNO and RL results

This report is regenerated from tracked numerical artifacts. Training weights/caches remain on the server. It adapts metrics from [arXiv:2608.03702](https://arxiv.org/pdf/2608.03702), not its molecule or hardware.

## Main interpretation

ThF+ has 192 molecular states, 12 Hamiltonian blocks, seven retained motional levels, and 312 controls (288 Raman + 24 exact primitives). Purification target is 0.98, horizon 80, temperature 4 K, and maximum Raman duration 6 ms. FNO replaces Raman propagation; primitives remain exact.

The early 8192-transition pilot did not establish a learned advantage: exact physics elimination achieved 51.455 average actions and 28% failure, versus PPO/SAC ~74 actions and 85–86.5% failure; sweeping failed 100%. Its surrogate incorrectly predicted 100% failure for physics elimination. Historical SAC also used an inconsistent entropy scale, corrected in the final grid.

Those are feasibility results only. A final learned ranking requires the locked complete grid and exact hold-out evaluation.

## Production FNO accuracy

Completed checkpoints with independent held-out tests: **24/24**. All completed models trained for 120 epochs; checkpoints are preselected `best_onres.pt`, never test-selected.

| Block | σ | On-resonance median infidelity, diffuse ↓ | On-resonance median, control mixture ↓ | Control-mixture P95 ↓ | Static/no-change median |
|---:|:---:|---:|---:|---:|---:|
| 0 | + | 0.00622 | 0.02051 | 0.02180 | 0.17603 |
| 1 | + | 0.00594 | 0.02051 | 0.02131 | 0.19463 |
| 2 | + | 0.00424 | 0.01245 | 0.01309 | 0.14371 |
| 3 | + | 0.00388 | 0.01345 | 0.01431 | 0.15958 |
| 4 | + | 0.00207 | 0.00613 | 0.00646 | 0.10285 |
| 5 | + | 0.00205 | 0.00647 | 0.00685 | 0.11909 |
| 6 | + | 0.00117 | 0.00360 | 0.00411 | 0.06656 |
| 7 | + | 0.00111 | 0.00346 | 0.00382 | 0.06888 |
| 8 | + | 0.00049 | 0.00145 | 0.00171 | 0.04142 |
| 9 | + | 0.00045 | 0.00150 | 0.00169 | 0.04153 |
| 10 | + | 0.00020 | 0.00040 | 0.00046 | 0.06502 |
| 11 | + | 0.04455 | 0.05790 | 0.07685 | 0.06502 |
| 0 | - | 0.00578 | 0.01864 | 0.01937 | 0.15703 |
| 1 | - | 0.00817 | 0.02600 | 0.02715 | 0.17144 |
| 2 | - | 0.00368 | 0.01141 | 0.01224 | 0.12984 |
| 3 | - | 0.00429 | 0.01362 | 0.01443 | 0.14811 |
| 4 | - | 0.00202 | 0.00648 | 0.00691 | 0.09330 |
| 5 | - | 0.00174 | 0.00595 | 0.00630 | 0.10941 |
| 6 | - | 0.00113 | 0.00316 | 0.00358 | 0.05713 |
| 7 | - | 0.00104 | 0.00322 | 0.00368 | 0.05737 |
| 8 | - | 0.00044 | 0.00131 | 0.00167 | 0.04071 |
| 9 | - | 0.00046 | 0.00129 | 0.00160 | 0.04093 |
| 10 | - | 0.00014 | 0.00036 | 0.00126 | 0.05233 |
| 11 | - | 0.00016 | 0.00034 | 0.00038 | 0.05233 |

![Production FNO independent errors](../results/thf_fno_blocks/thf_production_accuracy.png)

Each row uses 32 new frequencies per stratum × 128 initial states, seeds 20260916/17, with 200 single-pulse time samples. Values average over initial states and pulse times before computing frequency percentiles. Diffuse and peaked input ensembles are not interchangeable.

Accuracy on resonance improved strongly over the 30-epoch pilot, but a few-percent error floor remains on peaked beliefs. Off-resonance static predictions can outperform FNO; small unconditional error does not guarantee accurate normalized measurement branches or closed-loop purification.

### Preliminary CPU audit: block 0, σ=+

32 new frequencies/stratum × 128 initial states; device `cpu`. Reference: exact PyTorch, not CUDA-Q.

- Representative resonant trajectory: time-average population infidelity 0.0073515.
- Zero-time identity total-variation error: 0.052159 (ideal 0).
- Near-pure input mean/P95 infidelity: 0.0041929/0.005594.
- Near-pure conditional-branch P95 TV: 0.21304, excluding exact branch masses <10⁻³.
- Input-mixture linearity TV: 0.04103 (ideal 0).

![Paper-style FNO accuracy](../results/thf_fno_preview/rlprod120v2_sp_block0_accuracy.png)

![Near-pure measurement-branch errors](../results/thf_fno_preview/rlprod120v2_sp_block0_branch_errors.png)

Exact-build/FNO speedup range: 0.32–11.15×; cached-exact/FNO: 0.00153–0.0171×. These compare different amortization regimes, not the paper's CUDA-Q benchmark.

![Propagation timings](../results/thf_fno_preview/rlprod120v2_sp_block0_timing.png)

### Full paper-style audit: block 1, σ=-

100 new frequencies/stratum × 500 initial states; device `cuda:1`. Reference: exact PyTorch, not CUDA-Q.

- Representative resonant trajectory: time-average population infidelity 0.016159.
- Zero-time identity total-variation error: 0.067162 (ideal 0).
- Near-pure input mean/P95 infidelity: 0.004456/0.0065554.
- Near-pure conditional-branch P95 TV: 0.50592, excluding exact branch masses <10⁻³.
- Input-mixture linearity TV: 0.027968 (ideal 0).

![Paper-style FNO accuracy](../results/thf_fno_validation/rlprod120v2_sm_block1_accuracy.png)

![Near-pure measurement-branch errors](../results/thf_fno_validation/rlprod120v2_sm_block1_branch_errors.png)

Exact-build/FNO speedup range: 0.91–103.15×; cached-exact/FNO: 0.0145–0.141×. These compare different amortization regimes, not the paper's CUDA-Q benchmark.

![Propagation timings](../results/thf_fno_validation/rlprod120v2_sm_block1_timing.png)

### Full paper-style audit: block 0, σ=+

100 new frequencies/stratum × 500 initial states; device `cuda:0`. Reference: exact PyTorch, not CUDA-Q.

- Representative resonant trajectory: time-average population infidelity 0.0073515.
- Zero-time identity total-variation error: 0.052159 (ideal 0).
- Near-pure input mean/P95 infidelity: 0.0041929/0.005594.
- Near-pure conditional-branch P95 TV: 0.21304, excluding exact branch masses <10⁻³.
- Input-mixture linearity TV: 0.04103 (ideal 0).

![Paper-style FNO accuracy](../results/thf_fno_validation/rlprod120v2_sp_block0_accuracy.png)

![Near-pure measurement-branch errors](../results/thf_fno_validation/rlprod120v2_sp_block0_branch_errors.png)

Exact-build/FNO speedup range: 0.79–85.98×; cached-exact/FNO: 0.0124–0.116×. These compare different amortization regimes, not the paper's CUDA-Q benchmark.

![Propagation timings](../results/thf_fno_validation/rlprod120v2_sp_block0_timing.png)

The CPU preview finds 5.2% zero-time identity TV, 4.1% input-linearity TV, and conditional-branch P95 TV ≈21% despite near-pure mean joint infidelity ≈0.0042. Large MRE spikes are driven by small positive true populations; infidelity and absolute/TV diagnostics give complementary context. Low joint error is not a closed-loop certification.

For fixed-frequency state batches, exact propagation amortizes one eigendecomposition over many input states. Fresh frequency batches reach up to 11.2× FNO speedup, but cached exact is 58–655× faster across tested workloads. These CPU timings do not predict GPU timings; full audits log those separately. A compact fixed312-action problem can favor exact tables; FNO's stronger motivation is larger or changing/continuous control sets.

## Final exact-simulator ranking

Complete locked grid: 15 learned policies and three baselines. Each policy has 5000 exact and 5000 surrogate rollouts.

| Controller | Training seeds | Exact average actions ↓ (95% CI) | Exact failure ↓ (95% CI) | FNO average actions | FNO failure |
|---|---:|---:|---:|---:|---:|
| Physics elimination | 0 | 51.49 [50.78, 52.22] | 29.36% [28.11, 30.64] | 69.20 | 71.76% |
| Discrete SAC | 5 | 66.83 [62.10, 73.37] | 71.69% [61.18, 86.19] | 68.19 | 73.44% |
| PPO | 5 | 69.07 [67.17, 70.97] | 76.75% [73.14, 80.36] | 69.80 | 77.21% |
| Random | 0 | 74.57 [74.11, 75.01] | 85.62% [84.62, 86.57] | 76.18 | 88.46% |
| Sweeping | 0 | 79.97 [79.94, 80.00] | 99.92% [99.79, 99.97] | 79.93 | 99.82% |
| Double DQN | 5 | 80.00 [80.00, 80.00] | 100.00% [100.00, 100.00] | 79.86 | 99.60% |

![Final ThF+ RL ranking](../results/thf_rl_final/thf_final_ranking.png)

Order is descriptive: exact failure rate, then average actions. PPO/DDQN use γ=1; SAC uses γ=0.99 with an entropy bonus, so these are operational performance scores, not equal training objectives.

Learned-policy intervals bootstrap five training-seed means; baseline mean intervals bootstrap rollouts and failure intervals use Wilson bounds. Five seeds do not establish statistical dominance. Inspect individual JSONs and the FNO-to-exact gap before interpreting a learned advantage.


## What to improve next

1. If near-pure/branch errors remain large, prioritize FNO fidelity: train on exact collected beliefs and simplex vertices, enforce τ=0 identity and input linearity, and validate measurement-conditioned errors. More epochs alone may not remove softmax leakage.
2. Add a time-to-go feature to all critics/actors for the true finite-horizon task; current belief-only stationary policies do not distinguish identical beliefs with different remaining budgets.
3. Separate representation/control effects from RL optimization: test equal-action sweeping schedules, physics-informed action priors, and exact-trained RL controls. Current fixed-order sweeping uses only the first 80/312 actions before timeout.
4. Only after surrogate validity is checked, use a separate validation-seed Optuna search for PPO entropy/learning rate, DDQN exploration/target timescale, and SAC temperature/target entropy. Do not tune on the final 5000-rollout holdout.
5. Repeat matched-γ comparisons or report performance/objective differences explicitly; SAC's γ=0.99 soft objective is not the γ=1 shortest-path objective.

## Provenance and scope

Native qlsgym branch starts from main `2a7ee186f09c54b78d5987bcd0a6bb2399749e28`. FNO/RL experiment code and historical numerical records were selectively ported from HHTseng/rl_qls_paper_replication `FNO_RL_agents`, commit `d306d34`; unrelated molecule experiments/history were not merged.

Production status: complete. Missing pairs: `[]`. See [metric definitions](FNO_PAPER_METRICS.md) and [locked full study](THF_FINAL_RL_STUDY.md).
