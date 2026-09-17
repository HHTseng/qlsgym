# Completed ThF⁺ study: status and interpretation

Status checked 17 September 2026. Numerical sources: [locked summary](../results/thf_rl_final/summary.json), [individual runs](../results/thf_rl_final/runs), [all-block accuracy](../results/thf_fno_blocks/rlprod120v2.json), and [full GPU audits](../results/thf_fno_validation). Historical smoke runs are excluded.

## Completion and verification

All 24 production block/polarization FNO models trained for 120 epochs. Full independent GPU audits cover blocks 0/+ and 1/− with 100 frequencies per stratum, 500 initial states and 200 pulse-time samples; routine independent accuracy tests cover all 24 models. The full RL grid comprises PPO, categorical SAC and DDQN × seeds 0–4, plus physics elimination, random and sweeping baselines. Each controller has 5000 exact and 5000 FNO rollouts: 180,000 evaluation episodes overall.

The remaining FNO queue completed by 07:12 EDT; learned-agent jobs ran from 07:55 to approximately 14:27 EDT on 17 September. Training plus evaluation took roughly 47–54 minutes per learned job with two GPU queues. The study's commit was created at 14:27 EDT after **166 passed, 17 skipped, three deselected** tests. No study tmux sessions remained and both GPUs were idle at the 14:47 status check. There is no outstanding completion estimate: this study is finished. The automatic push failed on GitHub DNS resolution; its committed results were transferred locally for publication, without rerunning experiments.

The aggregation validator checks all 18 expected records, consistent checkpoint hashes for 24 models, identical task/evaluation contracts, actual training budgets and metrics recomputed from per-episode arrays. PPO collected 1,003,520 transitions per seed; SAC/DDQN collected 1,000,064 because vectorized collection overshoots the requested one million slightly.

This review reran the validator locally against all 180,000 episode scores and passed 16 targeted aggregation, PPO-GAE, evaluation and off-policy tests. No simulator retraining was required.

## What the operational scores mean

Let $H=80$, $T_j$ be the first successful action count in episode $j$ ($T_j=\infty$ for failure), and $L_j=\min(T_j,H)$. For $N$ evaluation episodes,

$$f=\frac1N\sum_j\mathbf1[T_j>H],\qquad A=\frac1N\sum_jL_j,\qquad C(h)=\frac1N\sum_j\mathbf1[T_j\le h].$$

Failure $f$ and failure-penalized average actions $A$ are **lower-better**; completion curve $C(h)$ is **higher-better**. Failure scores 80 actions even if an episode stops early; actual executed lengths are separately retained. Thus $A$ is neither a successful-only average nor an unconditional wall-clock cost. P85 and worst-10% CVaR are 80 for every controller in this study and cannot discriminate performance here. Prioritize $f$, then $A$ and the completion curve.

| Controller | Exact average actions ↓ | Exact failure ↓ | Exact success ↑ | FNO failure |
|---|---:|---:|---:|---:|
| Physics elimination | 51.49 | 29.36% | 70.64% | 71.76% |
| SAC | 66.83 | 71.69% | 28.31% | 73.44% |
| PPO | 69.07 | 76.75% | 23.25% | 77.21% |
| Random | 74.57 | 85.62% | 14.38% | 88.46% |
| Sweeping | 79.97 | 99.92% | 0.08% | 99.82% |
| DDQN | 80.00 | 100.00% | 0.00% | 99.60% |

![Exact ranking, uncertainty and completion curves](../results/thf_rl_final/thf_final_ranking.png)

Physics elimination beats the best learned average by 15.34 actions and 42.33 percentage points of failure. SAC improves over random by 7.73 actions and 13.93 percentage points; PPO improves by 5.50 actions and 8.87 points. These are descriptive operational differences, not established algorithmic superiority: SAC uses $\gamma=0.99$ and an entropy bonus, while PPO/DDQN use $\gamma=1$. All learned runs use branch-expected S18-style targets; this study does not estimate an S17-versus-S18 effect.

Learned confidence intervals bootstrap five training-seed means, not 25,000 independently trained policies. SAC failure CI is 61.18–86.19%; PPO is 73.14–80.36%. Their overlap, only five seeds, and different training objectives preclude a definitive SAC–PPO ranking. Baseline uncertainty comes from rollout bootstrap/Wilson bounds and has a different scope. DDQN's degenerate bootstrap CI at 100% means all five observed seed estimates coincide, not a theorem of universal failure.

## Seed-level diagnostics

| Seed | PPO exact actions / failure | SAC exact actions / failure | DDQN exact actions / failure |
|---:|---:|---:|---:|
| 0 | 70.78 / 77.72% | 63.69 / 65.20% | 80 / 100% |
| 1 | 66.38 / 71.54% | 67.26 / 71.44% | 80 / 100% |
| 2 | 71.76 / 82.66% | 60.03 / 56.06% | 80 / 100% |
| 3 | 66.68 / 72.44% | 63.37 / 66.26% | 80 / 100% |
| 4 | 69.75 / 79.38% | 79.82 / 99.48% | 80 / 100% |

SAC seed 4's last training minibatch policy entropy is **0.13 nats**, versus 3.12–4.00 for the other seeds (uniform over 312 actions would be $\log312\approx5.74$). This is consistent with policy collapse, but does not prove repeated-action trapping without action traces. Its cumulative training success is about 14.1%, illustrating why exploratory historical training performance does not represent the final deployed policy. Do not discard this seed or report seed 2 as the algorithm's score.

DDQN has 10.6–11.7% cumulative training success but zero exact success for its final greedy policies. Exploration and a changing training policy differ from deployment. Possible causes include poor greedy action selection, value-learning instability, inadequate exploration coverage, surrogate exploitation, and finite-horizon aliasing; these are hypotheses, not diagnoses established by the current artifacts. PPO's last-update episode statistics cover only that rollout/update, unlike off-policy cumulative training counts; those training summaries must not be compared directly.

Sweeping succeeds in **four of 5000** exact episodes. Its current fixed-order schedule tries at most the first 80 of 312 actions, so it is not an optimized spectroscopy sweep or a universal lower bound. Zero successes in a previous 200-episode smoke run is compatible with the present rate: $(1-0.0008)^{200}\approx0.852$. There is no numerical contradiction.

## Surrogate fidelity and transfer

For physics elimination, exact minus FNO failure is **−42.40 percentage points**, and exact minus FNO average actions is **−17.71**. This is a large *pessimistic* surrogate error, not an optimistic learned-agent artifact. SAC/PPO's smaller mean failure gaps (−1.75/−0.46 points) do not certify model accuracy: different policies visit different beliefs and controls, and mostly failing policies provide weak coverage of successful purification paths. Exact evaluation supplies exact beliefs to the policy; it does not validate a deployed FNO-only state estimator receiving real measurements.

All-block held-out control-mixture median infidelity identifies **block 11, σ=+, at 0.05790**, versus its static/no-change baseline 0.06502; diffuse infidelity is 0.04455. Its opposite polarization reaches 0.000344 control-mixture infidelity. This outlier warrants a separate audit. The server training summary confirms 120 completed epochs but selects its best on-resonance checkpoint at **epoch 26** (validation infidelity 0.03124). Simply extending training is therefore not justified by the present evidence. The two predeclared full GPU audits are not an exhaustive audit of the worst blocks discovered afterward.

For blocks 0/+ and 1/−, full GPU audits find zero-time identity TV errors 0.0522/0.0672 and input-mixture linearity TV errors 0.0410/0.0280 (both ideally zero). Near-pure conditional-branch P95 TV reaches 0.213/0.506, restricted to exact branch masses at least $10^{-3}$ in the selected resonant trajectory test. These are not typical errors across every control. Normalization amplifies small joint-population errors when branch probability is small: schematically, posterior error scales as $O(\epsilon/\pi_k)$ for joint error $\epsilon$. Low joint infidelity alone is insufficient for repeated measurement-conditioned purification. S18 expectation reduces branch-sampling variance, not surrogate bias.

Timing also needs qualification. Fresh 128-frequency batches show 86×/103× cold-exact-to-FNO speedups in the two audited blocks. Cached exact propagation is instead about 7–80× faster across tested workloads; fixed-frequency state batches amortize one exact propagator. These single-block PyTorch/RTX-2080-Ti timings are neither an end-to-end RL benchmark nor a reproduction of the paper's CUDA-Q/hardware ratio. A fixed 312-action library may favor exact tables; FNO is more motivated by changing or much larger continuous-control sets.

## Prioritized next comparisons — not launched

1. **Isolate learning from model error:** train matched PPO/SAC/DDQN controls with exact action tables, equal budgets and separate validation seeds. Record per-checkpoint completion and action traces. This distinguishes surrogate bias from optimization failure; current exact evaluation alone cannot do so.
2. **Resolve failure modes before large tuning:** compare DDQN greedy versus small-$\epsilon$ evaluation; inspect chosen-action repetition and Q gaps. For SAC, retain intermediate checkpoints, entropy curves and validation-only selection; test target entropy/temperature settings against seed collapse. Keep final holdout untouched.
3. **Audit and repair the surrogate:** independently test block 11/+; inspect its sampling, resonance coverage and learning curves. Include actual visited beliefs, simplex vertices and branch-conditioned validation. Enforce zero-time identity and population-map linearity, ideally through a nonnegative column-stochastic operator $G_\theta(s,\omega,\tau)=\widehat T_\theta(\omega,\tau)s$ with the correct identity embedding at $\tau=0$. More epochs alone is not the first remedy.
4. **Represent the finite horizon:** append normalized time-to-go $(H-t)/H$ to every actor/critic. Current stationary belief-only policies cannot distinguish equal beliefs with different remaining action budgets. Apply consistently to all learned agents and rerun a newly identified generation.
5. **Strengthen controls:** add coverage-aware/optimized sweeping and physics-informed action priors. Evaluate hybrids using physics elimination as fallback or imitation initialization rather than expecting random exploration to rediscover strong structure.
6. **Then tune with Optuna:** separate validation seeds, equal resource budgets, staged pruning, and objectives prioritizing failure then average actions. Tune PPO entropy/lr, DDQN exploration/update ratio and SAC entropy settings; compare matched discounts or retain explicit objective labels. Use new final seeds for any tuned-versus-untuned claim.

No new training, tuning or full-scale experiments were started during this status review.
