# Locked full ThF+ RL ranking

## Question and scope

Compare PPO, categorical SAC and Double DQN trained on the **same complete 24-pair production FNO**, versus sweeping, random and physics-elimination baselines. Judge operational performance in the exact environment, not training reward or a smoke test. No hyperparameter tuning or favorable-seed selection using this holdout.

Task: ThF+ 192-state belief, 312 actions (288 Raman and 24 primitives), thermal initialization 4 K, target $\max_i s_i\ge0.98$, horizon $H=80$, overlap-penalty weight $\rho=0$ (disabled). Each applied action has reward −1 including the successful action. Raman dynamics use FNO; primitives remain exact. Both polarizations/all 12 blocks must be trained; manifest coverage is **1.0**, no partial-coverage fallbacks accepted.

## Full grid

| Controller | Train seeds | Requested transitions/model | Actual transitions/model | Objective | Final policy |
|---|---|---:|---:|---|---|
| PPO | 0–4 | 1,000,000 | 1,003,520 | γ=1, S18 critic, GAE λ=0.95, entropy coefficient 0.01 | sampled categorical |
| Double DQN | 0–4 | 1,000,000 | 1,000,064 | γ=1, S18 branch expectation | greedy |
| Discrete SAC | 0–4 | 1,000,000 | 1,000,064 | γ=0.99, soft S18 target | sampled categorical |
| Sweeping/random/physics elimination | none | 0 | 0 | non-learning | fixed reference policies |

All learned networks: belief input $\sqrt{s}$, two 256-wide tanh hidden layers, LR $3\times10^{-4}$, 128 parallel environments. PPO rollout length32, four epochs/eight minibatches, clip0.2. DDQN/SAC replay500,000, batch256, warmup10,240, target Polyak0.005; four optimizer steps per128 collected transitions. DDQN epsilon1→0.02 over720,000 transitions. SAC target entropy0.5log312, automatic temperature: initial raw-reward $\alpha=0.05$, normalized $\alpha=0.05/H=0.000625$ because critic rewards are divided by $H$. Historical smoke used 0.05 after reward division, amplifying entropy by80; never pool it with this grid.

Quantum/Bellman critic target for branch $k\in\{0,1\}$:

$$y(s,a)=\sum_k\pi_k(s,a)\left[r_k(s,a)+\gamma c_k V(s'_k)\right],\qquad c_k=\mathbf1\{\text{branch nonterminal and budget remains}\}.$$

DDQN selects next action with online Q and evaluates target Q; SAC uses $V(s)=\sum_a\pi(a|s)[\min(Q_1,Q_2)-\alpha\log\pi(a|s)]$. PPO uses the environment's branch-expectation critic residual inside GAE; actions and continuing trajectories are still sampled, not an analytic full policy-gradient expectation. S18 weights are physical probabilities, not tunable pseudo-probabilities. All are predictions **under the learned model**; model bias is not removed.

Upstream `value_target="qmdp"` is one-step actor-critic and ignores `gae_lambda`. New `qmdp_gae` for final PPO uses $\delta_t=y_t-V(s_t)$, $A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}$ along the sampled trajectory. Branch masks inside $y_t$ stop each branch; sampled $d_t$ stops GAE across resets. Legacy behavior stays unchanged; λ=0 reduces to one-step qmdp. Tests check this equivalence and reset boundaries.

The belief-only stationary networks cannot represent the general finite-horizon value $V(s,h)$; time-to-go $h$ is not an observation in this study. Horizon continuation masks are correct but do not eliminate state aliasing. Interpret ranking as this stationary-policy implementation, not the optimal finite-horizon solution.

## Evaluation and metrics

Each of15 trained policies and3 baselines gets5000 exact +5000 FNO rollouts, common eval seed20001, chunk size128. Chunk seeds depend only on eval seed and chunk index, never training seed or agent. Training seeds0–4 and PPO's sole end-of-training diagnostic seed4242 are separate; no selection on final outcomes. Exact policies act on exact belief states; FNO policies act on FNO belief states. Thus this measures simulator transfer, not deployment with an FNO state-estimator applied to experimental observations.

Let $T_j$ be the first successful pulse count; $T_j=\infty$ if target is never reached. A policy returning no action without success is a failure. Define $L_j=\min(T_j,H)$ (all failures count as80), and $S_j=\mathbf1\{T_j\le H\}$.

$$\widehat f=1-\frac1N\sum_jS_j,\qquad \widehat A=\frac1N\sum_jL_j,\qquad C(h)=\frac1N\sum_j\mathbf1\{T_j\le h\}.$$

- **Unfinished/failure rate** $\widehat f$: lower better, reliable target attainment; historically called capped episodes.
- **Average actions** $\widehat A$: lower better, failures penalized80; historically restricted mean pulses. This is not successful-only mean; `actual_lengths` separately records physically applied actions if a policy stops early.
- **Finished-episode curve** $C(h)$: higher better at every budget; failures never enter it, including at80.
- **P85**: 85th percentile of $L_j$, lower better. **CVaR10**: mean of worst10% of $L_j$, lower better. Both saturate80 when enough episodes fail, reducing diagnostic power.
- **Transfer gaps** $A_{\mathrm{exact}}-A_{\mathrm{FNO}}$ and $f_{\mathrm{exact}}-f_{\mathrm{FNO}}$: positive means surrogate evaluation optimistic; negative means pessimistic. Near0 alone does not imply good scores.

Report each learned seed and equal-weight average overfive seeds. Learned mean/failure CIs use10,000 bootstrap resamples ofseed means; baseline mean CIs bootstrap episodes and failure CIs use Wilson binomial bounds. These are exploratory estimates: five seeds are limited, common random numbers do not make different policies' quantum outcomes identical, and endpoint bootstrap intervals can degenerate. No significance or statistical dominance claim based on sorted means. Operational ordering: exact failure first, then exact average actions. SAC’s discount/entropy objective differs from PPO/DDQN; do not call this an equal-objective training comparison.

## Execution and publication

The authorized server pipeline runs in tmux `thf_final_study`. It waits for the two existing FNO queues, validates all24 completed120-epoch checkpoints/audits, assembles one full manifest, runs the two full paper-style tests, evaluates baselines, then executes15 learned jobs across two sequential GPU queues. No competing training jobs share a GPU. Each job has an individual log; failed/partial checkpoints are never silently overwritten. A failed pipeline stops, leaving logs; learning jobs are not optimizer-state resumable, so an interrupted model needs inspection/new output folder.

```bash
export QLSGYM_WORK=/home/htseng/Downloads/qlsgym_work
export STUDY_PYTHON=/home/htseng/anaconda3/envs/qlsgym/bin/python
tmux new-session -d -s thf_final_study \
  'bash scripts/run_thf_final_study.sh > /home/htseng/Downloads/qlsgym_work/logs/final_study.log 2>&1'
```

Require exactly18 records, seeds0–4 per learned controller, identical task/library/manifest/hyperparameter contracts, full coverage, N5000, eval seed/batch unchanged, and recomputed per-episode metrics before declaring a final ranking. The finalizer regenerates figures, summaryJSON/MD and README/results report, runs unit tests, and commits/pushes only these study artifacts to HHTseng/qlsgym `FNO_RL_agents`. Weights/caches/logs stay outside Git.

Training queues began2026-09-16 22:10EDT; the two completed calibration blocks took~50–52minutes each. Remaining22 FNO pairs across two GPUs initially estimated~9–10hours; larger blocks and data generation can increase this. RL wall time must be estimated from the first full job, not extrapolated from a tiny low-width pilot. Check `final_study.log`, `final_gpu0.log`, `final_gpu1.log`, and per-job logs. README keeps pending status until the strict aggregator passes.
