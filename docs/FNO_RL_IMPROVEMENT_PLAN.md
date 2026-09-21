# FNO and RL improvement plan for ThF⁺

I interpret “RNO” as the current FNO/neural-operator dynamics model.

## Execution revision — 21 September 2026

The completed `FNO_RL_optuna` work changes the execution order without changing
the model diagnosis. Focused SAC tuning reduced exact failure from 69.41% to
57.07%, while optimized PPO reached 41.78%. These are stronger controls than
the original plan had available.

The revised, single-GPU sequence is:

1. Reuse the selected PPO and focused-SAC settings for a 100k-transition exact
   diagnosis, crossing agent with belief-only versus remaining-budget input.
2. Promote the best exact profile to five fresh one-million-transition seeds.
   This is the causal RL control for every later surrogate comparison.
3. Audit the downloaded manifest before retraining. The audit is manifest- and
   hash-aware, covers all 24 pairs, and reports structural/branch metrics by
   frequency stratum and population ensemble.
4. Pilot a transfer-column residual operator on blocks 0± and 1±. It enforces
   input linearity and τ=0 identity by construction and uses static identity for
   clearly off-resonant controls. A full 24-pair retrain is allowed only if the
   four-pair pilot materially improves the declared gates.
5. Train RL with the improved manifest using the retained Optuna settings, then
   compare exact-trained, downloaded-FNO-trained, improved-FNO-trained, and
   hybrid fine-tuned policies on exact dynamics.

All GPU commands set `CUDA_VISIBLE_DEVICES` to one device. Exact holdouts remain
outside surrogate selection. If the structural pilot fails, compute moves to
hybrid exact fine-tuning instead of scaling a model that still violates the
transition-map structure.

## Decision on the downloaded `mix` checkpoints

The model is useful for prototyping, error discovery, and generating candidate policies. It is not reliable enough to serve as the sole dynamics engine for scientific claims about RL performance.

For the present fixed library of 312 actions, cached exact transfer tables should be the primary RL environment. The reproduced benchmark took 72.5 s with cached exact PPO and 1,073.7 s with FNO PPO, so the FNO was 14.8 times slower in that workload. The FNO becomes attractive when frequencies or pulse parameters vary continuously and exact tables cannot be precomputed.

## Evidence

| Diagnostic | Downloaded `mix` result | Interpretation |
|---|---:|---|
| Manifest coverage | 24/24 pairs | Complete checkpoint coverage |
| Manifest independent validation field | `{}` | No manifest-level independent certification |
| Checkpoints with parity residual | 0/24 | Source-package conversion parity is not quantified |
| Embedded on-resonance median infidelity | median 0.00211; P95 0.00791; max 0.00805 | Promising single-step average, but insufficient for closed-loop use |
| Exact/FNO failure correlation, 15 learned policies | 0.993 | Broad learned-policy ordering is preserved |
| Mean absolute failure gap, 15 learned policies | 1.49 percentage points | Reasonable average transfer in the weak-policy distribution |
| Largest learned-policy failure gap | 5.52 points, SAC seed 0 | Some learned seeds still have material transfer error |
| Physics-elimination failure | exact 29.36%; FNO 63.14% | 33.78-point pessimistic closed-loop bias |
| Cached exact versus FNO PPO time | 72.5 s versus 1,073.7 s | Current FNO has no speed advantage for the fixed action library |

Small independent audits used 16 held-out frequencies per stratum and 64 held-out populations on the two checkpoint pairs with the largest embedded errors.

| Metric | Block 0, σ− | Block 1, σ+ | Desired behavior |
|---|---:|---:|---:|
| Zero-time identity TV | 0.0538 | 0.0416 | 0 |
| Input-linearity TV | 0.0645 | 0.0504 | 0 |
| Near-pure branch-probability MAE | 0.00351 | 0.00436 | close to 0 |
| Near-pure conditional-branch TV, P95 | 0.365 | 0.292 | close to 0 |
| Control-mixture on-resonance infidelity, median | 0.0238 | 0.0262 | lower is better |
| Control-mixture off-resonance infidelity, median | 0.0199 | 0.0203 | exact/static error is approximately 10⁻⁶ |

These errors explain why average population infidelity can look acceptable while an 80-pulse controller fails. For action $a$, exact dynamics produce joint branch populations

\[
u_k=T_{a,k}s,\qquad \pi_k=\mathbf 1^\top u_k,\qquad s'_k=\frac{u_k}{\pi_k}.
\]

The exact map is linear in the input population $s$. The current model instead applies a softmax to an unconstrained FNO. Softmax enforces nonnegativity and total probability, but it does not enforce input linearity or the identity map at pulse time zero. Conditional normalization can amplify a joint error by approximately $1/\pi_k$ on a rare measurement branch.

Agreement on weak PPO/SAC/DDQN policies does not certify the high-purity region. Those policies fail on 69–99% of exact episodes and rarely visit the states where the stronger physics controller exposes the largest surrogate error.

## Phase 1: causal baseline before retraining the FNO

Train PPO and SAC directly against cached exact tables with the same observation, reward, horizon, transition budget, seeds, and evaluation protocol used for the FNO runs.

The comparison must contain:

| Training engine | Evaluation engine | Purpose |
|---|---|---|
| Exact cached | Exact cached | Measures RL/optimization limitations without model bias |
| Current `mix` FNO | Exact cached | Existing result; combines RL and model error |
| Current `mix` FNO | `mix` FNO | Measures the policy's in-model score |

Use five seeds and 5,000 exact evaluation episodes per seed. If exact-trained PPO/SAC remain far below physics elimination, prioritize the RL formulation before spending more compute on the FNO. If exact-trained agents improve substantially, the FNO is the main bottleneck.

For the finite horizon, include remaining budget $b_t=(H-t)/H$ in the observation. Match the discount across algorithms for equal-objective comparisons. Consider potential-based shaping

\[
r'(s,a,s')=r(s,a,s')+\gamma\Phi(s')-\Phi(s),
\]

with a preregistered potential based on population concentration or entropy. This provides a denser learning signal while preserving the intended optimum when implemented consistently.

## Phase 2: complete audit of the current FNO

Extend `scripts/thf_fno_paper_metrics.py` to accept `--manifest`, so source-package checkpoints use the manifest loader and retain SHA/provenance checks.

Run all 24 block/polarization pairs with three frequency strata:

- on resonance, including resonance shoulders;
- uniform across the allowed window;
- explicitly off resonance, compared with the static identity predictor.

Evaluate four population ensembles separately:

- diffuse Dirichlet populations;
- the current training mixture;
- near-pure simplex vertices;
- states collected from exact physics, exact RL, and FNO RL trajectories.

Record median, P95, and P99 for joint infidelity, joint TV, branch-mass error, and conditional-branch TV. Report errors versus belief purity $\lVert s\rVert_\infty$, branch probability $\pi_k$, block, polarization, frequency detuning, and pulse time.

Initial engineering gates for proceeding to full RL are:

| Gate | Threshold |
|---|---:|
| Zero-time identity TV, maximum | ≤ 0.001 |
| Input-linearity TV, P95 | ≤ 0.001 |
| Off-resonance joint TV, P95 | ≤ 0.005 |
| Branch-probability absolute error, P95 | ≤ 0.005 |
| Conditional-branch TV, P95 for $\pi_k\ge10^{-2}$ | ≤ 0.05 |
| Conditional-branch TV, P95 for $\pi_k\ge10^{-3}$ | ≤ 0.10 |
| One-step termination classification error on policy states | ≤ 0.5% |
| Strong-policy exact/FNO failure gap | ≤ 2 percentage points |
| Strong-policy exact/FNO average-action gap | ≤ 1 action |

The thresholds are engineering gates, not values claimed by the paper. Tighten or relax them only after measuring how each error changes exact closed-loop success.

## Phase 3: training data focused on RL occupancy

Build an exact replay dataset with tuples

\[
(s,a,\{u_k,\pi_k,s'_k\}_{k=0,1}).
\]

Collect states from physics elimination, exact-trained agents, random exploration, failed FNO trajectories, and near-terminal beliefs with $\max_i s_i\in[0.90,0.99]$. Oversample rare branches and the four checkpoint pairs that currently have the largest embedded on-resonance errors: block 0/σ−, block 1/σ+, block 0/σ+, and block 1/σ−.

Use an iterative exact-query loop:

1. Train the surrogate on the current dataset.
2. Train or roll out a policy in the surrogate.
3. Query exact dynamics for visited state/action pairs with high uncertainty or large ensemble disagreement.
4. Add those samples and retrain.
5. Keep fixed validation and final test sets untouched.

This is a DAgger-style correction for model exploitation and distribution shift.

## Phase 4: physics-constrained neural operator

The preferred redesign predicts transfer columns rather than an arbitrary nonlinear population map:

\[
\widehat T_\theta(\omega,\tau,\sigma),\qquad
\widehat u=\widehat T_\theta s.
\]

Construct each column as a probability distribution over joint molecular/measurement outcomes. Then

\[
\widehat T\ge0,\qquad
\mathbf 1^\top\widehat T=\mathbf 1^\top,
\qquad \widehat T(\omega,0)=\begin{pmatrix}I\\0\end{pmatrix}.
\]

This architecture makes input linearity exact and can make zero-time identity exact. Keep the FNO along the pulse-time dimension if its spectral representation remains useful. Predict a residual from the static identity map and gate the residual by pulse time so it vanishes at $\tau=0$. For frequencies far from resonance, blend toward the exact static predictor.

Train with a branch-aware objective:

\[
\mathcal L=\lambda_I I_p+\lambda_{TV}D_{TV}(u,\hat u)
+\lambda_\pi|\pi-\hat\pi|
+\lambda_c D_{TV}(s'_k,\hat s'_k)
+\lambda_m\mathcal L_{\mathrm{multi-step}}.
\]

Compute the conditional term only above a declared branch-mass floor and report every floor. Add 2-, 4-, and 8-step losses on exact action sequences. Use an ensemble of three to five independently initialized models to estimate epistemic uncertainty.

## Phase 5: robust FNO plus RL

Use the improved model in one of two hybrid modes:

1. **FNO pretraining followed by exact fine-tuning.** Learn quickly in the surrogate, then fine-tune the policy and critic with cached exact transitions.
2. **Dyna training.** Use mostly surrogate rollouts but replace a fixed fraction with exact transitions. Increase the exact fraction for high-purity states, rare branches, or large ensemble disagreement.

Always select checkpoints and report final performance using exact dynamics. Penalize uncertain surrogate actions or fall back to exact dynamics when ensemble disagreement exceeds a calibrated threshold.

## Compute-saving experiment sequence

| Stage | Runs | Continue only if |
|---|---|---|
| Exact RL diagnosis | PPO and SAC, one seed, 100k transitions | Learning curves improve beyond random |
| Exact RL confirmation | Best exact agent, five seeds, 1M transitions | Improvement survives seed variation |
| FNO structural pilot | Blocks 0± and 1± | Identity, linearity, and branch gates improve materially |
| Full FNO audit | All 24 pairs | Every pair passes the structural and occupancy gates |
| Surrogate RL smoke test | One seed per agent, 100k transitions | Exact and FNO evaluations agree within gates |
| Final comparison | Five seeds, 1M transitions | Improved or hybrid agent approaches exact-trained performance |

The final table should compare exact-trained, current-FNO-trained, improved-FNO-trained, FNO-pretrained/exact-fine-tuned, and Dyna agents. Rank them only by exact failure rate and failure-penalized average actions. Report FNO scores as transfer diagnostics rather than final performance.
