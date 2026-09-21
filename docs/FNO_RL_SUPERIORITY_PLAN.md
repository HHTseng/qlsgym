# Plan to make FNO-assisted RL outperform non-ML ThF+ controllers

## 1. Decision target

All promotion decisions use the cached-exact environment.  The primary order is

1. unfinished fraction after the 80-pulse budget;
2. failure-penalized average actions, where a failed episode scores 80.

The initial references were:

| Controller | Exact unfinished | Exact actions |
|---|---:|---:|
| Physics elimination | 29.36% | 51.49 |
| Hybrid PPO: downloaded `mix` pretraining + 250k exact | 33.44% | 48.21 |
| Fixed-order sweeping | 99.92% | 79.97 |

During execution, a stronger `DescendingPopulationPolicy` control reached
19.98% unfinished and 40.00 actions. This replaced physics elimination as the
controller to beat. A method counts as superior only if its paired failure
difference has an upper 95% seed-level confidence bound below zero and its mean
action difference is also below zero.

Every selected configuration is confirmed with five training seeds and 5,000
exact rollouts per seed.  Selection and final evaluation use different random
seeds.  Common evaluation seeds and batch sizes are used across controllers.

## 2. Stage A: diagnose and rescue the failure tail

### A1. Frozen policy-to-physics fallback

Keep the five completed hybrid PPO checkpoints frozen.  Evaluate a batched,
row-private fallback controller that begins with PPO and permanently switches a
trajectory to physics elimination when either:

- the remaining budget falls below a threshold; or
- maximum belief has stopped setting a new best value for several pulses.

The diagnostic grid varies remaining-budget thresholds and stagnation windows.
It is selected by exact success first and exact actions second.  Report paired
rescue and loss counts relative to PPO on common simulation streams.

**Promotion gate:** mean exact unfinished below 29.36% and mean exact actions
below 51.49.  A stronger claim requires the upper 95% seed-level confidence
bound for failure to be below 29.36%.

### A2. Failure-sensitive exact PPO

The historical reward gives one negative unit per pulse.  A timeout at 80 is
therefore only slightly worse than a late success, whereas reporting ranks
failure first.  Add an explicit nonnegative timeout cost

\[
r_t=-1-\lambda_f\,\mathbf 1[t=80\ \text{and unfinished}],
\]

to sampled GAE and both branch-expected qMDP targets.  Search
\(\lambda_f\in\{0,20,40,80\}\), with and without normalized remaining budget
\(b_t=(80-t)/80\).  Start from the completed FNO-to-exact PPO checkpoints and
retain all other Optuna-selected settings.

Select on a separate 2,000-episode exact validation seed.  Confirm the selected
configuration for five source/training seeds.  Evaluate both the risk-sensitive
actor alone and the actor with the Stage A1 fallback.

**Stop rule:** if neither frozen fallback nor risk-sensitive fine-tuning lowers
the exact failure mean, do not spend compute on wider PPO architecture searches.
Proceed to the residual policy in Stage B.

## 3. Stage B: residual learning around physics elimination

Collect exact trajectories from physics elimination and disagreement states
where PPO chooses another action.  Train a policy with physics behavior cloning,
then exact fine-tune it.  Physics elimination's action is always present in the
candidate set.  A learned deviation is accepted only when an ensemble estimates
a conservative positive advantage.

The critic should represent two quantities:

\[
Q_s(s,a)=P(\text{finish before 80}\mid s,a),
\qquad
Q_t(s,a)=E[T_{\rm remaining}\mid s,a,\text{finish}].
\]

Actions maximize \(Q_s\) first and minimize \(Q_t\) among statistically tied
actions.  Add the remaining budget to the observation.  Oversample failures,
rare branches, physics/PPO disagreements, and beliefs with
\(\max_i s_i\in[0.90,0.99]\).

**Promotion gate:** the five-seed exact result must dominate physics elimination
on both primary metrics.  Otherwise retain physics elimination as the default
and use learned actions only behind the conservative gate.

## 4. Stage C: branch-aware structured surrogate

The downloaded FNO passes 0/24 structural audits.  The transfer-column pilot
fixed identity, linearity, and off-resonance behavior but made branch mass about
15.2 times worse and conditional total variation 2.6--4.1 times worse.  A full
24-pair retrain remains prohibited until a new four-pair pilot passes.

For outcome branch \(y\), action \(a\), and input belief
\(s\in\Delta^{n-1}\), model the unnormalized joint mass

\[
q_y=K_y(a)s,\qquad p_y=\mathbf 1^Tq_y,
\qquad s'_y=q_y/(p_y+\epsilon).
\]

The parameterization must enforce

\[
K_y(a)\ge0,
\qquad
\sum_y\mathbf1^TK_y(a)[:,i]=1,
\qquad
K(\tau=0)=K_0
\]

by construction.  Initialize from the downloaded checkpoints and learn a
constrained correction, using block/polarization adapters rather than a single
shared transfer head.  Train with joint-mass, branch-mass, conditional-state,
and action-ranking losses.  Balance all pairs and branch-mass strata; emphasize
the difficult `0-`, `1+`, `0+`, and `1-` pairs.

Compare FNO with Fourier-feature MLP and Chebyshev/interpolation controls at the
same data and latency budget.  The architecture name is secondary to producing
a reliable branch kernel.

### Four-pair pilot gate

Promote only when all four pairs satisfy the preregistered structural bounds,
both conditional-TV medians fall by at least 25%, at least three of five
accuracy medians fall by at least 25%, and no pair/metric materially regresses.
Only then train all 24 pairs.

## 5. Stage D: policy-distribution aggregation

Create exact data from physics elimination, the best residual policy, failed
episodes, rare outcomes, and near-terminal beliefs.  Repeat:

1. train the structured surrogate;
2. run the current policy under it;
3. replay that policy in cached-exact dynamics;
4. append visited and disagreement states;
5. retrain until closed-loop transfer stabilizes.

Use a three-to-five-model ensemble.  During training and short-horizon planning,
fall back to cached exact dynamics when ensemble disagreement is high.  Add an
action-ranking loss and report exact regret, top-1 agreement, and top-k recall.
Surrogate rollouts are limited to one to three steps before an exact-trained
value estimate to avoid compounding branch error.

**Closed-loop gate:** on both physics elimination and the selected learned
policy, FNO-to-exact failure differs by at most two percentage points and actions
by at most one.  This is required in addition to all 24 structural audits.

## 6. Stage E: stronger non-ML controls and final claim

Replace the order-dependent sweep, which visits only 80 of 312 controls, with:

- coverage-balanced block/polarization schedules;
- an information-gain-ordered sweep;
- an adaptive nonrepeating sweep;
- tuned physics-elimination tie-breaking; and
- a short-horizon cached-exact beam-search reference.

Lock all tuning on validation seeds.  The final test uses at least five training
seeds and 5,000 exact episodes per policy.  Report the exact failure/action
Pareto plot, success-by-pulse survival curves, paired rescue/loss counts,
seed-level uncertainty, FNO policy-occupancy error, and action regret.  Claim
superiority only when the paired 95% confidence interval for the failure
difference lies below zero and the learned method does not use more actions.

## 7. Compute and execution contract

- Tara GPUs are restricted to physical devices 0 and 2.
- At most one process runs on each GPU and at most two GPU processes run total.
- Diagnosis precedes five-seed confirmation.
- Results are persisted after every candidate and seed so interrupted jobs can
  resume without repeating completed work.
- The downloaded manifest and exact holdout contract remain unchanged.
- A failed promotion gate stops the dependent expensive stage; evidence is
  retained and summarized rather than hidden.

The first executable study covers Stages A1 and A2 because they directly target
the small remaining reliability gap without depending on another unvalidated
FNO.  Its result determines whether Stage B or the new Stage C pilot is the next
use of compute.

## 8. Execution outcome — 21 September 2026

The implementation used physical Tara GPUs 0 and 2, with at most one process
per GPU. Every final learned or hybrid row uses five policy seeds and 5,000
holdout episodes per seed. Exact dynamics determine all rankings.

### Controller result

| Controller | Type | Exact unfinished | Exact actions | Decision |
|---|---|---:|---:|---|
| FNO_RL_optuna PPO | actor only | 41.78% | 51.37 | historical reference |
| Failure-sensitive PPO | actor only | 24.92% | 45.20 | beats physics elimination, not descending population |
| Descending population | non-ML exact-table rule | 19.98% | 40.00 | strongest non-ML control |
| 15-pulse PPO + descending fallback | hybrid | 18.40% | 41.31 | fewer failures, but 1.31 more actions; reject superiority |
| **Exact candidate arbiter** | exact-table hybrid | **16.37%** | **38.70** | **passes the superiority rule** |

The arbiter compares the failure-sensitive PPO and descending-population action
at each state. Cached-exact one-step outcomes admit the learned proposal only
when it improves immediate success probability or expected posterior purity by
the selected margin, \(10^{-4}\). Relative to descending population, the paired
failure difference is \(-3.61\) percentage points with a 95% seed interval of
\([-4.75,-2.47]\), and the mean action difference is \(-1.30\). Across 25,000
paired episodes it rescues 1,478 and loses 576. This establishes a model-based
hybrid improvement. It does not establish that a standalone RL actor beats the
strongest non-ML controller.

### Structured FNO result

The branch-aware column-v2 pilot improved all five median accuracy metrics by
at least 66%, while enforcing exact identity and input linearity. The pilot
code applied the two structural bounds pairwise and used median improvement and
regression limits for the remaining metrics; its old label incorrectly implied
that all seven absolute bounds passed pairwise. That label is corrected in the
implementation.

The promoted 24-pair training then passed every absolute gate for only **10/24**
pairs. Identity, linearity, and off-resonance TV pass globally, but block
\(11,+\) has branch-mass P95 error 0.07254, conditional-TV P95 errors 0.27775
and 0.68902, and termination error 0.10705. The preregistered gate therefore
blocked improved-FNO PPO training.

The diagnostic closed-loop audit confirmed that this was the correct decision:

| Controller | Exact failure/actions | Downloaded `mix` | Column-v2 |
|---|---:|---:|---:|
| Failure-sensitive PPO | 24.92% / 45.20 | 28.98% / 48.40 | 89.51% / 73.62 |
| Descending population | 19.98% / 40.00 | 23.68% / 41.44 | 91.10% / 73.40 |

Column-v2 fixes several local structural errors yet fails catastrophically on
the visited closed-loop distribution. The next FNO work must target the failing
block/polarization pairs and policy-occupancy data before any RL training. Wider
generic architecture searches are not justified by these results.

Final artifacts are in
[`results/thf_fno_rl_superiority`](../results/thf_fno_rl_superiority), the full
audit is in
[`results/thf_column_fno_v2_full_audit`](../results/thf_column_fno_v2_full_audit),
and the closed-loop audit is in
[`results/thf_column_fno_v2_transfer`](../results/thf_column_fno_v2_transfer).
