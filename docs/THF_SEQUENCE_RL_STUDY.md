# Sequence-aware PPO and SAC under the downloaded ThF+ FNO

## Question

At decision time $t$, let

$$
s_t\in\Delta^{191},\qquad a_t\in\{0,\ldots,311\},\qquad k_t\in\{0,1\}
$$

denote the post-measurement molecular population belief, selected pulse, and
measured branch.  The experiment asks whether replacing the optimized MLP by a
finite-history encoder improves control under the fixed downloaded `mix` FNO.

The ideal environment is already Markov in the full belief and remaining
budget $b_t=(80-t)/80$:

$$
P(s_{t+1},r_t\mid s_{0:t},a_{0:t})
=P(s_{t+1},r_t\mid s_t,a_t).
$$

History therefore has no information-theoretic advantage in the nominal
model.  It can help only through finite-network approximation, FNO error, or a
latent variable omitted from $s_t$.  This makes the MLP a necessary matched
control rather than an obsolete architecture.

## Implemented model

For context length $K$, define

$$
H_t=(x_{t-K+1},\ldots,x_t),
$$

with token

$$
x_i=
E_s(\sqrt{s_i})+E_a(a_{i-1})+E_k(k_{i-1})+E_b(b_i)
+E_{\mathrm{phys}}(\alpha_{i-1})+E_{\mathrm{pos}}(i).
$$

Here

$$
\alpha(a)=(\widehat\omega_a,\widehat\tau_a,\sigma_a,q_a)
$$

contains normalized detuning and duration, polarization sign, and a
primitive-pulse indicator.  Learned START embeddings define $a_{-1}$ and
$k_{-1}$.  The code provides frame-stack, GRU, and pre-norm causal
Transformer encoders.  Transformer tokens can additionally use LayerNorm after
feature summation.  The MLP code path remains unchanged.

PPO maintains one right-aligned rolling history per parallel environment.  SAC
stores episode-safe observed histories in replay.  Neither implementation lets
a context cross an episode reset.  Actor and twin SAC critics use independent
encoders.

Both algorithms preserve the two-branch S18 target.  For the branch states
$s_{t+1}^{(k)}$, the code constructs distinct counterfactual histories

$$
H_{t+1}^{(k)}=
\mathrm{shiftAppend}(H_t,s_{t+1}^{(k)},a_t,k,b_{t+1}),
$$

and uses

$$
y_t=\sum_{k=0}^{1}p_{t,k}
\left[r_{t,k}+\gamma c_{t,k}V(H_{t+1}^{(k)})\right].
$$

For SAC, $V$ is the target soft value.  Thus the architecture experiment does
not replace quantum branch expectation by a sampled one-branch target.

## Bounded experiment

The run used Tara GPUs 0 and 2 from 14:10 to 16:22 EDT on 1 October 2026.  It
trained 44 policies, consumed approximately 5.1 million FNO transitions, and
evaluated 86,000 episodes across FNO and exact cached dynamics.

The single-seed screen tested 15 architectures for each algorithm:

- MLP;
- frame stacks and GRUs with $K\in\{4,8\}$;
- Transformers with $K\in\{1,4,8\}$;
- $K=8$ state-history and no-position ablations;
- five corresponding token-LayerNorm Transformer ablations.

PPO used 100,000 screen and 300,000 confirmation transitions.  SAC used 40,000
and 100,000 because its three independent sequence encoders made the original
budget incompatible with the six-hour limit.  The screen ranked candidates
using only the downloaded FNO.  The MLP and two best sequence candidates were
confirmed with training seeds 31001 and 31002.  Each confirmation used 2,000
FNO and 2,000 exact evaluation episodes.  Exact dynamics never selected a
candidate.

The plan's exact-first and latent-drift stages were not mixed into this study:
the requested intervention was architecture under the fixed downloaded FNO.
The exact simulator is instead a held-out transfer audit.  $K=16$, shuffled
order, and latent-physics variants were not promoted because every $K\leq8$
sequence family lost to its matched MLP during the bounded screen.

## Results

Lower unfinished fraction $f_E=P_E(T>80)$ and failure-penalized actions
$A_E=E_E[\min(T,80)]$ are better.  Values are means and sample standard
deviations over two training seeds.

| Agent | Encoder | $f_E$ | $A_E$ | Failure change from matched MLP | Action change |
|---|---|---:|---:|---:|---:|
| PPO | **MLP** | **27.53% +/- 1.59%** | **42.91 +/- 0.28** | 0 | 0 |
| PPO | GRU, $K=8$ | 73.78% +/- 4.70% | 62.51 +/- 2.04 | +46.25 pp | +19.60 |
| PPO | Transformer, $K=8$, state history | 90.50% +/- 13.01% | 74.91 +/- 7.02 | +62.98 pp | +32.00 |
| PPO | Transformer + token LN, $K=8$, state history | 89.35% +/- 9.33% | 73.07 +/- 6.04 | +61.82 pp | +30.16 |
| SAC | **MLP** | **44.38% +/- 1.73%** | **52.46 +/- 1.22** | 0 | 0 |
| SAC | frame stack, $K=4$ | 58.40% +/- 1.77% | 59.17 +/- 0.67 | +14.03 pp | +6.71 |
| SAC | Transformer, $K=8$, state history | 55.85% +/- 1.34% | 56.99 +/- 0.50 | +11.48 pp | +4.53 |

Every paired confirmation seed favors the MLP in both primary metrics.  The
result does not depend on one unlucky sequence seed.

The full-token SAC Transformers exhibit a stronger failure mode.  Their
single-seed unfinished fractions are 98.8%--100% for $K\in\{1,4,8\}$, and
actor entropy collapses toward zero during training.  Token LayerNorm does not
remove the collapse: its full-token variants fail 99%--100%.  Removing action,
outcome, and physical-action embeddings prevents the immediate entropy
collapse, but the resulting state-history Transformer still loses to the MLP.
The failure is therefore partly an optimization interaction among the
additive token channels and categorical SAC, while the negative state-history
result remains after removing those channels.

The new matched MLP means are better than the historical five-seed
`FNO_RL_optuna` references (PPO 41.78% failure, SAC 57.07%).  This comparison
uses different training budgets and only two new seeds, so it is evidence of
training-length/seed sensitivity, not evidence that the sequence change helped.
Architecture claims use only the matched seeds and budgets in the table.

## Interpretation

The downloaded FNO does not make temporal memory useful under the present
observation model.  The current $s_t$ already contains the posterior needed
by the transition kernel, and the extra sequence encoder makes optimization
harder.  The MLP remains the supported PPO and SAC architecture for this fixed
FNO.

A further sequence study is justified after adding a controlled latent process,
for example

$$
\delta\omega_{t+1}=\rho_\omega\delta\omega_t+\epsilon_t,
$$

finite-shot estimates of $s_t$, partial observation, or residual motional
memory.  In that POMDP, compare MLP, GRU, and Transformer at equal transition
budgets, and add a direct current-state residual path so attention models cannot
discard the sufficient present belief.  For nominal fixed-FNO training, more
Transformer tuning is not supported by the current evidence.

## Reproduction and artifacts

On Tara:

```bash
cd /home/htseng/qlsgym_FNO_seqRL
export QLSGYM_WORK=/home/htseng/qlsgym_work
export PYTHONPATH=$PWD/src

CUDA_VISIBLE_DEVICES=0 python scripts/run_thf_sequence_rl.py run-agent ppo \
  --device cuda --output results/thf_sequence_rl_mix \
  --screen-steps 100000 --confirm-steps 300000

CUDA_VISIBLE_DEVICES=2 python scripts/run_thf_sequence_rl.py run-agent sac \
  --device cuda --output results/thf_sequence_rl_mix \
  --screen-steps 40000 --confirm-steps 100000

python scripts/run_thf_sequence_rl.py summarize \
  --output results/thf_sequence_rl_mix
```

The manifest contract fixes fingerprint `d7deb43457d3`, SHA-256
`81ff309e7d4fb1663d384498ae46ce35c77146688e71db6045f168ebdb14d2ed`,
24/24 block-polarization pairs, 192 states, and 312 actions.  See
[`summary.md`](../results/thf_sequence_rl_mix/summary.md),
[`summary.json`](../results/thf_sequence_rl_mix/summary.json), [`run_index.json`](../results/thf_sequence_rl_mix/run_index.json), the
[confirmation figure](../results/thf_sequence_rl_mix/sequence_vs_mlp.png), and
the [complete screen](../results/thf_sequence_rl_mix/sequence_screen.png).
