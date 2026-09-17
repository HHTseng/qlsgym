# Paper-style validation of the ThF+ FNO

Source: [Inverse Design of Quantum Control Sequences with Fourier Neural Operators, arXiv:2608.03702v2](https://arxiv.org/pdf/2608.03702). We adapt Fig. 3, Fig. 4, Eqs. (27)–(31), and the finished-episode curves of Fig. 5. This is not a numerical reproduction of that paper: molecule, motional truncation, action set, hardware, propagation implementation and checkpoint criterion differ.

## Map and probability metric

For Hamiltonian block $f$ with $m_f$ molecular states, initial population $q\in\Delta^{m_f-1}$, polarization $\sigma\in\{+,-\}$, drive frequency $\omega$ (rad/ms), and pulse time $\tau$ (ms), the learned operator is

$$G_{\theta}^{f,\sigma}:(q,\omega)\mapsto\{\widehat p(\tau_r)\}_{r=1}^{200},\qquad \widehat p(\tau_r)\in\Delta^{2m_f-1}.$$

The two groups of $m_f$ output channels are joint probabilities of molecular state and motional outcome $k=0$ or $k=1$ (ground versus **all** excited motional levels). ThF+ actually propagates seven motional levels; the two-channel readout is not a two-level Hamiltonian truncation. Exact propagation gives $p(\tau)=T_{\omega,\sigma}(\tau)q$, linear in $q$.

The reference input density operator is $\rho(q)=\sum_i q_i|i,0\rangle\langle i,0|$. The environment reconstructs this diagonal, ground-motion input from each conditional population posterior before the next pulse: it does not carry inter-pulse coherences. “Exact” means propagation of this specified effective/truncated population-reset model. These tests do not certify arbitrary coherent inputs, motional-cutoff convergence or agreement with experiment; a separate exact cutoff test is useful before asserting physical convergence.

For normalized joint distributions, the paper's population fidelity and infidelity are

$$F_p(\tau)=\left[\sum_{b=1}^{2m_f}\sqrt{p_b(\tau)\widehat p_b(\tau)}\right]^2,\qquad I_p(\tau)=1-F_p(\tau)\in[0,1].$$

**Lower $I_p$ is better.** This compares population distributions, not quantum-state coherences or gate fidelity. For $J$ independent initial populations,

$$\overline I_{\omega}(\tau_r)=\frac1J\sum_{j=1}^J I_p(q_j,\omega,\tau_r),\qquad E_{\omega}=\frac1{200}\sum_r\overline I_{\omega}(\tau_r).$$

Fig. 3-style bands are frequency percentiles of $\overline I_\omega(\tau)$ **after** averaging initial states. Frequency scatter plots show $E_\omega$, not percentiles across individual channels/episodes. Two ensembles are reported separately: diffuse Dirichlet$(1)$ and the existing `alpha=-2` mixture of diffuse, concentrated and near-pure populations. See `random_mixed_populations` for its exact mixture; `-2` is a sampler selector, not a negative Dirichlet concentration.

Uniform frequency sampling reproduces the paper-style view but misses most narrow useful resonances: block 0's active frequency fraction is only ~0.0035. We therefore additionally draw explicitly on-resonance (within one linewidth) and off-resonance test frequencies, comparing the static predictor $\widehat p(\tau)=p(0)$. Static wins away from resonances when true dynamics barely move; reporting only the uniform aggregate can hide both errors and useful capabilities.

## Trajectory and closed-loop diagnostics

- Representative trajectory: strongest retained in-window resonance, selected by physical coupling **before examining prediction errors**, diffuse input seed 20260920. Plot at most 12 active channels; compute $I_p$ using **all** channels.
- Active positive-support MRE: channels moving by >$10^{-4}$, evaluated only where true population >$10^{-6}$. This deliberately avoids division by zero; its denominator convention is not the paper's unqualified MRE. Full-channel infidelity and identity leakage remain visible.
- Zero-time identity error: $D_{\mathrm{TV}}(G(q,\omega)|_{\tau=0},(q,0))$, ideal zero.
- Near-pure inputs: $q_i=0.995$, the remaining 0.005 distributed uniformly among other local states. Report mean and P95 time-averaged $I_p$ across vertices.
- Linearity error: average over pulse times of $D_{\mathrm{TV}}(G((q_1+q_2)/2),[G(q_1)+G(q_2)]/2)$, ideal zero.
- Branch error: for joint sub-vector $u_k$, mass $\pi_k=\sum_i u_{k,i}$, and conditional belief $s_k=u_k/\pi_k$, report probability MAE and conditional-TV P95 on near-pure inputs, excluding exact $\pi_k<10^{-3}$.

Here $D_{\mathrm{TV}}(p,\widehat p)=\tfrac12\|p-\widehat p\|_1\le\sqrt{I_p}$. For example $I_p=0.02$ permits TV up to 0.141. Conditional normalization can amplify errors as $O(D_{\mathrm{TV}}/\pi_k)$; an apparently small joint error may be disastrous for a rare measurement branch. S18's exact branch expectation reduces sampling variance **under the surrogate**, not its systematic dynamics error.

A physics-constrained next model could predict transfer columns once per frequency and use $G_\theta(q,\omega,\tau)=\widehat T_\theta(\omega,\tau)q$, with $\widehat T\ge0$, $\mathbf1^\top\widehat T=\mathbf1^\top$, and $\widehat T(\omega,0)=(I,0)^\top$. This guarantees input linearity/zero-time identity and amortizes inference across state batches; it is a proposed follow-up, **not** a constraint imposed on the present trained FNO or an experiment already executed.

## Timing: two honest baselines

For each batch size and workload, report synchronized median times after warmup and speedup $T_{\mathrm{exact}}/T_{\mathrm{FNO}}$ (larger faster). Workloads: (i) many input populations at fixed frequency; (ii) many frequencies at a fixed population. FNO time includes embedding and full 200-time forward prediction.

Two references matter: exact eigenpropagation **build + apply**, and **cached exact transfer columns + apply**. The latter can be faster than FNO once small fixed-control tables are available. The paper benchmarks CUDA-Q on A100 hardware; our reference is exact PyTorch on the explicitly logged device. Do not call our ratio a CUDA-Q speedup or compare absolute ratios across hardware/tasks. Table-generation amortization is outside the cached timing.

## Checkpoint and held-out design

Production preset: 24,000 candidate frequencies, 32,000 sampled training pairs, 400 validation pairs, 120 epochs, width 256, four FNO layers, 60 modes; AdamW initial LR $5\times10^{-4}$, StepLR factor 0.75 every 12 epochs, activity-weighted loss. Selection uses 64 on-resonance frequencies × 16 initial states to fit 11-GiB GPUs. This adapts rather than duplicates the paper's schedule/data/lowest-validation-loss checkpoint.

The locked `best_onres.pt` minimizes selection-set on-resonance infidelity; selection is not testing. Ordinary held-out tests use 32 frequencies/stratum ×128 initial states with seeds 20260916/17. Full paper-style tests on predeclared difficult blocks (0,+) and (1,−) use **100 frequencies/stratum ×500 initial states**, new seeds 20260918/19, and two input ensembles. Each stratum has 50,000 frequency–population pairs ×200 times. CPU previews use smaller explicitly logged sizes and are not pooled with these full tests. All 24 pairs retain ordinary independent audits, so testing only two difficult blocks is not represented as full-model coverage.

```bash
PYTHONPATH=src python scripts/thf_fno_paper_metrics.py \
  --work "$QLSGYM_WORK" --tag rlprod120v2 --block 0 --sigma + \
  --device cuda:0 --n-freq 100 --n-init 500 --batch-size 32
```

JSON logs checkpoint SHA256, physics fingerprint, seeds, input ensemble, batch sizes, hardware and exact reference. NPZs preserve raw frequency curves and trajectories; figures are reproducible, not a best-case hand-picked illustration.
