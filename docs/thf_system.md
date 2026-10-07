# The ThF⁺ system in qlsgym

What the ²³²Th¹⁹F⁺ molecule pack (`qlsgym.molecules.thf`) contains, where every number comes from,
and what is approximate. Every number marked "measured here" was computed from the current code and
tables (fingerprint `30547ef39855`, heff `0559bf1`). Every number marked "decided" is an input
setting, with the reason it was chosen; those settings were made in the predecessor project and
recorded in its decision ledger (`_legacy/FNO_NEW_MOLECULE/ThF_232_19/DECISIONS.md`, tags D1–D14).
Two were changed here on 2026-10-07: J_max 6 → 7 and n_nu 7 → 8 (sections 1 and 3).

## Pipeline

```
heff (MolecularHamiltonians/heff @ 0559bf1, parameter set thf_v1)
  │  qlsgym.molecules.thf_tables       scripts/build_thf_tables.py; build-time only
  ▼
data/thf/ThF232_levels_Jmax_7_B_20G_E_0Vcm.txt                 252 levels, E in MHz
data/thf/ThF232_two_photon_sigma_plus_modelA_Jmax_7_B_20G_E_0Vcm.txt   2136 σ⁺ couplings, rad/ms
  │  qlsgym.molecules.thf.build()      trap, window, task, carrier masks, cross-J primitives
  ▼
Molecule("thf")  ──►  physics/ (exact block propagation)  ──►  env/ (action tables, belief MDP)
                  └─►  surrogate/ (FNO per block and polarisation)
```

`tests/test_thf_tables.py` checks the coupling model's claims. It also checks that `data/thf`
equals a fresh build byte for byte, except the timestamp line. Training stamps every checkpoint
with the sha256 of both table files, so a table rebuild invalidates every surrogate.

## 1. Molecular structure (heff)

- **State:** X ³Δ₁, v = 0, Hund's case (c) basis |J, Ω = ±1, F, m_F⟩ with I(¹⁹F) = ½.
- **Rotational cutoff, J = 1…7 (252 states).** Decided 2026-10-07 (was J ≤ 6, D3). Measured
  here: a rigid rotor with B₀ = 7274.333 MHz holds 99.62% of the 4 K population in J ≤ 7 (98.57%
  in J ≤ 6, 99.92% in J ≤ 8). The missing 0.38% is a thermal tail that the model does not contain
  at all.
- **Basis truncation at the top J.** heff diagonalises J ≤ J_max only, so the highest J lacks its
  J + 1 partner. Measured here against a J ≤ 8 build: J = 7's in-window resonances move by at most
  1.9 Hz (under 1% of a strong line's ~300 Hz half-width; the 0.63-HWHM worst case is a 3 Hz-wide
  dark rung); J ≤ 6 resonances move by ≤ 0.001 Hz. Adding J = 7 changed the J ≤ 6 couplings by
  ≤ 0.8 Hz out of 8 kHz.
- **Fields: B_z = 20 G, E_z = 0.** Decided (D2).
  - E = 0 makes parity exact, which gives the clean (J, parity) blocks below.
  - 20 G separates the e/f parity combs and the Zeeman rungs for J ≤ 2. The rungs are only
    marginally resolved at J = 3 and not resolved above (section 5).
- **Parameter set `thf_v1`.** Every entry carries a status and a source; the full table is in
  each table header.
  - Measured: A_∥, D₀, d_mf, ω_ef.
  - Derived: B₀, G_zz.
  - Ab-initio: E_eff, W_TP.
  - **Estimates: G_xx, G_yy** (second-order sums, Petrov & Skripnikov 2025, about 20%
    uncertainty) **and c_I** (scaled from CsF).
- **Zeeman tensor.** heff's full G = diag(G_xx, G_yy, G_zz) is in the Hamiltonian.
  - Its parity-dependent part G_Δ = (G_xx − G_yy)/2 = −2.06e-4 makes e and f have different
    g-factors.
  - At J = 1, F = 3/2: g_f − g_e = +2.74e-4, and e (parity +(−1)^J) lies *below* f at every J.
  - Both are checked in `tests/test_thf_tables.py` against the shipped table: g_e = −0.015039,
    g_f = −0.014765, mean −0.014902 (Ng 2022 measured |g_F| = 0.0149).
- **Labels.** Each eigenstate gets (J, F, parity, m_F, ξ). These are its dominant ket, exact
  parity, and an energy order ξ inside its (J, parity, m_F) class.

## 2. Two-photon couplings: model A

No Raman rate inside X ³Δ₁ has been measured, so the coupling is modelled as
T = (d·ε₂*) P_int (d·ε₁)/Δ. Here P_int projects onto **one Ω = 1 intermediate electronic
state**, summed over all its sublevels with J'' ≤ 7, at **one common detuning**. Decided (D1).

**What follows from the model** (all checked in `tests/test_thf_tables.py`):
- It is parity-even, so couplings never cross parity.
- The channels are pure Δm_F = ±1 (σ⁺ := Δm_F = +1, realised by beams (π, σ⁻) in the Raman
  reading).
- The rank-1 part vanishes, so **the σ⁻ channel is the exact reverse of σ⁺** (|A₋| = |A₊|ᵀ). The
  exact engine builds σ⁻ from σ⁺ using this property.
- The operator is pure rank 2. That **suppresses the m = −½ → +½ rung** of every ladder, which
  is the dark-state structure the elimination primitives use.
- e and f twins couple identically in the ket basis. In the eigenbasis they differ by about
  1e-3 of the largest coupling, entirely through G_Δ: the control run with G_xx = G_yy removes it.

**Scale.** The reduced dipole and the detuning form one overall factor. It is fixed by setting
|J=1, F=3/2, +, m=+½⟩ → |m=+3/2⟩ to |Ω|/2π = **3 kHz**. Decided (D6): every in-window rung must
fit a π-pulse inside 6 ms. That reference rung's π-time is 2.47 ms.

**Coupling statistics (measured here).** 2136 σ⁺ couplings:

| ΔJ | count | \|Ω\|/2π range |
|---|---|---|
| 0 | 448 | 0.040–6.08 kHz |
| 1 | 744 | 3e-5–6.60 kHz |
| 2 | 560 | 9e-5–8.11 kHz |
| 3 | 384 | ≤ 1.1e-3 kHz (negligible) |

The ΔJ = 0 couplings lie inside the blocks. The ΔJ ≠ 0 couplings are far off-window, and only
the cross-J primitives use them.

**Approximations.** The model is approximate on purpose, in two ways:
- A real experiment resolves the intermediate rotational ladder (B_int ~ 7 GHz against a few-GHz
  detuning).
- Ω = 0 intermediates would add a ΔΩ = ±2 amplitude that makes e and f couple differently.

## 3. Trap and motion

All of these are decided (D4, D10). The trap parameters are assumptions: no ThF⁺/Ca⁺ trap has
been characterised.

| quantity | value | basis |
|---|---|---|
| shared mode | ThF⁺/Ca⁺ in-phase axial mode, ν_f/2π = 1112.7182 kHz | the H₃O⁺ paper's trap: its 5.164 MHz is read as the H₃O⁺/Ca⁺ stretch mode, which implies a single-Ca⁺ axial frequency of 2.328 MHz. The ThF⁺/Ca⁺ stretch mode would carry only 22% ThF⁺ amplitude (η = 0.009), so the in-phase mode is used |
| Lamb–Dicke η | 0.06776 | 769.6 nm Raman beams, counter-propagating along the axis |
| motional levels | n_nu = 8 (ν ≤ 7) propagated exactly | ν − m_F is conserved along a σ⁺ ladder and the rungs are nearly evenly spaced, so a pulse lets population climb several rungs at once. ν ≤ 1, the H₃O⁺ paper's cutoff, is wrong by 0.12–0.19 in branch probability for J ≥ 3 (legacy measurement, F4). n_nu = 7 sufficed for J ≤ 6 (1.2e-3) but is wrong by **0.67** on J = 7; n_nu = 8 brings J = 7 to 1.5e-3 (decided 2026-10-07; section 9 checks n_nu = 8 against 16, which closes the J = 7 ladder) |
| readout | lumped ν = 0 vs ν ≥ 1 | what a threshold Ca⁺ readout measures |

**Why n_nu = J + 1.** A σ⁺ step is |m, ν⟩ → |m+1, ν+1⟩, so ν − m_F is conserved, and every pulse
starts in ν = 0. When the rungs of an F = J + ½ ladder are unresolved, population climbs until the
rank-2 dark rung m = −½ → +½ stops it: J rungs, so ν up to J. Measured here (worst transfer entry
against 14–16 levels): J = 4 is 0.37 at n_nu = 4 and 3e-4 at 5; J = 6 is 0.47 at 6 and 1.1e-3 at 7;
J = 7 is 0.67 at 7 and 1.5e-3 at 8. The residual is leakage through the 40 Hz dark rung, which
only the full ladder (n_nu = 2J + 2) closes: J = 7 is still 1.6e-3 at 9 and 7.8e-4 at 10. J ≤ 3
rungs are partly resolved (J = 3: 0.35 kHz apart against 0.15–0.33 kHz half-widths), so the climb
dies out early.

**Open (OPEN-3):** reading the paper's 5.164 MHz as the stretch mode is a plausibility argument.
The in-phase reading would give η = 0.048 and π-times 1.4× longer.

## 4. Control window and pulses

| quantity | value | why |
|---|---|---|
| drive window | ν_f ± 470 kHz = [642.72, 1582.72] kHz | symmetric, so σ⁻ maps in by reflection (ω⁻ = 2ν_f − ω⁺); covers every strong σ⁺ resonance at 20 G |
| pulse durations | 200 points on [0, 6] ms, inclusive | the weakest strong in-window rungs need up to 5.1 ms (J = 7) |
| carrier masks | ±1 kHz around 906.32 and 921.66 kHz | the only in-window carriers (J = 1, F = ½). The sideband-only model is wrong on a carrier, so the masks are removed from the action grid and from training |
| RWA cutoff | 1e6 rad/ms | couplings detuned by more than this everywhere in the window are dropped |
| minimum coupling | 1 rad/ms (\|Ω\|/2π = 0.159 kHz) | weaker couplings get no grid action, no primitive and no embedding channel. They **stay in the dynamics**: the exact engine keeps every coupling within the RWA cutoff |
| embedding constants | Δ_max = 1e4 rad/ms, s_emb = 0.05, β = 0.01 | surrogate input only (section 7) |

## 5. Block structure (measured here)

With E = 0 and only ΔJ = 0 couplings in the window, the in-window dynamics split into **14
closed (J, parity) blocks**. Blocks are ordered by size, and within a pair the lower-energy
parity comes first.

| block | (J, parity) | states M | σ⁺ couplings | in-window σ⁺ resonances | thermal mass (4 K) | max \|Ω\|/2π (kHz) |
|---|---|---|---|---|---|---|
| 0, 1 | (7, −), (7, +) | 30 | 56 | 26 | 0.005 each | 6.08 |
| 2, 3 | (6, +), (6, −) | 26 | 48 | 22 | 0.015 each | 5.97 |
| 4, 5 | (5, −), (5, +) | 22 | 40 | 18 | 0.037 each | 5.76 |
| 6, 7 | (4, +), (4, −) | 18 | 32 | 14 | 0.073 each | 5.48 |
| 8, 9 | (3, −), (3, +) | 14 | 24 | 10 | 0.114 each | 4.92 |
| 10, 11 | (2, +), (2, −) | 10 | 16 | 6 | 0.138 each | 3.29 |
| 12, 13 | (1, −), (1, +) | 6 | 8 | 3 | 0.117 each | 3.61 |

- **Population.** Thermal mass by J is 0.234, 0.275, 0.228, 0.146, 0.075, 0.031, 0.011 for
  J = 1…7. The most populated single state holds 1.95%.
- **The largest blocks are 30 states, not H₃O⁺'s 40–64.** This is the main reason the
  surrogate problem is easier here (see `docs/h3o_fno_findings.md`).
- **Dark rungs.** In blocks 12 and 13 the middle rung of the J = 1, F = 3/2 ladder
  (m = −½ → +½) is the rank-2-suppressed one. Its coupling is 0.17 kHz, nonzero only through
  Zeeman mixing, so its π-time is 44 ms, far beyond the pulse window. No library pulse drives it
  to completion.
- **Resonances are crowded.** 141 of the 197 neighbouring pairs of strong in-window σ⁺
  resonances are closer than the sideband Rabi frequency of the stronger line (median gap: 0.31
  of it).
  - 120 of those pairs are rungs of one nearly evenly spaced Zeeman ladder, which only the
    quadratic Zeeman term separates.
  - 21 are lines from different blocks, and 12 of those essentially coincide (< 0.1 of a Rabi
    frequency). J = 7's lines (1150–1180 kHz) fall among J = 5's and J = 6's.

  So a pulse generally drives several transitions at once. The exact engine handles this
  correctly, because every block responds to every drive, but selective addressing is often
  impossible.

## 6. Actions and task

- **`ActionLibrary.physics_subset`: 416 actions.**
  - **392 grid pulses.** Each strong in-window σ⁺ resonance with a π-time ≤ 6 ms is taken at its
    π-time, together with its σ⁻ mirror. All 392 drives are distinct.
  - **24 cross-J primitives** (`cross_j_primitives`, mode `dark_state`): sideband Raman
    π-pulses with ΔJ = 1 (4 of them) or ΔJ = 2 (20), with π-times 1.0–2.1 ms. They empty the
    stretched states of the most populated ladders and the J = 1, F = ½ states into another J,
    which lets the policy reach population the in-window drives cannot move. Decided (D8).
  - Each primitive takes its source's strongest ΔJ = 1, 2 coupling (decided, kept 2026-10-07). With
    J = 7 in the model, that sends the four J = 5 stretched states *up* into J = 7 (the
    least-populated block) instead of down into J = 4 as at J ≤ 6.
  - The primitives' coupling magnitudes come from model A's closure limit (heff OPEN-23). Their
    π-times are assumptions, not predictions.
- **Task:**
  - Prior: Boltzmann at 4 K.
  - Success: one population ≥ 0.98.
  - Budgets: the task budget is 80 pulses. The benchmark horizon is 800, so that a full sweep of
    the library fits.
  - Readout: ν = 0 vs ν ≥ 1 after every pulse, followed by a Bayes update of the diagonal
    belief. Coherences are dropped (section 8).

## 7. Surrogate inputs

A per-block FNO maps a block's initial populations and a drive (σ, ω) to the trajectory of both
readout branches on the τ grid. Its input (`surrogate/embedding.py`, paper Eqs. 11–16) has
M + Q + 1 channels over the τ grid:
- the M initial populations, held constant in τ;
- one channel s_emb · sin(Δ_q τ) / (1 + β|Δ_q|) for each of the Q retained transitions, where
  Δ_q is the drive's detuning from that resonance;
- one channel encoding the drive frequency's position in the window.

A transition is retained when its resonance lies within Δ_max of the window and its coupling is
at least Ω_min. The coupling strengths themselves are *not* inputs: they are fixed per block, so
each block's network has to learn them.

Measured here: every J ≥ 2 block has two in-window resonances below Ω_min, its rank-2-suppressed
dark rungs at 40–57 Hz. The ground truth contains them, but the network gets no channel for them.
In a two-level estimate they can move at most 0.3–0.5% of the population in 6 ms: small, but not
zero against the accuracy a planner needs.

Each of the 14 blocks needs **its own model for each polarisation (28 models)**. Reusing a
parity partner's model, as H₃O⁺ did, is not possible here: the partner's resonances sit 36–76
half-linewidths away (legacy measurement, D14), where a surrogate predicts "nothing happens".

## 8. Not modelled

| omitted | size | consequence |
|---|---|---|
| Raman light shifts | ~Ω_R/2 ≈ 1.5 kHz, m-dependent | 7× the J = 3 rung anharmonicity; must be calibrated out before rung-resolved control is an experimental claim |
| sign of g_F | unmeasured (heff OPEN-4) | the theory-supported negative branch is used; flipping it maps m → −m and swaps σ⁺/σ⁻ (so the dark states swap) |
| hyperfine Ω-doubling e_Δ | 1–10 kHz (heff OPEN-8) | would make the doublet splitting F-dependent |
| rotational g_r | ≤ 0.6% of g_F (heff OPEN-7) | about 1% error on J ≥ 2 g-factors |
| J ≥ 8 | 0.38% of the 4 K population | absent from the prior and the dynamics |
| carriers and higher sidebands | dropped by the RWA | the two in-window carriers are masked |
| motion between pulses | re-cooled to ν = 0 before every pulse | every pulse uses only the ν = 0 transfer columns |
| coherence between pulses | analytic counterexamples exist; ≤ 1.2e-9 on sampled ThF⁺ resonances (2026-09 review) | the diagonal belief update is an assumption |

## 9. Verification status

| check | status |
|---|---|
| tables reproduce from heff @ `0559bf1`, byte-exact except line 1 | ✅ `tests/test_thf_tables.py` |
| model A's claimed properties (parity, Δm_F, σ⁻ reversal, rank 2, e/f via G_Δ) | ✅ `tests/test_thf_tables.py` |
| exact propagation agrees with an independent matrix exponential | ✅ < 1e-8 on every block and polarisation, at every in-window resonance and random drives (`tests/test_thf_physics.py`). The reference parses the raw tables itself, takes sideband elements from Laguerre polynomials and uses `scipy.linalg.expm` |
| motional cutoff n_nu = 8 against 16 (which closes the J = 7 ladder) | ✅ over the full window and τ grid, worst transfer-matrix entry: 1.5e-3 (J = 7), 1.0e-3 (J = 6), 4.0e-4 (J = 5), 1.8e-5 (J = 4), ≤ 1e-10 (J ≤ 3). Over the action library at its own τ: 2.1e-4, 3.8e-7 on the thermal ν ≥ 1 probability, 3.8e-13 on the primitives |
| Boltzmann prior agrees with an independent SI calculation | ✅ 4e-11 relative (the 11-digit k_B/h in `physics/thermal.py`) |

The truncation error at n_nu = 8 is two orders of magnitude below any surrogate error measured so
far (worst basis errors ≥ 0.05), so the exact engine is a sound ground truth for training *within
the model*. These checks validate the implementation of the stated model, not the model:
sections 3 and 8 list what remains assumed.
