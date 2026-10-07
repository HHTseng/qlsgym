# Quantum-logic-spectroscopy state-preparation gym

This is the code for running and developing the ML techniques for QLS state preparation,
the initial basis is (https://arxiv.org/pdf/2608.03702).

The package is structured as a Gymnasium environment and with access to the effective
Hamiltonian and training of neural operators for the dynamic surrogate.
The package is designed to be modular and extensible, allowing for integration of
 new molecules, action libraries, and dynamics surrogates.

Currently we support the ThF⁺ and H₃O⁺ molecules, and the exact, CUDA-Q or FNO dynamics engines.

**Status (2026-10-07): fresh start on ThF⁺.** Every previously trained surrogate and policy was
deleted; no checkpoint manifests exist until the ThF⁺ pipeline is retrained from scratch, so the
`*-fno` arms and `load_manifest` have nothing to load yet. The conclusions of the H₃O⁺
replication are kept in [docs/h3o_fno_findings.md](docs/h3o_fno_findings.md); H₃O⁺ stays as a
molecule pack but is not being developed.

## The problem

A molecular ion sits in a thermal mixture over its hyperfine states.
The agent applies Raman sideband pulses (`sigma`, `omega`, `tau`); a logic ion
then reads out `nu = 0` or `nu >= 1`. The belief over molecular populations is
Bayes-updated to the corresponding branch, sampled from the belief's own
predictive distribution (arXiv:2410.11839 Sec. II.2).

The episode ends when one population reaches
`p_target` (purity, default 0.98) or after `max_pulses` pulses.

## Hamiltonian Generation and analysis

The ThF⁺ tables in `data/thf/` are built from the effective Hamiltonian package heff
(https://github.com/arianjad/heff/tree/main) by `qlsgym.molecules.thf_tables`:

```bash
source scripts/env.sh                  # puts $HEFF_ROOT on PYTHONPATH
python scripts/build_thf_tables.py     # rewrites data/thf/*.txt; every trained surrogate must then be retrained
```

[docs/thf_system.md](docs/thf_system.md) documents the molecule end to end: the heff parameters and
their status, coupling model A, the trap, window, blocks, action library and what is not modelled.
`tests/test_thf_tables.py` and `tests/test_thf_physics.py` check it.

## Install

```bash
source scripts/env.sh
```

Activates the conda env and sets `QLSGYM_ROOT` and `QLSGYM_WORK` (caches, training splits, runs,
checkpoint manifests; defaults to netscratch, purged after 90 days). For a new env:
`pip install -e .[gym,fno]` for `gymnasium` and `neuraloperator`/`tensorly`.

## Quickstart

For the exact dynamics:

```python
from qlsgym import load_molecule
from qlsgym.physics.engines import ExactEngine
from qlsgym.env.actions import ActionLibrary
from qlsgym.env.cache import build_action_tables
from qlsgym.env.env import PurificationEnv

mol = load_molecule("thf")                          # or "h3o"
library = ActionLibrary.physics_subset(mol)          # or ActionLibrary(mol) for the full grid
tables = build_action_tables(mol, library)           # engine=None -> qlsgym.physics
env = PurificationEnv(mol, library, tables, batch=64)
transition = env.step(env.torch.randint(0, library.n_actions, (64,)))
```

For the FNO surrogate (instead of exact propagation), once a `mix` manifest has been trained
(`scripts/train_fno.py` per block, then `scripts/make_manifest.py --molecule thf --tag mix`):

```python
import numpy as np
from qlsgym.surrogate.manifest import load_manifest

engine = load_manifest(mol, "mix")                   # FnoEngine
belief = np.full(mol.n_states, 1 / mol.n_states)
p0, p1 = engine.branches_all_tau(belief, mol.trap.nu_f, "+")   # (n_tau, n_states) each
```

The code also integrates several engines for the dynamics:

| engine | how it gets there |
|---|---|
| `ExactEngine` | `eigh` of each block Hamiltonian, then `U(tau)` for the whole tau grid |
| `CudaqEngine` | `cudaq.evolve` on the `dynamics` (cuDensityMat) target: fixed-step RK4 on a GPU |
| `FnoEngine` | the trained surrogate, with one of the above as fallback |

To select one:

```python
from qlsgym.physics.engines import select_engine

engine = select_engine(mol, tau_indices)       # CUDA-Q if it can run here, else exact
engine.selection            # {'requested': 'auto', 'selected': ..., 'reason': ...}
```

`QLSGYM_ENGINE=auto|cudaq|exact`, `select_engine(prefer=...)` and `benchmark.py --engine` all
override it: `cudaq` raises rather than falling back, `exact` never probes for a GPU. `auto`
takes CUDA-Q only when the package is installed, a CUDA device is visible and
`set_target("dynamics")` succeeds, and prints the engine and the reason either way -- the
fallback is never silent. The choice, the integrator settings and the RK step count go into the
benchmark JSON under `engine`.

### How the CUDA-Q engine works

`BlockOperator.build` is affine in the drive frequency, so each sector is registered once as
`H(omega) = H_ref + (omega - omega_ref) dH/domega` -- two dense matrices shared by every drive on
that sector, not one per drive. A single `evolve` call per `(sector, sigma, omega)` propagates
the `nu = 0` basis states along a schedule that *is* the tau grid and keeps the intermediate
states, so the whole tau library comes from one integration and is then cached like the exact
engine's columns.

The integrator is fixed-step RK4 and CUDA-Q's own default step diverges here, so the step is
never left implicit. The knob is `atol`, the population the integration may lose over one pulse
(default `1e-6`): `step_size` turns it into a step per drive from `||H||_inf` and the pulse
length, capped by `max_step_size = 3e-5` ms. Every set of columns is checked against that budget
and a larger loss raises instead of returning a number. Measured against `ExactEngine` over the
production tau grid, worst block, window edges:

| molecule | dt (ms) | RK steps per drive | norm lost | max abs population error |
|---|---|---|---|---|
| H3O+ (dim 128, 4 ms) | 1.7e-5 | 2.3e5 | 1.0e-6 | 2.5e-8 |
| ThF+ (J ≤ 6, n_nu = 7: dim 182, 6 ms) | 1.7e-7 | 3.6e7 | 1.0e-6 | 3.3e-7 |

### Which one to use

**Exact, today, on both molecules.** Per RK step CUDA-Q costs `O(dim^2 M)` against `O(dim^3)`
once per tau for `eigh`, so it does `4 n_steps / (n_nu n_tau)` -- about 2300x for H3O+ -- more
arithmetic, and that ratio does not improve with `dim` because both sides are `O(dim^3)`. At
FNO_REPL's measured 0.65 ms per RK step on an A100 that is ~150 s per drive for H3O+ and ~6.5 h
for ThF+, against 0.2 s and 0.5 s for the whole exact tau grid on one CPU core;
`scripts/slurm/benchmark.sbatch` pins `ENGINE=exact` for that reason.

Nor is memory what it fixes: `operators.define` takes a dense `dim x dim` matrix, and the saved
all-tau states are the same `O(n_tau dim M)` the exact path already stores. The asymptotic win
needs the Hamiltonian expressed **sparsely** (ours has `O(dim)` non-zeros), which makes a step
`O(dim M)` and puts the crossover against `eigh` near `dim ~ 2e3`. Until then CUDA-Q buys a
workload with no dense eigendecomposition in it, not speed.

A single-trajectory Gymnasium env (`pip install qlsgym[gym]`):

```python
from qlsgym.env.gym import QLSGymEnv
gym_env = QLSGymEnv(mol, library, tables)
obs, info = gym_env.reset(seed=0)
```

Reference policies and Monte-Carlo evaluation:

```python
from qlsgym.policies.baselines import PhysicsEliminationPolicy, ScorePlannerPolicy
from qlsgym.policies.rollout import rollout

result = rollout(env, PhysicsEliminationPolicy(library))   # or rollout(engine, policy, library=library)
print(result.summary())
```

## Benchmark

One plot, six arms, the comparison of arXiv:2410.11839 Fig. 4c / Fig. S7: the percentage of
finished episodes against the number of pulses applied. Results and figures land in
`OUTPUTS/benchmark/<molecule>/` (`QLSGYM_OUTPUTS` moves them elsewhere).

### Runbook

Both molecules run the same way -- the horizon, the surrogate tag and the evaluation mode come
from the molecule, so only the name changes, and `run` prints the three it resolved:

```bash
source scripts/env.sh
for ARM in sweeping random planner-exact rl-exact rl-fno; do
  python scripts/benchmark.py run thf "$ARM" --n-episodes 200      # or h3o
done
python scripts/benchmark.py plot  thf        # OUTPUTS/benchmark/thf/finished.{png,pdf} + table
python scripts/benchmark.py table thf
```

`planner-fno` is the slow arm (one surrogate call per candidate pulse per step), so it is chunked
over a SLURM array and merged. Each part takes its own evaluation seed (`SEED + 1000 * part`) and
`merge` refuses parts that share one:

```bash
ARM=planner-fno MOL=thf NEPS=20 SEED=880777 sbatch --array=0-9 --partition=sapphire \
    --time=8:00:00 --cpus-per-task=4 --mem=32G scripts/slurm/benchmark.sbatch
python scripts/benchmark.py merge thf planner-fno    # thf/planner-fno.part*.json -> planner-fno.json
python scripts/benchmark.py plot  thf
```

`benchmark.sbatch` requires `ARM` and `MOL` and defaults everything else; `OUTROOT` is the only
way results leave `OUTPUTS/benchmark/`. `plot`/`table` never skip an arm quietly: part files with
no merged `<arm>.json`, and results whose `fingerprint` is not the molecule's current one, are
reported loudly and left out.

The RL arms want a GPU:

```bash
ARM=rl-fno MOL=thf NEPS=200 DEVICE=cuda sbatch --partition=gpu \
    --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=12:00:00 --cpus-per-task=8 --mem=64G \
    scripts/slurm/benchmark.sbatch
```

### What differs between the molecules

| | H3O+ | ThF+ |
|---|---|---|
| `--max-pulses` (horizon) | 400 | 800 |
| `physics_subset` library | 167 actions | 416 actions |
| `--ppo eval_greedy` | `true` | `false` |

* **Horizon.** One full sweep of the action library has to fit, so the horizon is not
  `molecule.task.max_pulses` (80 for both -- the *task* budget, drawn as the figure's vertical
  guide). `scripts/slurm/benchmark.sbatch` leaves `MAXPULSES` unset and lets these apply.
* **Tag.** `mix` unless `--fno-tag` says otherwise; an FNO arm without that manifest refuses to
  run rather than picking another tag.
* **Evaluation mode.** The argmax of ThF+'s 312-action stochastic policy (J ≤ 6 model) locks onto a single pulse
  (0.005 success against 0.985 sampled), so ThF+ is evaluated by sampling. Every run records the
  mode it used.

### Current numbers

None: the previous results were produced with surrogates that have since been deleted, and will be
regenerated by the ThF+ pipeline.

## Repo structure

```
src/qlsgym/
  spec.py          the contract: Molecule, Block, TauBatchedEngine
  molecules/       h3o, thf, synthetic -- build a Molecule from data/; thf_tables builds data/thf from heff
  physics/         exact dynamics: hamiltonian, propagate, spectrum, engines, cudaq_engine
  surrogate/       FNO: embedding, model, dataset, train, manifest, in-loop engine
  env/             action library, transfer-matrix cache, PurificationEnv, gym wrapper
  policies/        sweeping, physics elimination, score planner, rollout driver
  rl/              PPO
  benchmark/       arm protocol, finished-episodes curve, figure

data/              level energies and two-photon Rabi rates (h3o, thf)
scripts/           env.sh, build_thf_tables.py, train_fno.py, make_manifest.py, benchmark.py, slurm/
examples/          01 env, 02 baselines, 03 FNO engine
tests/             pytest suite
docs/              thf_system.md (the ThF+ model, end to end); h3o_fno_findings.md (H3O+ replication)
OUTPUTS/           benchmark results (JSON + figures), created by scripts/benchmark.py
```

Caches, training splits go to `$QLSGYM_WORK`.
