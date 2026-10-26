# Historical Tara ThF+ benchmark (2026-09-18)

These results were generated at commit `2a7ee18` and first published in
`412f511`. They use the older 192-state, J ≤ 6, seven-motional-level
ThF+ model and 24-checkpoint `mix` manifest. They are not benchmarks of the
current J ≤ 7, eight-motional-level upstream model. Old checkpoints must not
be reused with the new physical contract. The scripts named below are retained
as historical reproduction helpers and have not been ported or revalidated
against the current model.

The 24-checkpoint [`munozariasjm/thf_qls_fno`](https://huggingface.co/munozariasjm/thf_qls_fno)
manifest was evaluated on an NVIDIA H100 PCIe at commit `2a7ee18`.  The protocol uses the
312-action `physics_subset` library, target purity 0.98, an 800-pulse horizon and 200 exact-table
evaluation episodes.  PPO uses two million training transitions, training seed 0 and sampled
policy evaluation.  The horizon and evaluation mode must be explicit: ThF+'s repository default
is 80 pulses, and `PPOConfig.eval_greedy` defaults to `True`.

| controller | exact-evaluation success | mean pulses | P85 |
|---|---:|---:|---:|
| planner-exact | 1.000 | 50.8 | 90 |
| planner-fno | 1.000 | 51.3 | 98 |
| PPO-exact | 1.000 | 150.2 | 265 |
| PPO-fno | 0.990 | 124.0 | 220 |
| random | 0.930 | 323.4 | 619 |
| sweeping | 0.835 | 416.6 | not reached |

![ThF+ exact/FNO controller comparison](OUTPUTS/benchmark/thf/finished.png)

The planner pair closely reproduces the model-card result (exact/FNO: 53.1/51.3 mean pulses,
both 100% successful).  Fresh PPO runs reproduce the high-success conclusion but outperform the
model-card pulse counts; one training seed is not an exact numeric reproduction of a saved actor.
The PPO-FNO completion curve has 99.0% final completion and an 80-pulse median among successful
episodes.

![PPO-FNO belief and completion trajectories](OUTPUTS/benchmark/thf/rl_fno_trajectory_and_completion.png)

Run the PPO-FNO arm with:

```bash
export QLSGYM_WORK=/path/to/qlsgym_work
python scripts/benchmark.py run thf rl-fno \
  --device cuda --fno-tag mix --n-episodes 200 --seed 777 \
  --max-pulses 800 --p-target 0.98 \
  --ppo total_steps=2000000 --ppo seed=0 --ppo eval_greedy=false
```

All six JSON records, the merged completion curves and PDF figures are under
[`OUTPUTS/benchmark/thf`](OUTPUTS/benchmark/thf).  `scripts/thf_reproduction_figures.py` regenerates
the slide-style figures from the records and the selected PPO-FNO actor.

For this fixed action library, exact PPO uses cached transfer tables rather than an `eigh` per
pulse.  On Tara, end-to-end PPO-exact took 72.5 s and PPO-FNO took 1073.7 s, so the surrogate was
14.8 times slower than cached exact dynamics in this workload.  The FNO remains useful when the
control set changes or cannot be precomputed.  Planner timing is not directly comparable because
the exact reference was vectorized over cached actions and the FNO evaluation was split into three
Monte Carlo shards.
