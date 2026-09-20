# ThF+ downloaded-mix FNO RL optimization

The FNO checkpoints and action library were fixed. Optuna used only FNO validation during the broad search. Three configurations per agent were promoted, each with two new training seeds. The selected configuration was retrained with five seeds and evaluated on 5,000 FNO and 5,000 exact holdout episodes per seed.

| Agent | Exact failure baseline | Exact failure optimized | Baseline − optimized | Exact actions baseline | Exact actions optimized | Baseline − optimized |
|---|---:|---:|---:|---:|---:|---:|
| ppo | 73.89% | 41.78% | +32.11 pp | 68.15 | 51.37 | +16.77 |
| sac_discrete | 69.41% | 80.68% | -11.27 pp | 66.07 | 71.88 | -5.80 |
| ddqn | 98.56% | 94.61% | +3.95 pp | 78.88 | 76.00 | +2.88 |

| Agent | FNO failure baseline | FNO failure optimized | Baseline − optimized | FNO actions baseline | FNO actions optimized | Baseline − optimized |
|---|---:|---:|---:|---:|---:|---:|
| ppo | 73.84% | 38.90% | +34.94 pp | 69.14 | 50.86 | +18.28 |
| sac_discrete | 72.23% | 79.31% | -7.08 pp | 68.03 | 71.88 | -3.85 |
| ddqn | 98.97% | 95.09% | +3.88 pp | 79.69 | 77.23 | +2.46 |

## Selected configurations

### ppo

Broad trial `71` / `ppo_t71`:

```json
{
  "clip": 0.1,
  "ent_coef": 0.0005417800781171913,
  "epochs": 8,
  "eval_every": 15,
  "eval_greedy": false,
  "eval_rollouts": 256,
  "gae_lambda": 0.98,
  "gamma": 1.0,
  "hidden": 512,
  "lr": 0.001346363908197788,
  "max_grad_norm": 1.0,
  "minibatches": 16,
  "n_envs": 128,
  "n_hidden_layers": 1,
  "n_steps": 32,
  "obs": "sqrt",
  "reward_scale": 0.025,
  "seed": 10071,
  "total_steps": 250000,
  "value_target": "qmdp",
  "vf_coef": 0.5
}
```

### sac_discrete

Broad trial `58` / `sac_discrete_t58`:

```json
{
  "alpha": 0.002026280986674276,
  "autotune_alpha": false,
  "batch_size": 256,
  "buffer_size": 125000,
  "depth": 1,
  "gamma": 0.995,
  "gradient_steps": 4,
  "hidden": 256,
  "learning_starts": 5120,
  "lr": 3.7034611947922746e-05,
  "n_envs": 128,
  "seed": 10058,
  "target_entropy_ratio": 0.7713394752098243,
  "tau": 0.027856499454340163,
  "total_steps": 250000,
  "train_freq": 2
}
```

### ddqn

Broad trial `72` / `ddqn_t72`:

```json
{
  "batch_size": 512,
  "buffer_size": 125000,
  "depth": 2,
  "eps_end": 0.013933264600237711,
  "eps_fraction": 0.5297244240558068,
  "eps_start": 1.0,
  "gamma": 0.99,
  "gradient_steps": 1,
  "hidden": 128,
  "learning_starts": 20480,
  "lr": 0.0003863012590717139,
  "n_envs": 128,
  "seed": 10072,
  "tau": 0.00419740649400854,
  "total_steps": 250000,
  "train_freq": 1
}
```

## Figures

- `optuna_history.png`: every completed broad trial and the best-so-far curve.
- `parameter_importance.png`: PED-ANOVA importance for the ten most influential settings.
- `optimized_vs_baseline.png`: locked baseline versus optimized final evaluation.

Positive changes in the tables mean improvement. Failed episodes are charged the full 80-action horizon, so average actions jointly reflects success and speed.
