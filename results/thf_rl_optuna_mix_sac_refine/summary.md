# ThF+ SAC refinement from the successful H3O+ Optuna screen

The H3O+ result is a validation-only S17 screen on exact tables, so its score is not directly comparable to ThF+. This refinement transfers its optimizer settings while keeping the downloaded ThF+ `mix` FNO and the S18 two-branch target fixed.

| Run | Exact failure | Exact actions | FNO failure | FNO actions |
|---|---:|---:|---:|---:|
| Locked SAC baseline | 69.41% | 66.07 | 72.23% | 68.03 |
| Previous Optuna SAC | 80.68% | 71.88 | 79.31% | 71.88 |
| Refined SAC | 57.07% | 59.85 | 56.59% | 59.49 |

Positive deltas below mean improvement over the locked baseline:

- exact failure: +12.34 pp
- exact actions: +6.22
- FNO failure: +15.64 pp
- FNO actions: +8.53

## Selected configuration

```json
{
  "alpha": 0.013268918812005082,
  "autotune_alpha": true,
  "batch_size": 512,
  "buffer_size": 100000,
  "depth": 1,
  "gamma": 0.995,
  "gradient_steps": 1,
  "hidden": 128,
  "learning_starts": 1000,
  "lr": 0.00015047792717528454,
  "n_envs": 16,
  "obs": "sqrt",
  "return_scale": 20.0,
  "seed": null,
  "target_entropy_ratio": 0.49969706052497553,
  "tau": 0.003048864797383012,
  "total_steps": 300000,
  "train_freq": 1
}
```

## Figures

- `sac_refine_history.png`: paired-seed Optuna history.
- `sac_refined_vs_prior.png`: locked baseline, previous Optuna, and refined SAC.
- `sac_refine_importance.png`: focused-study parameter importance.
