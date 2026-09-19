## Original production FNO versus downloaded `mix` FNO

All entries use the same five-seed, one-million-transition, 5000-exact-rollout contract. Deltas are `mix - original`; negative is better for both exact metrics.

| Controller | Original exact actions | `mix` exact actions | Delta | Original exact failure | `mix` exact failure | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Sweeping | 79.97 | 79.97 | +0.00 | 99.92% | 99.92% | +0.00 pp |
| Random | 74.57 | 74.57 | +0.00 | 85.62% | 85.62% | +0.00 pp |
| Physics elimination | 51.49 | 51.49 | +0.00 | 29.36% | 29.36% | +0.00 pp |
| PPO | 69.07 | 68.15 | -0.92 | 76.75% | 73.89% | -2.86 pp |
| Discrete SAC | 66.83 | 66.07 | -0.76 | 71.69% | 69.41% | -2.28 pp |
| Double DQN | 80.00 | 78.88 | -1.12 | 100.00% | 98.56% | -1.44 pp |

Transfer failure gap is exact failure minus FNO failure. Negative values mean the FNO environment is pessimistic.

| Controller | Original transfer gap | `mix` transfer gap |
|---|---:|---:|
| Sweeping | +0.10 pp | +0.10 pp |
| Random | -2.84 pp | -0.58 pp |
| Physics elimination | -42.40 pp | -33.78 pp |
| PPO | -0.46 pp | +0.05 pp |
| Discrete SAC | -1.75 pp | -2.82 pp |
| Double DQN | +0.40 pp | -0.41 pp |
