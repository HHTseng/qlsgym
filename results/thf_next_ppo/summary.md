# PPO beyond Jose under the locked ThF+ contract

All final rows use five training seeds, 5,000 FNO episodes, and 5,000 exact
holdout episodes per seed. Lower failure and failure-penalized actions are better.

| PPO | exact failure | exact actions | FNO failure | paired failure vs Jose | paired actions vs Jose | wins |
|---|---:|---:|---:|---:|---:|---:|
| Jose PPO | 63.74% +/- 3.64% | 63.98 +/- 0.89 | 62.82% | -- | -- | -- |
| qMDP tuned, 2M | 34.35% +/- 5.65% | 47.28 +/- 2.69 | 31.10% | -29.40 pp | -16.70 | 5/5 |
| qMDP tuned, lr=0.001 | 37.78% +/- 2.93% | 50.52 +/- 1.64 | 35.52% | -25.97 pp | -13.46 | 5/5 |
