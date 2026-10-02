# Jose PPO comparison under the locked ThF+ contract

All rows use the downloaded `mix` FNO for training, `H=80`, `rho=0`,
5,000 FNO episodes, and 5,000 exact holdout episodes per seed.
Lower unfinished fraction and failure-penalized actions are better.

| PPO | seeds | selection | exact failure | exact actions | FNO failure | paired failure delta vs branch |
|---|---:|---|---:|---:|---:|---:|
| Branch PPO | 5 | exact-final-only | 73.89% +/- 2.37% | 68.15 +/- 1.01 | 73.84% | +0.00 pp |
| Jose GAE matched | 5 | fno | 81.21% +/- 2.68% | 72.10 +/- 1.24 | 80.50% | +7.32 pp |
| Jose PPO FNO-selected | 5 | fno | 63.74% +/- 3.64% | 63.98 +/- 0.89 | 62.82% | -10.14 pp |
| Jose PPO exact-selected | 3 | exact | 64.39% +/- 0.76% | 64.02 +/- 1.02 | 63.57% | -7.89 pp |
| Jose-tuned PPO FNO-selected | 5 | fno | 63.91% +/- 4.07% | 65.89 +/- 1.74 | 60.51% | -9.98 pp |
