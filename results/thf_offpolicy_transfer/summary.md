# Branch-aware off-policy transfer under the locked ThF+ contract

Each row uses five training seeds and 5,000 episodes per dynamics and seed.
Lower failure and failure-penalized actions are better.

| Agent | Exact failure | Exact actions | FNO failure | failure vs branch | actions vs branch | wins |
|---|---:|---:|---:|---:|---:|---:|
| SAC refined, 2M | 33.93% +/- 0.56% | 46.70 +/- 1.04 | 32.89% | -35.48 pp | -19.38 | 5/5 |
| DDQN raw-belief scaled, 1M | 92.46% +/- 3.99% | 74.46 +/- 3.37 | 92.56% | -6.10 pp | -4.42 | 5/5 |
