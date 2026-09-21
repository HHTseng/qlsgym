# Final FNO + RL superiority decision

All rankings use cached-exact dynamics, unfinished fraction first and failure-penalized actions second.

| Rank | Controller | Class | Exact failure | Exact actions | Inference requirement |
|---:|---|---|---:|---:|---|
| 1 | Exact candidate arbiter | hybrid | 16.37% | 38.70 | actor and baseline proposals scored by cached-exact tables |
| 2 | 15-pulse PPO + descending fallback | hybrid | 18.40% | 41.31 | actor prefix then cached-exact table rule |
| 3 | Descending population | non-ML | 19.98% | 40.00 | cached-exact table rule |
| 4 | Failure-sensitive PPO + fallback | hybrid | 20.72% | 45.25 | actor then physics fallback |
| 5 | Failure-sensitive PPO | standalone RL | 24.92% | 45.20 | actor only |
| 6 | Physics elimination | non-ML | 29.36% | 51.49 | physics rule |
| 7 | FNO_RL_optuna PPO | standalone RL | 41.78% | 51.37 | actor only |
| 8 | Coverage-balanced sweep | non-ML | 87.60% | 79.11 | fixed schedule |

The exact candidate arbiter beats descending population by 3.61 failure percentage points and 1.30 actions. Its paired failure-difference 95% seed interval is [-4.75, -2.47] percentage points.

The column-v2 FNO passes all seven gates for 10/24 pairs, so it was not promoted and no policy was trained under it.

![Final exact comparison](final_exact_comparison.png)
