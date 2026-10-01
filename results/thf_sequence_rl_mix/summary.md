# ThF+ sequence-aware PPO/SAC study

Training dynamics: fixed downloaded `mix` FNO. Candidate selection uses FNO validation; exact cached dynamics are a held-out transfer audit.

| agent | encoder | exact unfinished | exact actions | delta failure vs MLP | delta actions vs MLP | FNO unfinished | parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| ppo | mlp | 0.275 | 42.91 | +0.00 pp | +0.00 | 0.255 | 259,385 |
| ppo | gru_k8 | 0.738 | 62.51 | +46.25 pp | +19.60 | 0.785 | 206,521 |
| ppo | transformer_ln_k8_state_only | 0.893 | 73.07 | +61.82 pp | +30.16 | 0.897 | 372,665 |
| ppo | transformer_k8_state_only | 0.905 | 74.91 | +62.98 pp | +32.00 | 0.901 | 372,409 |
| sac | mlp | 0.444 | 52.46 | +0.00 pp | +0.00 | 0.433 | 194,857 |
| sac | transformer_k8_state_only | 0.558 | 56.99 | +11.48 pp | +4.53 | 0.588 | 1,116,841 |
| sac | stack_k4 | 0.584 | 59.17 | +14.03 pp | +6.71 | 0.637 | 517,417 |
