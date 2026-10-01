# Temporal FNO final result

Trained models: **48** over six block/polarization pairs; estimated training compute: **1.97 GPU-hours**.

| model | pairs passing all gates | branch-mass P95, worst | conditional TV P95, worst | on-resonance derivative error | on-resonance spectral error |
|---|---:|---:|---:|---:|---:|
| downloaded_mix | 0/6 | 0.0165931 | 0.125584 | 0.00206656 | 0.000199366 |
| column_v2 | 2/6 | 0.0725426 | 0.277755 | 0.00234047 | 0.000258726 |
| temporal_2layer_long | 4/6 | 0.0753247 | 0.329367 | 0.00234157 | 0.000172007 |

The matched-budget temporal model improves the easy pairs 0 and 1, raising complete gate passage from 2/6 to 4/6. It does not repair block 9/+ termination or the block 11/+ branch errors.

Full 24-pair training and new PPO/SAC training were not run because the preregistered pilot gate failed. Existing optimized PPO and refined SAC remain the valid RL references on the downloaded `mix` FNO.
