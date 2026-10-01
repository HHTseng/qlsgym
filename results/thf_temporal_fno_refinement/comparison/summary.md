# Temporal FNO six-pair pilot

| model | pairs passing all gates | worst branch-mass P95 | worst conditional TV (mass >= 1e-2) | median derivative error | median spectral error |
|---|---:|---:|---:|---:|---:|
| column_retrained | 0/6 | 0.100618 | 0.450368 | 0.00318966 | 0.000524184 |
| temporal_1layer | 0/6 | 0.0959031 | 0.410719 | 0.00353993 | 0.000964275 |
| temporal_2layer | 0/6 | 0.0956428 | 0.413443 | 0.00342445 | 0.00072586 |
| temporal_2layer_d1 | 0/6 | 0.0956636 | 0.412987 | 0.00322759 | 0.000529686 |
| temporal_2layer_d1spec | 0/6 | 0.0956518 | 0.412988 | 0.00338617 | 0.000656942 |
| shuffled_time | 0/6 | 0.040004 | 0.105074 | 0.00489331 | 0.00108473 |
| pure_transformer | 0/6 | 0.0478025 | 0.209986 | 0.0089726 | 0.0071123 |

Promotion: **no**

The full 24-pair study is permitted only after this pilot gate passes.
