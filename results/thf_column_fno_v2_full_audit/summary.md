# FNO manifest structural audit

Pairs passing every engineering gate: **10/24**.

| Gate | Worst value | Threshold | Worst pair | Pass |
|---|---:|---:|---|:---:|
| tau0_identity_tv_max | 0 | 0.001 | 0, - | yes |
| input_linearity_tv_p95 | 8.44219e-17 | 0.001 | 1, - | yes |
| off_joint_tv_p95 | 0.000944886 | 0.005 | 0, - | yes |
| branch_mass_abs_error_p95 | 0.0725426 | 0.005 | 11, + | no |
| conditional_tv_mass_ge_1e-2_p95 | 0.277755 | 0.05 | 11, + | no |
| conditional_tv_mass_ge_1e-3_p95 | 0.689024 | 0.1 | 11, + | no |
| block_local_termination_error | 0.107054 | 0.005 | 11, + | no |
