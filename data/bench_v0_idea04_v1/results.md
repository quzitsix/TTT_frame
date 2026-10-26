# Idea4 finite-response diagnostic

This is an Idea1 residual-MLP teacher prototype, not the complete Idea4 or official Spatial-TTT.
Warm20/copy/oracle require teacher responses; only the full arms start from zero and operate without a teacher at deployment.

| Arm | Validation | Exploratory test |
|---|---:|---:|
| teacher | 14/32 | 24/57 |
| recent_only | 14/32 | 24/57 |
| fixed_full | 14/32 | 24/57 |
| fixed_warm20 | 14/32 | 24/57 |
| fixed_copy | 14/32 | 24/57 |
| fixed_oracle | 14/32 | 24/57 |
| conditioned_full | 14/32 | 24/57 |
| conditioned_warm20 | 14/32 | 24/57 |
| conditioned_copy | 14/32 | 24/57 |
| conditioned_oracle | 14/32 | 24/57 |

All200 forecasts are frozen before a separate score stage opens silver answer labels.
Teacher trajectories (2.5GB) and full forecast tensors stay in the source run; their SHA256 values are archived.
The probe/model checkpoints are included. See docs/IDEA04_BENCH_V0_RESULTS.md for limitations and reproduction.
