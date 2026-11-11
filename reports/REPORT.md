# NER experiment report

Four best validation checkpoints evaluated on the official test split after freezing the winner.

Selected model: **distilbert** using the predeclared validation rule.

| Model | Validation F1 | Test F1 | Parameters | Train seconds | Peak CUDA allocated MB |
|---|---:|---:|---:|---:|---:|
| distilbert | 0.939164 | 0.9017841370782547 | 65197833 | 198.8 | 1732.4 |
| lstm | 0.847191 | 0.7863247863247863 | 1307301 | 283.0 | 102.4 |
| bilstm | 0.905123 | 0.8457408732999284 | 1426213 | 227.3 | 105.7 |
| bilstm_crf | 0.916186 | 0.8647972852295054 | 1426312 | 462.5 | 105.7 |

See comparison.json for per-type precision, recall, F1, support, timing, checkpoint sizes, gradients and learning history; CRF comparison files contain both corrections and regressions. Concurrent timing is not an isolated speed benchmark. Losses differ between CRF and cross-entropy and should not be directly ranked.

No optimizer-state resume is implemented. Completed compatible models are reusable after manifest verification. Interrupted models restart. Smoke and full have separate experiment identities.
