# synthetic — Greedy + MMTM (project adaptation) — seed 2

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 40.00 |
| macro precision | 54.85 |
| macro recall | 40.00 |
| macro f1 | 38.93 |
| weighted f1 | 38.93 |
| micro f1 | 40.00 |
| balanced accuracy | 40.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 53.85 | 28.00 | 36.84 | 25 |
| 1 | 50.00 | 24.00 | 32.43 | 25 |
| 2 | 83.33 | 20.00 | 32.26 | 25 |
| 3 | 72.73 | 32.00 | 44.44 | 25 |
| 4 | 27.27 | 84.00 | 41.18 | 25 |
| 5 | 41.94 | 52.00 | 46.43 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
