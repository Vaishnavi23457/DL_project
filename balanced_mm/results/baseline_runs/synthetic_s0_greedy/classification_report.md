# synthetic — Greedy + MMTM (project adaptation) — seed 0

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 52.00 |
| macro precision | 68.05 |
| macro recall | 52.00 |
| macro f1 | 53.27 |
| weighted f1 | 53.27 |
| micro f1 | 52.00 |
| balanced accuracy | 52.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 30.14 | 88.00 | 44.90 | 25 |
| 1 | 58.82 | 40.00 | 47.62 | 25 |
| 2 | 94.12 | 64.00 | 76.19 | 25 |
| 3 | 90.91 | 40.00 | 55.56 | 25 |
| 4 | 77.78 | 28.00 | 41.18 | 25 |
| 5 | 56.52 | 52.00 | 54.17 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
