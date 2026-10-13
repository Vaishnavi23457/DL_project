# synthetic — Greedy + MMTM (project adaptation) — seed 1

Held-out test n=150; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 41.33 |
| macro precision | 68.39 |
| macro recall | 41.33 |
| macro f1 | 36.48 |
| weighted f1 | 36.48 |
| micro f1 | 41.33 |
| balanced accuracy | 41.33 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 37.50 | 48.00 | 42.11 | 25 |
| 1 | 100.00 | 16.00 | 27.59 | 25 |
| 2 | 100.00 | 28.00 | 43.75 | 25 |
| 3 | 100.00 | 4.00 | 7.69 | 25 |
| 4 | 33.33 | 84.00 | 47.73 | 25 |
| 5 | 39.53 | 68.00 | 50.00 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
