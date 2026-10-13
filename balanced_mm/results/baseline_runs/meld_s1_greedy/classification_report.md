# meld — Greedy + MMTM (project adaptation) — seed 1

Held-out test n=70; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.57 |
| macro precision | 35.96 |
| macro recall | 28.57 |
| macro f1 | 29.31 |
| weighted f1 | 29.31 |
| micro f1 | 28.57 |
| balanced accuracy | 28.57 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 30.77 | 40.00 | 34.78 | 10 |
| disgust | 12.00 | 30.00 | 17.14 | 10 |
| fear | 50.00 | 10.00 | 16.67 | 10 |
| joy | 42.86 | 30.00 | 35.29 | 10 |
| neutral | 50.00 | 40.00 | 44.44 | 10 |
| sadness | 28.57 | 20.00 | 23.53 | 10 |
| surprise | 37.50 | 30.00 | 33.33 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
