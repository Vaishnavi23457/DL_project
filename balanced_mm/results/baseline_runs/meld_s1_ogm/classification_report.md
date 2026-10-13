# meld — OGM-GE (project) — seed 1

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 21.62 |
| macro recall | 27.14 |
| macro f1 | 22.64 |
| weighted f1 | 22.64 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 20.00 | 10.00 | 13.33 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 30.77 | 40.00 | 34.78 | 10 |
| joy | 20.00 | 20.00 | 20.00 | 10 |
| neutral | 33.33 | 80.00 | 47.06 | 10 |
| sadness | 22.22 | 20.00 | 21.05 | 10 |
| surprise | 25.00 | 20.00 | 22.22 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
