# meld — OPM (project) — seed 1

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 20.00 |
| macro precision | 20.52 |
| macro recall | 20.00 |
| macro f1 | 11.25 |
| weighted f1 | 11.25 |
| micro f1 | 20.00 |
| balanced accuracy | 20.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 26.67 | 40.00 | 32.00 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 16.98 | 90.00 | 28.57 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 100.00 | 10.00 | 18.18 | 10 |
| surprise | 0.00 | 0.00 | 0.00 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
