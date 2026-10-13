# meld — MMTM feature-level control — seed 1

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 31.43 |
| macro precision | 36.73 |
| macro recall | 31.43 |
| macro f1 | 27.47 |
| weighted f1 | 27.47 |
| micro f1 | 31.43 |
| balanced accuracy | 31.43 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 25.00 | 80.00 | 38.10 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 25.00 | 50.00 | 33.33 | 10 |
| joy | 50.00 | 20.00 | 28.57 | 10 |
| neutral | 57.14 | 40.00 | 47.06 | 10 |
| sadness | 50.00 | 10.00 | 16.67 | 10 |
| surprise | 50.00 | 20.00 | 28.57 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
