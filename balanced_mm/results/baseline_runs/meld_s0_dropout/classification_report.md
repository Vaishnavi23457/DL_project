# meld — Dropout — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 22.86 |
| macro precision | 23.37 |
| macro recall | 22.86 |
| macro f1 | 20.32 |
| weighted f1 | 20.32 |
| micro f1 | 22.86 |
| balanced accuracy | 22.86 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 20.83 | 50.00 | 29.41 | 10 |
| disgust | 20.00 | 20.00 | 20.00 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 36.36 | 40.00 | 38.10 | 10 |
| neutral | 27.27 | 30.00 | 28.57 | 10 |
| sadness | 9.09 | 10.00 | 9.52 | 10 |
| surprise | 50.00 | 10.00 | 16.67 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
