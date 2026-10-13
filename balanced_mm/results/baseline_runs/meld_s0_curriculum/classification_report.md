# meld — Curriculum dropout — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 34.80 |
| macro recall | 27.14 |
| macro f1 | 24.99 |
| weighted f1 | 24.99 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 25.00 | 40.00 | 30.77 | 10 |
| disgust | 18.18 | 20.00 | 19.05 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 41.67 | 50.00 | 45.45 | 10 |
| neutral | 44.44 | 40.00 | 42.11 | 10 |
| sadness | 14.29 | 30.00 | 19.35 | 10 |
| surprise | 100.00 | 10.00 | 18.18 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
