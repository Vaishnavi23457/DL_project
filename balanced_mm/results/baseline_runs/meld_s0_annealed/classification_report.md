# meld — Annealed dropout — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.29 |
| macro precision | 26.32 |
| macro recall | 24.29 |
| macro f1 | 22.76 |
| weighted f1 | 22.76 |
| micro f1 | 24.29 |
| balanced accuracy | 24.29 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 20.00 | 40.00 | 26.67 | 10 |
| disgust | 16.67 | 20.00 | 18.18 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 35.71 | 50.00 | 41.67 | 10 |
| neutral | 37.50 | 30.00 | 33.33 | 10 |
| sadness | 7.69 | 10.00 | 8.70 | 10 |
| surprise | 66.67 | 20.00 | 30.77 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
