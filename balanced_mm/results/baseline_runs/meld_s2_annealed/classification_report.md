# meld — Annealed dropout — seed 2

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 22.86 |
| macro precision | 26.40 |
| macro recall | 22.86 |
| macro f1 | 17.65 |
| weighted f1 | 17.65 |
| micro f1 | 22.86 |
| balanced accuracy | 22.86 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 20.00 | 70.00 | 31.11 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 100.00 | 20.00 | 33.33 | 10 |
| sadness | 14.81 | 40.00 | 21.62 | 10 |
| surprise | 50.00 | 30.00 | 37.50 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
