# meld — Curriculum dropout — seed 2

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 27.53 |
| macro recall | 27.14 |
| macro f1 | 24.89 |
| weighted f1 | 24.89 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 21.43 | 30.00 | 25.00 | 10 |
| disgust | 11.11 | 10.00 | 10.53 | 10 |
| fear | 33.33 | 10.00 | 15.38 | 10 |
| joy | 33.33 | 10.00 | 15.38 | 10 |
| neutral | 50.00 | 70.00 | 58.33 | 10 |
| sadness | 20.00 | 20.00 | 20.00 | 10 |
| surprise | 23.53 | 40.00 | 29.63 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
