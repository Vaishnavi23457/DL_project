# meld — Annealed dropout — seed 1

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 25.71 |
| macro precision | 31.00 |
| macro recall | 25.71 |
| macro f1 | 20.33 |
| weighted f1 | 20.33 |
| micro f1 | 25.71 |
| balanced accuracy | 25.71 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 22.58 | 70.00 | 34.15 | 10 |
| disgust | 42.86 | 30.00 | 35.29 | 10 |
| fear | 20.00 | 10.00 | 13.33 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 100.00 | 10.00 | 18.18 | 10 |
| surprise | 31.58 | 60.00 | 41.38 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
