# meld — ModDrop (feature adaptation) — seed 1

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 20.00 |
| macro precision | 16.24 |
| macro recall | 20.00 |
| macro f1 | 15.34 |
| weighted f1 | 15.34 |
| micro f1 | 20.00 |
| balanced accuracy | 20.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 25.00 | 10.00 | 14.29 | 10 |
| disgust | 22.22 | 20.00 | 21.05 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 25.00 | 10.00 | 14.29 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 15.15 | 50.00 | 23.26 | 10 |
| surprise | 26.32 | 50.00 | 34.48 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
