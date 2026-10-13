# meld — OGM-GE (project) — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 25.71 |
| macro precision | 41.06 |
| macro recall | 25.71 |
| macro f1 | 25.28 |
| weighted f1 | 25.28 |
| micro f1 | 25.71 |
| balanced accuracy | 25.71 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 26.32 | 50.00 | 34.48 | 10 |
| disgust | 13.33 | 20.00 | 16.00 | 10 |
| fear | 100.00 | 10.00 | 18.18 | 10 |
| joy | 57.14 | 40.00 | 47.06 | 10 |
| neutral | 27.27 | 30.00 | 28.57 | 10 |
| sadness | 13.33 | 20.00 | 16.00 | 10 |
| surprise | 50.00 | 10.00 | 16.67 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
