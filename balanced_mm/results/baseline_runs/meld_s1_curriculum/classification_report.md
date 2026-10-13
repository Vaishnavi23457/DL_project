# meld — Curriculum dropout — seed 1

Held-out test n=70; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 21.43 |
| macro precision | 20.01 |
| macro recall | 21.43 |
| macro f1 | 17.42 |
| weighted f1 | 17.42 |
| micro f1 | 21.43 |
| balanced accuracy | 21.43 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 26.09 | 60.00 | 36.36 | 10 |
| disgust | 28.57 | 40.00 | 33.33 | 10 |
| fear | 16.67 | 10.00 | 12.50 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 50.00 | 10.00 | 16.67 | 10 |
| surprise | 18.75 | 30.00 | 23.08 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
