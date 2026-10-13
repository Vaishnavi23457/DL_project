# food101 — OPM (project) — seed 2

Held-out test n=50; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 22.00 |
| macro precision | 28.80 |
| macro recall | 22.00 |
| macro f1 | 17.89 |
| weighted f1 | 17.89 |
| micro f1 | 22.00 |
| balanced accuracy | 22.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 18.18 | 40.00 | 25.00 | 5 |
| chocolate_cake | 26.67 | 80.00 | 40.00 | 5 |
| french_fries | 0.00 | 0.00 | 0.00 | 5 |
| fried_rice | 100.00 | 20.00 | 33.33 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 25.00 | 20.00 | 22.22 | 5 |
| pizza | 18.18 | 40.00 | 25.00 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 0.00 | 0.00 | 0.00 | 5 |
| tacos | 100.00 | 20.00 | 33.33 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
