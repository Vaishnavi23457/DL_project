# food101 — OGM-GE (project) — seed 0

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 44.00 |
| macro precision | 49.14 |
| macro recall | 44.00 |
| macro f1 | 42.21 |
| weighted f1 | 42.21 |
| micro f1 | 44.00 |
| balanced accuracy | 44.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 16.67 | 20.00 | 18.18 | 5 |
| chocolate_cake | 75.00 | 60.00 | 66.67 | 5 |
| french_fries | 36.36 | 80.00 | 50.00 | 5 |
| fried_rice | 30.00 | 60.00 | 40.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 50.00 | 80.00 | 61.54 | 5 |
| pizza | 66.67 | 40.00 | 50.00 | 5 |
| ramen | 50.00 | 20.00 | 28.57 | 5 |
| sushi | 66.67 | 40.00 | 50.00 | 5 |
| tacos | 100.00 | 40.00 | 57.14 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
