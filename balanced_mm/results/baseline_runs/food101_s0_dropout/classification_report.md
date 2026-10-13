# food101 — Dropout — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 38.00 |
| macro precision | 44.27 |
| macro recall | 38.00 |
| macro f1 | 35.38 |
| weighted f1 | 35.38 |
| micro f1 | 38.00 |
| balanced accuracy | 38.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 23.08 | 60.00 | 33.33 | 5 |
| chocolate_cake | 57.14 | 80.00 | 66.67 | 5 |
| french_fries | 25.00 | 40.00 | 30.77 | 5 |
| fried_rice | 33.33 | 40.00 | 36.36 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 37.50 | 60.00 | 46.15 | 5 |
| pizza | 50.00 | 20.00 | 28.57 | 5 |
| ramen | 50.00 | 20.00 | 28.57 | 5 |
| sushi | 66.67 | 40.00 | 50.00 | 5 |
| tacos | 100.00 | 20.00 | 33.33 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
