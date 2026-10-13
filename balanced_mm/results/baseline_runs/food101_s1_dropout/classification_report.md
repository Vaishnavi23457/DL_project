# food101 — Dropout — seed 1

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 38.00 |
| macro precision | 45.43 |
| macro recall | 38.00 |
| macro f1 | 34.44 |
| weighted f1 | 34.44 |
| micro f1 | 38.00 |
| balanced accuracy | 38.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 30.77 | 80.00 | 44.44 | 5 |
| chocolate_cake | 40.00 | 80.00 | 53.33 | 5 |
| french_fries | 40.00 | 40.00 | 40.00 | 5 |
| fried_rice | 40.00 | 40.00 | 40.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 28.57 | 40.00 | 33.33 | 5 |
| pizza | 50.00 | 40.00 | 44.44 | 5 |
| ramen | 25.00 | 20.00 | 22.22 | 5 |
| sushi | 100.00 | 20.00 | 33.33 | 5 |
| tacos | 100.00 | 20.00 | 33.33 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
