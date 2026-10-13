# food101 — Dropout — seed 2

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 36.00 |
| macro precision | 46.27 |
| macro recall | 36.00 |
| macro f1 | 31.73 |
| weighted f1 | 31.73 |
| micro f1 | 36.00 |
| balanced accuracy | 36.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 29.41 | 100.00 | 45.45 | 5 |
| chocolate_cake | 50.00 | 80.00 | 61.54 | 5 |
| french_fries | 25.00 | 40.00 | 30.77 | 5 |
| fried_rice | 50.00 | 20.00 | 28.57 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 25.00 | 40.00 | 30.77 | 5 |
| pizza | 100.00 | 20.00 | 33.33 | 5 |
| ramen | 33.33 | 20.00 | 25.00 | 5 |
| sushi | 100.00 | 20.00 | 33.33 | 5 |
| tacos | 50.00 | 20.00 | 28.57 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
