# food101 — Evolutional dropout — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 32.00 |
| macro precision | 32.18 |
| macro recall | 32.00 |
| macro f1 | 29.21 |
| weighted f1 | 29.21 |
| micro f1 | 32.00 |
| balanced accuracy | 32.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 18.18 | 40.00 | 25.00 | 5 |
| chocolate_cake | 75.00 | 60.00 | 66.67 | 5 |
| french_fries | 20.00 | 40.00 | 26.67 | 5 |
| fried_rice | 0.00 | 0.00 | 0.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 57.14 | 80.00 | 66.67 | 5 |
| pizza | 100.00 | 40.00 | 57.14 | 5 |
| ramen | 33.33 | 20.00 | 25.00 | 5 |
| sushi | 18.18 | 40.00 | 25.00 | 5 |
| tacos | 0.00 | 0.00 | 0.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
