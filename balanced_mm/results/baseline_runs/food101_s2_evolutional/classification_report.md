# food101 — Evolutional dropout — seed 2

Held-out test n=50; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 30.00 |
| macro precision | 34.47 |
| macro recall | 30.00 |
| macro f1 | 27.81 |
| weighted f1 | 27.81 |
| micro f1 | 30.00 |
| balanced accuracy | 30.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 28.57 | 40.00 | 33.33 | 5 |
| chocolate_cake | 40.00 | 80.00 | 53.33 | 5 |
| french_fries | 50.00 | 20.00 | 28.57 | 5 |
| fried_rice | 25.00 | 20.00 | 22.22 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 100.00 | 20.00 | 33.33 | 5 |
| pizza | 25.00 | 40.00 | 30.77 | 5 |
| ramen | 11.11 | 20.00 | 14.29 | 5 |
| sushi | 25.00 | 20.00 | 22.22 | 5 |
| tacos | 40.00 | 40.00 | 40.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
