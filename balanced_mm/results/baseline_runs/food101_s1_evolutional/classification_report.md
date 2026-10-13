# food101 — Evolutional dropout — seed 1

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 42.00 |
| macro precision | 51.38 |
| macro recall | 42.00 |
| macro f1 | 39.54 |
| weighted f1 | 39.54 |
| micro f1 | 42.00 |
| balanced accuracy | 42.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 25.00 | 60.00 | 35.29 | 5 |
| chocolate_cake | 57.14 | 80.00 | 66.67 | 5 |
| french_fries | 40.00 | 40.00 | 40.00 | 5 |
| fried_rice | 50.00 | 80.00 | 61.54 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 25.00 | 40.00 | 30.77 | 5 |
| pizza | 50.00 | 40.00 | 44.44 | 5 |
| ramen | 100.00 | 20.00 | 33.33 | 5 |
| sushi | 66.67 | 40.00 | 50.00 | 5 |
| tacos | 100.00 | 20.00 | 33.33 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
