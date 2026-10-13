# food101 — MMTM feature-level control — seed 0

Held-out test n=50; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.00 |
| macro precision | 25.73 |
| macro recall | 24.00 |
| macro f1 | 21.96 |
| weighted f1 | 21.96 |
| micro f1 | 24.00 |
| balanced accuracy | 24.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 11.11 | 20.00 | 14.29 | 5 |
| chocolate_cake | 28.57 | 80.00 | 42.11 | 5 |
| french_fries | 50.00 | 20.00 | 28.57 | 5 |
| fried_rice | 33.33 | 20.00 | 25.00 | 5 |
| hamburger | 14.29 | 20.00 | 16.67 | 5 |
| ice_cream | 50.00 | 40.00 | 44.44 | 5 |
| pizza | 50.00 | 20.00 | 28.57 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 0.00 | 0.00 | 0.00 | 5 |
| tacos | 20.00 | 20.00 | 20.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
