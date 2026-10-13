# food101 — MMTM feature-level control — seed 2

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 32.00 |
| macro precision | 28.22 |
| macro recall | 32.00 |
| macro f1 | 27.26 |
| weighted f1 | 27.26 |
| micro f1 | 32.00 |
| balanced accuracy | 32.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 26.67 | 80.00 | 40.00 | 5 |
| chocolate_cake | 50.00 | 80.00 | 61.54 | 5 |
| french_fries | 22.22 | 40.00 | 28.57 | 5 |
| fried_rice | 25.00 | 20.00 | 22.22 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 0.00 | 0.00 | 0.00 | 5 |
| pizza | 50.00 | 40.00 | 44.44 | 5 |
| ramen | 25.00 | 20.00 | 22.22 | 5 |
| sushi | 33.33 | 20.00 | 25.00 | 5 |
| tacos | 50.00 | 20.00 | 28.57 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
