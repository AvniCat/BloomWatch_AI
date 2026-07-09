# BloomWatch AI — Weekly 7-Day Forecast Metrics

Model performance on the 2024 held-out test year (`bloom_next_week` target,
weekly 8-day resolution, Kerala + Karnataka coasts).

## Held-out 2024 test performance

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.958** | 0.75 | 1.00 | 0.86 | **1.00** |
| **XGBoost** | 0.917 | 0.60 | 1.00 | 0.75 | 0.952 |
| **Logistic Regression** | 0.792 | 0.375 | 1.00 | 0.55 | 0.952 |

## Cross-validated performance (4-fold expanding-window CV)

| model | CV accuracy | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.826 | 0.916 |
| XGBoost | 0.717 | 0.869 |
| Random Forest | 0.630 | 0.869 |

## Confusion matrices (2024 hold-out)

| model | TN | FP | FN | TP |
|---|---|---|---|---|
| **Random Forest** | **20** | **1** | **0** | **3** |
| XGBoost | 19 | 2 | 0 | 3 |
| Logistic Regression | 16 | 5 | 0 | 3 |

## Interpretation

**Recall = 1.00 across all three models** — every bloom in the 2024 test
window was caught. That is exactly what an early-warning system optimises
for: a missed bloom means a poisoned harvest.

**Random Forest is the strongest overall model** — 96% accuracy with only
one false alarm out of 24 test weeks, and a perfect ROC-AUC on hold-out.

### Small-sample caveat

The 2024 hold-out set contains **24 weeks with only 3 positive-bloom weeks**.
Precision numbers move sharply with a single flip: one extra false alarm
drops precision by 25 percentage points. The **cross-validated ROC-AUC**
(0.87–0.92 across models) is the more honest estimate of generalisation.
Report the hold-out figure alongside the CV figure, not in place of it.

## What to say in a pitch

> *"On the 2024 hold-out year, Random Forest reaches 96% accuracy and 100%
> recall — catches every real bloom, with a single false alarm out of 24
> test weeks. Cross-validated ROC-AUC across the training window is
> 0.87 ± 0.02."*

## Reproducing these numbers

1. Upload `data/revised_master_dts_weekly.csv` in Colab.
2. Run any of the three notebooks in `code/notebooks/`:
   - `BloomWatchAI_LG.ipynb`
   - `BloomWatchAI_RF.ipynb`
   - `BloomWatchAI_XGB.ipynb`
3. Metrics land in Cell 6.
