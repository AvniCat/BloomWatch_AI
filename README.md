# HAB Prediction — Kerala & Karnataka Coasts

Machine-learning pipeline for identifying Harmful Algal Blooms (HABs) along
the southwest coast of India, using satellite, in-situ, and rainfall
observations from 2002–2024.

## Repo layout

```
hab-prediction/
├── code/
│   ├── src/                     training scripts (Python)
│   │   ├── feature_prep.py                  # shared feature engineering
│   │   ├── train_logistic_regression.py
│   │   ├── train_random_forest.py
│   │   └── train_xgboost.py
│   ├── notebooks/               Jupyter / Colab notebooks
│   │   └── train_all_models.ipynb
│   └── pipelines/               data-collection scripts (regenerate the CSVs)
│       ├── 01…05  CMFRI PDF extraction
│       ├── 06     MODIS-Aqua download + crop
│       ├── 07     IMD rainfall download + crop
│       ├── 08     USEPA HAB monitoring (reference-only, not merged)
│       ├── 09     merge master (long)  — India datasets 1-3 only
│       └── 10     merge India monthly wide
├── data/
│   ├── README.md                        sources, schemas, how to regenerate
│   └── dataset_merged_india_monthly_wide.csv   (only the small modelling table)
├── docs/                        research paper, pitch deck, diagrams
├── results/
│   ├── models/                          three pickled trained models
│   ├── predictions/                     test-set predictions per model
│   ├── feature_importance/              feature ranking per model
│   ├── model_metrics.csv                accuracy / precision / recall / F1 / ROC-AUC
│   └── figures/                         (add plots / screenshots here)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Modelling summary

**Target** — binary bloom flag from MODIS chlorophyll-a: `bloom = 1 if chlor_a_mean > 2 mg/m³ else 0`.

**Features** — SST (mean/min/max/std) + rainfall (monthly total, max daily, rainy days) + 1-and-2-month lags + cyclic month + region one-hot. 25 features total.

**Temporal split** — train Jan 2002 – Dec 2018 (370 rows), test Jan 2019 – Dec 2024 (141 rows).

| model | script | iterations | test accuracy | ROC-AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | `code/src/train_logistic_regression.py` | 28 (L-BFGS converged) | 0.766 | 0.836 | 0.571 |
| Random Forest | `code/src/train_random_forest.py` | 500 trees | **0.865** | **0.875** | **0.655** |
| XGBoost | `code/src/train_xgboost.py` | 175 (early-stopped from 2000) | 0.759 | 0.839 | 0.553 |

Top drivers converge across all three models:
1. Monthly seasonality (`month_sin`) — SW monsoon window
2. SST variability (`sst_std`, `sst_std_lag1`) — upwelling proxy
3. Rainy days at t-1 / t-2 — accumulated freshwater / nutrient input

Full metrics: `results/model_metrics.csv`.

## Quickstart — local

```bash
git clone <this-repo>
cd hab-prediction
pip install -r requirements.txt

# each script is standalone
python code/src/train_logistic_regression.py
python code/src/train_random_forest.py
python code/src/train_xgboost.py
```

## Quickstart — Google Colab

Open `code/notebooks/train_all_models.ipynb` in Colab (via
`https://colab.research.google.com/github/<YOUR_USER>/hab-prediction/blob/main/code/notebooks/train_all_models.ipynb`)
and hit **Runtime → Run all**. First cell handles `git clone` + `pip install`
automatically.

## Reusing a trained model

```python
import pickle
with open("results/models/RandomForest.pkl", "rb") as f:
    model = pickle.load(f)
proba = model.predict_proba(X_new)[:, 1]   # 25 features, same order as feature_prep.py
```

## Re-generating the datasets

Only `data/dataset_merged_india_monthly_wide.csv` is committed. To rebuild the
full stack (raw CSVs, master, wide):

```bash
python code/pipelines/06_build_dataset_2_modis.py   # needs Earthdata login
python code/pipelines/07_build_dataset_3_imd.py
python code/pipelines/09_merge_master.py            # merges India datasets 1-3
python code/pipelines/10_merge_india_wide.py
```

**Note on dataset_4 (USEPA):** The USEPA `provisional_habs` dataset is US
freshwater data and is *not* merged into the India-focused master dataset.
It is retained as a reference-only dataset (built by `08_build_dataset_4_epa.py`)
for future transfer-learning experiments. See `data/README.md` for details.

See `data/README.md` for source URLs, expected schemas, and licence notes.

## License

MIT (see `LICENSE`).
