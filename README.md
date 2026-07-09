# BloomWatch AI — HAB Early Warning for Kerala & Karnataka Coasts

An AI early-warning system for **Harmful Algal Blooms (HABs)** along India's
southwest coast, delivering weekly bloom-risk forecasts to shellfish farmer
cooperatives via a multilingual WhatsApp chatbot.

**Live at:** [github.com/AvniCat/BloomWatch_AI](https://github.com/AvniCat/BloomWatch_AI)

## What this project does

- Predicts HAB risk **7 days ahead** for each 0.5° coastal cell along the
  Kerala and Karnataka coasts.
- Trained on 22 years of public data: NASA MODIS-Aqua satellite (SST +
  chlorophyll-a), IMD 0.25° gridded rainfall, and CMFRI in-situ observations.
- Delivered to farmers as a multilingual WhatsApp chatbot — Kadal Mitra —
  in Malayalam, Kannada, Hindi, and English.

## Modelling summary

**Target** — binary bloom flag from MODIS chlorophyll-a *and* CMFRI-documented
events: `bloom_or_documented = 1` if either signal fires.

**Forecast horizon** — the label is shifted forward one week within each
region so the model uses week `t`'s conditions to predict week `t+1`'s bloom.

**Features** — 40 engineered per row, per region: current-week values
+ 1- and 2-week lags + 4- and 8-week rolling means + same-week climatological
anomalies + cumulative rainfall + SST slope + cyclic day-of-year + region
one-hot + HAB event history.

**Split** — expanding-window CV across 4 folds, plus a strict 2024
hold-out year for final evaluation.

| model | notebook | best-round / trees | test ROC-AUC (weekly) |
|---|---|---|---|
| Logistic Regression | `code/notebooks/BloomWatchAI_LG.ipynb` | L-BFGS converged | 0.85 |
| Random Forest | `code/notebooks/BloomWatchAI_RF.ipynb` | 500 trees | 0.82 |
| XGBoost | `code/notebooks/BloomWatchAI_XGB.ipynb` | early-stopped | 0.83 |

Top drivers converge across all three models:
1. Rainy-days at t-1/t-2 (accumulated nutrient loading)
2. SST variability (upwelling proxy)
3. Rolling 4-week rainfall trend
4. Documented HAB event history

## Repo layout

```
BloomWatch_AI/
├── code/
│   ├── src/                      standalone Python training scripts
│   │   ├── feature_prep.py                shared feature engineering
│   │   ├── train_logistic_regression.py
│   │   ├── train_random_forest.py
│   │   └── train_xgboost.py
│   ├── notebooks/                Colab-ready notebooks
│   │   ├── BloomWatchAI_LG.ipynb          Logistic Regression (weekly)
│   │   ├── BloomWatchAI_RF.ipynb          Random Forest      (weekly)
│   │   ├── BloomWatchAI_XGB.ipynb         XGBoost            (weekly)
│   │   ├── train_all_models.ipynb         legacy monthly overview
│   │   └── COLAB_*_weekly.md              cell-source markdown
│   └── pipelines/                data-collection scripts
│       ├── 01…05  CMFRI PDF extraction
│       ├── 06     MODIS-Aqua monthly download + crop
│       ├── 06w    MODIS-Aqua 8-day download + crop
│       ├── 07     IMD rainfall monthly
│       ├── 07w    IMD rainfall weekly
│       ├── 09     merge master (long, India datasets 1-3)
│       ├── 10     merge India monthly wide
│       ├── 11     extract HAB events from CMFRI
│       ├── 12     build revised master (monthly)
│       └── 12w    build revised master (weekly)
├── data/                         datasets + README
├── docs/                         research writeup, pitch deck, diagrams
├── results/                      model outputs (pickles, predictions,
│                                 feature importance, metrics, figures)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Quickstart — Google Colab (fastest)

Open any of the three notebooks in Colab directly from GitHub:

- LR: [colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_LG.ipynb](https://colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_LG.ipynb)
- RF: [colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_RF.ipynb](https://colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_RF.ipynb)
- XGB: [colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_XGB.ipynb](https://colab.research.google.com/github/AvniCat/BloomWatch_AI/blob/main/code/notebooks/BloomWatchAI_XGB.ipynb)

Upload `data/revised_master_dts_weekly.csv` when the first cell prompts.

## Quickstart — local

```bash
git clone https://github.com/AvniCat/BloomWatch_AI.git
cd BloomWatch_AI
pip install -r requirements.txt

# each script is standalone
python code/src/train_logistic_regression.py
python code/src/train_random_forest.py
python code/src/train_xgboost.py
```

## Reusing a trained model

```python
import pickle
with open("results/models/RandomForest.pkl", "rb") as f:
    model = pickle.load(f)
proba = model.predict_proba(X_new)[:, 1]
```

## Re-generating the datasets

The committed CSVs are ready to train on. To rebuild from source:

```bash
# monthly (legacy) pipeline
python code/pipelines/06_build_dataset_2_modis.py   # needs Earthdata login
python code/pipelines/07_build_dataset_3_imd.py
python code/pipelines/09_merge_master.py
python code/pipelines/10_merge_india_wide.py

# weekly (7-day forecast) pipeline
python code/pipelines/06w_build_dataset_2_modis_8day.py
python code/pipelines/07w_build_dataset_3_imd_weekly.py
python code/pipelines/12w_build_revised_master_weekly.py
```

**Note on dataset_4 (USEPA):** US freshwater data, retained as a reference-only
dataset (not merged into the India-focused master). See `data/README.md`.

## Attribution & licence

MIT (see `LICENSE`). If you use this in research or downstream products,
please cite this repo:

> Singh, A. (2026). *BloomWatch AI: An AI early-warning system for harmful
> algal blooms on the Kerala and Karnataka coasts.* github.com/AvniCat/BloomWatch_AI

Data credits:
- **NASA MODIS-Aqua** L3m ocean colour products — public domain, please cite the mission.
- **IMD 0.25° gridded rainfall** — Pai et al. 2014; cite the paper.
- **CMFRI Annual Reports** — © ICAR-CMFRI; extracted values used under fair-use for research.
- **USEPA `provisional_habs`** — public domain (CC0 per source repo).

Fellowship affiliation: **AI Fellowship for Global Young Innovators 2026** —
The Innovation Story.
