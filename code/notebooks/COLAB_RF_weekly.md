# Random Forest — WEEKLY forecast

Same 7-cell structure as the LR weekly notebook. Same `revised_master_dts_weekly.csv`.

## Cell 1 — upload
```python
from google.colab import files
uploaded = files.upload()   # select revised_master_dts_weekly.csv
```

## Cell 2 — imports + config
```python
import pathlib, pickle, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

DATA_PATH = pathlib.Path("/content/revised_master_dts_weekly.csv")
OUT_DIR   = pathlib.Path("/content/hab_outputs_rf_weekly"); OUT_DIR.mkdir(exist_ok=True)

TARGET_SOURCE   = "bloom_or_documented"
TRAIN_END_YEAR  = 2023
N_TREES         = 500
N_CV_SPLITS     = 4
CV_TEST_SIZE    = 46
```

## Cell 3 — feature engineering (identical to LR weekly Cell 3)
```python
df = pd.read_csv(DATA_PATH)
print(f"  raw rows: {len(df)}, columns: {len(df.columns)}")
df = df.sort_values(["region", "year", "doy_start"]).reset_index(drop=True)
df = df.dropna(subset=["chlor_a_mean"]).copy()

lag_cols = ["sst_mean","sst_min","sst_max","sst_std",
            "rainfall_mm_total_mean","rainfall_mm_max_daily","rainy_days_mean"]
for c in lag_cols:
    for lag in (1, 2):
        df[f"{c}_lag{lag}"] = df.groupby("region")[c].shift(lag)

roll_cols = ["sst_mean","sst_std","rainfall_mm_total_mean","rainy_days_mean"]
for c in roll_cols:
    df[f"{c}_roll4w"] = df.groupby("region")[c].shift(1).rolling(4).mean().reset_index(0, drop=True)
    df[f"{c}_roll8w"] = df.groupby("region")[c].shift(1).rolling(8).mean().reset_index(0, drop=True)

anom_cols = ["sst_mean","chlor_a_mean","rainfall_mm_total_mean"]
for c in anom_cols:
    clim = df.groupby(["region", "doy_start"])[c].transform("mean")
    df[f"{c}_anomaly"] = df[c] - clim

df["rainfall_cum_4w"] = df.groupby("region")["rainfall_mm_total_mean"].shift(1).rolling(4).sum().reset_index(0, drop=True)
df["rainfall_cum_8w"] = df.groupby("region")["rainfall_mm_total_mean"].shift(1).rolling(8).sum().reset_index(0, drop=True)
df["sst_slope_4w"] = (df.groupby("region")["sst_mean"].shift(1) - df.groupby("region")["sst_mean"].shift(5)) / 4

df["window_sin"] = np.sin(2 * np.pi * df["doy_start"] / 365)
df["window_cos"] = np.cos(2 * np.pi * df["doy_start"] / 365)
df["region_Kerala"]    = (df["region"] == "Kerala").astype(int)
df["region_Karnataka"] = (df["region"] == "Karnataka").astype(int)

FEATURES = (
    lag_cols
    + [f"{c}_lag{l}" for c in lag_cols for l in (1, 2)]
    + [c for c in df.columns if "_roll" in c or "_anomaly" in c or "_cum_" in c or "_slope_" in c]
    + ["window_sin","window_cos","region_Kerala","region_Karnataka"]
    + ["hab_event_documented","hab_events_last_52w"]
)
FEATURES = list(dict.fromkeys(FEATURES))
print(f"  total features: {len(FEATURES)}")
```

## Cell 4 — next-WEEK target + split
```python
df["bloom_next_week"] = df.groupby("region")[TARGET_SOURCE].shift(-1)
TARGET = "bloom_next_week"

df_model = df.dropna(subset=FEATURES + [TARGET]).copy().sort_values(
    ["year","doy_start","region"]).reset_index(drop=True)

train = df_model[df_model["year"] <= TRAIN_END_YEAR]
test  = df_model[df_model["year"] >  TRAIN_END_YEAR]
X_train, y_train = train[FEATURES], train[TARGET].astype(int)
X_test,  y_test  = test[FEATURES],  test[TARGET].astype(int)
print(f"  train: {len(X_train)} rows (pos rate {y_train.mean():.3f})")
print(f"  test:  {len(X_test)} rows (pos rate {y_test.mean():.3f})")
```

## Cell 5 — expanding-window CV
```python
X_cv = X_train.reset_index(drop=True); y_cv = y_train.reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, test_size=CV_TEST_SIZE)

fold_metrics = []
for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_cv), 1):
    X_tr, y_tr = X_cv.iloc[tr_idx], y_cv.iloc[tr_idx]
    X_te, y_te = X_cv.iloc[te_idx], y_cv.iloc[te_idx]

    fm = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                class_weight="balanced", bootstrap=True,
                                random_state=42, n_jobs=-1)
    fm.fit(X_tr, y_tr)
    p = fm.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    row = {"fold": fold, "train_n": len(tr_idx), "test_n": len(te_idx),
           "accuracy": accuracy_score(y_te, pred),
           "roc_auc": roc_auc_score(y_te, p) if y_te.nunique() > 1 else float("nan")}
    fold_metrics.append(row)
    print(f"fold {fold}: train={row['train_n']}, test={row['test_n']}, "
          f"acc={row['accuracy']:.3f}, AUC={row['roc_auc']:.3f}")

cv_df = pd.DataFrame(fold_metrics)
print(f"\n== CV summary ({N_CV_SPLITS} folds) ==")
print(f"  accuracy: {cv_df.accuracy.mean():.3f} ± {cv_df.accuracy.std():.3f}")
print(f"  ROC-AUC:  {cv_df.roc_auc.mean():.3f} ± {cv_df.roc_auc.std():.3f}")
```

## Cell 6 — final model + hold-out
```python
model = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                               class_weight="balanced", oob_score=True,
                               bootstrap=True, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"trees: {N_TREES}   OOB acc: {model.oob_score_:.4f}")

proba = model.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
metrics = {"accuracy": accuracy_score(y_test, pred),
           "precision": precision_score(y_test, pred, zero_division=0),
           "recall":    recall_score(y_test, pred, zero_division=0),
           "f1":        f1_score(y_test, pred, zero_division=0),
           "roc_auc":   roc_auc_score(y_test, proba)}
print("=== FINAL HELD-OUT METRICS (7-day forecast, 2024) ===")
for k, v in metrics.items(): print(f"  {k:>9s}: {v:.4f}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
```

## Cell 7 — save + download
```python
with open(OUT_DIR / "RandomForest_weekly.pkl", "wb") as f:
    pickle.dump(model, f)
test[["year","doy_start","date_start","region","chlor_a_mean",TARGET]].assign(
    proba=proba, pred=pred).to_csv(OUT_DIR / "RF_weekly_predictions.csv", index=False)

imp = pd.DataFrame({"feature":FEATURES, "importance":model.feature_importances_}
                  ).sort_values("importance", ascending=False)
imp.to_csv(OUT_DIR / "RF_weekly_feature_importance.csv", index=False)
cv_df.to_csv(OUT_DIR / "RF_weekly_cv_folds.csv", index=False)

print("=== TOP 10 FEATURES ===")
print(imp.head(10).to_string(index=False))

import shutil
from google.colab import files as _files
shutil.make_archive("rf_weekly_outputs", "zip", OUT_DIR)
_files.download("rf_weekly_outputs.zip")
```
