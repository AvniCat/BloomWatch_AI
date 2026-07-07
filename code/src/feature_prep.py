"""Shared feature engineering for the HAB bloom classifier scripts.

Loads the revised master training-ready dataset (`revised_master_dataset.csv`
built by `code/pipelines/12_build_revised_master.py`), engineers lag features,
and splits temporally (train 2002-2018, test 2019+).

Falls back to the older `dataset_merged_india_monthly_wide.csv` if the revised
file is not present, so old checkouts still run.

Path resolution is relative to the repo root (three levels up from this file:
code/src/feature_prep.py -> repo/), so the code runs identically on a laptop,
on Colab after cloning, or on any CI.
"""
import pandas as pd
import numpy as np
import pathlib
import pickle

REPO_ROOT   = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR    = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR  = RESULTS_DIR / "models"
PRED_DIR    = RESULTS_DIR / "predictions"
IMP_DIR     = RESULTS_DIR / "feature_importance"

REVISED_TABLE = DATA_DIR / "revised_master_dataset.csv"
LEGACY_TABLE  = DATA_DIR / "dataset_merged_india_monthly_wide.csv"

BLOOM_THRESHOLD = 2.0    # mg/m^3
TRAIN_END_YEAR  = 2018

# Which label column to train on:
#   "bloom"              -> chlor_a > 2 mg/m^3 (derived, MODIS-only)
#   "bloom_or_documented" -> above OR a documented CMFRI/lit HAB event
DEFAULT_LABEL = "bloom_or_documented"


def build_features(path=None, label=DEFAULT_LABEL):
    """Return (features_list, train_df, test_df) with target label column.

    Uses `revised_master_dataset.csv` if available (adds HAB-event features),
    else falls back to the legacy wide table.
    """
    if path is None:
        path = REVISED_TABLE if REVISED_TABLE.exists() else LEGACY_TABLE

    df = pd.read_csv(path)
    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)
    df = df.dropna(subset=["chlor_a_mean"]).copy()

    # Ensure the label column exists (recompute for legacy compatibility)
    if "bloom" not in df.columns:
        df["bloom"] = (df["chlor_a_mean"] > BLOOM_THRESHOLD).astype(int)
    else:
        df["bloom"] = df["bloom"].fillna(0).astype(int)

    if "bloom_or_documented" not in df.columns:
        # Legacy fallback — treat as identical to bloom
        df["bloom_or_documented"] = df["bloom"]
    else:
        df["bloom_or_documented"] = df["bloom_or_documented"].fillna(0).astype(int)

    if label not in df.columns:
        raise ValueError(f"label column {label!r} not found in {path.name}")

    # Environmental lag features
    lag_cols = [
        "sst_mean", "sst_min", "sst_max", "sst_std",
        "rainfall_mm_total_mean", "rainfall_mm_max_daily", "rainy_days_mean",
    ]
    for c in lag_cols:
        for lag in (1, 2):
            df[f"{c}_lag{lag}"] = df.groupby("region")[c].shift(lag)

    # Cyclic month encoding + region one-hot
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["region_Kerala"]    = (df["region"] == "Kerala").astype(int)
    df["region_Karnataka"] = (df["region"] == "Karnataka").astype(int)

    features = (lag_cols
                + [f"{c}_lag{l}" for c in lag_cols for l in (1, 2)]
                + ["month_sin", "month_cos", "region_Kerala", "region_Karnataka"])

    # HAB-event features (only present in the revised table)
    hab_cols = [c for c in ("hab_event_documented", "hab_events_last_24mo")
                if c in df.columns]
    features.extend(hab_cols)

    df_model = df.dropna(subset=features + [label]).copy()
    # Standardise the target column name to 'bloom' for downstream compatibility.
    # Drop the derived-only 'bloom' first if we're using a different label so
    # there's no duplicate column after renaming.
    if label != "bloom" and "bloom" in df_model.columns:
        df_model = df_model.drop(columns=["bloom"])
        df_model = df_model.rename(columns={label: "bloom"})
    train = df_model[df_model["year"] <= TRAIN_END_YEAR]
    test  = df_model[df_model["year"] >  TRAIN_END_YEAR]
    return features, train, test


def evaluate(name, y_true, y_pred, y_proba, train_n, test_n, n_iter=None):
    """Print + return a metrics dict for one model."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score,
                                 confusion_matrix)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = dict(
        model=name,
        n_iterations=n_iter if n_iter is not None else "n/a",
        accuracy =round(accuracy_score(y_true, y_pred), 4),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall   =round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1       =round(f1_score(y_true, y_pred, zero_division=0), 4),
        roc_auc  =round(roc_auc_score(y_true, y_proba), 4)
                       if pd.Series(y_true).nunique() == 2 else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        train_n=int(train_n), test_n=int(test_n),
    )
    print(f"=== {name} ===")
    for k, v in m.items():
        if k != "model":
            print(f"  {k:>13}: {v}")
    return m


def save_predictions(name, test_df, y_proba, y_pred):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = test_df[["year", "month", "region", "chlor_a_mean", "bloom"]].copy()
    out[f"{name}_proba"] = y_proba
    out[f"{name}_pred"]  = y_pred
    out.to_csv(PRED_DIR / f"{name}_predictions.csv", index=False)


def save_importance(name, features, importance, signed=None):
    IMP_DIR.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({"feature": features, "importance": importance})
    if signed is not None:
        imp["signed_weight"] = signed
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(IMP_DIR / f"{name}_feature_importance.csv", index=False)


def save_model(name, model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def append_metric_row(row_dict, path=None):
    """Append one row to results/model_metrics.csv (creating it if absent)."""
    if path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "model_metrics.csv"
    row = pd.DataFrame([row_dict])
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev["model"] != row_dict["model"]]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(path, index=False)
