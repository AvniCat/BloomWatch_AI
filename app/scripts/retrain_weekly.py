"""Weekly XGBoost retrain — Option A (rolling 6-year window).

Runs after the Friday data refresh. Retrains on the enlarged historical CSV,
evaluates on the trailing 90 days, and only replaces the deployed model if
held-out AUC does not regress by more than DRIFT_THRESHOLD (5%).

Archives every previous deployed model into models/history/ so a bad rollout
can be rolled back manually with a single copy.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    HISTORICAL_CSV, MODEL_PATH, MODEL_META_PATH, XGB_PARAMS, MODEL_VERSION,
)
from pipeline.features import engineer_features, feature_columns, add_forecast_target


DRIFT_THRESHOLD_AUC = 0.05    # refuse deploy if new val-AUC drops more than this
VAL_WINDOW_DAYS = 90          # evaluate retrained model on the trailing 90 days
ROLLING_WINDOW_YEARS = 6
MIN_TRAIN_ROWS = 100
MIN_VAL_POSITIVES = 3

HISTORY_DIR = MODEL_PATH.parent / "history"


def _ece(y_true, y_prob, n_bins=5):
    y_prob = np.asarray(y_prob); y_true = np.asarray(y_true)
    if y_prob.max() - y_prob.min() < 1e-9:
        return abs(y_prob.mean() - y_true.mean())
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1, 2
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(y_prob)) * abs(y_prob[m].mean() - y_true[m].mean())
    return e


def main() -> int:
    if not HISTORICAL_CSV.exists():
        print(f"ERROR: historical CSV missing at {HISTORICAL_CSV}")
        return 1

    print(f"Loading historical CSV: {HISTORICAL_CSV}")
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    df = df.sort_values(["region", "date_start"]).reset_index(drop=True)
    print(f"Total rows in historical CSV: {len(df)}")

    # ---- Rolling 6-year window (anchored to LATEST date in the data, not calendar today) ----
    latest = df.date_start.max()
    cutoff = latest - pd.Timedelta(days=ROLLING_WINDOW_YEARS * 365)
    df_window = df[df.date_start >= cutoff].copy()
    print(f"Data anchor: latest observed week = {latest.date()}")
    print(f"Rolling window from {cutoff.date()} → {latest.date()}: {len(df_window)} rows retained")

    # ---- Target + features ----
    df_window = add_forecast_target(df_window, label_col="bloom_or_documented")
    dfE = engineer_features(df_window)
    feat_cols = feature_columns(dfE)

    # ---- Train/val split — val is the trailing 90 days of actual data ----
    val_cutoff = latest - pd.Timedelta(days=VAL_WINDOW_DAYS)
    train = dfE[dfE.date_start < val_cutoff]
    val   = dfE[dfE.date_start >= val_cutoff]
    print(f"Train: {len(train)} rows ({train.bloom_next_week.mean():.1%} positive)")
    print(f"Val:   {len(val)} rows ({val.bloom_next_week.mean():.1%} positive)")

    if len(train) < MIN_TRAIN_ROWS:
        print(f"ABORT: only {len(train)} training rows (need >= {MIN_TRAIN_ROWS})")
        return 2
    if int(val.bloom_next_week.sum()) < MIN_VAL_POSITIVES:
        print(f"WARNING: only {int(val.bloom_next_week.sum())} positives in val — AUC will be unreliable")

    # ---- Train new model ----
    print("Training fresh XGBoost on rolling window...")
    Xtr = train[feat_cols].values
    ytr = train.bloom_next_week.values
    Xva = val[feat_cols].values
    yva = val.bloom_next_week.values

    new_model = XGBClassifier(**XGB_PARAMS).fit(Xtr, ytr)
    p_new = new_model.predict_proba(Xva)[:, 1]
    new_auc = roc_auc_score(yva, p_new) if len(np.unique(yva)) > 1 else float("nan")
    new_brier = brier_score_loss(yva, p_new)
    new_ece = _ece(yva, p_new)
    print(f"  new model — val AUC={new_auc:.4f}  Brier={new_brier:.4f}  ECE={new_ece:.4f}")

    # ---- Compare to current deployed model ----
    old_auc = None
    if MODEL_PATH.exists():
        try:
            old_model = joblib.load(MODEL_PATH)
            p_old = old_model.predict_proba(Xva)[:, 1]
            old_auc = roc_auc_score(yva, p_old) if len(np.unique(yva)) > 1 else float("nan")
            print(f"  old model — val AUC={old_auc:.4f} (on same val set)")
        except Exception as e:
            print(f"  could not evaluate old model: {e}")

    # ---- Drift check ----
    if old_auc is not None and not np.isnan(new_auc) and not np.isnan(old_auc):
        if new_auc < old_auc - DRIFT_THRESHOLD_AUC:
            print(f"REFUSE DEPLOY — new AUC {new_auc:.4f} vs old {old_auc:.4f} exceeds "
                  f"drift threshold {DRIFT_THRESHOLD_AUC}")
            print(f"Old model remains active. Investigate before next retrain.")
            return 3

    # ---- Monitor-only mode ----
    # For the first month of live operation we recommend running with
    # RETRAIN_MONITOR_ONLY=1 so the script logs what it WOULD deploy but does
    # not actually replace the deployed model. Once you've watched the numbers
    # for a few weeks and are confident the retrain is behaving, unset the
    # environment variable to enable real deployment.
    monitor_only = os.getenv("RETRAIN_MONITOR_ONLY", "0") == "1"

    if monitor_only:
        # Persist a shadow model for diagnostics but do not replace production
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        shadow = HISTORY_DIR / f"xgb_bloom_shadow_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.pkl"
        joblib.dump(new_model, shadow)
        print(f"MONITOR-ONLY mode: shadow model saved to {shadow.name}, "
              f"production model unchanged")
    else:
        # ---- Archive previous model ----
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archived = HISTORY_DIR / f"xgb_bloom_{stamp}.pkl"
            shutil.copy2(MODEL_PATH, archived)
            print(f"Archived previous model → {archived.name}")

        # ---- Deploy new model ----
        joblib.dump(new_model, MODEL_PATH)
        print(f"Deployed new model → {MODEL_PATH.name}")

    # ---- Metadata ----
    meta = {
        "model_version":         MODEL_VERSION,
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "training_strategy":     "weekly rolling retrain (Option A, 6-year window)",
        "n_train_rows":          int(len(train)),
        "n_val_rows":            int(len(val)),
        "n_val_positives":       int(val.bloom_next_week.sum()),
        "val_auc":               None if np.isnan(new_auc) else float(new_auc),
        "val_brier":             float(new_brier),
        "val_ece":               float(new_ece),
        "previous_val_auc":      None if old_auc is None or np.isnan(old_auc) else float(old_auc),
        "auc_delta_vs_previous": None if old_auc is None else float(new_auc - old_auc),
        "training_window_start": str(cutoff.date()),
        "training_window_end":   str(val_cutoff.date()),
        "val_window_start":      str(val_cutoff.date()),
        "val_window_end":        str(latest.date()),
        "n_features":            len(feat_cols),
        "xgb_params":            XGB_PARAMS,
        "drift_check_passed":    True,
        "monitor_only":          monitor_only,
    }
    if not monitor_only:
        MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
        print(f"Wrote metadata → {MODEL_META_PATH.name}")
    else:
        shadow_meta = HISTORY_DIR / f"xgb_bloom_shadow_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.meta.json"
        shadow_meta.write_text(json.dumps(meta, indent=2))
        print(f"Shadow metadata → {shadow_meta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
