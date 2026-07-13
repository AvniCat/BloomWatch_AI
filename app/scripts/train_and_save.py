"""Train XGBoost on the historical weekly CSV, save model + metadata sidecar."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# make `import config` work when running from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    HISTORICAL_CSV, MODEL_PATH, MODEL_META_PATH,
    XGB_PARAMS, MODEL_VERSION,
)
from pipeline.features import engineer_features, feature_columns, add_forecast_target


def main() -> None:
    print(f"Loading historical CSV: {HISTORICAL_CSV}")
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    df = df.sort_values(["region", "date_start"]).reset_index(drop=True)

    df = add_forecast_target(df, label_col="bloom_or_documented")
    dfE = engineer_features(df)
    feat_cols = feature_columns(dfE)
    print(f"Rows: {len(dfE)}   Features: {len(feat_cols)}")

    # Train on ALL historical rows — this is the production model.
    # Calibration/eval is documented separately in the calibration notebook.
    X = dfE[feat_cols].values
    y = dfE["bloom_next_week"].values

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(dfE)),
        "n_features": len(feat_cols),
        "feature_order": feat_cols,
        "positive_rate_train": float(np.mean(y)),
        "training_date_range": [
            str(dfE["date_start"].min().date()),
            str(dfE["date_start"].max().date()),
        ],
        "xgb_params": XGB_PARAMS,
        "label_column_used_for_target": "bloom_or_documented (shift -1)",
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata → {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
