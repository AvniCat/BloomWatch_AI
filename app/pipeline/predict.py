"""Load the pickled XGBoost model and produce per-region forecasts."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_PATH, MODEL_META_PATH, CURRENT_FORECAST_PATH, DATA_DIR


def qualitative_risk(p: float) -> str:
    """Risk bands recalibrated against reliability-diagram empirical rates.

    The reliability curve flattens around 0.35 on 2024 hold-out — predictions of
    0.5–0.7 correspond to empirical bloom rates of ~0.30–0.40, not 0.50–0.70.
    We surface the raw probability but present a hedged band label to farmers.
    """
    if p < 0.15:  return "Low"
    if p < 0.35:  return "Elevated caution"
    if p < 0.55:  return "Elevated caution (model less reliable in this range)"
    return "High risk — consult local extension officer"


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"ERROR: model missing at {MODEL_PATH}. Run scripts/train_and_save.py first.")
        return 1

    features_path = DATA_DIR / "live/current_week_features.json"
    if not features_path.exists():
        print(f"ERROR: feature vector missing at {features_path}. Run pipeline/build_features.py.")
        return 2

    model = joblib.load(MODEL_PATH)
    meta = json.loads(MODEL_META_PATH.read_text()) if MODEL_META_PATH.exists() else {}
    feats = json.loads(features_path.read_text())

    feature_order = feats["feature_order"]
    if meta.get("feature_order") and meta["feature_order"] != feature_order:
        print("WARNING: feature order in metadata does not match live features. "
              "Model may misbehave. Retrain if features have changed.")

    forecasts = {}
    for region, vec in feats["regions"].items():
        arr = np.array([[np.nan if v is None else v for v in vec]], dtype=float)
        p = float(model.predict_proba(arr)[0, 1])
        forecasts[region] = {
            "P_bloom_next_week": round(p, 4),
            "risk_band": qualitative_risk(p),
            "confidence_note": (
                "This forecast estimates the probability of an ELEVATED CHLOROPHYLL "
                "week (satellite chl-a > 2 mg/m³) — a leading indicator of algal "
                "bloom conditions, not confirmed toxic bloom initiation. Some "
                "harmful species (notably Trichodesmium) are poorly detected by "
                "standard satellite chlorophyll retrievals; treat the number as "
                "a caution flag, not a diagnostic call, and consult your local "
                "CMFRI or Fisheries Department extension officer for confirmation. "
                "Model AUC 0.83 on 2024 hold-out; probabilities above 0.35 flatten "
                "against empirical rates and should not be read as precise percentages."
            ),
        }

    payload = {
        "model_version": meta.get("model_version", "unknown"),
        "trained_at":    meta.get("trained_at"),
        "computed_at":   datetime.now(timezone.utc).isoformat(),
        "window_start":  feats["window_start"],
        "window_end":    feats["window_end"],
        "forecast_target": "P(bloom in next 8-day window)",
        "regions": forecasts,
    }
    CURRENT_FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_FORECAST_PATH.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {CURRENT_FORECAST_PATH}")
    for region, r in forecasts.items():
        print(f"  {region:10s}  P = {r['P_bloom_next_week']:.3f}  ({r['risk_band']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
