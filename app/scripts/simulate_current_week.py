"""Simulate a live refresh using the last week of the historical CSV.

Useful for testing the feature-engineering + inference pipeline end-to-end
without needing Earthdata credentials or waiting for a new MODIS composite.
Writes the same JSON artefacts that pull_modis.py + pull_imd.py would.
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    HISTORICAL_CSV, LIVE_MODIS_DIR, LIVE_IMD_DIR, REGIONS,
)


def main() -> int:
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    last_start = df["date_start"].max()
    last_end = df.loc[df["date_start"] == last_start, "date_end"].iloc[0]
    start_d = last_start.date(); end_d = last_end.date()
    print(f"Simulating live pipeline using window {start_d} → {end_d}")

    # Fake latest.json for pull_modis
    LIVE_MODIS_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_MODIS_DIR / "latest.json").write_text(json.dumps({
        "yyyyddd_start": start_d.isoformat(),
        "yyyyddd_end":   end_d.isoformat(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "simulated": True,
    }, indent=2))

    # Fake IMD current_window.json using the CSV's own rainfall stats
    LIVE_IMD_DIR.mkdir(parents=True, exist_ok=True)
    per_region = {}
    for region in REGIONS:
        row = df.loc[(df.date_start == last_start) & (df.region == region)].iloc[0]
        per_region[region] = {
            "rainfall_mm_total_mean": float(row["rainfall_mm_total_mean"]),
            "rainfall_mm_total_min":  float(row["rainfall_mm_total_min"]),
            "rainfall_mm_total_max":  float(row["rainfall_mm_total_max"]),
            "rainfall_mm_max_daily":  float(row["rainfall_mm_max_daily"]),
            "rainy_days_mean":        float(row["rainy_days_mean"]),
            "rainfall_n_cells":       int(row["rainfall_n_cells"]),
        }
    (LIVE_IMD_DIR / "current_window.json").write_text(json.dumps({
        "window_start": start_d.isoformat(),
        "window_end":   end_d.isoformat(),
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "regions":      per_region,
        "simulated":    True,
    }, indent=2))

    print("Wrote simulated MODIS + IMD summaries.")
    print("Now run:")
    print("  python -m pipeline.build_features   # in sim mode, uses last CSV row")
    print("  python -m pipeline.predict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
