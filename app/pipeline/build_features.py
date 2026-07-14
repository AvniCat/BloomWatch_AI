"""Build the current-week feature vector for both regions.

Reads the newly-downloaded MODIS + IMD summaries, appends a fresh row per
region to the historical CSV in memory, and runs the shared feature-engineering
pipeline. The last row per region is the live feature vector we want to predict on.

Outputs:
  data/live/current_week_row.csv        one row per region, raw values
  data/live/current_week_features.json  feature vector per region
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    HISTORICAL_CSV, LIVE_MODIS_DIR, LIVE_IMD_DIR, REGIONS, DATA_DIR,
)
from pipeline.features import engineer_features, feature_columns


def _read_modis_stats(product: str, start: date, end: date) -> dict:
    """Return per-region stats for one ocean-color product (SST or CHL).

    Data source is VIIRS-SNPP NRT (MODIS-Aqua retired Feb 2025). Filename read
    from latest.json so this stays robust to future source changes.
    """
    var_name = {"SST": "sst", "CHL": "chlor_a"}[product]
    latest = json.loads((LIVE_MODIS_DIR / "latest.json").read_text())
    fname = latest[f"{product.lower()}_filename"]
    path = LIVE_MODIS_DIR / fname
    if not path.exists(): raise FileNotFoundError(path)

    ds = xr.open_dataset(path, decode_times=False)
    da = ds[var_name]
    # dims are typically (lat, lon)
    lat_name = next(n for n in da.dims if "lat" in n.lower())
    lon_name = next(n for n in da.dims if "lon" in n.lower())

    per_region = {}
    for region_name, bbox in REGIONS.items():
        sub = da.sel({
            lat_name: slice(bbox["lat_max"], bbox["lat_min"]),  # MODIS lat descends
            lon_name: slice(bbox["lon_min"], bbox["lon_max"]),
        })
        vals = sub.values.astype(float)
        vals = vals[np.isfinite(vals) & (vals != da.attrs.get("_FillValue", -32767.0))]
        if vals.size == 0:
            per_region[region_name] = {"n_pixels": 0, "mean": np.nan, "max": np.nan,
                                       "min": np.nan, "std": np.nan}
        else:
            per_region[region_name] = {
                "n_pixels": int(vals.size),
                "mean": float(vals.mean()),
                "max":  float(vals.max()),
                "min":  float(vals.min()),
                "std":  float(vals.std()),
            }
    return per_region


def _read_simulated_modis_stats(start: date, end: date) -> tuple[dict, dict]:
    """Fallback for sim mode: pull the current-week MODIS stats from the historical CSV row."""
    hist = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    chl = {}; sst = {}
    for region in REGIONS:
        row = hist.loc[(hist.date_start == pd.Timestamp(start)) & (hist.region == region)]
        if row.empty:
            raise FileNotFoundError(f"No CSV row for {region} @ {start}")
        r = row.iloc[0]
        chl[region] = {
            "n_pixels": int(r["chlor_a_n_pixels"]) if pd.notna(r["chlor_a_n_pixels"]) else 0,
            "mean": float(r["chlor_a_mean"]) if pd.notna(r["chlor_a_mean"]) else np.nan,
            "max":  float(r["chlor_a_max"])  if pd.notna(r["chlor_a_max"])  else np.nan,
            "min":  float(r["chlor_a_min"])  if pd.notna(r["chlor_a_min"])  else np.nan,
            "std":  float(r["chlor_a_std"])  if pd.notna(r["chlor_a_std"])  else np.nan,
        }
        sst[region] = {
            "n_pixels": int(r["sst_n_pixels"]) if pd.notna(r["sst_n_pixels"]) else 0,
            "mean": float(r["sst_mean"]) if pd.notna(r["sst_mean"]) else np.nan,
            "max":  float(r["sst_max"])  if pd.notna(r["sst_max"])  else np.nan,
            "min":  float(r["sst_min"])  if pd.notna(r["sst_min"])  else np.nan,
            "std":  float(r["sst_std"])  if pd.notna(r["sst_std"])  else np.nan,
        }
    return chl, sst


def build_current_week_rows() -> pd.DataFrame:
    """Return a DataFrame with one row per region for the current week."""
    modis_latest = json.loads((LIVE_MODIS_DIR / "latest.json").read_text())
    imd = json.loads((LIVE_IMD_DIR / "current_window.json").read_text())
    start = date.fromisoformat(modis_latest["yyyyddd_start"])
    end   = date.fromisoformat(modis_latest["yyyyddd_end"])

    if modis_latest.get("simulated"):
        chl_stats, sst_stats = _read_simulated_modis_stats(start, end)
    else:
        chl_stats = _read_modis_stats("CHL", start, end)
        sst_stats = _read_modis_stats("SST", start, end)

    rows = []
    doy = (start - date(start.year, 1, 1)).days + 1
    for region in REGIONS:
        r = {
            "year": start.year,
            "doy_start": doy,
            "date_start": pd.Timestamp(start),
            "date_end":   pd.Timestamp(end),
            "region": region,

            "chlor_a_n_pixels": chl_stats[region]["n_pixels"],
            "sst_n_pixels":     sst_stats[region]["n_pixels"],
            "chlor_a_max":  chl_stats[region]["max"],
            "sst_max":      sst_stats[region]["max"],
            "chlor_a_mean": chl_stats[region]["mean"],
            "sst_mean":     sst_stats[region]["mean"],
            "chlor_a_min":  chl_stats[region]["min"],
            "sst_min":      sst_stats[region]["min"],
            "chlor_a_std":  chl_stats[region]["std"],
            "sst_std":      sst_stats[region]["std"],

            **imd["regions"][region],   # rainfall_mm_total_mean, _min, _max, _max_daily, rainy_days_mean, rainfall_n_cells

            "hab_event_documented": 0,
            "hab_events_last_52w": 0,   # will be recomputed after merge
            "bloom": 0,
            "bloom_or_documented": 0,
        }
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    if not HISTORICAL_CSV.exists():
        print(f"ERROR: historical CSV missing at {HISTORICAL_CSV}")
        return 1

    hist = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start", "date_end"])
    hist = hist.sort_values(["region", "date_start"]).reset_index(drop=True)

    current = build_current_week_rows()
    print(f"Current week: {current['date_start'].iloc[0].date()} → "
          f"{current['date_end'].iloc[0].date()}")

    # In sim mode, "current" is already in `hist`. Drop those rows before concat
    # to avoid duplicate weeks (which would corrupt lag/rolling features).
    modis_latest = json.loads((LIVE_MODIS_DIR / "latest.json").read_text())
    if modis_latest.get("simulated"):
        cur_start = current["date_start"].iloc[0]
        hist = hist[hist["date_start"] != cur_start].reset_index(drop=True)

    # ---- Live-data append: grow the historical CSV each real Friday refresh ----
    # This enables the weekly rolling retrain to actually see new data. Only append
    # for real live runs (not simulated) and only when the current week isn't
    # already in the historical CSV.
    if not modis_latest.get("simulated"):
        cur_start = current["date_start"].iloc[0]
        already_present = ((hist["date_start"] == cur_start).any()
                           and hist[hist["date_start"] == cur_start]["region"].nunique() == len(current))
        if not already_present:
            # Compute chl-a-derived bloom label from observed chl_a_mean
            current["bloom"] = (current["chlor_a_mean"].fillna(0) > 2.0).astype(int)
            current["bloom_or_documented"] = (
                current["bloom"] | (current["hab_event_documented"] == 1).astype(int)
            ).astype(int)
            # Concat and dedupe by (region, date_start) — keep the newer row
            grown = pd.concat([hist, current], ignore_index=True)
            grown = grown.sort_values(["region", "date_start"]).drop_duplicates(
                subset=["region", "date_start"], keep="last"
            ).reset_index(drop=True)
            grown.to_csv(HISTORICAL_CSV, index=False)
            print(f"Appended current week to historical CSV → now {len(grown)} rows")
            hist = grown

    # Recompute rolling HAB event history
    combined = pd.concat([hist, current], ignore_index=True)
    combined = combined.sort_values(["region", "date_start"]).reset_index(drop=True)
    combined["hab_events_last_52w"] = (
        combined.groupby("region")["hab_event_documented"]
                .transform(lambda s: s.shift(1).rolling(52, min_periods=1).sum())
                .fillna(0).astype(int)
    )

    # Engineer full feature set on combined data
    engineered = engineer_features(combined)
    feat_cols = feature_columns(engineered)

    # Extract the current-week rows (last per region)
    latest = (engineered.sort_values(["region", "date_start"])
                        .groupby("region").tail(1)
                        .reset_index(drop=True))

    # Save raw + feature vector
    DATA_DIR.joinpath("live").mkdir(parents=True, exist_ok=True)
    latest[["date_start", "date_end", "region"] + feat_cols].to_csv(
        DATA_DIR / "live/current_week_row.csv", index=False)

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "window_start": latest["date_start"].iloc[0].date().isoformat(),
        "window_end":   latest["date_end"].iloc[0].date().isoformat(),
        "feature_order": feat_cols,
        "regions": {
            row["region"]: [
                (float(v) if pd.notna(v) else None) for v in row[feat_cols].values
            ]
            for _, row in latest.iterrows()
        },
    }
    (DATA_DIR / "live/current_week_features.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote current_week_features.json ({len(feat_cols)} features × "
          f"{len(payload['regions'])} regions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
