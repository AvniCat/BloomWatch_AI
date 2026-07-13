"""Feature engineering used by both training and live inference.

This is the single source of truth for how the 69 features are constructed.
Keeping it here (imported by both scripts/train_and_save.py and pipeline/build_features.py)
guarantees identical treatment of training vs. live data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import NUMERIC_BASE, LAG_WEEKS, ROLL_WEEKS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Returns a new DataFrame."""
    g = df.copy().sort_values(["region", "date_start"]).reset_index(drop=True)

    for col in NUMERIC_BASE:
        for lag in LAG_WEEKS:
            g[f"{col}_lag{lag}"] = g.groupby("region")[col].shift(lag)
        for win in ROLL_WEEKS:
            g[f"{col}_roll{win}"] = (
                g.groupby("region")[col]
                 .shift(1)
                 .rolling(win, min_periods=1)
                 .mean()
                 .reset_index(0, drop=True)
            )

    # Climatological anomalies (per region × doy)
    clim = g.groupby(["region", "doy_start"])[["chlor_a_mean", "sst_mean"]].transform("mean")
    g["chlor_a_anom"] = g["chlor_a_mean"] - clim["chlor_a_mean"]
    g["sst_anom"] = g["sst_mean"] - clim["sst_mean"]

    # SST slope over last 4 weeks
    g["sst_slope4"] = g.groupby("region")["sst_mean"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
    )

    # Cumulative rainfall (4 weeks)
    g["rain_cum4"] = (
        g.groupby("region")["rainfall_mm_total_mean"]
         .shift(1).rolling(4, min_periods=1).sum().reset_index(0, drop=True)
    )

    # Cyclic day-of-year
    g["doy_sin"] = np.sin(2 * np.pi * g["doy_start"] / 366)
    g["doy_cos"] = np.cos(2 * np.pi * g["doy_start"] / 366)

    # Region one-hot
    g["is_kerala"] = (g["region"] == "Kerala").astype(int)
    g["is_karnataka"] = (g["region"] == "Karnataka").astype(int)

    return g


def feature_columns(df_engineered: pd.DataFrame) -> list[str]:
    """The 69 features, in a stable order."""
    exclude = {
        "date_start", "date_end", "region", "year", "doy_start",
        "bloom", "bloom_or_documented", "bloom_next_week", "hab_event_documented",
    }
    return [c for c in df_engineered.columns
            if c not in exclude and df_engineered[c].dtype != "O"]


def add_forecast_target(df: pd.DataFrame, label_col: str = "bloom_or_documented") -> pd.DataFrame:
    """Add bloom_next_week = shift(-1) of label within region."""
    g = df.copy().sort_values(["region", "date_start"]).reset_index(drop=True)
    g["bloom_next_week"] = (
        g.groupby("region")[label_col].shift(-1).fillna(0).astype(int)
    )
    return g
