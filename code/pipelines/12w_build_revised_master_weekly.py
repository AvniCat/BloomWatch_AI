"""Build revised_master_dts_weekly.csv — weekly-resolution training-ready CSV.

Inputs (all built by upstream pipelines):
  - data/dataset_2_8day.csv    (MODIS 8-day SST + Chl-a, long format)
  - data/dataset_3_8day.csv    (IMD weekly rainfall, wide format)
  - data/hab_events_india.csv  (documented HAB events)

Output: data/revised_master_dts_weekly.csv
  one row per (year, doy_start, region), containing:
  - MODIS SST + Chl-a stats
  - IMD rainfall stats
  - HAB event flags (matched to the 8-day window containing the event date)
  - bloom label = 1 if chlor_a_mean > 2 mg/m^3
  - bloom_or_documented = 1 if bloom OR event documented that week
"""
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

import pandas as pd
import numpy as np
from datetime import date, timedelta

BLOOM_THRESHOLD = 2.0

d2 = pd.read_csv(_DATA_DIR / "dataset_2_8day.csv")
d3 = pd.read_csv(_DATA_DIR / "dataset_3_8day.csv")
print(f"MODIS 8-day: {len(d2)} rows")
print(f"IMD  weekly: {len(d3)} rows")

# -----------------------------------------------------------------------------
# Pivot MODIS long -> wide: one row per (year, doy_start, region) with sst_* and chlor_a_* cols
# -----------------------------------------------------------------------------
d2_stats = d2[["year","doy_start","date_start","date_end","region","variable",
               "value_mean","value_min","value_max","value_std","n_valid_pixels"]]
piv = d2_stats.pivot_table(
    index=["year","doy_start","date_start","date_end","region"],
    columns="variable",
    values=["value_mean","value_min","value_max","value_std","n_valid_pixels"],
    aggfunc="first",
)
rename = {"value_mean":"mean","value_min":"min","value_max":"max",
          "value_std":"std","n_valid_pixels":"n_pixels"}
piv.columns = [f"{v}_{rename[stat]}" for stat, v in piv.columns]
piv = piv.reset_index()

# -----------------------------------------------------------------------------
# Slim IMD wide (already has one row per key)
# -----------------------------------------------------------------------------
d3_slim = d3[["year","doy_start","date_start","date_end","region",
              "rainfall_mm_total_mean","rainfall_mm_total_min",
              "rainfall_mm_total_max","rainfall_mm_max_daily",
              "rainy_days_mean","n_grid_cells"]].rename(
    columns={"n_grid_cells":"rainfall_n_cells"})

# -----------------------------------------------------------------------------
# Outer-join on (year, doy_start, region)
# -----------------------------------------------------------------------------
wide = piv.merge(d3_slim, on=["year","doy_start","region"],
                 how="outer", suffixes=("", "_imd"))
# collapse date_start/date_end (prefer MODIS side, fall back to IMD)
wide["date_start"] = wide["date_start"].fillna(wide["date_start_imd"])
wide["date_end"]   = wide["date_end"].fillna(wide["date_end_imd"])
wide = wide.drop(columns=[c for c in wide.columns if c.endswith("_imd")])

# -----------------------------------------------------------------------------
# HAB event flag — a window "contains" an event if event date in [d0, d1]
# -----------------------------------------------------------------------------
def parse_event_date(row):
    ed = str(row["event_date"])
    if len(ed) >= 10 and ed[4] == "-" and ed[7] == "-":
        return date.fromisoformat(ed)
    if len(ed) >= 7 and ed[4] == "-":
        y, m = int(ed[:4]), int(ed[5:7])
        return date(y, m, 15)
    return None

try:
    ev = pd.read_csv(_DATA_DIR / "hab_events_india.csv")
    ev["date"] = ev.apply(parse_event_date, axis=1)
    ev = ev.dropna(subset=["date"])
    def window_key(row):
        d = row["date"]; y = d.year
        doy = ((d.timetuple().tm_yday - 1) // 8) * 8 + 1
        return pd.Series({"year": y, "doy_start": doy})
    ev[["year","doy_start"]] = ev.apply(window_key, axis=1)
    ev["hab_event_documented"] = 1
    ev_counts = ev.groupby(["year","doy_start","region"]).size().rename("hab_event_documented").reset_index()
    wide = wide.merge(ev_counts, on=["year","doy_start","region"], how="left")
    wide["hab_event_documented"] = (wide["hab_event_documented"].fillna(0) > 0).astype(int)
except FileNotFoundError:
    wide["hab_event_documented"] = 0

# 52-week rolling count (per region)
wide = wide.sort_values(["region","year","doy_start"]).reset_index(drop=True)
wide["hab_events_last_52w"] = (
    wide.groupby("region")["hab_event_documented"]
        .rolling(window=52, min_periods=1).sum().reset_index(0, drop=True).astype(int)
)

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
wide["bloom"] = (wide["chlor_a_mean"] > BLOOM_THRESHOLD).astype("Int64")
wide.loc[wide["chlor_a_mean"].isna(), "bloom"] = pd.NA

wide["bloom_or_documented"] = (
    ((wide["bloom"] == 1) | (wide["hab_event_documented"] == 1)).astype("Int64")
)
wide.loc[wide["bloom"].isna() & (wide["hab_event_documented"] == 0),
         "bloom_or_documented"] = pd.NA

# round floats
for c in wide.columns:
    if wide[c].dtype.kind == "f":
        wide[c] = wide[c].round(4)

wide = wide.sort_values(["year","doy_start","region"]).reset_index(drop=True)

OUT = _DATA_DIR / "revised_master_dts_weekly.csv"
wide.to_csv(OUT, index=False)
print(f"\nwrote {OUT} with {len(wide)} rows, {len(wide.columns)} cols")
print()
print("=== columns ===")
for c in wide.columns:
    print(f"  {c}")
print()
print("=== coverage (non-null %) ===")
print((wide.notna().sum() / len(wide) * 100).round(1).astype(str) + "%")
print()
print("=== label distribution ===")
print("bloom:", wide["bloom"].value_counts(dropna=False).to_dict())
print("bloom_or_documented:", wide["bloom_or_documented"].value_counts(dropna=False).to_dict())
print("hab_event_documented:", wide["hab_event_documented"].value_counts(dropna=False).to_dict())
