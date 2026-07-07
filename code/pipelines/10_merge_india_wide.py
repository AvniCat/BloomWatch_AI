"""Wide-format join of datasets 2 (MODIS) and 3 (IMD) on (year, month, region).

Both are monthly, per-region (Kerala / Karnataka), so this is a clean 1:1 join.
Output: dataset_merged_india_monthly_wide.csv — the modeling-ready environmental
predictor table for the Kerala/Karnataka coasts.

CMFRI (dataset 1) is intentionally NOT joined here: it has a different time
granularity (mostly annual/seasonal/event), different sub-regions (Kochi,
Vizag, Mandapam, etc.), and no direct match to the MODIS/IMD 0.5-degree boxes.
Use dataset_merged.csv when you need CMFRI context.
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
import pandas as pd, pathlib

ROOT = pathlib.Path(str(_DATA_DIR))
OUT  = ROOT / "dataset_merged_india_monthly_wide.csv"

d2 = pd.read_csv(ROOT / "dataset_2.csv")
d3 = pd.read_csv(ROOT / "dataset_3.csv")

# --- pivot dataset_2 so 'sst' and 'chlor_a' become side-by-side columns ---
keep = ["year","month","region","variable",
        "value_mean","value_min","value_max","value_std","n_valid_pixels"]
d2 = d2[keep]
piv = d2.pivot_table(
    index=["year","month","region"],
    columns="variable",
    values=["value_mean","value_min","value_max","value_std","n_valid_pixels"],
    aggfunc="first",
)
# flatten MultiIndex columns: (value_mean, sst) -> sst_mean, (n_valid_pixels, chlor_a) -> chlor_a_n_pixels
rename = {
    "value_mean": "mean", "value_min": "min", "value_max": "max",
    "value_std": "std", "n_valid_pixels": "n_pixels",
}
piv.columns = [f"{v}_{rename[stat]}" for stat, v in piv.columns]
piv = piv.reset_index()

# --- dataset_3 is already 1 row per (year, month, region) ---
d3_slim = d3[["year","month","region",
              "rainfall_mm_total_mean","rainfall_mm_total_min","rainfall_mm_total_max",
              "rainfall_mm_max_daily","rainy_days_mean","n_grid_cells"]].rename(columns={
    "n_grid_cells": "rainfall_n_cells",
})

# --- outer join so we don't drop months where MODIS is missing ---
wide = piv.merge(d3_slim, on=["year","month","region"], how="outer")
wide = wide.sort_values(["year","month","region"]).reset_index(drop=True)

# round for readability
for c in wide.columns:
    if wide[c].dtype.kind == "f":
        wide[c] = wide[c].round(4)

wide.to_csv(OUT, index=False)
print(f"wrote {OUT} with {len(wide)} rows, {len(wide.columns)} cols")
print("\ncolumns:", list(wide.columns))
print("\ncoverage (non-null pct per column):")
print((wide.notna().sum() / len(wide) * 100).round(1).astype(str) + "%")
print("\nsample (2017 Kerala):")
print(wide[(wide.year==2017) & (wide.region=="Kerala")].to_string())
