"""Merge the India-focused datasets 1–3 into a single long-format CSV.

The USEPA `provisional_habs` dataset (previously dataset_4) has been removed
from the master merge because it is US freshwater data and does not align
with this project's Kerala + Karnataka focus. It is retained separately in
`reference_data/dataset_4_usepa_reference_analog.csv` for future
transfer-learning experiments.

The three datasets included here differ in time granularity (event / annual /
seasonal / monthly) and native shape (long vs wide), so we STACK (concat)
them with a harmonized column schema plus a 'dataset' tag so the user can
filter/pivot downstream.

Output: dataset_master.csv
Schema:
  dataset, source, country, region, sub_location,
  date, year, month, season_or_month, depth_or_layer,
  variable, value_min, value_max, value_mean,
  units, n_observations, notes
"""

import pathlib as _pl
# code/pipelines/xx_...py -> parents[2] is the repo root
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
import pandas as pd, pathlib, sys

ROOT = pathlib.Path(str(_DATA_DIR))
OUT  = ROOT / "dataset_master.csv"

COLS = ["dataset","source","country","region","sub_location",
        "date","year","month","season_or_month","depth_or_layer",
        "variable","value_min","value_max","value_mean",
        "units","n_observations","notes"]

def blanks(n): return [""] * n

# ------------------------------ dataset_1 -----------------------------------
d1 = pd.read_csv(ROOT / "dataset_1.csv")
d1_out = pd.DataFrame({
    "dataset": "dataset_1",
    "source":  d1["source_report"],
    "country": "India",
    "region":  d1["region"],
    "sub_location": d1["sub_location"].fillna(""),
    "date": "",
    "year": d1["data_year"],
    "month": "",
    "season_or_month": d1["season_or_month"].fillna(""),
    "depth_or_layer": d1["depth_or_layer"].fillna(""),
    "variable": d1["variable"],
    "value_min": d1["value_min"],
    "value_max": d1["value_max"],
    "value_mean": d1["value_mean"],
    "units": d1["units"],
    "n_observations": "",
    "notes": d1["dominant_taxa_or_notes"].fillna(""),
})

# ------------------------------ dataset_2 -----------------------------------
d2 = pd.read_csv(ROOT / "dataset_2.csv")
d2_out = pd.DataFrame({
    "dataset": "dataset_2",
    "source":  "MODIS-Aqua L3m 4km monthly",
    "country": "India",
    "region":  d2["region"],
    "sub_location": ("box " + d2["box_min_lat"].astype(str) + "-" + d2["box_max_lat"].astype(str)
                     + "N, " + d2["box_min_lon"].astype(str) + "-" + d2["box_max_lon"].astype(str) + "E"),
    "date": "",
    "year": d2["year"],
    "month": d2["month"],
    "season_or_month": "",
    "depth_or_layer": "surface",
    "variable": d2["variable"].map({"sst":"sea_surface_temperature","chlor_a":"chlorophyll_a"}).fillna(d2["variable"]),
    "value_min":  d2["value_min"],
    "value_max":  d2["value_max"],
    "value_mean": d2["value_mean"],
    "units": d2["variable"].map({"sst":"degC","chlor_a":"mg/m3"}).fillna(""),
    "n_observations": d2["n_valid_pixels"],
    "notes": "granule=" + d2["source_granule"].astype(str),
})

# ------------------------------ dataset_3 -----------------------------------
# wide format; melt into 3 canonical rows per (year, month, region):
#   rainfall_monthly_total_mm (mean/min/max)
#   rainfall_max_daily_mm (single scalar; put in value_mean)
#   rainy_days (scalar; put in value_mean)
d3 = pd.read_csv(ROOT / "dataset_3.csv")
common = dict(
    dataset="dataset_3", source="IMD 0.25deg gridded daily rainfall",
    country="India",
    depth_or_layer="", season_or_month="",
)
box_desc = ("box " + d3["box_min_lat"].astype(str) + "-" + d3["box_max_lat"].astype(str)
            + "N, " + d3["box_min_lon"].astype(str) + "-" + d3["box_max_lon"].astype(str) + "E")

d3_total = pd.DataFrame({**common,
    "region": d3["region"], "sub_location": box_desc,
    "date": "", "year": d3["year"], "month": d3["month"],
    "variable": "rainfall_monthly_total",
    "value_min":  d3["rainfall_mm_total_min"],
    "value_max":  d3["rainfall_mm_total_max"],
    "value_mean": d3["rainfall_mm_total_mean"],
    "units": "mm",
    "n_observations": d3["n_grid_cells"],
    "notes": "min/max/mean are across per-cell monthly totals within region",
})
d3_maxday = pd.DataFrame({**common,
    "region": d3["region"], "sub_location": box_desc,
    "date": "", "year": d3["year"], "month": d3["month"],
    "variable": "rainfall_max_daily",
    "value_min":  "", "value_max": "",
    "value_mean": d3["rainfall_mm_max_daily"],
    "units": "mm/day",
    "n_observations": d3["n_grid_cells"],
    "notes": "single value: max daily rainfall anywhere in region that month",
})
d3_rainy = pd.DataFrame({**common,
    "region": d3["region"], "sub_location": box_desc,
    "date": "", "year": d3["year"], "month": d3["month"],
    "variable": "rainy_days",
    "value_min":  "", "value_max": "",
    "value_mean": d3["rainy_days_mean"],
    "units": "days",
    "n_observations": d3["n_grid_cells"],
    "notes": "mean per-cell count of days with >=1 mm",
})
d3_out = pd.concat([d3_total, d3_maxday, d3_rainy], ignore_index=True)

# ------------------------------ stack ---------------------------------------
# NOTE: dataset_4 (USEPA) intentionally excluded — see module docstring.
merged = pd.concat([d1_out, d2_out, d3_out], ignore_index=True)[COLS]
merged = merged.sort_values(["dataset","country","region","year","month","variable"], na_position="last")
merged.to_csv(OUT, index=False)

print(f"wrote {OUT} with {len(merged):,} rows")
print()
print("=== rows per dataset ===")
print(merged["dataset"].value_counts())
print()
print("=== country x dataset ===")
print(pd.crosstab(merged["country"], merged["dataset"]))
print()
print("=== unique variables ===", merged["variable"].nunique())
