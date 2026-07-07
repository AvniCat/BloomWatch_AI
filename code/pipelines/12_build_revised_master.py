"""Build revised_master_dataset.csv — training-ready.

Takes the wide monthly modelling table (MODIS + IMD) and joins in the
documented HAB event flags derived by 11_extract_hab_events.py. Produces
a single CSV with all features + the bloom label pre-computed, so any
model can be trained from it without further feature engineering.

Added columns beyond the base wide table:
  - hab_event_documented       : 1 if a CMFRI/lit event was documented
                                  in this (year, month, region), else 0
  - hab_events_last_24mo       : rolling count of documented events
                                  in this region over the past 24 months
  - bloom                      : the label — 1 if chlor_a_mean > 2 mg/m^3
                                  (marine coastal HAB threshold)
  - bloom_or_documented        : 1 if either bloom==1 or hab_event_documented==1
                                  (a more inclusive label — recommended for
                                  supervised training)

Output: data/revised_master_dataset.csv
"""
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

import pandas as pd
import numpy as np

BLOOM_THRESHOLD = 2.0   # mg/m^3

# -- inputs
wide = pd.read_csv(_DATA_DIR / "dataset_merged_india_monthly_wide.csv")
events = pd.read_csv(_DATA_DIR / "hab_events_india.csv")

# -- documented-event flag
key = ["year","month","region"]
event_counts = (events
    .groupby(key)
    .size()
    .rename("hab_event_documented_count")
    .reset_index()
)
merged = wide.merge(event_counts, on=key, how="left")
merged["hab_event_documented_count"] = merged["hab_event_documented_count"].fillna(0).astype(int)
merged["hab_event_documented"] = (merged["hab_event_documented_count"] > 0).astype(int)

# -- rolling 24-month event count (per region)
def _rolling_events(sub):
    sub = sub.sort_values(["year","month"]).copy()
    sub["hab_events_last_24mo"] = (sub["hab_event_documented_count"]
        .rolling(window=24, min_periods=1).sum().astype(int))
    return sub

merged = (merged
    .groupby("region", group_keys=False)
    .apply(_rolling_events)
    .drop(columns=["hab_event_documented_count"]))

# -- label(s)
merged["bloom"] = (merged["chlor_a_mean"] > BLOOM_THRESHOLD).astype("Int64")
merged.loc[merged["chlor_a_mean"].isna(), "bloom"] = pd.NA
merged["bloom_or_documented"] = (
    ((merged["bloom"] == 1) | (merged["hab_event_documented"] == 1)).astype("Int64")
)
merged.loc[merged["bloom"].isna() & (merged["hab_event_documented"] == 0),
           "bloom_or_documented"] = pd.NA

# -- restore ordering
merged = merged.sort_values(["year","month","region"]).reset_index(drop=True)

OUT = _DATA_DIR / "revised_master_dataset.csv"
merged.to_csv(OUT, index=False)

print(f"wrote {OUT}")
print(f"rows: {len(merged)}, columns: {len(merged.columns)}")
print()
print("=== column list ===")
for c in merged.columns:
    print(f"  {c}")
print()
print("=== label distribution ===")
print(f"bloom (chlor_a > {BLOOM_THRESHOLD} mg/m3):")
print(merged["bloom"].value_counts(dropna=False).to_string())
print(f"\nhab_event_documented:")
print(merged["hab_event_documented"].value_counts(dropna=False).to_string())
print(f"\nbloom_or_documented (combined label):")
print(merged["bloom_or_documented"].value_counts(dropna=False).to_string())
print()
print("=== events per region ===")
print(events.groupby("region").size().to_string())
