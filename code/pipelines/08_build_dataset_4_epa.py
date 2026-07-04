"""Build dataset_4.csv from USEPA/provisional_habs (US freshwater HAB monitoring).

Source: https://github.com/USEPA/provisional_habs
Coverage: 4 US ponds (Mashapaug RI, Archer MA, Shubael & Hamblin MA, RWP Zoo),
sub-hourly automated monitoring, 2022-2026, ~887k observations.

Approach:
  - Filter to HAB-relevant variables (bloom pigments, env drivers, nutrients).
  - Aggregate sub-hourly obs to daily per (date, waterbody, site, depth, variable, units).
  - Long format matching dataset_1 schema.
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import pandas as pd, pathlib, glob, warnings

warnings.filterwarnings("ignore")

SRC = sorted(glob.glob("/tmp/provisional_habs/hab_provisional_data_*.csv"))
OUT = pathlib.Path(str(_DATA_DIR / "dataset_4.csv"))

# HAB-relevant variables to keep (case-insensitive contains-match)
KEEP = {
    "chlorophyll": "chlorophyll",           # RFU + ug/L, bloom biomass proxy
    "extracted chlorophyll": "chlorophyll_extracted",
    "phycocyanin": "phycocyanin",           # cyanobacteria pigment
    "extracted phycocyanin": "phycocyanin_extracted",
    "water temperature": "water_temperature",
    "dissolved oxygen conc.": "dissolved_oxygen",
    "dissolved oxygen sat.": "dissolved_oxygen_saturation",
    "turbidity": "turbidity",
    "ph": "pH",
    "specific conductivity": "specific_conductivity",
    "no3-n": "nitrate_N",
    "air temperature": "air_temperature",
    "wind speed": "wind_speed",
    "wind direction": "wind_direction",
    "barometric pressure": "barometric_pressure",
}

print("loading CSVs ...", flush=True)
dfs = []
for f in SRC:
    d = pd.read_csv(f, low_memory=False)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print(f"  raw rows: {len(df):,}")

# normalize types
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["date"]  = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "value", "variable"])

# canonicalize variable name
df["variable_lc"] = df["variable"].astype(str).str.strip().str.lower()
df = df[df["variable_lc"].isin(KEEP.keys())]
df["variable"] = df["variable_lc"].map(KEEP)
df = df.drop(columns=["variable_lc"])
print(f"  HAB-relevant rows: {len(df):,}")

# aggregate to daily per (waterbody, site, depth, variable, units)
group_cols = ["date", "waterbody", "site", "depth", "variable", "units"]
agg = df.groupby(group_cols, dropna=False)["value"].agg(
    value_mean="mean", value_min="min", value_max="max", n_observations="count"
).reset_index()
agg["source"]     = "USEPA/provisional_habs"
agg["data_year"]  = agg["date"].dt.year
agg["date"]       = agg["date"].dt.strftime("%Y-%m-%d")

# round for readability
for c in ("value_mean","value_min","value_max"):
    agg[c] = agg[c].round(3)

# column ordering matching dataset_1 style
cols = ["source","data_year","date","waterbody","site","depth",
        "variable","value_min","value_max","value_mean","units","n_observations"]
agg = agg[cols].sort_values(["date","waterbody","variable"])
agg.to_csv(OUT, index=False)
print(f"wrote {OUT} with {len(agg):,} rows")
print("\nsummary by variable:")
print(agg["variable"].value_counts())
print("\nsummary by waterbody:")
print(agg["waterbody"].value_counts())
