"""Download IMD gridded daily rainfall (0.25 deg) for 2010-2024, crop to
Kerala and Karnataka bounding boxes, aggregate to monthly per region,
save to dataset_3.csv.

IMD gridded rain: 0.25° x 0.25° from 6.5°N to 38.5°N and 66.5°E to 100°E,
one file per year, provided by IMD Pune. Fetched via `imdlib`.

Output schema (mirrors dataset_2 style):
  year, month, region, box_min_lat, box_max_lat, box_min_lon, box_max_lon,
  rainfall_mm_total_mean, rainfall_mm_total_min, rainfall_mm_total_max,
  rainfall_mm_max_daily, rainy_days_mean, n_grid_cells
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import imdlib, xarray as xr, numpy as np, pandas as pd
import pathlib, sys, gc, warnings, calendar

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(str(_DATA_DIR))
CACHE = ROOT / "_imd_cache"
CACHE.mkdir(exist_ok=True)
OUT_CSV = ROOT / "dataset_3.csv"

REGIONS = {
    "Kerala":    (8.0,  12.5, 74.0, 77.0),
    "Karnataka": (12.0, 15.0, 74.0, 76.0),
}

START_YEAR = 2002
END_YEAR   = 2024

def log(m):
    print(m, flush=True)

def main():
    log(f"target: {END_YEAR-START_YEAR+1} years x 12 months x {len(REGIONS)} regions = "
        f"{(END_YEAR-START_YEAR+1)*12*len(REGIONS)} rows")

    # imdlib downloads a full multi-year block at once.
    log("downloading IMD daily rainfall grids ...")
    data = imdlib.get_data("rain", START_YEAR, END_YEAR, fn_format="yearwise", file_dir=str(CACHE))
    # convert to xarray for easy cropping
    ds = data.get_xarray()   # dims: time, lat, lon ; var: rain (mm/day)
    log(f"loaded ds: dims={dict(ds.sizes)} time_range={str(ds.time.values[0])[:10]}..{str(ds.time.values[-1])[:10]}")

    rain = ds["rain"]
    # replace fill (IMD uses -999 or similar) with NaN just in case
    rain = rain.where(rain >= 0)

    rows = []
    for region, (minlat, maxlat, minlon, maxlon) in REGIONS.items():
        # IMD lat likely ascending 6.5 -> 38.5; use slice(min, max)
        sub = rain.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
        n_cells = int(sub.sizes["lat"] * sub.sizes["lon"])
        log(f"{region}: {sub.sizes['lat']} x {sub.sizes['lon']} = {n_cells} grid cells")
        # groupby (year, month)
        df = sub.to_dataframe(name="rain_mm").dropna(subset=["rain_mm"]).reset_index()
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        # For each (year, month), compute:
        #   rainfall_mm_total_mean = mean across cells of the cell's monthly total
        #   rainfall_mm_total_min/max = min/max across cells of monthly total
        #   rainfall_mm_max_daily = max daily value seen anywhere in region that month
        #   rainy_days_mean = mean per-cell count of days with >= 1 mm
        # First: per-cell monthly total & rainy-day count
        percell = df.groupby(["year","month","lat","lon"]).agg(
            monthly_total=("rain_mm", "sum"),
            rainy_days=("rain_mm", lambda s: int((s >= 1.0).sum())),
        ).reset_index()
        # Max daily anywhere in region
        max_daily = df.groupby(["year","month"])["rain_mm"].max().rename("max_daily")

        agg = percell.groupby(["year","month"]).agg(
            rainfall_mm_total_mean=("monthly_total","mean"),
            rainfall_mm_total_min =("monthly_total","min"),
            rainfall_mm_total_max =("monthly_total","max"),
            rainy_days_mean       =("rainy_days","mean"),
        ).join(max_daily).reset_index()

        for _, r in agg.iterrows():
            rows.append({
                "year": int(r.year), "month": int(r.month), "region": region,
                "box_min_lat": minlat, "box_max_lat": maxlat,
                "box_min_lon": minlon, "box_max_lon": maxlon,
                "rainfall_mm_total_mean": round(float(r.rainfall_mm_total_mean), 3),
                "rainfall_mm_total_min":  round(float(r.rainfall_mm_total_min), 3),
                "rainfall_mm_total_max":  round(float(r.rainfall_mm_total_max), 3),
                "rainfall_mm_max_daily":  round(float(r.max_daily), 3),
                "rainy_days_mean":        round(float(r.rainy_days_mean), 2),
                "n_grid_cells": n_cells,
            })

    out = pd.DataFrame(rows).sort_values(["year","month","region"])
    out.to_csv(OUT_CSV, index=False)
    log(f"wrote {OUT_CSV} with {len(out)} rows")

if __name__ == "__main__":
    main()
