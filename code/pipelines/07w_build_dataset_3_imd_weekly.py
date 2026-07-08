"""Aggregate IMD 0.25-deg gridded daily rainfall to 8-DAY windows for
2002-2024, cropped to Kerala + Karnataka.

Aligns with MODIS 8-day composites: windows start on Julian days
1, 9, 17, ..., 361 within each calendar year.

Output: data/dataset_3_8day.csv
Schema:
  year, doy_start, date_start, date_end, region,
  box_min_lat, box_max_lat, box_min_lon, box_max_lon,
  rainfall_mm_total_mean, rainfall_mm_total_min, rainfall_mm_total_max,
  rainfall_mm_max_daily, rainy_days_mean, n_grid_cells
"""
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

import imdlib, xarray as xr, numpy as np, pandas as pd
import pathlib, warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")

DATA_DIR = pathlib.Path(str(_DATA_DIR))
CACHE = DATA_DIR / "_imd_cache_8d"; CACHE.mkdir(exist_ok=True)
OUT = DATA_DIR / "dataset_3_8day.csv"

REGIONS = {
    "Kerala":    (8.0,  12.5, 74.0, 77.0),
    "Karnataka": (12.0, 15.0, 74.0, 76.0),
}

START_YEAR, END_YEAR = 2020, 2024   # scoped to 5 years for pitch timeline

def eight_day_windows(y0, y1):
    for y in range(y0, y1 + 1):
        for doy in range(1, 366, 8):
            d0 = date(y, 1, 1) + timedelta(days=doy - 1)
            d1 = d0 + timedelta(days=7)
            # last window truncates at year-end
            if doy == 361:
                d1 = date(y, 12, 31)
            yield y, doy, d0, d1

def main():
    print(f"target: {END_YEAR-START_YEAR+1} years x 46 windows x {len(REGIONS)} regions")

    print("downloading IMD daily rainfall grids...")
    data = imdlib.get_data("rain", START_YEAR, END_YEAR,
                           fn_format="yearwise", file_dir=str(CACHE))
    ds = data.get_xarray()
    print(f"  loaded ds: dims={dict(ds.sizes)} "
          f"time={str(ds.time.values[0])[:10]}..{str(ds.time.values[-1])[:10]}")

    rain = ds["rain"].where(ds["rain"] >= 0)

    rows = []
    for region, (minlat, maxlat, minlon, maxlon) in REGIONS.items():
        sub = rain.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
        n_cells = int(sub.sizes["lat"] * sub.sizes["lon"])
        print(f"  {region}: {sub.sizes['lat']}x{sub.sizes['lon']} = {n_cells} cells")

        df = sub.to_dataframe(name="rain_mm").dropna(subset=["rain_mm"]).reset_index()
        df["date"] = pd.to_datetime(df["time"]).dt.date

        for y, doy, d0, d1 in eight_day_windows(START_YEAR, END_YEAR):
            window = df[(df["date"] >= d0) & (df["date"] <= d1)]
            if window.empty:
                continue
            per_cell = window.groupby(["lat","lon"]).agg(
                window_total=("rain_mm","sum"),
                rainy_days=("rain_mm", lambda s: int((s >= 1.0).sum())),
            ).reset_index()
            max_daily = window["rain_mm"].max()
            rows.append({
                "year": y, "doy_start": doy,
                "date_start": d0.isoformat(), "date_end": d1.isoformat(),
                "region": region,
                "box_min_lat": minlat, "box_max_lat": maxlat,
                "box_min_lon": minlon, "box_max_lon": maxlon,
                "rainfall_mm_total_mean": round(float(per_cell.window_total.mean()), 3),
                "rainfall_mm_total_min":  round(float(per_cell.window_total.min()), 3),
                "rainfall_mm_total_max":  round(float(per_cell.window_total.max()), 3),
                "rainfall_mm_max_daily":  round(float(max_daily), 3),
                "rainy_days_mean":        round(float(per_cell.rainy_days.mean()), 2),
                "n_grid_cells": n_cells,
            })

    out = pd.DataFrame(rows).sort_values(["year","doy_start","region"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} with {len(out)} rows")

if __name__ == "__main__":
    main()
