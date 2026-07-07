"""Download MODIS-Aqua L3m monthly SST + Chl-a for 2002-07 through 2024-12,
crop to Kerala and Karnataka bounding boxes, aggregate per region, and append
to dataset_2.csv.

Design:
  - Resumable: skips (year, month, variable, region) rows already in the CSV.
  - Process-and-delete: each NetCDF file is downloaded to _modis_cache/,
    cropped, aggregated, then removed to keep disk footprint small.
  - Robust: any download / decode failure is logged and the loop continues.

Output schema (long format, matches dataset_1 style):
  year, month, variable, region, box_min_lat, box_max_lat, box_min_lon, box_max_lon,
  value_mean, value_min, value_max, value_std, n_valid_pixels, source_granule
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
import earthaccess, xarray as xr, numpy as np, pandas as pd
import pathlib, sys, os, traceback, gc, warnings, contextlib
from datetime import date

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(str(_DATA_DIR))
CACHE = ROOT / "_modis_cache"
CACHE.mkdir(exist_ok=True)
OUT_CSV = ROOT / "dataset_2.csv"
LOG = ROOT / "_modis_pipeline.log"

REGIONS = {
    # region_name: (min_lat, max_lat, min_lon, max_lon)
    "Kerala":    (8.0,  12.5, 74.0, 77.0),
    "Karnataka": (12.0, 15.0, 74.0, 76.0),
}

VARS = [
    # (short_name,               granule_name_glob,             xr_var, canonical_var_name, units)
    ("MODISA_L3m_SST", "*.MO.SST.sst.4km.nc",         "sst",     "sst",     "degC"),
    ("MODISA_L3m_CHL", "*.MO.CHL.chlor_a.4km.nc",     "chlor_a", "chlor_a", "mg/m3"),
]

def log(msg):
    line = f"[{date.today().isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def month_iter(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1; y += 1

def last_day(y, m):
    if m == 12: return 31
    from calendar import monthrange
    return monthrange(y, m)[1]

def already_done(df, y, m, canon_var, region):
    if df is None or df.empty: return False
    q = ((df.year == y) & (df.month == m)
         & (df.variable == canon_var) & (df.region == region))
    return bool(q.any())

def crop_stats(da, box):
    minlat, maxlat, minlon, maxlon = box
    # OceanColor L3m lat descends from +90 to -90 → use slice(max, min)
    sub = da.sel(lat=slice(maxlat, minlat), lon=slice(minlon, maxlon))
    arr = np.asarray(sub.values, dtype="float64")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return None
    return dict(
        value_mean=float(valid.mean()),
        value_min=float(valid.min()),
        value_max=float(valid.max()),
        value_std=float(valid.std()),
        n_valid_pixels=int(valid.size),
    )

def main():
    log("logging in")
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        log("AUTH FAILED"); sys.exit(1)

    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        log(f"resuming: {OUT_CSV.name} has {len(df)} rows")
    else:
        df = pd.DataFrame(columns=[
            "year","month","variable","region",
            "box_min_lat","box_max_lat","box_min_lon","box_max_lon",
            "value_mean","value_min","value_max","value_std",
            "n_valid_pixels","source_granule"])
        df.to_csv(OUT_CSV, index=False)
        log(f"created empty {OUT_CSV.name}")

    START = (2002, 7)
    END   = (2024, 12)
    total_months = (END[0]-START[0])*12 + (END[1]-START[1]) + 1
    log(f"target: {total_months} months × 2 vars × 2 regions = {total_months*4} rows")

    processed = 0
    for y, m in month_iter(*START, *END):
        for short_name, glob, xr_var, canon, units in VARS:
            # skip if BOTH regions already done for this (y,m,var)
            if all(already_done(df, y, m, canon, r) for r in REGIONS):
                continue
            t0 = f"{y}-{m:02d}-01"
            t1 = f"{y}-{m:02d}-{last_day(y,m):02d}"
            try:
                results = earthaccess.search_data(
                    short_name=short_name, temporal=(t0, t1),
                    granule_name=glob, count=3)
            except Exception as e:
                log(f"[SEARCH FAIL] {y}-{m:02d} {canon}: {e}")
                continue
            if not results:
                log(f"[NO GRANULE] {y}-{m:02d} {canon}")
                continue
            try:
                files = earthaccess.download(results[:1], str(CACHE))
            except Exception as e:
                log(f"[DL FAIL] {y}-{m:02d} {canon}: {e}")
                continue
            # earthaccess.download can put an Exception into `files` on 404s
            # instead of raising — filter those out.
            valid = [f for f in files if isinstance(f, (str, pathlib.Path))]
            if not valid:
                log(f"[DL FAIL] {y}-{m:02d} {canon}: 404/missing on server")
                continue
            nc = pathlib.Path(valid[0])
            try:
                with xr.open_dataset(nc) as ds:
                    if xr_var not in ds:
                        log(f"[NO VAR] {y}-{m:02d} {canon}: {list(ds.data_vars)}")
                        continue
                    da = ds[xr_var]
                    rows = []
                    for region, box in REGIONS.items():
                        if already_done(df, y, m, canon, region): continue
                        s = crop_stats(da, box)
                        if s is None:
                            log(f"[NO PX] {y}-{m:02d} {canon} {region}")
                            continue
                        rows.append({
                            "year": y, "month": m,
                            "variable": canon, "region": region,
                            "box_min_lat": box[0], "box_max_lat": box[1],
                            "box_min_lon": box[2], "box_max_lon": box[3],
                            **s,
                            "source_granule": nc.name,
                        })
                    if rows:
                        # append and flush
                        new = pd.DataFrame(rows)
                        new.to_csv(OUT_CSV, mode="a", index=False, header=False)
                        df = pd.concat([df, new], ignore_index=True)
            except Exception:
                log(f"[DECODE FAIL] {y}-{m:02d} {canon}: {traceback.format_exc().splitlines()[-1]}")
            finally:
                try: nc.unlink()
                except: pass
                gc.collect()
        processed += 1
        if processed % 12 == 0:
            log(f"progress: {processed}/{total_months} months  rows={len(df)}")

    log(f"done. rows={len(df)}")

if __name__ == "__main__":
    main()
