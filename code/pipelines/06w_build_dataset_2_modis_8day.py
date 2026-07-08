"""Download MODIS-Aqua L3m 8-DAY SST + Chl-a composites for 2020-2024 by
constructing direct oceandata URLs, cropping to Kerala + Karnataka boxes,
and appending to dataset_2_8day.csv.

Bypasses NASA CMR search (which drops connections during rapid queries) —
the L3m 8-day granule name is fully deterministic:
  AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.SST.sst.4km.nc
  AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.CHL.chlor_a.4km.nc

Auth: pulls Earthdata login from ~/.netrc via requests.
"""
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

import xarray as xr, numpy as np, pandas as pd
import pathlib, sys, os, traceback, gc, warnings, requests, netrc
from datetime import date, timedelta

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT  = pathlib.Path(str(_DATA_DIR))
CACHE = ROOT / "_modis_8d_cache"; CACHE.mkdir(exist_ok=True)
OUT   = ROOT / "dataset_2_8day.csv"
LOG   = ROOT / "_modis_8d_pipeline.log"

REGIONS = {
    "Kerala":    (8.0,  12.5, 74.0, 77.0),
    "Karnataka": (12.0, 15.0, 74.0, 76.0),
}

VARS = [
    # (canonical, url_key, xr_var, units)
    ("sst",     "SST.sst",         "sst",     "degC"),
    ("chlor_a", "CHL.chlor_a",     "chlor_a", "mg/m3"),
]

BASE = "https://oceandata.sci.gsfc.nasa.gov/ob/getfile"

def log(msg):
    line = f"[{date.today().isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def eight_day_windows(y0, y1):
    for y in range(y0, y1 + 1):
        for doy in range(1, 366, 8):
            d0 = date(y, 1, 1) + timedelta(days=doy - 1)
            if doy == 361:
                d1 = date(y, 12, 31)
            else:
                d1 = d0 + timedelta(days=7)
            yield y, doy, d0, d1

def already_done(df, y, doy, canon, region):
    if df is None or df.empty: return False
    return bool(((df.year == y) & (df.doy_start == doy)
                 & (df.variable == canon) & (df.region == region)).any())

def crop_stats(da, box):
    minlat, maxlat, minlon, maxlon = box
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

def get_auth():
    n = netrc.netrc(os.path.expanduser("~/.netrc"))
    a = n.authenticators("urs.earthdata.nasa.gov")
    if not a:
        raise RuntimeError("no earthdata credentials in ~/.netrc")
    return (a[0], a[2])

def download_file(url, dest, auth):
    """Download following URS OAuth redirects. Retries on network errors.
    Fresh session per attempt so a broken connection pool can't poison retries."""
    import time
    for attempt in range(4):
        try:
            with requests.Session() as s:
                s.auth = auth
                r = s.get(url, timeout=60, allow_redirects=True, stream=True)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        f.write(chunk)
                return True
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            if attempt < 3:
                time.sleep(2 ** attempt)  # 1, 2, 4 sec backoff
                continue
            log(f"[DL FAIL] {url.split('/')[-1]}: {err}")
            try: dest.unlink()
            except: pass
            return False

def main():
    auth = get_auth()
    log(f"authed as {auth[0]}")

    if OUT.exists():
        df = pd.read_csv(OUT)
        log(f"resuming: {OUT.name} has {len(df)} rows")
    else:
        df = pd.DataFrame(columns=[
            "year","doy_start","date_start","date_end","variable","region",
            "box_min_lat","box_max_lat","box_min_lon","box_max_lon",
            "value_mean","value_min","value_max","value_std",
            "n_valid_pixels","source_granule"])
        df.to_csv(OUT, index=False)
        log(f"created empty {OUT.name}")

    START_YEAR, END_YEAR = 2020, 2024
    windows = [(y, doy, d0, d1)
               for (y, doy, d0, d1) in eight_day_windows(START_YEAR, END_YEAR)
               if d0 >= date(2002, 7, 4)]
    log(f"target: {len(windows)} windows × 2 vars × 2 regions = {len(windows)*4} rows")

    processed = 0
    for y, doy, d0, d1 in windows:
        for canon, url_key, xr_var, units in VARS:
            if all(already_done(df, y, doy, canon, r) for r in REGIONS):
                continue
            gname = f"AQUA_MODIS.{d0:%Y%m%d}_{d1:%Y%m%d}.L3m.8D.{url_key}.4km.nc"
            url = f"{BASE}/{gname}"
            dest = CACHE / gname
            if not dest.exists():
                if not download_file(url, dest, auth):
                    continue
            try:
                with xr.open_dataset(dest) as ds:
                    if xr_var not in ds:
                        log(f"[NO VAR] {y} doy{doy:03d} {canon}: {list(ds.data_vars)}")
                        continue
                    da = ds[xr_var]
                    rows = []
                    for region, box in REGIONS.items():
                        if already_done(df, y, doy, canon, region): continue
                        s = crop_stats(da, box)
                        if s is None:
                            log(f"[NO PX] {y} doy{doy:03d} {canon} {region}")
                            continue
                        rows.append({
                            "year": y, "doy_start": doy,
                            "date_start": d0.isoformat(), "date_end": d1.isoformat(),
                            "variable": canon, "region": region,
                            "box_min_lat": box[0], "box_max_lat": box[1],
                            "box_min_lon": box[2], "box_max_lon": box[3],
                            **s,
                            "source_granule": gname,
                        })
                    if rows:
                        new = pd.DataFrame(rows)
                        new.to_csv(OUT, mode="a", index=False, header=False)
                        df = pd.concat([df, new], ignore_index=True)
            except Exception:
                log(f"[DECODE FAIL] {y} doy{doy:03d} {canon}: {traceback.format_exc().splitlines()[-1]}")
            finally:
                try: dest.unlink()
                except: pass
                gc.collect()
        processed += 1
        if processed % 23 == 0:  # every ~half year
            log(f"progress: {processed}/{len(windows)} windows  rows={len(df)}")

    log(f"done. rows={len(df)}")

if __name__ == "__main__":
    main()
