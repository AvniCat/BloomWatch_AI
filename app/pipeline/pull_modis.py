"""Fetch newest VIIRS-SNPP NRT 8-day SST + Chl-a composites (MODIS-Aqua retired 2025).

Uses NASA CMR to discover the newest granule, then downloads from obdaac-tea
(NASA cloud archive) with Earthdata credentials via a session-per-attempt loop.

Outputs (identical filenames as before so downstream stays unchanged):
  data/live/modis/<yyyyddd_start>.SST.nc
  data/live/modis/<yyyyddd_start>.CHL.nc
  data/live/modis/latest.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LIVE_MODIS_DIR

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Data sources (MODIS-Aqua retired Feb 2025):
#   CHL — Suomi-NPP VIIRS NRT (requires MERIS + Sentinel EULAs on Earthdata profile)
#   SST — Suomi-NPP VIIRS science-quality (SNPP publishes 8-day; JPSS1 doesn't)
# Both publish on the same 8-day schedule so a matched window exists.
COLLECTIONS = {
    "CHL": "VIIRSN_L3m_CHL_NRT",   # ~2 week lag
    "SST": "VIIRSN_L3m_SST",       # ~5 week lag
}
GRANULE_RE_CHL = re.compile(
    r"SNPP_VIIRS\.(\d{8})_(\d{8})\.L3m\.8D\.CHL\.chlor_a\.4km\.NRT\.nc$"
)
GRANULE_RE_SST = re.compile(
    r"SNPP_VIIRS\.(\d{8})_(\d{8})\.L3m\.8D\.SST\.sst\.4km\.nc$"
)

MAX_ATTEMPTS = 4
INITIAL_BACKOFF = 8
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 300


def _list_windows(product: str, days_back: int = 90) -> list[tuple[date, date, str]]:
    """Return all 8-day 4km windows for a product, newest first."""
    end = date.today(); start = end - timedelta(days=days_back)
    r = requests.get(CMR, params={
        "short_name": COLLECTIONS[product],
        "temporal": f"{start.isoformat()}T00:00:00Z,{end.isoformat()}T23:59:59Z",
        "page_size": 100,
        "sort_key[]": "-start_date",
    }, timeout=30)
    r.raise_for_status()
    regex = GRANULE_RE_CHL if product == "CHL" else GRANULE_RE_SST
    out = []
    for e in r.json().get("feed", {}).get("entry", []):
        title = e.get("title", "")
        m = regex.search(title)
        if not m: continue
        s_str, e_str = m.group(1), m.group(2)
        s = datetime.strptime(s_str, "%Y%m%d").date()
        e_ = datetime.strptime(e_str, "%Y%m%d").date()
        for link in e.get("links", []):
            href = link.get("href", "")
            if href.endswith(".nc") and ("obdaac" in href or "oceandata" in href):
                out.append((s, e_, href)); break
    return out


def find_matched_window() -> tuple[tuple[date, date, str], tuple[date, date, str]] | None:
    """Find the newest 8-day window where BOTH CHL and SST are available."""
    chl_list = _list_windows("CHL")
    sst_list = _list_windows("SST")
    sst_by_start = {s: (s, e, u) for s, e, u in sst_list}
    for s, e, u_chl in chl_list:
        if s in sst_by_start:
            return (s, e, u_chl), sst_by_start[s]
    return None


def download_file(url: str, dest: Path, user: str, pw: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.Session() as s:
                r = s.get(url, auth=(user, pw), stream=True,
                          timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                          allow_redirects=True)
                if r.status_code == 200:
                    ctype = r.headers.get("Content-Type", "")
                    if "html" in ctype:
                        # login page returned instead of file
                        print(f"    attempt {attempt}: got HTML (auth failed?)")
                    else:
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(chunk_size=64 * 1024):
                                if chunk: f.write(chunk)
                        return True
                else:
                    print(f"    attempt {attempt}: HTTP {r.status_code}")
        except (requests.ConnectionError, requests.Timeout) as ex:
            print(f"    attempt {attempt}: {type(ex).__name__}: {ex}")
        time.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))
    return False


def main() -> int:
    user = os.getenv("EARTHDATA_USER"); pw = os.getenv("EARTHDATA_PASS")
    if not user or not pw:
        print("ERROR: set EARTHDATA_USER and EARTHDATA_PASS in .env")
        return 1

    LIVE_MODIS_DIR.mkdir(parents=True, exist_ok=True)

    matched = find_matched_window()
    if not matched:
        print("No 8-day window with BOTH CHL and SST published.")
        return 2
    chl, sst = matched
    start, end = chl[0], chl[1]
    print(f"Newest matched VIIRS-SNPP window: {start} → {end}")

    ok_all = True
    for product, (_, _, url) in [("CHL", chl), ("SST", sst)]:
        fname = url.rsplit("/", 1)[-1]
        dest = LIVE_MODIS_DIR / fname
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"  {product}: cached ({fname})  {dest.stat().st_size/1e6:.1f} MB")
            continue
        print(f"  {product}: downloading {fname}")
        if not download_file(url, dest, user, pw):
            print(f"  {product}: FAILED")
            ok_all = False
        elif dest.exists():
            print(f"  {product}: OK ({dest.stat().st_size/1e6:.1f} MB)")

    latest = {
        "yyyyddd_start": start.isoformat(),
        "yyyyddd_end": end.isoformat(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "success": ok_all,
        "chl_filename": chl[2].rsplit("/", 1)[-1],
        "sst_filename": sst[2].rsplit("/", 1)[-1],
        "source": "VIIRS-SNPP NRT (MODIS-Aqua retired Feb 2025)",
    }
    (LIVE_MODIS_DIR / "latest.json").write_text(json.dumps(latest, indent=2))
    print(f"Wrote latest.json → window {start} to {end}")
    return 0 if ok_all else 3


if __name__ == "__main__":
    raise SystemExit(main())
