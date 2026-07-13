"""Scrape rainfall from IMD's district-level dashboard endpoint.

IMD's public NetCDF archive (rainfall_YYYY.nc) is intermittent — most of 2025-2026
returns 404. But their live monsoon-season dashboard
(https://mausam.imd.gov.in/responsive/rainfallinformation/rfi_district.inc.php)
exposes per-district cumulative rainfall as embedded JSON.

Strategy:
  1. Scrape the district endpoint for each state (Kerala + Karnataka).
  2. Keep only COASTAL districts that are ecologically relevant to shellfish farmers.
  3. Aggregate to region-level stats matching the historical CSV schema.
  4. Ledger each snapshot with timestamp so weekly refreshes can compute deltas.

Fallback: if IMD is unreachable, use climatological mean from historical CSV.

Outputs:
  data/live/imd/imd_snapshot_<yyyy-mm-dd>.json   raw scraped districts
  data/live/imd/current_window.json               per-region stats for the current window
"""
from __future__ import annotations
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LIVE_IMD_DIR, LIVE_MODIS_DIR, REGIONS, HISTORICAL_CSV

IMD_ENDPOINT = "https://mausam.imd.gov.in/responsive/rainfallinformation/rfi_district.inc.php"

# Coastal districts that matter for shellfish cooperatives.
# (IMD district names in UPPERCASE — normalise on comparison.)
COASTAL_DISTRICTS = {
    "Kerala": {
        "THIRUVANANTHAPURAM", "KOLLAM", "ALAPPUZHA", "ERNAKULAM", "THRISSUR",
        "MALAPPURAM", "KOZHIKODE", "KANNUR", "KASARGOD", "KOTTAYAM",
    },
    "Karnataka": {
        "DAKSHIN KANNADA", "UDUPI", "UTTAR KANNADA",
    },
}


def _fetch_state(state: str) -> list[dict]:
    """Fetch and parse per-district records for one state.

    Returns list of {district, actual_mm, normal_mm, departure_pct, date_iso}.
    """
    r = requests.get(IMD_ENDPOINT, params={"id": state}, timeout=30)
    r.raise_for_status()
    text = r.text

    # The page embeds each district as a JS object literal with balloonText
    # containing 'Date : YYYY-MM-DD', 'Actual : NN mm', 'Normal : NN mm'.
    pattern = re.compile(
        r'"title":\s*"([^"]+)"'
        r'.*?"info":\s*"([-\d.]+%)"'
        r'.*?Date\s*:\s*(\d{4}-\d{2}-\d{2})'
        r'.*?Actual\s*:\s*([\d.]+)\s*mm'
        r'.*?Normal\s*:\s*([\d.]+)\s*mm',
        re.DOTALL,
    )
    out = []
    for m in pattern.finditer(text):
        title, dep_str, date_iso, actual, normal = m.groups()
        try:
            out.append({
                "district": title.strip().upper(),
                "actual_mm": float(actual),
                "normal_mm": float(normal),
                "departure_pct": float(dep_str.rstrip("%")),
                "date_iso": date_iso,
            })
        except ValueError:
            continue
    return out


def _load_previous_snapshot(before: date) -> dict | None:
    """Return the most recent snapshot strictly BEFORE the given date, if any."""
    snapshots = sorted(LIVE_IMD_DIR.glob("imd_snapshot_*.json"))
    for path in reversed(snapshots):
        m = re.search(r"imd_snapshot_(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not m: continue
        d = date.fromisoformat(m.group(1))
        if d < before:
            return json.loads(path.read_text())
    return None


def _aggregate_region(records: list[dict], state: str) -> dict:
    """Aggregate district records for one region into pipeline-compatible stats."""
    keep = [r for r in records if r["district"] in COASTAL_DISTRICTS[state]]
    if not keep:
        return {
            "rainfall_mm_total_mean": 0.0, "rainfall_mm_total_min": 0.0,
            "rainfall_mm_total_max": 0.0, "rainfall_mm_max_daily": 0.0,
            "rainy_days_mean": 0.0, "rainfall_n_cells": 0,
        }
    actuals = [r["actual_mm"] for r in keep]
    return {
        "rainfall_mm_total_mean": float(sum(actuals) / len(actuals)),
        "rainfall_mm_total_min":  float(min(actuals)),
        "rainfall_mm_total_max":  float(max(actuals)),
        # Best proxy we have without a daily series: assume even distribution
        "rainfall_mm_max_daily":  float(max(actuals) / 40),  # ~40 day cumulative
        "rainy_days_mean":        float(sum(1 for a in actuals if a > 10) / len(actuals) * 8),
        "rainfall_n_cells":       len(keep),
    }


def _windowed_from_snapshots(curr: dict, prev: dict, state: str) -> dict | None:
    """Given current and previous IMD snapshots, compute the true 8-day window delta."""
    curr_records = {r["district"]: r for r in curr["states"].get(state, [])}
    prev_records = {r["district"]: r for r in prev["states"].get(state, [])}
    common = COASTAL_DISTRICTS[state] & set(curr_records) & set(prev_records)
    if not common: return None

    deltas = []
    for d in common:
        d_actual = curr_records[d]["actual_mm"] - prev_records[d]["actual_mm"]
        deltas.append(max(0.0, d_actual))  # guard against noise/reset

    if not deltas: return None
    return {
        "rainfall_mm_total_mean": float(sum(deltas) / len(deltas)),
        "rainfall_mm_total_min":  float(min(deltas)),
        "rainfall_mm_total_max":  float(max(deltas)),
        "rainfall_mm_max_daily":  float(max(deltas) / 8),
        "rainy_days_mean":        float(sum(1 for d in deltas if d > 5) / len(deltas) * 8),
        "rainfall_n_cells":       len(deltas),
    }


def _fallback_climatological(start: date) -> dict:
    """When IMD is unreachable, use climatological mean from historical CSV."""
    import pandas as pd
    hist = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start"])
    hist["doy"] = hist["date_start"].dt.dayofyear
    target_doy = (start - date(start.year, 1, 1)).days + 1

    per_region = {}
    for region in REGIONS:
        rows = hist[(hist.region == region) &
                    (hist.doy.between(target_doy - 4, target_doy + 4))]
        if rows.empty:
            per_region[region] = {
                "rainfall_mm_total_mean": 0.0, "rainfall_mm_total_min": 0.0,
                "rainfall_mm_total_max": 0.0, "rainfall_mm_max_daily": 0.0,
                "rainy_days_mean": 0.0, "rainfall_n_cells": 0,
            }
        else:
            per_region[region] = {
                "rainfall_mm_total_mean": float(rows["rainfall_mm_total_mean"].mean()),
                "rainfall_mm_total_min":  float(rows["rainfall_mm_total_min"].mean()),
                "rainfall_mm_total_max":  float(rows["rainfall_mm_total_max"].mean()),
                "rainfall_mm_max_daily":  float(rows["rainfall_mm_max_daily"].mean()),
                "rainy_days_mean":        float(rows["rainy_days_mean"].mean()),
                "rainfall_n_cells":       int(rows["rainfall_n_cells"].median()),
            }
    return per_region


def _read_window(modis_latest_path: Path) -> tuple[date, date] | None:
    if not modis_latest_path.exists(): return None
    j = json.loads(modis_latest_path.read_text())
    return date.fromisoformat(j["yyyyddd_start"]), date.fromisoformat(j["yyyyddd_end"])


def main() -> int:
    LIVE_IMD_DIR.mkdir(parents=True, exist_ok=True)
    window = _read_window(LIVE_MODIS_DIR / "latest.json")
    if window is None:
        print("ERROR: no MODIS window found — run pull_modis first")
        return 2
    start, end = window

    # ---- Scrape IMD for both states ----
    snapshot = {"scraped_at": datetime.now(timezone.utc).isoformat(), "states": {}}
    scrape_ok = True
    for state in ("Kerala", "Karnataka"):
        try:
            records = _fetch_state(state)
            snapshot["states"][state] = records
            n_coastal = sum(1 for r in records if r["district"] in COASTAL_DISTRICTS[state])
            print(f"  {state}: {len(records)} districts scraped, {n_coastal} coastal")
        except Exception as e:
            print(f"  {state}: scrape failed — {type(e).__name__}: {e}")
            scrape_ok = False

    # ---- Ledger the snapshot ----
    snapshot_date = date.today().isoformat()
    if scrape_ok:
        snap_path = LIVE_IMD_DIR / f"imd_snapshot_{snapshot_date}.json"
        snap_path.write_text(json.dumps(snapshot, indent=2))
        print(f"Wrote snapshot: {snap_path.name}")

    # ---- Prefer windowed delta if a previous snapshot exists ----
    per_region = {}
    source_used = "unknown"
    if scrape_ok:
        prev = _load_previous_snapshot(before=date.today())
        for region in REGIONS:
            if prev:
                delta = _windowed_from_snapshots(snapshot, prev, region)
                if delta:
                    per_region[region] = delta
                    source_used = "imd_windowed_delta"
                    continue
            per_region[region] = _aggregate_region(snapshot["states"].get(region, []), region)
            source_used = "imd_cumulative_snapshot"
    else:
        per_region = _fallback_climatological(start)
        source_used = "climatological_fallback"

    out = {
        "window_start": start.isoformat(),
        "window_end":   end.isoformat(),
        "computed_at":  datetime.now(timezone.utc).isoformat(),
        "source":       source_used,
        "regions":      per_region,
    }
    (LIVE_IMD_DIR / "current_window.json").write_text(json.dumps(out, indent=2))
    print(f"\nSource: {source_used}")
    for r, s in per_region.items():
        print(f"  {r}: {s['rainfall_mm_total_mean']:.1f} mm mean across {s['rainfall_n_cells']} districts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
