"""Find pages that mention many HAB variables together — likely water-quality tables."""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import pathlib, json, collections

HITS = pathlib.Path(str(_DATA_DIR / "_hits"))

# Only count variables likely to appear together in an env table
CORE = {"temperature","salinity","pH","phosphate","silicate","nitrite","nitrate","ammonia","chlorophyll","dissolved_O2","TSS","phytoplankton"}

for jf in sorted(HITS.glob("*.hits.json")):
    data = json.load(open(jf))
    page_groups = collections.defaultdict(set)
    for grp, entries in data["hits"].items():
        if grp not in CORE:
            continue
        for e in entries:
            page_groups[e["page"]].add(grp)
    dense = sorted(page_groups.items(), key=lambda kv: -len(kv[1]))
    print(f"\n=== {jf.stem} ===")
    for pg, groups in dense[:8]:
        print(f"  p{pg}: {len(groups)} vars -> {sorted(groups)}")
