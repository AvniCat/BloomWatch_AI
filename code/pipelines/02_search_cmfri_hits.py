"""Scan extracted CMFRI text for HAB-relevant sections.

For each report, find pages/passages that mention the target variables so we can
zoom into them for numeric extraction.
"""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import pathlib, re, json, sys

SRC = pathlib.Path(str(_DATA_DIR / "_text"))
OUT = pathlib.Path(str(_DATA_DIR / "_hits"))
OUT.mkdir(exist_ok=True)

# Keyword groups. Each group has a canonical name and regex patterns.
GROUPS = {
    "temperature":   r"(sea\s*surface\s*temperature|SST\b|bottom\s*temperature|water\s*temperature|delta\s*T|\bDelta\s*T\b|Δ\s*T)",
    "salinity":      r"salinity",
    "pH":            r"\bpH\b",
    "TSS":           r"(total\s*suspended\s*solids|\bTSS\b)",
    "phosphate":     r"phosphate",
    "silicate":      r"silicate",
    "nitrite":       r"nitrite",
    "nitrate":       r"nitrate",
    "ammonia":       r"(ammonia|ammonium)",
    "chlorophyll":   r"chlorophyll",
    "phytoplankton": r"phytoplankton",
    "upwelling":     r"upwell",
    "dissolved_O2":  r"(dissolved\s*oxygen|\bDO\b)",
    "HAB":           r"(harmful\s*algal\s*bloom|\bHAB\b|red\s*tide|algal\s*bloom)",
}

PAGE_RE = re.compile(r"^===PAGE (\d+)===\s*$")

def scan(txt_path):
    with open(txt_path) as f:
        raw = f.read()
    # split on page markers
    pages = {}
    cur = None
    buf = []
    for line in raw.splitlines():
        m = PAGE_RE.match(line)
        if m:
            if cur is not None:
                pages[cur] = "\n".join(buf)
            cur = int(m.group(1))
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        pages[cur] = "\n".join(buf)

    hits = {g: [] for g in GROUPS}
    for pg, body in pages.items():
        for g, pat in GROUPS.items():
            if re.search(pat, body, re.IGNORECASE):
                # capture short context
                snips = []
                for m in re.finditer(pat, body, re.IGNORECASE):
                    s = max(0, m.start()-80); e = min(len(body), m.end()+180)
                    snips.append(body[s:e].replace("\n", " "))
                    if len(snips) >= 3:
                        break
                hits[g].append({"page": pg, "snippets": snips})
    return hits, len(pages)

for txt in sorted(SRC.glob("*.txt")):
    hits, npages = scan(txt)
    summary = {g: len(v) for g, v in hits.items()}
    print(f"{txt.stem}: {npages} pages, hits per group: {summary}")
    out = OUT / (txt.stem + ".hits.json")
    with open(out, "w") as f:
        json.dump({"pages": npages, "hits": hits}, f, indent=1)
