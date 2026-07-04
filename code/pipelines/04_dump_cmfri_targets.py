"""Dump the top HAB-dense pages from each report into a single file for reading."""

import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
import pathlib, re, json, collections

TEXT = pathlib.Path(str(_DATA_DIR / "_text"))
HITS = pathlib.Path(str(_DATA_DIR / "_hits"))
OUT = pathlib.Path(str(_DATA_DIR / "_targets.txt"))

CORE = {"temperature","salinity","pH","phosphate","silicate","nitrite","nitrate","ammonia","chlorophyll","dissolved_O2","TSS","phytoplankton"}

# For each report, take top-N dense pages, plus a small window around each (page ±1).
PAGE_RE = re.compile(r"^===PAGE (\d+)===\s*$")

def parse_pages(txt):
    pages = {}; cur=None; buf=[]
    for line in txt.splitlines():
        m = PAGE_RE.match(line)
        if m:
            if cur is not None: pages[cur] = "\n".join(buf)
            cur = int(m.group(1)); buf=[]
        else:
            buf.append(line)
    if cur is not None: pages[cur] = "\n".join(buf)
    return pages

with open(OUT, "w") as fout:
    for jf in sorted(HITS.glob("*.hits.json")):
        stem = jf.name.replace(".hits.json", "")
        data = json.load(open(jf))
        pg_groups = collections.defaultdict(set)
        for grp, entries in data["hits"].items():
            if grp not in CORE: continue
            for e in entries:
                pg_groups[e["page"]].add(grp)
        dense = sorted(pg_groups.items(), key=lambda kv: -len(kv[1]))
        # take pages with >=4 co-occurring vars, top 4
        keep = [pg for pg,gs in dense if len(gs)>=4][:4]
        if not keep:
            keep = [pg for pg,_ in dense[:3]]
        # include ±1 neighbours to catch table continuations
        expand = set()
        for pg in keep:
            expand.update({pg-1, pg, pg+1})

        text_path = TEXT / (stem + ".txt")
        pages = parse_pages(text_path.read_text())
        fout.write(f"\n\n########## {stem} ##########\n")
        fout.write(f"# dense pages: {keep} (expanded {sorted(expand)})\n")
        for pg in sorted(expand):
            if pg in pages:
                fout.write(f"\n----- {stem} PAGE {pg} -----\n")
                fout.write(pages[pg])
print(f"wrote {OUT}, size = {OUT.stat().st_size} bytes")
