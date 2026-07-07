
import pathlib as _pl
_REPO_ROOT = _pl.Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
import pdfplumber, os, sys, pathlib

SRC = pathlib.Path(str(_DATA_DIR))
OUT = SRC / "_text"
OUT.mkdir(exist_ok=True)

pdfs = sorted(SRC.glob("*.pdf"))
for p in pdfs:
    target = OUT / (p.stem + ".txt")
    if target.exists() and target.stat().st_size > 0:
        print(f"skip {p.name}")
        continue
    print(f"extracting {p.name} ...", flush=True)
    try:
        with pdfplumber.open(p) as pdf, open(target, "w") as f:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                f.write(f"\n===PAGE {i+1}===\n")
                f.write(txt)
        print(f"  -> {target.name} ({target.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"  ERROR: {e}")
print("done")
