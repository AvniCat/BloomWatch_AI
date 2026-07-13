"""Chain the weekly refresh: pull_modis → pull_imd → build_features → predict.

Exits non-zero if any stage fails; the GitHub Action then flags it.
Meant to run every Friday when a new MODIS 8-day composite lands.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

STAGES = [
    ("MODIS fetch",     [sys.executable, "-m", "pipeline.pull_modis"]),
    ("IMD fetch",       [sys.executable, "-m", "pipeline.pull_imd"]),
    ("Feature build",   [sys.executable, "-m", "pipeline.build_features"]),
    ("Inference",       [sys.executable, "-m", "pipeline.predict"]),
]


def main() -> int:
    for label, cmd in STAGES:
        print(f"\n=== {label} ===")
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"STAGE FAILED: {label} (exit {r.returncode})")
            return r.returncode
    print("\nAll stages succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
