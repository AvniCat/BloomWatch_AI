"""CLI tool to build/update the training manifest from a folder of newly
collected shellfish photos. Gemini drafts a first-pass label; a human must
fill in `human_label` before train.py will use the row — the draft label
is an accelerant, never ground truth on its own (see design spec).

Usage:
    python -m pipeline.photo_diagnosis.label_photos --photo-dir path/to/photos \
        --source market_contact
"""
from __future__ import annotations
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from chatbot import llm

MANIFEST_COLUMNS = ["filepath", "source", "date", "draft_label", "human_label"]
_DRAFT_PROMPT = (
    "Look at this photo of a bivalve shellfish (oyster, mussel, or clam). "
    "Is the shell open and gaping, or closed/normal? Answer with exactly "
    "one word: 'gaping' or 'closed'."
)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_manifest(photo_dir: Path, source: str, manifest_path: Path) -> int:
    photo_dir = Path(photo_dir)
    manifest_path = Path(manifest_path)

    existing_paths = set()
    rows = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        existing_paths = {r["filepath"] for r in rows}

    new_count = 0
    for photo_path in sorted(photo_dir.iterdir()):
        if photo_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        filepath = str(photo_path.resolve())
        if filepath in existing_paths:
            continue
        draft = llm.vision_label(filepath, _DRAFT_PROMPT).strip().lower()
        draft = draft if draft in ("gaping", "closed") else "unclear"
        rows.append({
            "filepath": filepath,
            "source": source,
            "date": date.today().isoformat(),
            "draft_label": draft,
            "human_label": "",
        })
        new_count += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return new_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", required=True, type=Path)
    parser.add_argument("--source", required=True,
                         help="e.g. market_contact, cmfri, self_collected")
    parser.add_argument("--manifest", type=Path, default=config.PHOTO_MANIFEST_PATH)
    args = parser.parse_args()

    count = build_manifest(args.photo_dir, args.source, args.manifest)
    print(f"Added {count} new photo(s) to {args.manifest}")
    print("Open the manifest CSV and fill in `human_label` for every new row "
          "before running train.py — draft_label is a starting point, not "
          "ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
