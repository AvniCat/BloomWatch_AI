# Shellfish Photo Diagnosis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a binary "gaping shell / closed" photo classifier to BloomWatch AI, served through a new `/diagnose-photo` API endpoint and integrated into the existing chatbot's evidence-and-answer pipeline.

**Architecture:** A transfer-learning image classifier (frozen MobileNetV2 backbone + trained linear head) lives in a new `app/pipeline/photo_diagnosis/` package alongside a labeling CLI and a training CLI. The orchestrator gets a new `diagnose_photo()` function that mirrors the existing `answer()` function's evidence-assembly-then-LLM-call pattern. A new FastAPI endpoint accepts an uploaded image and returns a farmer-facing answer generated through the same plain-language system prompt already in production.

**Tech Stack:** PyTorch + torchvision (MobileNetV2, ImageNet-pretrained), Pillow for image I/O, existing FastAPI/pytest/Gemini stack.

## Global Constraints

- Target is binary: `gaping` vs `closed`. No species ID, no disease/toxin diagnosis, no water-discoloration classification (per spec).
- Never output a "safe to harvest" / "do not harvest" verdict — only a hedged distress signal + CMFRI referral for anything concerning.
- Classifier confidence and Gemini vision confidence are never blended into one fused number. Gemini vision (`vision_label`) is used only for (a) draft-labeling assistance during data collection and (b) generating the farmer-facing explanation text — never as an input to the trained classifier's output.
- Train/test splits must be by `source`/`batch`, never a random shuffle (matches the existing temporal-split convention in `code/notebooks/BloomWatchAI_Calibration.ipynb`).
- Reported metrics must include bootstrap confidence intervals, not point estimates alone.
- All new code follows the existing import convention: `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` then bare imports (see `app/pipeline/predict.py`, `app/chatbot/llm.py`).
- `app/chatbot/llm.py` remains the only module that talks to LLM APIs (per its own docstring) — the new vision call goes there, not into a new file.

---

## Task 1: Config additions for photo-diagnosis paths

**Files:**
- Modify: `app/config.py`
- Test: `app/tests/test_photo_diagnosis.py` (new file, created in this task)

**Interfaces:**
- Produces: `PHOTO_MODEL_PATH: Path`, `PHOTO_MODEL_META_PATH: Path`, `PHOTO_MANIFEST_PATH: Path`, `GAPING_CONFIDENCE_THRESHOLD: float` — all importable from `config`.

- [ ] **Step 1: Write the failing test**

Create `app/tests/test_photo_diagnosis.py`:

```python
"""Tests for the shellfish photo-diagnosis feature."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_photo_config_paths_exist():
    import config
    assert isinstance(config.PHOTO_MODEL_PATH, Path)
    assert isinstance(config.PHOTO_MODEL_META_PATH, Path)
    assert isinstance(config.PHOTO_MANIFEST_PATH, Path)
    assert 0.0 < config.GAPING_CONFIDENCE_THRESHOLD < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py::test_photo_config_paths_exist -v`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'PHOTO_MODEL_PATH'`

- [ ] **Step 3: Add the config values**

In `app/config.py`, immediately after the existing `MODEL_VERSION = "bloomwatch-xgb-v1"` line, add:

```python

# Photo diagnosis (gaping-shell classifier)
PHOTO_MODEL_DIR = APP_ROOT / "models" / "photo_diagnosis"
PHOTO_MODEL_PATH = Path(os.getenv("PHOTO_MODEL_PATH", PHOTO_MODEL_DIR / "gaping_classifier.pt"))
PHOTO_MODEL_META_PATH = PHOTO_MODEL_PATH.with_suffix(".meta.json")
PHOTO_MANIFEST_PATH = Path(os.getenv("PHOTO_MANIFEST_PATH", DATA_DIR / "photo_diagnosis/manifest.csv"))
GAPING_CONFIDENCE_THRESHOLD = float(os.getenv("GAPING_CONFIDENCE_THRESHOLD", "0.6"))
PHOTO_MODEL_VERSION = "bloomwatch-gaping-v1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py::test_photo_config_paths_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/tests/test_photo_diagnosis.py
git commit -m "Add config paths for photo-diagnosis feature"
```

---

## Task 2: Classifier model wrapper

**Files:**
- Create: `app/pipeline/photo_diagnosis/__init__.py`
- Create: `app/pipeline/photo_diagnosis/model.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `config.PHOTO_MODEL_PATH`, `config.PHOTO_MODEL_META_PATH`
- Produces: `build_model() -> torch.nn.Module`, `class GapingClassifier` with `.predict(image_path: str | Path) -> dict` (returns `{"label": "gaping" | "closed", "confidence": float}`), `GapingClassifier.load(path: Path) -> GapingClassifier`, `.save(self, path: Path) -> None`. Later tasks (5, 7) call `GapingClassifier.load(config.PHOTO_MODEL_PATH).predict(image_path)`.

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
import tempfile
from PIL import Image


def _make_test_image(path, color=(120, 150, 180)):
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)


def test_classifier_predict_returns_valid_shape():
    from pipeline.photo_diagnosis.model import GapingClassifier

    clf = GapingClassifier()  # untrained (ImageNet-pretrained backbone, random head)
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "test.jpg"
        _make_test_image(img_path)
        result = clf.predict(img_path)

    assert result["label"] in ("gaping", "closed")
    assert 0.0 <= result["confidence"] <= 1.0


def test_classifier_save_and_load_roundtrip():
    from pipeline.photo_diagnosis.model import GapingClassifier

    clf = GapingClassifier()
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "model.pt"
        clf.save(save_path)
        assert save_path.exists()

        loaded = GapingClassifier.load(save_path)
        img_path = Path(tmp) / "test.jpg"
        _make_test_image(img_path)
        result = loaded.predict(img_path)
        assert result["label"] in ("gaping", "closed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k classifier`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.photo_diagnosis'`

- [ ] **Step 3: Write the implementation**

Create `app/pipeline/photo_diagnosis/__init__.py` (empty file).

Create `app/pipeline/photo_diagnosis/model.py`:

```python
"""Gaping-shell binary classifier — transfer learning on a frozen MobileNetV2
backbone. Small dataset (dozens-to-low-hundreds of images), so we fine-tune
only the final classification head, not the whole network. See
docs/superpowers/specs/2026-08-18-shellfish-photo-diagnosis-design.md for
why transfer learning was chosen over training from scratch.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

LABELS = ["closed", "gaping"]  # index 0, 1 — must match training label encoding

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    """ImageNet-pretrained MobileNetV2 with a fresh 2-class head. Backbone
    frozen — only the head is meant to be trained (see train.py)."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(LABELS))
    return model


class GapingClassifier:
    def __init__(self, model: nn.Module | None = None):
        self.model = model if model is not None else build_model()
        self.model.eval()

    def predict(self, image_path: str | Path) -> dict:
        img = Image.open(image_path).convert("RGB")
        x = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return {"label": LABELS[idx], "confidence": float(probs[idx].item())}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "GapingClassifier":
        model = build_model()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k classifier`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/photo_diagnosis/ app/tests/test_photo_diagnosis.py
git commit -m "Add gaping-shell classifier (transfer learning on MobileNetV2)"
```

---

## Task 3: Low-confidence fallback logic

**Files:**
- Modify: `app/pipeline/photo_diagnosis/model.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `config.GAPING_CONFIDENCE_THRESHOLD`
- Produces: `apply_confidence_fallback(result: dict) -> dict` — adds `"fallback": bool` and, when `True`, replaces `label` with `"unclear"`. Task 7 (orchestrator) calls this on every classifier result before building evidence.

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
def test_confidence_fallback_triggers_below_threshold():
    from pipeline.photo_diagnosis.model import apply_confidence_fallback
    import config

    low = {"label": "gaping", "confidence": config.GAPING_CONFIDENCE_THRESHOLD - 0.1}
    result = apply_confidence_fallback(low)
    assert result["fallback"] is True
    assert result["label"] == "unclear"


def test_confidence_fallback_not_triggered_above_threshold():
    from pipeline.photo_diagnosis.model import apply_confidence_fallback
    import config

    high = {"label": "gaping", "confidence": config.GAPING_CONFIDENCE_THRESHOLD + 0.1}
    result = apply_confidence_fallback(high)
    assert result["fallback"] is False
    assert result["label"] == "gaping"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k fallback`
Expected: FAIL with `ImportError: cannot import name 'apply_confidence_fallback'`

- [ ] **Step 3: Write the implementation**

Append to `app/pipeline/photo_diagnosis/model.py`:

```python


def apply_confidence_fallback(result: dict) -> dict:
    """Below GAPING_CONFIDENCE_THRESHOLD, don't force a possibly-wrong call —
    surface it as unclear so the orchestrator can route to a hedged reply
    instead of a confident-sounding label."""
    is_low = result["confidence"] < config.GAPING_CONFIDENCE_THRESHOLD
    return {
        **result,
        "label": "unclear" if is_low else result["label"],
        "fallback": is_low,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k fallback`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/photo_diagnosis/model.py app/tests/test_photo_diagnosis.py
git commit -m "Add low-confidence fallback for gaping classifier"
```

---

## Task 4: Gemini vision draft-labeling function

**Files:**
- Modify: `app/chatbot/llm.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `GEMINI_API_KEY`, `GEMINI_MODEL` (existing module-level config in `llm.py`)
- Produces: `vision_label(image_path: str | Path, prompt: str) -> str`. Task 5's labeling script calls this; tests mock it rather than hitting the real API (matches the existing project convention — see `test_chatbot.py`'s docstring: "We don't run the actual LLM in unit tests").

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
def test_vision_label_is_callable_and_mockable(monkeypatch):
    from chatbot import llm

    def fake_gemini_vision(image_path, prompt):
        return "gaping"

    monkeypatch.setattr(llm, "_gemini_vision", fake_gemini_vision)
    result = llm.vision_label("fake/path.jpg", "Is this shell open or closed?")
    assert result == "gaping"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k vision_label`
Expected: FAIL with `AttributeError: module 'chatbot.llm' has no attribute 'vision_label'`

- [ ] **Step 3: Write the implementation**

In `app/chatbot/llm.py`, add after the existing `_gemini_chat` function (after line 36):

```python

def _gemini_vision(image_path, prompt: str) -> str:
    import google.generativeai as genai
    import PIL.Image
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    img = PIL.Image.open(image_path)
    resp = model.generate_content([prompt, img])
    return resp.text.strip()
```

Then add to the "Public API" section, after the existing `chat()` function (after line 89):

```python

def vision_label(image_path, prompt: str) -> str:
    """Draft-label an image via Gemini vision. Used only for (a) speeding up
    manual photo labeling during data collection and (b) generating
    farmer-facing explanation text — never as an input to the trained
    classifier's confidence score (see Global Constraints in the design
    spec: blending calibrated + uncalibrated confidence is explicitly
    rejected)."""
    return _gemini_vision(image_path, prompt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k vision_label`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/chatbot/llm.py app/tests/test_photo_diagnosis.py
git commit -m "Add Gemini vision draft-labeling function to llm.py"
```

---

## Task 5: Photo labeling CLI script

**Files:**
- Create: `app/pipeline/photo_diagnosis/label_photos.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `chatbot.llm.vision_label()`, `config.PHOTO_MANIFEST_PATH`
- Produces: `build_manifest(photo_dir: Path, source: str, manifest_path: Path) -> int` (returns count of rows written/updated). This is the tool Avni/Maya run against a folder of newly collected photos; it never sets `human_label` itself — that column starts blank and must be filled in by a human before `train.py` (Task 6) will use the row.

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
import csv


def test_build_manifest_writes_expected_columns(monkeypatch, tmp_path):
    from pipeline.photo_diagnosis.label_photos import build_manifest
    from chatbot import llm

    monkeypatch.setattr(llm, "vision_label", lambda path, prompt: "gaping")

    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    _make_test_image(photo_dir / "shell1.jpg")
    _make_test_image(photo_dir / "shell2.jpg")

    manifest_path = tmp_path / "manifest.csv"
    count = build_manifest(photo_dir, source="self_collected", manifest_path=manifest_path)

    assert count == 2
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"filepath", "source", "date", "draft_label", "human_label"}
    assert rows[0]["draft_label"] == "gaping"
    assert rows[0]["human_label"] == ""  # blank until a human fills it in
    assert rows[0]["source"] == "self_collected"


def test_build_manifest_appends_without_duplicating(monkeypatch, tmp_path):
    from pipeline.photo_diagnosis.label_photos import build_manifest
    from chatbot import llm

    monkeypatch.setattr(llm, "vision_label", lambda path, prompt: "closed")

    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    _make_test_image(photo_dir / "shell1.jpg")

    manifest_path = tmp_path / "manifest.csv"
    build_manifest(photo_dir, source="self_collected", manifest_path=manifest_path)
    count_second_run = build_manifest(photo_dir, source="self_collected", manifest_path=manifest_path)

    assert count_second_run == 0  # shell1.jpg already in manifest, not re-added
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k build_manifest`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.photo_diagnosis.label_photos'`

- [ ] **Step 3: Write the implementation**

Create `app/pipeline/photo_diagnosis/label_photos.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k build_manifest`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/photo_diagnosis/label_photos.py app/tests/test_photo_diagnosis.py
git commit -m "Add photo labeling CLI with Gemini-assisted draft labels"
```

---

## Task 6: Training script with source-based split and bootstrap CIs

**Files:**
- Create: `app/pipeline/photo_diagnosis/train.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `pipeline.photo_diagnosis.model.build_model`, `GapingClassifier`, `LABELS`
- Produces: `load_labeled_manifest(manifest_path: Path) -> list[dict]` (filters to rows with non-empty `human_label`), `source_split(rows: list[dict]) -> tuple[list[dict], list[dict]]` (train, test — split by distinct `source` values, never a random per-row shuffle), `bootstrap_ci(rows: list[dict], metric_fn, n_draws: int = 200) -> dict` (returns `{"mean": float, "lo": float, "hi": float}`), `train_and_evaluate(manifest_path: Path, out_dir: Path) -> dict` (the end-to-end entry point; returns the metrics dict it also writes to `out_dir/metrics.json`).

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
import random


def _write_synthetic_manifest(tmp_path, n_per_source=6):
    """Two sources, deterministic colors standing in for gaping vs closed,
    so the tiny trained model can actually learn something real in the test
    (not just prove the code runs) without needing real photos."""
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    rows = []
    random.seed(7)
    for source in ("source_a", "source_b"):
        for i in range(n_per_source):
            label = "gaping" if i % 2 == 0 else "closed"
            color = (200, 60, 60) if label == "gaping" else (60, 60, 200)
            path = photo_dir / f"{source}_{i}.jpg"
            _make_test_image(path, color=color)
            rows.append({
                "filepath": str(path.resolve()),
                "source": source,
                "date": "2026-08-01",
                "draft_label": label,
                "human_label": label,
            })
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "source", "date", "draft_label", "human_label"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_load_labeled_manifest_filters_unlabeled(tmp_path):
    from pipeline.photo_diagnosis.train import load_labeled_manifest

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "source", "date", "draft_label", "human_label"])
        writer.writeheader()
        writer.writerow({"filepath": "a.jpg", "source": "s1", "date": "2026-01-01", "draft_label": "gaping", "human_label": "gaping"})
        writer.writerow({"filepath": "b.jpg", "source": "s1", "date": "2026-01-01", "draft_label": "closed", "human_label": ""})

    rows = load_labeled_manifest(manifest_path)
    assert len(rows) == 1
    assert rows[0]["filepath"] == "a.jpg"


def test_source_split_never_mixes_sources_across_splits():
    from pipeline.photo_diagnosis.train import source_split

    rows = (
        [{"source": "s1", "filepath": f"a{i}.jpg", "human_label": "gaping"} for i in range(4)]
        + [{"source": "s2", "filepath": f"b{i}.jpg", "human_label": "closed"} for i in range(4)]
    )
    train_rows, test_rows = source_split(rows)
    train_sources = {r["source"] for r in train_rows}
    test_sources = {r["source"] for r in test_rows}
    assert train_sources.isdisjoint(test_sources)
    assert len(train_rows) > 0 and len(test_rows) > 0


def test_bootstrap_ci_returns_mean_lo_hi():
    from pipeline.photo_diagnosis.train import bootstrap_ci

    rows = [{"correct": 1} for _ in range(8)] + [{"correct": 0} for _ in range(2)]
    result = bootstrap_ci(rows, metric_fn=lambda rs: sum(r["correct"] for r in rs) / len(rs), n_draws=50)
    assert set(result.keys()) == {"mean", "lo", "hi"}
    assert 0.0 <= result["lo"] <= result["mean"] <= result["hi"] <= 1.0


def test_train_and_evaluate_end_to_end(tmp_path):
    from pipeline.photo_diagnosis.train import train_and_evaluate

    manifest_path = _write_synthetic_manifest(tmp_path)
    out_dir = tmp_path / "out"
    metrics = train_and_evaluate(manifest_path, out_dir)

    assert (out_dir / "gaping_classifier.pt").exists()
    assert (out_dir / "metrics.json").exists()
    assert "accuracy" in metrics
    assert set(metrics["accuracy"].keys()) == {"mean", "lo", "hi"}
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert metrics["test_sources"]  # non-empty list of held-out sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k "manifest_filters or source_split or bootstrap_ci or train_and_evaluate"`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.photo_diagnosis.train'`

- [ ] **Step 3: Write the implementation**

Create `app/pipeline/photo_diagnosis/train.py`:

```python
"""Train the gaping-shell classifier from a labeled manifest CSV.

Splits by source/batch (never a random per-row shuffle), fine-tunes only
the classification head on top of a frozen pretrained backbone, and reports
bootstrap confidence intervals rather than a bare point estimate — same
rigor pattern as the forecast model's power analysis.

Usage:
    python -m pipeline.photo_diagnosis.train --manifest path/to/manifest.csv \
        --out-dir models/photo_diagnosis
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from pipeline.photo_diagnosis.model import GapingClassifier, LABELS
from torchvision import transforms

# Augmentation applied only to training images, never at inference/predict
# time (see model.py's _TRANSFORM, which stays augmentation-free) — this is
# what lets a few dozen photos stand in for a larger dataset (per spec).
_TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_labeled_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("human_label", "").strip()]


def source_split(rows: list[dict], test_fraction: float = 0.3, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Hold out whole sources, not individual rows, so the test set proves
    generalization to a source the model never saw during training."""
    sources = sorted({r["source"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_test_sources = max(1, round(len(sources) * test_fraction))
    test_sources = set(sources[:n_test_sources])
    train_rows = [r for r in rows if r["source"] not in test_sources]
    test_rows = [r for r in rows if r["source"] in test_sources]
    return train_rows, test_rows


def bootstrap_ci(rows: list[dict], metric_fn, n_draws: int = 200, seed: int = 7) -> dict:
    rng = random.Random(seed)
    n = len(rows)
    draws = []
    for _ in range(n_draws):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        draws.append(metric_fn(sample))
    draws.sort()
    return {
        "mean": sum(draws) / len(draws),
        "lo": draws[int(0.025 * len(draws))],
        "hi": draws[min(len(draws) - 1, int(0.975 * len(draws)))],
    }


def _train_head(rows: list[dict], epochs: int = 5, lr: float = 1e-3) -> nn.Module:
    clf = GapingClassifier()
    model = clf.model
    model.train()
    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    images, targets = [], []
    for r in rows:
        img = Image.open(r["filepath"]).convert("RGB")
        images.append(_TRAIN_AUGMENT(img))
        targets.append(label_to_idx[r["human_label"].strip().lower()])
    x = torch.stack(images)
    y = torch.tensor(targets, dtype=torch.long)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def train_and_evaluate(manifest_path: Path, out_dir: Path) -> dict:
    rows = load_labeled_manifest(manifest_path)
    train_rows, test_rows = source_split(rows)

    model = _train_head(train_rows)
    clf = GapingClassifier(model)

    for r in test_rows:
        r["predicted_label"] = clf.predict(r["filepath"])["label"]
        r["correct"] = int(r["predicted_label"] == r["human_label"].strip().lower())

    accuracy = bootstrap_ci(test_rows, lambda rs: sum(r["correct"] for r in rs) / len(rs))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save(out_dir / "gaping_classifier.pt")

    metrics = {
        "model_version": config.PHOTO_MODEL_VERSION,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_sources": sorted({r["source"] for r in test_rows}),
        "accuracy": accuracy,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=config.PHOTO_MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.PHOTO_MODEL_DIR)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found at {args.manifest}. Run label_photos.py first.")
        return 1

    metrics = train_and_evaluate(args.manifest, args.out_dir)
    print(json.dumps(metrics, indent=2))
    print(
        f"\nTrained on {metrics['n_train']} photos, tested on "
        f"{metrics['n_test']} held out from source(s) {metrics['test_sources']}.\n"
        f"Accuracy: {metrics['accuracy']['mean']:.3f} "
        f"[{metrics['accuracy']['lo']:.3f}, {metrics['accuracy']['hi']:.3f}]\n"
        "Add these numbers, the exact N, and any untested conditions to "
        "LIMITATIONS.md before citing this model anywhere."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k "manifest_filters or source_split or bootstrap_ci or train_and_evaluate"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/photo_diagnosis/train.py app/tests/test_photo_diagnosis.py
git commit -m "Add training script with source-based split and bootstrap CIs"
```

---

## Task 7: Orchestrator integration

**Files:**
- Modify: `app/chatbot/orchestrator.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `pipeline.photo_diagnosis.model.GapingClassifier`, `apply_confidence_fallback`, `config.PHOTO_MODEL_PATH`, existing `chatbot.llm.chat`, existing `chatbot.vectorstore.retrieve`, existing `SYSTEM_PROMPT`
- Produces: `diagnose_photo(image_path: str | Path) -> dict` (returns `{"answer": str, "label": str, "confidence": float, "fallback": bool, "route": "photo_diagnosis"}`). Task 8 (API endpoint) calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
def test_diagnose_photo_assembles_evidence_and_calls_chat(monkeypatch, tmp_path):
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    class FakeClassifier:
        def predict(self, path):
            return {"label": "gaping", "confidence": 0.9}

    monkeypatch.setattr(photo_model.GapingClassifier, "load", classmethod(lambda cls, p: FakeClassifier()))

    captured = {}
    def fake_chat(prompt, system=None):
        captured["prompt"] = prompt
        captured["system"] = system
        return "Stop harvesting and contact your local CMFRI extension centre."
    monkeypatch.setattr(orchestrator, "chat", fake_chat)
    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])

    result = orchestrator.diagnose_photo(img_path)

    assert result["label"] == "gaping"
    assert result["confidence"] == 0.9
    assert result["fallback"] is False
    assert result["route"] == "photo_diagnosis"
    assert "gaping" in captured["prompt"].lower()
    assert captured["system"] == orchestrator.SYSTEM_PROMPT


def test_diagnose_photo_low_confidence_routes_to_fallback(monkeypatch, tmp_path):
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    class FakeClassifier:
        def predict(self, path):
            return {"label": "gaping", "confidence": 0.2}

    monkeypatch.setattr(photo_model.GapingClassifier, "load", classmethod(lambda cls, p: FakeClassifier()))

    captured = {}
    def fake_chat(prompt, system=None):
        captured["prompt"] = prompt
        return "I couldn't get a clear read from that photo — try a closer, well-lit shot."
    monkeypatch.setattr(orchestrator, "chat", fake_chat)
    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])

    result = orchestrator.diagnose_photo(img_path)

    assert result["fallback"] is True
    assert result["label"] == "unclear"
    assert "unclear" in captured["prompt"].lower() or "couldn't" in captured["prompt"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k diagnose_photo`
Expected: FAIL with `AttributeError: module 'chatbot.orchestrator' has no attribute 'diagnose_photo'`

- [ ] **Step 3: Write the implementation**

In `app/chatbot/orchestrator.py`, this file uses named imports from `config`
(`from config import CURRENT_FORECAST_PATH, REGIONS`), not `import config` —
match that style. Change that existing line to:

```python
from config import CURRENT_FORECAST_PATH, REGIONS, PHOTO_MODEL_PATH
```

Then add to the imports, after the existing `from chatbot.vectorstore import retrieve` line:

```python
from pipeline.photo_diagnosis.model import GapingClassifier, apply_confidence_fallback
```

Then append this function at the end of the file:

```python


def _format_photo_evidence(result: dict) -> str:
    if result["fallback"]:
        return (
            "PHOTO ANALYSIS: The model could not confidently classify this "
            "photo (low confidence). Tell the farmer you couldn't get a "
            "clear read and suggest a closer, well-lit photo, focused "
            "directly on the shell."
        )
    return (
        f"PHOTO ANALYSIS: The shellfish in this photo was classified as "
        f"'{result['label']}' (confidence {result['confidence']:.2f}). "
        f"A 'gaping' shell that won't close is a possible distress or "
        f"mortality sign. This is a screening signal, not a diagnosis — "
        f"advise the farmer accordingly and recommend contacting their "
        f"local CMFRI extension officer if they see this."
    )


def diagnose_photo(image_path: str | Path) -> dict:
    """Classify a farmer-submitted shellfish photo and generate a
    farmer-facing answer, mirroring answer()'s evidence-then-LLM pattern."""
    clf = GapingClassifier.load(PHOTO_MODEL_PATH)
    raw = clf.predict(image_path)
    result = apply_confidence_fallback(raw)

    evidence_parts = [_format_photo_evidence(result)]
    hits = retrieve("shellfish gaping shell distress symptoms mortality", k=2)
    if hits:
        evidence_parts.append(_format_rag_evidence(hits))
    evidence_text = "\n\n".join(evidence_parts)

    prompt = (
        f"A farmer submitted a photo of their shellfish for a health check.\n\n"
        f"{evidence_text}\n\n"
        f"Write a short, farmer-facing response explaining what this means "
        f"and what to do next."
    )
    answer_text = chat(prompt, system=SYSTEM_PROMPT)

    return {
        "answer": answer_text,
        "label": result["label"],
        "confidence": result["confidence"],
        "fallback": result["fallback"],
        "route": "photo_diagnosis",
    }
```

This relies on `PHOTO_MODEL_PATH` now being imported at the top of
`orchestrator.py` (added above) and on `_format_rag_evidence`, which
already exists in this file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k diagnose_photo`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/chatbot/orchestrator.py app/tests/test_photo_diagnosis.py
git commit -m "Wire photo diagnosis into the chatbot orchestrator"
```

---

## Task 8: API endpoint

**Files:**
- Modify: `app/api/main.py`
- Test: `app/tests/test_photo_diagnosis.py` (append)

**Interfaces:**
- Consumes: `chatbot.orchestrator.diagnose_photo`
- Produces: `POST /diagnose-photo` (multipart file upload) → JSON `{"answer": str, "label": str, "confidence": float, "fallback": bool}`

- [ ] **Step 1: Write the failing test**

Append to `app/tests/test_photo_diagnosis.py`:

```python
def test_diagnose_photo_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api_main

    def fake_diagnose_photo(image_path):
        return {"answer": "Looks fine.", "label": "closed", "confidence": 0.8, "fallback": False, "route": "photo_diagnosis"}
    monkeypatch.setattr(api_main, "diagnose_photo_orchestrated", fake_diagnose_photo)

    client = TestClient(api_main.app)
    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    with open(img_path, "rb") as f:
        resp = client.post("/diagnose-photo", files={"file": ("test.jpg", f, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "closed"
    assert body["answer"] == "Looks fine."


def test_diagnose_photo_endpoint_rejects_non_image(tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api_main

    client = TestClient(api_main.app)
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("hello")

    with open(bad_path, "rb") as f:
        resp = client.post("/diagnose-photo", files={"file": ("notes.txt", f, "text/plain")})

    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k diagnose_photo_endpoint`
Expected: FAIL with 404 (endpoint doesn't exist yet)

- [ ] **Step 3: Write the implementation**

In `app/api/main.py`, update the import line (line 23) from:

```python
from chatbot.orchestrator import answer as orchestrate
```

to:

```python
from chatbot.orchestrator import answer as orchestrate
from chatbot.orchestrator import diagnose_photo as diagnose_photo_orchestrated
```

Replace the existing `fastapi` import line (line 17):

```python
from fastapi import FastAPI, HTTPException, Query
```

with:

```python
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
```

Add this endpoint after the existing `chat_endpoint` function (after line 147):

```python


class PhotoDiagnosisResponse(BaseModel):
    answer: str
    label: str
    confidence: float
    fallback: bool


@app.post("/diagnose-photo", response_model=PhotoDiagnosisResponse)
def diagnose_photo_endpoint(file: UploadFile = File(...)) -> PhotoDiagnosisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    import tempfile
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        r = diagnose_photo_orchestrated(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Photo diagnosis error: {type(e).__name__}: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PhotoDiagnosisResponse(
        answer=r["answer"], label=r["label"], confidence=r["confidence"], fallback=r["fallback"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd app && pytest tests/test_photo_diagnosis.py -v -k diagnose_photo_endpoint`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/api/main.py app/tests/test_photo_diagnosis.py
git commit -m "Add POST /diagnose-photo API endpoint"
```

---

## Task 9: Dependencies and full test suite

**Files:**
- Modify: `app/requirements.txt`

- [ ] **Step 1: Add new dependencies**

In `app/requirements.txt`, add after the `joblib>=1.3` line:

```
torch>=2.0
torchvision>=0.15
pillow>=10.0
```

- [ ] **Step 2: Install and run the full suite**

Run: `cd app && pip install -r requirements.txt && pytest tests/ -v`
Expected: All tests pass, including every `test_photo_diagnosis.py` test from Tasks 1–8 and all pre-existing tests (`test_features.py`, `test_chatbot.py`, `test_pipeline_e2e.py`).

- [ ] **Step 3: Commit**

```bash
git add app/requirements.txt
git commit -m "Add torch/torchvision/pillow dependencies for photo diagnosis"
```

---

## What this plan does not cover (by design — see spec's Future Work)

- Actually collecting and labeling real photos — that's outreach/logistics for Avni and Maya to run in parallel with this implementation, using the `label_photos.py` tool built in Task 5.
- Running `train.py` on real data and writing the resulting `LIMITATIONS.md` disclosure (exact N, sources, untested conditions) — do this once real labeled data exists; the script and its honesty-focused output message (Task 6, Step 3) are what generates the numbers to write down.
- **K-fold cross-validation.** The spec allows for k-fold "if enough sources exist"; `train.py` (Task 6) implements only the always-valid single held-out-source split, since the real number of distinct sources isn't known until data collection happens. If three or more real sources end up available, extending `source_split`/`bootstrap_ci` to a proper k-fold loop is a small follow-up, not a redesign — flag it then rather than guessing source counts now.
- **The qualitative end-to-end test round** (a handful of real photos run through the deployed feature, mirroring the five canonical farmer-question tests already done for the text chatbot) needs real photos and a real trained model — do this after training, not as part of this plan's unit tests.
- Frontend UI for photo upload (camera capture button, result display) — this plan is API-only; a follow-up plan covers the Lovable-side UI once the API is deployed and verified.
- Mantle discoloration, water-discoloration classification, and a validated classifier+API ensemble — explicitly deferred in the spec.
