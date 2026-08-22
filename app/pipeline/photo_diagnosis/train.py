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


def _train_head(rows: list[dict], epochs: int = 5, lr: float = 1e-3, seed: int = 7) -> nn.Module:
    # Seeds both the fresh classification head's random init (build_model()'s
    # nn.Linear, via GapingClassifier() below) and _TRAIN_AUGMENT's stochastic
    # transforms, so accuracy is reproducible run-to-run rather than swinging
    # wildly (observed 0/6 to 6/6 on identical synthetic data when unseeded).
    # Same seed=7 convention already used by source_split/bootstrap_ci above.
    torch.manual_seed(seed)
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


def train_and_evaluate(manifest_path: Path, out_dir: Path, seed: int = 7) -> dict:
    rows = load_labeled_manifest(manifest_path)
    train_rows, test_rows = source_split(rows, seed=seed)

    model = _train_head(train_rows, seed=seed)
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
