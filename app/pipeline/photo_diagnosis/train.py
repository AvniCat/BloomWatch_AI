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
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from pipeline.photo_diagnosis.model import GapingClassifier, LABELS, apply_confidence_fallback
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
    # Load the raw PIL images once (cheap disk I/O), but re-apply
    # _TRAIN_AUGMENT fresh inside the epoch loop below. Applying it once up
    # front and reusing the same fixed tensors for every epoch would defeat
    # the entire point of augmentation — the small-dataset-stretching
    # strategy described in the module docstring above only works if every
    # epoch actually sees a newly-jittered/cropped/rotated version of each
    # image, not the same augmented tensor `epochs` times over.
    pil_images = [Image.open(r["filepath"]).convert("RGB") for r in rows]
    targets = [label_to_idx[r["human_label"].strip().lower()] for r in rows]
    y = torch.tensor(targets, dtype=torch.long)

    for _ in range(epochs):
        x = torch.stack([_TRAIN_AUGMENT(img) for img in pil_images])
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _precision(rows: list[dict]) -> float:
    """Precision for the 'gaping' positive class, over non-abstained rows.
    rows must have 'predicted_label' and 'true_label'. Returns 0.0 if the
    resample contains no positive predictions (avoids ZeroDivisionError
    during bootstrap resampling; matches sklearn's zero_division=0 default)."""
    predicted_positive = [r for r in rows if r["predicted_label"] == "gaping"]
    if not predicted_positive:
        return 0.0
    tp = sum(1 for r in predicted_positive if r["true_label"] == "gaping")
    return tp / len(predicted_positive)


def _recall(rows: list[dict]) -> float:
    """Recall for the 'gaping' positive class, over non-abstained rows."""
    actual_positive = [r for r in rows if r["true_label"] == "gaping"]
    if not actual_positive:
        return 0.0
    tp = sum(1 for r in actual_positive if r["predicted_label"] == "gaping")
    return tp / len(actual_positive)


_NULL_CI = {"mean": None, "lo": None, "hi": None}


def train_and_evaluate(manifest_path: Path, out_dir: Path, seed: int = 7) -> dict:
    rows = load_labeled_manifest(manifest_path)
    train_rows, test_rows = source_split(rows, seed=seed)

    model = _train_head(train_rows, seed=seed)
    clf = GapingClassifier(model)

    # Evaluate through the SAME post-fallback pipeline production actually
    # runs (orchestrator.diagnose_photo applies apply_confidence_fallback on
    # top of predict()) — scoring raw clf.predict() alone measures a
    # classifier nobody runs, since low-confidence calls get converted to
    # "unclear" in production before a farmer ever sees them.
    for r in test_rows:
        raw = clf.predict(r["filepath"])
        result = apply_confidence_fallback(raw)
        r["true_label"] = r["human_label"].strip().lower()
        r["predicted_label"] = result["label"]
        r["fallback"] = result["fallback"]
        r["correct"] = int((not result["fallback"]) and result["label"] == r["true_label"])

    # Abstention rate is computed over ALL test predictions (including the
    # ones that abstained) — it's a separate, honestly-reported number, not
    # folded into accuracy.
    abstention_rate = bootstrap_ci(test_rows, lambda rs: sum(r["fallback"] for r in rs) / len(rs), seed=seed)

    # Accuracy/precision/recall are computed only among non-abstained
    # predictions — an abstention is neither "correct" nor "incorrect",
    # so it must not silently deflate (or inflate) the accuracy number.
    non_abstained = [r for r in test_rows if not r["fallback"]]
    if non_abstained:
        accuracy = bootstrap_ci(non_abstained, lambda rs: sum(r["correct"] for r in rs) / len(rs), seed=seed)
        precision = bootstrap_ci(non_abstained, _precision, seed=seed)
        recall = bootstrap_ci(non_abstained, _recall, seed=seed)
    else:
        accuracy = dict(_NULL_CI)
        precision = dict(_NULL_CI)
        recall = dict(_NULL_CI)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save(out_dir / "gaping_classifier.pt")

    # Metadata sidecar — written next to the checkpoint using the same
    # gaping_classifier.pt -> gaping_classifier.meta.json naming convention
    # as config.PHOTO_MODEL_PATH / config.PHOTO_MODEL_META_PATH (so this
    # lines up with config.PHOTO_MODEL_META_PATH exactly when out_dir is the
    # production PHOTO_MODEL_DIR, while staying test-safe for callers that
    # pass a tmp out_dir). GapingClassifier.load() reads this to catch a
    # LABELS reorder before it silently inverts every prediction.
    meta = {
        "labels": LABELS,
        "model_version": config.PHOTO_MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "gaping_classifier.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    metrics = {
        "model_version": config.PHOTO_MODEL_VERSION,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_non_abstained": len(non_abstained),
        "test_sources": sorted({r["source"] for r in test_rows}),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "abstention_rate": abstention_rate,
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

    def _fmt_ci(ci: dict) -> str:
        if ci["mean"] is None:
            return "n/a (no non-abstained test predictions)"
        return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"

    print(
        f"\nTrained on {metrics['n_train']} photos, tested on "
        f"{metrics['n_test']} held out from source(s) {metrics['test_sources']} "
        f"({metrics['n_non_abstained']} scored, rest abstained).\n"
        f"Accuracy:        {_fmt_ci(metrics['accuracy'])}\n"
        f"Precision:       {_fmt_ci(metrics['precision'])}\n"
        f"Recall:          {_fmt_ci(metrics['recall'])}\n"
        f"Abstention rate: {_fmt_ci(metrics['abstention_rate'])}\n"
        "Add these numbers, the exact N, and any untested conditions to "
        "LIMITATIONS.md before citing this model anywhere."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
