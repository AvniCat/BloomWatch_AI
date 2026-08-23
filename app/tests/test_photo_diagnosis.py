"""Tests for the shellfish photo-diagnosis feature."""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


def test_photo_config_paths_exist():
    import config
    assert isinstance(config.PHOTO_MODEL_PATH, Path)
    assert isinstance(config.PHOTO_MODEL_META_PATH, Path)
    assert isinstance(config.PHOTO_MANIFEST_PATH, Path)
    assert 0.0 < config.GAPING_CONFIDENCE_THRESHOLD < 1.0


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


def test_vision_label_is_callable_and_mockable(monkeypatch):
    from chatbot import llm

    def fake_gemini_vision(image_path, prompt):
        return "gaping"

    monkeypatch.setattr(llm, "_gemini_vision", fake_gemini_vision)
    result = llm.vision_label("fake/path.jpg", "Is this shell open or closed?")
    assert result == "gaping"


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


def test_train_and_evaluate_is_deterministic_for_fixed_seed(tmp_path):
    """Same seed, same data -> same accuracy. Guards against unseeded model
    init / augmentation making the reported accuracy non-reproducible."""
    from pipeline.photo_diagnosis.train import train_and_evaluate

    manifest_path = _write_synthetic_manifest(tmp_path)

    metrics_1 = train_and_evaluate(manifest_path, tmp_path / "out1", seed=7)
    metrics_2 = train_and_evaluate(manifest_path, tmp_path / "out2", seed=7)

    assert metrics_1["accuracy"] == metrics_2["accuracy"]


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
