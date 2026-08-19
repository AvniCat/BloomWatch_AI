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
