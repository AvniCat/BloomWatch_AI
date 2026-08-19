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
