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
