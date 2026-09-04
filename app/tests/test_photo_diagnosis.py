"""Tests for the shellfish photo-diagnosis feature."""
import json
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


def test_confidence_band_maps_to_qualitative_labels():
    from pipeline.photo_diagnosis.model import confidence_band

    assert confidence_band({"confidence": 0.61}) == "Low"
    assert confidence_band({"confidence": 0.8}) == "Moderate"
    assert confidence_band({"confidence": 0.95}) == "High"


def test_build_model_pretrained_false_skips_weights_download():
    from pipeline.photo_diagnosis.model import build_model

    model = build_model(pretrained=False)
    assert model is not None
    # Still has a 2-class head wired up, same as the pretrained path.
    assert model.classifier[1].out_features == 2


def test_classifier_load_checks_meta_labels_and_raises_on_mismatch(tmp_path):
    import json as _json
    from pipeline.photo_diagnosis.model import GapingClassifier

    clf = GapingClassifier()
    save_path = tmp_path / "model.pt"
    clf.save(save_path)

    meta_path = save_path.with_suffix(".meta.json")
    meta_path.write_text(_json.dumps({
        "labels": ["gaping", "closed"],  # deliberately reversed vs. real LABELS
        "model_version": "test",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }))

    import pytest
    with pytest.raises(ValueError, match="label"):
        GapingClassifier.load(save_path)


def test_classifier_load_proceeds_when_meta_file_absent(tmp_path):
    """Saved/loaded models from before this fix (no meta sidecar) must keep
    working — the meta check is opportunistic, not mandatory."""
    from pipeline.photo_diagnosis.model import GapingClassifier

    clf = GapingClassifier()
    save_path = tmp_path / "model.pt"
    clf.save(save_path)
    assert not save_path.with_suffix(".meta.json").exists()

    loaded = GapingClassifier.load(save_path)
    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)
    result = loaded.predict(img_path)
    assert result["label"] in ("gaping", "closed")


def test_classifier_load_matching_meta_labels_succeeds(tmp_path):
    import json as _json
    from pipeline.photo_diagnosis.model import GapingClassifier, LABELS

    clf = GapingClassifier()
    save_path = tmp_path / "model.pt"
    clf.save(save_path)

    meta_path = save_path.with_suffix(".meta.json")
    meta_path.write_text(_json.dumps({
        "labels": LABELS,
        "model_version": "test",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }))

    loaded = GapingClassifier.load(save_path)
    assert loaded is not None


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


def test_train_and_evaluate_reports_precision_recall_and_abstention(tmp_path):
    """Design spec's Validation section requires accuracy, precision, and
    recall, each with bootstrap CIs — not accuracy alone."""
    from pipeline.photo_diagnosis.train import train_and_evaluate

    manifest_path = _write_synthetic_manifest(tmp_path)
    out_dir = tmp_path / "out"
    metrics = train_and_evaluate(manifest_path, out_dir)

    for key in ("accuracy", "precision", "recall", "abstention_rate"):
        assert key in metrics
        assert set(metrics[key].keys()) == {"mean", "lo", "hi"}

    assert 0.0 <= metrics["abstention_rate"]["mean"] <= 1.0
    assert "n_non_abstained" in metrics
    assert metrics["n_non_abstained"] <= metrics["n_test"]

    with open(out_dir / "metrics.json") as f:
        written = json.load(f)
    assert written == metrics


def test_train_and_evaluate_evaluates_through_confidence_fallback(monkeypatch, tmp_path):
    """Core of finding #1: accuracy must be scored on the SAME pipeline
    production runs (predict() -> apply_confidence_fallback()), not raw
    predict() alone. Forcing every test prediction into fallback should
    zero out n_non_abstained and leave accuracy/precision/recall unscoreable
    (None), while abstention_rate reports 1.0 — none of that would happen if
    train.py were still scoring raw clf.predict() output directly."""
    from pipeline.photo_diagnosis import train as train_mod
    from pipeline.photo_diagnosis.model import GapingClassifier
    import config

    manifest_path = _write_synthetic_manifest(tmp_path)

    monkeypatch.setattr(
        GapingClassifier, "predict",
        lambda self, path: {"label": "gaping", "confidence": config.GAPING_CONFIDENCE_THRESHOLD - 0.05},
    )

    out_dir = tmp_path / "out"
    metrics = train_mod.train_and_evaluate(manifest_path, out_dir)

    assert metrics["abstention_rate"]["mean"] == 1.0
    assert metrics["n_non_abstained"] == 0
    assert metrics["accuracy"] == {"mean": None, "lo": None, "hi": None}
    assert metrics["precision"] == {"mean": None, "lo": None, "hi": None}
    assert metrics["recall"] == {"mean": None, "lo": None, "hi": None}


def test_train_and_evaluate_writes_photo_model_meta(tmp_path):
    """Finding #7: a meta sidecar with labels/model_version/trained_at must
    be written next to the checkpoint so GapingClassifier.load() can detect
    a LABELS reorder."""
    from pipeline.photo_diagnosis.train import train_and_evaluate
    from pipeline.photo_diagnosis.model import LABELS

    manifest_path = _write_synthetic_manifest(tmp_path)
    out_dir = tmp_path / "out"
    train_and_evaluate(manifest_path, out_dir)

    meta_path = out_dir / "gaping_classifier.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["labels"] == LABELS
    assert "model_version" in meta
    assert "trained_at" in meta

    # Loading the freshly-trained checkpoint should succeed cleanly since
    # its own meta file's labels match the current LABELS.
    from pipeline.photo_diagnosis.model import GapingClassifier
    loaded = GapingClassifier.load(out_dir / "gaping_classifier.pt")
    assert loaded is not None


def test_train_head_reapplies_augmentation_each_epoch(monkeypatch, tmp_path):
    """Finding #3: _TRAIN_AUGMENT must be re-applied fresh every epoch, not
    once up front and reused — otherwise every epoch trains on an identical
    tensor and the augmentation strategy described in the module docstring
    isn't actually happening."""
    from pipeline.photo_diagnosis import train as train_mod

    manifest_path = _write_synthetic_manifest(tmp_path, n_per_source=2)
    rows = train_mod.load_labeled_manifest(manifest_path)

    call_count = {"n": 0}
    real_augment = train_mod._TRAIN_AUGMENT

    def counting_augment(img):
        call_count["n"] += 1
        return real_augment(img)

    monkeypatch.setattr(train_mod, "_TRAIN_AUGMENT", counting_augment)

    epochs = 4
    train_mod._train_head(rows, epochs=epochs)

    # One augmentation call per image per epoch — proves each epoch draws a
    # fresh augmented tensor rather than reusing one computed before the loop.
    assert call_count["n"] == len(rows) * epochs


def test_diagnose_photo_assembles_evidence_and_calls_chat(monkeypatch, tmp_path):
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    monkeypatch.setattr(orchestrator, "_classifier_cache", None)  # isolate from other tests' cache

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
    assert result["confidence_band"] in ("Low", "Moderate", "High")
    assert "gaping" in captured["prompt"].lower()
    assert captured["system"] == orchestrator.SYSTEM_PROMPT


def test_diagnose_photo_low_confidence_routes_to_fallback(monkeypatch, tmp_path):
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    monkeypatch.setattr(orchestrator, "_classifier_cache", None)  # isolate from other tests' cache

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
    assert result["confidence_band"] == "N/A (unclear)"
    assert "unclear" in captured["prompt"].lower() or "couldn't" in captured["prompt"].lower()
    # The raw confidence (0.2) belonged to the discarded "gaping" call and
    # must never be surfaced as a number in the prompt sent to the LLM.
    assert "0.2" not in captured["prompt"]


def test_diagnose_photo_prompt_forbids_harvest_verdict(monkeypatch, tmp_path):
    """The spec's 'never a safe-to-harvest / do-not-harvest verdict'
    constraint must be an explicit instruction in the photo-diagnosis
    prompt, not something left to the shared SYSTEM_PROMPT (which is also
    used for general chat that does discuss harvest decisions)."""
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    monkeypatch.setattr(orchestrator, "_classifier_cache", None)

    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    class FakeClassifier:
        def predict(self, path):
            return {"label": "gaping", "confidence": 0.95}

    monkeypatch.setattr(photo_model.GapingClassifier, "load", classmethod(lambda cls, p: FakeClassifier()))

    captured = {}
    def fake_chat(prompt, system=None):
        captured["prompt"] = prompt
        return "Contact your local CMFRI extension officer."
    monkeypatch.setattr(orchestrator, "chat", fake_chat)
    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])

    orchestrator.diagnose_photo(img_path)

    assert "safe to harvest" in captured["prompt"].lower()


def test_diagnose_photo_caches_classifier_across_calls(monkeypatch, tmp_path):
    """The classifier should be loaded once and reused, not reloaded from
    disk on every request (see model.py's build_model(pretrained=False) for
    the other half of this fix — avoiding a re-download of ImageNet
    weights)."""
    from chatbot import orchestrator
    from pipeline.photo_diagnosis import model as photo_model

    monkeypatch.setattr(orchestrator, "_classifier_cache", None)

    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    load_calls = {"n": 0}

    class FakeClassifier:
        def predict(self, path):
            return {"label": "closed", "confidence": 0.95}

    def fake_load(cls, p):
        load_calls["n"] += 1
        return FakeClassifier()

    monkeypatch.setattr(photo_model.GapingClassifier, "load", classmethod(fake_load))
    monkeypatch.setattr(orchestrator, "chat", lambda prompt, system=None: "ok")
    monkeypatch.setattr(orchestrator, "retrieve", lambda *a, **k: [])

    orchestrator.diagnose_photo(img_path)
    orchestrator.diagnose_photo(img_path)

    assert load_calls["n"] == 1


def _touch_fake_model(tmp_path):
    """A stand-in trained-model file, just so the endpoint's exists() check
    passes — its contents are never actually read since diagnose_photo_
    orchestrated is mocked in these tests."""
    p = tmp_path / "gaping_classifier.pt"
    p.write_bytes(b"fake")
    return p


def test_diagnose_photo_endpoint(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api_main

    monkeypatch.setattr(api_main, "PHOTO_MODEL_PATH", _touch_fake_model(tmp_path))

    def fake_diagnose_photo(image_path):
        return {
            "answer": "Looks fine.", "label": "closed", "confidence": 0.8,
            "confidence_band": "High", "fallback": False, "route": "photo_diagnosis",
        }
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
    assert body["confidence_band"] == "High"


def test_diagnose_photo_endpoint_rejects_non_image(tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api_main

    client = TestClient(api_main.app)
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("hello")

    with open(bad_path, "rb") as f:
        resp = client.post("/diagnose-photo", files={"file": ("notes.txt", f, "text/plain")})

    assert resp.status_code == 400


def test_diagnose_photo_endpoint_returns_503_when_model_missing(monkeypatch, tmp_path):
    """Mirrors the existing /forecast endpoint's clean 503 for 'not ready
    yet' instead of an opaque 500 FileNotFoundError — no trained photo
    model exists on disk yet at this point in the project."""
    from fastapi.testclient import TestClient
    import api.main as api_main

    monkeypatch.setattr(api_main, "PHOTO_MODEL_PATH", tmp_path / "does_not_exist.pt")

    client = TestClient(api_main.app)
    img_path = tmp_path / "test.jpg"
    _make_test_image(img_path)

    with open(img_path, "rb") as f:
        resp = client.post("/diagnose-photo", files={"file": ("test.jpg", f, "image/jpeg")})

    assert resp.status_code == 503
    assert "train.py" in resp.json()["detail"]


def test_diagnose_photo_endpoint_rejects_oversized_upload(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api_main

    monkeypatch.setattr(api_main, "PHOTO_MODEL_PATH", _touch_fake_model(tmp_path))
    monkeypatch.setattr(api_main, "MAX_PHOTO_UPLOAD_BYTES", 1024)  # shrink cap so the test is fast

    def fake_diagnose_photo(image_path):
        raise AssertionError("diagnose_photo_orchestrated should not be called for an oversized upload")
    monkeypatch.setattr(api_main, "diagnose_photo_orchestrated", fake_diagnose_photo)

    client = TestClient(api_main.app)
    oversized = b"x" * (1024 * 2)

    resp = client.post(
        "/diagnose-photo",
        files={"file": ("big.jpg", oversized, "image/jpeg")},
    )

    assert resp.status_code == 413
