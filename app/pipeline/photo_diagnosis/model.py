"""Gaping-shell binary classifier — transfer learning on a frozen MobileNetV2
backbone. Small dataset (dozens-to-low-hundreds of images), so we fine-tune
only the final classification head, not the whole network. See
docs/superpowers/specs/2026-08-18-shellfish-photo-diagnosis-design.md for
why transfer learning was chosen over training from scratch.
"""
from __future__ import annotations
import json
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

# Inference-time transform. Deliberately the standard companion of
# train.py's RandomResizedCrop(224, scale=(0.8, 1.0)): Resize(256) + a
# center 224 crop preserves aspect ratio, matching what the model was
# trained on. A plain Resize((224, 224)) here would squash non-square
# farmer phone photos into a square and distort geometry the model never
# saw during training (train/serve skew).
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with a fresh 2-class head. Backbone frozen — only the
    head is meant to be trained (see train.py).

    pretrained=False skips downloading ImageNet weights — use this when the
    caller is about to immediately overwrite every weight via
    load_state_dict() anyway (see GapingClassifier.load below), so we don't
    hit download.pytorch.org on every model load.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
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
        path = Path(path)
        # pretrained=False: we're about to overwrite every weight with the
        # checkpoint below, so there's no reason to fetch ImageNet weights
        # over the network on every request (see build_model's docstring).
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu"))

        # If a metadata sidecar exists, verify LABELS hasn't drifted since
        # this checkpoint was trained — a silent reorder would invert every
        # prediction with no error otherwise. Older checkpoints saved before
        # this check existed have no sidecar; we don't make it mandatory for
        # those, we just skip the check.
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("labels") != LABELS:
                raise ValueError(
                    f"Model checkpoint at {path} was trained with label "
                    f"order {meta.get('labels')!r}, but the code currently "
                    f"defines LABELS={LABELS!r}. Loading it would silently "
                    f"invert every prediction — retrain the model or fix "
                    f"LABELS before using this checkpoint."
                )

        return cls(model)


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


def confidence_band(result: dict) -> str:
    """Map a raw confidence float to a qualitative hedged band — the spec
    requires the classifier's output reach the farmer as a hedged band,
    never a raw unqualified percentage (mirrors pipeline/predict.py's
    qualitative_risk() banding pattern for the forecast model).

    Only meaningful when result['fallback'] is False — a fallback result's
    confidence score belongs to the label that was just discarded, so
    callers should not band it (see orchestrator._format_photo_evidence).
    """
    c = result["confidence"]
    if c < 0.75:
        return "Low"
    if c < 0.9:
        return "Moderate"
    return "High"
