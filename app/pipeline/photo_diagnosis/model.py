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
