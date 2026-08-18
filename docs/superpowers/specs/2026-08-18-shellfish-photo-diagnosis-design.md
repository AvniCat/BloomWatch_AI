# Shellfish photo-diagnosis feature — design

**Status:** Approved, ready for implementation planning
**Owners:** Avni Singh, Maya Candade
**Constraint:** Fully built and validated within 2–4 weeks (IRIS submission timeline)

## Problem

Farmers currently get bloom-risk forecasts and can chat with BloomWatch AI in
text, but have no way to get a read on a specific animal or catch they're
looking at right now. They want to take a photo of their harvest and get an
assessment of what's wrong, the same way they can already ask the chatbot a
text question.

## Scope

**In scope:** A photo of a farmed bivalve (oyster, mussel, or clam — the
three species the app already supports) is classified as one of two states:

- **Gaping** — shell open and will not close, a visible distress/possible-
  mortality sign
- **Closed/normal** — no gaping observed

This is a single binary target, not a general diagnosis.

**Explicitly out of scope for this version:**
- Species identification (the farmer already knows what they photographed)
- Disease or toxin diagnosis, or any claim about *why* the shell is open
- Water/bloom discoloration classification (a plausible future target, not
  this one)
- Mantle discoloration (oyster-specific alternate signal — deferred; see
  Future Work)
- Any "safe to harvest" verdict — output is a distress signal, never a
  food-safety call
- A trained-classifier + vision-API ensemble that blends confidence scores
  into one number (rejected — see Rejected Approaches)

### Why gaping shell, specifically

Chosen over the alternatives (mantle discoloration, a broader "looks
abnormal" catch-all, water discoloration) because it uniquely satisfies four
constraints at once, given a small, multi-source, 2–4-week dataset:

1. **Cross-species.** Unlike mantle tint (oysters only) or byssal-thread
   condition (mussels only), gaping is roughly universal across bivalves —
   one model covers all three supported species instead of needing
   per-species models with even less data each.
2. **Photographable in a single still frame.** Reduced filtration and
   siphon withdrawal are behaviors observed over time, not visible in one
   photo. Gaping and mantle tint are the only two static, photographable
   states — and gaping is the cross-species one.
3. **Low label ambiguity.** Open/closed is close to objectively binary,
   which matters because labels will come from three different sources
   (market/cooperative/CMFRI contact, self-collected, Gemini-assisted
   draft labels). "Abnormal tint" is inherently more subjective and would
   introduce more inter-labeler disagreement — the worst kind of noise for
   a small dataset.
4. **Maps to the highest-stakes outcome.** Gaping correlates with death or
   severe distress, not a subtler sublethal marker — consistent with the
   chatbot's own existing guidance that mass mortality is "the clearest
   sign of a bloom impact."

## Data collection & labeling

Pursue in parallel, starting immediately (outreach lead time is the actual
critical path on this timeline):

1. **Market/cooperative/CMFRI contact** — real shellfish photos with
   expert-assisted labeling. Preferred source: strongest ground-truth
   quality.
2. **Self-collected** — photos taken and labeled by Avni and Maya as a
   supplementary source.

**Labeling protocol:** every photo is labeled `gaping` / `closed` and
tagged with `source` and `date`. To speed up the labeling bottleneck,
Gemini's vision API drafts a first-pass label on each photo, but a human
(whichever source supplied the photo) must confirm or correct it before it
counts as ground truth — the API accelerates labeling, it never sets
ground truth unsupervised.

**Realistic volume:** dozens to low hundreds of images per class, not
thousands. This is the reason transfer learning was chosen over training
from scratch (see Model).

**Honesty requirement:** the final `LIMITATIONS.md`-style disclosure must
state the exact N, the sources used, and conditions not tested (e.g. if all
photos end up daylight-only, that's a stated limitation, not silently
omitted) — same standard as the rest of the paper's rigor.

## Model

- **Architecture:** pretrained lightweight CNN backbone (MobileNetV2 or
  EfficientNet-B0), frozen except the final classification head (and
  optionally the last conv block, if data volume supports it).
- **Rationale:** with a dataset in the dozens-to-low-hundreds range,
  transfer learning generalizes far better than training a CNN from
  scratch — the image-classification analog of the tree-model-beats-deep-
  learning-on-small-data reasoning already in the paper (Grinsztajn et
  al., cited in Table 1).
- **Training:** standard image augmentation (rotation, brightness/contrast
  jitter, crop) to stretch the small dataset.
- **Output:** binary label + confidence, mapped to a hedged band (see
  Integration) — never a raw unqualified percentage, consistent with the
  app's existing risk-band pattern.

## Validation

- **Split by source/batch, not randomly** — same principle as the
  forecast model's temporal splits (never randomly shuffled). A held-out
  test set must include at least one source/batch the model never trained
  on, to test real generalization instead of memorizing one photographer's
  lighting or camera.
- **Metrics:** accuracy, precision, recall, reported with bootstrap
  confidence intervals — same rigor pattern as the existing power
  analysis. Given small N, wide CIs are expected and must be reported
  honestly, not hidden.
- **Cross-validation:** k-fold across available sources if enough sources
  exist by the time training happens; otherwise a single clean held-out
  source-based test set.

## Integration

- **New endpoint:** `POST /diagnose-photo`, alongside the existing
  `/health`, `/forecast`, `/model_status`, `/chat`.
- **Pipeline** (mirrors the orchestrator's existing deterministic
  3-step pattern, not LLM tool-calling):
  1. Classify the photo (gaping / closed / low-confidence-fallback).
  2. Retrieve relevant RAG evidence from the existing knowledge base
     (shellfish symptom guide, mitigation docs, emergency contacts).
  3. Generate the farmer-facing answer through the existing chatbot LLM
     call, using the same plain-language system prompt already shipped
     (no technical jargon, observable-terms only).
- The classifier's structured result (label + hedged confidence band)
  is evidence fed into step 3, not a raw label shown to the farmer
  directly — same "translate the number into something actionable"
  principle as the forecast risk bands.

## Error handling

- **Out-of-distribution or low-quality photos** (not a shellfish, too
  blurry, wrong lighting): a low-confidence threshold routes to a fallback
  response ("couldn't get a clear read from this photo") rather than
  forcing a possibly-wrong classification.
- **No harvest verdicts.** This feature never outputs "safe to harvest" or
  "do not harvest" — only a hedged distress signal plus a recommendation
  to consult the local CMFRI extension officer for anything concerning,
  identical in spirit to the existing forecast risk-band language.

## Testing

- **Unit tests** for the classifier wrapper, matching the existing
  `test_features.py` / `test_chatbot.py` pattern in `app/tests/`.
- **Held-out evaluation** with bootstrap CIs, as described in Validation.
- **Qualitative round:** a handful of real end-to-end photo tests, same
  spirit as the five canonical farmer questions already used to test the
  text chatbot.

## Rejected approaches

- **Zero-shot Gemini-only classification, no training** (Approach B):
  fastest to ship, but produces no real validated accuracy number and is
  scientifically thinner than everything else in this project. Rejected
  in favor of a trained-and-validated classifier.
- **Ensembling the classifier's calibrated confidence with Gemini's
  uncalibrated confidence into one fused accuracy number:** rejected on
  methodological grounds. Gemini's vision confidence has not been tested
  against ground truth the way the classifier's has; blending an
  unvalidated confidence into a validated one is the same mistake the
  paper's own outcome-anchored-calibration finding already warns against
  — an unproven adjustment diluting an honest raw signal. The API is used
  for labeling assistance and the advisory/explanation layer only, never
  blended into the accuracy claim.

## Future work (explicitly deferred, not this version)

- Mantle discoloration as a second, oyster-specific target (rejected for
  v1 due to species restriction and higher label subjectivity — see
  "Why gaping shell, specifically").
- Water/bloom discoloration classification (the other candidate diagnosis
  target from initial scoping).
- A genuine, rigorously-tested ensemble of classifier + vision API,
  evaluated as one system against the same held-out test set — technically
  valid, but deferred because it's unproven extra scope on an already
  tight timeline.
