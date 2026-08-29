"""Chatbot orchestrator — the "brain" that takes a farmer question and
answers it with real forecast data + retrieved knowledge base context.

Design: rather than relying on LLM tool-calling (which behaves differently
per provider), we do a simple, deterministic 3-step flow:

  1. Route the question — is it about the CURRENT forecast, HISTORY, or GENERAL?
     Uses cheap keyword heuristics with LLM fallback for ambiguous cases.
  2. Assemble evidence — fetch forecast JSON if needed + retrieve top-k RAG chunks.
  3. Answer — single LLM call with system prompt + evidence + question.

This keeps the system explainable and testable — every answer is traceable
to specific evidence sources.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CURRENT_FORECAST_PATH, REGIONS, PHOTO_MODEL_PATH
from chatbot.llm import chat
from chatbot.vectorstore import retrieve
# NOTE: pipeline.photo_diagnosis.model (torch/torchvision) is intentionally
# NOT imported at module level — see diagnose_photo() below, which imports
# it lazily so requests that never touch the photo feature (including API
# boot) don't pay for loading torch+torchvision. Matches this codebase's
# established lazy-heavy-import convention (chatbot/vectorstore.py's
# _client(), chatbot/llm.py's _gemini_chat(), the /forecast endpoint's
# in-handler pandas import).


SYSTEM_PROMPT = """You are the assistant inside BloomWatch AI, an early-warning app
for harmful algal blooms used by shellfish cooperatives on the Kerala and Karnataka
coasts of India. Your users are shellfish farmers.

Rules:
- Be direct, practical, and specific. Talk like a helpful field officer, not a chatbot.
- Ground every claim in the evidence provided. If the evidence doesn't cover the
  question, say so honestly rather than guessing.
- Give the concrete forecast number when it's in the evidence, and translate it
  into a risk band the farmer can act on.
- Keep answers under 6 sentences unless the question demands more.
- Never invent a bloom event, closure date, or CMFRI advisory. If it's not in the
  evidence, it doesn't exist for you.
- WRITE FOR A FARMER, NOT A MARINE BIOLOGIST: use short sentences and everyday
  words. Avoid technical jargon ("byssal threads", "mantle tint", "hypoxia",
  "filtration rate") — describe what to look for in plain, observable terms
  instead (e.g. "the shell is open and won't close" rather than "gaping", "the
  animal is limp or doesn't react when touched" rather than "reduced
  responsiveness"). If a scientific word is unavoidable, explain it in the same
  sentence in plain language rather than assuming the reader knows it.
- HONESTY OVER REASSURANCE: If a LIMITATIONS block appears in the evidence, you
  MUST reflect those limitations in your answer. In particular, do not claim the
  model can detect Trichodesmium, does not have a blind spot, or covers regions
  outside Kerala/Karnataka if the limitations doc says otherwise. False
  reassurance about capability is worse than admitting a gap.
- Detect the farmer's language from the QUESTION. If the question is in English,
  answer in English. If it is in Hindi, answer in Hindi. If it is in Malayalam,
  answer in Malayalam. If it is in Kannada, answer in Kannada. Do not switch
  languages without being asked.
"""


REGION_KWS = {
    "Kerala":    ["kerala", "kochi", "cochin", "alappuzha", "kollam", "kannur",
                  "kozhikode", "trivandrum", "thiruvananthapuram", "malappuram",
                  "thrissur", "kasargod", "kottayam"],
    "Karnataka": ["karnataka", "mangalore", "mangaluru", "udupi", "karwar", "malpe",
                  "iddya", "surathkal", "kannada"],
}

FORECAST_KWS = ["risk", "chance", "bloom next", "next week", "this week", "should i harvest",
                "should i", "forecast", "predict", "probability", "trust", "reliable"]
HISTORY_KWS  = ["last bloom", "past bloom", "history", "before", "previous", "when was",
                "documented", "recorded", "cmfri"]


def _infer_region(text: str) -> str | None:
    t = text.lower()
    for region, kws in REGION_KWS.items():
        if any(k in t for k in kws):
            return region
    return None


def _route(question: str) -> Literal["forecast", "history", "general"]:
    q = question.lower()
    if any(k in q for k in FORECAST_KWS): return "forecast"
    if any(k in q for k in HISTORY_KWS):  return "history"
    return "general"


def _load_current_forecast() -> dict | None:
    if not CURRENT_FORECAST_PATH.exists(): return None
    return json.loads(CURRENT_FORECAST_PATH.read_text())


def _format_forecast_evidence(fc: dict, region: str | None) -> str:
    lines = [
        f"CURRENT FORECAST (window {fc['window_start']} → {fc['window_end']}):",
    ]
    for r, data in fc["regions"].items():
        if region and r != region: continue
        p = data["P_bloom_next_week"]
        band = data["risk_band"]
        lines.append(f"  {r}: {p*100:.1f}% chance of bloom next 8-day window ({band})")
    lines.append(
        f"Model version: {fc.get('model_version','?')}. "
        f"Computed at: {fc.get('computed_at','?')}."
    )
    return "\n".join(lines)


def _format_rag_evidence(hits: list[dict]) -> str:
    if not hits: return ""
    lines = ["KNOWLEDGE BASE EVIDENCE:"]
    for i, h in enumerate(hits, 1):
        topic = h["meta"].get("topic", "?")
        lines.append(f"[{i}] ({topic}) {h['text']}")
    return "\n".join(lines)


def answer(question: str, region_hint: str | None = None) -> dict:
    """Take a farmer question, return a structured answer.

    Returns:
      {answer: str, route: str, region: str|None, evidence: {...}}
    """
    route = _route(question)
    region = region_hint or _infer_region(question)

    evidence_parts: list[str] = []
    evidence_meta: dict = {"route": route, "region": region}

    # Include the current forecast ONLY when the question is genuinely asking
    # about it. Previously we injected the forecast for every "general" question
    # which padded mitigation/diagnosis answers with irrelevant risk numbers.
    _FORECAST_TRIGGERS = [
        "risk", "chance", "probability", "forecast", "predict", "predicted",
        "this week", "next week", "how bad", "how high", "trust this number",
        "reliable", "current", "right now", "today",
    ]
    q_lower = question.lower()
    forecast_wanted = (
        route == "forecast"
        or any(t in q_lower for t in _FORECAST_TRIGGERS)
    )
    fc = _load_current_forecast()
    if fc and forecast_wanted:
        evidence_parts.append(_format_forecast_evidence(fc, region))
        evidence_meta["forecast_included"] = True
    else:
        evidence_meta["forecast_included"] = False

    # RAG retrieval — filter by region if it's a history question about a specific area
    filter_region = region if route == "history" else None
    hits = retrieve(question, k=4, region=filter_region)

    # Force-include the limitations doc for capability / species / coverage questions.
    # These are exactly the questions where the model has historically over-claimed
    # (e.g. saying it can detect Trichodesmium when it cannot).
    _CAPABILITY_TRIGGERS = [
        "can your model", "can the model", "does your model", "does the model",
        "can bloomwatch", "does bloomwatch", "detect", "trichodesmium",
        "noctiluca", "alexandrium", "cyanobacter", "species", "which blooms",
        "what blooms", "blind spot", "limitation", "accurate", "accuracy",
        "reliable", "coverage", "cover", "chennai", "tamil nadu",
        "andhra", "goa", "gujarat", "mumbai", "chennai", "outside",
    ]
    if any(t in q_lower for t in _CAPABILITY_TRIGGERS):
        try:
            limits_hits = retrieve("model limitations trichodesmium blind spot", k=2)
            existing_ids = {h["id"] for h in hits}
            for lh in limits_hits:
                if lh["id"] not in existing_ids and lh["meta"].get("topic") == "limitations":
                    hits.insert(0, lh)  # put limitations at the top
                    break
        except Exception:
            pass

    if hits:
        evidence_parts.append(_format_rag_evidence(hits))
        evidence_meta["rag_hits"] = [
            {"topic": h["meta"].get("topic"), "id": h["id"], "distance": round(h["distance"], 3)}
            for h in hits
        ]

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(No evidence available.)"

    # Detect question language script (crude but works for English vs Malayalam/Kannada)
    if any("ഀ" <= c <= "ൿ" for c in question):
        lang_directive = "Answer in Malayalam."
    elif any("ಀ" <= c <= "೿" for c in question):
        lang_directive = "Answer in Kannada."
    else:
        lang_directive = "Answer in English."

    prompt = (
        f"FARMER QUESTION: {question}\n\n"
        f"{evidence_text}\n\n"
        f"Answer the farmer using only the evidence above. If evidence is missing "
        f"for a required fact, say so explicitly rather than guessing. "
        f"{lang_directive}"
    )

    reply = chat(prompt, system=SYSTEM_PROMPT)
    return {
        "answer": reply.strip(),
        "route": route,
        "region": region,
        "evidence": evidence_meta,
    }


def _format_photo_evidence(result: dict, band: str | None) -> str:
    if result["fallback"]:
        return (
            "PHOTO ANALYSIS: The model could not confidently classify this "
            "photo (low confidence). Tell the farmer you couldn't get a "
            "clear read and suggest a closer, well-lit photo, focused "
            "directly on the shell."
        )
    # Hedged band only — never the raw confidence number. When fallback is
    # True, `result['confidence']` is the score of the label that was just
    # discarded (e.g. a suppressed "gaping" call under "unclear"), so it
    # must never be surfaced here — the branch above returns before this
    # line runs in that case.
    return (
        f"PHOTO ANALYSIS: The shellfish in this photo was classified as "
        f"'{result['label']}' (confidence: {band}). "
        f"A 'gaping' shell that won't close is a possible distress or "
        f"mortality sign. This is a screening signal, not a diagnosis — "
        f"advise the farmer accordingly and recommend contacting their "
        f"local CMFRI extension officer if they see this."
    )


_classifier_cache = None  # lazily-populated GapingClassifier; see diagnose_photo() below


def diagnose_photo(image_path: str | Path) -> dict:
    """Classify a farmer-submitted shellfish photo and generate a
    farmer-facing answer, mirroring answer()'s evidence-then-LLM pattern."""
    global _classifier_cache

    # Lazy import: pipeline.photo_diagnosis.model pulls in torch/torchvision,
    # which we don't want to pay for at API boot for requests that never
    # touch this feature (see the NOTE near the top-of-file imports).
    from pipeline.photo_diagnosis.model import apply_confidence_fallback, confidence_band, GapingClassifier

    # Cache the loaded classifier across requests — GapingClassifier.load()
    # reads the checkpoint from disk every call otherwise, and (before the
    # pretrained=False fix in model.py) used to re-download ImageNet weights
    # on every single request.
    if _classifier_cache is None:
        _classifier_cache = GapingClassifier.load(PHOTO_MODEL_PATH)
    clf = _classifier_cache

    raw = clf.predict(image_path)
    result = apply_confidence_fallback(raw)
    band = None if result["fallback"] else confidence_band(result)

    evidence_parts = [_format_photo_evidence(result, band)]
    hits = retrieve("shellfish gaping shell distress symptoms mortality", k=2)
    if hits:
        evidence_parts.append(_format_rag_evidence(hits))
    evidence_text = "\n\n".join(evidence_parts)

    prompt = (
        f"A farmer submitted a photo of their shellfish for a health check.\n\n"
        f"{evidence_text}\n\n"
        f"Write a short, farmer-facing response explaining what this means "
        f"and what to do next. Do not tell the farmer whether the catch is "
        f"safe to harvest or not — that is a food-safety call this app "
        f"never makes. Describe what was seen and recommend contacting the "
        f"CMFRI extension officer for anything concerning."
    )
    answer_text = chat(prompt, system=SYSTEM_PROMPT)

    return {
        "answer": answer_text,
        "label": result["label"],
        "confidence": result["confidence"],
        "confidence_band": "N/A (unclear)" if result["fallback"] else band,
        "fallback": result["fallback"],
        "route": "photo_diagnosis",
    }


if __name__ == "__main__":
    canonical = [
        ("What's the bloom risk this week for Kerala?", None),
        ("Should I harvest my mussels now, off Kochi?", "Kerala"),
        ("What should I do if a bloom is coming?", "Kerala"),
        ("When was the last documented bloom near Kochi?", "Kerala"),
        ("How much can I trust this number?", None),
    ]
    for q, hint in canonical:
        print("=" * 70)
        print(f"Q: {q}  (region_hint={hint})")
        r = answer(q, region_hint=hint)
        print(f"[route={r['route']} region={r['region']}]")
        print(f"A: {r['answer']}")
        print()
