"""Non-LLM tests for the chatbot layer.

We don't run the actual LLM in unit tests (it's slow + non-deterministic).
Instead we test the routing, evidence assembly, and RAG retrieval logic.
End-to-end LLM tests live in scripts/demo_chatbot.py for manual runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.orchestrator import _route, _infer_region, _load_current_forecast
from chatbot.rag_docs import build_corpus
from chatbot.vectorstore import build_index, retrieve


def test_route_forecast():
    assert _route("What's the bloom risk this week?") == "forecast"
    assert _route("Should I harvest now?") == "forecast"
    assert _route("How reliable is this forecast?") == "forecast"


def test_route_history():
    assert _route("When was the last bloom near Kochi?") == "history"
    assert _route("Have there been any documented events?") == "history"


def test_infer_region():
    assert _infer_region("I farm near Kochi") == "Kerala"
    assert _infer_region("what about Udupi?") == "Karnataka"
    assert _infer_region("general question about blooms") is None


def test_current_forecast_loadable():
    fc = _load_current_forecast()
    assert fc is not None
    assert "regions" in fc
    for r in ("Kerala", "Karnataka"):
        assert r in fc["regions"]
        assert 0.0 <= fc["regions"][r]["P_bloom_next_week"] <= 1.0


def test_rag_retrieval():
    build_index(rebuild=True)
    hits = retrieve("what does 40 percent probability mean?", k=3)
    assert len(hits) >= 1
    # Must retrieve either forecast_interpretation or risk_bands
    topics = {h["meta"].get("topic") for h in hits}
    assert topics & {"forecast_interpretation", "risk_bands"}


def test_rag_history_retrieval():
    hits = retrieve("past bloom events near Kochi", k=3, region="Kerala")
    assert len(hits) >= 1
    assert any(h["meta"].get("topic") == "historical_event" for h in hits)
