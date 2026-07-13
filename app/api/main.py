"""BloomWatch API — FastAPI service exposing the forecast + chatbot.

Endpoints:
  GET  /health                 — liveness probe
  GET  /forecast?region=Kerala — current + recent-history forecast
  POST /chat                   — {question, region?} → answer
  GET  /providers              — LLM provider status (diagnostics)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CURRENT_FORECAST_PATH, HISTORICAL_CSV, REGIONS
from chatbot.orchestrator import answer as orchestrate
from chatbot.llm import provider_status
from chatbot.vectorstore import build_index, CHROMA_DIR


app = FastAPI(
    title="BloomWatch AI",
    description="Live HAB early-warning API for Kerala + Karnataka shellfish cooperatives",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_vector_index():
    """Build the RAG vector store if missing (first launch on a fresh host)."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        try:
            build_index(rebuild=False)
        except Exception as e:
            print(f"WARN: could not build vector index at startup: {e}")


# ---------- schemas ----------
class ChatRequest(BaseModel):
    question: str
    region: Optional[str] = None   # 'Kerala' | 'Karnataka' | None


class ChatResponse(BaseModel):
    answer: str
    route: str
    region: Optional[str] = None
    evidence: dict


# ---------- endpoints ----------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast_available": CURRENT_FORECAST_PATH.exists(),
    }


@app.get("/providers")
def providers() -> dict:
    return provider_status()


@app.get("/forecast")
def forecast(region: Optional[str] = Query(default=None, description="Kerala | Karnataka")) -> dict:
    if not CURRENT_FORECAST_PATH.exists():
        raise HTTPException(status_code=503, detail="No forecast yet. Run refresh_weekly.py first.")
    fc = json.loads(CURRENT_FORECAST_PATH.read_text())

    # Optionally filter to one region
    if region:
        if region not in REGIONS:
            raise HTTPException(status_code=400, detail=f"Unknown region {region}. Use one of {list(REGIONS)}.")
        fc = {**fc, "regions": {region: fc["regions"][region]}}

    # Attach last 4 weeks of historical context (chlor-a mean per region)
    try:
        import pandas as pd
        df = pd.read_csv(HISTORICAL_CSV, parse_dates=["date_start"])
        df = df.sort_values("date_start")
        history = {}
        for r in fc["regions"]:
            recent = df[df.region == r].tail(4)[["date_start", "chlor_a_mean", "sst_mean", "bloom"]]
            history[r] = [{
                "date": row["date_start"].date().isoformat(),
                "chlor_a_mean": None if pd.isna(row["chlor_a_mean"]) else float(round(row["chlor_a_mean"], 3)),
                "sst_mean":     None if pd.isna(row["sst_mean"])     else float(round(row["sst_mean"], 2)),
                "bloom":        int(row["bloom"]),
            } for _, row in recent.iterrows()]
        fc["recent_history"] = history
    except Exception as e:
        fc["recent_history_error"] = str(e)

    return fc


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    if req.region and req.region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region {req.region}")
    try:
        r = orchestrate(req.question, region_hint=req.region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {type(e).__name__}: {e}")
    return ChatResponse(**r)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
