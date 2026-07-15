"""LLM provider abstraction — Gemini primary, Ollama fallback + embeddings.

Design:
  - chat(prompt, system=None, ...) tries Gemini first if GEMINI_API_KEY is set
    and LLM_MODE != 'local'. Falls back to Ollama on any exception.
  - embed(texts) always uses Ollama (fast, local, unlimited).

This module is the ONLY place that talks to LLM APIs. Everything else
(orchestrator, RAG store, API layer) calls into these two functions.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # loads .env  # noqa: F401


# ---------- config ----------
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT     = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED    = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODE        = os.getenv("LLM_MODE", "primary").lower()   # primary | local


# ---------- Gemini (lazy import) ----------
def _gemini_chat(prompt: str, system: str | None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    resp = model.generate_content(prompt)
    return resp.text


# ---------- Ollama ----------
def _ollama_chat(prompt: str, system: str | None) -> str:
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat(model=OLLAMA_CHAT, messages=messages)
    return resp["message"]["content"]


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)
    out = []
    for t in texts:
        r = client.embeddings(model=OLLAMA_EMBED, prompt=t)
        out.append(r["embedding"])
    return out


def _gemini_embed(texts: list[str]) -> list[list[float]]:
    """Gemini embeddings — 768-dim, free tier, works on any hosted server."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    out = []
    for t in texts:
        r = genai.embed_content(model="models/gemini-embedding-001", content=t)
        out.append(r["embedding"])
    return out


# ---------- Public API ----------
class LLMError(RuntimeError):
    pass


def chat(prompt: str, system: str | None = None) -> str:
    """Route a chat prompt. Gemini primary, Ollama fallback."""
    tried = []
    if LLM_MODE != "local" and GEMINI_API_KEY:
        try:
            return _gemini_chat(prompt, system)
        except Exception as e:
            tried.append(f"Gemini: {type(e).__name__}: {e}")
    try:
        return _ollama_chat(prompt, system)
    except Exception as e:
        tried.append(f"Ollama: {type(e).__name__}: {e}")
    raise LLMError("All providers failed:\n  " + "\n  ".join(tried))


def embed(texts: Iterable[str]) -> list[list[float]]:
    """Embed a batch of texts. Prefers Ollama in local mode (fast, unlimited);
    prefers Gemini in primary mode (works on any hosted server)."""
    texts = list(texts)
    if not texts:
        return []
    tried = []
    # In local mode try Ollama first; in primary/hosted mode try Gemini first
    order = ([_ollama_embed, _gemini_embed] if LLM_MODE == "local"
             else [_gemini_embed, _ollama_embed])
    for fn in order:
        # Skip Gemini if key missing; skip Ollama if it seems unreachable
        if fn is _gemini_embed and not GEMINI_API_KEY:
            continue
        try:
            return fn(texts)
        except Exception as e:
            tried.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    raise LLMError("All embed providers failed:\n  " + "\n  ".join(tried))


def provider_status() -> dict:
    """Report which providers are configured, for diagnostics."""
    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL if GEMINI_API_KEY else None,
        "ollama_host": OLLAMA_HOST,
        "ollama_chat_model": OLLAMA_CHAT,
        "ollama_embed_model": OLLAMA_EMBED,
        "mode": LLM_MODE,
    }


if __name__ == "__main__":
    import json
    print("provider status:", json.dumps(provider_status(), indent=2))
    print("\ntesting chat…")
    try:
        answer = chat("In one sentence: what does BloomWatch AI forecast?")
        print("OK:", answer[:200])
    except LLMError as e:
        print("FAIL:", e)
    print("\ntesting embed…")
    try:
        v = embed(["harmful algal bloom forecast"])
        print(f"OK: {len(v[0])}-dim embedding")
    except LLMError as e:
        print("FAIL:", e)
