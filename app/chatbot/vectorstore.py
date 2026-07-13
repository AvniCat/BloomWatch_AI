"""ChromaDB vector store — indexes the RAG corpus, retrieves top-k relevant docs.

Uses Ollama nomic-embed-text (768-dim, local, free).
Single-file SQLite persistence at data/chroma/.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from chatbot.llm import embed
from chatbot.rag_docs import build_corpus, chunk_text

CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "bloomwatch_kb"


def _client():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(rebuild: bool = False) -> int:
    """(Re)build the ChromaDB collection from the RAG corpus. Returns docs indexed."""
    client = _client()
    if rebuild:
        try: client.delete_collection(COLLECTION_NAME)
        except Exception: pass
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    corpus = build_corpus()
    ids, docs, metas = [], [], []
    for d in corpus:
        for i, chunk in enumerate(chunk_text(d["text"])):
            ids.append(f"{d['id']}::{i}")
            docs.append(chunk)
            meta = {k: v for k, v in d.items() if k != "text" and isinstance(v, (str, int, float, bool))}
            meta["chunk_index"] = i
            metas.append(meta)

    print(f"Embedding {len(docs)} chunks…")
    vecs = embed(docs)
    col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
    print(f"Indexed {len(docs)} chunks across {len(corpus)} source docs.")
    return len(docs)


def retrieve(query: str, k: int = 4, topic: str | None = None,
             region: str | None = None) -> list[dict]:
    """Return top-k relevant chunks. Optional filter by topic or region metadata."""
    client = _client()
    try:
        col = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        build_index()
        col = client.get_collection(name=COLLECTION_NAME)

    qvec = embed([query])[0]
    where = None
    if topic and region:
        where = {"$and": [{"topic": topic}, {"region": region}]}
    elif topic:
        where = {"topic": topic}
    elif region:
        where = {"region": region}

    r = col.query(query_embeddings=[qvec], n_results=k, where=where)
    return [
        {"text": d, "id": id_, "meta": m, "distance": dist}
        for d, id_, m, dist in zip(
            r["documents"][0], r["ids"][0], r["metadatas"][0], r["distances"][0]
        )
    ]


if __name__ == "__main__":
    n = build_index(rebuild=True)
    print(f"\nRetrieval sanity tests:")
    for q in [
        "should I harvest this week?",
        "what does 40 percent bloom probability mean?",
        "past bloom events near Kochi",
    ]:
        hits = retrieve(q, k=2)
        print(f"\n> {q}")
        for h in hits:
            print(f"  [{h['meta'].get('topic','?')}] {h['text'][:110]}…")
