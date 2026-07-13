"""Fire the 5 canonical farmer questions at the chatbot. For manual demo runs.

Each question takes ~30-60s on local llama3.2:3b. Swap in Gemini for near-instant.
Run:  python scripts/demo_chatbot.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatbot.orchestrator import answer

CANONICAL = [
    ("What is the bloom risk this week for Kerala?", None),
    ("Should I harvest my mussels now off Kochi?", "Kerala"),
    ("What should I do if a bloom is coming?", "Kerala"),
    ("When was the last documented bloom near Kochi?", "Kerala"),
    ("How much can I trust this number?", None),
]

if __name__ == "__main__":
    for i, (q, hint) in enumerate(CANONICAL, 1):
        print("=" * 78)
        print(f"[{i}/{len(CANONICAL)}] Q: {q}   (region_hint={hint})")
        t0 = time.time()
        r = answer(q, region_hint=hint)
        dt = time.time() - t0
        print(f"    route={r['route']} | region={r['region']} | took {dt:.1f}s")
        print()
        print("A:", r["answer"])
        print()
