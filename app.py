"""FastAPI demo for the Document RAG Agent.

Run (after installing requirements):
    uvicorn app:app --reload

Try:
    POST /ask
    {"question": "What is the refund policy?"}

This file is optional for the portfolio demo. The core logic lives in
rag.py and works with plain Python, no server needed.
"""

import os

from rag import ask, load_docs

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = object


if FastAPI is None:
    raise SystemExit("Install requirements first: pip install -r requirements.txt")

app = FastAPI(title="Document RAG Agent")

_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
_CHUNKS = load_docs(_DOCS_DIR)


class Question(BaseModel):
    question: str = ""
    top: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(_CHUNKS)}


@app.post("/ask")
def ask_question(body: Question):
    results, _ = ask(body.question, _CHUNKS, top_n=body.top)
    return {
        "question": body.question,
        "found": bool(results),
        "answer": [r["text"] for r in results],
        "sources": [
            {"source": r["source"], "chunk": r["chunk"], "score": r["score"]}
            for r in results
        ],
    }
