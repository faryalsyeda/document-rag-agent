# Document RAG Agent

A small, working document question-answering demo. Ask a question over
your own documents and get an answer quoted from those documents,
with sources.

## What it does

1. Loads plain text documents from `sample_docs/`.
2. Splits each document into sections.
3. Scores each section against the question with keyword / TF-IDF
   retrieval (standard library only, no API key needed).
4. Returns the best matching sections, with the source file, section
   number, and score for each one.
5. If no section matches, it says "No relevant section found" instead
   of guessing an answer.

## Honest limits

- This demo does **not** use an LLM, so it does not write new text.
  Answers are quotes from the documents, not generated paragraphs.
- An LLM answer generator can be added later for a real client, with
  the same source quoting kept in place.
- Sample documents are fictional examples, not a real business. For a
  real client, replace `sample_docs/` with that client's documents.

## Run it

```bash
python3 rag.py --query "What is the refund policy?"
python3 rag.py --query "How much does the starter package cost?"
python3 rag.py --query "What services do you offer?"
```

## Optional API demo

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then send:

```json
POST /ask
{"question": "What is the refund policy?"}
```

## Sample results

| Question | Result |
|---|---|
| What is the refund policy? | Refund section from `returns_policy.txt`, 14 days, unused part only after full delivery |
| How much does the starter package cost? | Starter package section from `services_and_pricing.txt`, 200 dollars per month in this fictional example |
| What services do you offer? | Services section from `services_and_pricing.txt`, lead qualification, document question answering, follow-up automation |
| An unrelated question with no match | "No relevant section found", no guessed answer |

## Files

- `rag.py` — retrieval logic and command-line tool
- `app.py` — optional FastAPI demo
- `sample_docs/` — 3 fictional sample documents
- `requirements.txt` — FastAPI and Uvicorn, only needed for the API demo
