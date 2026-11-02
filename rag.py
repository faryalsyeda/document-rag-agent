"""Document RAG Agent (retrieval demo).

Ask a question over your own documents and get an answer quoted from
the documents, with sources. This demo uses keyword / TF-IDF retrieval
from the Python standard library only. It does not use an LLM, so it
does not generate new text or guess an answer: if no document section
matches the question, it says so honestly.

Run:
    python3 rag.py --query "What is the refund policy?"
    python3 rag.py --query "How much does the starter package cost?" --top 2

Sample documents live in sample_docs/. Replace that folder with a
client's own documents to use it on real content. Sample documents
are fictional examples, not a real business.
"""

import argparse
import math
import os
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "and", "or", "but", "if", "then", "than",
    "so", "such", "of", "at", "by", "for", "with", "about", "into",
    "through", "after", "over", "between", "out", "against", "during",
    "to", "from", "in", "on", "as", "can", "could", "should", "would",
    "will", "just", "me", "my", "your", "our", "their", "this", "that",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def load_docs(docs_dir):
    """Load .txt files and split them into paragraph chunks."""
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            chunks.append({
                "source": filename,
                "chunk": i + 1,
                "text": paragraph,
                "tokens": tokenize(paragraph),
            })
    return chunks


def ask(question, chunks, top_n=3):
    """Return the best matching chunks for a question, with scores."""
    query_tokens = tokenize(question)
    if not query_tokens or not chunks:
        return [], query_tokens

    # IDF over the chunks: rare words count more than common words.
    doc_count = len(chunks)
    doc_freq = Counter()
    for chunk in chunks:
        for token in set(chunk["tokens"]):
            doc_freq[token] += 1
    idf = {
        token: math.log((doc_count + 1) / (freq + 1)) + 1
        for token, freq in doc_freq.items()
    }

    scored = []
    for chunk in chunks:
        counts = Counter(chunk["tokens"])
        score = sum(counts[t] * idf.get(t, 0) for t in query_tokens if t in counts)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"score": round(score, 2), "source": chunk["source"],
         "chunk": chunk["chunk"], "text": chunk["text"]}
        for score, chunk in scored[:top_n]
    ], query_tokens


def format_answer(question, results):
    if not results:
        return (
            f"Question: {question}\n\n"
            "No relevant section found in the documents. "
            "This answer is quoted from the documents only, so it is "
            "not guessed when nothing matches."
        )
    lines = [f"Question: {question}", "", "Answer (quoted from the documents):", ""]
    for result in results:
        lines.append(f"- [{result['source']}, section {result['chunk']}] {result['text']}")
    lines.append("")
    lines.append("Sources: " + ", ".join(
        f"{r['source']} section {r['chunk']} (score {r['score']})" for r in results))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ask questions over documents.")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--docs", default="sample_docs", help="Folder of .txt documents")
    parser.add_argument("--top", type=int, default=3, help="How many sections to return")
    args = parser.parse_args()

    docs_dir = args.docs
    if not os.path.isdir(docs_dir):
        # Allow running from the project folder or its parent.
        docs_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    chunks = load_docs(docs_dir)
    results, _ = ask(args.query, chunks, top_n=args.top)
    print(format_answer(args.query, results))


if __name__ == "__main__":
    main()
