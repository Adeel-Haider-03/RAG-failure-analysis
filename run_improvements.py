"""
run_improvements.py
-------------------
Runs before/after comparisons for two pipeline improvements:

  Fix 1 — Sentence-aware chunking (addresses Q6: chunk boundary split)
    BEFORE: fixed 400-char chunking  + standard retrieval
    AFTER:  sentence-aware chunking  + standard retrieval

  Fix 2 — Diversity-enforced retrieval (addresses Q9: document length bias)
    BEFORE: fixed chunking + standard retrieval  (global top-5)
    AFTER:  fixed chunking + diverse retrieval   (max 2 chunks per source)

Usage:
    python run_improvements.py

Results saved to results/improvements_TIMESTAMP.txt and .json
Note: Fix 1 requires rebuilding the index with sentence-aware chunks.
      The original index (index/) is preserved. A new index is built in index_sentence/.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import (
    configure, load_documents, build_index, save_index, load_index, rag_query
)

# The two questions being tested
Q6 = {
    "id": "Q6",
    "question": "Can a Professor Emeritus serve in an administrative role at a university?",
    "correct_answer": "PE shall NOT carry formal statutory administrative positions where senior faculty are available. However, at young universities where senior faculty is not available, PE may be asked to take up administrative positions.",
    "fix": "sentence-aware chunking"
}

Q9 = {
    "id": "Q9",
    "question": "What is the difference between the authority that confers the Meritorious Professor title versus the DNP title?",
    "correct_answer": "MP is conferred at university level through Selection Board → Syndicate → Chancellor. DNP is conferred nationally by HEC through a national selection committee.",
    "fix": "diversity-enforced retrieval"
}


def run_single(question: str, chunks: list, embeddings, diverse: bool = False) -> dict:
    """Run a single question and return result."""
    result = rag_query(question, chunks, embeddings, diverse=diverse)
    sources = [c["source"] for c in result["retrieved_chunks"]]
    unique_sources = list(set(sources))
    return {
        "answer":         result["answer"],
        "sources":        sources,
        "unique_sources": unique_sources,
        "chunks_text":    [c["text"][:200] for c in result["retrieved_chunks"]]
    }


def check_answer(answer: str) -> str:
    """Classify answer as correct, partial, or failed."""
    answer_lower = answer.lower()
    if "i cannot find" in answer_lower or "not found" in answer_lower:
        return "FAILED — model could not find information"
    elif len(answer) < 80:
        return "PARTIAL — answer too brief"
    else:
        return "ANSWERED"


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Add GEMINI_API_KEY=your_key to your .env file")
        return

    configure(api_key)

    # ── Load original index (fixed-size chunks) ───────────────────
    print("\n" + "="*60)
    print("Loading original index (fixed-size chunking)...")
    if not Path("index/embeddings.npy").exists():
        print("ERROR: Original index not found. Run run_questions.py first.")
        return

    fixed_chunks, fixed_embeddings = load_index("index")

    # ── Build sentence-aware index ────────────────────────────────
    print("\n" + "="*60)
    print("Building sentence-aware index...")

    if Path("index_sentence/embeddings.npy").exists():
        print("Sentence index already exists — loading...")
        sentence_chunks, sentence_embeddings = load_index("index_sentence")
    else:
        print("Building new index with sentence-aware chunking...")
        sentence_chunks = load_documents("docs", chunking="sentence")
        sentence_chunks, sentence_embeddings = build_index(sentence_chunks)
        save_index(sentence_chunks, sentence_embeddings, "index_sentence")

    print(f"\nIndex comparison:")
    print(f"  Fixed-size chunks   : {len(fixed_chunks)}")
    print(f"  Sentence-aware chunks: {len(sentence_chunks)}")

    # ── FIX 1: Q6 — Sentence-aware chunking ──────────────────────
    print("\n" + "="*60)
    print("FIX 1: Q6 — Sentence-aware chunking")
    print("="*60)

    print("\n[BEFORE] Fixed-size chunking + standard retrieval")
    q6_before = run_single(Q6["question"], fixed_chunks, fixed_embeddings, diverse=False)
    print(f"  Sources: {[s.replace('Framework-','').replace('.pdf','') for s in q6_before['sources']]}")
    print(f"  Answer status: {check_answer(q6_before['answer'])}")
    print(f"  Answer: {q6_before['answer'][:200]}")

    # Show the problematic chunk
    print("\n  Key chunk retrieved (before):")
    for text in q6_before["chunks_text"]:
        if "available" in text.lower() and "young" in text.lower():
            print(f"  >>> {text[:300]}")
            break

    print("\n[AFTER] Sentence-aware chunking + standard retrieval")
    q6_after = run_single(Q6["question"], sentence_chunks, sentence_embeddings, diverse=False)
    print(f"  Sources: {[s.replace('Framework-','').replace('.pdf','') for s in q6_after['sources']]}")
    print(f"  Answer status: {check_answer(q6_after['answer'])}")
    print(f"  Answer: {q6_after['answer'][:200]}")

    # Show whether the correct chunk is now retrieved
    print("\n  Key chunk retrieved (after):")
    for text in q6_after["chunks_text"]:
        if "not" in text.lower() and ("administrative" in text.lower() or "young" in text.lower()):
            print(f"  >>> {text[:300]}")
            break

    # ── FIX 2: Q9 — Diversity-enforced retrieval ─────────────────
    print("\n" + "="*60)
    print("FIX 2: Q9 — Diversity-enforced retrieval")
    print("="*60)

    print("\n[BEFORE] Fixed-size chunking + standard retrieval")
    q9_before = run_single(Q9["question"], fixed_chunks, fixed_embeddings, diverse=False)
    print(f"  Sources: {[s.replace('Framework-','').replace('.pdf','') for s in q9_before['sources']]}")
    print(f"  Unique docs: {len(q9_before['unique_sources'])}")
    print(f"  Answer status: {check_answer(q9_before['answer'])}")
    print(f"  Answer: {q9_before['answer'][:200]}")

    print("\n[AFTER] Fixed-size chunking + diversity-enforced retrieval")
    q9_after = run_single(Q9["question"], fixed_chunks, fixed_embeddings, diverse=True)
    print(f"  Sources: {[s.replace('Framework-','').replace('.pdf','') for s in q9_after['sources']]}")
    print(f"  Unique docs: {len(q9_after['unique_sources'])}")
    print(f"  Answer status: {check_answer(q9_after['answer'])}")
    print(f"  Answer: {q9_after['answer'][:200]}")

    # ── Save results ──────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "index_comparison": {
            "fixed_size_chunks":    len(fixed_chunks),
            "sentence_aware_chunks": len(sentence_chunks)
        },
        "fix1_Q6": {
            "question":      Q6["question"],
            "correct_answer": Q6["correct_answer"],
            "before": q6_before,
            "after":  q6_after,
            "before_status": check_answer(q6_before["answer"]),
            "after_status":  check_answer(q6_after["answer"])
        },
        "fix2_Q9": {
            "question":      Q9["question"],
            "correct_answer": Q9["correct_answer"],
            "before": q9_before,
            "after":  q9_after,
            "before_status": check_answer(q9_before["answer"]),
            "after_status":  check_answer(q9_after["answer"])
        }
    }

    json_path = f"results/improvements_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    txt_path = f"results/improvements_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("IMPROVEMENT RESULTS — Before vs After\n")
        f.write(f"Generated: {results['generated']}\n")
        f.write("=" * 60 + "\n\n")

        f.write("INDEX COMPARISON\n")
        f.write(f"  Fixed-size chunks    : {len(fixed_chunks)}\n")
        f.write(f"  Sentence-aware chunks: {len(sentence_chunks)}\n\n")

        f.write("=" * 60 + "\n")
        f.write("FIX 1: Q6 — Sentence-aware chunking\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Question: {Q6['question']}\n\n")
        f.write(f"BEFORE — Status: {check_answer(q6_before['answer'])}\n")
        f.write(f"  Sources: {q6_before['sources']}\n")
        f.write(f"  Answer: {q6_before['answer']}\n\n")
        f.write(f"AFTER  — Status: {check_answer(q6_after['answer'])}\n")
        f.write(f"  Sources: {q6_after['sources']}\n")
        f.write(f"  Answer: {q6_after['answer']}\n\n")

        f.write("=" * 60 + "\n")
        f.write("FIX 2: Q9 — Diversity-enforced retrieval\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Question: {Q9['question']}\n\n")
        f.write(f"BEFORE — Status: {check_answer(q9_before['answer'])}\n")
        f.write(f"  Sources ({len(q9_before['unique_sources'])} doc): {q9_before['sources']}\n")
        f.write(f"  Answer: {q9_before['answer']}\n\n")
        f.write(f"AFTER  — Status: {check_answer(q9_after['answer'])}\n")
        f.write(f"  Sources ({len(q9_after['unique_sources'])} docs): {q9_after['sources']}\n")
        f.write(f"  Answer: {q9_after['answer']}\n\n")

    print(f"\n{'='*60}")
    print(f"Results saved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
