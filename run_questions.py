"""
run_questions.py
----------------
Runs all 12 questions through the RAG pipeline and saves results.

Usage:
    python run_questions.py                        # Standard retrieval
    python run_questions.py --diverse              # Diversity-enforced retrieval
    python run_questions.py --both                 # Run both modes

Results are saved to the results/ folder as JSON and plain-text summary.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import configure, load_documents, build_index, save_index, load_index, rag_query


def run_all(questions, chunks, embeddings, diverse=False) -> list[dict]:
    """Run all questions through the pipeline and return results."""
    mode_label = "DIVERSE" if diverse else "STANDARD"
    print(f"\n{'='*60}")
    print(f"Running {len(questions)} questions [{mode_label} retrieval]")
    print(f"{'='*60}")

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['id']}: {q['question'][:70]}...")
        result = rag_query(q["question"], chunks, embeddings, diverse=diverse)

        sources = [c["source"] for c in result["retrieved_chunks"]]
        unique  = len(set(sources))
        print(f"  Sources ({unique} doc{'s' if unique > 1 else ''}): "
              f"{[s.replace('Framework-','').replace('.pdf','') for s in sources]}")
        print(f"  Answer: {result['answer'][:120]}...")

        results.append({
            "id":               q["id"],
            "category":         q["category"],
            "question":         q["question"],
            "hypothesis":       q["hypothesis"],
            "retrieval_mode":   result["retrieval_mode"],
            "retrieved_chunks": result["retrieved_chunks"],
            "answer":           result["answer"],
        })

    return results


def save_results(results: list[dict], label: str):
    """Save results as JSON and human-readable text summary."""
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (full data)
    json_path = f"results/results_{label}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Plain text summary
    txt_path = f"results/summary_{label}_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"RAG Failure Analysis Results [{label.upper()}]\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write("=" * 60 + "\n")
            f.write(f"ID: {r['id']} | Category: {r['category']}\n")
            f.write(f"Question: {r['question']}\n\n")
            f.write(f"Hypothesis: {r['hypothesis']}\n\n")
            f.write(f"ANSWER:\n{r['answer']}\n\n")
            f.write("Retrieved from:\n")
            for c in r["retrieved_chunks"]:
                f.write(f"  - {c['source']} (score: {c['score']})\n")
            f.write("\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(description="Run RAG failure analysis questions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--diverse", action="store_true",
                       help="Use diversity-enforced retrieval (max 2 chunks per document)")
    group.add_argument("--both",    action="store_true",
                       help="Run both standard and diverse retrieval")
    args = parser.parse_args()

    # ── API setup ─────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Add GEMINI_API_KEY=your_key to your .env file")
        return
    configure(api_key)

    # ── Load or build index ───────────────────────────────────
    if Path("index/embeddings.npy").exists():
        chunks, embeddings = load_index()
    else:
        print("\nBuilding index from docs/ folder...")
        chunks = load_documents("docs")
        chunks, embeddings = build_index(chunks)
        save_index(chunks, embeddings)

    # ── Load questions ────────────────────────────────────────
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"\nLoaded {len(questions)} questions.")

    # ── Run ───────────────────────────────────────────────────
    if args.both:
        std_results = run_all(questions, chunks, embeddings, diverse=False)
        save_results(std_results, "standard")

        div_results = run_all(questions, chunks, embeddings, diverse=True)
        save_results(div_results, "diverse")


    elif args.diverse:
        results = run_all(questions, chunks, embeddings, diverse=True)
        save_results(results, "diverse")

    else:
        results = run_all(questions, chunks, embeddings, diverse=False)
        save_results(results, "standard")

    print("\nDone.")


if __name__ == "__main__":
    main()
