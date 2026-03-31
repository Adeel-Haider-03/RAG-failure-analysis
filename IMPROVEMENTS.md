# Improvements and Redesign
### Follow-up to RAG Failure Analysis

This document describes the follow-up work built on the original failure analysis.
It covers two targeted pipeline improvements (Part A) and a proposed redesign for
the structurally limited Q12 case (Part B).

The original analysis is documented in `README.md` and the failure analysis report.

---

## Part A — Targeted Improvements

Two failure types were selected from the original analysis for improvement:

| Fix | Failure Type | Question | Original Diagnosis |
|-----|-------------|----------|--------------------|
| Fix 1 | Chunk boundary split | Q6 | 400-char boundary severed "PE shall carry no formal statutory administrative positions" mid-sentence |
| Fix 2 | Document length bias | Q9 | DNP (73 chunks) monopolised all 5 retrieval slots, leaving MP (31 chunks) unretrieved |

---

### Fix 1 — Sentence-Aware Chunking (Q6)

**What changed in `rag_pipeline.py`:**

A new chunking function `split_into_sentences()` was added alongside the original
`split_into_chunks()`. Instead of splitting at fixed 400-character intervals,
it groups complete sentences into chunks up to 600 characters. A chunk boundary
never falls mid-sentence.

The `load_documents()` function now accepts a `chunking` argument:

```python
# Original fixed-size chunking (default)
chunks = load_documents("docs", chunking="fixed")

# New sentence-aware chunking
chunks = load_documents("docs", chunking="sentence")
```

**Index comparison:**

| | Fixed-Size | Sentence-Aware |
|--|-----------|----------------|
| Total chunks | 163 | 99 |
| Boundary rule | Every 400 characters | End of sentence only |

**Result:**

| | Before | After |
|--|--------|-------|
| Retrieved chunk | Starts mid-word: `"aculty is/are available..."` | Starts at sentence: `"However, at young universities..."` |
| Answer | Positive case only (negative default missing) | Positive case only (negative default still missing) |
| Status | Partial | Partial |

**What improved:** Mid-word chunk cuts eliminated. Chunks are more coherent and semantically complete.

**What did not improve:** The negative default rule ("PE shall carry no formal statutory
administrative positions where senior faculty is available") and the positive exception
("However, at young universities...") are in adjacent sentences that remain in separate
chunks even with sentence-aware splitting. The full conditional statement requires
paragraph-level or section-level chunking to stay intact.

---

### Fix 2 — Diversity-Enforced Retrieval (Q9)

**What changed:**

No new code was needed. `retrieve_diverse()` was already implemented in `rag_pipeline.py`
during the original analysis. This fix formally evaluates it as an improvement with
a before/after comparison.

The function enforces a maximum of 2 chunks per source document, preventing longer
documents from monopolising all top-k retrieval slots.

**Result:**

| | Before (Standard) | After (Diverse) |
|--|------------------|----------------|
| Sources retrieved | DNP ×5 | DNP ×2, PE ×1, MP ×2 |
| Unique documents | 1 | 3 |
| MP chunks | 0 | 2 |
| Answer | "I cannot find this information" | Correct comparison of both authority chains |
| Status | FAILED | ANSWERED |

**What improved:** Complete fix. Removing the length bias by enforcing source diversity
allowed MP chunks to appear in the retrieved set, and the model correctly described
both authority chains.

**What did not improve:** Diversity enforcement is not universally beneficial.
In Q6, enforcing diversity replaced relevant PE chunks with irrelevant content from
other documents, causing a regression. The optimal retrieval strategy depends on
whether the question requires cross-document breadth or single-document depth.

---

### How to Run the Improvements Comparison

```bash
python run_improvements.py
```

This script:
1. Loads the original index (fixed-size chunks) from `index/`
2. Builds a new index with sentence-aware chunking in `index_sentence/` (first run only)
3. Runs Q6 before and after Fix 1
4. Runs Q9 before and after Fix 2
5. Saves results to `results/improvements_*.json` and `results/improvements_*.txt`

> **Note:** Building the sentence-aware index requires API calls and will take
> approximately 3–4 minutes on the first run. Subsequent runs load the saved index.

---

## Part B — Q12: Structural Limitation and Redesign

**Why single-pass RAG cannot solve Q12:**

Q12 asks whether the international awards list changed between the 2024 MP framework
and the 2025 DNP framework. Three independent reasons explain why a standard
single-pass pipeline cannot answer this:

1. **No chunk encodes a version comparison.** The corpus contains the awards lists
   but no chunk says "this changed" or "this is identical to the previous version."
   Cosine similarity retrieval finds content that matches the query topic — but no
   author ever wrote a version comparison into the documents.

2. **Retrieval is uncoordinated.** A single-pass pipeline retrieves the top-5 most
   similar chunks globally with no mechanism to deliberately retrieve the MP annex
   AND the DNP annex as a matched pair for comparison.

3. **No concept of document versioning.** All chunks are treated as equally
   authoritative regardless of document date. The pipeline cannot reason about
   which version is newer or should take precedence.

---

**Proposed solution: Two-pass targeted retrieval**

See `two_pass_q12.py` for the full pseudocode and limitations analysis.

The design decouples the problem into two steps:

```
PASS 1 — Targeted annex retrieval:
  Filter chunks by source document + keywords ('annex', 'award', 'international')
  Retrieve the awards annex explicitly from both MP and DNP
  (bypasses similarity search — we know exactly what we need)

PASS 2 — Comparison generation:
  Send both annexes to the model with a comparison-specific prompt:
  "Here is the MP 2024 awards list and the DNP 2025 awards list.
   Are they identical? If not, what changed?"
  (different from standard RAG prompt — explicitly frames the comparison task)
```

**Key limitations of the proposed approach:**

- Annex detection by keyword is fragile — depends on knowing what to look for
- If the annex spans many chunks, the filter may return incomplete lists
- Model may hallucinate differences even when lists are identical
- Does not generalise without a query classifier to detect version comparison questions
- Cannot infer changes that no chunk explicitly records

---

## New Files Added

| File | Purpose |
|------|---------|
| `run_improvements.py` | Runs before/after comparison for Fix 1 and Fix 2 |
| `two_pass_q12.py` | Part B pseudocode and limitations analysis for Q12 redesign |

**Modified files:**

| File | What changed |
|------|-------------|
| `rag_pipeline.py` | Added `split_into_sentences()` function and `chunking` parameter to `load_documents()` |
