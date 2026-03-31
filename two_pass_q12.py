"""
two_pass_q12.py
---------------
Part B — Proposed system design for Q12 (Temporal and Version Confusion)

WHY SINGLE-PASS RAG CANNOT SOLVE Q12
--------------------------------------
Q12 asks: "Are there any criteria that changed between the 2024 MP framework
and the 2025 DNP framework, specifically around the international awards list?"

A standard single-pass RAG pipeline fails because:

1. No chunk was ever written to answer a version comparison question.
   The corpus contains the awards lists but no chunk says "this changed" or
   "this is identical across versions." Cosine similarity finds content that
   matches the query topic — but no chunk matches the meta-question about
   version differences.

2. RAG retrieves relevant chunks independently. It has no mechanism to
   retrieve the awards annex from MP AND the awards annex from DNP
   simultaneously as a pair for comparison.

3. The pipeline treats all chunks as equally current regardless of document
   date. It has no concept of version precedence.

PROPOSED SOLUTION: Two-Pass Retrieval with Targeted Comparison
---------------------------------------------------------------
Instead of asking the model to find version differences, we:
  1. Explicitly retrieve the awards annex from each document by source
  2. Send both annexes to the model with a direct comparison prompt

This is a targeted retrieval approach — we know which documents and which
sections we need, so we retrieve them directly rather than relying on
similarity-based retrieval.


PSEUDOCODE
----------

FUNCTION two_pass_q12(chunks, embeddings):

    # ── PASS 1: Targeted annex retrieval ─────────────────────────
    # Instead of similarity search, filter chunks by known source and keywords

    mp_annex_chunks  = []
    dnp_annex_chunks = []

    FOR each chunk in chunks:
        IF chunk.source == "Framework-MP.pdf":
            IF "annex" in chunk.text.lower() OR "award" in chunk.text.lower():
                mp_annex_chunks.append(chunk)

        IF chunk.source == "Framework-DNP.pdf":
            IF "annex" in chunk.text.lower() OR "award" in chunk.text.lower():
                dnp_annex_chunks.append(chunk)

    # Take the top 2 most relevant annex chunks from each document
    mp_annex  = sort_by_relevance(mp_annex_chunks,  query="international awards list")[:2]
    dnp_annex = sort_by_relevance(dnp_annex_chunks, query="international awards list")[:2]

    IF mp_annex is empty OR dnp_annex is empty:
        RETURN "Could not retrieve awards annex from one or both documents."

    # ── PASS 2: Comparison generation ────────────────────────────
    # Build a comparison-specific prompt — different from the standard RAG prompt

    comparison_prompt = """
    You are comparing two versions of an HEC policy document.

    DOCUMENT A — MP Framework (2024):
    {mp_annex_text}

    DOCUMENT B — DNP Framework (2025):
    {dnp_annex_text}

    QUESTION: Are the lists of recognized international awards identical
    across these two versions? If not, what has changed?

    Compare the two lists carefully and state specifically:
    1. Whether they are identical
    2. Any additions, removals, or modifications
    3. Which version should be considered current
    """

    answer = generate_answer_with_prompt(comparison_prompt)
    RETURN answer


LIMITATIONS OF THIS APPROACH
------------------------------
1. Annex detection by keyword is fragile.
   If the annex chunks do not contain the word "annex" or "award", they
   will be missed. The approach depends on knowing what to look for in advance.

2. If the annex is split across many small chunks, the two-chunk limit
   may not capture the full list. A larger top-k or a chunk merging step
   would be needed.

3. The model may still hallucinate differences even when given both lists.
   If the lists are identical, some models tend to find minor differences
   to appear helpful. The prompt needs to explicitly instruct the model
   that "identical" is a valid and expected answer.

4. This approach is question-specific — it only works for Q12.
   Generalising to arbitrary version comparison questions would require
   a query classifier that detects "is this a version comparison question?"
   and routes it to the two-pass system. That is a significantly more
   complex architecture.

5. It does not solve the deeper problem: if documents genuinely changed
   across versions and no chunk explicitly describes the change, the model
   still cannot infer it. The two-pass approach helps when the content is
   present but the comparison was never written — it does not help when
   the relevant content is absent entirely.
"""
