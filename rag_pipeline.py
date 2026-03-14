"""
rag_pipeline.py
---------------
Core RAG (Retrieval-Augmented Generation) pipeline.

What this file does, step by step:
  1. Read PDF documents from the docs/ folder
  2. Split each document into small text chunks
  3. Convert every chunk into a vector (embedding) using Gemini
  4. Save the vectors to disk so we don't recompute them every run
  5. When a question is asked:
     a. Convert the question into a vector
     b. Find the most similar chunk vectors (retrieval)
     c. Send those chunks + the question to Gemini (generation)
     d. Return the answer
"""

import os
import json
import time
import numpy as np
import fitz  # PyMuPDF — reads PDF files
from pathlib import Path
from google import genai


# ─────────────────────────────────────────────────────────────────
# Configuration — change these if needed
# ─────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400   # How many characters per chunk
CHUNK_OVERLAP = 80    # How many characters overlap between consecutive chunks
TOP_K         = 5     # How many chunks to retrieve per question

EMBEDDING_MODEL  = "gemini-embedding-001"       # Converts text to vectors
GENERATION_MODEL = "gemini-3.1-flash-lite-preview"           # Generates answers

# The instruction given to the generation model
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY
on the provided context documents. If the answer is not in the context, say:
"I cannot find this information in the provided documents."
Do not use any external knowledge. Be precise and cite which document your answer
comes from."""

# ─────────────────────────────────────────────────────────────────
# Step 0: Setup
# ─────────────────────────────────────────────────────────────────

_client = None  # Gemini client, created once in configure()

def configure(api_key: str):
    """Initialize the Gemini client with your API key."""
    global _client
    _client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")


# ─────────────────────────────────────────────────────────────────
# Step 1: Load PDFs
# ─────────────────────────────────────────────────────────────────

def load_documents(docs_folder: str = "docs") -> list[dict]:
    """
    Read all PDF files from the docs/ folder.
    Returns a list of chunks, each with: text, source filename, chunk index.
    """
    folder = Path(docs_folder)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_folder}/' folder.")

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")

        # Extract all text from the PDF
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # Split text into overlapping chunks
        chunks = split_into_chunks(full_text, pdf_path.name)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks created")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def split_into_chunks(text: str, source: str) -> list[dict]:
    """
    Split a long text string into overlapping fixed-size chunks.

    Why overlap? If an answer falls at the boundary between two chunks,
    the overlap ensures neither chunk loses that information.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()

        if chunk_text:  # Skip empty chunks
            chunks.append({
                "text": chunk_text,
                "source": source,
                "index": len(chunks)
            })

        # Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) to create overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ─────────────────────────────────────────────────────────────────
# Step 2: Embeddings
# ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into vectors using the Gemini embedding model.
    Returns a 2D numpy array of shape (len(texts), 3072).

    What is an embedding/vector?
    A list of 3072 numbers that represents the *meaning* of the text.
    Texts with similar meanings produce vectors that are mathematically close.
    """
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    vectors = [e.values for e in result.embeddings]
    return np.array(vectors, dtype=np.float32)


def build_index(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Embed all chunks and build the search index.
    Processes in batches of 50 with 60-second pauses to respect API rate limits.
    """
    texts = [c["text"] for c in chunks]
    print(f"\nBuilding embedding index for {len(texts)} chunks...")

    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding chunks {i+1}–{min(i+batch_size, len(texts))}...")
        batch_embeddings = embed_texts(batch)
        all_embeddings.append(batch_embeddings)

        # Pause between batches to avoid hitting the free-tier rate limit (100/min)
        if i + batch_size < len(texts):
            print("  Waiting 60s for rate limit...")
            time.sleep(60)

    embeddings = np.vstack(all_embeddings)
    print(f"Index built: {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}")
    return chunks, embeddings


def save_index(chunks: list[dict], embeddings: np.ndarray, path: str = "index"):
    """Save the index to disk. Avoids re-embedding on every run."""
    Path(path).mkdir(exist_ok=True)
    np.save(f"{path}/embeddings.npy", embeddings)
    with open(f"{path}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index saved to '{path}/'")


def load_index(path: str = "index") -> tuple[list[dict], np.ndarray]:
    """Load a previously saved index from disk."""
    embeddings = np.load(f"{path}/embeddings.npy")
    with open(f"{path}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Index loaded: {len(chunks)} chunks")
    return chunks, embeddings


# ─────────────────────────────────────────────────────────────────
# Step 3: Retrieval
# ─────────────────────────────────────────────────────────────────

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Measure how similar the query vector is to each chunk vector.

    Cosine similarity measures the angle between two vectors (0 to 1 scale).
    1.0 = identical direction (very similar meaning)
    0.0 = completely unrelated

    We use this instead of simple keyword search because it works even when
    the question and the answer use different words.
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms  = doc_vecs  / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a vector."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query]
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


def retrieve_standard(query: str, chunks: list[dict],
                      embeddings: np.ndarray) -> list[dict]:
    """
    Standard retrieval: return the TOP_K most similar chunks globally.

    Known limitation: longer documents have more chunks competing for the
    top-K slots, so they statistically dominate results regardless of
    actual relevance. This is called 'document length bias'.
    """
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(scores[idx])
        results.append(chunk)
    return results


def retrieve_diverse(query: str, chunks: list[dict],
                     embeddings: np.ndarray,
                     max_per_source: int = 2) -> list[dict]:
    """
    Diversity-enforced retrieval: at most max_per_source chunks per document.

    This prevents longer documents from monopolising all TOP_K slots.
    Added after observing that the DNP document (73 chunks) dominated
    retrieval in 5 of 12 questions when using standard retrieval.

    Trade-off: for single-document questions, this can introduce irrelevant
    chunks and actually harm performance (observed in Q5).
    """
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec, embeddings)
    ranked = np.argsort(scores)[::-1]

    results = []
    source_counts: dict[str, int] = {}

    for idx in ranked:
        if len(results) >= TOP_K:
            break
        source = chunks[idx]["source"]
        if source_counts.get(source, 0) < max_per_source:
            chunk = chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
            source_counts[source] = source_counts.get(source, 0) + 1

    return results


# ─────────────────────────────────────────────────────────────────
# Step 4: Generation
# ─────────────────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Send the retrieved chunks + question to Gemini and get an answer.
    Retries up to 3 times if a rate limit error occurs.
    """
    # Assemble context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────────────────────────
# Step 5: Full RAG Query (combines retrieval + generation)
# ─────────────────────────────────────────────────────────────────

def rag_query(query: str, chunks: list[dict],
              embeddings: np.ndarray,
              diverse: bool = False) -> dict:
    """
    Run a complete RAG query: retrieve relevant chunks, then generate an answer.

    Args:
        query:      The question to answer
        chunks:     All document chunks
        embeddings: All chunk vectors
        diverse:    If True, use diversity-enforced retrieval (max 2 chunks/doc)
                    If False, use standard retrieval (global top-K)

    Returns a dict with: query, retrieval_mode, retrieved_chunks, answer
    """
    if diverse:
        retrieved = retrieve_diverse(query, chunks, embeddings)
    else:
        retrieved = retrieve_standard(query, chunks, embeddings)

    answer = generate_answer(query, retrieved)

    return {
        "query": query,
        "retrieval_mode": "diverse" if diverse else "standard",
        "retrieved_chunks": [
            {
                "source": c["source"],
                "score":  round(c["score"], 4),
                "text":   c["text"]
            }
            for c in retrieved
        ],
        "answer": answer
    }
