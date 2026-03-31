"""
rag_pipeline.py
---------------
Core RAG pipeline with two chunking strategies:

  1. split_into_chunks()         — original fixed-size character chunking (400 chars, 80 overlap)
  2. split_into_sentences()      — improved sentence-aware chunking (groups complete sentences)

The sentence-aware chunker was added as Part A Fix 1 to address the chunk boundary split
failure observed in Q6, where the 400-character boundary severed a conditional sentence
mid-clause, removing the word "not" from the retrieved context.

Both chunkers are available so before/after comparisons can be run with the same pipeline.
"""

import os
import json
import time
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from google import genai


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

CHUNK_SIZE        = 400   # characters — used by fixed-size chunker only
CHUNK_OVERLAP     = 80    # characters — used by fixed-size chunker only
SENTENCE_CHUNK_SIZE = 600 # approximate max characters per sentence-aware chunk
TOP_K             = 5

EMBEDDING_MODEL  = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY
on the provided context documents. If the answer is not in the context, say:
"I cannot find this information in the provided documents."
Do not use any external knowledge. Be precise and cite which document your answer
comes from."""


# ─────────────────────────────────────────────────────────────────
# Step 0: Setup
# ─────────────────────────────────────────────────────────────────

_client = None

def configure(api_key: str):
    """Initialize the Gemini client."""
    global _client
    _client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")


# ─────────────────────────────────────────────────────────────────
# Step 1: Load PDFs
# ─────────────────────────────────────────────────────────────────

def load_documents(docs_folder: str = "docs",
                   chunking: str = "fixed") -> list[dict]:
    """
    Read all PDF files from the docs/ folder and split into chunks.

    Args:
        docs_folder: path to folder containing PDF files
        chunking:    "fixed"    — original 400-char fixed-size chunking
                     "sentence" — improved sentence-aware chunking

    Returns:
        List of chunk dicts with keys: text, source, index
    """
    folder = Path(docs_folder)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{docs_folder}/'")

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
    print(f"Chunking strategy: {chunking}")

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        if chunking == "sentence":
            chunks = split_into_sentences(full_text, pdf_path.name)
        else:
            chunks = split_into_chunks(full_text, pdf_path.name)

        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks created")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────────────────────────
# Chunking Strategy 1: Fixed-size character chunking (original)
# ─────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, source: str) -> list[dict]:
    """
    Original fixed-size character chunking.
    Splits text every CHUNK_SIZE characters with CHUNK_OVERLAP overlap.

    Known limitation: chunk boundaries fall at arbitrary character positions,
    which can split sentences mid-clause. This caused the Q6 failure where
    the word 'not' in a negative conditional was severed from its clause.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text":   chunk_text,
                "source": source,
                "index":  len(chunks)
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─────────────────────────────────────────────────────────────────
# Chunking Strategy 2: Sentence-aware chunking (improved)
# ─────────────────────────────────────────────────────────────────

def split_into_sentences(text: str, source: str) -> list[dict]:
    """
    Improved sentence-aware chunking.

    Groups complete sentences into chunks up to SENTENCE_CHUNK_SIZE characters.
    Never splits a sentence mid-clause — each chunk boundary falls at a sentence end.

    This directly addresses the Q6 failure: the conditional sentence
    'PE shall NOT carry formal statutory administrative positions at a university
    where senior faculty is/are available... However, at young universities...'
    will now stay in a single chunk rather than being split between two.

    Uses simple rule-based sentence splitting to avoid external dependencies.
    Splits on '. ', '? ', '! ', and newlines followed by capital letters.
    """
    import re

    # Split into sentences using punctuation boundaries
    # Pattern: split after . ! ? followed by space and capital letter, or after newlines
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z])', text)

    # Clean empty sentences
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed the limit and we already have content,
        # save the current chunk and start a new one
        if current_chunk and len(current_chunk) + len(sentence) + 1 > SENTENCE_CHUNK_SIZE:
            chunks.append({
                "text":   current_chunk.strip(),
                "source": source,
                "index":  len(chunks)
            })
            current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    # Save the last chunk
    if current_chunk.strip():
        chunks.append({
            "text":   current_chunk.strip(),
            "source": source,
            "index":  len(chunks)
        })

    return chunks


# ─────────────────────────────────────────────────────────────────
# Step 2: Embeddings
# ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert a list of text strings into embedding vectors."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    vectors = [e.values for e in result.embeddings]
    return np.array(vectors, dtype=np.float32)


def build_index(chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Embed all chunks and build the search index."""
    texts = [c["text"] for c in chunks]
    print(f"\nBuilding embedding index for {len(texts)} chunks...")

    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding chunks {i+1}–{min(i+batch_size, len(texts))}...")
        batch_embeddings = embed_texts(batch)
        all_embeddings.append(batch_embeddings)
        if i + batch_size < len(texts):
            print("  Waiting 60s for rate limit...")
            time.sleep(60)

    embeddings = np.vstack(all_embeddings)
    print(f"Index built: {embeddings.shape[0]} vectors of dim {embeddings.shape[1]}")
    return chunks, embeddings


def save_index(chunks: list[dict], embeddings: np.ndarray, path: str = "index"):
    """Save index to disk."""
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
    """Cosine similarity between query vector and all chunk vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms  = doc_vecs  / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query]
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)


def retrieve_standard(query: str, chunks: list[dict],
                       embeddings: np.ndarray) -> list[dict]:
    """
    Standard retrieval: global top-K by cosine similarity.
    Known limitation: longer documents dominate due to having more candidate chunks.
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
    Added as Part A Fix 2 to address document length bias (Q9 failure).
    Prevents longer documents from monopolising all TOP_K slots.
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
    """Generate an answer from retrieved chunks. Retries on rate limit."""
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
# Step 5: Full RAG Query
# ─────────────────────────────────────────────────────────────────

def rag_query(query: str, chunks: list[dict],
              embeddings: np.ndarray,
              diverse: bool = False) -> dict:
    """
    Run a complete RAG query.

    Args:
        query:      The question to answer
        chunks:     All document chunks
        embeddings: All chunk vectors
        diverse:    If True, use diversity-enforced retrieval (max 2 chunks/doc)
    """
    if diverse:
        retrieved = retrieve_diverse(query, chunks, embeddings)
    else:
        retrieved = retrieve_standard(query, chunks, embeddings)

    answer = generate_answer(query, retrieved)

    return {
        "query":            query,
        "retrieval_mode":   "diverse" if diverse else "standard",
        "retrieved_chunks": [
            {"source": c["source"], "score": round(c["score"], 4), "text": c["text"]}
            for c in retrieved
        ],
        "answer": answer
    }
