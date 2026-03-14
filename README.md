# RAG Failure Analysis

### Why Did This RAG Fail?

A structured failure analysis of a minimal RAG (Retrieval-Augmented Generation) system built on four real-world HEC Pakistan policy documents.

---

## Purpose

This project does **not** try to build a strong RAG system.  
The goal is to analyze **where and why** RAG systems fail — and whether those failures are fixable or structural.

---

## Corpus

Four publicly available HEC policy documents placed in the `docs/` folder:

| File                            | Framework                        | Year | Chunks |
| ------------------------------- | -------------------------------- | ---- | ------ |
| `Framework-MP.pdf`              | Meritorious Professor            | 2024 | 31     |
| `Framework-PE.pdf`              | Professor Emeritus               | 2024 | 21     |
| `Framework-DNP.pdf`             | Distinguished National Professor | 2025 | 73     |
| `Appointment-Prof-Practice.pdf` | Professor of Practice            | 2025 | 38     |

These documents were chosen because they share overlapping vocabulary, cross-reference each other, and describe different rules using similar language — ideal conditions for surfacing RAG failures.

---

## System Architecture

```
PDF files → Chunking (400 chars, 80 overlap) → Embeddings (gemini-embedding-001)
                                                        ↓
Question → Query embedding → Cosine similarity → Top-5 chunks → Gemini → Answer
```

**Intentional simplicity:** No re-ranking, no query expansion, no hybrid retrieval. Failures surface more clearly in a minimal system.

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Create a `.env` file** in your project folder:

```
GEMINI_API_KEY=your_key_here
```

**3. Place the four PDF files** in a `docs/` folder.

---

## How to Run

### Option A — Web UI (recommended)

Ask questions one at a time in your browser. Toggle between standard and diverse retrieval.

```bash
python app.py
```

Then open **http://localhost:5000**

### Option B — Run all 12 questions at once

```bash
# Standard retrieval only
python run_questions.py

# Diversity-enforced retrieval only
python run_questions.py --diverse

# Both modes, with side-by-side comparison
python run_questions.py --both
```

Results are saved to the `results/` folder as `.json` and `.txt` files.

---

## The Two Retrieval Modes

| Mode         | How it works                      | When it helps                                |
| ------------ | --------------------------------- | -------------------------------------------- |
| **Standard** | Global top-5 by cosine similarity | Fast, simple — but longer documents dominate |
| **Diverse**  | Max 2 chunks per source document  | Forces cross-document representation         |

The DNP document (73 chunks) is 3.5× longer than the PE document (21 chunks). In standard retrieval, it statistically dominates the top-5 results regardless of semantic relevance. This is the main failure mechanism observed in this project.

---

## Project Structure

```
├── docs/                         # PDF corpus (not included in repo)
├── index/                        # Saved embeddings (auto-generated, gitignored)
├── results/                      # Query outputs (auto-generated, gitignored)
├── rag_pipeline.py               # Core RAG logic — read this first
├── run_questions.py              # Batch runner for all 12 questions
├── app.py                        # Flask server for the web UI
├── ui.html                       # Web interface
├── questions.json                # 12 designed questions with hypotheses
├── requirements.txt
└── README.md
```

---

## What This Code Does NOT Cover

- Re-ranking or query expansion
- Hybrid retrieval (keyword + semantic)
- Semantic or structure-aware chunking
- Knowledge graphs or entity linking
- Any production-readiness concerns (authentication, error handling at scale, etc.)

These omissions are intentional. Each one is a potential fix for a specific failure type identified in the analysis.

---

## Questions

12 questions across 6 failure categories, each designed to trigger a specific RAG weakness:

| ID      | Category                             | Target Weakness                                 |
| ------- | ------------------------------------ | ----------------------------------------------- |
| Q1–Q2   | Same Numbers, Different Rules        | Numerical confusion across documents            |
| Q3–Q4   | Multi-Document Reasoning             | Cross-document synthesis                        |
| Q5–Q6   | Conditional Answers                  | Multi-clause logic dropped by generation        |
| Q7      | Buried Eligibility                   | Annex content ranked below body text            |
| Q8–Q9   | Process & Authority Confusion        | Similar process language, different authorities |
| Q10–Q11 | Category Confusion (PoP vs Academic) | Practitioner vs academic track                  |
| Q12     | Temporal & Version Confusion         | No mechanism for document versioning            |
