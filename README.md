# RAG Failure Analysis

### Why Did This RAG Fail?

A structured failure analysis of a minimal RAG (Retrieval-Augmented Generation) system built on four real-world HEC Pakistan policy documents.

---

## Purpose

This project does **not** try to build a strong RAG system.  
The goal is to analyze **where and why** RAG systems fail and whether those failures are fixable or structural.

---

## Corpus

Source: https://www.hec.gov.pk/english/policies/Pages/default.aspx

Four publicly available HEC policy documents placed in the `docs/` folder:

| File                            | Framework                        | Year | Chunks |
| ------------------------------- | -------------------------------- | ---- | ------ |
| `Framework-MP.pdf`              | Meritorious Professor            | 2024 | 31     |
| `Framework-PE.pdf`              | Professor Emeritus               | 2024 | 21     |
| `Framework-DNP.pdf`             | Distinguished National Professor | 2025 | 73     |
| `Appointment-Prof-Practice.pdf` | Professor of Practice            | 2025 | 38     |

These documents share overlapping vocabulary and similar policy structures while defining different eligibility rules. This makes them useful for testing retrieval confusion and cross-document reasoning failures.

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

# Both modes
python run_questions.py --both
```

Results are saved to the `results/` folder as `.json` and `.txt` files.

> **Note:** `run_questions.py` reads questions from `questions.json`, which must be present in the project folder. Each entry requires at minimum an `"id"`, `"question"`, `"category"`, and `"hypothesis"` field. The provided `questions.json` already contains all 12 questions in the correct format.

> `analysis.json` is a standalone research artifact, it is not read by any script. It contains manual failure classifications added after reviewing the pipeline output. Each entry uses the question `"id"` as the key and contains `"failure_type"` and `"notes"` fields. It exists for documentation and analysis purposes only.

---

## The Two Retrieval Modes

| Mode         | How it works                      | When it helps                                |
| ------------ | --------------------------------- | -------------------------------------------- |
| **Standard** | Global top-5 by cosine similarity | Fast, simple but longer documents dominate   |
| **Diverse**  | Max 2 chunks per source document  | Forces cross-document representation         |

The DNP document (73 chunks) is 3.5x longer than the PE document (21 chunks). In standard retrieval, the longer document has more chunks competing for the top-k positions, increasing the probability that it dominates the retrieved set even when other documents are relevant.

---

## Project Structure

```
├── docs/                         # PDF corpus (not included in repo)
├── index/                        # Saved embeddings (auto-generated, gitignored)
├── results/                      # Query outputs (auto-generated, gitignored)
├── example_outputs/              # Actual results from both retrieval runs
    ├── screenshots/              # UI screenshots
├── rag_pipeline.py               # Core RAG logic — read this first
├── run_questions.py              # Batch runner for all 12 questions
├── app.py                        # Flask server for the web UI
├── ui.html                       # Web interface
├── questions.json                # 12 designed questions with hypotheses
├── analysis.json                 # Manual failure annotations per question
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

12 questions across 7 failure categories, each designed to trigger a specific RAG weakness:

| ID      | Category                             | Target Weakness                                                                |
| ------- | ------------------------------------ | ------------------------------------------------------------------------------ |
| Q1–Q2   | Same Numbers, Different Rules        | Same values, different meaning across documents                                |
| Q3–Q4   | Multi-Document Reasoning             | Cross-document synthesis                                                       |
| Q5–Q6   | Conditional Answers                  | Multi-clause conditions, conditional structure vulnerable to chunk boundaries  |
| Q7      | Buried Eligibility                   | Annex content ranked below body text                                           |
| Q8–Q9   | Process & Authority Confusion        | Similar process language, different authorities                                |
| Q10–Q11 | Category Confusion (PoP vs Academic) | Practitioner vs academic track                                                 |
| Q12     | Temporal & Version Confusion         | No mechanism for document versioning                                           |

---

## Key Findings

- **No clear generation errors were observed in this experiment.** Incorrect answers were explainable by limitations in retrieval and chunking rather than the model’s reasoning.

- **Fixed-size chunking was the dominant failure cause.** Chunk boundary splits, intra-document chunk misses, and document length bias were all amplified by the use of fixed 400-character chunks without semantic boundaries.

- **Q12 was not directly answerable by this pipeline.** It requires cross-document comparison, which a standard single-pass RAG setup does not perform.

- **Diversity enforcement fixed some failures and created others.** Q9 was corrected while Q6 regressed, showing that no single retrieval configuration performed best across all question types in this experiment.

Manual annotations for each question are documented in `analysis.json`

---

## UI Screenshots

_Standard Retrieval_
![Standard Retrieval](example_outputs/screenshots/Q3%20standard%20retrieval.png)

_Diverse Retrieval_
![Diverse Retrieval](example_outputs/screenshots/Q3%20diverse%20retrieval.png)

## License

This project is for academic coursework and research demonstration purposes.
