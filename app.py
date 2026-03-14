"""
app.py
------
Local web interface for the RAG Failure Analysis system.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
You can ask questions one at a time and see the retrieved chunks and answers.
"""

import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import configure, load_documents, build_index, save_index, load_index, rag_query

app = Flask(__name__)

# Loaded once at startup, shared across all requests
_chunks     = None
_embeddings = None


@app.route("/")
def index():
    """Serve the UI."""
    return open("ui.html", encoding="utf-8").read()


@app.route("/questions")
def get_questions():
    """Return all questions from questions.json."""
    with open("questions.json", "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/ask", methods=["POST"])
def ask():
    """Run a single RAG query. Expects JSON: { question, diverse }"""
    data     = request.json or {}
    question = data.get("question", "").strip()
    diverse  = data.get("diverse", False)

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = rag_query(question, _chunks, _embeddings, diverse=diverse)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def startup():
    """Load the API key, build or load the index, then start the server."""
    global _chunks, _embeddings

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Add GEMINI_API_KEY=your_key to your .env file")
        exit(1)

    configure(api_key)

    if Path("index/embeddings.npy").exists():
        _chunks, _embeddings = load_index()
    else:
        print("No index found — building from docs/ folder...")
        _chunks = load_documents("docs")
        _chunks, _embeddings = build_index(_chunks)
        save_index(_chunks, _embeddings)

    print("\nReady. Open http://localhost:5000 in your browser.\n")


if __name__ == "__main__":
    startup()
    app.run(debug=False, port=5000)
