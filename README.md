# Job RAG Agent — Resume-to-Job Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)

Professional, modular pipeline that converts resumes into structured candidate profiles and matches them to job descriptions using a hybrid Retrieval-Augmented-Generation approach. The system blends lexical retrieval (BM25) with semantic embeddings (Chroma) and a configurable scoring pipeline to produce explainable job recommendations.

Overview
--------

This project is built for engineering teams and recruiters who need a robust, explainable matching system that can run locally or be deployed to cloud infrastructure. It is purposely modular so components (retriever, embedder, vector store, scoring) can be swapped with minimal friction.

Highlights
---------

- Hybrid BM25 + embedding retrieval for complementary recall/precision.
- Resume parsing and experience inference to normalize seniority and durations.
- Explainable scoring: each result includes component-level scores and evidence snippets.
- Local Vector DB (Chroma) support for offline or privacy-sensitive deployments.

Repository Structure
--------------------

- `analyze_resume.py` — CLI runner for quick resume-to-job matching demos.
- `config.py` — central configuration and environment mapping.
- `agents/` — core agent modules: `orchestrator.py`, `scorer.py`, `resume_parser.py`, `job_fetcher.py`, `rewriter.py`.
- `api/` — lightweight API server (FastAPI-style) with `main.py` and `routes.py`.
- `data/` — BM25 corpus and Chroma DB storage (kept out of Git; large files).
- `ui/` — small frontend demo (`index.html`, `app.js`) for interactive exploration.
- `utils/` — helpers: `bm25_index.py`, `chroma_client.py`, `embedder.py`, `pdf_extractor.py`, `exp_inference.py`, and others.
- `tests/` — unit and integration tests powered by `pytest`.


Tech Stack
----------

- Language: Python 3.10+
- Web/API: FastAPI (ASGI), compatible with Uvicorn/Gunicorn
- Retrieval: BM25 (custom on-disk corpus), hybrid orchestration in `utils/hybrid_retriever.py`
- Embeddings & Vector DB: Pluggable embedding providers (OpenAI/local) + Chroma for vector persistence
- Storage: JSON metadata, SQLite/Chroma files for vector shards, filesystem for artifacts
- Testing: pytest
- Packaging: `venv` + `pip` (use `requirements.txt`)


**Repository Layout**
- **Top-level scripts:** `analyze_resume.py`, `run_pipeline_test.py` — convenient runners.
- **Agents:** `agents/` — core modules for parsing, retrieval orchestration, scoring, and rewriting.
- **API:** `api/` — FastAPI-like routes and main app entry point.
- **Data:** `data/` — BM25 corpus and Chroma DB files (excluded from git via `.gitignore`).
- **Utils:** `utils/` — helpers for embedding, PDF extraction, indexing, and freshness logic.
- **UI:** `ui/` — sample single-page app to demo results.

**Installation (Local Development)**
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Populate environment variables (API keys, if using external embedder providers) via a `.env` file or system env vars.

3. Prepare data: put Chroma and BM25 data in `data/` or run provided ingestion scripts (not included in this repo snapshot).

**Quick Start — Run API (development)**

```powershell
# from repo root
python -m api.main
# or run analyze_resume.py for a local pipeline demo
python analyze_resume.py --resume sample_resumes/example.pdf
```

**Running Tests**
- Run unit tests with `pytest` from the repository root:

```powershell
pytest -q
```

**Configuration**
- `config.py` centralizes knobs: index paths, retriever weights, and embedder options.
- Adjust weights in `agents/scorer.py` to tune recall/precision tradeoffs.

**Development Notes & Best Practices**
- Keep heavy data files (vector DBs, corpora) out of Git — they are in `.gitignore` and should be stored in artifact storage or a data bucket.
- Use the modular `agents/` surface to swap alternative retrievers (e.g., Faiss) or embedder implementations by implementing the same interface in `utils/embedder.py`.
- When evaluating model-driven components, version your embedder model and persist the mapping from vector shard → model used.
- Tests are lightweight and focused; add edge cases for parsing different resume formats and malformed job posts.

**Deployment & GitHub**
- This project is ready for a GitHub repository. Follow these steps:

```powershell
git init
git add .
git commit -m "Initial commit: Job RAG Agent"
git branch -M main
git remote add origin <your-git-remote-url>
git push -u origin main
```

- Ensure secrets (API keys) are managed with GitHub Secrets or an external secret manager, never in the repository.

**Security & Privacy**
- Be careful storing resumes and personal data in public repos. Use encryption and access controls for storage containing PII.
- Audit any 3rd-party services and ensure you have user consent for processing resumes.

**Contributing**
- Fork, add tests for new behavior, and open a PR with a clear description of changes and rationale. Follow the existing code style and add unit tests for parsing and scoring changes.

**Where to Look First (Files of Interest)**
- `api/main.py` — API entrypoint.
- `agents/orchestrator.py` — retrieval → scoring flow.
- `agents/resume_parser.py` — resume parsing logic.
- `utils/bm25_index.py` and `utils/chroma_client.py` — retrieval backends.

**Contact & Maintainer Notes**
- Maintainer: repository owner (update with a real contact/email in a private repo).

---

This README is intended to be both an onboarding guide and a high-level design doc for engineers integrating, iterating, or deploying this system. If you want, I can also:
- Add a CONTRIBUTING.md and CODE_OF_CONDUCT.
- Create a minimal GitHub Actions workflow for CI that runs `pytest` on pushes.
- Scaffold an ingestion script to rebuild `data/bm25_index/corpus.json` from raw job sources.
