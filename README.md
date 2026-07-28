# AI RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) system that lets you upload PDFs and ask questions about them in natural language — with text **and** voice. Documents are parsed and chunked locally, embedded via the Gemini API, and stored in a persistent vector database — only the relevant retrieved chunks are sent to Google Gemini for answer generation.

**Live demo:** Flask UI talks to a FastAPI backend deployed on Render (`ui_flask/app.py` → `BACKEND_URL`).

---

## Why this project

Most "RAG chatbot" tutorials do naive top-k vector search and call it done. This one goes further:

- **Hybrid retrieval** — semantic (vector) search fused with BM25 keyword search using Reciprocal Rank Fusion, not just cosine similarity alone.
- **Gemini-first, low-memory embeddings** — embeddings are generated entirely through the Gemini API, so the backend never has to load a local embedding model into memory, which matters a lot on constrained hosting like Render's free tier.
- **Memory-safe ingestion** — the PDF pipeline streams page-by-page and chunk-batch-by-chunk instead of loading an entire document into memory, with explicit garbage collection between batches.
- **Voice, not just text** — a full speech-to-text → RAG → text-to-speech loop for hands-free conversation with your documents.
- **Tested** — 12 test modules covering the chunker, embedder, PDF parsing, Chroma operations, and API endpoints.

---

## Architecture

### 1. System Overview

Two independent services talk over HTTP — this separation matters because it means the UI and the RAG engine can be deployed, scaled, and restarted independently.

```
┌─────────────────┐        HTTP/JSON        ┌──────────────────────────┐
│   Flask Web UI    │ ───────────────────▶  │      FastAPI Backend       │
│   (port 5000)      │ ◀───────────────────  │      (port 8000)            │
│                       │                          │                                │
│  chat / voice /    │                          │  /documents  /ask            │
│  upload / docs      │                          │  /voice/conversation         │
└─────────────────┘                          └──────────────┬───────────────┘
                                                              │
                                    ┌─────────────────────────┼─────────────────────────┐
                                    │                         │                         │
                            Ingestion Engine           Query Engine              Voice Processor
                          (backend/ingest.py)        (backend/query.py)      (backend/voice_realtime.py)
                                    │                         │                         │
                                    └───────────► Chroma Vector DB ◄───────────┘
                                              (persistent, DuckDB+Parquet)
                                                              │
                                                    Google Gemini API
                                                (embeddings + answer generation)
```

**Why this split?** The Flask UI never talks to Chroma or Gemini directly — it only calls the FastAPI backend (`call_backend()` in `ui_flask/app.py`). This means the RAG logic is a proper standalone API that could serve a mobile app, a CLI, or another frontend with zero changes to the backend.

---

### 2. Ingestion Pipeline — "how a PDF becomes searchable"

```
PDF Upload
    │
    ▼
PyMuPDF: extract text page-by-page (generator, not all-at-once)
    │
    ├─▶ Page has < 10 chars of text? → mark as "failed page" (likely scanned/image-only), skip
    │
    ▼
MarkdownConverter: plain text → structured Markdown
    (detects headings, bullet lists, numbered lists via regex heuristics)
    │
    ▼
TextChunker: paragraph-aware token chunking
    (tiktoken, chunk_size=200 tokens, overlap=20 tokens)
    │
    ▼
Gemini Embedder.encode(chunks)  →  768-dim vectors
    (batches of 8, with retry + exponential backoff on failure)
    │
    ▼
ChromaDB.add_chunks(ids, documents, embeddings, metadata)
    metadata = {doc_id, source_filename, page_number, chunk_index, ingested_at}
```

**Talking point for interviewer:** *"I stream the PDF page-by-page and embed in small batches instead of loading the whole document into memory — a generator yields one page at a time, and I explicitly free memory (`gc.collect()`) after each batch. This keeps memory usage flat regardless of whether the PDF is 5 pages or 500 pages, which matters a lot for a low-resource deployment like Render's free tier."*

---

### 3. Query Pipeline — "how a question becomes an answer"

This is the core of the project and the part worth explaining in most depth.

```
User Question
    │
    ▼
Embed the query (same Gemini embedder used at ingestion time)
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      
Semantic Search        Keyword Search (BM25)       
(cosine similarity     (term-frequency based,       
 over Chroma vectors)   good for exact names/numbers)
    │                      │
    └──────────┬───────────┘
               ▼
    Reciprocal Rank Fusion (RRF)
    score(doc) = Σ 1 / (60 + rank_in_each_list)
    → merges both ranked lists into one, without needing
      to normalize semantic distance vs. BM25 score directly
               │
               ▼
    Deduplicate near-identical chunks (SequenceMatcher, ≥85% similarity)
               │
               ▼
    Build prompt: retrieved chunks + explicit "positive/negative" instructions
    ("use ONLY this context", "do not hallucinate", "say I don't know only if truly empty")
               │
               ▼
    Gemini 2.5 Flash generates the answer (temperature=0.3, for factual output)
               │
               ▼
    Return: answer + citations (source file + page number) + raw chunks
```

**Talking point for interviewer:** *"Pure vector search struggles with exact terms — like a product code or a person's name — because embeddings capture meaning, not exact tokens. So I run BM25 keyword search alongside semantic search and merge them with Reciprocal Rank Fusion, which is the same technique Elasticsearch and Azure AI Search use for hybrid retrieval. It's rank-based, so I don't have to worry about normalizing cosine similarity scores against BM25 scores, which use completely different scales."*

**Why RRF and not just re-ranking scores directly?** Cosine similarity and BM25 scores live on different, non-comparable scales — you can't just add them. RRF sidesteps this by only using each document's *rank position* in each list, which is why it's a standard, score-agnostic fusion method.

---

### 4. Voice Interaction Flow

```
User speaks (browser mic, WebM/Opus)
    │
    ▼
pydub/FFmpeg: convert WebM → 16kHz mono WAV
    │
    ▼
Google Speech Recognition: transcribe → text
    │
    ▼
Same Query Pipeline as above (steps 3) → generates text answer
    │
    ▼
gTTS: text answer → MP3 audio
    │
    ▼
Audio streamed back to browser, auto-played
```

This reuses the exact same `QueryService.answer_query()` used by the text chat — voice is just a different I/O layer wrapped around the same RAG core, not a separate implementation.

---

### 5. Resilience Design Choices (good things to mention proactively)

| Failure scenario | How the system handles it |
|---|---|
| Transient Gemini embedding API errors during ingestion | `retry_with_backoff()` retries 3x with exponential delay before giving up |
| BM25 keyword search returns nothing | Hybrid retriever falls back to semantic-only results automatically |
| Retrieval mode itself throws an exception | Falls back to semantic search as the last line of defense |
| A PDF page is scanned/image-only | Flagged in `failed_pages`, ingestion continues for the rest of the document instead of failing the whole upload |

**Talking point for interviewer:** *"I designed this with graceful degradation in mind — keyword search and retrieval mode both have a fallback path, a bad PDF page doesn't take down the whole upload, and transient Gemini API errors are retried with backoff rather than failing the request immediately."*

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Frontend UI | Flask, Jinja templates, vanilla JS |
| Vector DB | ChromaDB (DuckDB + Parquet persistence) |
| Embeddings | Gemini Embeddings (`gemini-embedding-001`) |
| Keyword search | BM25 (`rank-bm25`) |
| LLM | Google Gemini 2.5 Flash (`google-genai` SDK) |
| PDF parsing | PyMuPDF |
| Tokenization | `tiktoken` |
| Voice | SpeechRecognition (Google STT), gTTS, pydub/FFmpeg |
| Testing | Pytest |
| Deployment | Docker, Render |

---

## Key Features

- 📄 **PDF ingestion** with page-level failure tracking (flags image-only/scanned pages instead of silently skipping them)
- 🔀 **Configurable retrieval modes** — `hybrid`, `semantic`, or `keyword` via a single env var, with automatic fallback to semantic search if a mode fails
- 🧠 **Gemini-only embedding pipeline** — no local model to load or download, keeping the backend's memory footprint low; vectors are normalized to a fixed 768-dim space
- 🗣️ **Voice conversation endpoint** — upload/stream audio, get an audio response, one full RAG turn
- 📚 **Batch upload**, document listing, and deletion (with cascading chunk cleanup) via REST endpoints
- 🧹 **Duplicate chunk filtering** before context is sent to the LLM, to avoid repetitive/wasted context
- 🐳 **Dockerized** backend, ready to deploy

---

## Getting Started

### Prerequisites
- Python 3.11
- A [Google Gemini API key](https://makersuite.google.com/app/apikey)
- FFmpeg (for voice features)

### Setup

```bash
git clone https://github.com/Rajan-vish/AI-RAG-Chatbot-26.git
cd AI-RAG-Chatbot-26

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env           # then add your GOOGLE_API_KEY
```

### Run

```bash
# Terminal 1 — backend
python -m uvicorn backend.app:app --reload
# → http://localhost:8000/docs (Swagger UI)

# Terminal 2 — frontend
python ui_flask/app.py
# → http://localhost:5000
```

### Run tests

```bash
pytest backend/tests/ -v
```

---

## Configuration

All tunable via `.env` (see `.env.example` for full list):

| Variable | Default | Purpose |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` / `semantic` / `keyword` |
| `RETRIEVAL_K` | `5` | Chunks retrieved per query |
| `LLM_TEMPERATURE` | `0.3` | Gemini generation temperature |
| `DUPLICATE_THRESHOLD` | `0.85` | Similarity cutoff for chunk dedup |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |

---

## Known Limitations

- No OCR — scanned/image-only PDF pages are skipped and reported, not read
- Single-user, no auth/multi-tenancy
- English-tuned tokenizer and retrieval
- Depends entirely on the Gemini API for embeddings and generation — there's no local/offline fallback if the API is unavailable

---

# Synchronization Report

**Overview (intro paragraph)**
- Changed "Documents are parsed and embedded locally" to "Documents are parsed and chunked locally, embedded via the Gemini API" — the old wording directly contradicted the rest of the README (Architecture, Tech Stack) and your confirmation that embedding is now Gemini-first, not local.

**Why this project**
- Replaced the "Resilient embeddings — Gemini-primary, local-model-fallback strategy (`FallbackEmbedder`)" bullet with a bullet describing the actual current behavior: Gemini-only embeddings, chosen specifically for the low-memory deployment goal. This removes the fallback claim while keeping the same "why it matters for hosting" reasoning that was already there.
- Trimmed "embedders (local/Gemini/fallback)" in the testing bullet to "embedder," since there's now one embedding path, not three.

**Architecture → Ingestion Pipeline**
- Relabeled `Embedder.encode(chunks)` as `Gemini Embedder.encode(chunks)` for clarity now that it's the only embedding path. Left the 768-dim vector size, batch size (8), and retry/backoff behavior unchanged — nothing in your instructions indicated those changed, so I didn't touch them.

**Architecture → Query Pipeline**
- Updated "same embedder used at ingestion time" to "same Gemini embedder used at ingestion time" for consistency with the ingestion pipeline's new wording. No structural change.

**Architecture → System Overview / Voice Flow**
- No changes needed. The System Overview diagram already showed a single "Google Gemini API (embeddings + answer generation)" box with no local-model box, so it was already accurate. The Voice Interaction Flow doesn't reference the embedding provider at all.
- Note: this README uses plain text/ASCII diagrams, not Mermaid — there are no `mermaid` code blocks to update here, so items about "Mermaid diagram consistency" don't apply to this document as written.

**Architecture → Resilience Design Choices**
- Removed the row describing `FallbackEmbedder` switching to a local `nomic-embed-text-v1` model on Gemini API failure — that mechanism no longer exists per your update.
- Reworded the closing "talking point" quote so it no longer claims the embedding API itself has "a fallback path" (it doesn't anymore); it now accurately describes retry-with-backoff for transient Gemini errors, plus the keyword-search and bad-page resilience paths that are still real.

**Tech Stack**
- Changed the Embeddings row from "`nomic-embed-text-v1` (local) / Gemini Embeddings — auto-switching" to just "Gemini Embeddings (`gemini-embedding-001`)."

**Key Features**
- Rewrote the "Auto embedding provider" bullet (previously claiming Gemini-primary/local-fallback) to describe the actual Gemini-only pipeline and its memory benefit, without adding any new unverified capability.

**Getting Started**
- Removed `python download_model.py # downloads local embedding model (~100-500MB)` from Setup, since there's no local embedding model to download anymore.

**Configuration**
- Removed the `EMBEDDING_PROVIDER` (`auto` / `gemini` / `local`) row — that variable controlled the now-removed local/Gemini switching logic, so it's no longer applicable. All other variables (`RETRIEVAL_MODE`, `RETRIEVAL_K`, `LLM_TEMPERATURE`, `DUPLICATE_THRESHOLD`, `RRF_K`) are unrelated to the embedding provider and were left unchanged.

**Known Limitations**
- Added one new, real limitation that follows directly from removing the local fallback: the app now depends entirely on Gemini API availability for embeddings and generation, with no offline/local path if the API is down. This isn't a new invented feature — it's the direct, honest consequence of the change you described.

**Items reviewed but intentionally left unchanged**
- Lazy initialization, singleton resource initialization, and "reduced startup memory" were listed in your review checklist, but this README doesn't describe any of those mechanisms anywhere, and no source code was provided to confirm them. Rather than invent architecture that isn't documented, I left them out. If these are real and you'd like them documented, share the relevant backend module (or a short description) and I'll add an accurate section.
- Render deployment optimization beyond what's already stated (Docker, Render, memory-safe streaming ingestion) isn't described elsewhere in the README, so nothing further was added there either.
- Tests count (12 modules), Python version, FFmpeg requirement, RRF constant, dedup threshold, and temperature were all left as-is — none are related to the embedding migration and I have no basis to change them.