# LLM-Powered Research Workflow Automation Tool — Design Spec

**Date:** 2026-04-13
**Status:** Approved
**Approach:** Monolithic Streamlit App (Approach A)
**Quality Target:** Production-grade (tests, logging, error handling, CI-ready)

---

## 1. Overview

An internal research assistant tool that integrates OpenAI GPT-4 to automate literature review, summarization, and structured note organization for academic research workflows. Features a RAG pipeline using LangChain + FAISS to query and synthesize information from PDF documents, with an interactive Streamlit dashboard for tracking research progress.

## 2. Project Structure

```
research-workflow-tool/
├── app.py                      # Streamlit entry point (home/overview page)
├── pages/
│   ├── 01_Upload_Papers.py     # PDF upload & processing
│   ├── 02_Literature_Review.py # RAG-powered Q&A
│   ├── 03_Notes_Manager.py     # Structured notes & tagging
│   ├── 04_Citations.py         # Citation extraction & cross-doc comparison
│   └── 05_Reports.py           # Export & reporting dashboard
├── core/
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF text extraction (PyMuPDF)
│   ├── rag_pipeline.py         # LangChain + FAISS RAG pipeline
│   ├── summarizer.py           # GPT-4 summarization chains
│   ├── citation_extractor.py   # Automated citation parsing
│   └── cross_compare.py        # Cross-document comparison
├── db/
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM models
│   ├── database.py             # DB engine & session management
│   └── migrations/             # Alembic migrations
│       └── env.py
├── utils/
│   ├── __init__.py
│   ├── config.py               # Settings via pydantic-settings
│   └── logging_config.py       # Structured logging setup
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   ├── test_pdf_processor.py
│   ├── test_rag_pipeline.py
│   ├── test_summarizer.py
│   ├── test_citation_extractor.py
│   ├── test_cross_compare.py
│   └── test_db.py
├── data/
│   ├── uploads/                # Uploaded PDFs (gitignored)
│   ├── faiss_index/            # Persisted FAISS index (gitignored)
│   └── exports/                # Generated reports (gitignored)
├── .env.example                # Template for env vars
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── Makefile                    # Common commands (run, test, lint, migrate)
├── alembic.ini                 # Alembic config
└── README.md
```

## 3. Data Models

### Papers
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| title | String(500) | Extracted or user-provided |
| authors | String(1000) | Comma-separated |
| abstract | Text | First ~500 words or extracted abstract |
| file_path | String(500) | Relative path to uploaded PDF |
| file_hash | String(64) | SHA-256 for dedup |
| upload_date | DateTime | UTC, auto-set |
| page_count | Integer | |
| status | Enum | processing, ready, error |
| error_message | Text | Nullable, populated on failure |

### Chunks
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| paper_id | Integer (FK → Papers) | CASCADE delete |
| chunk_text | Text | Raw text content |
| chunk_index | Integer | Order within paper |
| page_number | Integer | Source page |
| embedding_id | String(100) | Reference to FAISS index position |

### Notes
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| paper_id | Integer (FK → Papers) | Nullable (free-form notes) |
| title | String(500) | |
| content | Text | Markdown-formatted |
| tags | JSON | List of string tags |
| created_at | DateTime | UTC |
| updated_at | DateTime | UTC, auto-update |

### Citations
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| paper_id | Integer (FK → Papers) | Source paper |
| raw_text | Text | Original reference string |
| parsed_title | String(500) | GPT-4 extracted |
| parsed_authors | String(1000) | GPT-4 extracted |
| parsed_year | Integer | Nullable |
| parsed_doi | String(200) | Nullable |

### ResearchSessions
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| query | Text | User's question |
| response | Text | LLM response |
| source_papers | JSON | List of {paper_id, chunk_ids, page_numbers} |
| model_used | String(50) | e.g. "gpt-4" |
| tokens_used | Integer | Total tokens for the call |
| created_at | DateTime | UTC |

### Reports
| Column | Type | Notes |
|--------|------|-------|
| id | Integer (PK) | Auto-increment |
| title | String(500) | |
| content | Text | Full report content |
| format | Enum | markdown, pdf, json |
| included_papers | JSON | List of paper_ids |
| created_at | DateTime | UTC |

## 4. Core Features

### 4.1 PDF Upload & Processing
- Streamlit file_uploader with multi-file support (PDF only)
- SHA-256 dedup check before processing
- Text extraction via PyMuPDF (fitz) — page-by-page
- Title/author extraction: attempt from PDF metadata first, fallback to GPT-4 parsing of first page
- Chunking: LangChain RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
- Embeddings: OpenAI text-embedding-3-small (1536 dims)
- FAISS index: IndexFlatIP (inner product), persisted to disk after each upload batch
- Progress bar in UI during processing
- Error handling: mark paper as "error" with message, allow retry

### 4.2 RAG Pipeline (Literature Review)
- User enters a research question in chat-style interface
- Query embedding → FAISS similarity search (top-k=5, configurable)
- Retrieved chunks formatted with source metadata
- LangChain RetrievalQA chain with GPT-4:
  - System prompt: "You are a research assistant. Answer based on the provided sources. Cite sources as [Paper Title, p.X]."
  - Stuff chain type (all retrieved docs in context)
- Response displayed with expandable source sections
- Session history stored in ResearchSessions table
- Token usage tracked per query

### 4.3 Summarization
- Single paper: select a paper → generate abstract-level summary, key findings, methodology
- Multi-paper synthesis: select multiple papers + topic → map-reduce chain
  - Map: summarize each paper's relevant sections
  - Reduce: synthesize into cohesive summary with comparisons
- Results saved as Notes with auto-generated tags

### 4.4 Citation Extraction
- Triggered per-paper after upload (async option) or on-demand
- Step 1: Regex patterns to identify reference section and individual entries
- Step 2: GPT-4 structured output to parse each entry → {title, authors, year, doi}
- Cross-reference: if a cited paper exists in the DB, link them
- UI: table view of all citations with filtering/search

### 4.5 Cross-Document Comparison
- User selects 2-5 papers from the library
- GPT-4 prompt: compare methodologies, key findings, limitations, and research gaps
- Output: structured markdown table + narrative comparison
- Saved as a Note with "comparison" tag

### 4.6 Research Dashboard (Plotly)
- Papers uploaded over time (line chart)
- Tag distribution across notes (horizontal bar chart)
- Research progress: papers processed vs. total uploaded (gauge/progress)
- Topic clusters: derived from note tags (treemap or sunburst)
- Recent activity feed (last 10 queries, notes, uploads)
- All charts interactive with Plotly

### 4.7 Report Export
- Select papers, notes, and/or research sessions to include
- Generate structured report with sections: Executive Summary, Literature Review, Key Findings, Citations
- Export formats: Markdown (native), PDF (via markdown→HTML→PDF), JSON
- Stored in Reports table and exported to data/exports/

## 5. Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Streamlit | >=1.30 |
| LLM | OpenAI GPT-4 via LangChain | latest |
| Embeddings | text-embedding-3-small | - |
| Vector Store | FAISS (faiss-cpu) | latest |
| Database | SQLite + SQLAlchemy 2.0 | - |
| Migrations | Alembic | latest |
| PDF Parsing | PyMuPDF (fitz) | latest |
| Visualization | Plotly | latest |
| Config | pydantic-settings | latest |
| Testing | pytest + pytest-cov + pytest-mock | latest |
| Logging | Python stdlib logging | structured JSON format |
| PDF Export | markdown + weasyprint | latest |

## 6. Configuration

Via `.env` file (read by pydantic-settings):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=sqlite:///data/research.db
FAISS_INDEX_PATH=data/faiss_index
UPLOAD_DIR=data/uploads
EXPORT_DIR=data/exports
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_TOP_K=5
LOG_LEVEL=INFO
```

## 7. Error Handling Strategy

- All core functions wrapped with try/except, errors logged with context
- PDF processing: individual file failures don't block batch; failed papers marked with status="error"
- OpenAI API: retry with exponential backoff (3 attempts) via tenacity
- FAISS: graceful degradation if index corrupted (rebuild option in UI)
- DB: SQLAlchemy session management with proper rollback on error
- UI: user-friendly error messages via st.error(), detailed logs in backend

## 8. Logging

- Structured JSON logging to stdout + rotating file handler
- Log levels: DEBUG (development), INFO (production)
- Key events logged: PDF upload, chunk creation, embedding generation, LLM calls (with token counts), DB operations, errors
- Request correlation IDs for tracing multi-step operations

## 9. Testing Strategy

- **Unit tests:** Each core module tested in isolation with mocked OpenAI calls
- **Integration tests:** DB operations with in-memory SQLite, RAG pipeline with small test PDFs
- **Fixtures:** conftest.py with sample papers, chunks, pre-built FAISS index
- **Coverage target:** >80%
- **Makefile targets:** `make test`, `make test-cov`, `make lint`

## 10. Non-Goals (YAGNI)

- Multi-user auth (this is a single-user internal tool)
- Cloud deployment (runs locally)
- Real-time collaboration
- Non-PDF document support (scope limited to PDFs)
- Custom fine-tuned models
