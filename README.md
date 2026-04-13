# LLM-Powered Research Workflow Automation Tool

An internal research assistant tool that integrates OpenAI GPT-4 to automate literature review, summarization, and structured note organization for academic research workflows. Features a RAG (Retrieval-Augmented Generation) pipeline using LangChain and FAISS vector store to query and synthesize information from PDF documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)

## Features

### PDF Upload & Processing
- Batch upload PDF research papers with automatic text extraction
- SHA-256 deduplication to prevent duplicate uploads
- Automatic metadata extraction (title, authors) from PDF properties
- Text chunking with LangChain's RecursiveCharacterTextSplitter for optimal RAG performance

### RAG-Powered Literature Review
- Chat-style Q&A interface for querying your paper library
- FAISS vector store with OpenAI embeddings for fast similarity search
- Source-cited responses with paper title and page number references
- Full query history tracking with token usage

### Summarization
- **Single paper summaries**: Structured analysis with key findings, methodology, and limitations
- **Multi-paper synthesis**: Map-reduce summarization across multiple papers with topic focus
- Automatic note creation from summaries

### Citation Extraction
- Regex + GPT-4 hybrid citation parser
- Extracts structured fields: title, authors, year, DOI
- Cross-reference detection against your paper library
- Searchable citation database

### Cross-Document Comparison
- Compare 2-5 papers side by side
- Structured comparison tables for methodology and findings
- Identifies agreements, disagreements, and research gaps

### Research Dashboard
- Interactive Plotly charts for research progress tracking
- Papers uploaded over time (cumulative line chart)
- Tag distribution across notes (horizontal bar chart)
- Research progress gauge (processed vs total papers)
- Recent activity feed

### Report Export
- Generate structured reports from papers, notes, and queries
- Export formats: Markdown, JSON
- SQL-backed storage for all reports
- Download directly from the dashboard

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit (multipage app) |
| LLM | OpenAI GPT-4 via LangChain |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (faiss-cpu) |
| Database | SQLite + SQLAlchemy 2.0 |
| Migrations | Alembic |
| PDF Parsing | PyMuPDF (fitz) |
| Visualization | Plotly |
| Config | pydantic-settings |
| Testing | pytest + pytest-cov |

## Architecture

```
app.py                      ← Streamlit dashboard (home page)
pages/
├── 01_Upload_Papers.py     ← PDF upload & processing
├── 02_Literature_Review.py ← RAG Q&A + summarization
├── 03_Notes_Manager.py     ← Notes CRUD with tags
├── 04_Citations.py         ← Citation extraction + comparison
└── 05_Reports.py           ← Report generation & export

core/
├── pdf_processor.py        ← PDF extraction & chunking
├── rag_pipeline.py         ← FAISS + LangChain RAG
├── summarizer.py           ← GPT-4 summarization chains
├── citation_extractor.py   ← Regex + LLM citation parser
└── cross_compare.py        ← Multi-paper comparison

db/
├── models.py               ← SQLAlchemy ORM models
├── database.py             ← Engine & session management
└── migrations/             ← Alembic migrations
```

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key with GPT-4 access

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sashankpotluru/LLM-Powered-Research-Workflow-Automation-Tool.git
   cd LLM-Powered-Research-Workflow-Automation-Tool
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

5. **Create data directories**
   ```bash
   mkdir -p data/uploads data/faiss_index data/exports
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Quick Start with Make

```bash
make setup   # Create venv, install deps, copy .env
make run     # Launch Streamlit app
make test    # Run test suite
make test-cov # Run tests with coverage report
```

## Usage Guide

### 1. Upload Papers
Navigate to **Upload Papers** and drag-and-drop your PDF files. The tool will:
- Extract text from each PDF
- Split text into chunks for the RAG pipeline
- Generate embeddings and add to the FAISS index
- Store metadata in the SQLite database

### 2. Ask Research Questions
Go to **Literature Review** and type your questions in the chat interface. The RAG pipeline will:
- Search your indexed papers for relevant chunks
- Generate an answer using GPT-4 with source citations
- Track the query and response in the database

### 3. Generate Summaries
Use the **Summarize** tab to:
- Select a single paper for structured summary
- Select multiple papers for a synthesis on a specific topic

### 4. Extract Citations
In **Citations**, select a paper and click "Extract Citations" to:
- Automatically find and parse the reference section
- Get structured citation data (title, authors, year, DOI)
- Discover cross-references to other papers in your library

### 5. Compare Papers
Select 2-5 papers in the **Cross-Document Comparison** tab to:
- Compare methodologies and findings in a structured table
- Identify agreements, contradictions, and gaps

### 6. Export Reports
Create structured reports in **Reports** by:
- Selecting papers, notes, and queries to include
- Choosing export format (Markdown or JSON)
- Downloading the generated report

## Configuration

All settings are managed via the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4` | LLM model for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `DATABASE_URL` | `sqlite:///data/research.db` | SQLite database path |
| `CHUNK_SIZE` | `1000` | Text chunk size for RAG |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=core --cov=db --cov=utils --cov-report=html

# Run specific test file
python -m pytest tests/test_pdf_processor.py -v
```

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── pages/                  # Streamlit multipage app pages
├── core/                   # Core business logic modules
├── db/                     # Database models and management
├── utils/                  # Configuration and logging
├── tests/                  # Test suite
├── data/                   # Runtime data (gitignored)
│   ├── uploads/            # Uploaded PDFs
│   ├── faiss_index/        # Persisted FAISS index
│   └── exports/            # Generated reports
├── docs/                   # Design documentation
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata & tool config
├── Makefile                # Common commands
└── alembic.ini             # Database migration config
```

## License

MIT License
