"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Paper, Chunk, Note, Citation, PaperStatus
from utils.config import get_settings


@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """Provide test settings that use temp directories."""
    # Clear the lru_cache so our mock takes effect
    get_settings.cache_clear()

    settings = MagicMock()
    settings.openai_api_key = "sk-test-key"
    settings.openai_model = "gpt-4"
    settings.embedding_model = "text-embedding-3-small"
    settings.database_url = f"sqlite:///{tmp_path}/test.db"
    settings.faiss_index_path = str(tmp_path / "faiss_index")
    settings.upload_dir = str(tmp_path / "uploads")
    settings.export_dir = str(tmp_path / "exports")
    settings.chunk_size = 500
    settings.chunk_overlap = 100
    settings.rag_top_k = 3
    settings.log_level = "DEBUG"
    settings.ensure_dirs = MagicMock()

    # Create directories
    (tmp_path / "faiss_index").mkdir(exist_ok=True)
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "exports").mkdir(exist_ok=True)

    with patch("utils.config.get_settings", return_value=settings):
        # Also patch in all modules that import get_settings
        with patch("core.pdf_processor.get_settings", return_value=settings):
            with patch("core.rag_pipeline.get_settings", return_value=settings):
                with patch("core.summarizer.get_settings", return_value=settings):
                    with patch("core.citation_extractor.get_settings", return_value=settings):
                        with patch("core.cross_compare.get_settings", return_value=settings):
                            with patch("db.database.get_settings", return_value=settings):
                                yield settings

    # Clear again after test
    get_settings.cache_clear()


@pytest.fixture
def db_engine(mock_settings):
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Provide a test database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_paper(db_session):
    """Create a sample paper in the test database."""
    paper = Paper(
        title="Test Paper on Machine Learning",
        authors="John Doe, Jane Smith",
        abstract="This paper explores machine learning techniques.",
        file_path="/tmp/test.pdf",
        file_hash="abc123def456",
        page_count=10,
        status=PaperStatus.READY,
    )
    db_session.add(paper)
    db_session.commit()
    return paper


@pytest.fixture
def sample_chunks(db_session, sample_paper):
    """Create sample chunks for the test paper."""
    chunks = []
    for i in range(5):
        chunk = Chunk(
            paper_id=sample_paper.id,
            chunk_text=f"This is chunk {i} of the test paper about machine learning.",
            chunk_index=i,
            page_number=i + 1,
            embedding_id=f"faiss_{sample_paper.id}_{i}",
        )
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.commit()
    return chunks


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal test PDF file."""
    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test Paper Title\n\nAuthor: Test Author\n\n")
        page.insert_text(
            (50, 150),
            "This is the content of the test paper. It discusses various topics "
            "related to artificial intelligence and machine learning.\n\n"
            "Section 1: Introduction\n"
            "Machine learning is a subset of artificial intelligence.\n\n"
            "References\n"
            "[1] Smith, J. (2023). Deep Learning Fundamentals. AI Journal, 15(2), 45-67.\n"
            "[2] Doe, A. & Lee, B. (2022). Neural Networks in Practice. ML Review, 8(1), 12-28.\n",
        )

        pdf_path = tmp_path / "test_paper.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return str(pdf_path)
    except Exception:
        # Fallback: create a minimal PDF manually
        pdf_path = tmp_path / "test_paper.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        return str(pdf_path)
