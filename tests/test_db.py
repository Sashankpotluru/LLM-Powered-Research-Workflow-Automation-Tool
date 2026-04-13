"""Tests for database models and operations."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import (
    Base,
    Paper,
    Chunk,
    Note,
    Citation,
    ResearchSession,
    Report,
    PaperStatus,
    ReportFormat,
)


@pytest.fixture
def test_db():
    """Create a fresh in-memory database for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestPaperModel:
    def test_create_paper(self, test_db):
        paper = Paper(
            title="Test Paper",
            authors="Author A",
            file_path="/tmp/test.pdf",
            file_hash="hash123",
            page_count=5,
            status=PaperStatus.READY,
        )
        test_db.add(paper)
        test_db.commit()

        result = test_db.query(Paper).first()
        assert result.title == "Test Paper"
        assert result.authors == "Author A"
        assert result.status == PaperStatus.READY
        assert result.id is not None

    def test_paper_default_status(self, test_db):
        paper = Paper(
            title="Test",
            file_path="/tmp/test.pdf",
            file_hash="hash456",
        )
        test_db.add(paper)
        test_db.commit()

        result = test_db.query(Paper).first()
        assert result.status == PaperStatus.PROCESSING

    def test_paper_unique_hash(self, test_db):
        paper1 = Paper(title="Paper 1", file_path="/tmp/1.pdf", file_hash="same_hash")
        test_db.add(paper1)
        test_db.commit()

        paper2 = Paper(title="Paper 2", file_path="/tmp/2.pdf", file_hash="same_hash")
        test_db.add(paper2)
        with pytest.raises(Exception):  # IntegrityError
            test_db.commit()


class TestChunkModel:
    def test_create_chunk(self, test_db):
        paper = Paper(title="Test", file_path="/tmp/t.pdf", file_hash="h1")
        test_db.add(paper)
        test_db.commit()

        chunk = Chunk(
            paper_id=paper.id,
            chunk_text="Some text content",
            chunk_index=0,
            page_number=1,
        )
        test_db.add(chunk)
        test_db.commit()

        result = test_db.query(Chunk).first()
        assert result.chunk_text == "Some text content"
        assert result.paper_id == paper.id

    def test_cascade_delete(self, test_db):
        paper = Paper(title="Test", file_path="/tmp/t.pdf", file_hash="h2")
        test_db.add(paper)
        test_db.commit()

        chunk = Chunk(paper_id=paper.id, chunk_text="Text", chunk_index=0, page_number=1)
        test_db.add(chunk)
        test_db.commit()

        test_db.delete(paper)
        test_db.commit()

        assert test_db.query(Chunk).count() == 0


class TestNoteModel:
    def test_create_note(self, test_db):
        note = Note(
            title="My Research Note",
            content="Important findings here.",
            tags=["ml", "nlp"],
        )
        test_db.add(note)
        test_db.commit()

        result = test_db.query(Note).first()
        assert result.title == "My Research Note"
        assert result.tags == ["ml", "nlp"]
        assert result.paper_id is None  # Free-form note

    def test_note_linked_to_paper(self, test_db):
        paper = Paper(title="Test", file_path="/tmp/t.pdf", file_hash="h3")
        test_db.add(paper)
        test_db.commit()

        note = Note(title="Paper Note", content="Notes about this paper", paper_id=paper.id)
        test_db.add(note)
        test_db.commit()

        result = test_db.query(Note).first()
        assert result.paper_id == paper.id


class TestCitationModel:
    def test_create_citation(self, test_db):
        paper = Paper(title="Test", file_path="/tmp/t.pdf", file_hash="h4")
        test_db.add(paper)
        test_db.commit()

        citation = Citation(
            paper_id=paper.id,
            raw_text="Smith, J. (2023). Paper Title. Journal.",
            parsed_title="Paper Title",
            parsed_authors="Smith, J.",
            parsed_year=2023,
        )
        test_db.add(citation)
        test_db.commit()

        result = test_db.query(Citation).first()
        assert result.parsed_title == "Paper Title"
        assert result.parsed_year == 2023


class TestResearchSessionModel:
    def test_create_session(self, test_db):
        rs = ResearchSession(
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            source_papers=[{"paper_id": 1, "page_number": 3}],
            model_used="gpt-4",
            tokens_used=150,
        )
        test_db.add(rs)
        test_db.commit()

        result = test_db.query(ResearchSession).first()
        assert result.query == "What is machine learning?"
        assert result.tokens_used == 150
        assert len(result.source_papers) == 1


class TestReportModel:
    def test_create_report(self, test_db):
        report = Report(
            title="Literature Review",
            content="# Report\n\nContent here",
            format=ReportFormat.MARKDOWN,
            included_papers=[1, 2, 3],
        )
        test_db.add(report)
        test_db.commit()

        result = test_db.query(Report).first()
        assert result.title == "Literature Review"
        assert result.format == ReportFormat.MARKDOWN
        assert result.included_papers == [1, 2, 3]
