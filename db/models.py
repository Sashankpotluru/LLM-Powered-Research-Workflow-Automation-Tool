"""SQLAlchemy ORM models for the research workflow tool."""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Enum,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class PaperStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class ReportFormat(str, enum.Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    authors = Column(String(1000), default="")
    abstract = Column(Text, default="")
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False)
    upload_date = Column(DateTime, default=utcnow)
    page_count = Column(Integer, default=0)
    status = Column(Enum(PaperStatus), default=PaperStatus.PROCESSING)
    error_message = Column(Text, nullable=True)

    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="paper")
    citations = relationship("Citation", back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, default=0)
    embedding_id = Column(String(100), default="")

    paper = relationship("Paper", back_populates="chunks")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, default="")
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    paper = relationship("Paper", back_populates="notes")


class Citation(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    raw_text = Column(Text, nullable=False)
    parsed_title = Column(String(500), default="")
    parsed_authors = Column(String(1000), default="")
    parsed_year = Column(Integer, nullable=True)
    parsed_doi = Column(String(200), nullable=True)

    paper = relationship("Paper", back_populates="citations")


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    source_papers = Column(JSON, default=list)
    model_used = Column(String(50), default="gpt-4")
    tokens_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    format = Column(Enum(ReportFormat), default=ReportFormat.MARKDOWN)
    included_papers = Column(JSON, default=list)
    created_at = Column(DateTime, default=utcnow)
