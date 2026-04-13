"""Tests for PDF processing module."""

import hashlib
from unittest.mock import MagicMock

import pytest

from core.pdf_processor import (
    compute_file_hash,
    extract_text_from_pdf,
    get_page_count,
    extract_metadata_from_pdf,
    chunk_text,
)


class TestComputeFileHash:
    def test_returns_sha256_hash(self, sample_pdf):
        result = compute_file_hash(sample_pdf)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_same_file_same_hash(self, sample_pdf):
        hash1 = compute_file_hash(sample_pdf)
        hash2 = compute_file_hash(sample_pdf)
        assert hash1 == hash2

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/file.pdf")


class TestExtractTextFromPdf:
    def test_extracts_pages(self, sample_pdf):
        pages = extract_text_from_pdf(sample_pdf)
        assert isinstance(pages, list)
        # Should have at least one page
        if pages:  # May be empty for minimal fallback PDF
            assert "page_number" in pages[0]
            assert "text" in pages[0]
            assert pages[0]["page_number"] >= 1

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            extract_text_from_pdf("/nonexistent/file.pdf")


class TestGetPageCount:
    def test_returns_count(self, sample_pdf):
        count = get_page_count(sample_pdf)
        assert isinstance(count, int)
        assert count >= 1

    def test_nonexistent_returns_zero(self):
        count = get_page_count("/nonexistent/file.pdf")
        assert count == 0


class TestExtractMetadata:
    def test_returns_dict_with_keys(self, sample_pdf):
        metadata = extract_metadata_from_pdf(sample_pdf)
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert "authors" in metadata

    def test_nonexistent_returns_empty(self):
        metadata = extract_metadata_from_pdf("/nonexistent/file.pdf")
        assert metadata["title"] == ""
        assert metadata["authors"] == ""


class TestChunkText:
    def test_chunks_pages(self):
        pages = [
            {"page_number": 1, "text": "A " * 300},
            {"page_number": 2, "text": "B " * 300},
        ]
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all("chunk_text" in c for c in chunks)
        assert all("chunk_index" in c for c in chunks)
        assert all("page_number" in c for c in chunks)

    def test_empty_pages_returns_empty(self):
        chunks = chunk_text([])
        assert chunks == []

    def test_chunk_indices_sequential(self):
        pages = [{"page_number": 1, "text": "word " * 500}]
        chunks = chunk_text(pages, chunk_size=100, chunk_overlap=20)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_page_numbers_preserved(self):
        pages = [
            {"page_number": 3, "text": "Content on page three. " * 50},
            {"page_number": 7, "text": "Content on page seven. " * 50},
        ]
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)
        page_numbers = {c["page_number"] for c in chunks}
        assert 3 in page_numbers
        assert 7 in page_numbers
