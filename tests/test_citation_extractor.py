"""Tests for citation extraction module."""

from unittest.mock import patch, MagicMock

import pytest

from core.citation_extractor import CitationExtractor


class TestCitationExtractor:
    def test_init(self):
        extractor = CitationExtractor()
        assert extractor._llm is None

    def test_find_reference_section_found(self):
        extractor = CitationExtractor()
        text = """
Introduction
Some text here.

Methods
More text.

References
[1] Smith, J. (2023). Paper Title. Journal, 1(1), 1-10.
[2] Doe, A. (2022). Another Paper. Review, 5(2), 20-30.
"""
        result = extractor.find_reference_section(text)
        assert "References" in result
        assert "Smith" in result

    def test_find_reference_section_bibliography(self):
        extractor = CitationExtractor()
        text = """Content here.

Bibliography
[1] Author, A. (2021). Title. Journal.
"""
        result = extractor.find_reference_section(text)
        assert "Bibliography" in result

    def test_find_reference_section_not_found(self):
        extractor = CitationExtractor()
        text = "This paper has no references section at all."
        result = extractor.find_reference_section(text)
        assert result == ""

    def test_split_references_numbered(self):
        extractor = CitationExtractor()
        ref_section = """References
[1] Smith, J. (2023). Deep Learning Fundamentals. AI Journal, 15(2), 45-67.
[2] Doe, A. & Lee, B. (2022). Neural Networks in Practice. ML Review, 8(1), 12-28.
[3] Johnson, K. (2021). Transformer Architectures. NLP Conference, 100-115.
"""
        refs = extractor.split_references(ref_section)
        assert len(refs) >= 2

    def test_split_references_empty(self):
        extractor = CitationExtractor()
        refs = extractor.split_references("References\n")
        assert refs == []

    @patch("core.citation_extractor.ChatOpenAI")
    def test_parse_citation_with_llm(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"title": "Deep Learning", "authors": "Smith, J.", "year": 2023, "doi": null}'
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        extractor = CitationExtractor()
        result = extractor.parse_citation_with_llm(
            "Smith, J. (2023). Deep Learning. AI Journal."
        )

        assert result["title"] == "Deep Learning"
        assert result["authors"] == "Smith, J."
        assert result["year"] == 2023

    @patch("core.citation_extractor.ChatOpenAI")
    def test_parse_citation_invalid_json(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON"
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        extractor = CitationExtractor()
        result = extractor.parse_citation_with_llm("Some citation")

        assert result["title"] == ""
        assert result["authors"] == ""

    @patch("core.citation_extractor.get_db_session")
    def test_extract_citations_paper_not_found(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        extractor = CitationExtractor()
        with pytest.raises(ValueError, match="not found"):
            extractor.extract_citations(999)
