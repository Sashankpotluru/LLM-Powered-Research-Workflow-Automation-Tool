"""Tests for the summarizer module."""

from unittest.mock import patch, MagicMock

import pytest
from tenacity import RetryError

from core.summarizer import Summarizer


class TestSummarizer:
    def test_init(self):
        summarizer = Summarizer()
        assert summarizer._llm is None

    @patch("core.summarizer.ChatOpenAI")
    def test_llm_property(self, mock_llm_cls):
        summarizer = Summarizer()
        _ = summarizer.llm
        mock_llm_cls.assert_called_once()

    @patch("core.summarizer.get_db_session")
    def test_summarize_paper_not_found(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        summarizer = Summarizer()
        with pytest.raises(RetryError):
            summarizer.summarize_paper(999)

    @patch("core.summarizer.get_db_session")
    def test_synthesize_no_valid_papers(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        summarizer = Summarizer()
        with pytest.raises(RetryError):
            summarizer.synthesize_papers([999, 998])

    @patch("core.summarizer.load_summarize_chain")
    @patch("core.summarizer.ChatOpenAI")
    @patch("core.summarizer.get_db_session")
    def test_summarize_paper_success(self, mock_db, mock_llm_cls, mock_chain_fn):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        mock_paper = MagicMock()
        mock_paper.id = 1
        mock_paper.title = "Test Paper"

        mock_chunk = MagicMock()
        mock_chunk.chunk_text = "Test content about machine learning"
        mock_chunk.chunk_index = 0

        mock_session.query.return_value.filter.return_value.first.return_value = mock_paper
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk]

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output_text": "## Summary\nThis paper discusses ML."}
        mock_chain_fn.return_value = mock_chain

        summarizer = Summarizer()
        result = summarizer.summarize_paper(1)

        assert "Summary" in result
        assert isinstance(result, str)
