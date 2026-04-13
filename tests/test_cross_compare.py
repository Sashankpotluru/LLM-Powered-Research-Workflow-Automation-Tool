"""Tests for cross-document comparison module."""

from unittest.mock import patch, MagicMock

import pytest
from tenacity import RetryError

from core.cross_compare import CrossComparer


class TestCrossComparer:
    def test_init(self):
        comparer = CrossComparer()
        assert comparer._llm is None

    def test_compare_too_few_papers(self):
        comparer = CrossComparer()
        with pytest.raises(RetryError):
            comparer.compare_papers([1])

    def test_compare_too_many_papers(self):
        comparer = CrossComparer()
        with pytest.raises(RetryError):
            comparer.compare_papers([1, 2, 3, 4, 5, 6])

    @patch("core.cross_compare.ChatOpenAI")
    @patch("core.cross_compare.get_db_session")
    def test_compare_papers_success(self, mock_db, mock_llm_cls):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        mock_paper1 = MagicMock()
        mock_paper1.id = 1
        mock_paper1.title = "Paper A"
        mock_paper1.authors = "Author A"

        mock_paper2 = MagicMock()
        mock_paper2.id = 2
        mock_paper2.title = "Paper B"
        mock_paper2.authors = "Author B"

        mock_chunk1 = MagicMock()
        mock_chunk1.chunk_text = "Content of paper A about neural networks"

        mock_chunk2 = MagicMock()
        mock_chunk2.chunk_text = "Content of paper B about deep learning"

        # Setup query chaining for papers and chunks
        def mock_query_side_effect(model):
            query = MagicMock()
            if model.__name__ == "Paper":
                def filter_side_effect(*args, **kwargs):
                    filter_result = MagicMock()
                    filter_result.first.side_effect = [mock_paper1, mock_paper2]
                    return filter_result
                query.filter = filter_side_effect
            else:
                def filter_side_effect(*args, **kwargs):
                    filter_result = MagicMock()
                    filter_result.order_by.return_value.all.side_effect = [
                        [mock_chunk1], [mock_chunk2]
                    ]
                    return filter_result
                query.filter = filter_side_effect
            return query

        mock_session.query = mock_query_side_effect

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "## Comparison\nPaper A and B both discuss neural networks."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        comparer = CrossComparer()
        result = comparer.compare_papers([1, 2])

        assert "Comparison" in result
        assert isinstance(result, str)

    @patch("core.cross_compare.get_db_session")
    def test_compare_not_enough_valid_papers(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        comparer = CrossComparer()
        with pytest.raises(RetryError):
            comparer.compare_papers([1, 2])
