"""Tests for RAG pipeline."""

from unittest.mock import patch, MagicMock

import pytest

from core.rag_pipeline import RAGPipeline


class TestRAGPipeline:
    def test_init(self):
        rag = RAGPipeline()
        assert rag._embeddings is None
        assert rag._vectorstore is None
        assert rag._llm is None

    @patch("core.rag_pipeline.OpenAIEmbeddings")
    def test_embeddings_property(self, mock_embeddings_cls):
        rag = RAGPipeline()
        _ = rag.embeddings
        mock_embeddings_cls.assert_called_once()

    @patch("core.rag_pipeline.ChatOpenAI")
    def test_llm_property(self, mock_llm_cls):
        rag = RAGPipeline()
        _ = rag.llm
        mock_llm_cls.assert_called_once()

    @patch("core.rag_pipeline.OpenAIEmbeddings")
    def test_generate_embeddings(self, mock_embeddings_cls):
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings_cls.return_value = mock_instance

        rag = RAGPipeline()
        result = rag.generate_embeddings(["text1", "text2"])
        assert len(result) == 2
        mock_instance.embed_documents.assert_called_once_with(["text1", "text2"])

    def test_query_no_index(self):
        """Query should return helpful message when no index exists."""
        rag = RAGPipeline()
        rag._vectorstore = None

        with patch.object(rag, "load_vectorstore", return_value=None):
            result = rag.query("test question")
            assert "No documents" in result["answer"]
            assert result["sources"] == []


class TestRAGPipelineAddChunks:
    @patch("core.rag_pipeline.LangChainFAISS", create=True)
    @patch("core.rag_pipeline.OpenAIEmbeddings")
    @patch("core.rag_pipeline.get_db_session")
    def test_add_paper_chunks_new_index(
        self, mock_db, mock_embeddings_cls, mock_faiss_cls
    ):
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_embeddings_cls.return_value = mock_embeddings_instance

        mock_vectorstore = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vectorstore

        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        rag = RAGPipeline()
        chunks = [{"chunk_text": "test", "chunk_index": 0, "page_number": 1}]
        rag.add_paper_chunks(1, chunks)

        mock_vectorstore.save_local.assert_called_once()

    def test_add_empty_chunks(self):
        rag = RAGPipeline()
        rag.add_paper_chunks(1, [])  # Should not raise
