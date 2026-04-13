"""RAG pipeline using LangChain, FAISS, and OpenAI."""

import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from db.database import get_db_session
from db.models import Chunk, Paper, ResearchSession
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Manages the RAG pipeline: embeddings, FAISS index, and retrieval QA."""

    def __init__(self):
        self.settings = get_settings()
        self._embeddings = None
        self._vectorstore = None
        self._llm = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                openai_api_key=self.settings.openai_api_key,
            )
        return self._embeddings

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name=self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                temperature=0.1,
            )
        return self._llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts with retry logic."""
        logger.info("Generating embeddings for %d texts", len(texts))
        return self.embeddings.embed_documents(texts)

    def add_paper_chunks(self, paper_id: int, chunks: list[dict]) -> None:
        """
        Generate embeddings for paper chunks and add to FAISS index.

        Updates chunk records with embedding_id.
        """
        if not chunks:
            logger.warning("No chunks to add for paper %d", paper_id)
            return

        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)

        # Build documents for LangChain FAISS
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    "paper_id": paper_id,
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                },
            )
            documents.append(doc)

        # Load existing or create new vectorstore
        index_path = Path(self.settings.faiss_index_path)
        if (index_path / "index.faiss").exists():
            self._vectorstore = LangChainFAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._vectorstore.add_documents(documents)
        else:
            self._vectorstore = LangChainFAISS.from_documents(
                documents, self.embeddings
            )

        # Save to disk
        index_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(index_path))

        # Update chunk records with embedding IDs
        with get_db_session() as session:
            db_chunks = (
                session.query(Chunk)
                .filter(Chunk.paper_id == paper_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            for db_chunk in db_chunks:
                db_chunk.embedding_id = f"faiss_{paper_id}_{db_chunk.chunk_index}"

        logger.info("Added %d chunks to FAISS index for paper %d", len(chunks), paper_id)

    def load_vectorstore(self) -> Optional[LangChainFAISS]:
        """Load the FAISS vectorstore from disk."""
        index_path = Path(self.settings.faiss_index_path)
        if not (index_path / "index.faiss").exists():
            logger.warning("No FAISS index found at %s", index_path)
            return None
        self._vectorstore = LangChainFAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self._vectorstore

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Run a RAG query: embed question, retrieve chunks, generate answer.

        Returns dict with keys: answer, sources, tokens_used
        """
        top_k = top_k or self.settings.rag_top_k

        if self._vectorstore is None:
            self.load_vectorstore()

        if self._vectorstore is None:
            return {
                "answer": "No documents have been indexed yet. Please upload some papers first.",
                "sources": [],
                "tokens_used": 0,
            }

        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])

        # Gather source info
        sources = []
        seen = set()
        for doc in source_docs:
            paper_id = doc.metadata.get("paper_id")
            page_num = doc.metadata.get("page_number", 0)
            key = (paper_id, page_num)
            if key not in seen:
                seen.add(key)
                # Look up paper title
                with get_db_session() as session:
                    paper = session.query(Paper).filter(Paper.id == paper_id).first()
                    title = paper.title if paper else f"Paper {paper_id}"
                sources.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page_number": page_num,
                    "chunk_text": doc.page_content[:200] + "...",
                })

        # Estimate tokens (rough)
        tokens_used = len(question.split()) + len(answer.split()) * 2

        # Save to research sessions
        with get_db_session() as session:
            research_session = ResearchSession(
                query=question,
                response=answer,
                source_papers=[
                    {"paper_id": s["paper_id"], "page_number": s["page_number"]}
                    for s in sources
                ],
                model_used=self.settings.openai_model,
                tokens_used=tokens_used,
            )
            session.add(research_session)

        logger.info("RAG query completed: %d sources, ~%d tokens", len(sources), tokens_used)

        return {
            "answer": answer,
            "sources": sources,
            "tokens_used": tokens_used,
        }

    def delete_paper_from_index(self, paper_id: int) -> None:
        """Rebuild the FAISS index excluding chunks from a specific paper."""
        with get_db_session() as session:
            remaining_chunks = (
                session.query(Chunk)
                .filter(Chunk.paper_id != paper_id)
                .all()
            )

        if not remaining_chunks:
            # No documents left, remove index
            index_path = Path(self.settings.faiss_index_path)
            for f in index_path.glob("*"):
                f.unlink()
            self._vectorstore = None
            logger.info("FAISS index cleared after removing paper %d", paper_id)
            return

        documents = []
        for chunk in remaining_chunks:
            doc = Document(
                page_content=chunk.chunk_text,
                metadata={
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                },
            )
            documents.append(doc)

        self._vectorstore = LangChainFAISS.from_documents(documents, self.embeddings)
        index_path = Path(self.settings.faiss_index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(index_path))
        logger.info("Rebuilt FAISS index after removing paper %d", paper_id)
