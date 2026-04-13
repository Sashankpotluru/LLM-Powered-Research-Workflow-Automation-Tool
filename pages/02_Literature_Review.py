"""RAG-powered literature review and Q&A interface."""

import streamlit as st

from core.rag_pipeline import RAGPipeline
from core.summarizer import Summarizer
from db.database import init_db, get_db_session
from db.models import Paper, PaperStatus, ResearchSession
from utils.logging_config import get_logger

logger = get_logger(__name__)
init_db()

st.set_page_config(page_title="Literature Review", page_icon="🔍", layout="wide")
st.title("🔍 Literature Review")
st.caption("Ask questions about your research papers using AI-powered retrieval")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Check if papers exist
with get_db_session() as session:
    paper_count = session.query(Paper).filter(Paper.status == PaperStatus.READY).count()

if paper_count == 0:
    st.warning("No processed papers found. Please upload papers first.")
    st.stop()

st.info(f"📚 {paper_count} papers available for querying")

# Tabs for different modes
tab_qa, tab_summarize, tab_history = st.tabs(["Q&A", "Summarize", "History"])

with tab_qa:
    st.subheader("Ask a Research Question")

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("sources"):
                with st.expander("📖 Sources"):
                    for src in entry["sources"]:
                        st.markdown(
                            f"- **{src['paper_title']}** (p. {src['page_number']})"
                        )
                        st.caption(src["chunk_text"])

    # Chat input
    question = st.chat_input("Ask a question about your papers...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing papers..."):
                try:
                    rag = RAGPipeline()
                    result = rag.query(question)

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("📖 Sources"):
                            for src in result["sources"]:
                                st.markdown(
                                    f"- **{src['paper_title']}** (p. {src['page_number']})"
                                )
                                st.caption(src["chunk_text"])

                    st.caption(f"~{result['tokens_used']} tokens used")

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as e:
                    logger.exception("RAG query failed")
                    st.error(f"Query failed: {str(e)}")

    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

with tab_summarize:
    st.subheader("Paper Summarization")

    col_single, col_multi = st.columns(2)

    with col_single:
        st.markdown("### Single Paper Summary")
        with get_db_session() as session:
            papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
            paper_options = {f"{p.title} (ID: {p.id})": p.id for p in papers}

        selected_paper = st.selectbox(
            "Select a paper to summarize",
            options=list(paper_options.keys()),
            key="single_summary",
        )

        if st.button("Generate Summary", type="primary"):
            if selected_paper:
                paper_id = paper_options[selected_paper]
                with st.spinner("Generating summary..."):
                    try:
                        summarizer = Summarizer()
                        summary = summarizer.summarize_paper(paper_id)
                        st.markdown(summary)
                    except Exception as e:
                        logger.exception("Summarization failed")
                        st.error(f"Summarization failed: {str(e)}")

    with col_multi:
        st.markdown("### Multi-Paper Synthesis")
        with get_db_session() as session:
            papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
            paper_options_multi = {f"{p.title} (ID: {p.id})": p.id for p in papers}

        selected_papers = st.multiselect(
            "Select papers to synthesize",
            options=list(paper_options_multi.keys()),
            key="multi_synthesis",
        )

        topic = st.text_input("Focus topic (optional)", key="synthesis_topic")

        if st.button("Generate Synthesis") and selected_papers:
            paper_ids = [paper_options_multi[p] for p in selected_papers]
            with st.spinner("Synthesizing papers..."):
                try:
                    summarizer = Summarizer()
                    synthesis = summarizer.synthesize_papers(paper_ids, topic)
                    st.markdown(synthesis)
                except Exception as e:
                    logger.exception("Synthesis failed")
                    st.error(f"Synthesis failed: {str(e)}")

with tab_history:
    st.subheader("Query History")

    with get_db_session() as session:
        sessions = (
            session.query(ResearchSession)
            .order_by(ResearchSession.created_at.desc())
            .limit(20)
            .all()
        )

        if sessions:
            for s in sessions:
                with st.expander(
                    f"🔍 {s.query[:80]}{'...' if len(s.query) > 80 else ''}"
                ):
                    st.markdown(f"**Query:** {s.query}")
                    st.markdown(f"**Response:** {s.response}")
                    st.caption(
                        f"Model: {s.model_used} | Tokens: ~{s.tokens_used} | "
                        f"Time: {s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else 'Unknown'}"
                    )
        else:
            st.info("No query history yet. Start asking questions!")
