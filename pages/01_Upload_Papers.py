"""Upload and process PDF research papers."""

import streamlit as st

from core.pdf_processor import (
    save_uploaded_file,
    extract_text_from_pdf,
    extract_metadata_from_pdf,
    get_page_count,
    chunk_text,
    compute_file_hash,
)
from core.rag_pipeline import RAGPipeline
from db.database import init_db, get_db_session
from db.models import Paper, Chunk, PaperStatus
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()
init_db()

st.set_page_config(page_title="Upload Papers", page_icon="📄", layout="wide")
st.title("📄 Upload Research Papers")
st.caption("Upload PDF documents to build your research library")

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF research papers",
)

if uploaded_files:
    st.info(f"Selected {len(uploaded_files)} file(s) for upload")

    if st.button("Process Papers", type="primary"):
        rag = RAGPipeline()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            status_text.text(f"Processing {file_name} ({i+1}/{len(uploaded_files)})...")

            try:
                # Save file
                file_path = save_uploaded_file(uploaded_file, file_name)
                file_hash = compute_file_hash(file_path)

                # Check for duplicates
                with get_db_session() as session:
                    existing = session.query(Paper).filter(Paper.file_hash == file_hash).first()
                    if existing:
                        st.warning(f"⚠️ **{file_name}** already exists as '{existing.title}'. Skipping.")
                        continue

                # Extract metadata
                metadata = extract_metadata_from_pdf(file_path)
                page_count = get_page_count(file_path)
                title = metadata["title"] or file_name.replace(".pdf", "")
                authors = metadata["authors"]

                # Create paper record
                with get_db_session() as session:
                    paper = Paper(
                        title=title,
                        authors=authors,
                        file_path=file_path,
                        file_hash=file_hash,
                        page_count=page_count,
                        status=PaperStatus.PROCESSING,
                    )
                    session.add(paper)
                    session.flush()
                    paper_id = paper.id

                # Extract and chunk text
                pages = extract_text_from_pdf(file_path)
                if not pages:
                    with get_db_session() as session:
                        paper = session.query(Paper).filter(Paper.id == paper_id).first()
                        paper.status = PaperStatus.ERROR
                        paper.error_message = "No text could be extracted from PDF"
                    st.error(f"❌ **{file_name}**: No text extracted")
                    continue

                chunks = chunk_text(pages)

                # Extract abstract (first ~500 words)
                full_text = " ".join(p["text"] for p in pages)
                abstract = " ".join(full_text.split()[:500])

                # Save chunks to DB
                with get_db_session() as session:
                    paper = session.query(Paper).filter(Paper.id == paper_id).first()
                    paper.abstract = abstract
                    for chunk_data in chunks:
                        chunk = Chunk(
                            paper_id=paper_id,
                            chunk_text=chunk_data["chunk_text"],
                            chunk_index=chunk_data["chunk_index"],
                            page_number=chunk_data["page_number"],
                        )
                        session.add(chunk)

                # Generate embeddings and add to FAISS
                rag.add_paper_chunks(paper_id, chunks)

                # Mark as ready
                with get_db_session() as session:
                    paper = session.query(Paper).filter(Paper.id == paper_id).first()
                    paper.status = PaperStatus.READY

                st.success(f"✅ **{title}** — {page_count} pages, {len(chunks)} chunks indexed")

            except Exception as e:
                logger.exception("Failed to process %s", file_name)
                st.error(f"❌ **{file_name}**: {str(e)}")
                # Mark as error if paper was created
                try:
                    with get_db_session() as session:
                        paper = session.query(Paper).filter(Paper.file_hash == file_hash).first()
                        if paper:
                            paper.status = PaperStatus.ERROR
                            paper.error_message = str(e)
                except Exception:
                    pass

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("Processing complete!")

st.divider()

# Paper library
st.subheader("📚 Paper Library")

with get_db_session() as session:
    papers = session.query(Paper).order_by(Paper.upload_date.desc()).all()

    if papers:
        for paper in papers:
            status_icon = {
                "ready": "✅",
                "processing": "⏳",
                "error": "❌",
            }.get(paper.status.value if paper.status else "processing", "❓")

            with st.expander(f"{status_icon} {paper.title}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Authors:** {paper.authors or 'Unknown'}")
                    st.markdown(f"**Pages:** {paper.page_count}")
                    st.markdown(f"**Uploaded:** {paper.upload_date.strftime('%Y-%m-%d %H:%M') if paper.upload_date else 'Unknown'}")
                    if paper.abstract:
                        st.markdown(f"**Abstract preview:** {paper.abstract[:300]}...")
                    if paper.error_message:
                        st.error(f"Error: {paper.error_message}")
                with col2:
                    chunk_count = session.query(Chunk).filter(Chunk.paper_id == paper.id).count()
                    st.metric("Chunks", chunk_count)

                    if st.button("🗑️ Delete", key=f"del_{paper.id}"):
                        try:
                            rag = RAGPipeline()
                            rag.delete_paper_from_index(paper.id)
                            session.delete(paper)
                            session.commit()
                            st.success("Paper deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
    else:
        st.info("No papers in your library yet. Upload some PDFs above!")
