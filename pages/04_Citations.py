"""Citation extraction and cross-document comparison."""

import streamlit as st
import pandas as pd

from core.citation_extractor import CitationExtractor
from core.cross_compare import CrossComparer
from db.database import init_db, get_db_session
from db.models import Paper, Citation, PaperStatus, Note
from utils.logging_config import get_logger

logger = get_logger(__name__)
init_db()

st.set_page_config(page_title="Citations", page_icon="📖", layout="wide")
st.title("📖 Citations & Cross-Document Analysis")
st.caption("Extract references and compare papers")

tab_extract, tab_browse, tab_compare = st.tabs(
    ["Extract Citations", "Browse Citations", "Cross-Document Comparison"]
)

with tab_extract:
    st.subheader("Extract Citations from Papers")

    with get_db_session() as session:
        papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
        paper_options = {f"{p.title} (ID: {p.id})": p.id for p in papers}

    if not paper_options:
        st.warning("No processed papers available. Upload papers first.")
    else:
        selected_paper = st.selectbox(
            "Select a paper",
            options=list(paper_options.keys()),
            key="citation_extract",
        )

        if st.button("Extract Citations", type="primary"):
            paper_id = paper_options[selected_paper]
            with st.spinner("Extracting and parsing citations... This may take a moment."):
                try:
                    extractor = CitationExtractor()
                    citations = extractor.extract_citations(paper_id)

                    if citations:
                        st.success(f"✅ Extracted {len(citations)} citations")

                        df = pd.DataFrame(citations)
                        display_cols = ["parsed_title", "parsed_authors", "year", "doi"]
                        available_cols = [c for c in display_cols if c in df.columns]
                        st.dataframe(
                            df[available_cols] if available_cols else df,
                            use_container_width=True,
                        )

                        # Check cross-references
                        cross_refs = extractor.get_cross_references(paper_id)
                        if cross_refs:
                            st.subheader("🔗 Cross-References Found")
                            st.markdown(
                                "These citations match papers in your library:"
                            )
                            for ref in cross_refs:
                                st.markdown(
                                    f"- Cites: **{ref['cited_paper_title']}** "
                                    f"(Paper ID: {ref['cited_paper_id']})"
                                )
                    else:
                        st.info("No citations found in this paper's reference section.")
                except Exception as e:
                    logger.exception("Citation extraction failed")
                    st.error(f"Extraction failed: {str(e)}")

with tab_browse:
    st.subheader("All Extracted Citations")

    # Filter by paper
    with get_db_session() as session:
        papers_with_citations = (
            session.query(Paper)
            .join(Citation)
            .distinct()
            .all()
        )
        filter_options = {"All Papers": None}
        filter_options.update({p.title: p.id for p in papers_with_citations})

    selected_filter = st.selectbox(
        "Filter by paper",
        options=list(filter_options.keys()),
        key="citation_filter",
    )

    search_term = st.text_input("Search citations", placeholder="Search by title, author, or DOI...")

    with get_db_session() as session:
        query = session.query(Citation)
        if filter_options[selected_filter]:
            query = query.filter(Citation.paper_id == filter_options[selected_filter])
        citations = query.order_by(Citation.id.desc()).all()

        if search_term:
            term = search_term.lower()
            citations = [
                c for c in citations
                if term in (c.parsed_title or "").lower()
                or term in (c.parsed_authors or "").lower()
                or term in (c.parsed_doi or "").lower()
            ]

        if citations:
            data = []
            for c in citations:
                paper = session.query(Paper).filter(Paper.id == c.paper_id).first()
                data.append({
                    "Source Paper": paper.title if paper else f"Paper {c.paper_id}",
                    "Cited Title": c.parsed_title or "Unknown",
                    "Authors": c.parsed_authors or "Unknown",
                    "Year": c.parsed_year or "",
                    "DOI": c.parsed_doi or "",
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, height=400)
            st.caption(f"Showing {len(citations)} citation(s)")
        else:
            st.info("No citations found. Extract citations from papers first.")

with tab_compare:
    st.subheader("Cross-Document Comparison")
    st.markdown("Select 2-5 papers to compare their methodologies, findings, and gaps.")

    with get_db_session() as session:
        papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
        compare_options = {f"{p.title} (ID: {p.id})": p.id for p in papers}

    selected_compare = st.multiselect(
        "Select papers to compare",
        options=list(compare_options.keys()),
        max_selections=5,
        key="compare_papers",
    )

    if len(selected_compare) < 2:
        st.info("Select at least 2 papers to compare.")

    if st.button("Compare Papers", type="primary") and len(selected_compare) >= 2:
        paper_ids = [compare_options[p] for p in selected_compare]
        with st.spinner("Analyzing and comparing papers..."):
            try:
                comparer = CrossComparer()
                comparison = comparer.compare_papers(paper_ids)
                st.markdown(comparison)

                # Save as note
                if st.button("💾 Save as Note"):
                    try:
                        with get_db_session() as session:
                            note = Note(
                                title=f"Comparison: {', '.join(p.split(' (ID')[0] for p in selected_compare[:3])}",
                                content=comparison,
                                tags=["comparison", "cross-document"],
                            )
                            session.add(note)
                        st.success("Comparison saved as a note!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

            except Exception as e:
                logger.exception("Comparison failed")
                st.error(f"Comparison failed: {str(e)}")
