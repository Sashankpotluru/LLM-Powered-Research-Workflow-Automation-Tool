"""
LLM-Powered Research Workflow Automation Tool

Main Streamlit application entry point.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

from db.database import init_db, get_db_session
from db.models import Paper, Note, ResearchSession, PaperStatus
from utils.config import get_settings
from utils.logging_config import setup_logging, get_logger

# Initialize
settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)
init_db()

# Page config
st.set_page_config(
    page_title="Research Workflow Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .stMetric > div {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Research Workflow Automation Tool</p>', unsafe_allow_html=True)
st.caption("LLM-powered research assistant for literature review, summarization, and analysis")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
    st.markdown("### Navigation")
    st.markdown("""
    - **Home** — Dashboard overview
    - **Upload Papers** — Add PDFs to your library
    - **Literature Review** — RAG-powered Q&A
    - **Notes Manager** — Organize your findings
    - **Citations** — Extract & compare references
    - **Reports** — Generate & export reports
    """)
    st.divider()
    st.markdown(f"**Model:** `{settings.openai_model}`")
    st.markdown(f"**Embeddings:** `{settings.embedding_model}`")

# Dashboard metrics
with get_db_session() as session:
    total_papers = session.query(Paper).count()
    ready_papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).count()
    total_notes = session.query(Note).count()
    total_queries = session.query(ResearchSession).count()

    # Recent papers
    recent_papers = (
        session.query(Paper)
        .order_by(Paper.upload_date.desc())
        .limit(5)
        .all()
    )

    # Recent queries
    recent_queries = (
        session.query(ResearchSession)
        .order_by(ResearchSession.created_at.desc())
        .limit(5)
        .all()
    )

    # Tag distribution
    all_notes = session.query(Note).all()
    tag_counts = {}
    for note in all_notes:
        if note.tags:
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Papers over time
    papers_list = session.query(Paper).order_by(Paper.upload_date).all()

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Papers", total_papers)
with col2:
    st.metric("Processed Papers", ready_papers)
with col3:
    st.metric("Research Notes", total_notes)
with col4:
    st.metric("Queries Made", total_queries)

st.divider()

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Papers Uploaded Over Time")
    if papers_list:
        df_papers = pd.DataFrame([
            {"date": p.upload_date.date() if p.upload_date else datetime.now().date(), "count": 1}
            for p in papers_list
        ])
        df_papers = df_papers.groupby("date").sum().reset_index()
        df_papers["cumulative"] = df_papers["count"].cumsum()
        fig = px.line(
            df_papers,
            x="date",
            y="cumulative",
            title="Cumulative Papers",
            labels={"date": "Date", "cumulative": "Total Papers"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No papers uploaded yet. Go to **Upload Papers** to get started.")

with chart_col2:
    st.subheader("Tag Distribution")
    if tag_counts:
        df_tags = pd.DataFrame(
            [{"tag": k, "count": v} for k, v in tag_counts.items()]
        ).sort_values("count", ascending=True)
        fig = px.bar(
            df_tags,
            x="count",
            y="tag",
            orientation="h",
            title="Notes by Tag",
            labels={"count": "Count", "tag": "Tag"},
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tagged notes yet. Add tags in the **Notes Manager**.")

st.divider()

# Recent activity
activity_col1, activity_col2 = st.columns(2)

with activity_col1:
    st.subheader("Recent Papers")
    if recent_papers:
        for paper in recent_papers:
            status_icon = {"ready": "✅", "processing": "⏳", "error": "❌"}.get(
                paper.status.value if paper.status else "processing", "❓"
            )
            st.markdown(f"{status_icon} **{paper.title}** — {paper.authors[:50] if paper.authors else 'Unknown'}")
    else:
        st.info("No papers uploaded yet.")

with activity_col2:
    st.subheader("Recent Queries")
    if recent_queries:
        for q in recent_queries:
            st.markdown(f"🔍 *{q.query[:80]}...*" if len(q.query) > 80 else f"🔍 *{q.query}*")
            st.caption(f"Sources: {len(q.source_papers)} papers | Tokens: ~{q.tokens_used}")
    else:
        st.info("No queries yet. Try the **Literature Review** page.")

# Research progress gauge
st.divider()
st.subheader("Research Progress")
if total_papers > 0:
    progress = ready_papers / total_papers
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ready_papers,
        delta={"reference": total_papers},
        gauge={
            "axis": {"range": [0, total_papers]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, total_papers * 0.5], "color": "#ffcccc"},
                {"range": [total_papers * 0.5, total_papers * 0.8], "color": "#ffffcc"},
                {"range": [total_papers * 0.8, total_papers], "color": "#ccffcc"},
            ],
        },
        title={"text": "Papers Processed"},
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload papers to track your research progress.")
