"""Report generation and export."""

import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import markdown

from db.database import init_db, get_db_session
from db.models import Paper, Note, ResearchSession, Report, ReportFormat, PaperStatus
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()
init_db()

st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")
st.title("📊 Reports & Export")
st.caption("Generate structured reports from your research")

tab_generate, tab_browse = st.tabs(["Generate Report", "Previous Reports"])

with tab_generate:
    st.subheader("Create New Report")

    report_title = st.text_input("Report Title", placeholder="Literature Review: Topic X")

    # Select content to include
    st.markdown("### Select Content to Include")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Papers**")
        with get_db_session() as session:
            papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
            paper_options = {f"{p.title}": p.id for p in papers}

        selected_papers = st.multiselect(
            "Select papers",
            options=list(paper_options.keys()),
            key="report_papers",
        )

    with col2:
        st.markdown("**Notes**")
        with get_db_session() as session:
            notes = session.query(Note).order_by(Note.updated_at.desc()).all()
            note_options = {f"{n.title}": n.id for n in notes}

        selected_notes = st.multiselect(
            "Select notes",
            options=list(note_options.keys()),
            key="report_notes",
        )

    include_queries = st.checkbox("Include recent research queries", value=False)

    export_format = st.selectbox(
        "Export Format",
        options=["Markdown", "JSON"],
        index=0,
    )

    if st.button("Generate Report", type="primary") and report_title:
        with st.spinner("Generating report..."):
            try:
                report_content = f"# {report_title}\n\n"
                report_content += f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n\n"

                included_paper_ids = []

                # Papers section
                if selected_papers:
                    report_content += "---\n\n## Papers Reviewed\n\n"
                    with get_db_session() as session:
                        for paper_name in selected_papers:
                            paper_id = paper_options[paper_name]
                            included_paper_ids.append(paper_id)
                            paper = session.query(Paper).filter(Paper.id == paper_id).first()
                            if paper:
                                report_content += f"### {paper.title}\n\n"
                                report_content += f"**Authors:** {paper.authors or 'Unknown'}\n\n"
                                report_content += f"**Pages:** {paper.page_count}\n\n"
                                if paper.abstract:
                                    report_content += f"**Abstract:** {paper.abstract[:500]}...\n\n"

                # Notes section
                if selected_notes:
                    report_content += "---\n\n## Research Notes\n\n"
                    with get_db_session() as session:
                        for note_name in selected_notes:
                            note_id = note_options[note_name]
                            note = session.query(Note).filter(Note.id == note_id).first()
                            if note:
                                report_content += f"### {note.title}\n\n"
                                if note.tags:
                                    report_content += f"*Tags: {', '.join(note.tags)}*\n\n"
                                report_content += f"{note.content}\n\n"

                # Queries section
                if include_queries:
                    report_content += "---\n\n## Research Queries & Findings\n\n"
                    with get_db_session() as session:
                        recent = (
                            session.query(ResearchSession)
                            .order_by(ResearchSession.created_at.desc())
                            .limit(10)
                            .all()
                        )
                        for q in recent:
                            report_content += f"**Q:** {q.query}\n\n"
                            report_content += f"**A:** {q.response}\n\n"
                            report_content += "---\n\n"

                # Determine format
                fmt = ReportFormat.MARKDOWN
                if export_format == "JSON":
                    fmt = ReportFormat.JSON
                    report_content = json.dumps({
                        "title": report_title,
                        "generated": datetime.now(timezone.utc).isoformat(),
                        "papers": selected_papers,
                        "notes": selected_notes,
                        "content": report_content,
                    }, indent=2)

                # Save to DB
                with get_db_session() as session:
                    report = Report(
                        title=report_title,
                        content=report_content,
                        format=fmt,
                        included_papers=included_paper_ids,
                    )
                    session.add(report)
                    session.flush()
                    report_id = report.id

                # Save to file
                export_dir = Path(settings.export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)

                ext = ".md" if fmt == ReportFormat.MARKDOWN else ".json"
                safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in report_title)
                file_name = f"{safe_title}_{report_id}{ext}"
                file_path = export_dir / file_name

                with open(file_path, "w") as f:
                    f.write(report_content)

                st.success(f"✅ Report generated and saved!")

                # Preview
                st.subheader("Report Preview")
                if fmt == ReportFormat.MARKDOWN:
                    st.markdown(report_content)
                else:
                    st.json(report_content)

                # Download button
                st.download_button(
                    label=f"⬇️ Download Report ({ext})",
                    data=report_content,
                    file_name=file_name,
                    mime="text/markdown" if fmt == ReportFormat.MARKDOWN else "application/json",
                )

            except Exception as e:
                logger.exception("Report generation failed")
                st.error(f"Report generation failed: {str(e)}")

    elif not report_title and st.button("Generate Report", type="primary", key="gen_no_title"):
        st.warning("Please enter a report title.")

with tab_browse:
    st.subheader("Previous Reports")

    with get_db_session() as session:
        reports = session.query(Report).order_by(Report.created_at.desc()).all()

        if reports:
            for report in reports:
                fmt_icon = {"markdown": "📄", "pdf": "📕", "json": "📋"}.get(
                    report.format.value if report.format else "markdown", "📄"
                )

                with st.expander(
                    f"{fmt_icon} {report.title} — "
                    f"{report.created_at.strftime('%Y-%m-%d %H:%M') if report.created_at else 'Unknown'}"
                ):
                    if report.format == ReportFormat.JSON:
                        st.json(report.content)
                    else:
                        st.markdown(report.content)

                    st.caption(
                        f"Format: {report.format.value if report.format else 'markdown'} | "
                        f"Papers included: {len(report.included_papers) if report.included_papers else 0}"
                    )

                    # Download
                    ext = ".md" if report.format == ReportFormat.MARKDOWN else ".json"
                    st.download_button(
                        label="⬇️ Download",
                        data=report.content,
                        file_name=f"{report.title}{ext}",
                        mime="text/markdown" if report.format == ReportFormat.MARKDOWN else "application/json",
                        key=f"dl_{report.id}",
                    )

                    # Delete
                    if st.button("🗑️ Delete Report", key=f"del_report_{report.id}"):
                        try:
                            with get_db_session() as del_session:
                                del_report = del_session.query(Report).filter(
                                    Report.id == report.id
                                ).first()
                                if del_report:
                                    del_session.delete(del_report)
                            st.success("Report deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.info("No reports generated yet. Create one in the 'Generate Report' tab!")
