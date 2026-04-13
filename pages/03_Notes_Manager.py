"""Structured notes management with tagging."""

import streamlit as st
from datetime import datetime, timezone

from db.database import init_db, get_db_session
from db.models import Note, Paper, PaperStatus
from utils.logging_config import get_logger

logger = get_logger(__name__)
init_db()

st.set_page_config(page_title="Notes Manager", page_icon="📝", layout="wide")
st.title("📝 Notes Manager")
st.caption("Organize your research findings with structured notes and tags")

# Tabs
tab_create, tab_browse, tab_search = st.tabs(["Create Note", "Browse Notes", "Search"])

with tab_create:
    st.subheader("Create New Note")

    with st.form("new_note_form"):
        title = st.text_input("Note Title", placeholder="Key findings from...")

        # Optional paper link
        with get_db_session() as session:
            papers = session.query(Paper).filter(Paper.status == PaperStatus.READY).all()
            paper_options = {"(No linked paper)": None}
            paper_options.update({f"{p.title}": p.id for p in papers})

        linked_paper = st.selectbox(
            "Link to Paper (optional)",
            options=list(paper_options.keys()),
        )

        content = st.text_area(
            "Note Content (Markdown supported)",
            height=300,
            placeholder="Write your research notes here...\n\n## Key Points\n- Finding 1\n- Finding 2",
        )

        tags_input = st.text_input(
            "Tags (comma-separated)",
            placeholder="machine-learning, nlp, transformers",
        )

        submitted = st.form_submit_button("Save Note", type="primary")

        if submitted and title:
            tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
            paper_id = paper_options[linked_paper]

            try:
                with get_db_session() as session:
                    note = Note(
                        title=title,
                        content=content,
                        tags=tags,
                        paper_id=paper_id,
                    )
                    session.add(note)
                st.success(f"✅ Note '{title}' saved!")
                st.rerun()
            except Exception as e:
                logger.exception("Failed to save note")
                st.error(f"Failed to save: {str(e)}")
        elif submitted:
            st.warning("Please enter a title for the note.")

with tab_browse:
    st.subheader("All Notes")

    # Filter by tag
    with get_db_session() as session:
        all_notes = session.query(Note).order_by(Note.updated_at.desc()).all()
        all_tags = set()
        for note in all_notes:
            if note.tags:
                all_tags.update(note.tags)

    tag_filter = st.multiselect("Filter by tags", options=sorted(all_tags))

    with get_db_session() as session:
        query = session.query(Note).order_by(Note.updated_at.desc())
        notes = query.all()

        # Client-side tag filtering
        if tag_filter:
            notes = [n for n in notes if n.tags and any(t in n.tags for t in tag_filter)]

        if notes:
            for note in notes:
                with st.expander(
                    f"📝 {note.title} — {', '.join(note.tags) if note.tags else 'No tags'}"
                ):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        # Edit mode
                        edit_key = f"edit_{note.id}"
                        if st.session_state.get(edit_key, False):
                            new_title = st.text_input(
                                "Title", value=note.title, key=f"title_{note.id}"
                            )
                            new_content = st.text_area(
                                "Content",
                                value=note.content,
                                height=200,
                                key=f"content_{note.id}",
                            )
                            new_tags = st.text_input(
                                "Tags",
                                value=", ".join(note.tags) if note.tags else "",
                                key=f"tags_{note.id}",
                            )

                            save_col, cancel_col = st.columns(2)
                            with save_col:
                                if st.button("💾 Save", key=f"save_{note.id}"):
                                    try:
                                        with get_db_session() as edit_session:
                                            db_note = edit_session.query(Note).filter(
                                                Note.id == note.id
                                            ).first()
                                            db_note.title = new_title
                                            db_note.content = new_content
                                            db_note.tags = [
                                                t.strip()
                                                for t in new_tags.split(",")
                                                if t.strip()
                                            ]
                                            db_note.updated_at = datetime.now(timezone.utc)
                                        st.session_state[edit_key] = False
                                        st.success("Note updated!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Update failed: {e}")
                            with cancel_col:
                                if st.button("Cancel", key=f"cancel_{note.id}"):
                                    st.session_state[edit_key] = False
                                    st.rerun()
                        else:
                            st.markdown(note.content or "*No content*")
                            if note.paper_id:
                                paper = session.query(Paper).filter(
                                    Paper.id == note.paper_id
                                ).first()
                                if paper:
                                    st.caption(f"📄 Linked to: {paper.title}")
                            st.caption(
                                f"Created: {note.created_at.strftime('%Y-%m-%d %H:%M') if note.created_at else 'Unknown'} | "
                                f"Updated: {note.updated_at.strftime('%Y-%m-%d %H:%M') if note.updated_at else 'Unknown'}"
                            )

                    with col2:
                        if not st.session_state.get(edit_key, False):
                            if st.button("✏️ Edit", key=f"editbtn_{note.id}"):
                                st.session_state[edit_key] = True
                                st.rerun()

                            if st.button("🗑️ Delete", key=f"delbtn_{note.id}"):
                                try:
                                    with get_db_session() as del_session:
                                        del_note = del_session.query(Note).filter(
                                            Note.id == note.id
                                        ).first()
                                        if del_note:
                                            del_session.delete(del_note)
                                    st.success("Note deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Delete failed: {e}")
        else:
            st.info("No notes found. Create one in the 'Create Note' tab!")

with tab_search:
    st.subheader("Search Notes")

    search_query = st.text_input("Search by keyword", placeholder="Enter search term...")

    if search_query:
        with get_db_session() as session:
            notes = session.query(Note).all()
            # Simple keyword search across title, content, and tags
            results = []
            query_lower = search_query.lower()
            for note in notes:
                if (
                    query_lower in (note.title or "").lower()
                    or query_lower in (note.content or "").lower()
                    or any(query_lower in t.lower() for t in (note.tags or []))
                ):
                    results.append(note)

            st.markdown(f"**{len(results)} result(s) found**")

            for note in results:
                with st.expander(f"📝 {note.title}"):
                    st.markdown(note.content or "*No content*")
                    if note.tags:
                        st.markdown(f"**Tags:** {', '.join(note.tags)}")
