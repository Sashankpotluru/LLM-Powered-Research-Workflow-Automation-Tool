"""Automated citation extraction using regex + GPT-4."""

import re
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from db.database import get_db_session
from db.models import Paper, Chunk, Citation
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Patterns to identify reference sections
REFERENCE_SECTION_PATTERNS = [
    r"(?i)\n\s*references?\s*\n",
    r"(?i)\n\s*bibliography\s*\n",
    r"(?i)\n\s*works?\s+cited\s*\n",
    r"(?i)\n\s*literature\s+cited\s*\n",
]

# Pattern to split individual references
REFERENCE_ENTRY_PATTERN = re.compile(
    r"(?:^\s*\[?\d+\]?\s*\.?\s*)|(?:^\s*•\s*)",
    re.MULTILINE,
)

CITATION_PARSE_SYSTEM_PROMPT = """You are a citation parser. Given a raw citation string from an academic paper, extract structured metadata.

Return a JSON object with these fields:
- title: the paper/book title (string)
- authors: comma-separated author names (string)
- year: publication year (integer or null)
- doi: DOI if present (string or null)

If a field cannot be determined, use an empty string for strings or null for year/doi.
Return ONLY the JSON object, no other text."""


class CitationExtractor:
    """Extracts and parses citations from research papers."""

    def __init__(self):
        self.settings = get_settings()
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name=self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                temperature=0.0,
            )
        return self._llm

    def find_reference_section(self, full_text: str) -> str:
        """
        Extract the references/bibliography section from the paper text.

        Returns the reference section text, or empty string if not found.
        """
        for pattern in REFERENCE_SECTION_PATTERNS:
            match = re.search(pattern, full_text)
            if match:
                ref_text = full_text[match.start():]
                # Trim to reasonable length (references usually < 5000 chars)
                if len(ref_text) > 10000:
                    ref_text = ref_text[:10000]
                return ref_text.strip()
        return ""

    def split_references(self, ref_section: str) -> list[str]:
        """Split a reference section into individual citation strings."""
        # Remove the section header
        lines = ref_section.split("\n")
        if lines:
            lines = lines[1:]  # Skip the "References" header

        ref_text = "\n".join(lines).strip()
        if not ref_text:
            return []

        # Try numbered references first [1], [2], etc.
        numbered = re.split(r"\n\s*\[?\d+\]?\s*\.?\s+", ref_text)
        if len(numbered) > 2:
            return [r.strip() for r in numbered if r.strip() and len(r.strip()) > 20]

        # Fall back to splitting by double newlines or blank lines
        entries = re.split(r"\n\s*\n", ref_text)
        if len(entries) > 1:
            return [r.strip() for r in entries if r.strip() and len(r.strip()) > 20]

        # Last resort: split by newlines and group
        return [r.strip() for r in ref_text.split("\n") if r.strip() and len(r.strip()) > 30]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def parse_citation_with_llm(self, raw_citation: str) -> dict:
        """
        Use GPT-4 to parse a raw citation string into structured fields.

        Returns dict with keys: title, authors, year, doi
        """
        messages = [
            SystemMessage(content=CITATION_PARSE_SYSTEM_PROMPT),
            HumanMessage(content=f"Parse this citation:\n{raw_citation}"),
        ]

        response = self.llm.invoke(messages)
        content = response.content.strip()

        # Extract JSON from response
        try:
            # Handle markdown code blocks
            if "```" in content:
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM citation response: %s", content[:200])
            parsed = {"title": "", "authors": "", "year": None, "doi": None}

        return {
            "title": parsed.get("title", ""),
            "authors": parsed.get("authors", ""),
            "year": parsed.get("year"),
            "doi": parsed.get("doi"),
        }

    def extract_citations(self, paper_id: int) -> list[dict]:
        """
        Extract all citations from a paper.

        Returns list of parsed citation dicts.
        """
        with get_db_session() as session:
            paper = session.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                raise ValueError(f"Paper {paper_id} not found")

            chunks = (
                session.query(Chunk)
                .filter(Chunk.paper_id == paper_id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            full_text = "\n\n".join(c.chunk_text for c in chunks)

        ref_section = self.find_reference_section(full_text)
        if not ref_section:
            logger.info("No reference section found for paper %d", paper_id)
            return []

        raw_refs = self.split_references(ref_section)
        logger.info("Found %d raw references for paper %d", len(raw_refs), paper_id)

        parsed_citations = []
        for raw_ref in raw_refs:
            parsed = self.parse_citation_with_llm(raw_ref)
            parsed["raw_text"] = raw_ref
            parsed_citations.append(parsed)

        # Save to database
        with get_db_session() as session:
            # Remove existing citations for this paper
            session.query(Citation).filter(Citation.paper_id == paper_id).delete()
            for citation_data in parsed_citations:
                citation = Citation(
                    paper_id=paper_id,
                    raw_text=citation_data["raw_text"],
                    parsed_title=citation_data["title"],
                    parsed_authors=citation_data["authors"],
                    parsed_year=citation_data["year"],
                    parsed_doi=citation_data["doi"],
                )
                session.add(citation)

        logger.info("Extracted %d citations for paper %d", len(parsed_citations), paper_id)
        return parsed_citations

    def get_cross_references(self, paper_id: int) -> list[dict]:
        """
        Find citations from this paper that match other papers in the database.

        Returns list of dicts with: citation_id, cited_paper_id, cited_paper_title
        """
        matches = []
        with get_db_session() as session:
            citations = (
                session.query(Citation)
                .filter(Citation.paper_id == paper_id)
                .all()
            )
            all_papers = session.query(Paper).filter(Paper.id != paper_id).all()

            for citation in citations:
                for paper in all_papers:
                    if (
                        citation.parsed_title
                        and paper.title
                        and citation.parsed_title.lower().strip()
                        in paper.title.lower().strip()
                    ):
                        matches.append({
                            "citation_id": citation.id,
                            "cited_paper_id": paper.id,
                            "cited_paper_title": paper.title,
                        })
        return matches
