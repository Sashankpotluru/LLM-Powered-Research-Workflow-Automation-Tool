"""GPT-4 powered summarization chains."""

from langchain_openai import ChatOpenAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from db.database import get_db_session
from db.models import Paper, Chunk
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

SINGLE_PAPER_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an academic research assistant. Analyze the following research paper text and provide a structured summary.

TEXT:
{text}

Provide your summary in the following format:

## Abstract Summary
A concise 2-3 sentence summary of the paper's main contribution.

## Key Findings
- Bullet point list of the most important findings

## Methodology
Brief description of the research methodology used.

## Limitations
Any noted limitations or gaps.

## Relevance
Why this paper matters and potential applications.
""",
)

SYNTHESIS_MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following section of a research paper, focusing on key findings, methodology, and conclusions:

{text}

CONCISE SUMMARY:""",
)

SYNTHESIS_REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are synthesizing multiple research paper summaries into a cohesive literature review.

SUMMARIES:
{text}

Create a unified synthesis that:
1. Identifies common themes and findings across papers
2. Notes any contradictions or disagreements
3. Highlights research gaps
4. Suggests potential future research directions

SYNTHESIS:""",
)


class Summarizer:
    """Handles single-paper and multi-paper summarization."""

    def __init__(self):
        self.settings = get_settings()
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name=self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                temperature=0.2,
            )
        return self._llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def summarize_paper(self, paper_id: int) -> str:
        """
        Generate a structured summary for a single paper.

        Returns the summary as markdown text.
        """
        with get_db_session() as session:
            paper = session.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                raise ValueError(f"Paper with id {paper_id} not found")

            chunks = (
                session.query(Chunk)
                .filter(Chunk.paper_id == paper_id)
                .order_by(Chunk.chunk_index)
                .all()
            )

            if not chunks:
                raise ValueError(f"No chunks found for paper {paper_id}")

            # Combine chunks into full text (truncate if very long)
            full_text = "\n\n".join(c.chunk_text for c in chunks)
            if len(full_text) > 15000:
                full_text = full_text[:15000] + "\n\n[... truncated for summary]"

        doc = Document(page_content=full_text)
        chain = load_summarize_chain(
            self.llm,
            chain_type="stuff",
            prompt=SINGLE_PAPER_PROMPT,
        )

        result = chain.invoke({"input_documents": [doc]})
        summary = result["output_text"]

        logger.info("Generated summary for paper %d (%d chars)", paper_id, len(summary))
        return summary

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def synthesize_papers(self, paper_ids: list[int], topic: str = "") -> str:
        """
        Generate a multi-paper synthesis using map-reduce.

        Args:
            paper_ids: List of paper IDs to synthesize.
            topic: Optional topic to focus the synthesis on.

        Returns the synthesis as markdown text.
        """
        documents = []

        with get_db_session() as session:
            for paper_id in paper_ids:
                paper = session.query(Paper).filter(Paper.id == paper_id).first()
                if not paper:
                    continue

                chunks = (
                    session.query(Chunk)
                    .filter(Chunk.paper_id == paper_id)
                    .order_by(Chunk.chunk_index)
                    .all()
                )

                text = "\n\n".join(c.chunk_text for c in chunks)
                if len(text) > 8000:
                    text = text[:8000] + "\n\n[... truncated]"

                doc = Document(
                    page_content=f"Paper: {paper.title}\nAuthors: {paper.authors}\n\n{text}",
                    metadata={"paper_id": paper_id, "title": paper.title},
                )
                documents.append(doc)

        if not documents:
            raise ValueError("No valid papers found for synthesis")

        # Adjust reduce prompt if topic provided
        reduce_prompt = SYNTHESIS_REDUCE_PROMPT
        if topic:
            reduce_prompt = PromptTemplate(
                input_variables=["text"],
                template=f"""You are synthesizing multiple research paper summaries into a cohesive literature review focused on: {topic}

SUMMARIES:
{{text}}

Create a unified synthesis that:
1. Identifies common themes and findings related to {topic}
2. Notes any contradictions or disagreements
3. Highlights research gaps related to {topic}
4. Suggests potential future research directions

SYNTHESIS:""",
            )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=SYNTHESIS_MAP_PROMPT,
            combine_prompt=reduce_prompt,
        )

        result = chain.invoke({"input_documents": documents})
        synthesis = result["output_text"]

        logger.info(
            "Generated synthesis for %d papers (%d chars)",
            len(paper_ids),
            len(synthesis),
        )
        return synthesis
