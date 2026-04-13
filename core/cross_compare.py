"""Cross-document comparison using GPT-4."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from db.database import get_db_session
from db.models import Paper, Chunk
from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

COMPARISON_SYSTEM_PROMPT = """You are an expert academic research analyst. You will be given text from multiple research papers. Your task is to perform a detailed cross-document comparison.

Provide your analysis in the following structured format:

## Comparison Overview
Brief overview of the papers being compared.

## Methodology Comparison
| Aspect | {paper_headers} |
|--------|{separators}|
| Approach | ... |
| Data Sources | ... |
| Sample Size | ... |
| Analysis Method | ... |

## Key Findings Comparison
| Finding Area | {paper_headers} |
|-------------|{separators}|
| Main Result | ... |
| Secondary Findings | ... |
| Statistical Significance | ... |

## Agreements
Points where the papers reach similar conclusions.

## Disagreements
Points where the papers contradict or diverge.

## Research Gaps
Gaps identified across the papers that future research could address.

## Synthesis
An integrated view combining insights from all papers.
"""


class CrossComparer:
    """Performs cross-document comparison of research papers."""

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
                max_tokens=4000,
            )
        return self._llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def compare_papers(self, paper_ids: list[int]) -> str:
        """
        Compare 2-5 papers and generate a structured comparison.

        Args:
            paper_ids: List of 2-5 paper IDs to compare.

        Returns markdown-formatted comparison.
        """
        if len(paper_ids) < 2:
            raise ValueError("Need at least 2 papers to compare")
        if len(paper_ids) > 5:
            raise ValueError("Maximum 5 papers for comparison")

        papers_data = []
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
                # Truncate per paper to fit in context
                max_per_paper = 6000 // len(paper_ids)
                if len(text) > max_per_paper:
                    text = text[:max_per_paper] + "\n\n[... truncated]"

                papers_data.append({
                    "id": paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "text": text,
                })

        if len(papers_data) < 2:
            raise ValueError("Could not load enough papers for comparison")

        # Build the prompt
        paper_headers = " | ".join(f"Paper {i+1}: {p['title'][:30]}" for i, p in enumerate(papers_data))
        separators = " | ".join("---" for _ in papers_data)

        system_prompt = COMPARISON_SYSTEM_PROMPT.replace("{paper_headers}", paper_headers)
        system_prompt = system_prompt.replace("{separators}", separators)

        papers_text = ""
        for i, paper in enumerate(papers_data):
            papers_text += f"\n\n--- PAPER {i+1}: {paper['title']} ---\n"
            papers_text += f"Authors: {paper['authors']}\n\n"
            papers_text += paper["text"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Compare these papers:{papers_text}"),
        ]

        response = self.llm.invoke(messages)
        comparison = response.content

        logger.info("Generated comparison for %d papers", len(papers_data))
        return comparison
