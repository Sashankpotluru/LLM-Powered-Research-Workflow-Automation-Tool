"""PDF text extraction and chunking pipeline."""

import hashlib
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file page by page.

    Returns list of dicts with keys: page_number, text
    """
    pages = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_number": page_num + 1, "text": text.strip()})
        doc.close()
        logger.info("Extracted text from %d pages in %s", len(pages), file_path)
    except Exception:
        logger.exception("Failed to extract text from %s", file_path)
        raise
    return pages


def get_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF."""
    try:
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        logger.exception("Failed to get page count for %s", file_path)
        return 0


def extract_metadata_from_pdf(file_path: str) -> dict:
    """
    Extract metadata (title, author) from PDF metadata fields.

    Returns dict with keys: title, authors
    """
    metadata = {"title": "", "authors": ""}
    try:
        doc = fitz.open(file_path)
        pdf_meta = doc.metadata
        if pdf_meta:
            metadata["title"] = pdf_meta.get("title", "") or ""
            metadata["authors"] = pdf_meta.get("author", "") or ""
        doc.close()
    except Exception:
        logger.exception("Failed to extract metadata from %s", file_path)
    return metadata


def chunk_text(
    pages: list[dict],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[dict]:
    """
    Split extracted page texts into chunks for embedding.

    Returns list of dicts with keys: chunk_text, chunk_index, page_number
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk_text_str in page_chunks:
            chunks.append({
                "chunk_text": chunk_text_str,
                "chunk_index": chunk_index,
                "page_number": page_data["page_number"],
            })
            chunk_index += 1

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def save_uploaded_file(uploaded_file, filename: str) -> str:
    """
    Save an uploaded file to the uploads directory.

    Returns the file path.
    """
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Ensure unique filename
    file_path = upload_dir / filename
    counter = 1
    while file_path.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        file_path = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info("Saved uploaded file to %s", file_path)
    return str(file_path)
