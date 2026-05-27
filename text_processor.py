import logging
from pathlib import Path
from typing import List

import nltk
from docx import Document  # Reserved for future support
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader  # Reserved for future support

LOGGER = logging.getLogger(__name__)


def _ensure_punkt() -> None:
    """Ensure NLTK punkt tokenizer is available."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            LOGGER.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def extract_text_from_txt(file_path: str) -> str:
    """Read and return UTF-8 text from a .txt file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    return path.read_text(encoding="utf-8")


def extract_text_from_pdf(file_path: str) -> str:
    """Placeholder for future PDF extraction support."""
    raise NotImplementedError("PDF extraction will be added later.")


def extract_text_from_docx(file_path: str) -> str:
    """Placeholder for future DOCX extraction support."""
    raise NotImplementedError("DOCX extraction will be added later.")


def split_into_chunks(text: str, max_tokens: int = 400) -> List[str]:
    """
    Split text into sentence-based chunks.

    Uses a practical character limit equivalent to about 400 tokens.
    Approximation: 1 token ~ 0.75 word. Here we cap chunk by ~1200 chars.
    """
    _ = max_tokens  # Reserved for future token-level splitter
    _ensure_punkt()

    max_chars = 1200
    sentences = sent_tokenize(text, language="russian")
    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        prospective = f"{current_chunk} {sentence}".strip()
        if len(prospective) <= max_chars:
            current_chunk = prospective
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if len(sentence) <= max_chars:
            current_chunk = sentence
        else:
            start = 0
            while start < len(sentence):
                chunks.append(sentence[start : start + max_chars])
                start += max_chars
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
