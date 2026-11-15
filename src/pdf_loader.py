"""
PDF text extraction with page-number tracking.

Reuses the pypdf extraction approach from the original load_pdf..py,
but tags every chunk with its source page number(s).
"""

from pypdf import PdfReader


def extract_pages(path: str) -> list[dict]:
    """Return a list of {'page': int, 'text': str} for every non-empty page."""
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def split_into_chunks(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split page texts into overlapping character chunks.

    Each chunk records the PDF page(s) it came from.
    Returns a list of {'text': str, 'pages': list[int]}.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be between zero and chunk size minus one.")

    chunks: list[dict] = []

    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "pages": [page_num]})
            if end >= len(text):
                break
            start += chunk_size - overlap

    return chunks
