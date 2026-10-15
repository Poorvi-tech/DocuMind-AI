from __future__ import annotations

import re
from typing import List

import fitz

from backend.app.schemas import PageText


def _clean_text(raw_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    return cleaned


def extract_pdf_text(file_bytes: bytes, filename: str) -> List[PageText]:
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                raise ValueError("This PDF appears to be empty or unreadable.")

            pages: List[PageText] = []
            for page_number in range(document.page_count):
                page = document[page_number]
                text = page.get_text("text")
                cleaned = _clean_text(text)
                if not cleaned:
                    continue
                pages.append(PageText(page_number=page_number + 1, text=cleaned))

            if not pages:
                raise ValueError(
                    "No readable text was found in this PDF. It may be scanned, image-only, or encrypted."
                )
            return pages
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError("The PDF could not be processed. Please upload a readable file.") from exc
