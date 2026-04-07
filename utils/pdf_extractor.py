"""
pdf_extractor.py  –  Robust PDF Text Extractor
================================================
Strategy:
  1. pdfplumber (primary) – best layout-aware extraction
  2. pypdf fallback – when pdfplumber fails or returns empty
  3. Text cleaning: normalise unicode, bullets, whitespace
  4. Raises clear errors for scanned/empty PDFs
"""

import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract clean text from a PDF file.
    Tries pdfplumber first, falls back to pypdf.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    text = _try_pdfplumber(str(path))

    if not text or len(text.strip()) < 50:
        text = _try_pypdf(str(path))

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Could not extract readable text from PDF. "
            "The file may be scanned/image-based or empty."
        )

    return _clean_text(text)


def _try_pdfplumber(pdf_path: str) -> str:
    try:
        import pdfplumber
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    pages.append(t)
        return "\n".join(pages)
    except ImportError:
        return ""
    except Exception as exc:
        print(f"  pdfplumber error: {exc}")
        return ""


def _try_pypdf(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except ImportError:
        return ""
    except Exception as exc:
        print(f"  pypdf error: {exc}")
        return ""


def _clean_text(text: str) -> str:
    """Normalise extracted resume text."""
    # Unicode dashes / bullets → ASCII equivalents
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2022": "-", "\u25cf": "-", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u00a0": " ",  # non-breaking space
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)

    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces → 1
    text = re.sub(r" {2,}", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return basic PDF metadata."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return {
                "pages": len(pdf.pages),
                "file_name": Path(pdf_path).name,
                "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 1),
            }
    except Exception:
        return {
            "pages": "?",
            "file_name": Path(pdf_path).name,
            "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 1),
        }