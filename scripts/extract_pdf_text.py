#!/usr/bin/env python3

import argparse
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalized text from a PDF.")
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--keep-page-markers", action="store_true")
    return parser.parse_args()


def _clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if re.fullmatch(r"\d{1,3}", line):
            continue
        cleaned_lines.append(line)

    paragraphs = []
    current: list[str] = []
    for line in cleaned_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = _clean_page_text(page.extract_text() or "")
        if not text:
            continue
        if args.keep_page_markers:
            chunks.append(f"[[PAGE {page_index}]]\n{text}")
        else:
            chunks.append(text)

    output_path.write_text("\n\n".join(chunks).strip() + "\n", encoding="utf-8")
    print(
        f"[done] extracted {len(chunks)} pages from {pdf_path} to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
