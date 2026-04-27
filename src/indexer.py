"""Knowledge-base indexer.

Loads markdown files from `data/`, splits them into per-section passages, and
prepares them for retrieval. The chunking unit is a level-2 markdown heading
(`## Title`) so that each passage is a coherent topic (one artist, one genre,
one song review).

This is the offline half of the RAG pipeline: it runs once at startup. The
retriever consumes its output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from .schemas import Passage, SourceType


SOURCE_FILES: Dict[str, SourceType] = {
    "artist_bios.md": "artist_bio",
    "genre_descriptions.md": "genre_description",
    "listener_reviews.md": "listener_review",
}


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "section"


def _split_markdown_by_h2(markdown: str) -> List[tuple[str, str]]:
    """Split a markdown document into (title, body) pairs at every `## ` heading.

    Content before the first `## ` heading is discarded (treated as preamble).
    """
    sections: List[tuple[str, str]] = []
    current_title: str | None = None
    current_body: List[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)

    if current_title is not None:
        sections.append((current_title, "\n".join(current_body).strip()))

    return [(t, b) for t, b in sections if b]


def load_passages(data_dir: str | Path) -> List[Passage]:
    """Load and chunk all known knowledge-base files into passages."""
    data_path = Path(data_dir)
    passages: List[Passage] = []

    for filename, source_type in SOURCE_FILES.items():
        path = data_path / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for title, body in _split_markdown_by_h2(text):
            passage_id = f"{source_type}:{_slugify(title)}"
            passages.append(
                Passage(
                    id=passage_id,
                    source_type=source_type,
                    source_file=filename,
                    title=title,
                    text=body,
                )
            )

    return passages
