"""Tests for indexer + retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.indexer import load_passages
from src.retriever import Retriever


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    passages = load_passages(DATA_DIR)
    return Retriever.build(passages)


def test_passages_cover_all_three_sources() -> None:
    passages = load_passages(DATA_DIR)
    types = {p.source_type for p in passages}
    assert types == {"artist_bio", "genre_description", "listener_review"}
    assert len(passages) > 20


def test_retriever_returns_top_k(retriever: Retriever) -> None:
    results = retriever.retrieve("calm lofi study background", top_k=5)
    assert 1 <= len(results) <= 5
    titles = " ".join(r.passage.title.lower() for r in results)
    assert "lofi" in titles or "calm" in titles or "soft static" in titles


def test_retriever_similarity_descending(retriever: Retriever) -> None:
    results = retriever.retrieve("intense rock workout", top_k=5)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_retriever_filter_by_source(retriever: Retriever) -> None:
    results = retriever.retrieve(
        "high energy pop", top_k=5, source_types=["genre_description"]
    )
    assert len(results) >= 1
    assert all(r.passage.source_type == "genre_description" for r in results)


def test_empty_query_returns_empty(retriever: Retriever) -> None:
    assert retriever.retrieve("   ", top_k=3) == []
