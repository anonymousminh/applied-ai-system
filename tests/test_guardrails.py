"""Tests for input validation and guardrail behavior."""

from __future__ import annotations

import pytest

from src.guardrails import (
    check_groundedness,
    retrieval_is_usable,
    validate_profile_dict,
)
from src.schemas import (
    Citation,
    Passage,
    RetrievedPassage,
)


def test_validate_profile_accepts_known_genre_and_mood() -> None:
    profile, warnings = validate_profile_dict(
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
            "likes_acoustic": False,
        }
    )
    assert profile.favorite_genre == "pop"
    assert warnings == []


def test_validate_profile_warns_on_unknown_genre() -> None:
    _, warnings = validate_profile_dict(
        {
            "favorite_genre": "kpop",
            "favorite_mood": "energetic",
            "target_energy": 0.8,
            "likes_acoustic": False,
        }
    )
    assert any("kpop" in w for w in warnings)


def test_validate_profile_rejects_out_of_range_energy() -> None:
    with pytest.raises(ValueError):
        validate_profile_dict(
            {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 5.0,
                "likes_acoustic": False,
            }
        )


def test_validate_profile_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        validate_profile_dict(
            {
                "favorite_genre": "pop",
                "target_energy": 0.5,
            }
        )


def _fake_retrieved(pid: str, sim: float) -> RetrievedPassage:
    p = Passage(
        id=pid,
        source_type="artist_bio",
        source_file="x.md",
        title="t",
        text="body",
    )
    return RetrievedPassage(passage=p, similarity=sim)


def test_retrieval_is_usable_thresholds() -> None:
    assert retrieval_is_usable([_fake_retrieved("a", 0.5)]) is True
    assert retrieval_is_usable([_fake_retrieved("a", 0.01)]) is False
    assert retrieval_is_usable([]) is False


def test_groundedness_passes_when_citation_in_retrieved() -> None:
    retrieved = [_fake_retrieved("artist_bio:foo", 0.3)]
    citation = Citation(
        passage_id="artist_bio:foo", source_type="artist_bio", title="t"
    )
    assert check_groundedness("text", [citation], retrieved) is True


def test_groundedness_fails_when_citation_unknown() -> None:
    retrieved = [_fake_retrieved("artist_bio:foo", 0.3)]
    citation = Citation(
        passage_id="artist_bio:bar", source_type="artist_bio", title="t"
    )
    assert check_groundedness("text", [citation], retrieved) is False


def test_groundedness_fails_when_no_citations() -> None:
    retrieved = [_fake_retrieved("artist_bio:foo", 0.3)]
    assert check_groundedness("text", [], retrieved) is False
