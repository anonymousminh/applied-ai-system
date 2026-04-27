"""Tests for the rule-based scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schemas import Song, UserProfile
from src.scorer import load_songs, rank_songs, score_song


DATA = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


@pytest.fixture
def songs() -> list[Song]:
    return load_songs(DATA)


def test_load_songs_parses_full_catalog(songs: list[Song]) -> None:
    assert len(songs) == 18
    assert all(isinstance(s, Song) for s in songs)


def test_genre_match_increases_score() -> None:
    song = Song(
        id="t1",
        title="Test",
        artist="X",
        genre="pop",
        mood="happy",
        energy=0.8,
        tempo_bpm=120,
        valence=0.8,
        danceability=0.7,
        acousticness=0.1,
    )
    pop_profile = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rock_profile = pop_profile.model_copy(update={"favorite_genre": "rock"})

    pop_score = score_song(song, pop_profile).score
    rock_score = score_song(song, rock_profile).score

    assert pop_score > rock_score
    assert pop_score - rock_score == pytest.approx(2.0, abs=0.001)


def test_rank_returns_top_k_sorted(songs: list[Song]) -> None:
    profile = UserProfile(
        favorite_genre="lofi",
        favorite_mood="calm",
        target_energy=0.3,
        likes_acoustic=True,
    )
    top = rank_songs(songs, profile, top_k=3)
    assert len(top) == 3
    assert top[0].score >= top[1].score >= top[2].score
    assert top[0].song.genre == "lofi"
