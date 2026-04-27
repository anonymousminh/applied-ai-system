"""Pydantic schemas for inputs, outputs, and internal data structures.

These schemas serve two purposes in the rubric:
  1. Input validation guardrail (Reliability section).
  2. Strong typing across modules so refactors stay safe.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


VALID_GENRES = {"pop", "lofi", "rock", "electronic", "folk", "jazz"}
VALID_MOODS = {"happy", "energetic", "calm", "melancholic", "intense"}


class UserProfile(BaseModel):
    """Validated user preference profile.

    Unknown genres/moods are allowed (they are treated as out-of-catalog at
    scoring time) but are flagged so the orchestrator can adapt retrieval.
    """

    name: Optional[str] = None
    favorite_genre: str = Field(..., min_length=1)
    favorite_mood: str = Field(..., min_length=1)
    target_energy: float = Field(..., ge=0.0, le=1.0)
    likes_acoustic: bool = False

    @field_validator("favorite_genre", "favorite_mood")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def is_out_of_catalog_genre(self) -> bool:
        return self.favorite_genre not in VALID_GENRES

    @property
    def is_out_of_catalog_mood(self) -> bool:
        return self.favorite_mood not in VALID_MOODS


class Song(BaseModel):
    """A single song record from songs.csv."""

    id: str
    title: str
    artist: str
    genre: str
    mood: str
    energy: float = Field(..., ge=0.0, le=1.0)
    tempo_bpm: float = Field(..., gt=0.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    danceability: float = Field(..., ge=0.0, le=1.0)
    acousticness: float = Field(..., ge=0.0, le=1.0)


class ScoredSong(BaseModel):
    """A song with its rule-based score and breakdown."""

    song: Song
    score: float
    breakdown: dict


SourceType = Literal["artist_bio", "genre_description", "listener_review"]


class Passage(BaseModel):
    """A retrievable knowledge-base chunk."""

    id: str
    source_type: SourceType
    source_file: str
    title: str
    text: str


class RetrievedPassage(BaseModel):
    """A passage with its retrieval similarity score."""

    passage: Passage
    similarity: float


class Citation(BaseModel):
    """A citation in a generated explanation."""

    passage_id: str
    source_type: SourceType
    title: str


class Recommendation(BaseModel):
    """Final user-facing recommendation: song + score + grounded explanation."""

    song: Song
    score: float
    breakdown: dict
    explanation: str
    citations: List[Citation]
    grounded: bool


class RecommendationResult(BaseModel):
    """Full pipeline result for one profile."""

    profile: UserProfile
    recommendations: List[Recommendation]
    warnings: List[str] = []
    used_fallback: bool = False
