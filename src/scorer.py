"""Rule-based song scorer.

Carried over from the original VibeFinder 1.0 logic so that the rubric can
clearly show what is "baseline" vs what is "new RAG-enhanced." Genre and mood
are fixed-weight features; energy, tempo, valence, and danceability contribute
similarity points; acousticness is conditional on user preference.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .schemas import ScoredSong, Song, UserProfile


GENRE_WEIGHT = 2.0
MOOD_WEIGHT = 1.0
ENERGY_WEIGHT = 2.0
TEMPO_WEIGHT = 0.5
VALENCE_WEIGHT = 0.5
DANCEABILITY_WEIGHT = 0.5
ACOUSTIC_BONUS = 0.5

TEMPO_REFERENCE_BPM = 60.0


def load_songs(csv_path: str | Path) -> List[Song]:
    """Load and validate songs from CSV."""
    df = pd.read_csv(csv_path)
    return [Song(**row) for row in df.to_dict(orient="records")]


def _energy_similarity(song_energy: float, target_energy: float) -> float:
    return max(0.0, 1.0 - abs(song_energy - target_energy))


def _tempo_similarity(song_tempo: float, target_energy: float) -> float:
    target_tempo = 80 + 80 * target_energy
    diff = abs(song_tempo - target_tempo) / TEMPO_REFERENCE_BPM
    return max(0.0, 1.0 - diff)


def _valence_similarity(song_valence: float, mood: str) -> float:
    mood_to_valence = {
        "happy": 0.85,
        "energetic": 0.65,
        "calm": 0.55,
        "melancholic": 0.25,
        "intense": 0.45,
    }
    target = mood_to_valence.get(mood, 0.5)
    return max(0.0, 1.0 - abs(song_valence - target))


def score_song(song: Song, profile: UserProfile) -> ScoredSong:
    """Score a single song against a user profile and return a breakdown."""
    breakdown: dict = {}
    score = 0.0

    genre_pts = GENRE_WEIGHT if song.genre == profile.favorite_genre else 0.0
    breakdown["genre"] = genre_pts
    score += genre_pts

    mood_pts = MOOD_WEIGHT if song.mood == profile.favorite_mood else 0.0
    breakdown["mood"] = mood_pts
    score += mood_pts

    energy_pts = _energy_similarity(song.energy, profile.target_energy) * ENERGY_WEIGHT
    breakdown["energy"] = round(energy_pts, 3)
    score += energy_pts

    tempo_pts = _tempo_similarity(song.tempo_bpm, profile.target_energy) * TEMPO_WEIGHT
    breakdown["tempo"] = round(tempo_pts, 3)
    score += tempo_pts

    valence_pts = _valence_similarity(song.valence, profile.favorite_mood) * VALENCE_WEIGHT
    breakdown["valence"] = round(valence_pts, 3)
    score += valence_pts

    dance_pts = song.danceability * DANCEABILITY_WEIGHT
    breakdown["danceability"] = round(dance_pts, 3)
    score += dance_pts

    if profile.likes_acoustic:
        acoustic_pts = song.acousticness * ACOUSTIC_BONUS
        breakdown["acoustic_bonus"] = round(acoustic_pts, 3)
        score += acoustic_pts
    else:
        breakdown["acoustic_bonus"] = 0.0

    return ScoredSong(song=song, score=round(score, 3), breakdown=breakdown)


def rank_songs(songs: List[Song], profile: UserProfile, top_k: int = 5) -> List[ScoredSong]:
    """Score every song and return the top_k descending by score."""
    scored = [score_song(s, profile) for s in songs]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]
