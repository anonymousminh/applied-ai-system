"""Reliability guardrails.

Three layered checks satisfy the rubric's reliability/guardrail requirement:

  1. Input validation: Pydantic does most of this in `schemas.UserProfile`,
     but `validate_profile_dict` wraps it with friendly error messages and
     surfaces "soft" warnings (e.g. unknown genre still passes but is flagged).

  2. Retrieval fallback: `retrieval_is_usable` returns False when the
     retriever produces nothing useful, so the orchestrator can degrade
     gracefully to a baseline-only response instead of inventing context.

  3. Groundedness guardrail: `check_groundedness` verifies that any
     generated explanation actually cites at least one retrieved passage.
     If not, the explanation is replaced with a conservative fallback.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from pydantic import ValidationError

from .schemas import (
    Citation,
    Recommendation,
    RetrievedPassage,
    ScoredSong,
    UserProfile,
)


def validate_profile_dict(raw: dict) -> Tuple[UserProfile, List[str]]:
    """Validate raw input. Raises ValueError on hard failures, returns warnings list.

    Hard failures: missing required field, out-of-range numeric, wrong type.
    Soft failures (warnings): unknown genre/mood that still parses.
    """
    try:
        profile = UserProfile(**raw)
    except ValidationError as e:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        raise ValueError(f"Invalid user profile: {msg}") from e

    warnings: List[str] = []
    if profile.is_out_of_catalog_genre:
        warnings.append(
            f"Genre '{profile.favorite_genre}' is not in the catalog; "
            "results will rely on mood and audio-feature similarity."
        )
    if profile.is_out_of_catalog_mood:
        warnings.append(
            f"Mood '{profile.favorite_mood}' is not in the standard set; "
            "valence-based scoring may be less precise."
        )
    return profile, warnings


def retrieval_is_usable(
    retrieved: Sequence[RetrievedPassage], min_similarity: float = 0.05
) -> bool:
    """Decide whether retrieval produced usable context.

    Returns False if:
      - the list is empty, or
      - the best result's similarity is below `min_similarity`.
    """
    if not retrieved:
        return False
    return retrieved[0].similarity >= min_similarity


def check_groundedness(
    explanation: str,
    citations: Sequence[Citation],
    retrieved: Sequence[RetrievedPassage],
) -> bool:
    """A claim is grounded if at least one citation refers to a retrieved passage.

    This is intentionally simple: a stronger version would parse sentences and
    check each one. For a class-scale system, "has citations and they came from
    real retrieval" is a meaningful baseline.
    """
    if not explanation.strip():
        return False
    if not citations:
        return False
    retrieved_ids = {r.passage.id for r in retrieved}
    return any(c.passage_id in retrieved_ids for c in citations)


def fallback_explanation(scored: ScoredSong, profile: UserProfile) -> str:
    """Conservative explanation used when generation fails or is ungrounded."""
    s = scored.song
    return (
        f"'{s.title}' by {s.artist} is recommended primarily on rule-based "
        f"feature similarity to your profile (genre={profile.favorite_genre}, "
        f"mood={profile.favorite_mood}, target_energy={profile.target_energy:.2f}). "
        "No grounded narrative context was available, so this explanation is "
        "kept minimal."
    )


def finalize_recommendation(
    scored: ScoredSong,
    profile: UserProfile,
    explanation: str,
    citations: List[Citation],
    retrieved: Sequence[RetrievedPassage],
) -> Recommendation:
    """Assemble a Recommendation, applying the groundedness guardrail."""
    grounded = check_groundedness(explanation, citations, retrieved)
    if not grounded:
        explanation = fallback_explanation(scored, profile)
        citations = []
    return Recommendation(
        song=scored.song,
        score=scored.score,
        breakdown=scored.breakdown,
        explanation=explanation,
        citations=citations,
        grounded=grounded,
    )
