"""End-to-end orchestrator that wires all components together.

Pipeline:
  raw_profile -> validate -> rule-based score -> per-candidate retrieve ->
  generate explanation -> guardrail check -> finalize -> return result
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .generator import (
    OpenAIGenerator,
    TemplateGenerator,
    make_generator,
    profile_summary,
    song_summary,
)
from .guardrails import (
    finalize_recommendation,
    retrieval_is_usable,
    validate_profile_dict,
)
from .indexer import load_passages
from .retriever import Retriever, build_query
from .schemas import (
    Recommendation,
    RecommendationResult,
    ScoredSong,
    UserProfile,
)
from .scorer import load_songs, rank_songs


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Orchestrator:
    """Long-lived object that holds the song catalog, vector store, and generator."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        generator: Optional[TemplateGenerator | OpenAIGenerator] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.songs = load_songs(self.data_dir / "songs.csv")
        passages = load_passages(self.data_dir)
        self.retriever = Retriever.build(passages)
        self.generator = generator if generator is not None else make_generator()

    def recommend(
        self,
        raw_profile: dict,
        top_k: int = 3,
        retrieve_k: int = 3,
    ) -> RecommendationResult:
        """Run the full pipeline for one raw profile dict."""
        profile, warnings = validate_profile_dict(raw_profile)

        scored_top: List[ScoredSong] = rank_songs(self.songs, profile, top_k=top_k)

        used_fallback = False
        recommendations: List[Recommendation] = []
        prof_summary = profile_summary(profile)

        for scored in scored_top:
            song_text = song_summary(scored)
            query = build_query(prof_summary, song_text)
            retrieved = self.retriever.retrieve(query, top_k=retrieve_k)

            if not retrieval_is_usable(retrieved):
                used_fallback = True
                rec = finalize_recommendation(
                    scored, profile, explanation="", citations=[], retrieved=[]
                )
                recommendations.append(rec)
                continue

            explanation, citations = self.generator.generate(
                profile, scored, retrieved
            )
            rec = finalize_recommendation(
                scored, profile, explanation, citations, retrieved
            )
            if not rec.grounded:
                used_fallback = True
            recommendations.append(rec)

        return RecommendationResult(
            profile=profile,
            recommendations=recommendations,
            warnings=warnings,
            used_fallback=used_fallback,
        )
