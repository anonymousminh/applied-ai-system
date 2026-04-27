"""Explanation generator.

The generator is the "G" in RAG: given (a) the user profile, (b) the candidate
song, and (c) retrieved passages, it produces a short, grounded explanation
with citations.

Two backends are supported:

  - "template" (default): a deterministic, dependency-free generator that
    composes the explanation from retrieved passages and structured features.
    Reproducible, free, and used in tests.

  - "openai": calls an OpenAI chat model when OPENAI_API_KEY is set and
    USE_LLM=true. The prompt forces the model to ground every sentence in a
    cited passage and returns JSON, which is parsed and validated.

The orchestrator picks the backend at construction time, so the rest of the
pipeline stays identical.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence

from .schemas import Citation, RetrievedPassage, ScoredSong, UserProfile


SYSTEM_PROMPT = """You are a music recommendation explainer.

You will be given:
  - A user profile (genre, mood, energy, acoustic preference)
  - One candidate song with audio features
  - A small set of retrieved knowledge-base passages

Your job: write 2-3 sentences explaining why this song fits the user.

STRICT RULES:
  1. Every claim must be supported by a retrieved passage OR a numeric feature
     of the song. Do NOT invent facts about the song or artist.
  2. Cite passages by their `id` (format: "source_type:slug") in a JSON list.
  3. Keep tone neutral and concise.
  4. Output ONLY valid JSON matching:
     {"explanation": "...", "citations": ["passage_id_1", ...]}
"""


def profile_summary(profile: UserProfile) -> str:
    parts = [
        f"Likes {profile.favorite_genre}",
        f"prefers a {profile.favorite_mood} mood",
        f"target energy around {profile.target_energy:.2f}",
    ]
    if profile.likes_acoustic:
        parts.append("enjoys acoustic textures")
    return ", ".join(parts) + "."


def song_summary(scored: ScoredSong) -> str:
    s = scored.song
    return (
        f"'{s.title}' by {s.artist} is a {s.genre} track with a {s.mood} mood, "
        f"energy {s.energy:.2f}, tempo {s.tempo_bpm:.0f} BPM, "
        f"valence {s.valence:.2f}, acousticness {s.acousticness:.2f}."
    )


class TemplateGenerator:
    """Deterministic, no-API fallback generator.

    Builds a coherent multi-sentence explanation by combining:
      1. Numeric feature alignment (genre / mood / energy match).
      2. Up to two of the most relevant retrieved passages.

    Citations include only passages whose content is actually referenced in
    the explanation, so the groundedness guardrail will pass.
    """

    def generate(
        self,
        profile: UserProfile,
        scored: ScoredSong,
        retrieved: Sequence[RetrievedPassage],
    ) -> tuple[str, List[Citation]]:
        song = scored.song
        sentences: List[str] = []
        citations: List[Citation] = []

        feature_bits: List[str] = []
        if scored.breakdown.get("genre", 0) > 0:
            feature_bits.append(
                f"its {song.genre} genre matches your preference"
            )
        if scored.breakdown.get("mood", 0) > 0:
            feature_bits.append(
                f"its {song.mood} mood lines up with what you want"
            )
        energy_pts = scored.breakdown.get("energy", 0)
        if energy_pts >= 1.6:
            feature_bits.append(
                f"its energy ({song.energy:.2f}) closely matches your target ({profile.target_energy:.2f})"
            )

        if feature_bits:
            sentences.append(
                f"'{song.title}' by {song.artist} fits because " + ", and ".join(feature_bits) + "."
            )
        else:
            sentences.append(
                f"'{song.title}' by {song.artist} is included as a near-match across audio features rather than an exact preference hit."
            )

        used = 0
        for r in retrieved:
            if used >= 2:
                break
            p = r.passage
            referenced = (
                song.title.lower() in p.text.lower()
                or song.artist.lower() in p.text.lower()
                or song.genre.lower() in p.title.lower()
                or song.mood.lower() in p.title.lower()
                or song.genre.lower() in p.text.lower()
            )
            if not referenced:
                continue
            snippet = p.text.split(". ")[0].strip().rstrip(".")
            sentences.append(f"Context: {snippet} [{p.id}].")
            citations.append(
                Citation(
                    passage_id=p.id,
                    source_type=p.source_type,
                    title=p.title,
                )
            )
            used += 1

        if not citations and retrieved:
            r = retrieved[0]
            p = r.passage
            snippet = p.text.split(". ")[0].strip().rstrip(".")
            sentences.append(f"Related context: {snippet} [{p.id}].")
            citations.append(
                Citation(passage_id=p.id, source_type=p.source_type, title=p.title)
            )

        return " ".join(sentences), citations


class OpenAIGenerator:
    """LLM-backed generator. Gracefully falls back if API call fails."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._fallback = TemplateGenerator()

    def generate(
        self,
        profile: UserProfile,
        scored: ScoredSong,
        retrieved: Sequence[RetrievedPassage],
    ) -> tuple[str, List[Citation]]:
        passages_block = "\n\n".join(
            f"[{r.passage.id}] {r.passage.title}\n{r.passage.text}"
            for r in retrieved
        )
        user_prompt = (
            f"USER PROFILE: {profile_summary(profile)}\n\n"
            f"CANDIDATE SONG: {song_summary(scored)}\n\n"
            f"RETRIEVED PASSAGES:\n{passages_block}\n\n"
            "Return JSON only."
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            explanation = str(data.get("explanation", "")).strip()
            cited_ids = [str(x) for x in data.get("citations", [])]

            id_to_passage = {r.passage.id: r.passage for r in retrieved}
            citations = [
                Citation(
                    passage_id=pid,
                    source_type=id_to_passage[pid].source_type,
                    title=id_to_passage[pid].title,
                )
                for pid in cited_ids
                if pid in id_to_passage
            ]

            if not explanation or not citations:
                return self._fallback.generate(profile, scored, retrieved)
            return explanation, citations
        except Exception:
            return self._fallback.generate(profile, scored, retrieved)


def make_generator() -> TemplateGenerator | OpenAIGenerator:
    """Construct the configured generator based on environment variables."""
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if use_llm and api_key:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return OpenAIGenerator(model=model)
    return TemplateGenerator()
