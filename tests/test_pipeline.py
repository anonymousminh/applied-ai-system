"""End-to-end pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestrator import Orchestrator


PROFILES_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_profiles.json"


@pytest.fixture(scope="module")
def orch() -> Orchestrator:
    return Orchestrator()


@pytest.fixture(scope="module")
def demo_profiles() -> list[dict]:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def test_pipeline_produces_recommendations_for_each_profile(
    orch: Orchestrator, demo_profiles: list[dict]
) -> None:
    for raw in demo_profiles:
        result = orch.recommend(raw, top_k=3)
        assert len(result.recommendations) == 3
        for rec in result.recommendations:
            assert rec.song.id
            assert rec.score >= 0
            assert isinstance(rec.explanation, str) and rec.explanation


def test_pipeline_marks_grounded_recommendations(
    orch: Orchestrator, demo_profiles: list[dict]
) -> None:
    pop = demo_profiles[0]
    result = orch.recommend(pop, top_k=3)
    assert any(r.grounded for r in result.recommendations)


def test_pipeline_warns_on_unknown_genre(orch: Orchestrator) -> None:
    raw = {
        "name": "kpop test",
        "favorite_genre": "kpop",
        "favorite_mood": "energetic",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }
    result = orch.recommend(raw, top_k=2)
    assert result.warnings
    assert len(result.recommendations) == 2


def test_pipeline_rejects_invalid_input(orch: Orchestrator) -> None:
    with pytest.raises(ValueError):
        orch.recommend(
            {"favorite_genre": "pop", "target_energy": 99}, top_k=2
        )


def test_pipeline_consistent_for_same_input(
    orch: Orchestrator, demo_profiles: list[dict]
) -> None:
    raw = demo_profiles[1]
    r1 = orch.recommend(raw, top_k=3)
    r2 = orch.recommend(raw, top_k=3)
    ids1 = [rec.song.id for rec in r1.recommendations]
    ids2 = [rec.song.id for rec in r2.recommendations]
    assert ids1 == ids2
