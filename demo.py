"""End-to-end demo runner.

Loads `examples/demo_profiles.json` and runs the full pipeline for every
profile, printing a human-readable report. This script is the primary
artifact for the rubric's "Functional End-to-End System Demonstration."

Usage:
    python demo.py
    python demo.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from src.schemas import RecommendationResult


def _format(result: RecommendationResult) -> str:
    lines: list[str] = []
    p = result.profile
    header = (
        f"Profile: {p.name or '(unnamed)'} | "
        f"genre={p.favorite_genre} mood={p.favorite_mood} "
        f"energy={p.target_energy:.2f} acoustic={p.likes_acoustic}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for w in result.warnings:
        lines.append(f"  ! {w}")
    if result.used_fallback:
        lines.append("  ! at least one recommendation triggered the fallback guardrail.")
    for i, rec in enumerate(result.recommendations, start=1):
        lines.append(
            f"  #{i} {rec.song.title} - {rec.song.artist}  "
            f"[score={rec.score:.2f}, grounded={rec.grounded}]"
        )
        lines.append(f"      {rec.explanation}")
        if rec.citations:
            cites = ", ".join(c.passage_id for c in rec.citations)
            lines.append(f"      citations: {cites}")
    return "\n".join(lines)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="VibeFinder 2.0 demo runner")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--profiles",
        type=str,
        default="examples/demo_profiles.json",
    )
    args = parser.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))

    orch = Orchestrator()

    print("=" * 70)
    print("VibeFinder 2.0  -  RAG-enhanced music recommender")
    print("=" * 70)

    for raw in profiles:
        print()
        try:
            result = orch.recommend(raw, top_k=args.top_k)
            print(_format(result))
        except ValueError as e:
            name = raw.get("name", "(unnamed)")
            print(f"Profile: {name}")
            print(f"  ! input rejected by guardrail: {e}")

    print()
    print("=" * 70)
    print(f"Done. Ran {len(profiles)} profiles.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
