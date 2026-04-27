"""CLI entry point: `python -m src.main`.

Reads a single profile from `--profile` (path to JSON) or accepts inline flags
and prints recommendations with explanations and citations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .orchestrator import Orchestrator
from .schemas import RecommendationResult


def _format_result(result: RecommendationResult) -> str:
    lines: list[str] = []
    p = result.profile
    header = f"Profile: {p.name or '(unnamed)'} | genre={p.favorite_genre} mood={p.favorite_mood} energy={p.target_energy:.2f} acoustic={p.likes_acoustic}"
    lines.append(header)
    lines.append("=" * len(header))

    for w in result.warnings:
        lines.append(f"  ! warning: {w}")
    if result.used_fallback:
        lines.append("  ! note: at least one recommendation used the guardrail fallback path.")

    for i, rec in enumerate(result.recommendations, start=1):
        lines.append("")
        lines.append(
            f"#{i}  {rec.song.title} - {rec.song.artist}  "
            f"[score={rec.score:.2f}, grounded={rec.grounded}]"
        )
        lines.append(f"     {rec.explanation}")
        if rec.citations:
            cites = ", ".join(f"{c.passage_id}" for c in rec.citations)
            lines.append(f"     citations: {cites}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="VibeFinder 2.0 - RAG music recommender")
    parser.add_argument("--profile", type=str, help="Path to a JSON profile file.")
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--mood", type=str, default=None)
    parser.add_argument("--energy", type=float, default=None)
    parser.add_argument("--acoustic", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args(argv)

    if args.profile:
        raw = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    else:
        if not all([args.genre, args.mood, args.energy is not None]):
            parser.error("Provide --profile or all of --genre, --mood, --energy.")
        raw = {
            "name": args.name,
            "favorite_genre": args.genre,
            "favorite_mood": args.mood,
            "target_energy": args.energy,
            "likes_acoustic": args.acoustic,
        }

    orch = Orchestrator()
    result = orch.recommend(raw, top_k=args.top_k)
    print(_format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
