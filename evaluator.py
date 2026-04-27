"""Evaluation harness for the rubric's stretch "Test Harness" feature.

Runs every demo profile through the orchestrator and reports four metrics:

  - precision_at_k:   fraction of recommendations whose genre OR mood matches
                      the user profile (a coarse but interpretable proxy).
  - diversity_at_k:   number of distinct artists in the top-k.
  - groundedness:     fraction of recommendations whose generated explanation
                      cites at least one retrieved passage.
  - guardrail_ok:     no hard validation failures during the run.

Each profile's metrics are printed individually, then summarized across the
suite with a pass/fail decision per profile based on configurable thresholds.

Usage:
    python evaluator.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.orchestrator import Orchestrator
from src.schemas import RecommendationResult


def precision_at_k(result: RecommendationResult) -> float:
    if not result.recommendations:
        return 0.0
    profile = result.profile
    hits = sum(
        1
        for r in result.recommendations
        if r.song.genre == profile.favorite_genre
        or r.song.mood == profile.favorite_mood
    )
    return hits / len(result.recommendations)


def diversity_at_k(result: RecommendationResult) -> float:
    if not result.recommendations:
        return 0.0
    artists = {r.song.artist for r in result.recommendations}
    return len(artists) / len(result.recommendations)


def groundedness(result: RecommendationResult) -> float:
    if not result.recommendations:
        return 0.0
    grounded = sum(1 for r in result.recommendations if r.grounded)
    return grounded / len(result.recommendations)


THRESHOLDS = {
    "precision_at_k": 0.34,
    "diversity_at_k": 0.34,
    "groundedness": 0.5,
}


def evaluate(result: RecommendationResult) -> dict:
    metrics = {
        "precision_at_k": precision_at_k(result),
        "diversity_at_k": diversity_at_k(result),
        "groundedness": groundedness(result),
        "n": len(result.recommendations),
        "used_fallback": result.used_fallback,
    }
    metrics["pass"] = all(
        metrics[k] >= v for k, v in THRESHOLDS.items()
    )
    return metrics


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="VibeFinder 2.0 evaluation harness")
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
    print("Evaluation harness")
    print(f"Thresholds: {THRESHOLDS}")
    print("=" * 70)

    rows: List[dict] = []
    n_passed = 0
    n_total = 0

    for raw in profiles:
        name = raw.get("name", "(unnamed)")
        try:
            result = orch.recommend(raw, top_k=args.top_k)
        except ValueError as e:
            print(f"\n[REJECTED] {name}: {e}")
            rows.append({"name": name, "pass": False, "error": str(e)})
            n_total += 1
            continue

        m = evaluate(result)
        m["name"] = name
        rows.append(m)
        n_total += 1
        if m["pass"]:
            n_passed += 1

        verdict = "PASS" if m["pass"] else "FAIL"
        print(
            f"\n[{verdict}] {name}\n"
            f"   precision@k = {m['precision_at_k']:.2f}\n"
            f"   diversity@k = {m['diversity_at_k']:.2f}\n"
            f"   groundedness = {m['groundedness']:.2f}\n"
            f"   used_fallback = {m['used_fallback']}"
        )

    print("\n" + "=" * 70)
    print(f"Summary: {n_passed}/{n_total} profiles passed all thresholds.")
    print("=" * 70)

    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
