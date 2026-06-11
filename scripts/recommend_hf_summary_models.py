from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pozify import load_local_env  # type: ignore[attr-defined]
from pozify.hf_summary_model_recommender import fetch_router_models, recommend_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Hugging Face router chat models and recommend the best options for Pozify summary generation."
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of top recommendations to print.")
    parser.add_argument(
        "--show-candidates",
        type=int,
        default=10,
        help="How many ranked candidates to print after the top recommendations.",
    )
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()
    records = fetch_router_models()
    recommendations = recommend_models(records, limit=args.limit)
    ranked_candidates = recommend_models(records, limit=max(args.show_candidates, args.limit))

    if not recommendations:
        raise SystemExit("No chat/instruct models were found in the Hugging Face router response.")

    print("Top Hugging Face summary model recommendations:\n")
    for index, candidate in enumerate(recommendations, start=1):
        context = candidate.context_length or "unknown"
        provider = candidate.provider_hint or "auto"
        reasons = "; ".join(candidate.reasons[:4]) or "general chat/instruct fit"
        print(f"{index}. {candidate.model_id}")
        print(f"   provider: {provider}")
        print(f"   context: {context}")
        print(f"   score: {candidate.score}")
        print(f"   why: {reasons}")
        print(
            "   run: "
            f"POZIFY_SUMMARY_PROVIDER=slm_cloud "
            f"POZIFY_SUMMARY_CLOUD_MODEL='{candidate.model_id}' "
            "uv run python app.py"
        )
        print()

    print("Additional ranked candidates:\n")
    for candidate in ranked_candidates:
        print(
            f"- {candidate.model_id} | score={candidate.score} | "
            f"context={candidate.context_length or 'unknown'} | "
            f"provider={candidate.provider_hint or 'auto'}"
        )


if __name__ == "__main__":
    main()
