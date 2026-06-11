from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from pozify.knowledge_cards.schema import KnowledgeCard


CARDS_DIR = Path(__file__).resolve().parent / "cards"
CARD_FILES = (
    "exercises.json",
    "variations.json",
    "issues.json",
    "goals.json",
    "safety.json",
)


def _as_str_tuple(value: Any, field_name: str, card_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{card_id}.{field_name} must be a list of strings")
    return tuple(value)


def _coerce_card(raw_card: dict[str, Any]) -> KnowledgeCard:
    required = {"id", "type", "label", "summary"}
    missing = sorted(required - raw_card.keys())
    if missing:
        raise ValueError(f"knowledge card missing fields: {', '.join(missing)}")

    card_id = raw_card["id"]
    if not isinstance(card_id, str):
        raise ValueError("knowledge card id must be a string")
    if not isinstance(raw_card["type"], str) or not isinstance(raw_card["label"], str):
        raise ValueError(f"{card_id} type and label must be strings")
    if not isinstance(raw_card["summary"], str):
        raise ValueError(f"{card_id}.summary must be a string")

    return KnowledgeCard(
        id=card_id,
        type=raw_card["type"],
        label=raw_card["label"],
        aliases=_as_str_tuple(raw_card.get("aliases", []), "aliases", card_id),
        summary=raw_card["summary"],
        good_signals=_as_str_tuple(raw_card.get("good_signals", []), "good_signals", card_id),
        common_misreads=_as_str_tuple(raw_card.get("common_misreads", []), "common_misreads", card_id),
        coaching_cues=_as_str_tuple(raw_card.get("coaching_cues", []), "coaching_cues", card_id),
        safety_notes=_as_str_tuple(raw_card.get("safety_notes", []), "safety_notes", card_id),
        contraindicated_claims=_as_str_tuple(
            raw_card.get("contraindicated_claims", []),
            "contraindicated_claims",
            card_id,
        ),
    )


@lru_cache(maxsize=1)
def load_knowledge_cards() -> tuple[KnowledgeCard, ...]:
    cards: list[KnowledgeCard] = []
    seen_ids: set[str] = set()
    for filename in CARD_FILES:
        payload = json.loads((CARDS_DIR / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{filename} must contain a list of cards")
        for raw_card in payload:
            if not isinstance(raw_card, dict):
                raise ValueError(f"{filename} contains a non-object card")
            card = _coerce_card(raw_card)
            if card.id in seen_ids:
                raise ValueError(f"duplicate knowledge card id: {card.id}")
            seen_ids.add(card.id)
            cards.append(card)
    return tuple(cards)
