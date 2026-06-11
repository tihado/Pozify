from __future__ import annotations

from pozify.knowledge_cards.loader import load_knowledge_cards
from pozify.knowledge_cards.schema import KnowledgeCard, RetrievedCards, RetrievalTrace


CARD_TYPE_ORDER = ("exercise", "variation", "issue", "goal", "safety")


def _matches(card: KnowledgeCard, label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return False
    if card.label.lower() == normalized:
        return True
    return normalized in {alias.lower() for alias in card.aliases}


def retrieve_cards(
    *,
    exercise: str,
    variation: str,
    issues: list[str],
    goal: str,
) -> RetrievedCards:
    cards = load_knowledge_cards()
    requested_labels = [
        exercise,
        variation,
        *sorted(set(issues)),
        goal,
        "no_diagnosis",
        "no_injury_prevention_claim",
        "confidence_language",
    ]

    matched_cards: list[KnowledgeCard] = []
    matched_ids: set[str] = set()
    missing_labels: list[str] = []

    for label in requested_labels:
        match = next((card for card in cards if _matches(card, label)), None)
        if match is None:
            missing_labels.append(label)
            continue
        if match.id in matched_ids:
            continue
        matched_ids.add(match.id)
        matched_cards.append(match)

    matched_cards.sort(
        key=lambda card: (
            CARD_TYPE_ORDER.index(card.type) if card.type in CARD_TYPE_ORDER else len(CARD_TYPE_ORDER),
            card.id,
        )
    )
    return RetrievedCards(
        cards=tuple(matched_cards),
        trace=RetrievalTrace(
            requested_labels=tuple(requested_labels),
            matched_card_ids=tuple(card.id for card in matched_cards),
            missing_labels=tuple(missing_labels),
        ),
    )
