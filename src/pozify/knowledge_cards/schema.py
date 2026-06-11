from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    type: str
    label: str
    aliases: tuple[str, ...] = ()
    summary: str = ""
    good_signals: tuple[str, ...] = ()
    common_misreads: tuple[str, ...] = ()
    coaching_cues: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    contraindicated_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalTrace:
    requested_labels: tuple[str, ...]
    matched_card_ids: tuple[str, ...]
    missing_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedCards:
    cards: tuple[KnowledgeCard, ...]
    trace: RetrievalTrace
