from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.knowledge_cards import load_knowledge_cards, retrieve_cards


class KnowledgeCardTests(unittest.TestCase):
    def test_loads_required_starter_cards(self) -> None:
        cards = load_knowledge_cards()
        labels = {card.label for card in cards}
        self.assertIn("squat", labels)
        self.assertIn("push_up", labels)
        self.assertIn("shoulder_press", labels)
        self.assertIn("wide_grip_push_up", labels)
        self.assertIn("knee_push_up", labels)
        self.assertIn("shallow_depth", labels)
        self.assertIn("hip_sag", labels)
        self.assertIn("incomplete_lockout", labels)

    def test_retrieval_is_deterministic(self) -> None:
        retrieved = retrieve_cards(
            exercise="squat",
            variation="bodyweight_squat",
            issues=["shallow_depth"],
            goal="strength",
        )
        self.assertEqual(
            [card.type for card in retrieved.cards],
            ["exercise", "issue", "goal", "safety", "safety", "safety"],
        )
        self.assertEqual(retrieved.trace.requested_labels[0], "squat")


if __name__ == "__main__":
    unittest.main()
