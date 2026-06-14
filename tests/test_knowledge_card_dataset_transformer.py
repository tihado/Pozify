from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.knowledge_card_dataset_transformer import (
    load_dataset_rows,
    transform_dataset_rows,
    write_card_pack,
)


class KnowledgeCardDatasetTransformerTests(unittest.TestCase):
    def test_transform_dataset_rows_maps_common_fields(self) -> None:
        pack = transform_dataset_rows(
            [
                {
                    "name": "Push-Up",
                    "aliases": ["Push Up", "Pushup"],
                    "category": "Strength",
                    "equipment": ["Bodyweight"],
                    "primary_muscles": ["Chest", "Triceps"],
                    "secondary_muscles": ["Shoulders"],
                    "instructions": [
                        "Keep a straight line from shoulders through ankles.",
                        "Lower until the bottom position is controlled.",
                    ],
                    "description": "A classic upper-body bodyweight exercise.",
                }
            ],
            source_dataset="demo/exercises",
        )

        self.assertEqual(pack["source_dataset"], "demo/exercises")
        self.assertEqual(pack["card_count"], 1)
        card = pack["cards"][0]
        self.assertEqual(card["card_id"], "exercise:push_up")
        self.assertEqual(card["labels"], ["push_up"])
        self.assertIn("Bodyweight", card["summary"])
        self.assertIn("Chest", " ".join(card["allowed_interpretations"]))
        self.assertEqual(
            card["coaching_points"][0],
            "Keep a straight line from shoulders through ankles.",
        )

    def test_load_dataset_rows_supports_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "Squat", "equipment": ["Bodyweight"]}),
                        json.dumps({"name": "Shoulder Press", "equipment": ["Dumbbell"]}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = load_dataset_rows(dataset_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "Squat")

    def test_write_card_pack_writes_transform_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "dataset.json"
            output_path = Path(temp_dir) / "pack.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "Exercise_Name": "Shoulder Press",
                            "Equipment": ["Dumbbell"],
                            "Target Muscle": ["Shoulders"],
                            "Instructions": "Press the weights overhead.\nLower with control.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            pack = write_card_pack(
                input_path=input_path,
                output_path=output_path,
                source_dataset="demo/export",
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(pack["card_count"], 1)
        self.assertEqual(written["source_dataset"], "demo/export")
        self.assertEqual(written["cards"][0]["card_id"], "exercise:shoulder_press")
        self.assertEqual(
            written["cards"][0]["coaching_points"],
            ["Press the weights overhead.", "Lower with control."],
        )


if __name__ == "__main__":
    unittest.main()
