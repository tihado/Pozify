from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import PoseFrame, PoseSequence, UserProfile, validate_contract
from pozify.ml.exercise_router_evaluation import evaluate_router_predictions, select_router_candidate
from pozify.ml.exercise_router_features import (
    ROUTER_LABELS,
    extract_router_windows,
    window_tensor_feature_names,
    window_vector_feature_names,
)
from pozify.ml.exercise_router_inference import (
    RouterModelBundle,
    WindowRouterPrediction,
    aggregate_window_predictions,
    load_router_model,
)
from pozify.steps import exercise_classifier
from pozify.steps.pose_backends.landmarks import LANDMARK_NAMES


def _profile(intended_exercise: str = "auto") -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise=intended_exercise,
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


def _base_landmarks(visibility: float) -> dict[str, dict[str, float]]:
    return {
        name: {
            "x": 0.5,
            "y": 0.5,
            "z": 0.0,
            "visibility": visibility,
            "normalized_x": 0.0,
            "normalized_y": 0.0,
            "normalized_z": 0.0,
        }
        for name in LANDMARK_NAMES
    }


def _set_landmark(
    landmarks: dict[str, dict[str, float]],
    name: str,
    x: float,
    y: float,
    visibility: float,
) -> None:
    landmarks[name].update(
        {
            "x": x,
            "y": y,
            "z": 0.0,
            "visibility": visibility,
            "normalized_x": x - 0.5,
            "normalized_y": y - 0.5,
            "normalized_z": 0.0,
        }
    )


def _landmarks_for_exercise(exercise: str, phase: float, visibility: float) -> dict[str, dict[str, float]]:
    landmarks = _base_landmarks(visibility)
    wave = (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
    if exercise == "squat":
        _set_landmark(landmarks, "left_shoulder", 0.42, 0.3 + wave * 0.04, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.58, 0.3 + wave * 0.04, visibility)
        _set_landmark(landmarks, "left_hip", 0.43, 0.52 + wave * 0.16, visibility)
        _set_landmark(landmarks, "right_hip", 0.57, 0.52 + wave * 0.16, visibility)
        _set_landmark(landmarks, "left_knee", 0.42 + wave * 0.05, 0.72, visibility)
        _set_landmark(landmarks, "right_knee", 0.58 - wave * 0.05, 0.72, visibility)
        _set_landmark(landmarks, "left_ankle", 0.41, 0.92, visibility)
        _set_landmark(landmarks, "right_ankle", 0.59, 0.92, visibility)
    elif exercise == "shoulder_press":
        _set_landmark(landmarks, "left_shoulder", 0.42, 0.42, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.58, 0.42, visibility)
        _set_landmark(landmarks, "left_hip", 0.43, 0.7, visibility)
        _set_landmark(landmarks, "right_hip", 0.57, 0.7, visibility)
        _set_landmark(landmarks, "left_elbow", 0.4 - wave * 0.05, 0.62 - wave * 0.12, visibility)
        _set_landmark(landmarks, "right_elbow", 0.6 + wave * 0.05, 0.62 - wave * 0.12, visibility)
        _set_landmark(landmarks, "left_wrist", 0.4, 0.78 - wave * 0.34, visibility)
        _set_landmark(landmarks, "right_wrist", 0.6, 0.78 - wave * 0.34, visibility)
    else:
        _set_landmark(landmarks, "left_shoulder", 0.3, 0.38 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_shoulder", 0.7, 0.38 + wave * 0.14, visibility)
        _set_landmark(landmarks, "left_elbow", 0.36, 0.47 + wave * 0.08, visibility)
        _set_landmark(landmarks, "right_elbow", 0.64, 0.47 + wave * 0.08, visibility)
        _set_landmark(landmarks, "left_wrist", 0.34, 0.52, visibility)
        _set_landmark(landmarks, "right_wrist", 0.66, 0.52, visibility)
        _set_landmark(landmarks, "left_hip", 0.42, 0.48 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_hip", 0.58, 0.48 + wave * 0.14, visibility)
        _set_landmark(landmarks, "left_ankle", 0.44, 0.56 + wave * 0.14, visibility)
        _set_landmark(landmarks, "right_ankle", 0.56, 0.56 + wave * 0.14, visibility)
    return landmarks


def _sequence(exercise: str = "push_up", frame_count: int = 45, visibility: float = 0.95) -> PoseSequence:
    frames = [
        PoseFrame(
            frame_index=index,
            timestamp_sec=round(index / 30.0, 3),
            landmarks=_landmarks_for_exercise(exercise, index / 24.0, visibility),
            world_landmarks={},
            pose_quality={"mean_visibility": visibility, "normalized": True},
        )
        for index in range(frame_count)
    ]
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0 if visibility >= 0.2 else 0.4,
    )


class _FakePushUpModel:
    classes_ = np.asarray(ROUTER_LABELS)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.03, 0.91, 0.02, 0.04]]), (values.shape[0], 1))


class ExerciseRouterFeatureTests(unittest.TestCase):
    def test_extracts_windows_for_supported_exercise_motion(self) -> None:
        for exercise in ("squat", "push_up", "shoulder_press"):
            with self.subTest(exercise=exercise):
                windows = extract_router_windows(_sequence(exercise))

                self.assertGreaterEqual(len(windows), 2)
                self.assertEqual(windows[0].tensor.shape, (30, len(window_tensor_feature_names())))
                self.assertEqual(windows[0].vector.shape, (len(window_vector_feature_names()),))
                self.assertGreater(windows[0].mean_visibility, 0.9)

    def test_empty_and_low_visibility_sequences_do_not_produce_windows(self) -> None:
        empty = PoseSequence(
            frames=[],
            normalized=True,
            smoothing_method="none",
            pose_valid_ratio=0.0,
        )

        self.assertEqual(extract_router_windows(empty), [])
        self.assertEqual(extract_router_windows(_sequence(visibility=0.05)), [])


class ExerciseRouterAggregationTests(unittest.TestCase):
    def test_aggregates_confident_windows(self) -> None:
        predictions = [
            WindowRouterPrediction(0.0, 1.0, "push_up", 0.91, {"squat": 0.03, "push_up": 0.91, "shoulder_press": 0.02, "unknown": 0.04}),
            WindowRouterPrediction(0.5, 1.5, "push_up", 0.88, {"squat": 0.05, "push_up": 0.88, "shoulder_press": 0.02, "unknown": 0.05}),
        ]

        aggregated = aggregate_window_predictions(predictions)

        self.assertEqual(aggregated.label, "push_up")
        self.assertFalse(aggregated.fallback_required)
        self.assertGreaterEqual(aggregated.confidence, 0.85)

    def test_low_confidence_and_inconsistent_windows_fallback_to_unknown(self) -> None:
        low_confidence = [
            WindowRouterPrediction(0.0, 1.0, "push_up", 0.45, {"squat": 0.25, "push_up": 0.45, "shoulder_press": 0.15, "unknown": 0.15}),
            WindowRouterPrediction(0.5, 1.5, "push_up", 0.48, {"squat": 0.22, "push_up": 0.48, "shoulder_press": 0.15, "unknown": 0.15}),
        ]
        inconsistent = [
            WindowRouterPrediction(0.0, 1.0, "push_up", 0.8, {"squat": 0.05, "push_up": 0.8, "shoulder_press": 0.1, "unknown": 0.05}),
            WindowRouterPrediction(0.5, 1.5, "squat", 0.8, {"squat": 0.8, "push_up": 0.05, "shoulder_press": 0.1, "unknown": 0.05}),
            WindowRouterPrediction(1.0, 2.0, "shoulder_press", 0.8, {"squat": 0.1, "push_up": 0.05, "shoulder_press": 0.8, "unknown": 0.05}),
        ]

        self.assertTrue(aggregate_window_predictions(low_confidence).fallback_required)
        self.assertEqual(aggregate_window_predictions(low_confidence).label, "unknown")
        self.assertTrue(aggregate_window_predictions(inconsistent).fallback_required)
        self.assertEqual(aggregate_window_predictions(inconsistent).label, "unknown")


class ExerciseRouterModelLoadingTests(unittest.TestCase):
    def test_selection_file_controls_active_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifact_path = model_dir / "selected.joblib"
            joblib.dump(
                {
                    "model": _FakePushUpModel(),
                    "labels": ["unknown", "squat", "push_up", "shoulder_press"],
                    "model_kind": "baseline",
                },
                artifact_path,
            )
            (model_dir / "router_selection.json").write_text(
                json.dumps({"selected_artifact": artifact_path.name}),
                encoding="utf-8",
            )

            bundle = load_router_model(model_dir)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.labels, tuple(ROUTER_LABELS))
        result = exercise_classifier.run(
            _sequence("push_up"),
            _profile("auto"),
            mock=False,
            model_bundle=bundle,
        )
        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)


class ExerciseClassifierStepTests(unittest.TestCase):
    def test_manual_override_bypasses_model_and_validates_contract(self) -> None:
        result = exercise_classifier.run(
            _sequence("squat"),
            _profile("push_up"),
            mock=False,
        )

        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)
        self.assertEqual(result.confidence, 0.98)
        validate_contract("exercise_classification.json", result)

    def test_missing_model_falls_back_to_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = exercise_classifier.run(
                _sequence("push_up"),
                _profile("auto"),
                mock=False,
                model_dir=Path(temp_dir),
            )

        self.assertEqual(result.exercise, "unknown")
        self.assertTrue(result.fallback_required)
        validate_contract("exercise_classification.json", result)

    def test_fake_model_routes_and_persists_window_predictions(self) -> None:
        result = exercise_classifier.run(
            _sequence("push_up"),
            _profile("auto"),
            mock=False,
            model_bundle=RouterModelBundle(model=_FakePushUpModel(), labels=ROUTER_LABELS),
        )

        self.assertEqual(result.exercise, "push_up")
        self.assertFalse(result.fallback_required)
        self.assertGreater(result.confidence, 0.9)
        self.assertGreater(len(result.window_predictions), 0)
        self.assertEqual(
            sorted(result.window_predictions[0]),
            ["confidence", "end_sec", "label", "start_sec"],
        )
        validate_contract("exercise_classification.json", result)

    def test_low_pose_valid_ratio_falls_back_before_model_inference(self) -> None:
        sequence = _sequence("push_up", visibility=0.95)
        sequence = PoseSequence(
            frames=sequence.frames,
            normalized=sequence.normalized,
            smoothing_method=sequence.smoothing_method,
            pose_valid_ratio=0.4,
        )

        result = exercise_classifier.run(
            sequence,
            _profile("auto"),
            mock=False,
            model_bundle=RouterModelBundle(model=_FakePushUpModel(), labels=ROUTER_LABELS),
        )

        self.assertEqual(result.exercise, "unknown")
        self.assertTrue(result.fallback_required)


class ExerciseRouterEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_accuracy_and_confusion_matrix(self) -> None:
        evaluation = evaluate_router_predictions(
            ["squat", "push_up", "shoulder_press", "unknown"],
            ["squat", "push_up", "unknown", "unknown"],
        )

        self.assertEqual(evaluation.accuracy, 0.75)
        self.assertEqual(evaluation.confusion_matrix["shoulder_press"]["unknown"], 1)
        self.assertEqual(evaluation.unknown_rejection_rate, 1.0)

    def test_selects_temporal_only_when_metrics_win(self) -> None:
        baseline = {"name": "baseline", "accuracy": 0.91, "unknown_rejection_rate": 0.8}
        temporal = {"name": "temporal", "accuracy": 0.92, "unknown_rejection_rate": 0.7}

        self.assertEqual(select_router_candidate([baseline, temporal]), temporal)

    def test_selects_baseline_on_metric_tie(self) -> None:
        baseline = {"name": "baseline", "accuracy": 0.91, "unknown_rejection_rate": 0.8}
        temporal = {"name": "temporal", "accuracy": 0.91, "unknown_rejection_rate": 0.8}

        self.assertEqual(select_router_candidate([baseline, temporal]), baseline)


if __name__ == "__main__":
    unittest.main()
