from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Exercise = Literal["squat", "push_up", "shoulder_press", "unknown"]


@dataclass(frozen=True)
class UserProfile:
    goal: str
    experience_level: str
    intended_exercise: str = "auto"
    intended_variation: str | None = None
    known_limitations: list[str] = field(default_factory=list)
    equipment: str = "unknown"


@dataclass(frozen=True)
class VideoManifest:
    video_path: str | None
    fps: float
    duration_sec: float
    total_frames: int
    sampled_frames: int
    quality_warnings: list[str]
    analysis_allowed: bool


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_sec: float
    landmarks: dict[str, dict[str, float]]
    world_landmarks: dict[str, dict[str, float]]
    pose_quality: dict[str, Any]


@dataclass(frozen=True)
class PoseSequence:
    frames: list[PoseFrame]
    normalized: bool
    smoothing_method: str
    pose_valid_ratio: float


@dataclass(frozen=True)
class ExerciseClassification:
    exercise: Exercise
    confidence: float
    window_predictions: list[dict[str, Any]]
    fallback_required: bool


@dataclass(frozen=True)
class Rep:
    rep_id: int
    start_frame: int
    mid_frame: int
    end_frame: int
    start_sec: float
    mid_sec: float
    end_sec: float


@dataclass(frozen=True)
class Reps:
    exercise: Exercise
    reps: list[Rep]
    partial_reps: list[dict[str, Any]]


@dataclass(frozen=True)
class RepAnalysisItem:
    rep_id: int
    duration_sec: float
    range_of_motion_score: float
    stability_score: float
    symmetry_score: float
    metrics: dict[str, Any]
    variation_hints: list[str]


@dataclass(frozen=True)
class RepAnalysis:
    exercise: Exercise
    items: list[RepAnalysisItem]
    aggregate_metrics: dict[str, Any]


@dataclass(frozen=True)
class Variation:
    exercise: Exercise
    detected_variation: str
    variation_confidence: float
    not_issues: list[str]


@dataclass(frozen=True)
class IssueMarker:
    rep_id: int
    issue: str
    severity: float
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    affected_joints: list[str]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class IssueMarkers:
    issues: list[IssueMarker]


@dataclass(frozen=True)
class CoachSummary:
    summary: str
    what_went_well: list[str]
    main_findings: list[str]
    variation_explanation: str
    top_fixes: list[str]
    next_session_plan: list[str]
    confidence_notes: list[str]


@dataclass(frozen=True)
class Verification:
    passed: bool
    checks: dict[str, bool]
    notes: list[str]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value

