from __future__ import annotations

from dataclasses import dataclass

from pozify.contracts import ExerciseClassification, GOALS, IssueMarkers, UserProfile, Variation


CARD_TYPE_ORDER = {
    "exercise": 0,
    "variation": 1,
    "issue": 2,
    "goal": 3,
    "safety_rule": 4,
}


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    card_type: str
    labels: tuple[str, ...]
    title: str
    summary: str
    evidence_rules: tuple[str, ...]
    coaching_points: tuple[str, ...]
    allowed_interpretations: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    related_cards: tuple[str, ...] = ()


def _card(
    card_id: str,
    card_type: str,
    labels: tuple[str, ...],
    title: str,
    summary: str,
    evidence_rules: tuple[str, ...],
    coaching_points: tuple[str, ...],
    *,
    allowed_interpretations: tuple[str, ...] = (),
    forbidden_claims: tuple[str, ...] = (),
    related_cards: tuple[str, ...] = (),
) -> KnowledgeCard:
    return KnowledgeCard(
        card_id=card_id,
        card_type=card_type,
        labels=labels,
        title=title,
        summary=summary,
        evidence_rules=evidence_rules,
        coaching_points=coaching_points,
        allowed_interpretations=allowed_interpretations,
        forbidden_claims=forbidden_claims,
        related_cards=related_cards,
    )


CARD_REGISTRY: tuple[KnowledgeCard, ...] = (
    _card(
        "exercise:squat",
        "exercise",
        ("squat",),
        "Squat",
        "A squat summary should describe depth, balance, and torso position "
        "from structured rep evidence.",
        (
            "Use rep analysis and issue markers instead of inferring directly from the video.",
            "Valid stance variations should not be framed as faults by default.",
        ),
        (
            "Call out depth, stance, and torso control only when they appear "
            "in structured evidence.",
            "Keep fixes simple and specific to the detected issue labels.",
        ),
    ),
    _card(
        "exercise:push_up",
        "exercise",
        ("push_up",),
        "Push-up",
        "A push-up summary should focus on body line, depth, and rep consistency "
        "from structured evidence.",
        (
            "Treat hand placement or knee support as variation context when the "
            "variation detector marks them as not-issues.",
            "Do not infer shoulder or wrist pain from the movement.",
        ),
        (
            "Explain whether the set looked controlled before suggesting changes.",
            "Use issue labels such as `hip_sag` or `incomplete_depth` only "
            "when they are present in JSON.",
        ),
    ),
    _card(
        "exercise:shoulder_press",
        "exercise",
        ("shoulder_press",),
        "Shoulder Press",
        "A shoulder press summary should focus on lockout, symmetry, and rep "
        "consistency from structured evidence.",
        (
            "Partial range can be a valid variation context and should not be "
            "automatically overcorrected.",
            "Use rep analysis and issue markers instead of diagnosing shoulder limitations.",
        ),
        (
            "Separate partial range context from incomplete lockout issue markers.",
            "Use `asymmetry` only when it is explicitly present in JSON.",
        ),
    ),
    _card(
        "variation:wide_grip_push_up",
        "variation",
        ("wide_grip_push_up", "wide_hand_placement"),
        "Wide-Grip Push-up",
        "A wide-grip push-up is a valid push-up variation when detected by the "
        "variation step.",
        (
            "If `wide_hand_placement` appears in not_issues, treat hand width "
            "as context, not a fault.",
        ),
        (
            "Acknowledge the wide-grip setup without asking the athlete to "
            "normalize it unless another issue requires it.",
        ),
        allowed_interpretations=("Variation, not automatically an issue.",),
    ),
    _card(
        "variation:knee_push_up",
        "variation",
        ("knee_push_up", "knee_contact"),
        "Knee Push-up",
        "A knee push-up is a valid push-up variation when knee support is intentionally detected.",
        (
            "If `knee_contact` appears in not_issues, do not correct knee support as an error.",
        ),
        (
            "Explain the movement as a valid regression or variation rather than a mistake.",
        ),
        allowed_interpretations=("Variation, not automatically an issue.",),
    ),
    _card(
        "issue:shallow_depth",
        "issue",
        ("shallow_depth",),
        "Shallow Depth",
        "The squat bottom position stayed above the expected depth threshold in the issue markers.",
        (
            "Only mention this issue when `shallow_depth` exists in `issue_markers.json`.",
        ),
        (
            "Sit slightly deeper before standing up.",
            "Slow the bottom portion so depth stays consistent.",
        ),
    ),
    _card(
        "issue:hip_sag",
        "issue",
        ("hip_sag",),
        "Hip Sag",
        "The push-up body line dropped below the body-line threshold across a sustained interval.",
        (
            "Only mention this issue when `hip_sag` exists in `issue_markers.json`.",
        ),
        (
            "Keep shoulders, hips, and ankles moving as one line.",
            "Reduce speed if body line drops on later reps.",
        ),
    ),
    _card(
        "issue:incomplete_lockout",
        "issue",
        ("incomplete_lockout",),
        "Incomplete Lockout",
        "The elbows did not reach the lockout threshold at the top of the shoulder press.",
        (
            "Only mention this issue when `incomplete_lockout` exists in `issue_markers.json`.",
        ),
        (
            "Finish each rep by reaching a cleaner top position.",
            "Use a slower press so the top range stays consistent.",
        ),
    ),
    _card(
        "issue:incomplete_depth",
        "issue",
        ("incomplete_depth",),
        "Incomplete Depth",
        "The push-up bottom position stayed above the depth threshold at the bottom of the rep.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Lower a bit more at the bottom if control stays clean.",
            "Use slower reps to make bottom depth repeatable.",
        ),
    ),
    _card(
        "issue:knee_valgus",
        "issue",
        ("knee_valgus",),
        "Knee Valgus",
        "The knees tracked inward relative to the ankles beyond the configured threshold.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Keep the knees tracking more evenly over the feet.",
            "Use a slightly slower descent so knee path stays consistent.",
        ),
    ),
    _card(
        "issue:excessive_torso_lean",
        "issue",
        ("excessive_torso_lean",),
        "Excessive Torso Lean",
        "The torso lean exceeded the configured threshold near the squat bottom.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Keep the chest taller through the bottom.",
            "Use a controlled descent so the torso angle stays steadier.",
        ),
    ),
    _card(
        "issue:asymmetry",
        "issue",
        ("asymmetry",),
        "Asymmetry",
        "The left-right wrist height difference exceeded the configured symmetry threshold.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Try to finish both sides at a more even height.",
            "Use a slower tempo to keep both arms in sync.",
        ),
    ),
    _card(
        "goal:strength",
        "goal",
        ("strength",),
        "Strength Goal",
        "Strength-oriented coaching should prioritize a few high-value fixes over many cues.",
        ("Keep the plan focused and repeatable.",),
        ("Use 1 to 2 form priorities for the next session.",),
    ),
    _card(
        "goal:hypertrophy",
        "goal",
        ("hypertrophy",),
        "Hypertrophy Goal",
        "Hypertrophy-oriented coaching should emphasize repeatable reps and manageable fixes.",
        ("Keep cues practical for multi-rep sets.",),
        ("Prioritize consistency over perfect-looking single reps.",),
    ),
    _card(
        "goal:endurance",
        "goal",
        ("endurance",),
        "Endurance Goal",
        "Endurance-oriented coaching should emphasize repeatability across the full set.",
        ("Call out late-set drift when the rep analysis shows it.",),
        ("Use pacing and consistency cues for the next session.",),
    ),
    _card(
        "goal:mobility",
        "goal",
        ("mobility",),
        "Mobility Goal",
        "Mobility-oriented coaching should stay descriptive and avoid medical claims.",
        ("Describe range findings without diagnosing restrictions.",),
        ("Use easy controlled reps next session to compare range consistency.",),
    ),
    _card(
        "goal:beginner_practice",
        "goal",
        ("beginner_practice",),
        "Beginner Practice Goal",
        "Beginner practice coaching should stay simple, encouraging, and concrete.",
        ("Limit the number of corrections in a single summary.",),
        ("Pick the top one or two form priorities for next time.",),
    ),
    _card(
        "safety:no_diagnosis",
        "safety_rule",
        ("no_diagnosis",),
        "No Diagnosis",
        "The summary must not diagnose pain, injury, imbalance, mobility deficits, or pathology.",
        ("Do not use diagnostic language.",),
        ("Use uncertainty language when evidence is limited.",),
        forbidden_claims=("diagnosis", "injury", "pathology", "medical assessment"),
    ),
    _card(
        "safety:no_injury_prevention_claim",
        "safety_rule",
        ("no_injury_prevention_claim",),
        "No Injury Prevention Claim",
        "The summary must not claim that a cue will prevent injury.",
        ("Do not promise injury prevention.",),
        ("Keep coaching language descriptive and performance-focused.",),
        forbidden_claims=("injury prevention", "prevent injury"),
    ),
    _card(
        "safety:grounded_only",
        "safety_rule",
        ("grounded_only",),
        "Grounded Only",
        "The summary must explain only the structured evidence and retrieved knowledge cards.",
        ("Do not infer new issues that are absent from JSON.",),
        ("State confidence limits when the evidence is thin.",),
    ),
    _card(
        "safety:variation_not_issue",
        "safety_rule",
        ("variation_not_issue",),
        "Variation Is Not Automatically An Issue",
        "A detected variation or listed not-issue should not be overcorrected as a mistake.",
        ("Keep valid variation context separate from issue language.",),
        ("Explain why the variation is treated as context when needed.",),
    ),
)


CARDS_BY_ID = {card.card_id: card for card in CARD_REGISTRY}
CARDS_BY_LABEL = {
    label: card
    for card in CARD_REGISTRY
    for label in card.labels
}
KNOWN_ISSUE_LABELS = frozenset(
    label for card in CARD_REGISTRY if card.card_type == "issue" for label in card.labels
)
KNOWN_VARIATION_LABELS = frozenset(
    label for card in CARD_REGISTRY if card.card_type == "variation" for label in card.labels
)


def get_card_by_label(label: str) -> KnowledgeCard | None:
    return CARDS_BY_LABEL.get(label)


def cards_for_labels(labels: list[str] | tuple[str, ...]) -> list[KnowledgeCard]:
    cards = {
        card.card_id: card
        for label in labels
        for card in [get_card_by_label(label)]
        if card is not None
    }
    return sorted(
        cards.values(),
        key=lambda card: (CARD_TYPE_ORDER.get(card.card_type, 99), card.card_id),
    )


def retrieve_cards(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    variation: Variation,
    issues: IssueMarkers,
) -> list[KnowledgeCard]:
    labels = [
        classification.exercise,
        variation.detected_variation,
        *variation.not_issues,
        *[issue.issue for issue in issues.issues],
        profile.goal,
        "no_diagnosis",
        "no_injury_prevention_claim",
        "grounded_only",
        "variation_not_issue",
    ]
    return cards_for_labels(labels)


def supported_goal_labels() -> frozenset[str]:
    return frozenset(GOALS)
