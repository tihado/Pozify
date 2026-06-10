from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pozify.steps.rep_signals import SignalSample


ExtremaKind = Literal["min", "max"]


@dataclass(frozen=True)
class Extrema:
    kind: ExtremaKind
    sample: SignalSample


@dataclass(frozen=True)
class RepSegment:
    start: SignalSample
    middle: SignalSample
    end: SignalSample
    amplitude: float


def find_local_extrema(samples: list[SignalSample], *, min_gap_frames: int = 4) -> list[Extrema]:
    if len(samples) < 2:
        return []

    extrema: list[Extrema] = []
    first, second = samples[0], samples[1]
    if first.value < second.value:
        extrema.append(Extrema(kind="min", sample=first))
    elif first.value > second.value:
        extrema.append(Extrema(kind="max", sample=first))

    for index in range(1, len(samples) - 1):
        previous = samples[index - 1]
        current = samples[index]
        following = samples[index + 1]
        if current.value >= previous.value and current.value > following.value:
            kind: ExtremaKind | None = "max"
        elif current.value <= previous.value and current.value < following.value:
            kind = "min"
        else:
            kind = None
        if kind is None:
            continue
        if (
            extrema
            and current.frame_index - extrema[-1].sample.frame_index < min_gap_frames
            and extrema[-1].kind == kind
        ):
            previous_extrema = extrema[-1]
            if kind == "max" and current.value > previous_extrema.sample.value:
                extrema[-1] = Extrema(kind=kind, sample=current)
            elif kind == "min" and current.value < previous_extrema.sample.value:
                extrema[-1] = Extrema(kind=kind, sample=current)
            continue
        extrema.append(Extrema(kind=kind, sample=current))

    previous, last = samples[-2], samples[-1]
    if previous.value < last.value:
        kind: ExtremaKind | None = "max"
    elif previous.value > last.value:
        kind = "min"
    else:
        kind = None
    if kind is not None:
        if (
            extrema
            and last.frame_index - extrema[-1].sample.frame_index < min_gap_frames
            and extrema[-1].kind == kind
        ):
            replace = kind == "max" and last.value > extrema[-1].sample.value
            replace = replace or (kind == "min" and last.value < extrema[-1].sample.value)
            if replace:
                extrema[-1] = Extrema(kind=kind, sample=last)
        else:
            extrema.append(Extrema(kind=kind, sample=last))
    return extrema


def segment_low_high_low(
    extrema: list[Extrema],
    *,
    min_cycle_frames: int,
    min_phase_frames: int,
    min_amplitude: float,
) -> list[RepSegment]:
    segments: list[RepSegment] = []
    index = 0
    while index <= len(extrema) - 3:
        first, second, third = extrema[index : index + 3]
        if (first.kind, second.kind, third.kind) != ("min", "max", "min"):
            index += 1
            continue

        if second.sample.frame_index - first.sample.frame_index < min_phase_frames:
            index += 1
            continue
        if third.sample.frame_index - second.sample.frame_index < min_phase_frames:
            index += 1
            continue
        if third.sample.frame_index - first.sample.frame_index < min_cycle_frames:
            index += 1
            continue

        left_amplitude = second.sample.value - first.sample.value
        right_amplitude = second.sample.value - third.sample.value
        amplitude = min(left_amplitude, right_amplitude)
        if amplitude < min_amplitude:
            index += 1
            continue

        segments.append(
            RepSegment(
                start=first.sample,
                middle=second.sample,
                end=third.sample,
                amplitude=amplitude,
            )
        )
        index += 2
    return segments
