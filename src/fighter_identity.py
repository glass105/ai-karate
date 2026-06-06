"""Recover a named fighter across tracker ID changes."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, sqrt
from typing import Iterable, Sequence

Descriptor = Sequence[float]
Keypoints = Sequence[Sequence[float]]


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    pose: Descriptor = ()
    appearance: Descriptor = ()
    red_glove_score: float = 0.0
    white_glove_score: float = 0.0
    white_uniform_score: float = 0.0
    black_belt_score: float = 0.0
    blue_glove_score: float = 0.0
    standing_score: float = 1.0
    reference_match_score: float = 0.0
    pose_reference_match_score: float = 0.0
    face_match_score: float = 0.0
    face_detected: bool = False

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def pose_descriptor(box: TrackedBox, keypoints: Keypoints) -> tuple[float, ...]:
    """Normalize pose coordinates within a detection box for comparison."""
    descriptor = []
    for point in keypoints:
        if len(point) < 2 or (point[0] == 0 and point[1] == 0):
            descriptor.extend((0.0, 0.0))
        else:
            descriptor.extend(
                (
                    (float(point[0]) - box.x1) / box.width,
                    (float(point[1]) - box.y1) / box.height,
                )
            )
    return tuple(descriptor)


class FighterIdentity:
    """Assign a named fighter and recover their label after tracker swaps."""

    def __init__(
        self,
        fighter_a_name: str,
        fighter_a_start: str,
        recovery_threshold: float = 1.35,
        expect_red_gloves: bool = False,
        expect_white_gloves: bool = False,
        expect_white_uniform: bool = False,
        expect_black_belt: bool = False,
        expect_taller: bool = False,
        require_red_gloves: bool = False,
        require_white_gloves: bool = False,
        reject_blue_gloves: bool = False,
        require_standing: bool = False,
        expect_reference_match: bool = False,
        require_reference_match: bool = False,
        expect_pose_reference_match: bool = False,
        expect_face_match: bool = False,
        reject_face_mismatch: bool = False,
        min_red_glove_score: float = 0.35,
        min_white_glove_score: float = 0.02,
        min_standing_score: float = 0.45,
        min_reference_match_score: float = 0.05,
        min_face_match_score: float = 0.35,
        reset_after_missing_frames: int = 15,
        recovery_confirmation_frames: int = 2,
        fighter_candidate_limit: int = 4,
        strong_recovery_threshold: float = 0.35,
        lineup_pause_frames: int = 45,
        lineup_motion_threshold: float = 0.06,
        lineup_separation_threshold: float = 1.50,
    ) -> None:
        if fighter_a_start not in {"left", "right"}:
            raise ValueError("fighter_a_start must be 'left' or 'right'")
        if recovery_confirmation_frames < 1:
            raise ValueError("recovery_confirmation_frames must be at least 1")
        if fighter_candidate_limit < 1:
            raise ValueError("fighter_candidate_limit must be at least 1")
        if lineup_pause_frames < 1:
            raise ValueError("lineup_pause_frames must be at least 1")
        self.fighter_a_name = fighter_a_name
        self.fighter_a_start = fighter_a_start
        self.recovery_threshold = recovery_threshold
        self.expect_red_gloves = expect_red_gloves
        self.expect_white_gloves = expect_white_gloves
        self.expect_white_uniform = expect_white_uniform
        self.expect_black_belt = expect_black_belt
        self.expect_taller = expect_taller
        self.require_red_gloves = require_red_gloves
        self.require_white_gloves = require_white_gloves
        self.reject_blue_gloves = reject_blue_gloves
        self.require_standing = require_standing
        self.expect_reference_match = expect_reference_match
        self.require_reference_match = require_reference_match
        self.expect_pose_reference_match = expect_pose_reference_match
        self.expect_face_match = expect_face_match
        self.reject_face_mismatch = reject_face_mismatch
        self.min_red_glove_score = min_red_glove_score
        self.min_white_glove_score = min_white_glove_score
        self.min_standing_score = min_standing_score
        self.min_reference_match_score = min_reference_match_score
        self.min_face_match_score = min_face_match_score
        self.reset_after_missing_frames = reset_after_missing_frames
        self.recovery_confirmation_frames = recovery_confirmation_frames
        self.fighter_candidate_limit = fighter_candidate_limit
        self.strong_recovery_threshold = strong_recovery_threshold
        self.lineup_pause_frames = lineup_pause_frames
        self.lineup_motion_threshold = lineup_motion_threshold
        self.lineup_separation_threshold = lineup_separation_threshold
        self.fighter_a_track_id: int | None = None
        self.fighter_a_track_ids: set[int] = set()
        self.recovery_count = 0
        self.reset_count = 0
        self._last_observation: TrackedBox | None = None
        self._appearance_template: tuple[float, ...] = ()
        self._missing_frames = 0
        self._visible_track_id: int | None = None
        self._pending_track_id: int | None = None
        self._pending_frames = 0
        self._velocity = (0.0, 0.0)
        self._lineup_stable_frames = 0
        self._lineup_reset_applied = False
        self._previous_foreground: dict[int, TrackedBox] = {}
        self._foreground_max_height = 1.0

    def observe(self, boxes: Iterable[TrackedBox]) -> int | None:
        boxes = list(boxes)
        if not boxes:
            self._missing_frames += 1
            self._visible_track_id = None
            self._clear_pending()
            self._update_lineup_state([])
            return None
        foreground = self._fighter_candidates(boxes)
        self._foreground_max_height = max(box.height for box in foreground)
        if self.fighter_a_track_id is None:
            ordered = sorted(self._eligible_recovery_candidates(foreground), key=lambda box: box.center_x)
            if not ordered:
                return None
            selected = ordered[0] if self.fighter_a_start == "left" else ordered[-1]
            self._accept(selected)
            self._update_lineup_state(foreground)
            return selected.track_id

        lineup_reset = self._update_lineup_state(foreground)
        if lineup_reset is not None:
            if lineup_reset.track_id != self.fighter_a_track_id:
                self.recovery_count += 1
            self.reset_count += 1
            self._accept(lineup_reset)
            return lineup_reset.track_id

        current = next(
            (box for box in foreground if box.track_id == self.fighter_a_track_id),
            None,
        )
        if current is not None and self._is_plausible(current):
            self._accept(current)
            return current.track_id

        self._missing_frames += 1
        self._visible_track_id = None
        if self._last_observation is None:
            return None
        resetting = self._missing_frames >= self.reset_after_missing_frames
        reset_side_x = None
        if resetting:
            centers = [box.center_x for box in foreground]
            reset_side_x = min(centers) if self.fighter_a_start == "left" else max(centers)
        eligible = self._eligible_recovery_candidates(foreground)
        if not eligible:
            self._clear_pending()
            return None
        scored = sorted([
            (self._match_score(box, reset_side_x=reset_side_x), box)
            for box in eligible
        ], key=lambda item: item[0])
        score, selected = scored[0]
        if score > self.recovery_threshold:
            self._clear_pending()
            return None
        if score <= self.strong_recovery_threshold:
            if selected.track_id != self.fighter_a_track_id:
                self.recovery_count += 1
            if resetting:
                self.reset_count += 1
            self._accept(selected)
            return selected.track_id
        if selected.track_id == self._pending_track_id:
            self._pending_frames += 1
        else:
            self._pending_track_id = selected.track_id
            self._pending_frames = 1
        if self._pending_frames < self.recovery_confirmation_frames:
            return None
        if selected.track_id != self.fighter_a_track_id:
            self.recovery_count += 1
        if resetting:
            self.reset_count += 1
        self._accept(selected)
        return selected.track_id

    def label(self, track_id: int) -> str:
        if track_id == self._visible_track_id:
            return self.fighter_a_name
        return f"fighter-{track_id}"

    @property
    def active_track_id(self) -> int | None:
        return self._visible_track_id

    def _is_plausible(self, box: TrackedBox) -> bool:
        if self._last_observation is None:
            return True
        return self._match_score(box) <= self.recovery_threshold

    def _accept(self, box: TrackedBox) -> None:
        if self._last_observation is not None:
            movement = (
                box.center_x - self._last_observation.center_x,
                box.center_y - self._last_observation.center_y,
            )
            self._velocity = (
                self._velocity[0] * 0.65 + movement[0] * 0.35,
                self._velocity[1] * 0.65 + movement[1] * 0.35,
            )
        self.fighter_a_track_id = box.track_id
        self._visible_track_id = box.track_id
        self.fighter_a_track_ids.add(box.track_id)
        self._last_observation = box
        self._missing_frames = 0
        self._clear_pending()
        if box.appearance:
            if not self._appearance_template:
                self._appearance_template = tuple(box.appearance)
            else:
                self._appearance_template = tuple(
                    old * 0.85 + new * 0.15
                    for old, new in zip(self._appearance_template, box.appearance)
                )

    def _match_score(self, box: TrackedBox, *, reset_side_x: float | None = None) -> float:
        if self._last_observation is None:
            return 0.0
        previous = self._last_observation
        position_scale = max(previous.width, previous.height, box.width, box.height)
        position_distance = dist(
            (previous.center_x + self._velocity[0], previous.center_y + self._velocity[1]),
            (box.center_x, box.center_y),
        ) / position_scale
        pose_distance = _mean_absolute_distance(previous.pose, box.pose, default=0.5)
        appearance_distance = _cosine_distance(
            self._appearance_template,
            box.appearance,
            default=0.5,
        )
        score = position_distance * 0.50 + pose_distance * 0.20 + appearance_distance * 0.15
        if self.expect_red_gloves:
            score += (1.0 - box.red_glove_score) * 0.10
        if self.expect_white_gloves:
            score += (1.0 - box.white_glove_score) * 0.10
        if self.expect_white_uniform:
            score += (1.0 - box.white_uniform_score) * 0.05
        if self.expect_black_belt:
            score += (1.0 - box.black_belt_score) * 0.10
        if self.expect_taller:
            score += (1.0 - box.height / self._foreground_max_height) * 0.10
        if self.expect_reference_match:
            score += (1.0 - box.reference_match_score) * 0.20
        if self.expect_pose_reference_match:
            score += (1.0 - box.pose_reference_match_score) * 0.20
        if self.expect_face_match and box.face_detected:
            score += (1.0 - box.face_match_score) * 0.35
        if reset_side_x is not None and box.center_x != reset_side_x:
            score += 0.75
        return score

    def _clear_pending(self) -> None:
        self._pending_track_id = None
        self._pending_frames = 0

    def _fighter_candidates(self, boxes: list[TrackedBox]) -> list[TrackedBox]:
        """Limit identity decisions to the largest foreground people."""
        return sorted(boxes, key=lambda box: box.area, reverse=True)[: self.fighter_candidate_limit]

    def _eligible_recovery_candidates(self, boxes: list[TrackedBox]) -> list[TrackedBox]:
        """Reject people who cannot be Gabriel before assigning a new identity."""
        return [
            box
            for box in boxes
            if (not self.require_red_gloves or box.red_glove_score >= self.min_red_glove_score)
            and (not self.require_white_gloves or box.white_glove_score >= self.min_white_glove_score)
            and (
                not self.reject_blue_gloves
                or box.blue_glove_score < max(box.red_glove_score, box.white_glove_score)
            )
            and (not self.require_standing or box.standing_score >= self.min_standing_score)
            and (
                not self.require_reference_match
                or box.reference_match_score >= self.min_reference_match_score
            )
            and (
                not self.reject_face_mismatch
                or not box.face_detected
                or box.face_match_score >= self.min_face_match_score
            )
        ]

    def _update_lineup_state(self, boxes: list[TrackedBox]) -> TrackedBox | None:
        """Re-anchor to the configured side once fighters pause in a separated lineup."""
        ordered = sorted(boxes, key=lambda box: box.center_x)
        separated = (
            len(ordered) >= 2
            and (ordered[-1].center_x - ordered[0].center_x)
            / max(ordered[-1].height, ordered[0].height)
            >= self.lineup_separation_threshold
        )
        shared = [
            (self._previous_foreground[box.track_id], box)
            for box in boxes
            if box.track_id in self._previous_foreground
        ]
        stable = (
            separated
            and len(shared) >= 2
            and {box.track_id for box in boxes} == set(self._previous_foreground)
            and max(
                dist((previous.center_x, previous.center_y), (current.center_x, current.center_y))
                / max(previous.height, current.height)
                for previous, current in shared
            )
            <= self.lineup_motion_threshold
        )
        self._previous_foreground = {box.track_id: box for box in boxes}
        if not stable:
            self._lineup_stable_frames = 0
            self._lineup_reset_applied = False
            return None
        self._lineup_stable_frames += 1
        if self._lineup_reset_applied or self._lineup_stable_frames < self.lineup_pause_frames:
            return None
        self._lineup_reset_applied = True
        eligible = sorted(self._eligible_recovery_candidates(ordered), key=lambda box: box.center_x)
        if not eligible:
            return None
        selected = eligible[0] if self.fighter_a_start == "left" else eligible[-1]
        return selected if selected.track_id != self.fighter_a_track_id else None


def _mean_absolute_distance(
    first: Descriptor,
    second: Descriptor,
    *,
    default: float,
) -> float:
    if not first or len(first) != len(second):
        return default
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


def _cosine_distance(
    first: Descriptor,
    second: Descriptor,
    *,
    default: float,
) -> float:
    if not first or len(first) != len(second):
        return default
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return default
    similarity = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    return 1.0 - max(0.0, min(1.0, similarity))
