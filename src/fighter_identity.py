"""Recover a named fighter across tracker ID changes."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    exclude_reference_match_score: float = 0.0
    exclude_body_match_score: float = 0.0
    exclude_face_match_score: float = 0.0
    exclude_face_detected: bool = False
    competition_fighter_score: float = 1.0

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


@dataclass(frozen=True)
class IdentityScore:
    track_id: int
    gabriel_candidate_score: float
    red_wrist_score: float
    white_glove_score: float
    blue_glove_score: float
    pose_reference_score: float
    face_match_score: float
    face_detected: bool
    exclude_reference_score: float
    exclude_body_score: float
    exclude_face_score: float
    exclude_face_detected: bool
    competition_fighter_score: float
    continuity_score: float
    reset_side_score: float
    match_gap: float
    rejection_reason: str = ""
    hard_reject_active: bool = False
    confirmed_not_top: bool = False
    confirmed: bool = False
    tentative: bool = False


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
        expect_blue_gloves: bool = False,
        expect_white_uniform: bool = False,
        expect_black_belt: bool = False,
        expect_taller: bool = False,
        require_red_gloves: bool = False,
        require_white_gloves: bool = False,
        require_blue_gloves: bool = False,
        reject_red_gloves: bool = False,
        reject_white_gloves: bool = False,
        reject_blue_gloves: bool = False,
        require_standing: bool = False,
        require_competition_fighter: bool = False,
        expect_reference_match: bool = False,
        require_reference_match: bool = False,
        expect_pose_reference_match: bool = False,
        expect_face_match: bool = False,
        reject_face_mismatch: bool = False,
        reject_exclude_reference_match: bool = False,
        exclude_reference_hard_veto: bool = False,
        exclude_veto_allow_strong_face_match: bool = False,
        min_red_glove_score: float = 0.15,
        min_white_glove_score: float = 0.02,
        min_blue_glove_score: float = 0.15,
        min_standing_score: float = 0.45,
        min_reference_match_score: float = 0.05,
        min_face_match_score: float = 0.25,
        strong_face_match_score: float = 0.70,
        min_exclude_reference_match_score: float = 0.80,
        min_exclude_body_match_score: float | None = None,
        min_exclude_face_match_score: float = 0.45,
        exclude_veto_confirmation_frames: int = 3,
        reset_after_missing_frames: int = 10,
        recovery_confirmation_frames: int = 3,
        fighter_candidate_limit: int = 4,
        strong_recovery_threshold: float = 0.35,
        visual_confidence_threshold: float = 0.45,
        visual_grace_frames: int = 15,
        lineup_pause_frames: int = 30,
        lineup_motion_threshold: float = 0.10,
        lineup_separation_threshold: float = 1.20,
        locked_fighter_exclude_grace_score: float = 0.96,
        locked_fighter_min_continuity_score: float = 0.60,
        locked_fighter_drop_confirmation_frames: int = 10,
        identity_switch_confirmation_frames: int | None = None,
        confirmed_lock_min_frames: int = 30,
    ) -> None:
        if fighter_a_start not in {"left", "right"}:
            raise ValueError("fighter_a_start must be 'left' or 'right'")
        if recovery_confirmation_frames < 1:
            raise ValueError("recovery_confirmation_frames must be at least 1")
        if fighter_candidate_limit < 1:
            raise ValueError("fighter_candidate_limit must be at least 1")
        if lineup_pause_frames < 1:
            raise ValueError("lineup_pause_frames must be at least 1")
        if locked_fighter_drop_confirmation_frames < 1:
            raise ValueError("locked_fighter_drop_confirmation_frames must be at least 1")
        if exclude_veto_confirmation_frames < 1:
            raise ValueError("exclude_veto_confirmation_frames must be at least 1")
        if identity_switch_confirmation_frames is not None and identity_switch_confirmation_frames < 1:
            raise ValueError("identity_switch_confirmation_frames must be at least 1")
        if confirmed_lock_min_frames < 1:
            raise ValueError("confirmed_lock_min_frames must be at least 1")
        self.fighter_a_name = fighter_a_name
        self.fighter_a_start = fighter_a_start
        self.recovery_threshold = recovery_threshold
        self.expect_red_gloves = expect_red_gloves
        self.expect_white_gloves = expect_white_gloves
        self.expect_blue_gloves = expect_blue_gloves
        self.expect_white_uniform = expect_white_uniform
        self.expect_black_belt = expect_black_belt
        self.expect_taller = expect_taller
        self.require_red_gloves = require_red_gloves
        self.require_white_gloves = require_white_gloves
        self.require_blue_gloves = require_blue_gloves
        self.reject_red_gloves = reject_red_gloves
        self.reject_white_gloves = reject_white_gloves
        self.reject_blue_gloves = reject_blue_gloves
        self.require_standing = require_standing
        self.require_competition_fighter = require_competition_fighter
        self.expect_reference_match = expect_reference_match
        self.require_reference_match = require_reference_match
        self.expect_pose_reference_match = expect_pose_reference_match
        self.expect_face_match = expect_face_match
        self.reject_face_mismatch = reject_face_mismatch
        self.reject_exclude_reference_match = reject_exclude_reference_match
        self.exclude_reference_hard_veto = exclude_reference_hard_veto
        self.exclude_veto_allow_strong_face_match = exclude_veto_allow_strong_face_match
        self.min_red_glove_score = min_red_glove_score
        self.min_white_glove_score = min_white_glove_score
        self.min_blue_glove_score = min_blue_glove_score
        self.min_standing_score = min_standing_score
        self.min_reference_match_score = min_reference_match_score
        self.min_face_match_score = min_face_match_score
        self.strong_face_match_score = strong_face_match_score
        self.min_exclude_reference_match_score = min_exclude_reference_match_score
        self.min_exclude_body_match_score = (
            min_exclude_body_match_score
            if min_exclude_body_match_score is not None
            else min_exclude_reference_match_score
        )
        self.min_exclude_face_match_score = min_exclude_face_match_score
        self.exclude_veto_confirmation_frames = exclude_veto_confirmation_frames
        self.reset_after_missing_frames = reset_after_missing_frames
        self.recovery_confirmation_frames = recovery_confirmation_frames
        self.fighter_candidate_limit = fighter_candidate_limit
        self.strong_recovery_threshold = strong_recovery_threshold
        self.visual_confidence_threshold = visual_confidence_threshold
        self.visual_grace_frames = visual_grace_frames
        self.lineup_pause_frames = lineup_pause_frames
        self.lineup_motion_threshold = lineup_motion_threshold
        self.lineup_separation_threshold = lineup_separation_threshold
        self.locked_fighter_exclude_grace_score = locked_fighter_exclude_grace_score
        self.locked_fighter_min_continuity_score = locked_fighter_min_continuity_score
        self.locked_fighter_drop_confirmation_frames = locked_fighter_drop_confirmation_frames
        self.identity_switch_confirmation_frames = (
            identity_switch_confirmation_frames
            if identity_switch_confirmation_frames is not None
            else recovery_confirmation_frames
        )
        self.confirmed_lock_min_frames = confirmed_lock_min_frames
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
        self._last_scores: dict[int, IdentityScore] = {}
        self._visual_track_id: int | None = None
        self._visual_box: TrackedBox | None = None
        self._visual_tentative = False
        self._lock_frames = 0
        self._drop_reason: str | None = None
        self._drop_frames = 0

    def observe(self, boxes: Iterable[TrackedBox]) -> int | None:
        boxes = list(boxes)
        if not boxes:
            self._missing_frames += 1
            self._visible_track_id = None
            self._set_grace_visual()
            self._clear_pending()
            self._update_lineup_state([])
            self._last_scores = {}
            return None
        foreground = self._fighter_candidates(boxes)
        active_fighters = self._active_fighter_candidates(foreground)
        self._foreground_max_height = max(box.height for box in foreground)
        self._last_scores = self._score_candidates(foreground, active_fighters)
        if self.fighter_a_track_id is None:
            ordered = sorted(self._eligible_recovery_candidates(active_fighters), key=lambda box: box.center_x)
            if not ordered:
                self._set_tentative_visual(active_fighters)
                return None
            selected = ordered[0] if self.fighter_a_start == "left" else ordered[-1]
            self._accept(selected)
            self._update_lineup_state(active_fighters)
            self._mark_visual(selected, tentative=False)
            return selected.track_id

        lineup_reset = self._update_lineup_state(active_fighters)
        if lineup_reset is not None:
            if lineup_reset.track_id != self.fighter_a_track_id:
                self.recovery_count += 1
            self.reset_count += 1
            self._accept(lineup_reset)
            self._mark_visual(lineup_reset, tentative=False)
            return lineup_reset.track_id

        current = next(
            (box for box in foreground if box.track_id == self.fighter_a_track_id),
            None,
        )
        if current is not None:
            current_reason = self._rejection_reason(current, {box.track_id for box in active_fighters})
            if self._should_keep_current_lock(current, current_reason):
                keep_drop_pending = current_reason in {"exclude_reference", "face_mismatch"}
                self._accept(current, clear_drop_pending=not keep_drop_pending)
                self._mark_visual(current, tentative=False)
                return current.track_id

        self._missing_frames += 1
        self._visible_track_id = None
        if self._last_observation is None:
            self._set_tentative_visual(active_fighters)
            return None
        resetting = self._missing_frames >= self.reset_after_missing_frames
        reset_side_x = None
        if resetting:
            centers = [box.center_x for box in active_fighters]
            reset_side_x = min(centers) if self.fighter_a_start == "left" else max(centers)
        eligible = self._eligible_recovery_candidates(active_fighters, allow_soft_red=not resetting)
        if not eligible:
            self._clear_pending()
            if self._has_exclude_reference_reject(active_fighters):
                self._clear_visual()
                return None
            if not self._set_tentative_visual(active_fighters):
                self._set_grace_visual()
            return None
        scored = sorted([
            (self._match_score(box, reset_side_x=reset_side_x), box)
            for box in eligible
        ], key=lambda item: item[0])
        score, selected = scored[0]
        if score > self.recovery_threshold:
            self._clear_pending()
            if not self._set_tentative_visual(active_fighters):
                self._set_grace_visual()
            return None
        if score <= self.strong_recovery_threshold:
            if selected.track_id != self.fighter_a_track_id:
                self.recovery_count += 1
            if resetting:
                self.reset_count += 1
            self._accept(selected)
            self._mark_visual(selected, tentative=False)
            return selected.track_id
        if selected.track_id == self._pending_track_id:
            self._pending_frames += 1
        else:
            self._pending_track_id = selected.track_id
            self._pending_frames = 1
        required_confirmation_frames = (
            self.identity_switch_confirmation_frames
            if selected.track_id != self.fighter_a_track_id
            else self.recovery_confirmation_frames
        )
        if self._pending_frames < required_confirmation_frames:
            self._mark_visual(selected, tentative=True)
            return None
        if selected.track_id != self.fighter_a_track_id:
            self.recovery_count += 1
        if resetting:
            self.reset_count += 1
        self._accept(selected)
        self._mark_visual(selected, tentative=False)
        return selected.track_id

    def label(self, track_id: int) -> str:
        if track_id == self._visible_track_id:
            return self.fighter_a_name
        return f"fighter-{track_id}"

    @property
    def active_track_id(self) -> int | None:
        return self._visible_track_id

    @property
    def visual_track_id(self) -> int | None:
        return self._visual_track_id

    @property
    def visual_box(self) -> TrackedBox | None:
        return self._visual_box

    @property
    def visual_tentative(self) -> bool:
        return self._visual_tentative

    @property
    def identity_scores(self) -> dict[int, IdentityScore]:
        return self._last_scores

    def _is_plausible(self, box: TrackedBox) -> bool:
        if self._matches_exclude_reference(box):
            return False
        if self._last_observation is None:
            return True
        return self._match_score(box) <= self.recovery_threshold

    def _should_keep_current_lock(self, box: TrackedBox, reason: str) -> bool:
        if reason in {
            "",
            "red_below_threshold_continuity_ok",
            "white_below_threshold_continuity_ok",
            "blue_below_threshold_continuity_ok",
            "exclude_reference_locked_ok",
        }:
            self._clear_drop_pending()
            return self._is_positionally_plausible(box)
        if reason in {"red_below_threshold", "white_below_threshold", "blue_below_threshold"}:
            self._clear_drop_pending()
            return self._is_positionally_plausible(box)
        if reason in {"blue_glove", "not_standing", "not_competition_fighter", "reference_below_threshold"}:
            self._clear_drop_pending()
            return False
        if reason not in {"exclude_reference", "face_mismatch"}:
            self._clear_drop_pending()
            return False
        if self._lock_frames < self.confirmed_lock_min_frames:
            self._mark_drop_pending(reason)
            return True
        self._mark_drop_pending(reason)
        confirmation_frames = (
            self.exclude_veto_confirmation_frames
            if reason == "exclude_reference" and self.exclude_reference_hard_veto
            else self.locked_fighter_drop_confirmation_frames
        )
        return self._drop_frames < confirmation_frames

    def _is_positionally_plausible(self, box: TrackedBox) -> bool:
        if self._last_observation is None:
            return True
        return self._match_score(box) <= self.recovery_threshold

    def _accept(self, box: TrackedBox, *, clear_drop_pending: bool = True) -> None:
        previous_track_id = self.fighter_a_track_id
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
        self._lock_frames = self._lock_frames + 1 if previous_track_id == box.track_id else 1
        if clear_drop_pending:
            self._clear_drop_pending()
        self._clear_pending()
        self._mark_visual(box, tentative=False)
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
        if self.expect_blue_gloves:
            score += (1.0 - box.blue_glove_score) * 0.10
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

    def _clear_drop_pending(self) -> None:
        self._drop_reason = None
        self._drop_frames = 0

    def _mark_drop_pending(self, reason: str) -> None:
        if self._drop_reason == reason:
            self._drop_frames += 1
        else:
            self._drop_reason = reason
            self._drop_frames = 1

    def _clear_visual(self) -> None:
        self._visual_track_id = None
        self._visual_box = None
        self._visual_tentative = False

    def _fighter_candidates(self, boxes: list[TrackedBox]) -> list[TrackedBox]:
        """Limit identity decisions to the largest foreground people."""
        return sorted(boxes, key=lambda box: box.area, reverse=True)[: self.fighter_candidate_limit]

    def _active_fighter_candidates(self, boxes: list[TrackedBox]) -> list[TrackedBox]:
        """Focus identity decisions on the two most likely standing fighters."""
        standing = [
            box
            for box in boxes
            if not self.require_standing or box.standing_score >= self.min_standing_score
        ]
        competition = [
            box
            for box in standing
            if not self.require_competition_fighter or box.competition_fighter_score >= 1.0
        ]
        ordered = sorted(competition or standing or boxes, key=lambda box: box.area, reverse=True)[:2]
        current = next((box for box in boxes if box.track_id == self.fighter_a_track_id), None)
        if current is not None and current not in ordered:
            ordered = [current] + [box for box in ordered if box.track_id != current.track_id]
            ordered = ordered[:2]
        return ordered

    def _eligible_recovery_candidates(
        self,
        boxes: list[TrackedBox],
        *,
        allow_soft_red: bool = False,
    ) -> list[TrackedBox]:
        """Reject people who cannot be Gabriel before assigning a new identity."""
        return [
            box
            for box in boxes
            if (
                not self.require_red_gloves
                or box.red_glove_score >= self.min_red_glove_score
                or (allow_soft_red and self._has_strong_continuity(box))
            )
            and (
                not self.require_white_gloves
                or box.white_glove_score >= self.min_white_glove_score
                or (allow_soft_red and self._has_strong_continuity(box))
            )
            and (
                not self.require_blue_gloves
                or box.blue_glove_score >= self.min_blue_glove_score
                or (allow_soft_red and self._has_strong_continuity(box))
            )
            and (
                not self.reject_red_gloves
                or box.red_glove_score < max(box.white_glove_score, box.blue_glove_score)
            )
            and (
                not self.reject_white_gloves
                or box.white_glove_score < max(box.red_glove_score, box.blue_glove_score)
            )
            and (
                not self.reject_blue_gloves
                or box.blue_glove_score < max(box.red_glove_score, box.white_glove_score)
            )
            and (not self.require_standing or box.standing_score >= self.min_standing_score)
            and (not self.require_competition_fighter or box.competition_fighter_score >= 1.0)
            and (
                not self.require_reference_match
                or box.reference_match_score >= self.min_reference_match_score
            )
            and (
                not self.reject_face_mismatch
                or not box.face_detected
                or box.face_match_score >= self.min_face_match_score
            )
            and not self._matches_exclude_reference(box)
        ]

    def _has_strong_continuity(self, box: TrackedBox) -> bool:
        if self._last_observation is None:
            return False
        return self._match_score(box) <= self.strong_recovery_threshold

    def _mark_visual(self, box: TrackedBox, *, tentative: bool) -> None:
        self._visual_track_id = box.track_id
        self._visual_box = box
        self._visual_tentative = tentative
        if box.track_id in self._last_scores:
            score = self._last_scores[box.track_id]
            self._last_scores[box.track_id] = replace(
                score,
                confirmed=not tentative,
                tentative=tentative,
                confirmed_not_top=not tentative and score.match_gap < 0,
            )

    def _set_grace_visual(self) -> bool:
        if self._last_observation is None or self._missing_frames > self.visual_grace_frames:
            self._visual_track_id = None
            self._visual_box = None
            self._visual_tentative = False
            return False
        self._visual_track_id = self._last_observation.track_id
        self._visual_box = self._last_observation
        self._visual_tentative = True
        return True

    def _set_tentative_visual(self, boxes: list[TrackedBox]) -> bool:
        scored = sorted(
            (
                (self._last_scores.get(box.track_id), box)
                for box in boxes
                if box.track_id in self._last_scores
                and not self._last_scores[box.track_id].hard_reject_active
            ),
            key=lambda item: item[0].gabriel_candidate_score if item[0] is not None else 0.0,
            reverse=True,
        )
        if not scored or scored[0][0] is None:
            self._visual_track_id = None
            self._visual_box = None
            self._visual_tentative = False
            return False
        score, box = scored[0]
        if score.gabriel_candidate_score < self.visual_confidence_threshold:
            return False
        self._mark_visual(box, tentative=True)
        return True

    def _score_candidates(
        self,
        boxes: list[TrackedBox],
        active_fighters: list[TrackedBox],
    ) -> dict[int, IdentityScore]:
        active_ids = {box.track_id for box in active_fighters}
        reset_side_x = None
        if active_fighters:
            centers = [box.center_x for box in active_fighters]
            reset_side_x = min(centers) if self.fighter_a_start == "left" else max(centers)
        scored_entries = []
        for box in boxes:
            continuity = self._continuity_score(box)
            reset_side_score = 1.0 if reset_side_x is not None and box.center_x == reset_side_x else 0.0
            cue_values = []
            glove_scores = self._expected_glove_scores(box)
            if glove_scores:
                cue_values.append((max(glove_scores), 0.30))
            cue_values.append((box.standing_score, 0.10))
            if self.expect_reference_match:
                cue_values.append((box.reference_match_score, 0.15))
            if self.expect_pose_reference_match:
                cue_values.append((box.pose_reference_match_score, 0.20))
            if self.expect_face_match and box.face_detected:
                cue_values.append((box.face_match_score, 0.25))
            if self._last_observation is not None:
                cue_values.append((continuity, 0.25))
            else:
                cue_values.append((reset_side_score, 0.15))
            total_weight = sum(weight for _, weight in cue_values)
            candidate_score = sum(value * weight for value, weight in cue_values) / total_weight
            reason = self._rejection_reason(box, active_ids)
            scored_entries.append((box, max(0.0, min(1.0, candidate_score)), continuity, reset_side_score, reason))
        scores = {}
        for box, candidate_score, continuity, reset_side_score, reason in scored_entries:
            other_scores = [
                score
                for other_box, score, *_ in scored_entries
                if other_box.track_id != box.track_id
            ]
            next_best_score = max(other_scores) if other_scores else 0.0
            scores[box.track_id] = IdentityScore(
                track_id=box.track_id,
                gabriel_candidate_score=candidate_score,
                red_wrist_score=box.red_glove_score,
                white_glove_score=box.white_glove_score,
                blue_glove_score=box.blue_glove_score,
                pose_reference_score=box.pose_reference_match_score,
                face_match_score=box.face_match_score,
                face_detected=box.face_detected,
                exclude_reference_score=box.exclude_reference_match_score,
                exclude_body_score=self._exclude_body_score(box),
                exclude_face_score=box.exclude_face_match_score,
                exclude_face_detected=box.exclude_face_detected,
                competition_fighter_score=box.competition_fighter_score,
                continuity_score=continuity,
                reset_side_score=reset_side_score,
                match_gap=candidate_score - next_best_score,
                rejection_reason=reason,
                hard_reject_active=self._is_hard_reject(reason),
            )
        return scores

    def _continuity_score(self, box: TrackedBox) -> float:
        if self._last_observation is None:
            return 0.0
        score = self._match_score(box)
        return max(0.0, min(1.0, 1.0 - score / self.recovery_threshold))

    def _rejection_reason(self, box: TrackedBox, active_ids: set[int]) -> str:
        if box.track_id not in active_ids:
            return "not_active_fighter_pair"
        if self.require_competition_fighter and box.competition_fighter_score < 1.0:
            return "not_competition_fighter"
        if self.reject_red_gloves and box.red_glove_score >= max(box.white_glove_score, box.blue_glove_score):
            return "red_glove"
        if self.reject_white_gloves and box.white_glove_score >= max(box.red_glove_score, box.blue_glove_score):
            return "white_glove"
        if self.reject_blue_gloves and box.blue_glove_score >= max(box.red_glove_score, box.white_glove_score):
            return "blue_glove"
        if self.require_standing and box.standing_score < self.min_standing_score:
            return "not_standing"
        if self.reject_face_mismatch and box.face_detected and box.face_match_score < self.min_face_match_score:
            return "face_mismatch"
        if self._matches_exclude_reference(box):
            if box.track_id == self.fighter_a_track_id and self._locked_exclude_reference_is_allowed(box):
                return "exclude_reference_locked_ok"
            return "exclude_reference"
        if self.require_reference_match and box.reference_match_score < self.min_reference_match_score:
            return "reference_below_threshold"
        if self.require_red_gloves and box.red_glove_score < self.min_red_glove_score:
            if self._has_strong_continuity(box):
                return "red_below_threshold_continuity_ok"
            return "red_below_threshold"
        if self.require_white_gloves and box.white_glove_score < self.min_white_glove_score:
            if self._has_strong_continuity(box):
                return "white_below_threshold_continuity_ok"
            return "white_below_threshold"
        if self.require_blue_gloves and box.blue_glove_score < self.min_blue_glove_score:
            if self._has_strong_continuity(box):
                return "blue_below_threshold_continuity_ok"
            return "blue_below_threshold"
        return ""

    def _is_hard_reject(self, reason: str) -> bool:
        return reason in {
            "exclude_reference",
            "red_glove",
            "white_glove",
            "blue_glove",
            "face_mismatch",
            "not_standing",
            "not_competition_fighter",
            "reference_below_threshold",
        }

    def _matches_exclude_reference(self, box: TrackedBox) -> bool:
        if not self.reject_exclude_reference_match:
            return False
        if (
            box.exclude_face_detected
            and box.exclude_face_match_score >= self.min_exclude_face_match_score
        ):
            return True
        return self._body_exclude_is_hard(box)

    def _body_exclude_is_hard(self, box: TrackedBox) -> bool:
        if self._exclude_body_score(box) < self.min_exclude_body_match_score:
            return False
        if self.exclude_reference_hard_veto:
            return not self._has_strong_gabriel_face_match(box)
        return not self._has_meaningful_gabriel_evidence(box)

    def _exclude_body_score(self, box: TrackedBox) -> float:
        if box.exclude_body_match_score > 0.0:
            return box.exclude_body_match_score
        return box.exclude_reference_match_score

    def _has_meaningful_gabriel_evidence(self, box: TrackedBox) -> bool:
        expected_gloves = self._expected_glove_thresholds(box)
        if expected_gloves and any(score >= threshold for score, threshold in expected_gloves):
            return True
        if box.face_detected and box.face_match_score >= self.min_face_match_score:
            return True
        if box.pose_reference_match_score >= 0.70:
            return True
        return self._has_strong_continuity(box)

    def _has_strong_gabriel_face_match(self, box: TrackedBox) -> bool:
        return (
            self.exclude_veto_allow_strong_face_match
            and box.face_detected
            and box.face_match_score >= self.strong_face_match_score
        )

    def _expected_glove_scores(self, box: TrackedBox) -> list[float]:
        scores = []
        if self.expect_red_gloves or self.require_red_gloves:
            scores.append(box.red_glove_score)
        if self.expect_white_gloves or self.require_white_gloves:
            scores.append(box.white_glove_score)
        if self.expect_blue_gloves or self.require_blue_gloves:
            scores.append(box.blue_glove_score)
        return scores

    def _expected_glove_thresholds(self, box: TrackedBox) -> list[tuple[float, float]]:
        scores = []
        if self.expect_red_gloves or self.require_red_gloves:
            scores.append((box.red_glove_score, self.min_red_glove_score))
        if self.expect_white_gloves or self.require_white_gloves:
            scores.append((box.white_glove_score, self.min_white_glove_score))
        if self.expect_blue_gloves or self.require_blue_gloves:
            scores.append((box.blue_glove_score, self.min_blue_glove_score))
        return scores

    def _locked_exclude_reference_is_allowed(self, box: TrackedBox) -> bool:
        if self.exclude_reference_hard_veto:
            return False
        if box.track_id != self.fighter_a_track_id:
            return False
        if (
            box.exclude_face_detected
            and box.exclude_face_match_score >= self.min_exclude_face_match_score
        ):
            return False
        if self._lock_frames < self.confirmed_lock_min_frames:
            return True
        if box.exclude_reference_match_score < self.locked_fighter_exclude_grace_score:
            return True
        return self._continuity_score(box) >= self.locked_fighter_min_continuity_score

    def _has_exclude_reference_reject(self, boxes: list[TrackedBox]) -> bool:
        return any(
            self._last_scores.get(box.track_id) is not None
            and self._last_scores[box.track_id].rejection_reason == "exclude_reference"
            for box in boxes
        )

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
