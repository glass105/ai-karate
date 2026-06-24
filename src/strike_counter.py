"""Pose heuristics and debug metrics for prototype strike counting."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import dist
from typing import Sequence

Point = Sequence[float]
Keypoints = Sequence[Point]


@dataclass
class StrikeDebug:
    """Per-frame strike decision metrics for CSV/debug review."""

    punch_score: float = 0.0
    kick_score: float = 0.0
    punch_endpoint_motion: float = 0.0
    kick_endpoint_motion: float = 0.0
    punch_extension_delta: float = 0.0
    kick_extension_delta: float = 0.0
    punch_extension_ratio: float = 0.0
    kick_extension_ratio: float = 0.0
    kick_foot_height_change: float = 0.0
    kick_foot_elevation_body_heights: float = 0.0
    kick_support_foot_motion_body_heights: float = 0.0
    kick_opponent_distance_body_heights: float | None = None
    kick_used_recent_opponent: bool = False
    punch_commitment_frames: int = 0
    kick_commitment_frames: int = 0
    punch_cooldown: int = 0
    kick_cooldown: int = 0
    punch_armed: bool = True
    kick_armed: bool = True
    strike_candidate_type: str = ""
    strike_confirmed: bool = False
    strike_rejection_reason: str = ""

    def as_csv_fields(self) -> dict[str, object]:
        return {
            "strike_punch_score": round(self.punch_score, 4),
            "strike_kick_score": round(self.kick_score, 4),
            "strike_punch_endpoint_motion": round(self.punch_endpoint_motion, 4),
            "strike_kick_endpoint_motion": round(self.kick_endpoint_motion, 4),
            "strike_punch_extension_delta": round(self.punch_extension_delta, 4),
            "strike_kick_extension_delta": round(self.kick_extension_delta, 4),
            "strike_punch_extension_ratio": round(self.punch_extension_ratio, 4),
            "strike_kick_extension_ratio": round(self.kick_extension_ratio, 4),
            "strike_kick_foot_height_change": round(self.kick_foot_height_change, 4),
            "strike_kick_foot_elevation_body_heights": round(
                self.kick_foot_elevation_body_heights, 4
            ),
            "strike_kick_support_foot_motion_body_heights": round(
                self.kick_support_foot_motion_body_heights, 4
            ),
            "strike_kick_opponent_distance_body_heights": (
                round(self.kick_opponent_distance_body_heights, 4)
                if self.kick_opponent_distance_body_heights is not None
                else ""
            ),
            "strike_kick_used_recent_opponent": self.kick_used_recent_opponent,
            "strike_punch_commitment_frames": self.punch_commitment_frames,
            "strike_kick_commitment_frames": self.kick_commitment_frames,
            "strike_punch_cooldown": self.punch_cooldown,
            "strike_kick_cooldown": self.kick_cooldown,
            "strike_punch_armed": self.punch_armed,
            "strike_kick_armed": self.kick_armed,
            "strike_candidate_type": self.strike_candidate_type,
            "strike_confirmed": self.strike_confirmed,
            "strike_rejection_reason": self.strike_rejection_reason,
        }


@dataclass
class LimbMotion:
    endpoint_motion: float = 0.0
    extension_delta: float = 0.0
    extension_ratio: float = 0.0
    commitment_frames: int = 0
    score: float = 0.0
    details: dict[str, float] = field(default_factory=dict)


class StrikeCounter:
    """Estimate and explain strike candidates from short pose history."""

    def __init__(
        self,
        fps: float,
        history_frames: int = 6,
        punch_cooldown_seconds: float = 0.35,
        kick_cooldown_seconds: float = 0.50,
        min_punch_endpoint_motion: float = 25.0,
        min_kick_endpoint_motion: float = 25.0,
        min_kick_foot_motion: float = 0.0,
        min_punch_extension_delta: float = 12.0,
        min_kick_extension_delta: float = 12.0,
        min_punch_extension_ratio: float = 1.20,
        min_kick_extension_ratio: float = 1.20,
        min_punch_commitment_frames: int = 3,
        min_punch_commitment_ratio: float = 0.0,
        min_kick_commitment_frames: int = 2,
        min_kick_foot_height_change: float = 20.0,
        min_kick_score: float = 0.0,
        strong_kick_score: float = 0.0,
        max_kick_extension_ratio: float = 0.0,
        min_kick_foot_elevation_body_heights: float = 0.0,
        strong_kick_foot_elevation_body_heights: float = 0.0,
        max_kick_support_foot_motion_body_heights: float = 0.0,
        max_kick_opponent_distance_body_heights: float = 0.0,
        kick_opponent_memory_frames: int = 0,
        min_strike_score: float = 1.0,
        strike_rearm_score: float = 0.60,
        min_strike_score_gap: float = 0.10,
    ) -> None:
        self.history_frames = history_frames
        self.punch_cooldown_frames = max(1, int(fps * punch_cooldown_seconds))
        self.kick_cooldown_frames = max(1, int(fps * kick_cooldown_seconds))
        self.min_punch_endpoint_motion = min_punch_endpoint_motion
        self.min_kick_endpoint_motion = min_kick_endpoint_motion
        self.min_kick_foot_motion = min_kick_foot_motion
        self.min_punch_extension_delta = min_punch_extension_delta
        self.min_kick_extension_delta = min_kick_extension_delta
        self.min_punch_extension_ratio = min_punch_extension_ratio
        self.min_kick_extension_ratio = min_kick_extension_ratio
        self.min_punch_commitment_frames = max(1, min_punch_commitment_frames)
        self.min_punch_commitment_ratio = (
            min_punch_commitment_ratio if min_punch_commitment_ratio > 0 else min_punch_extension_ratio
        )
        self.min_kick_commitment_frames = max(1, min_kick_commitment_frames)
        self.min_kick_foot_height_change = min_kick_foot_height_change
        self.min_kick_score = min_kick_score if min_kick_score > 0 else min_strike_score
        self.strong_kick_score = max(0.0, strong_kick_score)
        self.max_kick_extension_ratio = max(0.0, max_kick_extension_ratio)
        self.min_kick_foot_elevation_body_heights = max(
            0.0, min_kick_foot_elevation_body_heights
        )
        self.strong_kick_foot_elevation_body_heights = max(
            0.0, strong_kick_foot_elevation_body_heights
        )
        self.max_kick_support_foot_motion_body_heights = max(
            0.0, max_kick_support_foot_motion_body_heights
        )
        self.max_kick_opponent_distance_body_heights = max(0.0, max_kick_opponent_distance_body_heights)
        self.kick_opponent_memory_frames = max(0, kick_opponent_memory_frames)
        self.min_strike_score = min_strike_score
        self.strike_rearm_score = strike_rearm_score
        self.min_strike_score_gap = min_strike_score_gap
        self.history: dict[int, deque[Keypoints]] = defaultdict(
            lambda: deque(maxlen=self.history_frames)
        )
        self.cooldowns: dict[int, dict[str, int]] = defaultdict(
            lambda: {"punch": 0, "fake_punch": 0, "kick": 0}
        )
        self.counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"punches": 0, "fake_punches": 0, "kicks": 0}
        )
        self.armed: dict[int, dict[str, bool]] = defaultdict(
            lambda: {"punch": True, "kick": True}
        )
        self.last_debug: dict[int, StrikeDebug] = defaultdict(StrikeDebug)
        self.last_frame_index: dict[int, int] = {}
        self.pending_punch_commitment: dict[int, int] = {}
        self.last_opponent_observation: dict[int, tuple[int, float]] = {}

    def update(
        self,
        track_id: int,
        keypoints: Keypoints,
        count_enabled: bool = True,
        opponent_distance_body_heights: float | None = None,
        frame_index: int | None = None,
    ) -> str:
        if track_id < 0 or len(keypoints) < 17:
            self.last_debug[track_id] = StrikeDebug(strike_rejection_reason="invalid_pose")
            return ""

        if frame_index is not None:
            previous_frame = self.last_frame_index.get(track_id)
            if previous_frame is not None and frame_index != previous_frame + 1:
                self.history[track_id].clear()
                self.pending_punch_commitment.pop(track_id, None)
            self.last_frame_index[track_id] = frame_index
        effective_opponent_distance = opponent_distance_body_heights
        used_recent_opponent = False
        if frame_index is not None and opponent_distance_body_heights is not None:
            self.last_opponent_observation[track_id] = (
                frame_index,
                opponent_distance_body_heights,
            )
        elif frame_index is not None and self.kick_opponent_memory_frames > 0:
            previous_opponent = self.last_opponent_observation.get(track_id)
            if (
                previous_opponent is not None
                and frame_index - previous_opponent[0] <= self.kick_opponent_memory_frames
            ):
                effective_opponent_distance = previous_opponent[1]
                used_recent_opponent = True

        self.history[track_id].append(keypoints)
        self._tick_cooldowns(track_id)
        if len(self.history[track_id]) < self.history_frames:
            self.last_debug[track_id] = StrikeDebug(
                punch_cooldown=self.cooldowns[track_id]["punch"],
                kick_cooldown=self.cooldowns[track_id]["kick"],
                kick_opponent_distance_body_heights=effective_opponent_distance,
                kick_used_recent_opponent=used_recent_opponent,
                strike_rejection_reason="history_warmup",
            )
            return ""

        punch = self._punch_motion(track_id)
        kick = self._kick_motion(track_id)
        self._rearm_if_retracted(track_id, "punch", punch.score)
        self._rearm_if_retracted(track_id, "kick", kick.score)
        debug = StrikeDebug(
            punch_score=punch.score,
            kick_score=kick.score,
            punch_endpoint_motion=punch.endpoint_motion,
            kick_endpoint_motion=kick.endpoint_motion,
            punch_extension_delta=punch.extension_delta,
            kick_extension_delta=kick.extension_delta,
            punch_extension_ratio=punch.extension_ratio,
            kick_extension_ratio=kick.extension_ratio,
            kick_foot_height_change=kick.details.get("height_change", 0.0),
            kick_foot_elevation_body_heights=kick.details.get(
                "foot_elevation_body_heights", 0.0
            ),
            kick_support_foot_motion_body_heights=kick.details.get(
                "support_foot_motion_body_heights", 0.0
            ),
            kick_opponent_distance_body_heights=effective_opponent_distance,
            kick_used_recent_opponent=used_recent_opponent,
            punch_commitment_frames=punch.commitment_frames,
            kick_commitment_frames=kick.commitment_frames,
            punch_cooldown=self.cooldowns[track_id]["punch"],
            kick_cooldown=self.cooldowns[track_id]["kick"],
            punch_armed=self.armed[track_id]["punch"],
            kick_armed=self.armed[track_id]["kick"],
        )
        action = "punch" if punch.score >= kick.score else "kick"
        action_score = max(punch.score, kick.score)
        other_score = kick.score if action == "punch" else punch.score
        action_qualified = (
            punch.score >= self.min_strike_score
            if action == "punch"
            else self._kick_score_qualified(kick)
        )
        debug.strike_candidate_type = action if action_qualified else ""

        if not count_enabled and debug.strike_candidate_type:
            debug.strike_rejection_reason = "not_selected_fighter"
            self.last_debug[track_id] = debug
            return ""

        if action_qualified and action_score - other_score < self.min_strike_score_gap:
            debug.strike_rejection_reason = "ambiguous_strike"
            self.last_debug[track_id] = debug
            return ""

        if action == "punch" and punch.score >= self.min_strike_score:
            if self.cooldowns[track_id]["punch"] > 0:
                debug.strike_rejection_reason = "punch_cooldown"
            elif not self.armed[track_id]["punch"]:
                debug.strike_rejection_reason = "punch_not_rearmed"
            elif punch.commitment_frames < self.min_punch_commitment_frames:
                if self.cooldowns[track_id]["fake_punch"] > 0:
                    debug.strike_rejection_reason = "fake_punch_cooldown"
                else:
                    self.pending_punch_commitment[track_id] = max(
                        punch.commitment_frames,
                        self.pending_punch_commitment.get(track_id, 0),
                    )
                    debug.strike_rejection_reason = "punch_pending_commitment"
            else:
                self.pending_punch_commitment.pop(track_id, None)
                self.counts[track_id]["punches"] += 1
                self.cooldowns[track_id]["punch"] = self.punch_cooldown_frames
                self.armed[track_id]["punch"] = False
                debug.punch_cooldown = self.cooldowns[track_id]["punch"]
                debug.punch_armed = False
                debug.strike_candidate_type = "punch"
                debug.strike_confirmed = True
                self.last_debug[track_id] = debug
                return "punch"
            self.last_debug[track_id] = debug
            return ""

        if action == "kick" and self._kick_score_qualified(kick):
            strong_kinematics = self._strong_kick_kinematics(kick)
            if (
                self.max_kick_extension_ratio > 0
                and kick.extension_ratio > self.max_kick_extension_ratio
            ):
                debug.strike_rejection_reason = "kick_pose_ratio_implausible"
            elif (
                self.min_kick_foot_elevation_body_heights > 0
                and debug.kick_foot_elevation_body_heights
                < self.min_kick_foot_elevation_body_heights
            ):
                debug.strike_rejection_reason = "kick_insufficient_foot_elevation"
            elif (
                self.max_kick_support_foot_motion_body_heights > 0
                and debug.kick_support_foot_motion_body_heights
                > self.max_kick_support_foot_motion_body_heights
                and not strong_kinematics
            ):
                debug.strike_rejection_reason = "kick_support_foot_moving"
            elif kick.commitment_frames < self.min_kick_commitment_frames:
                debug.strike_rejection_reason = "kick_not_committed"
            elif (
                self.max_kick_opponent_distance_body_heights > 0
                and effective_opponent_distance is None
                and not strong_kinematics
            ):
                debug.strike_rejection_reason = "kick_no_opponent"
            elif (
                self.max_kick_opponent_distance_body_heights > 0
                and effective_opponent_distance is not None
                and effective_opponent_distance > self.max_kick_opponent_distance_body_heights
                and not strong_kinematics
            ):
                debug.strike_rejection_reason = "kick_opponent_too_far"
            elif self.cooldowns[track_id]["kick"] > 0:
                debug.strike_rejection_reason = "kick_cooldown"
            elif not self.armed[track_id]["kick"]:
                debug.strike_rejection_reason = "kick_not_rearmed"
            else:
                self.counts[track_id]["kicks"] += 1
                self.cooldowns[track_id]["kick"] = self.kick_cooldown_frames
                self.armed[track_id]["kick"] = False
                debug.kick_cooldown = self.cooldowns[track_id]["kick"]
                debug.kick_armed = False
                debug.strike_candidate_type = "kick"
                debug.strike_confirmed = True
                self.last_debug[track_id] = debug
                return "kick"
            self.last_debug[track_id] = debug
            return ""

        if (
            count_enabled
            and track_id in self.pending_punch_commitment
            and punch.score < self.strike_rearm_score
        ):
            pending_commitment = self.pending_punch_commitment.pop(track_id)
            if self.cooldowns[track_id]["fake_punch"] <= 0:
                self.counts[track_id]["fake_punches"] += 1
                self.cooldowns[track_id]["fake_punch"] = self.punch_cooldown_frames
                debug.punch_commitment_frames = max(
                    debug.punch_commitment_frames,
                    pending_commitment,
                )
                debug.strike_candidate_type = "fake_punch"
                debug.strike_confirmed = True
                debug.strike_rejection_reason = "fake_punch"
                self.last_debug[track_id] = debug
                return "fake_punch"

        if debug.strike_candidate_type:
            debug.strike_rejection_reason = f"{debug.strike_candidate_type}_cooldown"
        else:
            debug.strike_rejection_reason = "below_threshold"
        self.last_debug[track_id] = debug
        return ""

    def _rearm_if_retracted(self, track_id: int, strike_type: str, score: float) -> None:
        if score < self.strike_rearm_score:
            self.armed[track_id][strike_type] = True

    def _tick_cooldowns(self, track_id: int) -> None:
        for strike_type in ("punch", "fake_punch", "kick"):
            current = self.cooldowns[track_id][strike_type]
            self.cooldowns[track_id][strike_type] = max(0, current - 1)

    def _punch_motion(self, track_id: int) -> LimbMotion:
        old, new = self.history[track_id][0], self.history[track_id][-1]
        candidates: list[LimbMotion] = []
        for shoulder, wrist in ((5, 9), (6, 10)):
            motion = self._limb_motion(
                old[shoulder],
                old[wrist],
                new[shoulder],
                new[wrist],
                self.min_punch_endpoint_motion,
                self.min_punch_extension_delta,
                self.min_punch_extension_ratio,
            )
            motion.commitment_frames = self._punch_commitment_frames(track_id, shoulder, wrist)
            candidates.append(motion)
        return max(candidates, key=lambda motion: motion.score)

    def _punch_commitment_frames(self, track_id: int, shoulder: int, wrist: int) -> int:
        first = self.history[track_id][0]
        old_extension = dist(self._xy(first[shoulder]), self._xy(first[wrist]))
        if old_extension <= 1e-6:
            return 0

        committed = 0
        for keypoints in reversed(self.history[track_id]):
            extension = dist(self._xy(keypoints[shoulder]), self._xy(keypoints[wrist]))
            extension_delta = extension - old_extension
            extension_ratio = extension / old_extension
            if (
                extension_delta >= self.min_punch_extension_delta
                and extension_ratio >= self.min_punch_commitment_ratio
            ):
                committed += 1
            else:
                break
        return committed

    def _kick_motion(self, track_id: int) -> LimbMotion:
        old, new = self.history[track_id][0], self.history[track_id][-1]
        candidates: list[LimbMotion] = []
        for hip, knee, ankle, support_ankle in ((11, 13, 15, 16), (12, 14, 16, 15)):
            context = self._kick_context(track_id, hip, ankle, support_ankle)
            hip_to_ankle = self._limb_motion(
                old[hip],
                old[ankle],
                new[hip],
                new[ankle],
                self.min_kick_endpoint_motion,
                self.min_kick_extension_delta,
                self.min_kick_extension_ratio,
                min_height_change=self.min_kick_foot_height_change,
                old_motion_anchor=old[hip],
                new_motion_anchor=new[hip],
            )
            hip_to_ankle.commitment_frames = self._kick_commitment_frames(
                track_id, hip, hip, ankle, self.min_kick_endpoint_motion
            )
            hip_to_ankle.details.update(context)
            knee_to_ankle = self._limb_motion(
                old[knee],
                old[ankle],
                new[knee],
                new[ankle],
                self.min_kick_foot_motion or self.min_kick_endpoint_motion,
                self.min_kick_extension_delta,
                self.min_kick_extension_ratio,
                min_height_change=self.min_kick_foot_height_change,
                old_motion_anchor=old[hip],
                new_motion_anchor=new[hip],
            )
            knee_to_ankle.commitment_frames = self._kick_commitment_frames(
                track_id,
                knee,
                hip,
                ankle,
                self.min_kick_foot_motion or self.min_kick_endpoint_motion,
            )
            knee_to_ankle.details.update(context)
            candidates.extend((hip_to_ankle, knee_to_ankle))
            if self.min_kick_foot_motion > 0:
                foot_travel = self._foot_travel_motion(
                    old[hip],
                    old[knee],
                    old[ankle],
                    new[hip],
                    new[knee],
                    new[ankle],
                )
                foot_travel.details.update(context)
                candidates.append(foot_travel)
        return max(candidates, key=lambda motion: motion.score)

    def _kick_context(
        self,
        track_id: int,
        hip: int,
        ankle: int,
        support_ankle: int,
    ) -> dict[str, float]:
        old = self.history[track_id][0]
        new = self.history[track_id][-1]
        body_height = self._pose_body_height(new)
        foot_elevation = max(
            0.0,
            max(
                float(keypoints[support_ankle][1]) - float(keypoints[ankle][1])
                for keypoints in self.history[track_id]
            ),
        )
        old_support = self._relative_xy(old[support_ankle], old[hip])
        new_support = self._relative_xy(new[support_ankle], new[hip])
        return {
            "foot_elevation_body_heights": foot_elevation / body_height,
            "support_foot_motion_body_heights": dist(old_support, new_support)
            / body_height,
        }

    def _kick_score_qualified(self, kick: LimbMotion) -> bool:
        if kick.score >= self.min_kick_score:
            return True
        elevation = kick.details.get("foot_elevation_body_heights", 0.0)
        return (
            kick.score >= self.min_strike_score
            and self.strong_kick_foot_elevation_body_heights > 0
            and elevation >= self.strong_kick_foot_elevation_body_heights
        )

    def _strong_kick_kinematics(self, kick: LimbMotion) -> bool:
        elevation = kick.details.get("foot_elevation_body_heights", 0.0)
        return (
            (
                kick.score >= self.min_kick_score
                and self.strong_kick_foot_elevation_body_heights > 0
                and elevation >= self.strong_kick_foot_elevation_body_heights
            )
            or (
                self.strong_kick_score > 0
                and self.min_kick_foot_elevation_body_heights > 0
                and kick.score >= self.strong_kick_score
                and elevation >= self.min_kick_foot_elevation_body_heights
            )
        )

    @classmethod
    def _pose_body_height(cls, keypoints: Keypoints) -> float:
        y_values = [
            cls._xy(keypoints[index])[1]
            for index in (5, 6, 11, 12, 13, 14, 15, 16)
            if index < len(keypoints)
            and not (float(keypoints[index][0]) == 0 and float(keypoints[index][1]) == 0)
        ]
        return max(1.0, max(y_values) - min(y_values)) if y_values else 1.0

    def _kick_commitment_frames(
        self,
        track_id: int,
        extension_anchor: int,
        motion_anchor: int,
        ankle: int,
        min_endpoint_motion: float,
    ) -> int:
        first = self.history[track_id][0]
        old_extension = dist(self._xy(first[extension_anchor]), self._xy(first[ankle]))
        old_relative = self._relative_xy(first[ankle], first[motion_anchor])
        if old_extension <= 1e-6:
            return 0

        committed = 0
        for keypoints in reversed(self.history[track_id]):
            new_extension = dist(self._xy(keypoints[extension_anchor]), self._xy(keypoints[ankle]))
            new_relative = self._relative_xy(keypoints[ankle], keypoints[motion_anchor])
            endpoint_motion = dist(old_relative, new_relative)
            extension_delta = new_extension - old_extension
            extension_ratio = new_extension / old_extension
            height_change = abs(new_relative[1] - old_relative[1])
            if (
                endpoint_motion >= min_endpoint_motion
                and extension_delta >= self.min_kick_extension_delta
                and extension_ratio >= self.min_kick_extension_ratio
                and (
                    self.min_kick_foot_height_change <= 0
                    or height_change >= self.min_kick_foot_height_change
                )
            ):
                committed += 1
            else:
                break
        return committed

    def _foot_travel_motion(
        self,
        old_hip: Point,
        old_knee: Point,
        old_ankle: Point,
        new_hip: Point,
        new_knee: Point,
        new_ankle: Point,
    ) -> LimbMotion:
        old_hip_xy = self._xy(old_hip)
        old_knee_xy = self._xy(old_knee)
        old_ankle_xy = self._xy(old_ankle)
        new_hip_xy = self._xy(new_hip)
        new_knee_xy = self._xy(new_knee)
        new_ankle_xy = self._xy(new_ankle)

        old_relative_ankle = (
            old_ankle_xy[0] - old_hip_xy[0],
            old_ankle_xy[1] - old_hip_xy[1],
        )
        new_relative_ankle = (
            new_ankle_xy[0] - new_hip_xy[0],
            new_ankle_xy[1] - new_hip_xy[1],
        )
        endpoint_motion = dist(old_relative_ankle, new_relative_ankle)
        old_full_extension = dist(old_hip_xy, old_ankle_xy)
        new_full_extension = dist(new_hip_xy, new_ankle_xy)
        old_lower_extension = dist(old_knee_xy, old_ankle_xy)
        new_lower_extension = dist(new_knee_xy, new_ankle_xy)
        full_delta = new_full_extension - old_full_extension
        lower_delta = new_lower_extension - old_lower_extension
        extension_delta = max(full_delta, lower_delta)
        extension_ratio = (
            max(
                new_full_extension / old_full_extension if old_full_extension > 1e-6 else 0.0,
                new_lower_extension / old_lower_extension if old_lower_extension > 1e-6 else 0.0,
            )
        )
        height_change = abs(new_relative_ankle[1] - old_relative_ankle[1])

        motion_score = endpoint_motion / self.min_kick_foot_motion if self.min_kick_foot_motion else 0.0
        delta_score = extension_delta / self.min_kick_extension_delta if self.min_kick_extension_delta else 0.0
        ratio_score = extension_ratio / self.min_kick_extension_ratio if self.min_kick_extension_ratio else 0.0
        support_score = max(delta_score, ratio_score)
        if self.min_kick_foot_height_change > 0:
            support_score = max(support_score, height_change / self.min_kick_foot_height_change)
        score = min(motion_score, support_score)
        return LimbMotion(
            endpoint_motion=endpoint_motion,
            extension_delta=extension_delta,
            extension_ratio=extension_ratio,
            commitment_frames=1 if score > 0 else 0,
            score=max(0.0, score),
            details={"height_change": height_change},
        )

    @staticmethod
    def _xy(point: Point) -> tuple[float, float]:
        return float(point[0]), float(point[1])

    @classmethod
    def _relative_xy(cls, point: Point, anchor: Point) -> tuple[float, float]:
        point_xy = cls._xy(point)
        anchor_xy = cls._xy(anchor)
        return point_xy[0] - anchor_xy[0], point_xy[1] - anchor_xy[1]

    @classmethod
    def _limb_motion(
        cls,
        old_anchor: Point,
        old_endpoint: Point,
        new_anchor: Point,
        new_endpoint: Point,
        min_endpoint_motion: float,
        min_extension_delta: float,
        min_extension_ratio: float,
        min_height_change: float = 0.0,
        old_motion_anchor: Point | None = None,
        new_motion_anchor: Point | None = None,
    ) -> LimbMotion:
        old_anchor_xy = cls._xy(old_anchor)
        old_endpoint_xy = cls._xy(old_endpoint)
        new_anchor_xy = cls._xy(new_anchor)
        new_endpoint_xy = cls._xy(new_endpoint)
        old_extension = dist(old_anchor_xy, old_endpoint_xy)
        new_extension = dist(new_anchor_xy, new_endpoint_xy)
        if old_motion_anchor is not None and new_motion_anchor is not None:
            old_motion_endpoint = cls._relative_xy(old_endpoint, old_motion_anchor)
            new_motion_endpoint = cls._relative_xy(new_endpoint, new_motion_anchor)
        else:
            old_motion_endpoint = old_endpoint_xy
            new_motion_endpoint = new_endpoint_xy
        endpoint_motion = dist(old_motion_endpoint, new_motion_endpoint)
        extension_delta = new_extension - old_extension
        extension_ratio = new_extension / old_extension if old_extension > 1e-6 else 0.0
        height_change = abs(new_motion_endpoint[1] - old_motion_endpoint[1])
        motion_score = endpoint_motion / min_endpoint_motion if min_endpoint_motion else 0.0
        delta_score = extension_delta / min_extension_delta if min_extension_delta else 0.0
        ratio_score = extension_ratio / min_extension_ratio if min_extension_ratio else 0.0
        scores = [motion_score, delta_score, ratio_score]
        if min_height_change > 0:
            scores.append(height_change / min_height_change)
        score = min(scores)
        return LimbMotion(
            endpoint_motion=endpoint_motion,
            extension_delta=extension_delta,
            extension_ratio=extension_ratio,
            commitment_frames=1 if score > 0 else 0,
            score=max(0.0, score),
            details={"height_change": height_change},
        )
