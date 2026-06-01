"""Simple pose heuristics for prototype strike counting."""

from collections import defaultdict, deque
from math import dist
from typing import Sequence

Point = Sequence[float]
Keypoints = Sequence[Point]


class StrikeCounter:
    """Estimate strike candidates from pose extension over a short history."""

    def __init__(
        self,
        fps: float,
        history_frames: int = 6,
        punch_cooldown_seconds: float = 0.35,
        kick_cooldown_seconds: float = 0.50,
    ) -> None:
        self.history_frames = history_frames
        self.punch_cooldown_frames = max(1, int(fps * punch_cooldown_seconds))
        self.kick_cooldown_frames = max(1, int(fps * kick_cooldown_seconds))
        self.history: dict[int, deque[Keypoints]] = defaultdict(
            lambda: deque(maxlen=self.history_frames)
        )
        self.cooldowns: dict[int, dict[str, int]] = defaultdict(
            lambda: {"punch": 0, "kick": 0}
        )
        self.counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"punches": 0, "kicks": 0}
        )

    def update(self, track_id: int, keypoints: Keypoints) -> str:
        if track_id < 0 or len(keypoints) < 17:
            return ""

        self.history[track_id].append(keypoints)
        self._tick_cooldowns(track_id)
        if len(self.history[track_id]) < self.history_frames:
            return ""

        if self._detect_punch(track_id) and self.cooldowns[track_id]["punch"] == 0:
            self.counts[track_id]["punches"] += 1
            self.cooldowns[track_id]["punch"] = self.punch_cooldown_frames
            return "punch"

        if self._detect_kick(track_id) and self.cooldowns[track_id]["kick"] == 0:
            self.counts[track_id]["kicks"] += 1
            self.cooldowns[track_id]["kick"] = self.kick_cooldown_frames
            return "kick"

        return ""

    def _tick_cooldowns(self, track_id: int) -> None:
        for strike_type in ("punch", "kick"):
            current = self.cooldowns[track_id][strike_type]
            self.cooldowns[track_id][strike_type] = max(0, current - 1)

    def _detect_punch(self, track_id: int) -> bool:
        old, new = self.history[track_id][0], self.history[track_id][-1]
        return any(
            self._extension_increased(old[shoulder], old[wrist], new[shoulder], new[wrist])
            for shoulder, wrist in ((5, 9), (6, 10))
        )

    def _detect_kick(self, track_id: int) -> bool:
        old, new = self.history[track_id][0], self.history[track_id][-1]
        return any(
            self._extension_increased(old[hip], old[ankle], new[hip], new[ankle])
            for hip, ankle in ((11, 15), (12, 16))
        )

    @staticmethod
    def _extension_increased(
        old_anchor: Point,
        old_endpoint: Point,
        new_anchor: Point,
        new_endpoint: Point,
    ) -> bool:
        old_extension = dist(old_anchor, old_endpoint)
        new_extension = dist(new_anchor, new_endpoint)
        endpoint_motion = dist(old_endpoint, new_endpoint)
        return new_extension > old_extension * 1.25 and endpoint_motion > 25
