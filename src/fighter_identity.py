"""Map a named fighter to a tracker ID using the starting side of the frame."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2


class FighterIdentity:
    """Assign fighter A once and retain the tracker mapping for the clip."""

    def __init__(self, fighter_a_name: str, fighter_a_start: str) -> None:
        if fighter_a_start not in {"left", "right"}:
            raise ValueError("fighter_a_start must be 'left' or 'right'")
        self.fighter_a_name = fighter_a_name
        self.fighter_a_start = fighter_a_start
        self.fighter_a_track_id: int | None = None

    def observe(self, boxes: Iterable[TrackedBox]) -> int | None:
        boxes = list(boxes)
        if self.fighter_a_track_id is not None:
            return self.fighter_a_track_id
        if not boxes:
            return None

        ordered = sorted(boxes, key=lambda box: box.center_x)
        selected = ordered[0] if self.fighter_a_start == "left" else ordered[-1]
        self.fighter_a_track_id = selected.track_id
        return self.fighter_a_track_id

    def label(self, track_id: int) -> str:
        if track_id == self.fighter_a_track_id:
            return self.fighter_a_name
        return f"fighter-{track_id}"
