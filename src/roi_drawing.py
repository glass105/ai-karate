"""ROI-filtered pose annotation helpers."""

from __future__ import annotations

from typing import Any

import cv2

COCO_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def candidate_inside_roi(
    candidate: dict[str, Any],
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> bool:
    center_x, center_y = candidate["center"]
    return roi[0] * width <= center_x <= roi[2] * width and roi[1] * height <= center_y <= roi[3] * height


def clip_box_to_roi(
    box: tuple[int, int, int, int],
    roi_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(box[0], roi_box[0])
    y1 = max(box[1], roi_box[1])
    x2 = min(box[2], roi_box[2])
    y2 = min(box[3], roi_box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_pose_candidates(
    annotated: Any,
    candidates: list[dict[str, Any]],
    roi: tuple[float, float, float, float],
    color: tuple[int, int, int] = (0, 220, 0),
    score_threshold: float = 0.35,
) -> tuple[int, int, int, int]:
    height, width = annotated.shape[:2]
    roi_box = (
        int(roi[0] * width),
        int(roi[1] * height),
        int(roi[2] * width),
        int(roi[3] * height),
    )
    for candidate in candidates:
        points = candidate["keypoints"]
        for start, end in COCO_SKELETON:
            if start >= len(points) or end >= len(points):
                continue
            if not (_point_inside_roi(points[start], roi_box) and _point_inside_roi(points[end], roi_box)):
                continue
            cv2.line(
                annotated,
                (int(points[start][0]), int(points[start][1])),
                (int(points[end][0]), int(points[end][1])),
                color,
                2,
            )
        for point in points:
            if _point_inside_roi(point, roi_box):
                cv2.circle(annotated, (int(point[0]), int(point[1])), 3, color, -1)
        draw_points = candidate.get("draw_keypoints")
        draw_scores = candidate.get("draw_scores")
        if draw_points is None or draw_scores is None:
            continue
        for point, score in zip(draw_points, draw_scores):
            if score < score_threshold or not _point_inside_roi(point, roi_box):
                continue
            cv2.circle(annotated, (int(point[0]), int(point[1])), 2, (0, 180, 255), -1)
    return roi_box


def _point_inside_roi(
    point: list[float] | tuple[float, float],
    roi_box: tuple[int, int, int, int],
) -> bool:
    return roi_box[0] <= point[0] <= roi_box[2] and roi_box[1] <= point[1] <= roi_box[3]
