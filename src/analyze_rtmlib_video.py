"""Analyze a karate video with RTMPose or RTMW through rtmlib."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.fighter_identity import FighterIdentity, TrackedBox, pose_descriptor
from src.roi_drawing import candidate_inside_roi, clip_box_to_roi, draw_pose_candidates
from src.strike_counter import StrikeCounter

COCO_BODY_KEYPOINT_COUNT = 17


@dataclass
class PoseTrack:
    track_id: int
    center: tuple[float, float]
    missing_frames: int = 0


class PoseTracker:
    """Assign short-lived IDs to RTM detections using center proximity."""

    def __init__(self, max_distance: float = 180.0, max_missing_frames: int = 15) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: dict[int, PoseTrack] = {}

    def update(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unmatched_track_ids = set(self._tracks)
        for candidate in sorted(candidates, key=lambda item: item["center"][0]):
            available = [
                (dist(candidate["center"], self._tracks[track_id].center), track_id)
                for track_id in unmatched_track_ids
            ]
            distance, track_id = min(available, default=(self.max_distance + 1, -1))
            if distance > self.max_distance:
                track_id = self._next_track_id
                self._next_track_id += 1
            else:
                unmatched_track_ids.remove(track_id)
            candidate["track_id"] = track_id
            self._tracks[track_id] = PoseTrack(track_id, candidate["center"])
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                del self._tracks[track_id]
        return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("rtmpose", "rtmw"), required=True)
    parser.add_argument("--fighter-a-name", default="Gabriel")
    parser.add_argument(
        "--fighter-a-start",
        choices=("left", "right"),
        default="left",
        help="Side where Gabriel starts and resets after stoppages/lineups.",
    )
    parser.add_argument("--fighter-a-black-belt", action="store_true")
    parser.add_argument("--fighter-a-taller", action="store_true")
    parser.add_argument(
        "--fighter-a-glove-color",
        choices=("red", "white", "blue", "none"),
        default="red",
        help=(
            "Expected Gabriel glove color for this run. The selected color is required for initial lock/reset; "
            "'none' disables positive glove-color evidence while reject-glove flags still apply."
        ),
    )
    parser.add_argument("--fighter-a-require-red-gloves", action="store_true")
    parser.add_argument("--fighter-a-require-white-gloves", action="store_true")
    parser.add_argument("--fighter-a-require-blue-gloves", action="store_true")
    parser.add_argument("--fighter-a-reject-red-gloves", action="store_true")
    parser.add_argument("--fighter-a-reject-white-gloves", action="store_true")
    parser.add_argument("--fighter-a-reject-blue-gloves", action="store_true")
    parser.add_argument("--fighter-a-require-standing", action="store_true")
    parser.add_argument("--fighter-a-min-red-glove-score", default=0.15, type=float)
    parser.add_argument("--fighter-a-min-white-glove-score", default=0.02, type=float)
    parser.add_argument("--fighter-a-min-blue-glove-score", default=0.15, type=float)
    parser.add_argument("--fighter-a-min-standing-score", default=0.45, type=float)
    parser.add_argument("--reset-to-start-side-after-missing", default=10, type=int)
    parser.add_argument("--identity-recovery-confirmation-frames", default=3, type=int)
    parser.add_argument("--fighter-candidate-limit", default=4, type=int)
    parser.add_argument("--lineup-pause-frames", default=30, type=int)
    parser.add_argument("--lineup-motion-threshold", default=0.10, type=float)
    parser.add_argument("--lineup-separation-threshold", default=1.20, type=float)
    parser.add_argument("--arena-roi", default="0.2,0.1,0.8,0.9")
    parser.add_argument("--keypoint-threshold", default=0.35, type=float)
    return parser


def glove_color_settings(args: argparse.Namespace) -> dict[str, bool]:
    target = args.fighter_a_glove_color
    return {
        "expect_red": target == "red" or args.fighter_a_require_red_gloves,
        "expect_white": target == "white" or args.fighter_a_require_white_gloves,
        "expect_blue": target == "blue" or args.fighter_a_require_blue_gloves,
        "require_red": target == "red" or args.fighter_a_require_red_gloves,
        "require_white": target == "white" or args.fighter_a_require_white_gloves,
        "require_blue": target == "blue" or args.fighter_a_require_blue_gloves,
        "reject_red": args.fighter_a_reject_red_gloves,
        "reject_white": args.fighter_a_reject_white_gloves,
        "reject_blue": args.fighter_a_reject_blue_gloves,
    }


def parse_roi(value: str) -> tuple[float, float, float, float]:
    roi = tuple(float(part) for part in value.split(","))
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise SystemExit("--arena-roi must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return roi


def box_for_pose(points: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int, int] | None:
    visible = points[scores >= threshold]
    if len(visible) < 4:
        return None
    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def coco_body_keypoints(points: np.ndarray) -> list[list[float]]:
    """Normalize RTMPose and RTMW output to the COCO body layout."""
    if len(points) < COCO_BODY_KEYPOINT_COUNT:
        raise ValueError("RTM pose output must contain at least 17 body keypoints")
    return [[float(x), float(y)] for x, y in points[:COCO_BODY_KEYPOINT_COUNT]]


def color_scores(
    frame: Any,
    points: np.ndarray,
    scores: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    low_red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    high_red = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(low_red, high_red)
    blue_mask = cv2.inRange(hsv, (90, 80, 60), (135, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 145), (180, 85, 255))

    def mask_score(mask: Any, center_x: int, center_y: int, sample_radius: int) -> float | None:
        crop = mask[
            max(0, center_y - sample_radius) : min(height, center_y + sample_radius),
            max(0, center_x - sample_radius) : min(width, center_x + sample_radius),
        ]
        if not crop.size:
            return None
        return float(crop.mean() / 255.0)

    wrist_values = []
    white_wrist_values = []
    blue_wrist_values = []
    radius = max(6, int(min(box_width, box_height) * 0.10))
    for index in (9, 10):
        if index >= len(points) or scores[index] < 0.2:
            continue
        wrist_x, wrist_y = (int(value) for value in points[index])
        red_value = mask_score(red_mask, wrist_x, wrist_y, radius)
        if red_value is not None:
            wrist_values.append(red_value)
        white_value = mask_score(white_mask, wrist_x, wrist_y, radius)
        if white_value is not None:
            white_wrist_values.append(white_value)
        blue_value = mask_score(blue_mask, wrist_x, wrist_y, radius)
        if blue_value is not None:
            blue_wrist_values.append(blue_value)

    torso_red = red_mask[
        max(0, int(y1 + box_height * 0.20)) : min(height, int(y1 + box_height * 0.65)),
        max(0, int(x1 + box_width * 0.20)) : min(width, int(x2 - box_width * 0.20)),
    ]
    torso_red_score = float(torso_red.mean() / 255.0) if torso_red.size else 0.0
    elbow_values = []
    for index in (7, 8):
        if index >= len(points) or scores[index] < 0.2:
            continue
        elbow_x, elbow_y = (int(value) for value in points[index])
        elbow_value = mask_score(red_mask, elbow_x, elbow_y, radius)
        if elbow_value is not None:
            elbow_values.append(elbow_value)
    elbow_red_score = sum(elbow_values) / len(elbow_values) if elbow_values else 0.0
    clothing_red_score = max(torso_red_score, elbow_red_score * 0.80)
    adjusted_wrist_values = [max(0.0, value - clothing_red_score * 0.75) for value in wrist_values]
    red_score = sum(adjusted_wrist_values) / len(adjusted_wrist_values) if adjusted_wrist_values else 0.0
    white_glove_score = sum(white_wrist_values) / len(white_wrist_values) if white_wrist_values else 0.0
    blue_score = sum(blue_wrist_values) / len(blue_wrist_values) if blue_wrist_values else 0.0

    torso = white_mask[
        max(0, int(y1 + box_height * 0.20)) : min(height, int(y1 + box_height * 0.65)),
        max(0, int(x1 + box_width * 0.20)) : min(width, int(x2 - box_width * 0.20)),
    ]
    white_score = float(torso.mean() / 255.0) if torso.size else 0.0
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    belt = black_mask[
        max(0, int(y1 + box_height * 0.52)) : min(height, int(y1 + box_height * 0.72)),
        max(0, int(x1 + box_width * 0.15)) : min(width, int(x2 - box_width * 0.15)),
    ]
    black_belt_score = float(belt.mean() / 255.0) if belt.size else 0.0
    return red_score, white_glove_score, white_score, black_belt_score, blue_score


def standing_score(points: np.ndarray, scores: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Estimate whether a pose is upright enough to exclude seated spectators."""
    visible = [
        points[index][1]
        for index in (5, 6, 11, 12, 13, 14, 15, 16)
        if index < len(points) and scores[index] >= 0.2
    ]
    if len(visible) < 4:
        return 0.0
    box_height = max(1, box[3] - box[1])
    return min(1.0, (max(visible) - min(visible)) / box_height)


def appearance_descriptor(frame: Any, box: TrackedBox) -> tuple[float, ...]:
    """Summarize crop color so identity recovery has a lightweight visual cue."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return ()
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    return tuple(float(value) for value in cv2.normalize(histogram, histogram).flatten())


def identity_box(candidate: dict[str, Any], frame: Any) -> TrackedBox:
    x1, y1, x2, y2 = candidate["box"]
    tracked = TrackedBox(candidate["track_id"], x1, y1, x2, y2)
    return TrackedBox(
        tracked.track_id,
        tracked.x1,
        tracked.y1,
        tracked.x2,
        tracked.y2,
        pose_descriptor(tracked, candidate["keypoints"]),
        appearance_descriptor(frame, tracked),
        candidate["red_glove_score"],
        candidate["white_glove_score"],
        candidate["white_uniform_score"],
        candidate["black_belt_score"],
        candidate["blue_glove_score"],
        candidate["standing_score"],
        candidate.get("reference_match_score", 0.0),
        candidate.get("pose_reference_match_score", 0.0),
        candidate.get("face_match_score", 0.0),
        candidate.get("face_detected", False),
        candidate.get("exclude_reference_match_score", 0.0),
        candidate.get("exclude_body_match_score", 0.0),
        candidate.get("exclude_face_match_score", 0.0),
        candidate.get("exclude_face_detected", False),
        candidate.get("competition_fighter_score", 1.0),
    )


def select_gabriel(
    candidates: list[dict[str, Any]],
    start_side: str,
    last_center: tuple[float, float] | None,
    missing_frames: int,
    reset_after_missing: int,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    resetting = last_center is None or missing_frames >= reset_after_missing
    centers = [candidate["center"] for candidate in candidates]
    side_x = min(center[0] for center in centers) if start_side == "left" else max(center[0] for center in centers)

    def score(candidate: dict[str, Any]) -> float:
        center_x, center_y = candidate["center"]
        box = candidate["box"]
        scale = max(1, box[2] - box[0], box[3] - box[1])
        distance_score = 0.0
        if last_center is not None:
            distance_score = (((center_x - last_center[0]) ** 2 + (center_y - last_center[1]) ** 2) ** 0.5) / scale
        appearance_score = (1.0 - candidate["red_glove_score"]) * 0.30 + (
            1.0 - candidate["white_uniform_score"]
        ) * 0.15
        side_penalty = 0.0 if not resetting or center_x == side_x else 0.50
        return distance_score * (0.25 if resetting else 0.55) + appearance_score + side_penalty

    return min(candidates, key=score)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    from rtmlib import Body, Wholebody

    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_roi(args.arena_roi)
    estimator = (
        Body(mode="performance", backend="onnxruntime", device="cuda")
        if args.backend == "rtmpose"
        else Wholebody(mode="performance", backend="onnxruntime", device="cuda")
    )
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = args.output_dir / f"{args.input.stem}_annotated.mp4"
    summary_path = args.output_dir / f"{args.input.stem}_summary.json"
    csv_path = args.output_dir / f"{args.input.stem}_tracks.csv"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = PoseTracker()
    counter = StrikeCounter(fps=fps)
    glove_settings = glove_color_settings(args)
    identity = FighterIdentity(
        args.fighter_a_name,
        args.fighter_a_start,
        expect_red_gloves=glove_settings["expect_red"],
        expect_white_gloves=glove_settings["expect_white"],
        expect_blue_gloves=glove_settings["expect_blue"],
        expect_white_uniform=True,
        expect_black_belt=args.fighter_a_black_belt,
        expect_taller=args.fighter_a_taller,
        require_red_gloves=glove_settings["require_red"],
        require_white_gloves=glove_settings["require_white"],
        require_blue_gloves=glove_settings["require_blue"],
        reject_red_gloves=glove_settings["reject_red"],
        reject_white_gloves=glove_settings["reject_white"],
        reject_blue_gloves=glove_settings["reject_blue"],
        require_standing=args.fighter_a_require_standing,
        min_red_glove_score=args.fighter_a_min_red_glove_score,
        min_white_glove_score=args.fighter_a_min_white_glove_score,
        min_blue_glove_score=args.fighter_a_min_blue_glove_score,
        min_standing_score=args.fighter_a_min_standing_score,
        reset_after_missing_frames=args.reset_to_start_side_after_missing,
        recovery_confirmation_frames=args.identity_recovery_confirmation_frames,
        fighter_candidate_limit=args.fighter_candidate_limit,
        lineup_pause_frames=args.lineup_pause_frames,
        lineup_motion_threshold=args.lineup_motion_threshold,
        lineup_separation_threshold=args.lineup_separation_threshold,
    )
    frame_count = 0
    gabriel_frames = 0
    named_counts = {"punches": 0, "kicks": 0}
    fieldnames = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "fighter_label",
        "bbox",
        "keypoints",
        "red_glove_score",
        "white_glove_score",
        "blue_glove_score",
        "white_uniform_score",
        "black_belt_score",
        "standing_score",
        "estimated_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            points_batch, scores_batch = estimator(frame)
            candidates = []
            for points, scores in zip(points_batch, scores_batch):
                points = np.asarray(points)
                scores = np.asarray(scores)
                box = box_for_pose(points, scores, args.keypoint_threshold)
                if box is None:
                    continue
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                red_score, white_glove_score, white_score, black_belt_score, blue_score = color_scores(frame, points, scores, box)
                candidate = {
                    "box": box,
                    "center": center,
                    "keypoints": coco_body_keypoints(points),
                    "draw_keypoints": [[float(x), float(y)] for x, y in points],
                    "draw_scores": [float(score) for score in scores],
                    "red_glove_score": red_score,
                    "white_glove_score": white_glove_score,
                    "white_uniform_score": white_score,
                    "black_belt_score": black_belt_score,
                    "blue_glove_score": blue_score,
                    "standing_score": standing_score(points, scores, box),
                }
                if candidate_inside_roi(candidate, roi, width, height):
                    candidates.append(candidate)
            annotated = frame.copy()
            roi_box = draw_pose_candidates(annotated, candidates, roi, score_threshold=args.keypoint_threshold)
            candidates = tracker.update(candidates)
            selected_track_id = identity.observe(identity_box(candidate, frame) for candidate in candidates)
            selected = next(
                (candidate for candidate in candidates if candidate["track_id"] == selected_track_id),
                None,
            )
            if selected is not None:
                gabriel_frames += 1
                clipped = clip_box_to_roi(selected["box"], roi_box)
                if clipped is not None:
                    x1, y1, x2, y2 = clipped
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(annotated, args.fighter_a_name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for candidate in candidates:
                track_id = candidate["track_id"]
                action = counter.update(track_id, candidate["keypoints"])
                fighter_label = identity.label(track_id)
                if fighter_label == args.fighter_a_name and action:
                    named_counts[f"{action}es" if action == "punch" else "kicks"] += 1
                csv_writer.writerow(
                    {
                        "frame": frame_count,
                        "timestamp_seconds": round((frame_count - 1) / fps, 3),
                        "track_id": track_id,
                        "fighter_label": fighter_label,
                        "bbox": json.dumps(candidate["box"]),
                        "keypoints": json.dumps(candidate["keypoints"]),
                        "red_glove_score": round(candidate["red_glove_score"], 4),
                        "white_glove_score": round(candidate["white_glove_score"], 4),
                        "blue_glove_score": round(candidate["blue_glove_score"], 4),
                        "white_uniform_score": round(candidate["white_uniform_score"], 4),
                        "black_belt_score": round(candidate["black_belt_score"], 4),
                        "standing_score": round(candidate["standing_score"], 4),
                        "estimated_action": action,
                    }
                )
            cv2.rectangle(
                annotated,
                (roi_box[0], roi_box[1]),
                (roi_box[2], roi_box[3]),
                (255, 255, 0),
                2,
            )
            writer.write(annotated)
    capture.release()
    writer.release()
    summary = {
        "input": str(args.input),
        "model_family": args.backend,
        "runtime": "rtmlib",
        "mode": "performance",
        "fps": fps,
        "frames_processed": frame_count,
        "fighter_a_name": args.fighter_a_name,
        "fighter_a_cues": {
            "start_side": args.fighter_a_start,
            "reset_side": args.fighter_a_start,
            "white_uniform": True,
            "glove_color": args.fighter_a_glove_color,
            "glove_positive_evidence": args.fighter_a_glove_color != "none"
            or glove_settings["require_red"]
            or glove_settings["require_white"]
            or glove_settings["require_blue"],
            "glove_rejection_only": args.fighter_a_glove_color == "none"
            and (
                glove_settings["reject_red"]
                or glove_settings["reject_white"]
                or glove_settings["reject_blue"]
            ),
            "red_gloves": glove_settings["expect_red"],
            "white_gloves": glove_settings["expect_white"],
            "blue_gloves": glove_settings["expect_blue"],
            "black_belt": args.fighter_a_black_belt,
            "taller_height": args.fighter_a_taller,
            "require_red_gloves": glove_settings["require_red"],
            "require_white_gloves": glove_settings["require_white"],
            "require_blue_gloves": glove_settings["require_blue"],
            "reject_red_gloves": glove_settings["reject_red"],
            "reject_white_gloves": glove_settings["reject_white"],
            "reject_blue_gloves": glove_settings["reject_blue"],
            "require_standing": args.fighter_a_require_standing,
            "min_red_glove_score": args.fighter_a_min_red_glove_score,
            "min_white_glove_score": args.fighter_a_min_white_glove_score,
            "min_blue_glove_score": args.fighter_a_min_blue_glove_score,
            "min_standing_score": args.fighter_a_min_standing_score,
            "reset_to_start_side_after_missing_frames": args.reset_to_start_side_after_missing,
            "recovery_confirmation_frames": args.identity_recovery_confirmation_frames,
            "fighter_candidate_limit": args.fighter_candidate_limit,
            "lineup_pause_frames": args.lineup_pause_frames,
            "lineup_motion_threshold": args.lineup_motion_threshold,
            "lineup_separation_threshold": args.lineup_separation_threshold,
        },
        "arena_roi": roi,
        "gabriel_frames": gabriel_frames,
        "fighter_a_track_id": identity.fighter_a_track_id,
        "fighter_a_track_ids": sorted(identity.fighter_a_track_ids),
        "identity_recoveries": identity.recovery_count,
        "identity_resets": identity.reset_count,
        "counts_by_track_id": counter.counts,
        "named_fighter_candidate_counts": named_counts,
        "artifacts": {
            "annotated_video": str(output_path),
            "tracks_csv": str(csv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
