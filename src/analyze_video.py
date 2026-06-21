"""Analyze a karate video with Ultralytics pose tracking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.fighter_identity import FighterIdentity, TrackedBox, pose_descriptor
from src.strike_counter import StrikeCounter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input fight video")
    parser.add_argument("--output-dir", default=Path("videos/output"), type=Path)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--fighter-a-name", default="Fighter A")
    parser.add_argument(
        "--fighter-a-start",
        choices=("left", "right"),
        default="left",
        help="Side where Gabriel starts and resets after stoppages/lineups.",
    )
    parser.add_argument("--fighter-a-red-gloves", action="store_true")
    parser.add_argument("--fighter-a-white-uniform", action="store_true")
    parser.add_argument("--fighter-a-black-belt", action="store_true")
    parser.add_argument("--fighter-a-taller", action="store_true")
    parser.add_argument(
        "--reset-to-start-side-after-missing",
        default=10,
        type=int,
        help="Prefer the configured starting side after this many missing frames",
    )
    parser.add_argument("--identity-recovery-confirmation-frames", default=3, type=int)
    parser.add_argument("--fighter-candidate-limit", default=4, type=int)
    parser.add_argument("--lineup-pause-frames", default=30, type=int)
    parser.add_argument("--lineup-motion-threshold", default=0.10, type=float)
    parser.add_argument("--lineup-separation-threshold", default=1.20, type=float)
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument(
        "--arena-roi",
        default="0,0,1,1",
        help="Normalized competition area as x1,y1,x2,y2; detections outside are ignored",
    )
    return parser


def parse_arena_roi(value: str) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise SystemExit("--arena-roi must contain four comma-separated numbers") from exc
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise SystemExit("--arena-roi values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return x1, y1, x2, y2


def is_inside_arena(
    box: TrackedBox,
    arena_roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = arena_roi
    return x1 * width <= box.center_x <= x2 * width and y1 * height <= box.center_y <= y2 * height


def appearance_descriptor(frame: Any, box: TrackedBox, cv2: Any) -> tuple[float, ...]:
    """Summarize crop color so identity recovery has a lightweight visual cue."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return ()
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return tuple(float(value) for value in histogram)


def fighter_color_scores(
    frame: Any,
    box: TrackedBox,
    keypoints: list[list[float]],
    cv2: Any,
) -> tuple[float, float, float]:
    """Estimate red gloves around wrists and a white uniform across the torso."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def crop_score(center_x: float, center_y: float, radius: int, mask: Any) -> float:
        x1, x2 = max(0, int(center_x) - radius), min(width, int(center_x) + radius)
        y1, y2 = max(0, int(center_y) - radius), min(height, int(center_y) + radius)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(mask[y1:y2, x1:x2].mean() / 255.0)

    low_red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    high_red = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(low_red, high_red)
    wrist_radius = max(6, int(min(box.width, box.height) * 0.10))
    wrist_scores = [
        crop_score(point[0], point[1], wrist_radius, red_mask)
        for point in (keypoints[index] for index in (9, 10))
        if len(point) >= 2 and not (point[0] == 0 and point[1] == 0)
    ]
    red_glove_score = sum(wrist_scores) / len(wrist_scores) if wrist_scores else 0.0

    white_mask = cv2.inRange(hsv, (0, 0, 145), (180, 85, 255))
    torso_x1 = max(0, int(box.x1 + box.width * 0.20))
    torso_x2 = min(width, int(box.x2 - box.width * 0.20))
    torso_y1 = max(0, int(box.y1 + box.height * 0.20))
    torso_y2 = min(height, int(box.y1 + box.height * 0.65))
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        white_uniform_score = 0.0
    else:
        white_uniform_score = float(
            white_mask[torso_y1:torso_y2, torso_x1:torso_x2].mean() / 255.0
        )
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    belt_y1 = max(0, int(box.y1 + box.height * 0.52))
    belt_y2 = min(height, int(box.y1 + box.height * 0.72))
    belt_x1 = max(0, int(box.x1 + box.width * 0.15))
    belt_x2 = min(width, int(box.x2 - box.width * 0.15))
    black_belt_score = (
        float(black_mask[belt_y1:belt_y2, belt_x1:belt_x2].mean() / 255.0)
        if belt_x2 > belt_x1 and belt_y2 > belt_y1
        else 0.0
    )
    return red_glove_score, white_uniform_score, black_belt_score


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    csv_path = args.output_dir / f"{stem}_tracks.csv"
    json_path = args.output_dir / f"{stem}_summary.json"
    video_path = args.output_dir / f"{stem}_annotated.mp4"

    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    arena_roi = parse_arena_roi(args.arena_roi)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    identity = FighterIdentity(
        args.fighter_a_name,
        args.fighter_a_start,
        expect_red_gloves=args.fighter_a_red_gloves,
        expect_white_uniform=args.fighter_a_white_uniform,
        expect_black_belt=args.fighter_a_black_belt,
        expect_taller=args.fighter_a_taller,
        reset_after_missing_frames=args.reset_to_start_side_after_missing,
        recovery_confirmation_frames=args.identity_recovery_confirmation_frames,
        fighter_candidate_limit=args.fighter_candidate_limit,
        lineup_pause_frames=args.lineup_pause_frames,
        lineup_motion_threshold=args.lineup_motion_threshold,
        lineup_separation_threshold=args.lineup_separation_threshold,
    )
    counter = StrikeCounter(fps=fps)
    model = YOLO(args.model)
    results = model.track(
        source=str(args.input),
        tracker=args.tracker,
        persist=True,
        stream=True,
        conf=args.confidence,
        iou=args.iou,
        verbose=False,
    )

    fieldnames = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "fighter_label",
        "confidence",
        "bbox",
        "keypoints",
        "red_glove_score",
        "white_uniform_score",
        "black_belt_score",
        "estimated_action",
    ]
    frame_count = 0
    named_counts = {"punches": 0, "fake_punches": 0, "kicks": 0}
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for frame_count, result in enumerate(results, start=1):
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or keypoints is None or boxes.id is None:
                writer.write(result.plot())
                continue

            tracked_boxes = []
            accepted = []
            for index, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().tolist()
                tracked = TrackedBox(int(box.id[0]), xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                if not is_inside_arena(tracked, arena_roi, width, height):
                    continue
                points = keypoints.xy[index].cpu().tolist()
                red_glove_score, white_uniform_score, black_belt_score = fighter_color_scores(
                    result.orig_img,
                    tracked,
                    points,
                    cv2,
                )
                tracked = TrackedBox(
                    tracked.track_id,
                    tracked.x1,
                    tracked.y1,
                    tracked.x2,
                    tracked.y2,
                    pose_descriptor(tracked, points),
                    appearance_descriptor(result.orig_img, tracked, cv2),
                    red_glove_score,
                    white_uniform_score,
                    black_belt_score,
                )
                tracked_boxes.append(tracked)
                accepted.append((index, box, tracked, points))
            identity.observe(tracked_boxes)

            annotated = result.plot()
            x1, y1, x2, y2 = arena_roi
            cv2.rectangle(
                annotated,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                (255, 255, 0),
                2,
            )
            for _, _, tracked, _ in accepted:
                if tracked.track_id == identity.active_track_id:
                    cv2.rectangle(
                        annotated,
                        (int(tracked.x1), int(tracked.y1)),
                        (int(tracked.x2), int(tracked.y2)),
                        (0, 255, 255),
                        3,
                    )
                    cv2.putText(
                        annotated,
                        args.fighter_a_name,
                        (int(tracked.x1), max(20, int(tracked.y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
            writer.write(annotated)

            for _, box, tracked, points in accepted:
                track_id = tracked.track_id
                xyxy = [tracked.x1, tracked.y1, tracked.x2, tracked.y2]
                action = counter.update(track_id, points, frame_index=frame_count)
                fighter_label = identity.label(track_id)
                if fighter_label == args.fighter_a_name and action:
                    action_key = {"punch": "punches", "fake_punch": "fake_punches", "kick": "kicks"}.get(action)
                    if action_key:
                        named_counts[action_key] += 1
                csv_writer.writerow(
                    {
                        "frame": frame_count,
                        "timestamp_seconds": round((frame_count - 1) / fps, 3),
                        "track_id": track_id,
                        "fighter_label": fighter_label,
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": json.dumps(xyxy),
                        "keypoints": json.dumps(points),
                        "red_glove_score": round(tracked.red_glove_score, 4),
                        "white_uniform_score": round(tracked.white_uniform_score, 4),
                        "black_belt_score": round(tracked.black_belt_score, 4),
                        "estimated_action": action,
                    }
                )
    writer.release()

    summary = {
        "input": str(args.input),
        "model": args.model,
        "tracker": args.tracker,
        "fps": fps,
        "frames_processed": frame_count,
        "fighter_a_name": args.fighter_a_name,
        "fighter_a_track_id": identity.fighter_a_track_id,
        "fighter_a_track_ids": sorted(identity.fighter_a_track_ids),
        "identity_recoveries": identity.recovery_count,
        "arena_roi": arena_roi,
        "fighter_a_cues": {
            "start_side": args.fighter_a_start,
            "reset_side": args.fighter_a_start,
            "red_gloves": args.fighter_a_red_gloves,
            "white_uniform": args.fighter_a_white_uniform,
            "black_belt": args.fighter_a_black_belt,
            "taller_height": args.fighter_a_taller,
            "reset_to_start_side_after_missing_frames": args.reset_to_start_side_after_missing,
            "recovery_confirmation_frames": args.identity_recovery_confirmation_frames,
            "fighter_candidate_limit": args.fighter_candidate_limit,
            "lineup_pause_frames": args.lineup_pause_frames,
            "lineup_motion_threshold": args.lineup_motion_threshold,
            "lineup_separation_threshold": args.lineup_separation_threshold,
        },
        "counts_by_track_id": counter.counts,
        "named_fighter_candidate_counts": named_counts,
        "artifacts": {
            "annotated_video": str(video_path),
            "tracks_csv": str(csv_path),
        },
        "limitations": [
            "Strike counts are heuristic candidates and require review.",
            "Identity recovery is heuristic and should be checked against the annotated video.",
            "Tune --arena-roi for each camera angle so spectators and officials are ignored.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
