"""Analyze a karate video with Ultralytics pose tracking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.fighter_identity import FighterIdentity, TrackedBox
from src.strike_counter import StrikeCounter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input fight video")
    parser.add_argument("--output-dir", default=Path("videos/output"), type=Path)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--fighter-a-name", default="Fighter A")
    parser.add_argument("--fighter-a-start", choices=("left", "right"), default="left")
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    return parser


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

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    identity = FighterIdentity(args.fighter_a_name, args.fighter_a_start)
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
        "estimated_action",
    ]
    frame_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for frame_count, result in enumerate(results, start=1):
            boxes = result.boxes
            keypoints = result.keypoints
            writer.write(result.plot())
            if boxes is None or keypoints is None or boxes.id is None:
                continue

            tracked_boxes = []
            for box in boxes:
                xyxy = box.xyxy[0].cpu().tolist()
                tracked_boxes.append(
                    TrackedBox(int(box.id[0]), xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                )
            identity.observe(tracked_boxes)

            for index, box in enumerate(boxes):
                track_id = int(box.id[0])
                xyxy = box.xyxy[0].cpu().tolist()
                points = keypoints.xy[index].cpu().tolist()
                action = counter.update(track_id, points)
                csv_writer.writerow(
                    {
                        "frame": frame_count,
                        "timestamp_seconds": round((frame_count - 1) / fps, 3),
                        "track_id": track_id,
                        "fighter_label": identity.label(track_id),
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": json.dumps(xyxy),
                        "keypoints": json.dumps(points),
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
        "counts_by_track_id": counter.counts,
        "artifacts": {
            "annotated_video": str(video_path),
            "tracks_csv": str(csv_path),
        },
        "limitations": [
            "Strike counts are heuristic candidates and require review.",
            "Starting-side identity mapping does not yet recover from tracker ID swaps.",
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
