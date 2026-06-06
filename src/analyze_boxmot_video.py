"""Analyze karate video with YOLO or RTM poses and BoxMOT OC-SORT trackers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.analyze_rtmlib_video import (
    box_for_pose,
    coco_body_keypoints,
    color_scores,
    identity_box,
    parse_roi,
    standing_score,
)
from src.fighter_identity import FighterIdentity, TrackedBox, pose_descriptor
from src.roi_drawing import candidate_inside_roi, clip_box_to_roi, draw_pose_candidates
from src.strike_counter import StrikeCounter


RTMW_POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
RTMPOSE_BODY_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pose-backend", choices=("yolo", "rtmpose", "rtmw", "yolo-rtmw"), required=True)
    parser.add_argument("--model", default="yolo26l-pose.pt")
    parser.add_argument(
        "--rtm-detector-backend",
        choices=("yolo", "rtmlib"),
        default="yolo",
        help="Person detector used before RTM pose inference. yolo keeps RTMW pose but avoids RTMLib YOLOX downloads.",
    )
    parser.add_argument("--tracker", choices=("ocsort", "deepocsort", "hybridsort", "strongsort", "boosttrack"), required=True)
    parser.add_argument("--reid-weights", default="models/osnet_x0_25_msmt17.pt", type=Path)
    parser.add_argument("--fighter-a-name", default="Gabriel")
    parser.add_argument("--fighter-a-start", choices=("left", "right"), default="left")
    parser.add_argument("--fighter-a-black-belt", action="store_true")
    parser.add_argument("--fighter-a-taller", action="store_true")
    parser.add_argument("--fighter-a-require-red-gloves", action="store_true")
    parser.add_argument("--fighter-a-require-white-gloves", action="store_true")
    parser.add_argument("--fighter-a-reject-blue-gloves", action="store_true")
    parser.add_argument("--fighter-a-require-standing", action="store_true")
    parser.add_argument(
        "--fighter-a-reference-images",
        nargs="*",
        default=[],
        type=Path,
        help="Gabriel reference image files or directories used as an appearance cue.",
    )
    parser.add_argument("--fighter-a-require-reference-match", action="store_true")
    parser.add_argument("--fighter-a-min-reference-match-score", default=0.05, type=float)
    parser.add_argument("--fighter-a-enable-face-match", action="store_true")
    parser.add_argument("--fighter-a-reject-face-mismatch", action="store_true")
    parser.add_argument("--fighter-a-min-face-match-score", default=0.35, type=float)
    parser.add_argument("--fighter-a-min-red-glove-score", default=0.35, type=float)
    parser.add_argument("--fighter-a-min-white-glove-score", default=0.02, type=float)
    parser.add_argument("--fighter-a-min-standing-score", default=0.45, type=float)
    parser.add_argument("--experiment-label")
    parser.add_argument("--reset-to-start-side-after-missing", default=15, type=int)
    parser.add_argument("--identity-recovery-confirmation-frames", default=2, type=int)
    parser.add_argument("--fighter-candidate-limit", default=4, type=int)
    parser.add_argument("--lineup-pause-frames", default=45, type=int)
    parser.add_argument("--lineup-motion-threshold", default=0.06, type=float)
    parser.add_argument("--lineup-separation-threshold", default=1.50, type=float)
    parser.add_argument("--arena-roi", default="0.2,0.1,0.8,0.9")
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--keypoint-threshold", default=0.35, type=float)
    return parser


class BoxGuidedRtmPoseEstimator:
    """Run RTMPose/RTMW on externally supplied person boxes."""

    def __init__(self, pose_model: Any, keypoint_count: int) -> None:
        self.pose_model = pose_model
        self.keypoint_count = keypoint_count
        self.boxes: list[tuple[int, int, int, int]] = []

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def __call__(self, image: Any) -> tuple[np.ndarray, np.ndarray]:
        if not self.boxes:
            return (
                np.empty((0, self.keypoint_count, 2), dtype=np.float32),
                np.empty((0, self.keypoint_count), dtype=np.float32),
            )
        return self.pose_model(image, bboxes=[list(box) for box in self.boxes])


def build_box_guided_rtm_estimator(pose_backend: str) -> BoxGuidedRtmPoseEstimator:
    from rtmlib import RTMPose

    if pose_backend == "rtmpose":
        pose_url = RTMPOSE_BODY_URL
        keypoint_count = 17
    else:
        pose_url = RTMW_POSE_URL
        keypoint_count = 133
    pose_model = RTMPose(
        pose_url,
        model_input_size=(288, 384),
        backend="onnxruntime",
        device="cuda",
    )
    return BoxGuidedRtmPoseEstimator(pose_model, keypoint_count)


def build_tracker(tracker_name: str, reid_weights: Path) -> Any:
    from boxmot.trackers.tracker_zoo import create_tracker

    kwargs: dict[str, Any] = {"tracker_type": tracker_name, "per_class": False}
    if tracker_name in {"deepocsort", "hybridsort", "strongsort", "boosttrack"}:
        kwargs.update({"reid_weights": reid_weights, "device": "0", "half": True})
    return create_tracker(**kwargs)


def detections_array(candidates: list[dict[str, Any]]) -> np.ndarray:
    if not candidates:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(
        [
            [
                candidate["box"][0],
                candidate["box"][1],
                candidate["box"][2],
                candidate["box"][3],
                candidate["confidence"],
                0,
            ]
            for candidate in candidates
        ],
        dtype=np.float32,
    )


def attach_tracks(candidates: list[dict[str, Any]], tracks: Any) -> list[dict[str, Any]]:
    if tracks.size == 0:
        return []
    attached = []
    for index, track_id in zip(tracks.det_ind, tracks.id):
        if 0 <= index < len(candidates):
            candidate = candidates[index]
            candidate["track_id"] = int(track_id)
            attached.append(candidate)
    return attached


def yolo_candidates(result: Any, frame: Any, threshold: float) -> list[dict[str, Any]]:
    candidates = []
    if result.boxes is not None and result.keypoints is not None:
        for index, box in enumerate(result.boxes):
            points = np.asarray(result.keypoints.xy[index].cpu().tolist())
            scores = np.asarray(result.keypoints.conf[index].cpu().tolist())
            xyxy = box.xyxy[0].cpu().tolist()
            candidate_box = tuple(int(value) for value in xyxy)
            red_score, white_glove_score, white_score, black_belt_score, blue_score = color_scores(frame, points, scores, candidate_box)
            candidates.append(
                {
                    "box": candidate_box,
                    "center": ((candidate_box[0] + candidate_box[2]) / 2, (candidate_box[1] + candidate_box[3]) / 2),
                    "keypoints": coco_body_keypoints(points),
                    "identity_keypoints": [[float(x), float(y)] for x, y in points],
                    "draw_keypoints": [[float(x), float(y)] for x, y in points],
                    "draw_scores": [float(score) for score in scores],
                    "red_glove_score": red_score,
                    "white_glove_score": white_glove_score,
                    "white_uniform_score": white_score,
                    "black_belt_score": black_belt_score,
                    "blue_glove_score": blue_score,
                    "standing_score": standing_score(points, scores, candidate_box),
                    "confidence": float(box.conf[0]),
                }
            )
    return candidates


def reference_image_paths(paths: list[Path]) -> list[Path]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                sorted(child for child in path.rglob("*") if child.suffix.lower() in image_suffixes)
            )
        elif path.suffix.lower() in image_suffixes:
            expanded.append(path)
    return expanded


def image_descriptor(image: Any) -> tuple[float, ...]:
    """Summarize a reference or candidate crop for lightweight visual matching."""
    if image is None or image.size == 0:
        return ()
    resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    descriptor = []
    for y1, y2 in ((0, 42), (42, 86), (86, 128)):
        region = hsv[y1:y2, :]
        histogram = cv2.calcHist([region], [0, 1], None, [12, 4], [0, 180, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        descriptor.extend(float(value) for value in histogram)
    return tuple(descriptor)


def load_reference_descriptors(paths: list[Path]) -> tuple[list[Path], list[tuple[float, ...]]]:
    image_paths = reference_image_paths(paths)
    descriptors = []
    loaded_paths = []
    for path in image_paths:
        image = cv2.imread(str(path))
        descriptor = image_descriptor(image)
        if descriptor:
            loaded_paths.append(path)
            descriptors.append(descriptor)
    return loaded_paths, descriptors


def crop_descriptor(frame: Any, box: tuple[int, int, int, int]) -> tuple[float, ...]:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return ()
    return image_descriptor(frame[y1:y2, x1:x2])


def reference_match_score(
    frame: Any,
    box: tuple[int, int, int, int],
    reference_descriptors: list[tuple[float, ...]],
) -> float:
    if not reference_descriptors:
        return 0.0
    descriptor = crop_descriptor(frame, box)
    if not descriptor:
        return 0.0
    return max(_cosine_similarity(descriptor, reference) for reference in reference_descriptors)


def load_pose_reference_descriptors(
    estimator: Any,
    image_paths: list[Path],
    threshold: float,
    detector_model: Any = None,
    confidence_threshold: float = 0.35,
) -> list[tuple[float, ...]]:
    descriptors = []
    if estimator is None:
        return descriptors
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if detector_model is not None and hasattr(estimator, "set_boxes"):
            result = detector_model.predict(image, conf=confidence_threshold, verbose=False)[0]
            estimator.set_boxes(yolo_person_boxes(result, confidence_threshold))
        points_batch, scores_batch = estimator(image)
        for points, scores in zip(points_batch, scores_batch):
            points = np.asarray(points)
            scores = np.asarray(scores)
            box = box_for_pose(points, scores, threshold)
            if box is None:
                continue
            descriptor = pose_descriptor(TrackedBox(-1, *box), [[float(x), float(y)] for x, y in points])
            if descriptor:
                descriptors.append(descriptor)
    return descriptors


def pose_reference_match_score(
    candidate: dict[str, Any],
    reference_descriptors: list[tuple[float, ...]],
) -> float:
    if not reference_descriptors:
        return 0.0
    descriptor = pose_descriptor(
        TrackedBox(-1, *candidate["box"]),
        candidate.get("identity_keypoints", candidate["keypoints"]),
    )
    if not descriptor:
        return 0.0
    return max(_pose_similarity(descriptor, reference) for reference in reference_descriptors)


def _cosine_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if not first or len(first) != len(second):
        return 0.0
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if not first_norm or not second_norm:
        return 0.0
    similarity = sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)
    return max(0.0, min(1.0, float(similarity)))


def _pose_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if not first or len(first) != len(second):
        return 0.0
    distance = sum(abs(a - b) for a, b in zip(first, second)) / len(first)
    return max(0.0, min(1.0, 1.0 - distance))


def rtm_candidates(
    estimator: Any,
    frame: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    points_batch, scores_batch = estimator(frame)
    source_boxes = getattr(estimator, "boxes", [])
    candidates = []
    for index, (points, scores) in enumerate(zip(points_batch, scores_batch)):
        points = np.asarray(points)
        scores = np.asarray(scores)
        candidate_box = (
            tuple(int(value) for value in source_boxes[index])
            if index < len(source_boxes)
            else box_for_pose(points[:17], scores[:17], threshold)
        )
        if candidate_box is None:
            continue
        red_score, white_glove_score, white_score, black_belt_score, blue_score = color_scores(frame, points, scores, candidate_box)
        candidates.append(
            {
                "box": candidate_box,
                "center": ((candidate_box[0] + candidate_box[2]) / 2, (candidate_box[1] + candidate_box[3]) / 2),
                "keypoints": coco_body_keypoints(points),
                "identity_keypoints": [[float(x), float(y)] for x, y in points],
                "draw_keypoints": [[float(x), float(y)] for x, y in points],
                "draw_scores": [float(score) for score in scores],
                "red_glove_score": red_score,
                "white_glove_score": white_glove_score,
                "white_uniform_score": white_score,
                "black_belt_score": black_belt_score,
                "blue_glove_score": blue_score,
                "standing_score": standing_score(points, scores, candidate_box),
                "confidence": float(np.mean(scores[:17])),
            }
        )
    return candidates


def yolo_person_boxes(result: Any, confidence_threshold: float) -> list[tuple[int, int, int, int]]:
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls = int(box.cls[0]) if box.cls is not None else 0
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if cls != 0 or conf < confidence_threshold:
            continue
        boxes.append(tuple(int(value) for value in box.xyxy[0].cpu().tolist()))
    return boxes


def filter_candidates_by_boxes(
    candidates: list[dict[str, Any]],
    boxes: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    matched = []
    for candidate in candidates:
        center_x, center_y = candidate["center"]
        candidate_box = candidate["box"]
        if any(_point_inside_box(center_x, center_y, box) or _iou(candidate_box, box) >= 0.10 for box in boxes):
            matched.append(candidate)
    return matched


def _point_inside_box(x: float, y: float, box: tuple[int, int, int, int]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_roi(args.arena_roi)
    reference_paths, reference_descriptors = load_reference_descriptors(args.fighter_a_reference_images)
    face_matcher = None
    face_reference_count = 0
    if args.fighter_a_enable_face_match:
        from src.face_identity import FaceMatcher

        face_matcher = FaceMatcher(args.fighter_a_reference_images)
        face_reference_count = face_matcher.reference_count
    tracker = build_tracker(args.tracker, args.reid_weights)
    needs_yolo_detector = args.pose_backend in {"yolo", "yolo-rtmw"} or (
        args.pose_backend in {"rtmpose", "rtmw"} and args.rtm_detector_backend == "yolo"
    )
    if needs_yolo_detector:
        from ultralytics import YOLO

        pose_model = YOLO(args.model)
    else:
        pose_model = None
    if args.pose_backend != "yolo":
        if args.rtm_detector_backend == "yolo" or args.pose_backend == "yolo-rtmw":
            estimator = build_box_guided_rtm_estimator("rtmw" if args.pose_backend == "yolo-rtmw" else args.pose_backend)
        else:
            from rtmlib import Body, Wholebody

            estimator = (
                Body(mode="performance", backend="onnxruntime", device="cuda")
                if args.pose_backend == "rtmpose"
                else Wholebody(mode="performance", backend="onnxruntime", device="cuda")
            )
    else:
        estimator = None
    pose_reference_descriptors_list = (
        load_pose_reference_descriptors(
            estimator,
            reference_paths,
            args.keypoint_threshold,
            detector_model=pose_model if hasattr(estimator, "set_boxes") else None,
            confidence_threshold=args.confidence,
        )
        if estimator is not None and reference_paths
        else []
    )
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = args.input.stem
    video_path = args.output_dir / f"{stem}_annotated.mp4"
    csv_path = args.output_dir / f"{stem}_tracks.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    counter = StrikeCounter(fps=fps)
    identity = FighterIdentity(
        args.fighter_a_name,
        args.fighter_a_start,
        expect_red_gloves=True,
        expect_white_gloves=args.fighter_a_require_white_gloves,
        expect_white_uniform=True,
        expect_black_belt=args.fighter_a_black_belt,
        expect_taller=args.fighter_a_taller,
        require_red_gloves=args.fighter_a_require_red_gloves,
        require_white_gloves=args.fighter_a_require_white_gloves,
        reject_blue_gloves=args.fighter_a_reject_blue_gloves,
        require_standing=args.fighter_a_require_standing,
        expect_reference_match=bool(reference_descriptors),
        expect_pose_reference_match=bool(pose_reference_descriptors_list),
        expect_face_match=face_reference_count > 0,
        require_reference_match=args.fighter_a_require_reference_match,
        reject_face_mismatch=args.fighter_a_reject_face_mismatch and face_reference_count > 0,
        min_red_glove_score=args.fighter_a_min_red_glove_score,
        min_white_glove_score=args.fighter_a_min_white_glove_score,
        min_standing_score=args.fighter_a_min_standing_score,
        min_reference_match_score=args.fighter_a_min_reference_match_score,
        min_face_match_score=args.fighter_a_min_face_match_score,
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
        "frame", "timestamp_seconds", "track_id", "fighter_label", "confidence",
        "bbox", "keypoints", "red_glove_score", "white_glove_score", "blue_glove_score", "white_uniform_score",
        "black_belt_score", "standing_score", "reference_match_score", "pose_reference_match_score",
        "face_detected", "face_match_score", "estimated_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            if args.pose_backend == "yolo":
                result = pose_model.predict(frame, conf=args.confidence, verbose=False)[0]
                candidates = yolo_candidates(result, frame, args.keypoint_threshold)
            elif args.pose_backend == "yolo-rtmw":
                result = pose_model.predict(frame, conf=args.confidence, verbose=False)[0]
                yolo_boxes = yolo_person_boxes(result, args.confidence)
                if hasattr(estimator, "set_boxes"):
                    estimator.set_boxes(yolo_boxes)
                candidates = rtm_candidates(estimator, frame, args.keypoint_threshold)
                if not hasattr(estimator, "set_boxes"):
                    candidates = filter_candidates_by_boxes(candidates, yolo_boxes)
            elif args.pose_backend in {"rtmpose", "rtmw"} and args.rtm_detector_backend == "yolo":
                result = pose_model.predict(frame, conf=args.confidence, verbose=False)[0]
                estimator.set_boxes(yolo_person_boxes(result, args.confidence))
                candidates = rtm_candidates(estimator, frame, args.keypoint_threshold)
            else:
                candidates = rtm_candidates(estimator, frame, args.keypoint_threshold)
            candidates = [candidate for candidate in candidates if candidate_inside_roi(candidate, roi, width, height)]
            annotated = frame.copy()
            roi_box = draw_pose_candidates(annotated, candidates, roi, score_threshold=args.keypoint_threshold)
            for candidate in candidates:
                candidate["reference_match_score"] = reference_match_score(
                    frame,
                    candidate["box"],
                    reference_descriptors,
                )
                candidate["pose_reference_match_score"] = pose_reference_match_score(
                    candidate,
                    pose_reference_descriptors_list,
                )
                if face_matcher is not None:
                    face_score, face_detected = face_matcher.match_candidate(frame, candidate["box"])
                    candidate["face_match_score"] = face_score
                    candidate["face_detected"] = face_detected
                else:
                    candidate["face_match_score"] = 0.0
                    candidate["face_detected"] = False
            tracked = attach_tracks(candidates, tracker.update(detections_array(candidates), frame))
            selected_track_id = identity.observe(identity_box(candidate, frame) for candidate in tracked)
            selected = next(
                (candidate for candidate in tracked if candidate["track_id"] == selected_track_id),
                None,
            )
            if selected is not None:
                gabriel_frames += 1
                clipped = clip_box_to_roi(selected["box"], roi_box)
                if clipped is not None:
                    x1, y1, x2, y2 = clipped
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(annotated, args.fighter_a_name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for candidate in tracked:
                track_id = candidate["track_id"]
                action = counter.update(track_id, candidate["keypoints"])
                label = identity.label(track_id)
                if label == args.fighter_a_name and action:
                    named_counts[f"{action}es" if action == "punch" else "kicks"] += 1
                csv_writer.writerow(
                    {
                        "frame": frame_count,
                        "timestamp_seconds": round((frame_count - 1) / fps, 3),
                        "track_id": track_id,
                        "fighter_label": label,
                        "confidence": round(candidate["confidence"], 4),
                        "bbox": json.dumps(candidate["box"]),
                        "keypoints": json.dumps(candidate["keypoints"]),
                        "red_glove_score": round(candidate["red_glove_score"], 4),
                        "white_glove_score": round(candidate["white_glove_score"], 4),
                        "blue_glove_score": round(candidate["blue_glove_score"], 4),
                        "white_uniform_score": round(candidate["white_uniform_score"], 4),
                        "black_belt_score": round(candidate["black_belt_score"], 4),
                        "standing_score": round(candidate["standing_score"], 4),
                        "reference_match_score": round(candidate.get("reference_match_score", 0.0), 4),
                        "pose_reference_match_score": round(candidate.get("pose_reference_match_score", 0.0), 4),
                        "face_detected": candidate.get("face_detected", False),
                        "face_match_score": round(candidate.get("face_match_score", 0.0), 4),
                        "estimated_action": action,
                    }
                )
            cv2.rectangle(annotated, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (255, 255, 0), 2)
            writer.write(annotated)
    capture.release()
    writer.release()
    summary = {
        "input": str(args.input),
        "pose_backend": args.pose_backend,
        "model": args.model if needs_yolo_detector else args.pose_backend,
        "rtm_detector_backend": args.rtm_detector_backend if args.pose_backend != "yolo" else None,
        "tracker": args.tracker,
        "experiment_label": args.experiment_label,
        "fps": fps,
        "frames_processed": frame_count,
        "fighter_a_name": args.fighter_a_name,
        "fighter_a_cues": {
            "start_side": args.fighter_a_start,
            "white_uniform": True,
            "red_gloves": True,
            "white_gloves": args.fighter_a_require_white_gloves,
            "black_belt": args.fighter_a_black_belt,
            "taller_height": args.fighter_a_taller,
            "require_red_gloves": args.fighter_a_require_red_gloves,
            "require_white_gloves": args.fighter_a_require_white_gloves,
            "reject_blue_gloves": args.fighter_a_reject_blue_gloves,
            "require_standing": args.fighter_a_require_standing,
            "reference_images": [str(path) for path in reference_paths],
            "reference_image_count": len(reference_descriptors),
            "pose_reference_count": len(pose_reference_descriptors_list),
            "face_reference_count": face_reference_count,
            "require_reference_match": args.fighter_a_require_reference_match,
            "face_match": args.fighter_a_enable_face_match,
            "reject_face_mismatch": args.fighter_a_reject_face_mismatch and face_reference_count > 0,
            "min_red_glove_score": args.fighter_a_min_red_glove_score,
            "min_white_glove_score": args.fighter_a_min_white_glove_score,
            "min_standing_score": args.fighter_a_min_standing_score,
            "min_reference_match_score": args.fighter_a_min_reference_match_score,
            "min_face_match_score": args.fighter_a_min_face_match_score,
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
        "artifacts": {"annotated_video": str(video_path), "tracks_csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
