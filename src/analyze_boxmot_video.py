"""Analyze karate video with YOLO or RTM poses and BoxMOT OC-SORT trackers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import MethodType
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
    parser.add_argument(
        "--fighter-a-require-competition-fighter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidates outside the configured competition ROI or seated/crouched on the ROI edge.",
    )
    parser.add_argument(
        "--fighter-a-reference-images",
        nargs="*",
        default=[],
        type=Path,
        help="Gabriel reference image files or directories used as an appearance cue.",
    )
    parser.add_argument(
        "--fighter-a-exclude-reference-images",
        nargs="*",
        default=[Path("reference/exclude")],
        type=Path,
        help="Reference image files or directories that should never be labeled as Gabriel.",
    )
    parser.add_argument("--fighter-a-require-reference-match", action="store_true")
    parser.add_argument("--fighter-a-min-reference-match-score", default=0.05, type=float)
    parser.add_argument("--fighter-a-min-exclude-reference-match-score", default=0.90, type=float)
    parser.add_argument("--fighter-a-min-exclude-body-match-score", default=0.97, type=float)
    parser.add_argument("--fighter-a-min-exclude-face-match-score", default=0.45, type=float)
    parser.add_argument(
        "--fighter-a-exclude-reference-hard-veto",
        action="store_true",
        help="Treat exclude reference matches as hard negatives that red/pose/continuity cannot rescue.",
    )
    parser.add_argument(
        "--fighter-a-exclude-veto-confirmation-frames",
        default=4,
        type=int,
        help="Consecutive exclude-veto frames before dropping an already locked Gabriel track.",
    )
    parser.add_argument(
        "--fighter-a-exclude-allow-strong-face-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a strong Gabriel face match to override a body-only exclude match.",
    )
    parser.add_argument("--fighter-a-enable-face-match", action="store_true")
    parser.add_argument(
        "--face-match-backend",
        choices=("insightface", "deepface-arcface"),
        default="insightface",
        help="Face embedding backend used for ROI-filtered fighter candidate identity checks.",
    )
    parser.add_argument(
        "--deepface-detector-backend",
        default="opencv",
        help="DeepFace detector backend used when --face-match-backend deepface-arcface is selected.",
    )
    parser.add_argument("--fighter-a-reject-face-mismatch", action="store_true")
    parser.add_argument("--fighter-a-min-face-match-score", default=0.25, type=float)
    parser.add_argument("--fighter-a-strong-face-match-score", default=0.75, type=float)
    parser.add_argument("--fighter-a-min-red-glove-score", default=0.15, type=float)
    parser.add_argument("--fighter-a-min-white-glove-score", default=0.02, type=float)
    parser.add_argument("--fighter-a-min-blue-glove-score", default=0.15, type=float)
    parser.add_argument("--fighter-a-min-standing-score", default=0.45, type=float)
    parser.add_argument("--experiment-label")
    parser.add_argument("--reset-to-start-side-after-missing", default=10, type=int)
    parser.add_argument("--identity-recovery-confirmation-frames", default=3, type=int)
    parser.add_argument("--fighter-candidate-limit", default=4, type=int)
    parser.add_argument("--lineup-pause-frames", default=30, type=int)
    parser.add_argument("--lineup-motion-threshold", default=0.10, type=float)
    parser.add_argument("--lineup-separation-threshold", default=1.20, type=float)
    parser.add_argument("--locked-fighter-exclude-grace-score", default=0.98, type=float)
    parser.add_argument("--locked-fighter-min-continuity-score", default=0.55, type=float)
    parser.add_argument("--locked-fighter-drop-confirmation-frames", default=15, type=int)
    parser.add_argument("--identity-switch-confirmation-frames", default=12, type=int)
    parser.add_argument("--confirmed-lock-min-frames", default=30, type=int)
    parser.add_argument("--arena-roi", default="0.2,0.1,0.8,0.9")
    parser.add_argument("--confidence", default=0.35, type=float)
    parser.add_argument("--keypoint-threshold", default=0.35, type=float)
    parser.add_argument("--max-seconds", default=0.0, type=float, help="Stop after this many seconds; 0 means full video.")
    parser.add_argument("--strike-history-frames", default=6, type=int)
    parser.add_argument("--min-punch-endpoint-motion", default=25.0, type=float)
    parser.add_argument("--min-kick-endpoint-motion", default=25.0, type=float)
    parser.add_argument(
        "--min-kick-foot-motion",
        default=0.0,
        type=float,
        help="Minimum ankle/foot travel for kick snap scoring; 0 reuses --min-kick-endpoint-motion.",
    )
    parser.add_argument("--min-punch-extension-delta", default=12.0, type=float)
    parser.add_argument("--min-kick-extension-delta", default=12.0, type=float)
    parser.add_argument("--min-punch-extension-ratio", default=1.20, type=float)
    parser.add_argument("--min-kick-extension-ratio", default=1.20, type=float)
    parser.add_argument(
        "--min-punch-commitment-frames",
        default=5,
        type=int,
        help="Consecutive ending frames the arm must stay extended before a punch is counted as real.",
    )
    parser.add_argument(
        "--min-punch-commitment-ratio",
        default=0.0,
        type=float,
        help="Extension ratio for committed punch frames; 0 reuses --min-punch-extension-ratio.",
    )
    parser.add_argument("--min-kick-foot-height-change", default=0.0, type=float)
    parser.add_argument("--strike-min-score", default=1.0, type=float)
    parser.add_argument("--strike-rearm-score", default=0.60, type=float)
    parser.add_argument("--min-strike-score-gap", default=0.10, type=float)
    parser.add_argument("--punch-cooldown-seconds", default=0.35, type=float)
    parser.add_argument("--kick-cooldown-seconds", default=0.50, type=float)
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
    patch_rtm_fp16_input(pose_model)
    return BoxGuidedRtmPoseEstimator(pose_model, keypoint_count)


def patch_rtm_fp16_input(pose_model: Any) -> None:
    """Allow RTMLib to run alternate RTMW ONNX exports used by some mirrors."""
    session = getattr(pose_model, "session", None)
    if session is None or not hasattr(session, "get_inputs"):
        return
    inputs = session.get_inputs()
    if not inputs:
        return
    input_meta = inputs[0]
    input_type = getattr(input_meta, "type", "")
    input_shape = list(getattr(input_meta, "shape", []) or [])
    expects_fp16 = input_type == "tensor(float16)"
    expects_nhwc = len(input_shape) == 4 and input_shape[-1] == 3
    if not expects_fp16 and not expects_nhwc:
        return

    def inference(self: Any, img: Any) -> list[Any]:
        dtype = np.float16 if expects_fp16 else np.float32
        if expects_nhwc:
            input_tensor = np.ascontiguousarray(img[None, :, :, :], dtype=dtype)
        else:
            img_chw = img.transpose(2, 0, 1)
            input_tensor = np.ascontiguousarray(img_chw[None, :, :, :], dtype=dtype)
        sess_input = {self.session.get_inputs()[0].name: input_tensor}
        sess_output = [out.name for out in self.session.get_outputs()]
        return self.session.run(sess_output, sess_input)

    pose_model.inference = MethodType(inference, pose_model)
    outputs = session.get_outputs()
    if len(outputs) == 1 and list(getattr(outputs[0], "shape", []) or [])[-1:] == [3]:

        def postprocess(self: Any, outputs: list[Any], center: Any, scale: Any) -> tuple[Any, Any]:
            direct = np.asarray(outputs[0])
            if direct.ndim == 3:
                direct = direct[0]
            keypoints = direct[:, :2].astype(np.float32)
            scores = direct[:, 2].astype(np.float32)
            keypoints = keypoints / np.asarray(self.model_input_size, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
            keypoints = keypoints + np.asarray(center, dtype=np.float32) - np.asarray(scale, dtype=np.float32) / 2
            return keypoints[None, :, :], scores[None, :]

        pose_model.postprocess = MethodType(postprocess, pose_model)


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


def draw_dashed_rectangle(
    frame: Any,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
) -> None:
    x1, y1, x2, y2 = box
    for start in range(x1, x2, dash * 2):
        cv2.line(frame, (start, y1), (min(start + dash, x2), y1), color, thickness)
        cv2.line(frame, (start, y2), (min(start + dash, x2), y2), color, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, start), (x1, min(start + dash, y2)), color, thickness)
        cv2.line(frame, (x2, start), (x2, min(start + dash, y2)), color, thickness)


def competition_fighter_score(
    candidate: dict[str, Any],
    roi_box: tuple[int, int, int, int],
) -> float:
    """Reject obvious mat-edge/background people before identity assignment."""
    x1, _, x2, y2 = candidate["box"]
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
    roi_width = max(1, roi_x2 - roi_x1)
    roi_height = max(1, roi_y2 - roi_y1)
    if y2 > roi_y2:
        return 0.0
    if x2 < roi_x1 or x1 > roi_x2:
        return 0.0
    overlap_width = max(0, min(x2, roi_x2) - max(x1, roi_x1))
    if overlap_width / max(1, x2 - x1) < 0.60:
        return 0.0
    bottom_fraction = (y2 - roi_y1) / roi_height
    center_x = (x1 + x2) / 2
    side_margin = min(center_x - roi_x1, roi_x2 - center_x) / roi_width
    standing = candidate.get("standing_score", 1.0)
    if bottom_fraction >= 0.96 and standing < 0.75:
        return 0.0
    if bottom_fraction >= 0.92 and side_margin <= 0.10 and standing < 0.80:
        return 0.0
    return 1.0


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi = parse_roi(args.arena_roi)
    reference_paths, reference_descriptors = load_reference_descriptors(args.fighter_a_reference_images)
    exclude_reference_paths, exclude_reference_descriptors = load_reference_descriptors(
        args.fighter_a_exclude_reference_images
    )
    face_matcher = None
    exclude_face_matcher = None
    face_reference_count = 0
    exclude_face_reference_count = 0
    if args.fighter_a_enable_face_match:
        from src.face_identity import FaceMatcher

        face_matcher = FaceMatcher(
            args.fighter_a_reference_images,
            backend=args.face_match_backend,
            detector_backend=args.deepface_detector_backend,
        )
        face_reference_count = face_matcher.reference_count
        exclude_face_matcher = FaceMatcher(
            args.fighter_a_exclude_reference_images,
            backend=args.face_match_backend,
            detector_backend=args.deepface_detector_backend,
        )
        exclude_face_reference_count = exclude_face_matcher.reference_count
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
    max_frames = int(args.max_seconds * fps) if args.max_seconds and args.max_seconds > 0 else 0
    counter = StrikeCounter(
        fps=fps,
        history_frames=args.strike_history_frames,
        punch_cooldown_seconds=args.punch_cooldown_seconds,
        kick_cooldown_seconds=args.kick_cooldown_seconds,
        min_punch_endpoint_motion=args.min_punch_endpoint_motion,
        min_kick_endpoint_motion=args.min_kick_endpoint_motion,
        min_kick_foot_motion=args.min_kick_foot_motion,
        min_punch_extension_delta=args.min_punch_extension_delta,
        min_kick_extension_delta=args.min_kick_extension_delta,
        min_punch_extension_ratio=args.min_punch_extension_ratio,
        min_kick_extension_ratio=args.min_kick_extension_ratio,
        min_punch_commitment_frames=args.min_punch_commitment_frames,
        min_punch_commitment_ratio=args.min_punch_commitment_ratio,
        min_kick_foot_height_change=args.min_kick_foot_height_change,
        min_strike_score=args.strike_min_score,
        strike_rearm_score=args.strike_rearm_score,
        min_strike_score_gap=args.min_strike_score_gap,
    )
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
        require_competition_fighter=args.fighter_a_require_competition_fighter,
        expect_reference_match=bool(reference_descriptors),
        expect_pose_reference_match=bool(pose_reference_descriptors_list),
        expect_face_match=face_reference_count > 0,
        require_reference_match=args.fighter_a_require_reference_match,
        reject_face_mismatch=args.fighter_a_reject_face_mismatch and face_reference_count > 0,
        reject_exclude_reference_match=bool(exclude_reference_descriptors) or exclude_face_reference_count > 0,
        exclude_reference_hard_veto=args.fighter_a_exclude_reference_hard_veto,
        exclude_veto_allow_strong_face_match=args.fighter_a_exclude_allow_strong_face_match,
        min_red_glove_score=args.fighter_a_min_red_glove_score,
        min_white_glove_score=args.fighter_a_min_white_glove_score,
        min_blue_glove_score=args.fighter_a_min_blue_glove_score,
        min_standing_score=args.fighter_a_min_standing_score,
        min_reference_match_score=args.fighter_a_min_reference_match_score,
        min_face_match_score=args.fighter_a_min_face_match_score,
        strong_face_match_score=args.fighter_a_strong_face_match_score,
        min_exclude_reference_match_score=args.fighter_a_min_exclude_reference_match_score,
        min_exclude_body_match_score=args.fighter_a_min_exclude_body_match_score,
        min_exclude_face_match_score=args.fighter_a_min_exclude_face_match_score,
        exclude_veto_confirmation_frames=args.fighter_a_exclude_veto_confirmation_frames,
        reset_after_missing_frames=args.reset_to_start_side_after_missing,
        recovery_confirmation_frames=args.identity_recovery_confirmation_frames,
        fighter_candidate_limit=args.fighter_candidate_limit,
        lineup_pause_frames=args.lineup_pause_frames,
        lineup_motion_threshold=args.lineup_motion_threshold,
        lineup_separation_threshold=args.lineup_separation_threshold,
        locked_fighter_exclude_grace_score=args.locked_fighter_exclude_grace_score,
        locked_fighter_min_continuity_score=args.locked_fighter_min_continuity_score,
        locked_fighter_drop_confirmation_frames=args.locked_fighter_drop_confirmation_frames,
        identity_switch_confirmation_frames=args.identity_switch_confirmation_frames,
        confirmed_lock_min_frames=args.confirmed_lock_min_frames,
    )
    frame_count = 0
    gabriel_frames = 0
    named_counts = {"punches": 0, "fake_punches": 0, "kicks": 0}
    fieldnames = [
        "frame", "timestamp_seconds", "track_id", "fighter_label", "confidence",
        "bbox", "keypoints", "red_glove_score", "white_glove_score", "blue_glove_score", "white_uniform_score",
        "black_belt_score", "standing_score", "reference_match_score", "pose_reference_match_score",
        "competition_fighter_score",
        "exclude_reference_match_score", "exclude_body_match_score", "exclude_face_detected",
        "exclude_face_match_score", "face_detected", "face_match_score", "gabriel_candidate_score", "id_red_wrist_score",
        "id_white_glove_score", "id_blue_glove_score", "id_pose_reference_score", "id_face_detected", "id_face_match_score",
        "id_exclude_reference_score", "id_exclude_body_score", "id_exclude_face_detected",
        "id_exclude_face_score", "id_competition_fighter_score", "id_continuity_score", "id_reset_side_score", "id_match_gap",
        "id_confirmed_not_top", "id_hard_reject_active", "id_rejection_reason", "id_visual_state",
        "strike_punch_score", "strike_kick_score", "strike_punch_endpoint_motion", "strike_kick_endpoint_motion",
        "strike_punch_extension_delta", "strike_kick_extension_delta", "strike_punch_extension_ratio",
        "strike_kick_extension_ratio", "strike_punch_commitment_frames", "strike_punch_cooldown", "strike_kick_cooldown",
        "strike_punch_armed", "strike_kick_armed", "strike_candidate_type", "strike_confirmed",
        "strike_rejection_reason",
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
            if max_frames and frame_count > max_frames:
                frame_count -= 1
                break
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
                candidate["competition_fighter_score"] = competition_fighter_score(candidate, roi_box)
                candidate["reference_match_score"] = reference_match_score(
                    frame,
                    candidate["box"],
                    reference_descriptors,
                )
                candidate["pose_reference_match_score"] = pose_reference_match_score(
                    candidate,
                    pose_reference_descriptors_list,
                )
                candidate["exclude_body_match_score"] = reference_match_score(
                    frame,
                    candidate["box"],
                    exclude_reference_descriptors,
                )
                candidate["exclude_reference_match_score"] = candidate["exclude_body_match_score"]
                candidate["exclude_face_match_score"] = 0.0
                candidate["exclude_face_detected"] = False
                if face_matcher is not None:
                    face_score, face_detected = face_matcher.match_candidate(frame, candidate["box"])
                    candidate["face_match_score"] = face_score
                    candidate["face_detected"] = face_detected
                    if exclude_face_matcher is not None:
                        exclude_face_score, exclude_face_detected = exclude_face_matcher.match_candidate(
                            frame,
                            candidate["box"],
                        )
                        candidate["exclude_face_match_score"] = exclude_face_score
                        candidate["exclude_face_detected"] = exclude_face_detected
                        candidate["exclude_reference_match_score"] = max(
                            candidate["exclude_body_match_score"],
                            exclude_face_score,
                        )
                else:
                    candidate["face_match_score"] = 0.0
                    candidate["face_detected"] = False
                    candidate["exclude_face_match_score"] = 0.0
                    candidate["exclude_face_detected"] = False
            tracked = attach_tracks(candidates, tracker.update(detections_array(candidates), frame))
            tracked_identity_boxes = [identity_box(candidate, frame) for candidate in tracked]
            selected_track_id = identity.observe(tracked_identity_boxes)
            identity_scores = identity.identity_scores
            if identity_scores:
                debug_parts = []
                for score in sorted(identity_scores.values(), key=lambda item: item.gabriel_candidate_score, reverse=True)[:4]:
                    debug_parts.append(
                        "track={track} score={score:.3f} red={red:.3f} white={white:.3f} blue={blue:.3f} pose={pose:.3f} "
                        "face_detected={face_detected} face={face:.3f} "
                        "excl_body={exclude_body:.3f} excl_face={exclude_face:.3f} excl_final={exclude:.3f} "
                        "comp={competition:.3f} "
                        "cont={cont:.3f} reset={reset:.3f} "
                        "gap={gap:.3f} confirmed_not_top={confirmed_not_top} hard_reject={hard_reject} "
                        "state={state} reject={reject}".format(
                            track=score.track_id,
                            score=score.gabriel_candidate_score,
                            red=score.red_wrist_score,
                            white=score.white_glove_score,
                            blue=score.blue_glove_score,
                            pose=score.pose_reference_score,
                            face_detected=score.face_detected,
                            face=score.face_match_score,
                            exclude=score.exclude_reference_score,
                            exclude_body=score.exclude_body_score,
                            exclude_face=score.exclude_face_score,
                            competition=score.competition_fighter_score,
                            cont=score.continuity_score,
                            reset=score.reset_side_score,
                            gap=score.match_gap,
                            confirmed_not_top=score.confirmed_not_top,
                            hard_reject=score.hard_reject_active,
                            state="confirmed" if score.confirmed else "tentative" if score.tentative else "candidate",
                            reject=score.rejection_reason or "none",
                        )
                    )
                print(f"ID confidence scores frame={frame_count}: " + " | ".join(debug_parts), flush=True)
            selected = next(
                (candidate for candidate in tracked if candidate["track_id"] == selected_track_id),
                None,
            )
            for candidate in tracked:
                score = identity_scores.get(candidate["track_id"])
                if score is None:
                    continue
                clipped = clip_box_to_roi(candidate["box"], roi_box)
                if clipped is None:
                    continue
                x1, y1, _, y2 = clipped
                rejection = score.rejection_reason or "ok"
                overlay_text = (
                    f"id={candidate['track_id']} gab={score.gabriel_candidate_score:.2f} "
                    f"red={score.red_wrist_score:.2f} white={score.white_glove_score:.2f} "
                    f"blue={score.blue_glove_score:.2f} xb={score.exclude_body_score:.2f} "
                    f"xf={score.exclude_face_score:.2f} x={score.exclude_reference_score:.2f} "
                    f"comp={score.competition_fighter_score:.2f} {rejection}"
                )
                cv2.putText(
                    annotated,
                    overlay_text,
                    (x1, min(max(20, y1 + 16), max(20, y2 - 4))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
            if selected is not None:
                gabriel_frames += 1
                clipped = clip_box_to_roi(selected["box"], roi_box)
                if clipped is not None:
                    x1, y1, x2, y2 = clipped
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(annotated, args.fighter_a_name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif identity.visual_box is not None:
                clipped = clip_box_to_roi(
                    (
                        int(identity.visual_box.x1),
                        int(identity.visual_box.y1),
                        int(identity.visual_box.x2),
                        int(identity.visual_box.y2),
                    ),
                    roi_box,
                )
                if clipped is not None:
                    x1, y1, x2, y2 = clipped
                    draw_dashed_rectangle(annotated, clipped, (0, 255, 255), 2)
                    cv2.putText(annotated, f"{args.fighter_a_name}?", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for candidate in tracked:
                track_id = candidate["track_id"]
                label = identity.label(track_id)
                action = counter.update(
                    track_id,
                    candidate["keypoints"],
                    count_enabled=label == args.fighter_a_name,
                )
                strike_debug = counter.last_debug[track_id]
                score = identity_scores.get(track_id)
                visual_state = ""
                if score is not None:
                    visual_state = "confirmed" if score.confirmed else "tentative" if score.tentative else "candidate"
                if label == args.fighter_a_name and action:
                    action_key = {"punch": "punches", "fake_punch": "fake_punches", "kick": "kicks"}.get(action)
                    if action_key:
                        named_counts[action_key] += 1
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
                        "competition_fighter_score": round(candidate.get("competition_fighter_score", 1.0), 4),
                        "exclude_reference_match_score": round(candidate.get("exclude_reference_match_score", 0.0), 4),
                        "exclude_body_match_score": round(candidate.get("exclude_body_match_score", 0.0), 4),
                        "exclude_face_detected": candidate.get("exclude_face_detected", False),
                        "exclude_face_match_score": round(candidate.get("exclude_face_match_score", 0.0), 4),
                        "face_detected": candidate.get("face_detected", False),
                        "face_match_score": round(candidate.get("face_match_score", 0.0), 4),
                        "gabriel_candidate_score": round(score.gabriel_candidate_score, 4) if score else 0.0,
                        "id_red_wrist_score": round(score.red_wrist_score, 4) if score else 0.0,
                        "id_white_glove_score": round(score.white_glove_score, 4) if score else 0.0,
                        "id_blue_glove_score": round(score.blue_glove_score, 4) if score else 0.0,
                        "id_pose_reference_score": round(score.pose_reference_score, 4) if score else 0.0,
                        "id_face_detected": score.face_detected if score else False,
                        "id_face_match_score": round(score.face_match_score, 4) if score else 0.0,
                        "id_exclude_reference_score": round(score.exclude_reference_score, 4) if score else 0.0,
                        "id_exclude_body_score": round(score.exclude_body_score, 4) if score else 0.0,
                        "id_exclude_face_detected": score.exclude_face_detected if score else False,
                        "id_exclude_face_score": round(score.exclude_face_score, 4) if score else 0.0,
                        "id_competition_fighter_score": round(score.competition_fighter_score, 4) if score else 0.0,
                        "id_continuity_score": round(score.continuity_score, 4) if score else 0.0,
                        "id_reset_side_score": round(score.reset_side_score, 4) if score else 0.0,
                        "id_match_gap": round(score.match_gap, 4) if score else 0.0,
                        "id_confirmed_not_top": score.confirmed_not_top if score else False,
                        "id_hard_reject_active": score.hard_reject_active if score else False,
                        "id_rejection_reason": score.rejection_reason if score else "",
                        "id_visual_state": visual_state,
                        **strike_debug.as_csv_fields(),
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
        "max_seconds": args.max_seconds,
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
            "require_competition_fighter": args.fighter_a_require_competition_fighter,
            "reference_images": [str(path) for path in reference_paths],
            "reference_image_count": len(reference_descriptors),
            "exclude_reference_images": [str(path) for path in exclude_reference_paths],
            "exclude_reference_image_count": len(exclude_reference_descriptors),
            "pose_reference_count": len(pose_reference_descriptors_list),
            "face_reference_count": face_reference_count,
            "exclude_face_reference_count": exclude_face_reference_count,
            "require_reference_match": args.fighter_a_require_reference_match,
            "face_match": args.fighter_a_enable_face_match,
            "face_match_backend": args.face_match_backend,
            "deepface_detector_backend": args.deepface_detector_backend,
            "reject_face_mismatch": args.fighter_a_reject_face_mismatch and face_reference_count > 0,
            "min_red_glove_score": args.fighter_a_min_red_glove_score,
            "min_white_glove_score": args.fighter_a_min_white_glove_score,
            "min_blue_glove_score": args.fighter_a_min_blue_glove_score,
            "min_standing_score": args.fighter_a_min_standing_score,
            "min_reference_match_score": args.fighter_a_min_reference_match_score,
            "min_face_match_score": args.fighter_a_min_face_match_score,
            "strong_face_match_score": args.fighter_a_strong_face_match_score,
            "min_exclude_reference_match_score": args.fighter_a_min_exclude_reference_match_score,
            "min_exclude_body_match_score": args.fighter_a_min_exclude_body_match_score,
            "min_exclude_face_match_score": args.fighter_a_min_exclude_face_match_score,
            "exclude_reference_hard_veto": args.fighter_a_exclude_reference_hard_veto,
            "exclude_veto_confirmation_frames": args.fighter_a_exclude_veto_confirmation_frames,
            "exclude_allow_strong_face_match": args.fighter_a_exclude_allow_strong_face_match,
            "reset_to_start_side_after_missing_frames": args.reset_to_start_side_after_missing,
            "recovery_confirmation_frames": args.identity_recovery_confirmation_frames,
            "fighter_candidate_limit": args.fighter_candidate_limit,
            "lineup_pause_frames": args.lineup_pause_frames,
            "lineup_motion_threshold": args.lineup_motion_threshold,
            "lineup_separation_threshold": args.lineup_separation_threshold,
            "locked_fighter_exclude_grace_score": args.locked_fighter_exclude_grace_score,
            "locked_fighter_min_continuity_score": args.locked_fighter_min_continuity_score,
            "locked_fighter_drop_confirmation_frames": args.locked_fighter_drop_confirmation_frames,
            "identity_switch_confirmation_frames": args.identity_switch_confirmation_frames,
            "confirmed_lock_min_frames": args.confirmed_lock_min_frames,
        },
        "arena_roi": roi,
        "gabriel_frames": gabriel_frames,
        "fighter_a_track_id": identity.fighter_a_track_id,
        "fighter_a_track_ids": sorted(identity.fighter_a_track_ids),
        "identity_recoveries": identity.recovery_count,
        "identity_resets": identity.reset_count,
        "counts_by_track_id": counter.counts,
        "named_fighter_candidate_counts": named_counts,
        "strike_counter": {
            "history_frames": args.strike_history_frames,
            "min_punch_endpoint_motion": args.min_punch_endpoint_motion,
            "min_kick_endpoint_motion": args.min_kick_endpoint_motion,
            "min_kick_foot_motion": args.min_kick_foot_motion,
            "min_punch_extension_delta": args.min_punch_extension_delta,
            "min_kick_extension_delta": args.min_kick_extension_delta,
            "min_punch_extension_ratio": args.min_punch_extension_ratio,
            "min_kick_extension_ratio": args.min_kick_extension_ratio,
            "min_punch_commitment_frames": args.min_punch_commitment_frames,
            "min_punch_commitment_ratio": (
                args.min_punch_commitment_ratio
                if args.min_punch_commitment_ratio > 0
                else args.min_punch_extension_ratio
            ),
            "min_kick_foot_height_change": args.min_kick_foot_height_change,
            "strike_min_score": args.strike_min_score,
            "strike_rearm_score": args.strike_rearm_score,
            "min_strike_score_gap": args.min_strike_score_gap,
            "punch_cooldown_seconds": args.punch_cooldown_seconds,
            "kick_cooldown_seconds": args.kick_cooldown_seconds,
            "counts_only_selected_fighter": True,
        },
        "artifacts": {"annotated_video": str(video_path), "tracks_csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
