import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from src.analyze_boxmot_video import (
    attach_tracks,
    build_parser,
    detections_array,
    load_reference_descriptors,
    patch_rtm_fp16_input,
    reference_match_score,
    rtm_candidates,
)


class FakeTracks:
    size = 16
    det_ind = np.asarray([1, 0])
    id = np.asarray([20, 10])


class EmptyTracks:
    size = 0


class FakeBoxGuidedEstimator:
    boxes = [(10, 20, 110, 220)]

    def __call__(self, frame):
        points = np.zeros((1, 133, 2), dtype=np.float32)
        scores = np.ones((1, 133), dtype=np.float32)
        points[0, :17, 0] = np.linspace(20, 100, 17)
        points[0, :17, 1] = np.linspace(30, 210, 17)
        points[0, 120] = [500, 500]
        return points, scores


class FakeOnnxInput:
    name = "images"
    type = "tensor(float16)"


class FakeOnnxOutput:
    name = "simcc"


class FakeFp16Session:
    def __init__(self) -> None:
        self.received_dtype = None

    def get_inputs(self):
        return [FakeOnnxInput()]

    def get_outputs(self):
        return [FakeOnnxOutput()]

    def run(self, outputs, inputs):
        self.received_dtype = inputs["images"].dtype
        return [np.zeros((1, 1), dtype=np.float32)]


class FakePoseModel:
    def __init__(self) -> None:
        self.session = FakeFp16Session()

    def inference(self, img):
        return []


class AnalyzeBoxmotVideoTests(unittest.TestCase):
    def test_builds_boxmot_detection_layout(self) -> None:
        detections = detections_array([{"box": (1, 2, 3, 4), "confidence": 0.8}])

        self.assertEqual(detections.tolist(), [[1.0, 2.0, 3.0, 4.0, 0.800000011920929, 0.0]])

    def test_maps_track_results_back_to_pose_candidates(self) -> None:
        candidates = [{"name": "first"}, {"name": "second"}]

        attached = attach_tracks(candidates, FakeTracks())

        self.assertEqual(attached, [{"name": "second", "track_id": 20}, {"name": "first", "track_id": 10}])

    def test_ignores_empty_track_results(self) -> None:
        self.assertEqual(attach_tracks([{"name": "first"}], EmptyTracks()), [])

    def test_accepts_requested_boxmot_trackers(self) -> None:
        parser = build_parser()
        for tracker in ("ocsort", "deepocsort", "hybridsort", "strongsort", "boosttrack"):
            args = parser.parse_args(
                [
                    "--input",
                    "input.mp4",
                    "--output-dir",
                    "out",
                    "--pose-backend",
                    "rtmw",
                    "--tracker",
                    tracker,
                ]
            )
            self.assertEqual(args.tracker, tracker)

    def test_accepts_yolo_rtmw_backend(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--input",
                "input.mp4",
                "--output-dir",
                "out",
                "--pose-backend",
                "yolo-rtmw",
                "--tracker",
                "boosttrack",
            ]
        )

        self.assertEqual(args.pose_backend, "yolo-rtmw")

    def test_accepts_reference_images(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "--input",
                "input.mp4",
                "--output-dir",
                "out",
                "--pose-backend",
                "rtmw",
                "--tracker",
                "boosttrack",
                "--fighter-a-reference-images",
                "refs",
                "gabriel.png",
                "--fighter-a-require-reference-match",
                "--fighter-a-min-reference-match-score",
                "0.2",
            ]
        )

        self.assertEqual(args.fighter_a_reference_images, [Path("refs"), Path("gabriel.png")])
        self.assertTrue(args.fighter_a_require_reference_match)
        self.assertEqual(args.fighter_a_min_reference_match_score, 0.2)

    def test_reference_match_scores_similar_crop_higher(self) -> None:
        with TemporaryDirectory() as temp:
            reference_path = Path(temp) / "gabriel.png"
            red_image = np.zeros((128, 64, 3), dtype=np.uint8)
            red_image[:, :] = (0, 0, 255)
            blue_image = np.zeros((128, 64, 3), dtype=np.uint8)
            blue_image[:, :] = (255, 0, 0)
            cv2.imwrite(str(reference_path), red_image)
            _, descriptors = load_reference_descriptors([reference_path])

            red_score = reference_match_score(red_image, (0, 0, 64, 128), descriptors)
            blue_score = reference_match_score(blue_image, (0, 0, 64, 128), descriptors)

            self.assertGreater(red_score, 0.99)
            self.assertLess(blue_score, 0.1)

    def test_box_guided_rtm_candidates_keep_detector_box(self) -> None:
        frame = np.full((600, 600, 3), 255, dtype=np.uint8)

        candidates = rtm_candidates(FakeBoxGuidedEstimator(), frame, 0.35)

        self.assertEqual(candidates[0]["box"], (10, 20, 110, 220))

    def test_patches_fp16_rtm_session_input_dtype(self) -> None:
        pose_model = FakePoseModel()
        image = np.zeros((4, 4, 3), dtype=np.float32)

        patch_rtm_fp16_input(pose_model)
        pose_model.inference(image)

        self.assertEqual(pose_model.session.received_dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
