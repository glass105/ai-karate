import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from src.analyze_boxmot_video import (
    attach_tracks,
    build_parser,
    competition_fighter_score,
    detections_array,
    glove_color_settings,
    load_reference_descriptors,
    opponent_distance_body_heights,
    opponent_target_context,
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
    shape = [1, 3, 4, 4]


class FakeNhwcFp16Input:
    name = "image"
    type = "tensor(float16)"
    shape = [1, 4, 4, 3]


class FakeOnnxOutput:
    name = "simcc"
    shape = [1, 1]


class FakeDirectKeypointOutput:
    name = "keypoints"
    shape = [1, 133, 3]


class FakeFp16Session:
    def __init__(self, input_meta=None, output_meta=None) -> None:
        self.input_meta = input_meta or FakeOnnxInput()
        self.output_meta = output_meta or FakeOnnxOutput()
        self.received_dtype = None
        self.received_shape = None

    def get_inputs(self):
        return [self.input_meta]

    def get_outputs(self):
        return [self.output_meta]

    def run(self, outputs, inputs):
        input_tensor = next(iter(inputs.values()))
        self.received_dtype = input_tensor.dtype
        self.received_shape = input_tensor.shape
        return [np.zeros((1, 1), dtype=np.float32)]


class FakePoseModel:
    def __init__(self, input_meta=None, output_meta=None) -> None:
        self.session = FakeFp16Session(input_meta, output_meta)
        self.model_input_size = (4, 4)

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
                "--fighter-a-exclude-reference-images",
                "reference/exclude",
                "--fighter-a-require-reference-match",
                "--fighter-a-min-reference-match-score",
                "0.2",
                "--fighter-a-min-exclude-reference-match-score",
                "0.8",
            ]
        )

        self.assertEqual(args.fighter_a_reference_images, [Path("refs"), Path("gabriel.png")])
        self.assertEqual(args.fighter_a_exclude_reference_images, [Path("reference/exclude")])
        self.assertTrue(args.fighter_a_require_reference_match)
        self.assertEqual(args.fighter_a_min_reference_match_score, 0.2)
        self.assertEqual(args.fighter_a_min_exclude_reference_match_score, 0.8)

    def test_identity_threshold_defaults(self) -> None:
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
            ]
        )

        self.assertEqual(args.fighter_a_min_red_glove_score, 0.15)
        self.assertEqual(args.fighter_a_glove_color, "red")
        self.assertEqual(args.fighter_a_start, "left")
        self.assertEqual(args.fighter_a_min_white_glove_score, 0.02)
        self.assertEqual(args.fighter_a_min_blue_glove_score, 0.15)
        self.assertEqual(args.fighter_a_min_exclude_reference_match_score, 0.9)
        self.assertEqual(args.fighter_a_min_exclude_body_match_score, 0.97)
        self.assertEqual(args.fighter_a_min_exclude_face_match_score, 0.45)
        self.assertFalse(args.fighter_a_exclude_reference_hard_veto)
        self.assertEqual(args.fighter_a_exclude_veto_confirmation_frames, 4)
        self.assertTrue(args.fighter_a_exclude_allow_strong_face_match)
        self.assertEqual(args.fighter_a_min_face_match_score, 0.25)
        self.assertEqual(args.fighter_a_strong_face_match_score, 0.75)
        self.assertEqual(args.identity_recovery_confirmation_frames, 3)
        self.assertEqual(args.reset_to_start_side_after_missing, 10)
        self.assertEqual(args.lineup_pause_frames, 30)
        self.assertEqual(args.lineup_motion_threshold, 0.10)
        self.assertEqual(args.lineup_separation_threshold, 1.20)
        self.assertEqual(args.locked_fighter_exclude_grace_score, 0.98)
        self.assertEqual(args.locked_fighter_min_continuity_score, 0.55)
        self.assertEqual(args.locked_fighter_drop_confirmation_frames, 15)
        self.assertEqual(args.identity_switch_confirmation_frames, 12)
        self.assertEqual(args.confirmed_lock_min_frames, 30)
        self.assertEqual(args.face_match_backend, "insightface")
        self.assertEqual(args.deepface_detector_backend, "opencv")
        self.assertEqual(args.min_punch_commitment_frames, 3)
        self.assertEqual(args.min_kick_commitment_frames, 2)
        self.assertEqual(args.min_kick_foot_height_change, 20.0)
        self.assertEqual(args.min_punch_score, 1.10)
        self.assertEqual(args.max_punch_extension_ratio, 4.0)
        self.assertEqual(args.max_punch_opponent_distance_body_heights, 0.0)
        self.assertEqual(args.punch_target_gate_score, 1.50)
        self.assertEqual(args.max_untargeted_punch_distance_body_heights, 0.75)
        self.assertEqual(args.max_mixed_strike_punch_target_distance_body_heights, 0.60)
        self.assertEqual(args.max_new_lock_punch_target_distance_body_heights, 0.45)
        self.assertEqual(args.post_kick_combo_window_frames, 25)
        self.assertEqual(args.duplicate_punch_window_frames, 12)
        self.assertEqual(args.min_duplicate_punch_score, 1.20)
        self.assertEqual(args.min_punch_count_lock_frames, 30)
        self.assertEqual(args.post_kick_punch_suppression_frames, 8)
        self.assertEqual(args.min_strike_id_match_gap, 0.03)
        self.assertEqual(args.min_kick_score, 1.35)
        self.assertEqual(args.strong_kick_score, 1.75)
        self.assertEqual(args.max_kick_extension_ratio, 4.0)
        self.assertEqual(args.min_kick_foot_elevation_body_heights, 0.10)
        self.assertEqual(args.strong_kick_foot_elevation_body_heights, 0.28)
        self.assertEqual(args.max_kick_support_foot_motion_body_heights, 0.08)
        self.assertEqual(args.max_kick_opponent_distance_body_heights, 0.70)
        self.assertEqual(args.kick_opponent_memory_frames, 12)
        self.assertEqual(args.punch_cooldown_seconds, 0.35)

    def test_keeps_lunging_fighter_near_roi_bottom(self) -> None:
        candidate = {"box": (490, 331, 602, 643), "standing_score": 0.7438}

        score = competition_fighter_score(candidate, (256, 144, 1024, 648))

        self.assertEqual(score, 1.0)

    def test_normalizes_opponent_distance_by_average_fighter_height(self) -> None:
        fighter = {"track_id": 7, "box": (0, 0, 100, 200), "competition_fighter_score": 1.0}
        opponent = {"track_id": 8, "box": (150, 0, 250, 200), "competition_fighter_score": 1.0}

        distance = opponent_distance_body_heights(fighter, [fighter, opponent])

        self.assertEqual(distance, 0.75)

    def test_ignores_noncompetition_opponent_candidates(self) -> None:
        fighter = {"track_id": 7, "box": (0, 0, 100, 200), "competition_fighter_score": 1.0}
        opponent = {"track_id": 8, "box": (150, 0, 250, 200), "competition_fighter_score": 0.0}

        self.assertIsNone(opponent_distance_body_heights(fighter, [fighter, opponent]))

    def test_returns_nearest_opponent_target_context(self) -> None:
        fighter = {"track_id": 7, "box": (0, 0, 100, 200), "competition_fighter_score": 1.0}
        opponent = {"track_id": 8, "box": (150, 0, 250, 200), "competition_fighter_score": 1.0}

        context = opponent_target_context(fighter, [fighter, opponent])

        self.assertIsNotNone(context)
        distance, center, height = context
        self.assertEqual(distance, 0.75)
        self.assertEqual(center, (200.0, 100.0))
        self.assertEqual(height, 200.0)

    def test_accepts_start_and_reset_side_parameter(self) -> None:
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
                "hybridsort",
                "--fighter-a-start",
                "right",
            ]
        )

        self.assertEqual(args.fighter_a_start, "right")

    def test_accepts_selected_glove_color_and_thresholds(self) -> None:
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
                "hybridsort",
                "--fighter-a-glove-color",
                "blue",
                "--fighter-a-min-blue-glove-score",
                "0.35",
                "--fighter-a-min-white-glove-score",
                "0.25",
            ]
        )

        self.assertEqual(args.fighter_a_glove_color, "blue")
        self.assertEqual(args.fighter_a_min_blue_glove_score, 0.35)
        self.assertEqual(args.fighter_a_min_white_glove_score, 0.25)

    def test_glove_color_selects_requirement_without_implicit_rejects(self) -> None:
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
                "hybridsort",
                "--fighter-a-glove-color",
                "blue",
            ]
        )
        settings = glove_color_settings(args)

        self.assertTrue(settings["require_blue"])
        self.assertFalse(settings["require_red"])
        self.assertFalse(settings["require_white"])
        self.assertFalse(settings["reject_red"])
        self.assertFalse(settings["reject_white"])
        self.assertFalse(settings["reject_blue"])

    def test_glove_none_keeps_gloves_as_rejection_only(self) -> None:
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
                "hybridsort",
                "--fighter-a-glove-color",
                "none",
                "--fighter-a-reject-red-gloves",
                "--fighter-a-reject-blue-gloves",
            ]
        )
        settings = glove_color_settings(args)

        self.assertFalse(settings["expect_red"])
        self.assertFalse(settings["expect_white"])
        self.assertFalse(settings["expect_blue"])
        self.assertFalse(settings["require_red"])
        self.assertFalse(settings["require_white"])
        self.assertFalse(settings["require_blue"])
        self.assertTrue(settings["reject_red"])
        self.assertFalse(settings["reject_white"])
        self.assertTrue(settings["reject_blue"])

    def test_accepts_deepface_arcface_backend(self) -> None:
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
                "--fighter-a-enable-face-match",
                "--face-match-backend",
                "deepface-arcface",
                "--deepface-detector-backend",
                "retinaface",
            ]
        )

        self.assertTrue(args.fighter_a_enable_face_match)
        self.assertEqual(args.face_match_backend, "deepface-arcface")
        self.assertEqual(args.deepface_detector_backend, "retinaface")

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
        self.assertEqual(pose_model.session.received_shape, (1, 3, 4, 4))

    def test_patches_nhwc_fp16_rtm_session_layout(self) -> None:
        pose_model = FakePoseModel(FakeNhwcFp16Input())
        image = np.zeros((4, 4, 3), dtype=np.float32)

        patch_rtm_fp16_input(pose_model)
        pose_model.inference(image)

        self.assertEqual(pose_model.session.received_dtype, np.float16)
        self.assertEqual(pose_model.session.received_shape, (1, 4, 4, 3))

    def test_patches_direct_keypoint_rtm_output(self) -> None:
        pose_model = FakePoseModel(FakeNhwcFp16Input(), FakeDirectKeypointOutput())
        output = np.zeros((1, 133, 3), dtype=np.float16)
        output[0, 0] = [2.0, 2.0, 0.75]

        patch_rtm_fp16_input(pose_model)
        keypoints, scores = pose_model.postprocess([output], center=np.array([10.0, 20.0]), scale=np.array([8.0, 16.0]))

        self.assertEqual(keypoints.shape, (1, 133, 2))
        self.assertEqual(scores.shape, (1, 133))
        self.assertTrue(np.allclose(keypoints[0, 0], [10.0, 20.0]))
        self.assertAlmostEqual(float(scores[0, 0]), 0.75)


if __name__ == "__main__":
    unittest.main()
