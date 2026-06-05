import unittest

import numpy as np

from src.roi_drawing import candidate_inside_roi, clip_box_to_roi, draw_pose_candidates


class RoiDrawingTests(unittest.TestCase):
    def test_candidate_center_must_be_inside_roi(self) -> None:
        roi = (0.1, 0.2, 0.9, 1.0)

        self.assertTrue(candidate_inside_roi({"center": (50, 50)}, roi, 100, 100))
        self.assertFalse(candidate_inside_roi({"center": (5, 50)}, roi, 100, 100))
        self.assertFalse(candidate_inside_roi({"center": (50, 10)}, roi, 100, 100))

    def test_clips_selected_box_to_roi(self) -> None:
        self.assertEqual(clip_box_to_roi((0, 0, 80, 80), (10, 20, 90, 100)), (10, 20, 80, 80))
        self.assertIsNone(clip_box_to_roi((0, 0, 5, 10), (10, 20, 90, 100)))

    def test_draws_only_points_and_lines_inside_roi(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        candidate = {
            "keypoints": [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [20, 30],
                [80, 30],
                [20, 50],
                [95, 10],
                [20, 70],
                [95, 10],
                [20, 80],
                [80, 80],
                [20, 90],
                [80, 90],
                [20, 95],
                [80, 95],
            ]
        }

        roi_box = draw_pose_candidates(frame, [candidate], (0.1, 0.2, 0.9, 1.0))

        self.assertEqual(roi_box, (10, 20, 90, 100))
        self.assertGreater(frame[30:96, 20:81].sum(), 0)
        self.assertEqual(frame[:20].sum(), 0)
        self.assertEqual(frame[:, :10].sum(), 0)
        self.assertEqual(frame[:, 91:].sum(), 0)

    def test_draws_extra_landmarks_only_inside_roi(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        candidate = {
            "keypoints": [[50, 50] for _ in range(17)],
            "draw_keypoints": [[50, 50], [95, 50], [50, 10]],
            "draw_scores": [0.9, 0.9, 0.9],
        }

        draw_pose_candidates(frame, [candidate], (0.1, 0.2, 0.9, 1.0), score_threshold=0.35)

        self.assertGreater(frame[48:53, 48:53].sum(), 0)
        self.assertEqual(frame[:, 92:].sum(), 0)
        self.assertEqual(frame[:20].sum(), 0)

    def test_skips_low_confidence_extra_landmarks(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        candidate = {
            "keypoints": [],
            "draw_keypoints": [[50, 50]],
            "draw_scores": [0.1],
        }

        draw_pose_candidates(frame, [candidate], (0.1, 0.2, 0.9, 1.0), score_threshold=0.35)

        self.assertEqual(frame.sum(), 0)


if __name__ == "__main__":
    unittest.main()
