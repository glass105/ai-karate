import unittest

import cv2
import numpy as np

from src.analyze_rtmlib_video import PoseTracker, coco_body_keypoints, color_scores, select_gabriel


class AnalyzeRtmlibVideoTests(unittest.TestCase):
    def test_maps_first_17_rtmw_keypoints_to_coco_body(self) -> None:
        points = np.asarray([[index, index + 0.5] for index in range(133)])

        mapped = coco_body_keypoints(points)

        self.assertEqual(len(mapped), 17)
        self.assertEqual(mapped[0], [0.0, 0.5])
        self.assertEqual(mapped[16], [16.0, 16.5])

    def test_pose_tracker_reuses_nearby_track_id(self) -> None:
        tracker = PoseTracker(max_distance=50)
        first = tracker.update([{"center": (100.0, 100.0)}])
        second = tracker.update([{"center": (120.0, 100.0)}])

        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

    def test_selects_one_gabriel_candidate(self) -> None:
        candidates = [
            {"track_id": 1, "center": (100.0, 100.0), "box": (50, 50, 150, 150), "red_glove_score": 1.0, "white_uniform_score": 1.0},
            {"track_id": 2, "center": (300.0, 100.0), "box": (250, 50, 350, 150), "red_glove_score": 0.0, "white_uniform_score": 0.0},
        ]

        selected = select_gabriel(candidates, "left", None, 0, 15)

        self.assertEqual(selected["track_id"], 1)

    def test_red_glove_score_requires_red_near_wrists_not_torso(self) -> None:
        frame = np.full((200, 120, 3), 255, dtype=np.uint8)
        box = (10, 10, 110, 190)
        points = np.zeros((17, 2), dtype=np.float32)
        scores = np.zeros(17, dtype=np.float32)
        for index, point in {
            5: (40, 55),
            6: (80, 55),
            7: (35, 90),
            8: (85, 90),
            9: (30, 140),
            10: (90, 140),
            11: (45, 120),
            12: (75, 120),
            13: (45, 165),
            14: (75, 165),
            15: (45, 185),
            16: (75, 185),
        }.items():
            points[index] = point
            scores[index] = 1.0

        red = (0, 0, 255)
        red_gloves = frame.copy()
        cv2.circle(red_gloves, (30, 140), 14, red, -1)
        cv2.circle(red_gloves, (90, 140), 14, red, -1)
        glove_score, *_ = color_scores(red_gloves, points, scores, box)

        red_shirt = frame.copy()
        cv2.rectangle(red_shirt, (25, 45), (95, 125), red, -1)
        shirt_score, *_ = color_scores(red_shirt, points, scores, box)

        self.assertGreater(glove_score, 0.50)
        self.assertLess(shirt_score, 0.05)


if __name__ == "__main__":
    unittest.main()
