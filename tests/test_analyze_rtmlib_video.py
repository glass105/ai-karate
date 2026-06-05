import unittest

import numpy as np

from src.analyze_rtmlib_video import PoseTracker, coco_body_keypoints, select_gabriel


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


if __name__ == "__main__":
    unittest.main()
