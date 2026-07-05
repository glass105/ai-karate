import unittest

from src.analyze_video import is_inside_arena, parse_arena_roi
from src.fighter_identity import TrackedBox


class AnalyzeVideoTests(unittest.TestCase):
    def test_parses_normalized_arena_roi(self) -> None:
        self.assertEqual(parse_arena_roi("0.1,0.2,0.9,0.95"), (0.1, 0.2, 0.9, 0.95))

    def test_rejects_invalid_arena_roi(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arena_roi("0.9,0.2,0.1,0.95")

    def test_filters_detection_outside_arena(self) -> None:
        arena = (0.1, 0.2, 0.9, 0.95)
        self.assertTrue(is_inside_arena(TrackedBox(1, 100, 100, 200, 300), arena, 1000, 1000))
        self.assertFalse(is_inside_arena(TrackedBox(2, 0, 100, 50, 300), arena, 1000, 1000))


if __name__ == "__main__":
    unittest.main()
