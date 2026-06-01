import unittest

from src.strike_counter import StrikeCounter


def pose() -> list[list[float]]:
    return [[0.0, 0.0] for _ in range(17)]


class StrikeCounterTests(unittest.TestCase):
    def test_counts_extended_wrist_as_punch(self) -> None:
        counter = StrikeCounter(fps=30, history_frames=2)
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "punch")
        self.assertEqual(counter.counts[1]["punches"], 1)

    def test_counts_extended_ankle_as_kick(self) -> None:
        counter = StrikeCounter(fps=30, history_frames=2)
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "kick")
        self.assertEqual(counter.counts[1]["kicks"], 1)


if __name__ == "__main__":
    unittest.main()
