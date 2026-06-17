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

    def test_counts_lower_leg_snap_as_kick(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_endpoint_motion=100.0,
            min_kick_foot_motion=25.0,
            min_kick_extension_delta=10.0,
            min_kick_extension_ratio=1.2,
        )
        first = pose()
        first[11] = [0, 0]
        first[13] = [40, 0]
        first[15] = [55, 0]
        second = pose()
        second[11] = [0, 0]
        second[13] = [40, 0]
        second[15] = [85, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "kick")
        self.assertEqual(counter.counts[1]["kicks"], 1)

    def test_counts_foot_travel_with_height_change_as_kick(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_endpoint_motion=100.0,
            min_kick_foot_motion=30.0,
            min_kick_extension_delta=100.0,
            min_kick_extension_ratio=10.0,
            min_kick_foot_height_change=20.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[13] = [20, 30]
        first[15] = [40, 60]
        second = pose()
        second[11] = [0, 0]
        second[13] = [20, 30]
        second[15] = [80, 20]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "kick")
        self.assertEqual(counter.counts[1]["kicks"], 1)

    def test_does_not_recount_without_rearm(self) -> None:
        counter = StrikeCounter(fps=30, history_frames=2, punch_cooldown_seconds=0.0)
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]
        third = pose()
        third[5] = [0, 0]
        third[9] = [100, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "punch")
        self.assertEqual(counter.update(1, third), "")
        self.assertEqual(counter.counts[1]["punches"], 1)
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_not_rearmed")

    def test_rearms_after_score_drops(self) -> None:
        counter = StrikeCounter(fps=30, history_frames=2, punch_cooldown_seconds=0.0)
        retracted = pose()
        retracted[5] = [0, 0]
        retracted[9] = [20, 0]
        extended = pose()
        extended[5] = [0, 0]
        extended[9] = [60, 0]

        self.assertEqual(counter.update(1, retracted), "")
        self.assertEqual(counter.update(1, extended), "punch")
        self.assertEqual(counter.update(1, retracted), "")
        self.assertEqual(counter.update(1, extended), "punch")
        self.assertEqual(counter.counts[1]["punches"], 2)

    def test_counts_stronger_kick_when_punch_is_also_above_threshold(self) -> None:
        counter = StrikeCounter(fps=30, history_frames=2)
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [45, 0]
        second[11] = [0, 0]
        second[15] = [70, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "kick")
        self.assertEqual(counter.counts[1]["punches"], 0)
        self.assertEqual(counter.counts[1]["kicks"], 1)


if __name__ == "__main__":
    unittest.main()
