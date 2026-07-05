import unittest

from src.strike_counter import StrikeCounter


def pose() -> list[list[float]]:
    return [[0.0, 0.0] for _ in range(17)]


class StrikeCounterTests(unittest.TestCase):
    def test_counts_short_default_extension_as_fake_punch(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]
        third = pose()
        third[5] = [0, 0]
        third[9] = [20, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.update(1, third), "fake_punch")
        self.assertEqual(counter.counts[1]["punches"], 0)
        self.assertEqual(counter.counts[1]["fake_punches"], 1)

    def test_counts_short_extension_as_fake_punch_when_commitment_required(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_commitment_frames=2,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]
        third = pose()
        third[5] = [0, 0]
        third[9] = [20, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_pending_commitment")
        self.assertEqual(counter.update(1, third), "fake_punch")
        self.assertEqual(counter.counts[1]["punches"], 0)
        self.assertEqual(counter.counts[1]["fake_punches"], 1)
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "fake_punch")
        self.assertEqual(counter.last_debug[1].punch_commitment_frames, 1)

    def test_counts_committed_extension_as_real_punch(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=3,
            min_punch_commitment_frames=2,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]
        third = pose()
        third[5] = [0, 0]
        third[9] = [60, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.update(1, third), "punch")
        self.assertEqual(counter.counts[1]["punches"], 1)
        self.assertEqual(counter.counts[1]["fake_punches"], 0)
        self.assertEqual(counter.last_debug[1].punch_commitment_frames, 2)

    def test_rejects_punch_when_identity_lock_is_too_new(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_commitment_frames=1,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]

        self.assertEqual(counter.update(1, first, lock_frames=1, min_count_lock_frames=5), "")
        self.assertEqual(counter.update(1, second, lock_frames=2, min_count_lock_frames=5), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_identity_lock_too_new")
        self.assertEqual(counter.counts[1]["punches"], 0)

    def test_rejects_punch_when_opponent_is_too_far(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_commitment_frames=1,
            max_punch_opponent_distance_body_heights=0.85,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]

        self.assertEqual(counter.update(1, first, opponent_distance_body_heights=1.2), "")
        self.assertEqual(counter.update(1, second, opponent_distance_body_heights=1.2), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_opponent_too_far")
        self.assertEqual(counter.counts[1]["punches"], 0)

    def test_rejects_implausible_punch_extension_ratio(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_commitment_frames=1,
            max_punch_extension_ratio=4.0,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [10, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [70, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_pose_ratio_implausible")
        self.assertEqual(counter.counts[1]["punches"], 0)

    def test_rejects_mid_strength_punch_moving_away_from_target(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_score=1.2,
            min_punch_commitment_frames=1,
            punch_target_gate_score=1.5,
            max_untargeted_punch_distance_body_heights=0.75,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [56, 0]

        counter.update(
            1,
            first,
            opponent_center=(100.0, 0.0),
            opponent_height=100.0,
        )
        self.assertEqual(
            counter.update(
                1,
                second,
                opponent_center=(-100.0, 0.0),
                opponent_height=100.0,
            ),
            "",
        )
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_not_targeted")

    def test_keeps_mid_strength_punch_approaching_target(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_score=1.2,
            min_punch_commitment_frames=1,
            punch_target_gate_score=1.5,
            max_untargeted_punch_distance_body_heights=0.75,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [60, 0]

        counter.update(
            1,
            first,
            opponent_center=(100.0, 0.0),
            opponent_height=100.0,
        )
        self.assertEqual(
            counter.update(
                1,
                second,
                opponent_center=(100.0, 0.0),
                opponent_height=100.0,
            ),
            "punch",
        )
        self.assertEqual(counter.last_debug[1].punch_target_alignment, 1.0)

    def test_rejects_punch_like_motion_during_kick_when_hand_misses_target(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_score=1.2,
            min_punch_commitment_frames=1,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            max_mixed_strike_punch_target_distance_body_heights=0.60,
            min_strike_score_gap=0.0,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[5] = [0, 0]
        second[9] = [80, 0]
        second[11] = [0, 0]
        second[15] = [55, 0]

        counter.update(
            1,
            first,
            opponent_center=(180.0, 0.0),
            opponent_height=100.0,
        )
        self.assertEqual(
            counter.update(
                1,
                second,
                opponent_center=(180.0, 0.0),
                opponent_height=100.0,
            ),
            "",
        )
        self.assertEqual(
            counter.last_debug[1].strike_rejection_reason,
            "punch_kick_setup_not_targeted",
        )

    def test_suppresses_punch_immediately_after_confirmed_kick(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_commitment_frames=1,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            post_kick_punch_suppression_frames=5,
            punch_cooldown_seconds=0.0,
        )
        kick_first = pose()
        kick_first[11] = [0, 0]
        kick_first[15] = [20, 0]
        kick_second = pose()
        kick_second[11] = [0, 0]
        kick_second[15] = [60, 0]
        punch_first = pose()
        punch_first[5] = [0, 0]
        punch_first[9] = [20, 0]
        punch_second = pose()
        punch_second[5] = [0, 0]
        punch_second[9] = [60, 0]

        counter.update(1, kick_first)
        self.assertEqual(counter.update(1, kick_second), "kick")
        counter.update(1, punch_first)
        self.assertEqual(counter.update(1, punch_second), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_after_recent_kick")

    def test_rejects_low_score_duplicate_punch_shortly_after_confirmed_punch(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_punch_score=1.10,
            min_punch_commitment_frames=1,
            duplicate_punch_window_frames=12,
            min_duplicate_punch_score=1.20,
            punch_cooldown_seconds=0.0,
        )
        first = pose()
        first[5] = [0, 0]
        first[9] = [20, 0]
        strong = pose()
        strong[5] = [0, 0]
        strong[9] = [60, 0]
        reset = pose()
        reset[5] = [0, 0]
        reset[9] = [20, 0]
        weak = pose()
        weak[5] = [0, 0]
        weak[9] = [49, 0]

        counter.update(1, first, frame_index=1)
        self.assertEqual(counter.update(1, strong, frame_index=2), "punch")
        counter.update(1, reset, frame_index=3)
        counter.update(1, reset, frame_index=12)
        self.assertEqual(counter.update(1, weak, frame_index=13), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "punch_duplicate_low_score")

    def test_counts_extended_ankle_as_kick(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
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
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
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

    def test_counts_kick_when_opponent_is_close(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            max_kick_opponent_distance_body_heights=0.75,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        self.assertEqual(counter.update(1, first, opponent_distance_body_heights=0.5), "")
        self.assertEqual(counter.update(1, second, opponent_distance_body_heights=0.5), "kick")
        self.assertEqual(counter.last_debug[1].kick_opponent_distance_body_heights, 0.5)

    def test_rejects_kick_when_opponent_is_too_far(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            max_kick_opponent_distance_body_heights=0.75,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        self.assertEqual(counter.update(1, first, opponent_distance_body_heights=1.0), "")
        self.assertEqual(counter.update(1, second, opponent_distance_body_heights=1.0), "")
        self.assertEqual(counter.counts[1]["kicks"], 0)
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "kick_opponent_too_far")

    def test_rejects_kick_when_no_opponent_is_available(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            max_kick_opponent_distance_body_heights=0.75,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.counts[1]["kicks"], 0)
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "kick_no_opponent")

    def test_counts_foot_travel_with_height_change_as_kick(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_endpoint_motion=100.0,
            min_kick_foot_motion=30.0,
            min_kick_extension_delta=100.0,
            min_kick_extension_ratio=10.0,
            min_kick_commitment_frames=1,
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
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            punch_cooldown_seconds=0.0,
            min_punch_commitment_frames=1,
        )
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
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            punch_cooldown_seconds=0.0,
            min_punch_commitment_frames=1,
        )
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
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
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

    def test_rejects_setup_step_after_subtracting_hip_translation(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[13] = [10, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [40, 0]
        second[13] = [50, 0]
        second[15] = [75, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.counts[1]["kicks"], 0)

    def test_clears_pose_history_after_tracking_gap(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=3,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [22, 0]
        after_gap = pose()
        after_gap[11] = [0, 0]
        after_gap[15] = [80, 0]

        self.assertEqual(counter.update(1, first, frame_index=1), "")
        self.assertEqual(counter.update(1, second, frame_index=2), "")
        self.assertEqual(counter.update(1, after_gap, frame_index=10), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "history_warmup")
        self.assertEqual(counter.counts[1]["kicks"], 0)

    def test_requires_two_committed_kick_frames(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=3,
            min_kick_commitment_frames=2,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]
        third = pose()
        third[11] = [0, 0]
        third[15] = [62, 0]

        self.assertEqual(counter.update(1, first), "")
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(counter.update(1, third), "kick")
        self.assertEqual(counter.last_debug[1].kick_commitment_frames, 2)

    def test_rejects_one_frame_kick_spike_as_not_committed(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=3,
            min_kick_commitment_frames=2,
            min_kick_foot_height_change=0.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [20, 0]
        spike = pose()
        spike[11] = [0, 0]
        spike[15] = [60, 0]

        counter.update(1, first)
        counter.update(1, second)
        self.assertEqual(counter.update(1, spike), "")
        self.assertEqual(counter.last_debug[1].strike_rejection_reason, "kick_not_committed")

    def test_rejects_kick_without_foot_elevation(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            min_kick_foot_elevation_body_heights=0.10,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        first[16] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]
        second[16] = [20, 0]

        counter.update(1, first)
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(
            counter.last_debug[1].strike_rejection_reason,
            "kick_insufficient_foot_elevation",
        )

    def test_rejects_implausible_kick_extension_ratio(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            max_kick_extension_ratio=4.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [1, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        counter.update(1, first)
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(
            counter.last_debug[1].strike_rejection_reason,
            "kick_pose_ratio_implausible",
        )

    def test_rejects_step_when_support_foot_moves(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            min_kick_score=1.35,
            min_kick_foot_elevation_body_heights=0.10,
            max_kick_support_foot_motion_body_heights=0.08,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 40]
        first[16] = [0, 100]
        second = pose()
        second[11] = [0, 0]
        second[15] = [80, 20]
        second[16] = [40, 100]

        counter.update(1, first)
        self.assertEqual(counter.update(1, second), "")
        self.assertEqual(
            counter.last_debug[1].strike_rejection_reason,
            "kick_support_foot_moving",
        )

    def test_uses_recent_opponent_during_brief_occlusion(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=0.0,
            max_kick_opponent_distance_body_heights=0.70,
            kick_opponent_memory_frames=2,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        second = pose()
        second[11] = [0, 0]
        second[15] = [60, 0]

        counter.update(
            1,
            first,
            opponent_distance_body_heights=0.5,
            frame_index=1,
        )
        self.assertEqual(counter.update(1, second, frame_index=2), "kick")
        self.assertTrue(counter.last_debug[1].kick_used_recent_opponent)
        self.assertEqual(
            counter.last_debug[1].kick_opponent_distance_body_heights,
            0.5,
        )

    def test_requires_relative_foot_height_change(self) -> None:
        counter = StrikeCounter(
            fps=30,
            history_frames=2,
            min_kick_commitment_frames=1,
            min_kick_foot_height_change=20.0,
        )
        first = pose()
        first[11] = [0, 0]
        first[15] = [20, 0]
        flat = pose()
        flat[11] = [0, 0]
        flat[15] = [60, 0]

        counter.update(1, first)
        self.assertEqual(counter.update(1, flat), "")
        self.assertEqual(counter.counts[1]["kicks"], 0)


if __name__ == "__main__":
    unittest.main()
