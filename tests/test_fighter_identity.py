import unittest

from src.fighter_identity import FighterIdentity, TrackedBox, pose_descriptor


class FighterIdentityTests(unittest.TestCase):
    def test_assigns_leftmost_fighter(self) -> None:
        identity = FighterIdentity("Gabriel", "left")
        track_id = identity.observe(
            [
                TrackedBox(20, 500, 0, 600, 100),
                TrackedBox(10, 100, 0, 200, 100),
            ]
        )
        self.assertEqual(track_id, 10)
        self.assertEqual(identity.label(10), "Gabriel")

    def test_recovers_mapping_when_tracker_id_changes(self) -> None:
        identity = FighterIdentity("Gabriel", "right", recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0))])
        identity.observe([TrackedBox(20, 110, 0, 210, 100, appearance=(1, 0))])
        self.assertEqual(identity.fighter_a_track_id, 20)
        self.assertEqual(identity.fighter_a_track_ids, {10, 20})
        self.assertEqual(identity.label(20), "Gabriel")
        self.assertEqual(identity.label(10), "fighter-10")
        self.assertEqual(identity.recovery_count, 1)

    def test_does_not_reassign_to_distant_person(self) -> None:
        identity = FighterIdentity("Gabriel", "left")
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0))])
        identity.observe([TrackedBox(20, 1000, 0, 1100, 100, appearance=(0, 1))])
        self.assertEqual(identity.fighter_a_track_id, 10)

    def test_normalizes_pose_within_box(self) -> None:
        box = TrackedBox(10, 100, 200, 300, 600)
        self.assertEqual(pose_descriptor(box, [[100, 200], [300, 600]]), (0, 0, 1, 1))

    def test_red_glove_cue_prefers_visual_match(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_red_gloves=True, recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100, red_glove_score=1.0)])
        identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, red_glove_score=0.0),
                TrackedBox(30, 120, 0, 220, 100, red_glove_score=1.0),
            ]
        )
        self.assertEqual(identity.fighter_a_track_id, 30)

    def test_black_belt_cue_prefers_visual_match(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_black_belt=True, recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100, black_belt_score=1.0)])
        identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, black_belt_score=0.0),
                TrackedBox(30, 120, 0, 220, 100, black_belt_score=1.0),
            ]
        )
        self.assertEqual(identity.fighter_a_track_id, 30)

    def test_taller_cue_prefers_taller_candidate(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_taller=True, recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 200)])
        identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100),
                TrackedBox(30, 120, 0, 220, 200),
            ]
        )
        self.assertEqual(identity.fighter_a_track_id, 30)

    def test_reference_match_cue_prefers_visual_match(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_reference_match=True, recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100, reference_match_score=1.0)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, reference_match_score=0.0),
                TrackedBox(30, 120, 0, 220, 100, reference_match_score=1.0),
            ]
        )

        self.assertEqual(selected, 30)

    def test_requires_reference_match_for_initial_assignment(self) -> None:
        identity = FighterIdentity("Gabriel", "left", require_reference_match=True, min_reference_match_score=0.5)

        selected = identity.observe([TrackedBox(10, 100, 0, 200, 100, reference_match_score=0.1)])

        self.assertIsNone(selected)

    def test_requires_red_gloves_for_initial_assignment(self) -> None:
        identity = FighterIdentity("Gabriel", "left", require_red_gloves=True)

        selected = identity.observe([TrackedBox(10, 100, 0, 200, 100, red_glove_score=0.0)])

        self.assertIsNone(selected)

    def test_requires_white_gloves_for_initial_assignment(self) -> None:
        identity = FighterIdentity("Gabriel", "left", require_white_gloves=True)

        selected = identity.observe([TrackedBox(10, 100, 0, 200, 100, white_glove_score=0.0)])

        self.assertIsNone(selected)

    def test_rejects_blue_gloves_during_recovery(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_red_gloves=True,
            reject_blue_gloves=True,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, red_glove_score=1.0)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, red_glove_score=0.1, blue_glove_score=0.8),
                TrackedBox(30, 120, 0, 220, 100, red_glove_score=0.8, blue_glove_score=0.1),
            ]
        )

        self.assertEqual(selected, 30)

    def test_rejects_seated_person_during_recovery(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_red_gloves=True,
            require_standing=True,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 200, red_glove_score=1.0, standing_score=1.0)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, red_glove_score=1.0, standing_score=0.1),
                TrackedBox(30, 120, 0, 220, 200, red_glove_score=1.0, standing_score=0.9),
            ]
        )

        self.assertEqual(selected, 30)

    def test_reset_prefers_left_side_after_missing_frames(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reset_after_missing_frames=2,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 400, 0, 500, 100)])
        identity.observe([])
        identity.observe([])
        identity.observe(
            [
                TrackedBox(20, 300, 0, 400, 100),
                TrackedBox(30, 500, 0, 600, 100),
            ]
        )
        self.assertEqual(identity.fighter_a_track_id, 20)

    def test_does_not_transfer_label_to_one_frame_impostor(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            recovery_confirmation_frames=3,
            strong_recovery_threshold=0.0,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0))])

        selected = identity.observe([TrackedBox(20, 110, 0, 210, 100, appearance=(1, 0))])

        self.assertIsNone(selected)
        self.assertEqual(identity.label(20), "fighter-20")
        self.assertEqual(identity.fighter_a_track_id, 10)

    def test_recovers_label_after_confirmation_window(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            recovery_confirmation_frames=3,
            strong_recovery_threshold=0.0,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0))])
        replacement = [TrackedBox(20, 110, 0, 210, 100, appearance=(1, 0))]

        self.assertIsNone(identity.observe(replacement))
        self.assertIsNone(identity.observe(replacement))
        self.assertEqual(identity.observe(replacement), 20)
        self.assertEqual(identity.label(20), "Gabriel")
        self.assertEqual(identity.recovery_count, 1)

    def test_recovers_strong_track_replacement_immediately(self) -> None:
        identity = FighterIdentity("Gabriel", "left")
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0))])

        selected = identity.observe([TrackedBox(20, 102, 0, 202, 100, appearance=(1, 0))])

        self.assertEqual(selected, 20)
        self.assertEqual(identity.label(20), "Gabriel")

    def test_ignores_small_background_person_during_recovery(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            fighter_candidate_limit=2,
            recovery_confirmation_frames=1,
            strong_recovery_threshold=0.0,
        )
        identity.observe([TrackedBox(10, 100, 0, 300, 400, appearance=(1, 0))])

        selected = identity.observe(
            [
                TrackedBox(20, 105, 0, 155, 80, appearance=(1, 0)),
                TrackedBox(30, 110, 0, 310, 400, appearance=(1, 0)),
                TrackedBox(40, 500, 0, 700, 400, appearance=(0, 1)),
            ]
        )

        self.assertEqual(selected, 30)
        self.assertEqual(identity.label(20), "fighter-20")

    def test_lineup_pause_reanchors_to_initial_left_side(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "right",
            lineup_pause_frames=2,
            lineup_separation_threshold=0.75,
        )
        identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400),
                TrackedBox(20, 500, 0, 700, 400),
            ]
        )
        identity.fighter_a_start = "left"

        identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400),
                TrackedBox(20, 500, 0, 700, 400),
            ]
        )
        selected = identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400),
                TrackedBox(20, 500, 0, 700, 400),
            ]
        )

        self.assertEqual(selected, 10)
        self.assertEqual(identity.label(10), "Gabriel")
        self.assertEqual(identity.reset_count, 1)

    def test_lineup_pause_reanchors_to_initial_right_side_once(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            lineup_pause_frames=2,
            lineup_separation_threshold=0.75,
        )
        lineup = [
            TrackedBox(10, 100, 0, 300, 400),
            TrackedBox(20, 500, 0, 700, 400),
        ]
        identity.observe(lineup)
        identity.fighter_a_start = "right"

        identity.observe(lineup)
        self.assertEqual(identity.observe(lineup), 20)
        self.assertEqual(identity.observe(lineup), 20)
        self.assertEqual(identity.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
