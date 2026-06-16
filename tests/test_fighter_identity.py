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

    def test_red_score_is_not_positive_cue_without_expected_glove_color(self) -> None:
        identity = FighterIdentity("Gabriel", "left")
        low_red = TrackedBox(10, 100, 0, 200, 100, red_glove_score=0.0)
        high_red = TrackedBox(20, 100, 0, 200, 100, red_glove_score=1.0)

        scores = identity._score_candidates([low_red, high_red], [low_red, high_red])

        self.assertEqual(
            scores[10].gabriel_candidate_score,
            scores[20].gabriel_candidate_score,
        )

    def test_competition_fighter_gate_blocks_mat_edge_candidate(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_competition_fighter=True,
            fighter_candidate_limit=4,
        )

        selected = identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400, competition_fighter_score=0.0),
                TrackedBox(20, 500, 0, 700, 400, competition_fighter_score=1.0),
            ]
        )

        self.assertEqual(selected, 20)

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

    def test_pose_reference_match_cue_prefers_visual_match(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_pose_reference_match=True, recovery_confirmation_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100, pose_reference_match_score=1.0)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, pose_reference_match_score=0.0),
                TrackedBox(30, 120, 0, 220, 100, pose_reference_match_score=1.0),
            ]
        )

        self.assertEqual(selected, 30)

    def test_rejects_visible_face_mismatch_during_recovery(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_face_mismatch=True,
            min_face_match_score=0.5,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, face_match_score=1.0, face_detected=True)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, face_match_score=0.1, face_detected=True),
                TrackedBox(30, 120, 0, 220, 100, face_match_score=0.8, face_detected=True),
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

    def test_requires_blue_gloves_for_initial_assignment(self) -> None:
        identity = FighterIdentity("Gabriel", "left", require_blue_gloves=True, min_blue_glove_score=0.3)

        selected = identity.observe([TrackedBox(10, 100, 0, 200, 100, blue_glove_score=0.1)])

        self.assertIsNone(selected)

    def test_blue_glove_cue_prefers_visual_match(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            expect_blue_gloves=True,
            require_blue_gloves=True,
            min_blue_glove_score=0.3,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, blue_glove_score=1.0)])

        selected = identity.observe(
            [
                TrackedBox(20, 110, 0, 210, 100, blue_glove_score=0.1),
                TrackedBox(30, 120, 0, 220, 100, blue_glove_score=0.8),
            ]
        )

        self.assertEqual(selected, 30)

    def test_rejects_non_target_glove_color(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_blue_gloves=True,
            reject_red_gloves=True,
            min_blue_glove_score=0.2,
            recovery_confirmation_frames=1,
        )

        selected = identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400, blue_glove_score=0.9, red_glove_score=0.1),
                TrackedBox(20, 500, 0, 700, 400, blue_glove_score=0.2, red_glove_score=0.8),
            ]
        )

        self.assertEqual(selected, 10)
        self.assertEqual(identity.identity_scores[20].rejection_reason, "red_glove")

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
        self.assertEqual(identity.visual_track_id, 20)
        self.assertTrue(identity.visual_tentative)

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

    def test_keeps_tentative_grace_visual_when_missing(self) -> None:
        identity = FighterIdentity("Gabriel", "left", visual_grace_frames=2)
        identity.observe([TrackedBox(10, 100, 0, 200, 100)])

        selected = identity.observe([])

        self.assertIsNone(selected)
        self.assertIsNotNone(identity.visual_box)
        self.assertEqual(identity.visual_track_id, 10)
        self.assertTrue(identity.visual_tentative)

    def test_expires_tentative_grace_visual_after_missing_window(self) -> None:
        identity = FighterIdentity("Gabriel", "left", visual_grace_frames=1)
        identity.observe([TrackedBox(10, 100, 0, 200, 100)])

        identity.observe([])
        identity.observe([])

        self.assertIsNone(identity.visual_box)

    def test_soft_red_glove_drop_keeps_strong_continuity_recovery(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_red_gloves=True,
            recovery_confirmation_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 200, 100, appearance=(1, 0), red_glove_score=1.0)])

        selected = identity.observe([TrackedBox(20, 102, 0, 202, 100, appearance=(1, 0), red_glove_score=0.05)])

        self.assertEqual(selected, 20)
        self.assertEqual(identity.label(20), "Gabriel")

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

    def test_initial_assignment_ignores_background_outside_active_pair(self) -> None:
        identity = FighterIdentity("Gabriel", "left", require_red_gloves=True, fighter_candidate_limit=4)

        selected = identity.observe(
            [
                TrackedBox(10, 50, 0, 100, 80, red_glove_score=1.0),
                TrackedBox(20, 200, 0, 400, 400, red_glove_score=0.8),
                TrackedBox(30, 500, 0, 700, 400, red_glove_score=0.8),
            ]
        )

        self.assertEqual(selected, 20)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "not_active_fighter_pair")

    def test_identity_scores_explain_rejections(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            require_red_gloves=True,
            reject_blue_gloves=True,
            require_standing=True,
        )

        identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400, red_glove_score=0.1, standing_score=1.0),
                TrackedBox(20, 500, 0, 700, 400, red_glove_score=0.8, blue_glove_score=0.9, standing_score=1.0),
            ]
        )

        self.assertEqual(identity.identity_scores[10].rejection_reason, "red_below_threshold")
        self.assertEqual(identity.identity_scores[20].rejection_reason, "blue_glove")
        self.assertGreaterEqual(identity.identity_scores[20].gabriel_candidate_score, 0.0)
        self.assertEqual(identity.identity_scores[20].blue_glove_score, 0.9)
        self.assertTrue(identity.identity_scores[20].hard_reject_active)

    def test_identity_scores_report_match_gap_and_confirmed_not_top(self) -> None:
        identity = FighterIdentity("Gabriel", "left", expect_red_gloves=True)

        identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400, red_glove_score=0.2),
                TrackedBox(20, 500, 0, 700, 400, red_glove_score=0.8),
            ]
        )

        self.assertEqual(identity.label(10), "Gabriel")
        self.assertLess(identity.identity_scores[10].match_gap, 0.0)
        self.assertTrue(identity.identity_scores[10].confirmed_not_top)
        self.assertGreater(identity.identity_scores[20].match_gap, 0.0)

    def test_identity_scores_include_face_detected(self) -> None:
        identity = FighterIdentity("Gabriel", "left", reject_face_mismatch=True, min_face_match_score=0.5)

        identity.observe([TrackedBox(10, 100, 0, 300, 400, face_detected=True, face_match_score=0.1)])

        self.assertTrue(identity.identity_scores[10].face_detected)
        self.assertTrue(identity.identity_scores[10].hard_reject_active)

    def test_exclude_reference_match_hard_rejects_candidate(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            min_exclude_reference_match_score=0.75,
        )

        selected = identity.observe(
            [
                TrackedBox(10, 100, 0, 300, 400, exclude_reference_match_score=0.9),
                TrackedBox(20, 500, 0, 700, 400, exclude_reference_match_score=0.1),
            ]
        )

        self.assertEqual(selected, 20)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "exclude_reference")
        self.assertTrue(identity.identity_scores[10].hard_reject_active)

    def test_exclude_face_match_does_not_get_tentative_visual(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            min_exclude_reference_match_score=0.75,
            min_exclude_face_match_score=0.75,
            require_red_gloves=True,
            visual_confidence_threshold=0.1,
        )

        selected = identity.observe(
            [
                TrackedBox(
                    10,
                    100,
                    0,
                    300,
                    400,
                    red_glove_score=1.0,
                    exclude_reference_match_score=0.9,
                    exclude_body_match_score=0.1,
                    exclude_face_match_score=0.9,
                    exclude_face_detected=True,
                )
            ]
        )

        self.assertIsNone(selected)
        self.assertIsNone(identity.visual_track_id)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "exclude_reference")

    def test_body_exclude_match_is_soft_when_gabriel_evidence_is_strong(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            min_exclude_body_match_score=0.75,
            require_red_gloves=True,
            visual_confidence_threshold=0.1,
        )

        selected = identity.observe(
            [
                TrackedBox(
                    10,
                    100,
                    0,
                    300,
                    400,
                    red_glove_score=1.0,
                    exclude_reference_match_score=0.9,
                    exclude_body_match_score=0.9,
                )
            ]
        )

        self.assertEqual(selected, 10)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "")

    def test_hard_veto_exclude_match_rejects_even_with_red_and_pose_evidence(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            exclude_reference_hard_veto=True,
            min_exclude_body_match_score=0.75,
            require_red_gloves=True,
            min_red_glove_score=0.15,
        )

        selected = identity.observe(
            [
                TrackedBox(
                    10,
                    100,
                    0,
                    300,
                    400,
                    red_glove_score=0.40,
                    pose_reference_match_score=0.90,
                    exclude_reference_match_score=0.90,
                    exclude_body_match_score=0.90,
                ),
                TrackedBox(20, 500, 0, 700, 400, red_glove_score=0.20),
            ]
        )

        self.assertEqual(selected, 20)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "exclude_reference")
        self.assertTrue(identity.identity_scores[10].hard_reject_active)

    def test_hard_veto_body_exclude_can_be_overridden_by_strong_gabriel_face(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            exclude_reference_hard_veto=True,
            exclude_veto_allow_strong_face_match=True,
            strong_face_match_score=0.70,
            min_exclude_body_match_score=0.75,
            require_red_gloves=True,
            min_red_glove_score=0.15,
        )

        selected = identity.observe(
            [
                TrackedBox(
                    10,
                    100,
                    0,
                    300,
                    400,
                    red_glove_score=0.40,
                    face_detected=True,
                    face_match_score=0.80,
                    exclude_reference_match_score=0.90,
                    exclude_body_match_score=0.90,
                )
            ]
        )

        self.assertEqual(selected, 10)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "")

    def test_exclude_reference_match_does_not_immediately_drop_existing_lock(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            min_exclude_reference_match_score=0.75,
            locked_fighter_exclude_grace_score=0.95,
            locked_fighter_drop_confirmation_frames=3,
            confirmed_lock_min_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 300, 400, exclude_reference_match_score=0.1)])

        selected = identity.observe([TrackedBox(10, 102, 0, 302, 400, exclude_reference_match_score=0.9)])

        self.assertEqual(selected, 10)
        self.assertEqual(identity.label(10), "Gabriel")
        self.assertEqual(identity.visual_track_id, 10)
        self.assertEqual(identity.identity_scores[10].rejection_reason, "")

    def test_repeated_strong_exclude_with_weak_continuity_drops_existing_lock(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            min_exclude_reference_match_score=0.75,
            locked_fighter_exclude_grace_score=0.95,
            locked_fighter_min_continuity_score=0.99,
            locked_fighter_drop_confirmation_frames=2,
            confirmed_lock_min_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 300, 400, exclude_reference_match_score=0.1)])

        first = identity.observe([TrackedBox(10, 500, 0, 700, 400, exclude_reference_match_score=0.99)])
        second = identity.observe([TrackedBox(10, 500, 0, 700, 400, exclude_reference_match_score=0.99)])

        self.assertEqual(first, 10)
        self.assertIsNone(second)
        self.assertEqual(identity.label(10), "fighter-10")
        self.assertEqual(identity.identity_scores[10].rejection_reason, "exclude_reference")

    def test_hard_veto_drops_existing_lock_after_veto_confirmation_frames(self) -> None:
        identity = FighterIdentity(
            "Gabriel",
            "left",
            reject_exclude_reference_match=True,
            exclude_reference_hard_veto=True,
            min_exclude_reference_match_score=0.75,
            exclude_veto_confirmation_frames=2,
            locked_fighter_drop_confirmation_frames=10,
            confirmed_lock_min_frames=1,
        )
        identity.observe([TrackedBox(10, 100, 0, 300, 400, exclude_reference_match_score=0.1)])

        first = identity.observe([TrackedBox(10, 102, 0, 302, 400, exclude_reference_match_score=0.90)])
        second = identity.observe([TrackedBox(10, 102, 0, 302, 400, exclude_reference_match_score=0.90)])

        self.assertEqual(first, 10)
        self.assertIsNone(second)
        self.assertEqual(identity.label(10), "fighter-10")
        self.assertEqual(identity.identity_scores[10].rejection_reason, "exclude_reference")

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
