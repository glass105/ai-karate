import unittest

from src.fighter_identity import FighterIdentity, TrackedBox


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

    def test_retains_original_mapping(self) -> None:
        identity = FighterIdentity("Gabriel", "right")
        identity.observe([TrackedBox(10, 100, 0, 200, 100)])
        identity.observe([TrackedBox(20, 500, 0, 600, 100)])
        self.assertEqual(identity.fighter_a_track_id, 10)


if __name__ == "__main__":
    unittest.main()
