import unittest

from standing_lunges import (
    joint_angle,
    StandingLungeAnalyzer
)


class TestStandingLunges(unittest.TestCase):

    # check knee-angle calculation
    def test_joint_angle(self):

        hip = (0, 1)
        knee = (0, 0)
        ankle = (1, 0)

        angle = joint_angle(hip, knee, ankle)

        self.assertAlmostEqual(angle, 90, places=2)


    # check one complete lunge
    def test_one_complete_lunge(self):

        analyzer = StandingLungeAnalyzer()

        analyzer.update_angles(170, 170)
        analyzer.update_angles(115, 165)
        analyzer.update_angles(105, 160)
        result = analyzer.update_angles(170, 170)

        self.assertEqual(result.reps, 1)


    # check two complete lunges
    def test_two_complete_lunges(self):

        analyzer = StandingLungeAnalyzer()

        test_angles = [
            (170, 170),
            (115, 165),
            (105, 160),
            (170, 170),

            (170, 170),
            (115, 165),
            (105, 160),
            (170, 170)
        ]

        for left, right in test_angles:
            result = analyzer.update_angles(left, right)

        self.assertEqual(result.reps, 2)


    # partial lunge is not counted
    def test_partial_lunge(self):

        analyzer = StandingLungeAnalyzer()

        analyzer.update_angles(170, 170)
        analyzer.update_angles(115, 165)
        analyzer.update_angles(130, 165)
        result = analyzer.update_angles(170, 170)

        self.assertEqual(result.reps, 0)


    # lunge does not start without standing
    def test_starting_position(self):

        analyzer = StandingLungeAnalyzer()

        analyzer.update_angles(110, 160)
        result = analyzer.update_angles(105, 160)

        self.assertEqual(result.reps, 0)
        self.assertEqual(result.phase, "waiting")


    # check missing knee-angle information
    def test_missing_angle(self):

        analyzer = StandingLungeAnalyzer()

        analyzer.update_angles(170, 170)
        analyzer.update_angles(115, 165)
        analyzer.update_angles(105, 160)

        result = analyzer.update_angles(None, 160)

        self.assertEqual(result.reps, 0)
        self.assertEqual(result.phase, "lunged")


    # check analyzer reset
    def test_reset(self):

        analyzer = StandingLungeAnalyzer()

        analyzer.update_angles(170, 170)
        analyzer.update_angles(115, 165)
        analyzer.update_angles(105, 160)
        analyzer.update_angles(170, 170)

        analyzer.reset()

        self.assertEqual(analyzer.reps, 0)
        self.assertEqual(analyzer.phase, "waiting")


if __name__ == "__main__":
    unittest.main()
