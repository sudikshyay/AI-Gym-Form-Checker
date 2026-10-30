
import unittest
from types import SimpleNamespace
from bicep_curls import BicepCurlAnalyzer, joint_angle


def p(x, y):
    return SimpleNamespace(x=x, y=y)


class CurlTests(unittest.TestCase):
    def test_elbow_angle(self):
        self.assertAlmostEqual(joint_angle(p(0, 1), p(0, 0), p(1, 0)), 90)

    def test_full_two_reps(self):
        tracker = BicepCurlAnalyzer()
        for angle in [170, 130, 90, 55, 100, 155, 170, 100, 50, 170]:
            result = tracker.update_angle(angle)
        self.assertEqual(result.reps, 2)

    def test_no_false_count_on_partial(self):
        tracker = BicepCurlAnalyzer()
        for angle in [170, 120, 90, 115, 170]:
            result = tracker.update_angle(angle)
        self.assertEqual(result.reps, 0)

    def test_no_start_mid_curl(self):
        tracker = BicepCurlAnalyzer()
        for angle in [50, 160]:
            result = tracker.update_angle(angle)
        self.assertEqual(result.reps, 0)

    def test_missing_pose_does_not_increment(self):
        tracker = BicepCurlAnalyzer()
        for angle in [170, 55, None, 160]:
            result = tracker.update_angle(angle)
        self.assertEqual(result.reps, 1)

    def test_swing_feedback(self):
        result = BicepCurlAnalyzer().update_angle(100, swing=True)
        self.assertEqual(result.form_error, 'upper_arm_swing')


if __name__ == '__main__':
    unittest.main()

