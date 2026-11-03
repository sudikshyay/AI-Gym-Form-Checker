from dataclasses import dataclass
from math import acos, degrees
from typing import Optional

# Calculate Knee Angle

def joint_angle(hip, knee, ankle):

    if hip is None or knee is None or ankle is None:
        return None

    vector_a = (
        hip[0] - knee[0],
        hip[1] - knee[1]
    )

    vector_b = (
        ankle[0] - knee[0],
        ankle[1] - knee[1]
    )

    dot_product = (
        vector_a[0] * vector_b[0]
        + vector_a[1] * vector_b[1]
    )

    magnitude_a = (
        vector_a[0] ** 2 + vector_a[1] ** 2
    ) ** 0.5

    magnitude_b = (
        vector_b[0] ** 2 + vector_b[1] ** 2
    ) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return None

    cosine = dot_product / (magnitude_a * magnitude_b)

    # Keep cosine within the valid range.
    cosine = max(-1, min(1, cosine))

    angle = degrees(acos(cosine))

    return angle

# Store Lunge Results

@dataclass
class LungeResult:

    reps: int
    phase: str
    left_angle: Optional[float]
    right_angle: Optional[float]
    feedback: str

# Standing Lunge Analyzer

class StandingLungeAnalyzer:

    def __init__(self):

        self.reps = 0
        self.phase = "waiting"
        self.has_lunged = False

        # Knee angle thresholds
        self.standing_threshold = 150
        self.lowering_threshold = 120
        self.deep_lunge_threshold = 110

    # Reset repetition counter
    def reset(self):

        self.reps = 0
        self.phase = "waiting"
        self.has_lunged = False

    # Analyze left and right knee angles
    def update_angles(self, left_angle, right_angle):

        # Check if both knee angles are available
        if left_angle is None or right_angle is None:

            return LungeResult(
                self.reps,
                self.phase,
                left_angle,
                right_angle,
                "Move so both legs are visible"
            )

        # Detect standing position
        standing = (
            left_angle >= self.standing_threshold
            and right_angle >= self.standing_threshold
        )

        # Detect lowering into a lunge
        lowered = (
            left_angle <= self.lowering_threshold
            or right_angle <= self.lowering_threshold
        )

        # Detect reaching the bottom of a lunge
        deep_lunge = (
            left_angle <= self.deep_lunge_threshold
            or right_angle <= self.deep_lunge_threshold
        )

        feedback = "Keep moving"

        # Waiting for the starting position
        if self.phase == "waiting":

            if standing:
                self.phase = "standing"
                feedback = "Ready"

            else:
                feedback = "Stand upright to begin"

        # Detect movement into a lunge
        elif self.phase == "standing":

            if lowered:
                self.phase = "lowering"
                feedback = "Lower into the lunge"

            else:
                feedback = "Step into a lunge"

        # Detect the bottom of the lunge
        elif self.phase == "lowering":

            if deep_lunge:
                self.phase = "lunged"
                self.has_lunged = True
                feedback = "Good depth - push back up"

            elif standing:
                self.phase = "standing"
                feedback = "Try lowering a little further"

            else:
                feedback = "Continue lowering"

        # Detect return to standing
        elif self.phase == "lunged":

            if standing and self.has_lunged:

                self.reps += 1

                self.phase = "standing"
                self.has_lunged = False

                feedback = "Repetition completed"

            else:
                feedback = "Push back up to standing"

        # Return current results
        return LungeResult(
            self.reps,
            self.phase,
            left_angle,
            right_angle,
            feedback
        )


# Test the Lunge Analyzer

if __name__ == "__main__":

    analyzer = StandingLungeAnalyzer()

    test_angles = [
        (170, 170),  # Standing
        (115, 165),  # Lowering
        (105, 160),  # Deep lunge
        (170, 170)   # Return to standing
    ]

    for left, right in test_angles:

        result = analyzer.update_angles(left, right)

        print(
            "Left Angle:", left,
            "| Right Angle:", right,
            "| Phase:", result.phase,
            "| Reps:", result.reps,
            "| Feedback:", result.feedback
        )
