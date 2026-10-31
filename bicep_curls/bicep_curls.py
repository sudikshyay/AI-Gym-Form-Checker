
import math
from dataclasses import dataclass
from typing import Optional


def joint_angle(shoulder, elbow, wrist):
    """Calculate the elbow angle in degrees."""
    a = (shoulder.x - elbow.x, shoulder.y - elbow.y)
    b = (wrist.x - elbow.x, wrist.y - elbow.y)

    dot = a[0] * b[0] + a[1] * b[1]
    magnitude_a = math.hypot(*a)
    magnitude_b = math.hypot(*b)

    if magnitude_a == 0 or magnitude_b == 0:
        return None

    cosine = dot / (magnitude_a * magnitude_b)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


@dataclass
class CurlResult:
    reps: int
    phase: str
    angle: Optional[float]
    form_error: Optional[str]


class BicepCurlAnalyzer:
    """Tracks bicep curl repetitions and basic form errors."""

    def __init__(self):
        self.reps = 0
        self.phase = "waiting"
        self.has_curled = False

    def update_angle(self, angle, swing=False):
        form_error = None

        if swing:
            form_error = "upper_arm_swing"

        if angle is None:
            return CurlResult(
                self.reps, self.phase, None, form_error
            )

        if self.phase == "waiting":
            if angle >= 150:
                self.phase = "extended"

        elif self.phase == "extended":
            if angle <= 65:
                self.phase = "curled"
                self.has_curled = True

        elif self.phase == "curled":
            if angle >= 150 and self.has_curled:
                self.reps += 1
                self.phase = "extended"
                self.has_curled = False

        return CurlResult(
            self.reps, self.phase, angle, form_error
        )
