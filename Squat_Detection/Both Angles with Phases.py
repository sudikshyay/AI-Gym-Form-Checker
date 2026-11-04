
import cv2
import mediapipe as mp
import math
import time
import csv
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# SETTINGS
MODEL_PATH = "pose_landmarker_full.task"

STANDING_ANGLE = 160
BOTTOM_ANGLE = 105
MIN_MOVEMENT_CHANGE = 0.5


# HELPER FUNCTION: CALCULATE JOINT ANGLE
def calculate_angle(a, b, c):
    """
    Calculate the angle at point b using points a, b, and c.
    For a knee angle:
    a = hip, b = knee, c = ankle
    """

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle

# HELPER FUNCTION: CHECK LANDMARK VISIBILITY
def is_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= 0.5



# SQUAT PHASE DETECTOR
class SquatPhaseDetector:

    def __init__(self):
        self.previous_angle = None
        self.current_phase = "STANDING"

    def update(self, average_angle):

        if average_angle is None:
            return self.current_phase

        # Standing position
        if average_angle >= STANDING_ANGLE:
            self.current_phase = "STANDING"

        # Lowest part of the squat
        elif average_angle <= BOTTOM_ANGLE:
            self.current_phase = "BOTTOM"

        else:
            # Detect movement using the change in knee angle
            if self.previous_angle is not None:

                angle_change = average_angle - self.previous_angle

                # Knee angle is decreasing: moving down
                if angle_change < -MIN_MOVEMENT_CHANGE:
                    self.current_phase = "GOING DOWN"

                # Knee angle is increasing: moving up
                elif angle_change > MIN_MOVEMENT_CHANGE:
                    self.current_phase = "GOING UP"

        self.previous_angle = average_angle

        return self.current_phase



# MEDIAPIPE POSE LANDMARKER SETUP
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)


# OPEN WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    detector.close()
    raise SystemExit

print("Webcam opened successfully.")
print("Press ESC to exit.")


# CSV FILE SETUP
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"squat_measurements_{timestamp}.csv"

csv_file = open(
    csv_filename,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Elapsed_Time_Seconds",
    "Left_Knee_Angle_Degrees",
    "Right_Knee_Angle_Degrees",
    "Average_Knee_Angle_Degrees",
    "Squat_Phase"
])


# INITIALIZE TRACKING


phase_detector = SquatPhaseDetector()

start_time = time.monotonic()
last_timestamp_ms = -1


# MAIN LOOP


try:

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert OpenCV frame to MediaPipe image
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe requires increasing timestamps
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        # Detect body landmarks
        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        left_angle = None
        right_angle = None
        average_angle = None

        # PROCESS BODY LANDMARKS
      

        if results.pose_landmarks:

            landmarks = results.pose_landmarks[0]

            # Left leg: hip 23, knee 25, ankle 27
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]

            # Right leg: hip 24, knee 26, ankle 28
            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]

            # Calculate angles only when landmarks are visible
            if (
                is_visible(left_hip)
                and is_visible(left_knee)
                and is_visible(left_ankle)
            ):
                left_angle = calculate_angle(
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y),
                    (left_ankle.x, left_ankle.y)
                )

            if (
                is_visible(right_hip)
                and is_visible(right_knee)
                and is_visible(right_ankle)
            ):
                right_angle = calculate_angle(
                    (right_hip.x, right_hip.y),
                    (right_knee.x, right_knee.y),
                    (right_ankle.x, right_ankle.y)
                )

            # Draw the leg landmarks and connecting lines
            h, w, _ = frame.shape

            leg_connections = [
                (23, 25),
                (25, 27),
                (24, 26),
                (26, 28)
            ]

            for start_idx, end_idx in leg_connections:

                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                start_xy = (
                    int(start_point.x * w),
                    int(start_point.y * h)
                )

                end_xy = (
                    int(end_point.x * w),
                    int(end_point.y * h)
                )

                # Green for left leg, blue for right leg
                if start_idx in (23, 25):
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.line(
                    frame,
                    start_xy,
                    end_xy,
                    color,
                    3
                )

            # Draw knee points
            if is_visible(left_knee):
                cv2.circle(
                    frame,
                    (
                        int(left_knee.x * w),
                        int(left_knee.y * h)
                    ),
                    8,
                    (0, 255, 0),
                    -1
                )

            if is_visible(right_knee):
                cv2.circle(
                    frame,
                    (
                        int(right_knee.x * w),
                        int(right_knee.y * h)
                    ),
                    8,
                    (255, 0, 0),
                    -1
                )

            # Average available knee angles
            available_angles = []

            if left_angle is not None:
                available_angles.append(left_angle)

            if right_angle is not None:
                available_angles.append(right_angle)

            if available_angles:
                average_angle = sum(available_angles) / len(
                    available_angles
                )

        # DETECT SQUAT PHASE
      

        phase = phase_detector.update(average_angle)

        # DISPLAY INFORMATION
      

        if left_angle is not None:
            cv2.putText(
                frame,
                f"Left Knee: {left_angle:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if right_angle is not None:
            cv2.putText(
                frame,
                f"Right Knee: {right_angle:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        if average_angle is not None:
            cv2.putText(
                frame,
                f"Average Knee: {average_angle:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"Squat Phase: {phase}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Gym Posture Checker",
            frame
        )

      
        # PRINT AND SAVE DATA
   

        elapsed_time = time.monotonic() - start_time

        left_text = (
            f"{left_angle:.1f}"
            if left_angle is not None
            else "N/A"
        )

        right_text = (
            f"{right_angle:.1f}"
            if right_angle is not None
            else "N/A"
        )

        average_text = (
            f"{average_angle:.1f}"
            if average_angle is not None
            else "N/A"
        )

        print(
            f"Time: {elapsed_time:.2f}s | "
            f"Left: {left_text} | "
            f"Right: {right_text} | "
            f"Average: {average_text} | "
            f"Phase: {phase}"
        )

        csv_writer.writerow([
            round(elapsed_time, 3),
            round(left_angle, 2) if left_angle is not None else "",
            round(right_angle, 2) if right_angle is not None else "",
            round(average_angle, 2) if average_angle is not None else "",
            phase
        ])

        # Press ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    cap.release()
    detector.close()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"\nCamera closed.")
    print(f"Measurements saved to: {csv_filename}")
