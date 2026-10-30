
"""
Senior Design Project: AI Gym Form Checker
Exercise: Bicep Curls

Features:
- MediaPipe pose detection
- Right-arm elbow angle calculation
- Automatic repetition counting
- On-screen exercise feedback
- Voice announcements using macOS text-to-speech
- CSV logging

Press ESC to exit.
"""

import csv
import time
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp

from bicep_curls import BicepCurlAnalyzer, joint_angle


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

# The model is in the MAIN project directory.
MODEL = (
    Path(__file__).resolve().parent.parent
    / "pose_landmarker_full.task"
)

# Save CSV results in the bicep_curls directory.
LOG = (
    Path(__file__).resolve().parent
    / "bicep_curl_results.csv"
)

if not MODEL.exists() or MODEL.stat().st_size == 0:
    raise FileNotFoundError(
        f"MediaPipe model is missing or empty: {MODEL}"
    )


# --------------------------------------------------
# 2. VOICE FEEDBACK SYSTEM
# --------------------------------------------------

last_spoken_time = float("-inf")
last_rep_count = 0
voice_process = None

VOICE_COOLDOWN = 1.5


def speak(message):
    """
    Speak using the built-in macOS 'say' command.
    Prevent overlapping speech and excessive announcements.
    """
    global last_spoken_time, voice_process

    current_time = time.monotonic()

    if current_time - last_spoken_time < VOICE_COOLDOWN:
        return

    # Do not overlap announcements.
    if (
        voice_process is not None
        and voice_process.poll() is None
    ):
        return

    voice_process = subprocess.Popen(
        ["say", message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    last_spoken_time = current_time


# --------------------------------------------------
# 3. MEDIAPIPE SETUP
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)


# --------------------------------------------------
# 4. INITIALIZE BICEP CURL ANALYZER
# --------------------------------------------------

analyzer = BicepCurlAnalyzer()

start_time = time.monotonic()
last_timestamp = -1


# --------------------------------------------------
# 5. CAMERA SETUP
# --------------------------------------------------

# Uses the selected camera at index 0.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

print("Camera opened successfully.")
print("Bicep Curl Detection Started.")
print("Voice feedback enabled.")
print("Press ESC to exit.")


# --------------------------------------------------
# 6. START MEDIAPIPE DETECTION
# --------------------------------------------------

try:

    with PoseLandmarker.create_from_options(options) as landmarker, \
            LOG.open("w", newline="") as output:

        writer = csv.writer(output)

        writer.writerow([
            "elapsed_ms",
            "side",
            "elbow_angle_deg",
            "phase",
            "reps",
            "form_error",
            "valid_pose"
        ])

        while True:

            # --------------------------------------
            # READ CAMERA FRAME
            # --------------------------------------

            success, frame = cap.read()

            if not success:
                print("Could not read camera frame.")
                break

            # --------------------------------------
            # CONVERT BGR TO RGB
            # --------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # --------------------------------------
            # GENERATE TIMESTAMP
            # --------------------------------------

            timestamp_ms = max(
                last_timestamp + 1,
                int(
                    (time.monotonic() - start_time) * 1000
                )
            )

            last_timestamp = timestamp_ms

            # --------------------------------------
            # DETECT BODY LANDMARKS
            # --------------------------------------

            detection = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            angle = None

            if detection.pose_landmarks:

                landmarks = detection.pose_landmarks[0]

                # Right-arm landmark indices
                shoulder = landmarks[12]
                elbow = landmarks[14]
                wrist = landmarks[16]

                # Check if the arm is visible.
                if all(
                    getattr(point, "visibility", 1.0) > 0.5
                    for point in (shoulder, elbow, wrist)
                ):

                    # Calculate elbow angle.
                    angle = joint_angle(
                        shoulder,
                        elbow,
                        wrist
                    )

                    # --------------------------------
                    # DRAW ARM LANDMARKS
                    # --------------------------------

                    height, width = frame.shape[:2]

                    shoulder_point = (
                        int(shoulder.x * width),
                        int(shoulder.y * height)
                    )

                    elbow_point = (
                        int(elbow.x * width),
                        int(elbow.y * height)
                    )

                    wrist_point = (
                        int(wrist.x * width),
                        int(wrist.y * height)
                    )

                    cv2.line(
                        frame,
                        shoulder_point,
                        elbow_point,
                        (0, 255, 0),
                        3
                    )

                    cv2.line(
                        frame,
                        elbow_point,
                        wrist_point,
                        (0, 255, 0),
                        3
                    )

                    for point in (
                        shoulder_point,
                        elbow_point,
                        wrist_point
                    ):

                        cv2.circle(
                            frame,
                            point,
                            6,
                            (0, 255, 0),
                            -1
                        )

            # --------------------------------------
            # ANALYZE BICEP CURL
            # --------------------------------------

            result = analyzer.update_angle(angle)

            # --------------------------------------
            # VOICE FEEDBACK
            # --------------------------------------

            # Speak only when a new repetition is counted.
            if result.reps > last_rep_count:

                last_rep_count = result.reps

                speak(
                    f"Rep {result.reps} completed."
                )

            # --------------------------------------
            # DISPLAY REPETITION COUNT
            # --------------------------------------

            cv2.putText(
                frame,
                f"BICEP CURLS | REPS: {result.reps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # --------------------------------------
            # DISPLAY ELBOW ANGLE
            # --------------------------------------

            if result.angle is not None:

                angle_text = (
                    f"Elbow Angle: {result.angle:.1f} deg"
                )

            else:

                angle_text = "Elbow Angle: Unavailable"

            cv2.putText(
                frame,
                angle_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # --------------------------------------
            # DISPLAY EXERCISE PHASE
            # --------------------------------------

            cv2.putText(
                frame,
                f"Phase: {result.phase}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # --------------------------------------
            # DISPLAY FORM FEEDBACK
            # --------------------------------------

            if result.angle is None:

                feedback = "Position arm in camera"

            elif result.form_error == "upper_arm_swing":

                feedback = "Keep upper arm steady"

            else:

                feedback = "Tracking curl"

            cv2.putText(
                frame,
                f"Feedback: {feedback}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # --------------------------------------
            # SAVE RESULTS TO CSV
            # --------------------------------------

            writer.writerow([
                timestamp_ms,
                "right",
                (
                    round(result.angle, 2)
                    if result.angle is not None
                    else ""
                ),
                result.phase,
                result.reps,
                result.form_error or "",
                result.angle is not None
            ])

            # --------------------------------------
            # DISPLAY CAMERA WINDOW
            # --------------------------------------

            cv2.imshow(
                "Senior Design - Bicep Curl Checker",
                frame
            )

            # Press ESC to exit.
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    # Stop an active voice announcement.
    if voice_process is not None:
        if voice_process.poll() is None:
            voice_process.terminate()

    print("Bicep Curl Detection Stopped.")
    print(f"Results saved to: {LOG}")
