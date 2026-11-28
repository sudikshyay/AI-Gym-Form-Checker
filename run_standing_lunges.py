import cv2
import mediapipe as mp
import csv
import time

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from standing_lunges import StandingLungeAnalyzer, joint_angle

# Locate MediaPipe Model

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_FOLDER / "pose_landmarker_full.task"

if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_full.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Could not find pose_landmarker_full.task. "
        "Place it in the main project folder."
    )

# MediaPipe

BaseOptions = python.BaseOptions

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

# Initialize Lunge Analyzer

analyzer = StandingLungeAnalyzer()

# Open Webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Create CSV File

CSV_PATH = (
    Path(__file__).resolve().parent
    / "standing_lunge_results.csv"
)

csv_file = None

try:
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "Timestamp",
        "Left Knee Angle",
        "Right Knee Angle",
        "Phase",
        "Repetitions",
        "Feedback"
    ])

    # Start Pose Detection

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        frame_number = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Unable to read webcam.")
                break

            frame_number += 1

            # Flip image for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert image to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert to MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Process video frame
            timestamp = frame_number * 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            left_angle = None
            right_angle = None

            # Detect Body Landmarks

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                # Left leg landmarks
                left_hip = landmarks[23]
                left_knee = landmarks[25]
                left_ankle = landmarks[27]

                # Right leg landmarks
                right_hip = landmarks[24]
                right_knee = landmarks[26]
                right_ankle = landmarks[28]

                # Check landmark visibility
                left_visible = (
                    left_hip.visibility > 0.5
                    and left_knee.visibility > 0.5
                    and left_ankle.visibility > 0.5
                )

                right_visible = (
                    right_hip.visibility > 0.5
                    and right_knee.visibility > 0.5
                    and right_ankle.visibility > 0.5
                )

                # Calculate left knee angle
                if left_visible:

                    left_angle = joint_angle(
                        (left_hip.x, left_hip.y),
                        (left_knee.x, left_knee.y),
                        (left_ankle.x, left_ankle.y)
                    )

                # Calculate right knee angle
                if right_visible:

                    right_angle = joint_angle(
                        (right_hip.x, right_hip.y),
                        (right_knee.x, right_knee.y),
                        (right_ankle.x, right_ankle.y)
                    )

                # Draw Leg Landmarks

                height, width, _ = frame.shape

                for hip, knee, ankle in [
                    (left_hip, left_knee, left_ankle),
                    (right_hip, right_knee, right_ankle)
                ]:

                    points = [hip, knee, ankle]

                    for point in points:

                        x = int(point.x * width)
                        y = int(point.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            7,
                            (0, 255, 0),
                            -1
                        )

                    # Draw lines connecting landmarks
                    for first, second in [
                        (hip, knee),
                        (knee, ankle)
                    ]:

                        x1 = int(first.x * width)
                        y1 = int(first.y * height)

                        x2 = int(second.x * width)
                        y2 = int(second.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            3
                        )

            # Update Lunge Analyzer

            lunge_result = analyzer.update_angles(
                left_angle,
                right_angle
            )

            # Display Workout Information

            cv2.putText(
                frame,
                f"Reps: {lunge_result.reps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            left_text = (
                f"Left Knee: {left_angle:.1f}"
                if left_angle is not None
                else "Left Knee: --"
            )

            cv2.putText(
                frame,
                left_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            right_text = (
                f"Right Knee: {right_angle:.1f}"
                if right_angle is not None
                else "Right Knee: --"
            )

            cv2.putText(
                frame,
                right_text,
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Phase: {lunge_result.phase}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                lunge_result.feedback,
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Save Workout Data

            csv_writer.writerow([
                time.time(),
                left_angle,
                right_angle,
                lunge_result.phase,
                lunge_result.reps,
                lunge_result.feedback
            ])

            # Display Webcam

            cv2.imshow(
                "AI Gym Form Checker - Standing Lunges",
                frame
            )

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break


finally:

    # Close CSV file
    if csv_file is not None:
        csv_file.close()

    # Release webcam
    cap.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()

    print("Standing lunge program closed.")
