import cv2
import mediapipe as mp

# MediaPipe Tasks API

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Pose mode

MODEL_PATH = "pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Open webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully.")
print("Press ESC to exit.")

# Start MediaPipe Pose Landmarker


with PoseLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:

        # Read frame from webcam
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect pose
        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

      
        # Process landmarks
      

        if results.pose_landmarks:

            # Get the first detected person's landmarks
            landmarks = results.pose_landmarks[0]

            # Draw all 33 pose landmarks
      

            for landmark in landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

          
            # Get important landmarks
       

            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]

            # Print landmark coordinates
        

            print(
                f"Left Hip:   x={left_hip.x:.2f}  y={left_hip.y:.2f}"
            )

            print(
                f"Left Knee:  x={left_knee.x:.2f}  y={left_knee.y:.2f}"
            )

            print(
                f"Left Ankle: x={left_ankle.x:.2f}  y={left_ankle.y:.2f}"
            )

      
        # Show webcam
        cv2.imshow(
            "AI Gym Posture Checker - Pose Detection",
            frame
        )

        # Increase timestamp
        frame_timestamp_ms += 33

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break


# Close everything
cap.release()
cv2.destroyAllWindows()

print("Camera closed.")