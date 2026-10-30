# AI-Gym-Form-Checker

# Project Overview
The AI Gym Posture Checker is a senior design project that uses computer vision to analyze exercise movements and provide real-time feedback. The goal is to help users improve their exercise form using a webcam.

# Technologies Used
- Python
- OpenCV
- MediaPipe Pose Landmarker
  
# Current Progress
- Implemented webcam-based pose detection.
- Implemented left and right knee-angle calculations.
- Developed basic squat movement phase detection (standing, going down, bottom, and going up).
- Added CSV logging to record knee angles and squat phases.
- Tested squat detection using multiple repetitions.
  
# How to Run
1. Install Python and connect a webcam.
2. Install the required libraries using python -m pip install -r requirements.txt.
3. Download the MediaPipe Pose Landmarker Full model and save it as pose_landmarker_full.task in the project folder.
4. Run python squat_phase_detection.py.
5. Stand where your body is visible to the camera and perform squats.
6. Press ESC to close the program.

# Next Steps
- Implement automatic squat repetition counting.
- Improve squat-phase detection reliability.
- Develop exercise form-error detection.
- Add real-time corrective feedback.
- Test system performance and explore Raspberry Pi integration.
  
# Project Status
Development and testing are ongoing
