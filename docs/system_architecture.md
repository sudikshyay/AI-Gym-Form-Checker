# AI Gym Form Checker — System Architecture

## Project Overview

The AI Gym Form Checker is a computer-vision-based exercise monitoring system intended to analyze body movements, count exercise repetitions, and provide real-time feedback.

The selected exercises are squats, lunges, and bicep curls.

## Proposed Processing Pipeline

1. **Camera Input:** Captures video of the user performing an exercise.
2. **Pose Estimation:** MediaPipe Pose Landmarker detects body landmarks.
3. **Joint-Angle Calculation:** Relevant joint angles are calculated from landmark coordinates.
4. **Exercise Analysis:** Exercise-specific algorithms determine movement phases and count repetitions.
5. **Feedback System:** Displays exercise metrics and generates feedback messages or voice announcements.
6. **Data Logging:** Records measurements and detection results in CSV files.
7. **Hardware Deployment:** Integrates the system with Raspberry Pi hardware.

## Current Implementation Status

| Component                     | Status |
| Webcam-based pose detection   | Implemented in Python |
| Squat knee-angle calculation  | Implemented |
| Squat phase detection         | Implemented |
| Bicep curl elbow-angle calculation | Implemented |
| Bicep curl repetition-counting logic | Implemented |
| Bicep curl voice rep announcements | Implemented in code; live validation pending |
| Lunge analysis | Development status to be confirmed |
| Form-error detection | In development |
| Raspberry Pi integration | Hardware test status to be confirmed |
| Complete system integration | Pending |

## Next Development Steps

The team will focus on completing exercise-specific analysis, validating repetition-counting accuracy, implementing meaningful corrective feedback, integrating the modules into a shared interface, and deploying the prototype on Raspberry Pi hardware.