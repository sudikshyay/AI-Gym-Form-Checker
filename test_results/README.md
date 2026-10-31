This folder contains CSV files from squat movement testing, including knee-angle measurements and detected squat phases.

## Bicep Curl Detection Module

The bicep curl module uses MediaPipe Pose Landmarker to detect shoulder, elbow, and wrist positions through a webcam.

### Features Implemented
- Real-time right-arm landmark tracking
- Elbow-angle calculation using geometric vector analysis
- Automatic repetition counting using angle thresholds
- Exercise-phase identification (waiting, extended, curled)
- Live visualization of elbow angle and repetition count
- Voice announcements for completed repetitions using macOS text-to-speech
- CSV logging for exercise data analysis
- Unit tests for angle calculation and repetition-counting behavior

### Running the Bicep Curl Module

From the main project directory:

```bash
source .venv/bin/activate
python bicep_curls/run_bicep_curls.py
```

The MediaPipe model file, `pose_landmarker_full.task`, must be located in the main project directory.

Press ESC to exit the application.

### Current Limitations

The prototype currently tracks the right arm. Upper-arm swing detection is not yet connected to live landmark measurements. The voice system announces repetitions but does not yet provide automatic corrective voice instructions.

Exercise recognition accuracy, performance measurements, and Raspberry Pi deployment remain to be validated.

### Module Development

Developed as an individual contribution on the `feature/bicep-curls` branch.