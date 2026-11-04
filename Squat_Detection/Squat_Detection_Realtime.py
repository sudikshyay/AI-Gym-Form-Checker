
import cv2
import mediapipe as mp
import math
import time
import csv
import os
import threading
import subprocess
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# SETTING

MODEL_PATH = "pose_landmarker_full.task"

STANDING_ANGLE = 160
BOTTOM_ANGLE = 95
SET_TARGET = 10

INITIAL_STANDING_FRAMES = 10
STANDING_CONFIRM_FRAMES = 3
COUNTDOWN_SECONDS = 3

SHALLOW_WARNING_MIN_BEND = 20
ASCENDING_CONFIRM_FRAMES = 3

# Real-time coaching settings
COACHING_ZONE_MIN = 100
COACHING_ZONE_MAX = 125

# Degrees per second: movement slower than this is
# considered slow enough for a coaching cue.
SLOW_DESCENT_SPEED = 35

# Number of consecutive frames needed to confirm
# a movement direction or coaching condition.
COACH_CONFIRM_FRAMES = 3
RISE_CONFIRM_FRAMES = 3

# Avoid excessive speech.
COACH_COOLDOWN_SECONDS = 1.5

AUDIO_ENABLED = True


# WINDOWS VOICE FEEDBACK
class AudioFeedback:
    """
    Windows text-to-speech using a background worker.

    Uses one PowerShell process for the entire session.
    Short-lived coaching messages can be cancelled before
    they are spoken if the user's movement has changed.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.closed = False
        self.condition = threading.Condition()
        self.pending = None
        self.generation = 0
        self.thread = None

        if enabled:
            self.thread = threading.Thread(
                target=self._worker,
                daemon=True
            )
            self.thread.start()

    def say(self, message, kind="general", valid_for=1.5):
        if not self.enabled or self.closed:
            return

        with self.condition:
            self.generation += 1

            self.pending = {
                "text": str(message),
                "kind": kind,
                "expires": time.monotonic() + valid_for,
                "generation": self.generation
            }

            self.condition.notify()

    def cancel_coaching(self):
        if not self.enabled:
            return

        with self.condition:
            if (
                self.pending is not None
                and self.pending["kind"] == "coaching"
            ):
                self.pending = None
                self.generation += 1

    def _worker(self):
        # A persistent PowerShell process avoids launching
        # a separate process for every spoken message.
        powershell_script = r"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 1

try {
    while ($null -ne ($line = [Console]::ReadLine())) {
        if ($line -eq "__EXIT__") {
            break
        }

        try {
            $bytes = [Convert]::FromBase64String($line)
            $message = [Text.Encoding]::UTF8.GetString($bytes)
            $speaker.Speak($message)
        } catch {
            [Console]::Error.WriteLine($_.Exception.Message)
        }
    }
} finally {
    $speaker.Dispose()
}
"""

        process = None

        try:
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    powershell_script
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )

            import base64

            while True:
                with self.condition:
                    while self.pending is None and not self.closed:
                        self.condition.wait()

                    if self.closed:
                        break

                    item = self.pending
                    self.pending = None

                if time.monotonic() > item["expires"]:
                    continue

                if process.poll() is not None:
                    print("Voice process stopped.")
                    break

                encoded = base64.b64encode(
                    item["text"].encode("utf-8")
                ).decode("ascii")

                try:
                    process.stdin.write(encoded + "\n")
                    process.stdin.flush()
                    print("Speaking:", item["text"])
                except (BrokenPipeError, OSError):
                    print("Could not send voice message.")
                    break

        except Exception as error:
            print("Voice system error:", error)

        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.stdin.write("__EXIT__\n")
                        process.stdin.flush()
                        process.wait(timeout=2)
                except Exception:
                    process.terminate()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()

        if self.thread is not None:
            self.thread.join(timeout=3)



# ANGLE CALCULATIONS

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def calculate_torso_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    return math.degrees(
        math.atan2(abs(dx), abs(dy))
    )


def is_visible(point):
    return getattr(point, "visibility", 1.0) >= 0.5


def format_value(value):
    return f"{value:.1f}" if value is not None else "N/A"


def csv_value(value):
    return round(value, 2) if value is not None else ""



# REAL-TIME SQUAT DETECTION

class SquatPhaseDetector:

    def __init__(self, audio):
        self.audio = audio

        self.squat_count = 0
        self.shallow_count = 0

        self.phase = "WAITING FOR STANDING"
        self.feedback = "Stand still to begin"
        self.feedback_time = time.monotonic()

        self.ready = False
        self.initial_standing_frames = 0

        self.countdown_start = None
        self.last_countdown_number = None
        self.countdown_finished = False
        self.start_message_time = None

        self.previous_angle = None
        self.previous_time = None
        self.smoothed_speed = 0.0

        self.movement_started = False
        self.reached_bottom = False
        self.started_ascending = False

        self.minimum_angle = 180
        self.standing_frames = 0
        self.rising_frames = 0
        self.coaching_frames = 0

        self.coached_this_rep = False
        self.shallow_warning_given = False
        self.last_coaching_time = -100.0

        self.set_complete_announced = False

    def set_feedback(self, message):
        self.feedback = message
        self.feedback_time = time.monotonic()
        print("FEEDBACK:", message)

    def reset_rep(self):
        self.movement_started = False
        self.reached_bottom = False
        self.started_ascending = False

        self.minimum_angle = 180
        self.standing_frames = 0
        self.rising_frames = 0
        self.coaching_frames = 0

        self.coached_this_rep = False
        self.shallow_warning_given = False

        self.audio.cancel_coaching()

    def update(self, angle):
        now = time.monotonic()

        # WAIT FOR INITIAL STANDING POSITION

        if not self.ready:
            if angle is not None and angle >= STANDING_ANGLE:
                self.initial_standing_frames += 1
            else:
                self.initial_standing_frames = 0

            if self.initial_standing_frames >= INITIAL_STANDING_FRAMES:
                self.ready = True
                self.countdown_start = now
                self.phase = "GET READY"
                self.audio.say(
                    "Get ready",
                    valid_for=2.0
                )
                self.set_feedback("Standing confirmed")

            return self.phase

      
        # COUNTDOWN

        if not self.countdown_finished:
            if angle is None or angle < STANDING_ANGLE:
                self.ready = False
                self.initial_standing_frames = 0
                self.countdown_start = None
                self.last_countdown_number = None
                self.phase = "WAITING FOR STANDING"
                self.set_feedback("Stand still to begin")
                return self.phase

            elapsed = now - self.countdown_start

            if elapsed < COUNTDOWN_SECONDS:
                remaining = COUNTDOWN_SECONDS - int(elapsed)
                self.phase = f"GET READY: {remaining}"

                if remaining != self.last_countdown_number:
                    self.last_countdown_number = remaining
                    self.audio.say(
                        str(remaining),
                        valid_for=0.9
                    )

                return self.phase

            self.countdown_finished = True
            self.start_message_time = now
            self.phase = "START!"
            self.previous_angle = angle
            self.previous_time = now

            self.audio.say("Start", valid_for=1.0)
            self.set_feedback("Begin your squats")
            return self.phase


        # HANDLE MISSING LANDMARKS

        if angle is None:
            self.previous_angle = None
            self.previous_time = None
            self.audio.cancel_coaching()
            return self.phase

       
        # CALCULATE MOVEMENT SPEED-

        speed = 0.0

        if (
            self.previous_angle is not None
            and self.previous_time is not None
        ):
            dt = now - self.previous_time

            if dt > 0.001:
                speed = (angle - self.previous_angle) / dt

        # Negative speed = descending
        # Positive speed = ascending

        self.smoothed_speed = (
            0.65 * self.smoothed_speed
            + 0.35 * speed
        )

        self.previous_angle = angle
        self.previous_time = now

     
        # DETECT PHASE
        if angle >= STANDING_ANGLE:
            self.phase = "STANDING"

        elif angle <= BOTTOM_ANGLE:
            self.phase = "BOTTOM"

        elif self.smoothed_speed < -8:
            self.phase = "GOING DOWN"

        elif self.smoothed_speed > 8:
            self.phase = "GOING UP"

        else:
            self.phase = "MOVING / TRANSITION"

      
        # TRACK SQUAT MOVEMENT

        if angle < STANDING_ANGLE - 5:
            self.movement_started = True

        if self.movement_started:
            self.minimum_angle = min(
                self.minimum_angle,
                angle
            )

     
        # FULL DEPTH REACHED

        if angle <= BOTTOM_ANGLE:

            if not self.reached_bottom:
                self.reached_bottom = True

                self.audio.cancel_coaching()
                self.set_feedback("Good depth!")

                self.audio.say(
                    "Good!",
                    kind="success",
                    valid_for=0.8
                )

            self.coaching_frames = 0
            self.rising_frames = 0
            self.standing_frames = 0


        # DETECT ASCENDING DIRECTION

        elif (
            self.movement_started
            and self.smoothed_speed > 12
            and angle < STANDING_ANGLE
        ):
            self.rising_frames += 1
            self.coaching_frames = 0

            if self.rising_frames >= RISE_CONFIRM_FRAMES:
                self.audio.cancel_coaching()

                if self.reached_bottom:
                    self.started_ascending = True

                elif not self.shallow_warning_given:
                    bent_enough = (
                        self.minimum_angle
                        <= STANDING_ANGLE - SHALLOW_WARNING_MIN_BEND
                    )

                    if bent_enough:
                        self.shallow_warning_given = True
                        self.set_feedback(
                            "Shallow squat - lower next rep"
                        )

                        self.audio.say(
                            "Lower next rep",
                            kind="correction",
                            valid_for=1.0
                        )

     
        # REAL-TIME DEPTH COACHING

        elif (
            self.movement_started
            and not self.reached_bottom
            and not self.started_ascending
            and not self.shallow_warning_given
            and not self.coached_this_rep
            and COACHING_ZONE_MIN <= angle <= COACHING_ZONE_MAX
            and self.smoothed_speed < 0
            and abs(self.smoothed_speed) < SLOW_DESCENT_SPEED
        ):
            self.coaching_frames += 1
            self.rising_frames = 0

            if (
                self.coaching_frames >= COACH_CONFIRM_FRAMES
                and now - self.last_coaching_time
                >= COACH_COOLDOWN_SECONDS
            ):
                self.coached_this_rep = True
                self.last_coaching_time = now

                self.set_feedback(
                    "A little lower, if comfortable"
                )

                self.audio.say(
                    "A little lower",
                    kind="coaching",
                    valid_for=0.55
                )

        else:
            self.coaching_frames = 0

            if self.smoothed_speed < -8:
                self.rising_frames = 0

     
        # COUNT REP WHEN USER RETURNS TO STANDING

        if angle >= STANDING_ANGLE and self.movement_started:
            self.standing_frames += 1

            if self.standing_frames >= STANDING_CONFIRM_FRAMES:

                if self.reached_bottom and self.started_ascending:
                    self.squat_count += 1

                    self.set_feedback(
                        f"Squat {self.squat_count} completed"
                    )

                    self.audio.say(
                        str(self.squat_count),
                        kind="count",
                        valid_for=1.0
                    )

                    print(
                        "Completed squat:",
                        self.squat_count
                    )

                    if (
                        self.squat_count >= SET_TARGET
                        and not self.set_complete_announced
                    ):
                        self.set_complete_announced = True
                        self.set_feedback("Set complete!")

                        self.audio.say(
                            "Great job! Set completed",
                            kind="completion",
                            valid_for=3.0
                        )

                else:
                    bent_enough = (
                        self.minimum_angle
                        <= STANDING_ANGLE - SHALLOW_WARNING_MIN_BEND
                    )

                    if bent_enough:
                        self.shallow_count += 1

                        print(
                            "Shallow squat:",
                            self.shallow_count
                        )

                        self.set_feedback(
                            "Shallow squat detected"
                        )

                        if not self.shallow_warning_given:
                            self.audio.say(
                                "Lower next rep",
                                kind="correction",
                                valid_for=1.0
                            )

                self.reset_rep()

        elif angle < STANDING_ANGLE:
            self.standing_frames = 0

        return self.phase



# CAMERA SETUP

def open_camera():
    for index in [1, 0, 2, 3]:
        print(f"Trying camera {index}...")

        camera = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        for attempt in range(8):
            success, frame = camera.read()

            if success and frame is not None:
                if frame.mean() >= 5:
                    print(f"Using camera {index}")
                    return camera

        camera.release()

    raise RuntimeError(
        "Could not find a working webcam. "
        "Check the USB connection and privacy cover."
    )



# MAIN PROGRAM
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}. "
            "Put pose_landmarker_full.task "
            "in the same working folder."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    detector = vision.PoseLandmarker.create_from_options(
        options
    )

    camera = None
    audio = None
    csv_file = None

    try:
        camera = open_camera()
        audio = AudioFeedback(AUDIO_ENABLED)
        squat = SquatPhaseDetector(audio)

        filename = (
            "squat_measurements_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        csv_file = open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        )

        writer = csv.writer(csv_file)

        writer.writerow([
            "Time_Seconds",
            "Left_Knee",
            "Right_Knee",
            "Average_Knee",
            "Left_Hip",
            "Right_Hip",
            "Average_Hip",
            "Torso_Lean",
            "Phase",
            "Completed_Squats",
            "Shallow_Squats",
            "Feedback"
        ])

        start_time = time.monotonic()
        last_timestamp = -1

        print("Webcam opened successfully.")
        print("Press ESC to exit.")

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("ERROR: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp = max(
                timestamp,
                last_timestamp + 1
            )

            last_timestamp = timestamp

            result = detector.detect_for_video(
                mp_image,
                timestamp
            )

            measurements = {
                "Left Knee": None,
                "Right Knee": None,
                "Average Knee": None,
                "Left Hip": None,
                "Right Hip": None,
                "Average Hip": None,
                "Torso Lean": None
            }

            if result.pose_landmarks:
                points = result.pose_landmarks[0]

                def point(index):
                    p = points[index]
                    return (p.x, p.y)

                def joint_angle(a, b, c):
                    if not all(
                        is_visible(points[i])
                        for i in (a, b, c)
                    ):
                        return None

                    return calculate_angle(
                        point(a),
                        point(b),
                        point(c)
                    )

                measurements["Left Knee"] = joint_angle(
                    23, 25, 27
                )

                measurements["Right Knee"] = joint_angle(
                    24, 26, 28
                )

                measurements["Left Hip"] = joint_angle(
                    11, 23, 25
                )

                measurements["Right Hip"] = joint_angle(
                    12, 24, 26
                )

                def average(*values):
                    valid = [
                        value for value in values
                        if value is not None
                    ]
                    return (
                        sum(valid) / len(valid)
                        if valid else None
                    )

                measurements["Average Knee"] = average(
                    measurements["Left Knee"],
                    measurements["Right Knee"]
                )

                measurements["Average Hip"] = average(
                    measurements["Left Hip"],
                    measurements["Right Hip"]
                )

                torso_values = []

                for shoulder, hip in [(11, 23), (12, 24)]:
                    if (
                        is_visible(points[shoulder])
                        and is_visible(points[hip])
                    ):
                        torso_values.append(
                            calculate_torso_angle(
                                (
                                    points[shoulder].x * width,
                                    points[shoulder].y * height
                                ),
                                (
                                    points[hip].x * width,
                                    points[hip].y * height
                                )
                            )
                        )

                measurements["Torso Lean"] = average(
                    *torso_values
                )

                # Draw left and right leg/torso connections
                connections = [
                    (11, 23), (23, 25), (25, 27),
                    (12, 24), (24, 26), (26, 28)
                ]

                for a, b in connections:
                    if not (
                        is_visible(points[a])
                        and is_visible(points[b])
                    ):
                        continue

                    p1 = (
                        int(points[a].x * width),
                        int(points[a].y * height)
                    )

                    p2 = (
                        int(points[b].x * width),
                        int(points[b].y * height)
                    )

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (0, 255, 0),
                        3
                    )

                for index in [11, 12, 23, 24, 25, 26, 27, 28]:
                    if not is_visible(points[index]):
                        continue

                    center = (
                        int(points[index].x * width),
                        int(points[index].y * height)
                    )

                    cv2.circle(
                        frame,
                        center,
                        6,
                        (255, 0, 0),
                        -1
                    )

            # Update squat phase and feedback
            phase = squat.update(
                measurements["Average Knee"]
            )

      
            # DISPLAY MEASUREMENTS

            display_items = [
                ("Left Knee", (0, 255, 0)),
                ("Right Knee", (255, 0, 0)),
                ("Average Knee", (255, 255, 255)),
                ("Left Hip", (0, 255, 0)),
                ("Right Hip", (255, 0, 0)),
                ("Average Hip", (255, 255, 255)),
                ("Torso Lean", (0, 255, 255))
            ]

            for i, (label, color) in enumerate(display_items):
                value = measurements[label]

                cv2.putText(
                    frame,
                    f"{label}: {format_value(value)}",
                    (20, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            cv2.putText(
                frame,
                f"Phase: {phase}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Completed: {squat.squat_count}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Shallow: {squat.shallow_count}",
                (20, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame,
                squat.feedback,
                (20, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "AI Gym Posture Checker - Real-Time Coaching",
                frame
            )

            elapsed = time.monotonic() - start_time

            writer.writerow([
                round(elapsed, 3),
                csv_value(measurements["Left Knee"]),
                csv_value(measurements["Right Knee"]),
                csv_value(measurements["Average Knee"]),
                csv_value(measurements["Left Hip"]),
                csv_value(measurements["Right Hip"]),
                csv_value(measurements["Average Hip"]),
                csv_value(measurements["Torso Lean"]),
                phase,
                squat.squat_count,
                squat.shallow_count,
                squat.feedback
            ])

            if cv2.waitKey(1) & 0xFF == 27:
                break

        print("Completed squats:", squat.squat_count)
        print("Shallow squats:", squat.shallow_count)
        print("CSV saved:", filename)

    finally:
        if camera is not None:
            camera.release()

        detector.close()

        if csv_file is not None:
            csv_file.close()

        cv2.destroyAllWindows()

        if audio is not None:
            audio.close()


if __name__ == "__main__":
    main()
