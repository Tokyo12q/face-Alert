# ─────────────────────────────────────────────────────────────────────────────
# src/gui.py  –  Main Application Window (Dashboard)
# Purpose : Defines ProctorDashboard, the top-level Tkinter window.
#           It owns ALL logic components and drives the frame-by-frame
#           proctoring loop via Tkinter's after() scheduler.
#
# Architecture overview
# ─────────────────────
#  ProctorDashboard (CTk window)
#   ├── ProctorDetector    – MediaPipe face/landmark detection
#   ├── GazeTracker        – pupil centre → gaze direction
#   ├── IntegrityManager   – score + calibration state machine
#   ├── ProctorNotifier    – audible alert beep (with cooldown)
#   ├── ProctorRecorder    – SQLite DB + screenshot evidence
#   └── update_frame()     – called every 10 ms via self.after()
# ─────────────────────────────────────────────────────────────────────────────

import customtkinter as ctk   # Modern dark-mode Tkinter widgets.
import cv2                    # OpenCV – webcam capture and image flipping.
import numpy as np            # NumPy – blank eye-view placeholder array.
from PIL import Image, ImageTk  # Pillow – frame conversion (used via ProctorUI).

# Import all logic/rendering classes from the src package.
from src.detector  import ProctorDetector
from src.processor import ProctorProcessor
from src.tracker   import GazeTracker, IntegrityManager
from src.ui        import ProctorUI
from src.notifier  import ProctorNotifier
from src.recorder  import ProctorRecorder


class ProctorDashboard(ctk.CTk):
    """
    The main application window.  Inherits from ctk.CTk (CustomTkinter's
    root window class, itself a subclass of tkinter.Tk).
    """

    # ─── Initialisation ───────────────────────────────────────────────────────

    def __init__(self):
        """Set up the window, initialise all logic components, build the UI."""

        # Call CTk.__init__() to create the underlying Tk window object.
        # This MUST be called before any other CTk/Tk operations.
        super().__init__()

        # ── Window properties ────────────────────────────────────────────────

        # Set the text shown in the window's title bar.
        self.title("Smart Proctoring System Dashboard")

        # Set the initial window size in pixels (width × height).
        self.geometry("1100x700")

        # Switch CustomTkinter to dark mode (dark background, light text).
        ctk.set_appearance_mode("dark")

        # Use the built-in "blue" accent colour theme for buttons and highlights.
        ctk.set_default_color_theme("blue")

        # ── Logic Components ─────────────────────────────────────────────────

        # Face & landmark detector (loads the MediaPipe model once here).
        self.detector = ProctorDetector()

        # Pupil-centre tracker; default 15-pixel deviation threshold.
        self.gaze_tracker = GazeTracker()

        # Integrity score manager:
        #   suspicion_threshold=3.0  → 3 s of continuous look-aside = VIOLATION.
        #   auto_calibrate_duration=5.0 → 5 s calibration window.
        self.integrity_manager = IntegrityManager(
            suspicion_threshold=3.0,
            auto_calibrate_duration=5.0
        )

        # Alert sound manager; 1-second cooldown between beeps.
        self.notifier = ProctorNotifier(cooldown_seconds=1)

        # Database + screenshot evidence manager.
        self.recorder = ProctorRecorder()

        # VideoCapture object; starts as None until start_system() is called.
        self.cap = None

        # Flag that controls whether update_frame() keeps rescheduling itself.
        self.is_running = False

        # ── Build the UI ─────────────────────────────────────────────────────

        # Delegate all widget creation to a private helper method.
        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        """Create and layout all GUI widgets."""

        # ── Grid Configuration ───────────────────────────────────────────────

        # Column 1 (main video area) expands when the window is resized.
        # Column 0 (sidebar) keeps its fixed width.
        self.grid_columnconfigure(1, weight=1)

        # Row 0 expands vertically to fill the window height.
        self.grid_rowconfigure(0, weight=1)

        # ── 1. Sidebar Panel ─────────────────────────────────────────────────

        # Create a fixed-width CTkFrame for the left sidebar.
        # corner_radius=0 makes it flush to the window edges.
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)

        # Place the sidebar in grid cell (row=0, col=0), stretching to fill
        # all four sides of the cell (North, South, East, West).
        self.sidebar.grid(row=0, column=0, sticky="nsew")



        # ── Start button ─────────────────────────────────────────────────────
        # Clicking "Start System" calls self.start_system().
        self.btn_start = ctk.CTkButton(
            self.sidebar, text="Start System", command=self.start_system
        )
        self.btn_start.pack(pady=10, padx=20)  # padx adds horizontal margin

        # ── Stop button ──────────────────────────────────────────────────────
        # state="disabled" means the button is greyed out and unclickable
        # until the system is running.
        self.btn_stop = ctk.CTkButton(
            self.sidebar, text="Stop System",
            command=self.stop_system, state="disabled"
        )
        self.btn_stop.pack(pady=10, padx=20)

        # ── Recalibrate button ───────────────────────────────────────────────
        # Restarts the auto-calibration phase without stopping the camera.
        self.btn_calibrate = ctk.CTkButton(
            self.sidebar, text="Recalibrate",
            command=self.recalibrate, state="disabled"
        )
        self.btn_calibrate.pack(pady=10, padx=20)

        # ── Integrity score label ────────────────────────────────────────────
        # Displays the current score; updated every frame by update_frame().
        self.score_label = ctk.CTkLabel(
            self.sidebar,
            text="Integrity Score: 100",
            font=ctk.CTkFont(size=14)
        )
        self.score_label.pack(pady=20)

        # ── Status label ─────────────────────────────────────────────────────
        # Shows the current proctoring status string.
        # wraplength=180 wraps long status strings to fit inside the sidebar.
        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Status: IDLE", wraplength=180
        )
        self.status_label.pack(pady=10)

        # ── 2. Main Video Feed Area ───────────────────────────────────────────

        # A container frame for the main camera feed.
        self.main_feed_frame = ctk.CTkFrame(self)

        # Place it in column 1 with padding; sticky="nsew" makes it fill the cell.
        self.main_feed_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # A label widget used as a canvas to display the video PhotoImage.
        # text="" means no text; the image replaces text.
        self.video_label = ctk.CTkLabel(self.main_feed_frame, text="")

        # expand=True + fill="both" makes the label fill the container frame.
        self.video_label.pack(expand=True, fill="both")

        # ── 3. Small Eye Tracking Monitor (bottom of sidebar) ─────────────────

        # Title label above the eye monitor panel.
        self.eye_monitor_label = ctk.CTkLabel(
            self.sidebar,
            text="Eye Tracking Monitor",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        # side="bottom" stacks this below the previous pack() widgets.
        self.eye_monitor_label.pack(side="bottom", pady=(10, 5))

        # Small black placeholder label that will display the binary eye image.
        # width=150, height=100 sets the initial dimensions.
        # fg_color="black" fills the background before any frame is shown.
        self.eye_video_label = ctk.CTkLabel(
            self.sidebar, text="",
            width=150, height=100, fg_color="black"
        )
        self.eye_video_label.pack(side="bottom", pady=(0, 20), padx=25)

    # ─── System Control ───────────────────────────────────────────────────────

    def start_system(self):
        """Open the webcam and begin the frame-update loop."""

        # cv2.VideoCapture(0) opens the first (default) webcam.
        # Index 0 → built-in laptop camera; 1+ → external USB cameras.
        self.cap = cv2.VideoCapture(0)

        # isOpened() returns False if the camera could not be accessed
        # (e.g. no webcam, camera in use by another app).
        if not self.cap.isOpened():
            self.status_label.configure(text="Error: Camera not found")
            return  # Abort startup gracefully.

        # Mark the system as running so update_frame() keeps scheduling itself.
        self.is_running = True

        # Swap button states: disable Start, enable Stop & Recalibrate.
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_calibrate.configure(state="normal")

        # Kick off the recurring frame update loop (calls itself every 10 ms).
        self.update_frame()

    def stop_system(self):
        """Stop the frame loop, release the webcam, and reset the UI."""

        # Setting is_running=False causes update_frame() to return early
        # on its next invocation, effectively stopping the loop.
        self.is_running = False

        # Release the VideoCapture object so the webcam is no longer occupied.
        if self.cap:
            self.cap.release()

        # Restore button states: Start re-enabled, Stop & Calibrate disabled.
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_calibrate.configure(state="disabled")

        # Clear the video display widgets (show nothing instead of last frame).
        self.video_label.configure(image="")
        self.eye_video_label.configure(image="")

        # Reset the status text.
        self.status_label.configure(text="Status: IDLE")

        # Print the session summary report to the console via the recorder.
        self.recorder.generate_summary()

    def recalibrate(self):
        """Reset calibration state so the system re-learns the baseline gaze."""

        # Mark the IntegrityManager as not calibrated so handle_calibration()
        # will start collecting samples again.
        self.integrity_manager.is_calibrated = False

        # Reset the calibration timer to 'now' so a fresh window begins.
        self.integrity_manager.start_time = time.time()

        # Clear any previously collected pupil-X samples.
        self.integrity_manager.calibration_samples = []

        # Update the status label to inform the operator.
        self.status_label.configure(text="Status: RECALIBRATING")

    # ─── Frame Update Loop ────────────────────────────────────────────────────

    def update_frame(self):
        """
        Process one webcam frame: detect faces, analyse gaze, update UI.
        Reschedules itself every 10 ms using self.after() to achieve
        ~100 fps (in practice limited by detection speed, ~15-30 fps).
        """

        # Guard: if the system was stopped, exit immediately without
        # rescheduling.  This is what terminates the loop.
        if not self.is_running:
            return

        # Read one frame from the webcam.
        # ret  : bool – True if a frame was successfully captured.
        # frame: np.ndarray – the BGR image array.
        ret, frame = self.cap.read()

        # If the camera feed is lost (unplugged, end of file), stop gracefully.
        if not ret:
            self.stop_system()
            return

        # ── Flip the frame horizontally (mirror effect) ───────────────────────
        # cv2.flip(frame, 1) mirrors left↔right so the video feels natural
        # (like looking in a mirror) rather than reversed.
        frame = cv2.flip(frame, 1)

        # ── Face Detection ────────────────────────────────────────────────────

        # Run MediaPipe on the current frame; returns a list of face dicts.
        faces = self.detector.detect_faces(frame)

        # Select the primary (largest) face.  Returns None if no face found.
        primary_face = self.detector.get_primary_face(faces)

        # ── Default values (overwritten below if a face is found) ─────────────

        # Default status while we determine what is happening.
        status = self.integrity_manager.update(False)

        # Black 100×100 pixel placeholder for the eye monitor display.
        binary_eye_view = np.zeros((100, 100), dtype=np.uint8)

        # Empty list of eye bounding rectangles.
        eye_rects = []

        # ── Primary face processing branch ────────────────────────────────────

        if primary_face is not None:

            # Crop the eye regions from the frame using landmark positions.
            eye_rois = self.detector.get_eye_rois(frame, primary_face)

            # Extract just the (x, y, w, h) coordinates for overlay drawing.
            eye_rects = [roi['coords'] for roi in eye_rois]

            # ── Calibration sub-branch ────────────────────────────────────────
            if not self.integrity_manager.is_calibrated:

                # cx_to_calibrate holds the pupil X if found, else None.
                cx_to_calibrate = None

                if eye_rois:
                    # Take the first eye ROI (left eye by convention).
                    eye_roi_bgr = eye_rois[0]['roi']

                    # Convert the BGR crop to grayscale for processing.
                    eye_roi_gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)

                    # Run the 4-step image processing pipeline.
                    binary_eye_view = ProctorProcessor.process_eye_region(eye_roi_gray)

                    # Find the pupil centre in the binary image.
                    center = self.gaze_tracker.find_pupil_center(binary_eye_view)

                    if center:
                        # Extract just the horizontal component.
                        cx_to_calibrate = center[0]

                # Feed the pupil X into the calibration state machine.
                # Returns avg_cx (float) when calibration is done, False otherwise.
                calibration_result = self.integrity_manager.handle_calibration(cx_to_calibrate)

                # isinstance check: if a numeric value was returned, calibration
                # just completed – pass the average X to the gaze tracker.
                if isinstance(calibration_result, (float, int)):
                    self.gaze_tracker.calibrate(calibration_result)

                # Override the status text during calibration.
                status = "AUTO-CALIBRATING..."

            # ── Live monitoring sub-branch (calibration complete) ─────────────
            elif eye_rois:

                # Use the first eye ROI for pupil tracking.
                eye_roi_bgr  = eye_rois[0]['roi']
                eye_roi_gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)

                # Process the eye region into a clean binary image.
                binary_eye_view = ProctorProcessor.process_eye_region(eye_roi_gray)

                # Locate the pupil centre in the binary image.
                center = self.gaze_tracker.find_pupil_center(binary_eye_view)

                if center:
                    # check_behavior returns (label_string, is_suspicious_bool).
                    # We only need the boolean here.
                    _, is_suspicious = self.gaze_tracker.check_behavior(center[0])

                    # Update the integrity state machine with the suspicion flag.
                    status = self.integrity_manager.update(is_suspicious)

                    # If a violation was just declared, log it + play alert.
                    if "VIOLATION" in status:
                        self.recorder.log_event(
                            "VIOLATION", status,
                            self.integrity_manager.score, frame
                        )
                        self.notifier.play_alert_sound()

                else:
                    # Pupil not detected in the eye crop (e.g. blink, blur).
                    status = "SEARCHING PUPIL"
                    # Count as suspicious so prolonged lid-closure is penalised.
                    self.integrity_manager.update(True)

            else:
                # Eye ROIs were not extractable (face at extreme angle, etc.).
                status = "SEARCHING EYES"
                self.integrity_manager.update(True)

        # ── No primary face branch ────────────────────────────────────────────
        else:
            status = "FACE NOT DETECTED"

            # Sound the alert immediately when the face disappears.
            self.notifier.play_alert_sound()

            # Log a face-missing violation only after calibration is complete
            # (during calibration it is normal to briefly not detect a face).
            if self.integrity_manager.is_calibrated:
                self.recorder.log_event(
                    "VIOLATION", "Face missing",
                    self.integrity_manager.score, frame
                )
                # Count as suspicious to degrade the score.
                self.integrity_manager.update(True)

        # ── Multi-person detection ────────────────────────────────────────────

        # If more than one face is in the frame, override status immediately.
        if len(faces) > 1:
            status = "VIOLATION: MULTI-PERSON"
            self.recorder.log_event(
                "VIOLATION", "Multi-face",
                self.integrity_manager.score, frame
            )
            self.notifier.play_alert_sound()

        # ── Update sidebar widgets ────────────────────────────────────────────

        # Refresh the integrity score display.
        # int() truncates the float to a whole number for cleaner display.
        self.score_label.configure(
            text=f"Integrity Score: {int(self.integrity_manager.score)}"
        )

        # Refresh the status text.
        self.status_label.configure(text=f"Status: {status}")

        # ── Render main video feed ────────────────────────────────────────────

        # Draw all visual overlays (boxes, landmarks, status text) onto a copy
        # of the frame so the original is not permanently modified.
        display_frame = ProctorUI.draw_overlays(
            frame.copy(), faces, primary_face, eye_rects, status
        )

        # Convert the annotated BGR frame to a Tkinter PhotoImage (800×600).
        img_main = ProctorUI.convert_frame(display_frame, 800, 600)

        # Update the main video label with the new image.
        self.video_label.configure(image=img_main)

        # Keep a reference to prevent Python's garbage collector from deleting
        # the PhotoImage object (a common Tkinter pitfall).
        self.video_label.image = img_main

        # ── Render eye monitor ────────────────────────────────────────────────

        # Convert the binary eye image to a small (150×100) Tkinter image.
        img_eye = ProctorUI.convert_frame(binary_eye_view, 150, 100)

        # Update the sidebar eye monitor label.
        self.eye_video_label.configure(image=img_eye)

        # Again, hold a reference to prevent garbage collection.
        self.eye_video_label.image = img_eye

        # ── Schedule next frame ───────────────────────────────────────────────

        # self.after(ms, callback) asks Tkinter to call update_frame() again
        # after 10 milliseconds, creating a continuous loop.
        # This is non-blocking: Tkinter keeps processing events between calls.
        self.after(10, self.update_frame)

    # ─── Window Close Handler ─────────────────────────────────────────────────

    def on_closing(self):
        """
        Called when the user clicks the window's × button.
        Registered in main.py via app.protocol("WM_DELETE_WINDOW", …).
        """

        # Stop the camera and frame loop cleanly before closing.
        self.stop_system()

        # Destroy the Tk window and exit the mainloop().
        self.destroy()


# NOTE: 'import time' is placed at the bottom because only recalibrate()
# references it directly (the rest of the time-related logic lives in tracker.py
# which imports time itself).  Moving it to the top would be cleaner but this
# placement works and matches the original code structure.
import time  # For time tracking in recalibrate
