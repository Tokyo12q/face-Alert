import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from src.detector import ProctorDetector
from src.processor import ProctorProcessor
from src.tracker import GazeTracker, IntegrityManager
from src.ui import ProctorUI
from src.notifier import ProctorNotifier
from src.recorder import ProctorRecorder

class ProctorDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Smart Proctoring System Dashboard")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize Logic Components
        self.detector = ProctorDetector()
        self.gaze_tracker = GazeTracker()
        self.integrity_manager = IntegrityManager(suspicion_threshold=3.0, auto_calibrate_duration=5.0)
        self.notifier = ProctorNotifier(cooldown_seconds=1)
        self.recorder = ProctorRecorder()
        self.cap = None
        self.is_running = False

        self._build_ui()

    def _build_ui(self):
        # Configure Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="PROCTOR AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.btn_start = ctk.CTkButton(self.sidebar, text="Start System", command=self.start_system)
        self.btn_start.pack(pady=10, padx=20)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop System", command=self.stop_system, state="disabled")
        self.btn_stop.pack(pady=10, padx=20)

        self.btn_calibrate = ctk.CTkButton(self.sidebar, text="Recalibrate", command=self.recalibrate, state="disabled")
        self.btn_calibrate.pack(pady=10, padx=20)

        self.score_label = ctk.CTkLabel(self.sidebar, text="Integrity Score: 100", font=ctk.CTkFont(size=14))
        self.score_label.pack(pady=20)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: IDLE", wraplength=180)
        self.status_label.pack(pady=10)

        # 2. Main Video Feed
        self.main_feed_frame = ctk.CTkFrame(self)
        self.main_feed_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.main_feed_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # 3. Small Eye Monitor (In Sidebar or Corner)
        self.eye_monitor_label = ctk.CTkLabel(self.sidebar, text="Eye Tracking Monitor", font=ctk.CTkFont(size=12, weight="bold"))
        self.eye_monitor_label.pack(side="bottom", pady=(10, 5))
        
        self.eye_video_label = ctk.CTkLabel(self.sidebar, text="", width=150, height=100, fg_color="black")
        self.eye_video_label.pack(side="bottom", pady=(0, 20), padx=25)

    def start_system(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Error: Camera not found")
            return
            
        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_calibrate.configure(state="normal")
        self.update_frame()

    def stop_system(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_calibrate.configure(state="disabled")
        self.video_label.configure(image="")
        self.eye_video_label.configure(image="")
        self.status_label.configure(text="Status: IDLE")
        self.recorder.generate_summary()

    def recalibrate(self):
        self.integrity_manager.is_calibrated = False
        self.integrity_manager.start_time = time.time()
        self.integrity_manager.calibration_samples = []
        self.status_label.configure(text="Status: RECALIBRATING")

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_system()
            return

        frame = cv2.flip(frame, 1)
        faces = self.detector.detect_faces(frame)
        primary_face = self.detector.get_primary_face(faces)
        
        status = self.integrity_manager.update(False)
        binary_eye_view = np.zeros((100, 100), dtype=np.uint8)
        eye_rects = []

        if primary_face is not None:
            eye_rois = self.detector.get_eye_rois(frame, primary_face)
            eye_rects = [roi['coords'] for roi in eye_rois]
            
            if not self.integrity_manager.is_calibrated:
                cx_to_calibrate = None
                if eye_rois:
                    eye_roi_bgr = eye_rois[0]['roi']
                    eye_roi_gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
                    binary_eye_view = ProctorProcessor.process_eye_region(eye_roi_gray)
                    center = self.gaze_tracker.find_pupil_center(binary_eye_view)
                    if center:
                        cx_to_calibrate = center[0]
                
                calibration_result = self.integrity_manager.handle_calibration(cx_to_calibrate)
                if isinstance(calibration_result, (float, int)):
                    self.gaze_tracker.calibrate(calibration_result)
                status = "AUTO-CALIBRATING..."
            
            elif eye_rois:
                eye_roi_bgr = eye_rois[0]['roi']
                eye_roi_gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
                binary_eye_view = ProctorProcessor.process_eye_region(eye_roi_gray)
                center = self.gaze_tracker.find_pupil_center(binary_eye_view)
                
                if center:
                    _, is_suspicious = self.gaze_tracker.check_behavior(center[0])
                    status = self.integrity_manager.update(is_suspicious)
                    
                    if "VIOLATION" in status:
                        self.recorder.log_event("VIOLATION", status, self.integrity_manager.score, frame)
                        self.notifier.play_alert_sound()
                else:
                    status = "SEARCHING PUPIL"
                    self.integrity_manager.update(True)
            else:
                status = "SEARCHING EYES"
                self.integrity_manager.update(True)
        else:
            status = "FACE NOT DETECTED"
            self.notifier.play_alert_sound()
            if self.integrity_manager.is_calibrated:
                self.recorder.log_event("VIOLATION", "Face missing", self.integrity_manager.score, frame)
                self.integrity_manager.update(True)

        if len(faces) > 1:
            status = "VIOLATION: MULTI-PERSON"
            self.recorder.log_event("VIOLATION", "Multi-face", self.integrity_manager.score, frame)
            self.notifier.play_alert_sound()

        # Update GUI widgets
        self.score_label.configure(text=f"Integrity Score: {int(self.integrity_manager.score)}")
        self.status_label.configure(text=f"Status: {status}")

        # Process main video for display
        display_frame = ProctorUI.draw_overlays(frame.copy(), faces, primary_face, eye_rects, status)
        img_main = ProctorUI.convert_frame(display_frame, 800, 600)
        self.video_label.configure(image=img_main)
        self.video_label.image = img_main

        # Update Small Eye Monitor
        img_eye = ProctorUI.convert_frame(binary_eye_view, 150, 100)
        self.eye_video_label.configure(image=img_eye)
        self.eye_video_label.image = img_eye

        # Schedule next update
        self.after(10, self.update_frame)

    def on_closing(self):
        self.stop_system()
        self.destroy()

import time # For time tracking in recalibrate
