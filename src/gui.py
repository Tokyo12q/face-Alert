import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.detector  import ProctorDetector
from src.processor import ProctorProcessor
from src.tracker   import GazeTracker, IntegrityManager
from src.ui        import ProctorUI
from src.notifier  import ProctorNotifier
from src.recorder  import ProctorRecorder

class ProctorDashboard(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("Smart Proctoring System Dashboard")

        self.geometry("1100x700")

        ctk.set_appearance_mode("dark")

        ctk.set_default_color_theme("blue")

        self.detector = ProctorDetector()

        self.gaze_tracker = GazeTracker()

        self.integrity_manager = IntegrityManager(
            suspicion_threshold=3.0,
            auto_calibrate_duration=5.0
        )

        self.notifier = ProctorNotifier(cooldown_seconds=1)

        self.recorder = ProctorRecorder()

        self.cap = None

        self.is_running = False

        self._build_ui()

    def _build_ui(self):

        self.bg_color = "#f8fafc"
        self.sidebar_color = "#0f172a"
        self.card_color = "#ffffff"
        self.accent_color = "#2563eb"
        self.danger_color = "#ef4444"
        self.configure(fg_color=self.bg_color)

        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=self.sidebar_color)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="FACE-ALERT", 
            font=ctk.CTkFont(size=26, weight="bold", family="Inter"),
            text_color="#ffffff"
        )
        self.logo_label.pack(pady=(50, 60))

        self.status_badge = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", corner_radius=12, height=45)
        self.status_badge.pack(padx=30, fill="x", pady=(0, 40))
        self.status_badge.pack_propagate(False)

        self.status_dot = ctk.CTkLabel(
            self.status_badge, text="●", text_color="gray", font=ctk.CTkFont(size=14)
        )
        self.status_dot.pack(side="left", padx=(15, 5))

        self.status_label = ctk.CTkLabel(
            self.status_badge, text="SYSTEM STANDBY", font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#94a3b8"
        )
        self.status_label.pack(side="left")

        self.btn_start = ctk.CTkButton(
            self.sidebar, text="Start Session", height=48, corner_radius=10,
            font=ctk.CTkFont(weight="bold"), fg_color=self.accent_color, hover_color="#1d4ed8",
            command=self.start_system
        )
        self.btn_start.pack(pady=8, padx=30, fill="x")

        self.btn_stop = ctk.CTkButton(
            self.sidebar, text="End Session", height=48, corner_radius=10,
            font=ctk.CTkFont(weight="bold"), fg_color="transparent", border_width=1, border_color=self.danger_color,
            text_color=self.danger_color, hover_color="#450a0a",
            command=self.stop_system, state="disabled"
        )
        self.btn_stop.pack(pady=8, padx=30, fill="x")

        self.btn_calibrate = ctk.CTkButton(
            self.sidebar, text="Recalibrate Gaze", height=40, corner_radius=10,
            font=ctk.CTkFont(size=12), fg_color="#1e293b", hover_color="#334155",
            text_color="#f1f5f9",
            command=self.recalibrate, state="disabled"
        )
        self.btn_calibrate.pack(pady=(40, 0), padx=30, fill="x")

        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.pack(side="right", expand=True, fill="both", padx=40, pady=40)

        self.ws_header = ctk.CTkLabel(
            self.workspace, text="Session Monitoring", 
            font=ctk.CTkFont(size=32, weight="bold", family="Inter"), anchor="w",
            text_color="#0f172a"
        )
        self.ws_header.pack(fill="x", pady=(0, 30))

        self.content_container = ctk.CTkFrame(self.workspace, fg_color="transparent")
        self.content_container.pack(expand=True, fill="both")

        self.video_card = ctk.CTkFrame(self.content_container, fg_color="#000000", corner_radius=20, border_width=1, border_color="#e2e8f0")
        self.video_card.pack(side="left", expand=True, fill="both", padx=(0, 25))

        self.video_label = ctk.CTkLabel(self.video_card, text="Camera Offline", font=ctk.CTkFont(size=14))
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.insights_panel = ctk.CTkFrame(self.content_container, width=320, fg_color="transparent")
        self.insights_panel.pack(side="right", fill="y")
        self.insights_panel.pack_propagate(False)

        self.score_card = ctk.CTkFrame(self.insights_panel, fg_color=self.card_color, corner_radius=20, border_width=1, border_color="#e2e8f0")
        self.score_card.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(self.score_card, text="INTEGRITY SCORE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#64748b").pack(pady=(20, 0))
        self.score_label = ctk.CTkLabel(
            self.score_card, text="100", 
            font=ctk.CTkFont(size=64, weight="bold", family="Inter"),
            text_color=self.accent_color
        )
        self.score_label.pack(pady=(0, 25))

        self.eye_card = ctk.CTkFrame(self.insights_panel, fg_color=self.card_color, corner_radius=20, border_width=1, border_color="#e2e8f0")
        self.eye_card.pack(fill="both", expand=True)

        ctk.CTkLabel(self.eye_card, text="GAZE FOCUS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#64748b").pack(pady=(20, 10))

        self.eye_video_label = ctk.CTkLabel(self.eye_card, text="", width=260, height=160, fg_color="#000000", corner_radius=12)
        self.eye_video_label.pack(pady=(0, 25), padx=25)

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

                eye_roi_bgr  = eye_rois[0]['roi']
                eye_roi_gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)

                binary_eye_view = ProctorProcessor.process_eye_region(eye_roi_gray)

                center = self.gaze_tracker.find_pupil_center(binary_eye_view)

                if center:
                    _, is_suspicious = self.gaze_tracker.check_behavior(center[0])

                    status = self.integrity_manager.update(is_suspicious)

                    if "VIOLATION" in status:
                        self.recorder.log_event(
                            "VIOLATION", status,
                            self.integrity_manager.score, frame
                        )
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
                self.recorder.log_event(
                    "VIOLATION", "Face missing",
                    self.integrity_manager.score, frame
                )
                self.integrity_manager.update(True)

        if len(faces) > 1:
            status = "VIOLATION: MULTI-PERSON"
            self.recorder.log_event(
                "VIOLATION", "Multi-face",
                self.integrity_manager.score, frame
            )
            self.notifier.play_alert_sound()

        self.score_label.configure(text=str(int(self.integrity_manager.score)))

        display_status = status
        status_color = "#94a3b8"
        dot_color = "#94a3b8"

        if "VIOLATION" in status:
            status_color = "#ef4444"
            dot_color = "#ef4444"
        elif "SUSPICION" in status or "SEARCHING" in status:
            status_color = "#f59e0b"
            dot_color = "#f59e0b"
        elif "CALIBRATING" in status:
            status_color = "#3b82f6"
            dot_color = "#3b82f6"
        elif "NORMAL" in status:
            status_color = "#10b981"
            dot_color = "#10b981"

        self.status_label.configure(text=display_status.upper(), text_color=status_color)
        self.status_dot.configure(text_color=dot_color)

        display_frame = ProctorUI.draw_overlays(
            frame.copy(), faces, primary_face, eye_rects, status
        )

        img_main = ProctorUI.convert_frame(display_frame, 800, 600)

        self.video_label.configure(image=img_main)

        self.video_label.image = img_main

        img_eye = ProctorUI.convert_frame(binary_eye_view, 150, 100)

        self.eye_video_label.configure(image=img_eye)

        self.eye_video_label.image = img_eye

        self.after(10, self.update_frame)

    def on_closing(self):

        self.stop_system()

        self.destroy()

import time
