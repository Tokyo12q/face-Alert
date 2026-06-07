import cv2
import numpy as np

import time

class GazeTracker:
    def __init__(self, threshold_offset=15):
        self.threshold_offset = threshold_offset
        self.baseline_x = None

    def find_pupil_center(self, binary_eye_roi):
        # Invert because pupil is dark (0) and we need white (255) for contours
        inverted = cv2.bitwise_not(binary_eye_roi)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the pupil/iris
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        return None

    def calibrate(self, cx):
        self.baseline_x = cx

    def check_behavior(self, cx):
        if self.baseline_x is None:
            return "CALIBRATING", False
        
        deviation = abs(cx - self.baseline_x)
        if deviation > self.threshold_offset:
            return "WARNING: SUSPICIOUS BEHAVIOR (LOOKING ASIDE)", True
        return "BEHAVIOR: NORMAL", False

class IntegrityManager:
    def __init__(self, suspicion_threshold=3.0, auto_calibrate_duration=10.0):
        self.score = 100
        self.suspicion_threshold = suspicion_threshold
        self.auto_calibrate_duration = auto_calibrate_duration
        self.start_time = time.time()
        self.suspicion_start = None
        self.is_calibrated = False
        self.calibration_samples = []

    def update(self, is_suspicious):
        if not self.is_calibrated:
            return "AUTO-CALIBRATING"

        if is_suspicious:
            if self.suspicion_start is None:
                self.suspicion_start = time.time()
            
            elapsed = time.time() - self.suspicion_start
            if elapsed > self.suspicion_threshold:
                self.score = max(0, self.score - 0.1) # Gradual decrease
                return f"VIOLATION: PERSISTENT LOOK ASIDE (Score: {int(self.score)})"
            return f"SUSPICION: LOOKING ASIDE ({int(elapsed)}s)"
        else:
            self.suspicion_start = None
            return f"NORMAL (Score: {int(self.score)})"

    def handle_calibration(self, cx):
        if self.is_calibrated:
            return True
        
        elapsed = time.time() - self.start_time
        if elapsed < self.auto_calibrate_duration:
            if cx is not None:
                self.calibration_samples.append(cx)
            return False
        else:
            if self.calibration_samples:
                avg_cx = sum(self.calibration_samples) / len(self.calibration_samples)
                self.is_calibrated = True
                return avg_cx
            else:
                # Reset timer if no samples were found in the window
                self.start_time = time.time()
                return False
