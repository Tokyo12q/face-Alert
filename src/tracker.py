# ─────────────────────────────────────────────────────────────────────────────
# src/tracker.py  –  Gaze Tracking & Integrity Scoring
# Purpose : Two classes work together:
#   • GazeTracker      – finds the pupil in a binary eye image and compares
#                        its horizontal position to a calibrated baseline.
#   • IntegrityManager – maintains a 0-100 integrity score, manages the
#                        auto-calibration window, and decides when a
#                        suspicious gaze deviation becomes a VIOLATION.
# ─────────────────────────────────────────────────────────────────────────────

import cv2    # OpenCV – contour detection and moment calculation.
import numpy as np  # NumPy – (imported but available for future use).

import time   # Built-in module for getting wall-clock timestamps.


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 1 – GazeTracker
# ═══════════════════════════════════════════════════════════════════════════════

class GazeTracker:
    """
    Detects the pupil position in a binary eye image and flags suspicious
    horizontal gaze deviations relative to a calibrated centre baseline.
    """

    def __init__(self, threshold_offset=15):
        """
        Parameters
        ----------
        threshold_offset : int
            Maximum allowed horizontal deviation (in pixels) from the
            calibrated baseline before the gaze is considered suspicious.
            Default is 15 pixels.
        """

        # Store the allowed horizontal deviation in pixels.
        # If the pupil moves more than this many pixels left or right of
        # the baseline, check_behavior() returns is_suspicious=True.
        self.threshold_offset = threshold_offset

        # The calibrated centre X-position of the pupil.
        # None means calibration has not yet occurred.
        self.baseline_x = None

    # ─── Pupil Detection ──────────────────────────────────────────────────────

    def find_pupil_center(self, binary_eye_roi):
        """
        Locate the pupil centre in a binary (black/white) eye image.

        In the binary image produced by ProctorProcessor, the pupil is the
        darkest region (value ≈ 0).  We invert the image so the pupil becomes
        the brightest blob, then use contour detection to find it.

        Parameters
        ----------
        binary_eye_roi : np.ndarray  – binary grayscale eye crop (0 or 255).

        Returns
        -------
        (cx, cy) tuple of ints if a pupil is found, or None.
        """

        # Invert the binary image: dark pupil (0) → bright (255) and
        # bright sclera (255) → dark (0).
        # cv2.bitwise_not() flips every bit: 0→255, 255→0.
        inverted = cv2.bitwise_not(binary_eye_roi)

        # Find all contours (outlines of white blobs) in the inverted image.
        # RETR_EXTERNAL   – retrieve only the outermost contours (ignore holes).
        # CHAIN_APPROX_SIMPLE – compress horizontal/vertical runs into endpoints.
        # The function returns a list of contours and a hierarchy (ignored with _).
        contours, _ = cv2.findContours(inverted,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Only proceed if at least one contour was found.
        if contours:
            # Assume the largest contour by area represents the pupil/iris.
            # max() with key=cv2.contourArea picks the contour with the
            # biggest enclosed pixel area.
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute image moments of the largest contour.
            # Moments are weighted averages of pixel positions and are used
            # to calculate the centroid (geometric centre) of a shape.
            M = cv2.moments(largest_contour)

            # m00 is the zeroth moment = total area of the contour.
            # If m00 == 0 the contour has no area (degenerate), so skip it.
            if M["m00"] != 0:
                # m10 / m00 gives the centroid X-coordinate.
                # m01 / m00 gives the centroid Y-coordinate.
                # These are the standard centroid formulas from image moments.
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Return the pupil centre as a (cx, cy) tuple.
                return cx, cy

        # Return None if no contour was found or the area was zero.
        return None

    # ─── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, cx):
        """
        Record the current pupil X-position as the 'looking-straight-ahead'
        baseline.  Called by IntegrityManager once auto-calibration is done.
        """

        # Save the horizontal pupil centre as the reference baseline.
        # Future deviations are measured relative to this value.
        self.baseline_x = cx

    # ─── Behaviour Check ──────────────────────────────────────────────────────

    def check_behavior(self, cx):
        """
        Compare the current pupil X-position to the baseline.

        Parameters
        ----------
        cx : int  – current horizontal pupil centre from find_pupil_center().

        Returns
        -------
        (status_string, is_suspicious_bool)
        """

        # If calibration has not been done yet, we cannot make a judgement.
        if self.baseline_x is None:
            return "CALIBRATING", False

        # Calculate how many pixels the pupil has moved from the centre.
        # abs() makes the result positive regardless of direction.
        deviation = abs(cx - self.baseline_x)

        # If deviation exceeds the allowed threshold → suspicious.
        if deviation > self.threshold_offset:
            return "WARNING: SUSPICIOUS BEHAVIOR (LOOKING ASIDE)", True

        # Pupil is within the acceptable range → normal behaviour.
        return "BEHAVIOR: NORMAL", False


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 2 – IntegrityManager
# ═══════════════════════════════════════════════════════════════════════════════

class IntegrityManager:
    """
    Manages the exam integrity score and the automatic calibration window.

    Auto-calibration phase: For the first `auto_calibrate_duration` seconds,
    the system collects pupil X samples to establish the baseline.

    Live monitoring phase: After calibration, every suspicious gaze event
    starts a timer.  If suspicion persists longer than `suspicion_threshold`
    seconds, the score is reduced and a VIOLATION status is issued.
    """

    def __init__(self, suspicion_threshold=3.0, auto_calibrate_duration=10.0):
        """
        Parameters
        ----------
        suspicion_threshold      : float – seconds of continuous suspicious
                                   gaze before a VIOLATION is declared (default 3 s).
        auto_calibrate_duration  : float – seconds to collect baseline samples
                                   before calibration completes (default 10 s).
        """

        # Integrity score starts at 100 (perfect) and decreases on violations.
        self.score = 100

        # How many seconds of continuous suspicious gaze triggers a VIOLATION.
        self.suspicion_threshold = suspicion_threshold

        # How long (seconds) to collect calibration samples.
        self.auto_calibrate_duration = auto_calibrate_duration

        # Record the wall-clock time when this object was created.
        # Used as the start of the calibration window.
        self.start_time = time.time()

        # Timestamp when suspicious gaze first started.
        # None means no current suspicion.
        self.suspicion_start = None

        # Flag: has calibration completed successfully?
        self.is_calibrated = False

        # Buffer that collects pupil X values during the calibration window.
        self.calibration_samples = []

    # ─── Live Monitoring Update ───────────────────────────────────────────────

    def update(self, is_suspicious):
        """
        Update the integrity state for the current frame and return a status
        string.

        Parameters
        ----------
        is_suspicious : bool – True if the gaze check flagged a deviation.

        Returns
        -------
        str – human-readable status for display in the GUI.
        """

        # During calibration, don't penalise any behaviour yet.
        if not self.is_calibrated:
            return "AUTO-CALIBRATING"

        if is_suspicious:
            # Record when suspicion first started (only on the first frame).
            if self.suspicion_start is None:
                self.suspicion_start = time.time()

            # Calculate how many seconds the current suspicious gaze has lasted.
            elapsed = time.time() - self.suspicion_start

            # If suspicion has lasted longer than the threshold → VIOLATION.
            if elapsed > self.suspicion_threshold:
                # Decrease score by 0.1 per frame (gradual degradation).
                # max(0, …) prevents the score from going below zero.
                self.score = max(0, self.score - 0.1)
                return f"VIOLATION: PERSISTENT LOOK ASIDE (Score: {int(self.score)})"

            # Suspicion detected but not yet long enough to be a violation.
            return f"SUSPICION: LOOKING ASIDE ({int(elapsed)}s)"

        else:
            # Gaze is normal: reset the suspicion timer.
            self.suspicion_start = None
            return f"NORMAL (Score: {int(self.score)})"

    # ─── Calibration Handler ──────────────────────────────────────────────────

    def handle_calibration(self, cx):
        """
        Accumulate pupil X samples during the calibration window and finalise
        the baseline when the window closes.

        Parameters
        ----------
        cx : int or None  – current pupil X from find_pupil_center(), or None
                            if the pupil was not found this frame.

        Returns
        -------
        • True           if already calibrated (should not normally be called).
        • False          if still collecting samples.
        • float (avg_cx) when calibration completes successfully – this value
                         is passed directly to GazeTracker.calibrate().
        """

        # If already calibrated, nothing more to do.
        if self.is_calibrated:
            return True

        # Check how many seconds have elapsed since calibration started.
        elapsed = time.time() - self.start_time

        if elapsed < self.auto_calibrate_duration:
            # Still inside the calibration window.
            # Append the pupil X position to the sample buffer (if valid).
            if cx is not None:
                self.calibration_samples.append(cx)
            return False  # Signal: calibration not yet complete.

        else:
            # The calibration window has closed.
            if self.calibration_samples:
                # Calculate the average pupil X across all collected samples.
                # This average represents the "looking straight ahead" position.
                avg_cx = sum(self.calibration_samples) / len(self.calibration_samples)

                # Mark calibration as done so future calls skip this block.
                self.is_calibrated = True

                # Return the average X so the caller can pass it to
                # GazeTracker.calibrate().
                return avg_cx
            else:
                # No valid pupil samples were collected (e.g. no face detected
                # during the whole window).  Reset the timer and try again.
                self.start_time = time.time()
                return False
