# ─────────────────────────────────────────────────────────────────────────────
# src/detector.py  –  Face & Eye Detection Engine
# Purpose : Uses Google's MediaPipe Face Landmarker (Tasks API) to locate every
#           face in a webcam frame, extract 468 3-D landmarks per face, compute
#           a bounding box, and crop out the eye regions for pupil tracking.
#
# Key design decisions:
#   • The heavy .task model file is downloaded automatically on first run
#     and cached in the project's models/ folder.
#   • Up to 2 faces are detected (primary student + potential cheater).
#   • The "primary" face is simply the largest one (closest to camera).
# ─────────────────────────────────────────────────────────────────────────────

import cv2            # OpenCV – colour conversion and array slicing.
import mediapipe as mp  # Google MediaPipe – provides the Face Landmarker model.
import numpy as np    # NumPy – efficient array math for landmark coordinates.
import os             # Built-in – file/path operations.
import urllib.request # Built-in – downloads the model file over HTTPS.


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (computed once at import time)
# ─────────────────────────────────────────────────────────────────────────────

# URL of the pre-trained MediaPipe Face Landmarker model (float16 precision).
# This model detects 468 3-D facial landmarks per face.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Absolute path to the models/ directory (one level up from this file's src/).
# os.path.dirname(__file__) → absolute path to the src/ folder.
# ".." navigates up to the project root.
# os.path.join combines them into a cross-platform path.
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Full path to the cached model file.
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


# ─────────────────────────────────────────────────────────────────────────────
# Private helper – download model on first use
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model():
    """Download the face-landmarker model if it isn't already present."""

    # os.path.exists() returns True if the file is already on disk.
    # Skip the download to avoid waiting on every run.
    if not os.path.exists(_MODEL_PATH):

        # Create the models/ folder (and any missing parent dirs) if needed.
        # exist_ok=True means no error if the folder already exists.
        os.makedirs(_MODEL_DIR, exist_ok=True)

        print(f"[detector] Downloading face landmarker model to {_MODEL_PATH} ...")

        # urllib.request.urlretrieve(url, filename) downloads the file at 'url'
        # and saves it to 'filename', blocking until the download is complete.
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        print("[detector] Model downloaded successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Class
# ═══════════════════════════════════════════════════════════════════════════════

class ProctorDetector:
    """
    Wraps the MediaPipe Face Landmarker Tasks API.

    Public methods
    --------------
    detect_faces(bgr_frame)            → list of face dicts
    get_primary_face(faces)            → face dict or None
    get_eye_rois(bgr_frame, face)      → list of eye dicts
    """

    def __init__(self):
        """Initialise the Face Landmarker, downloading the model if needed."""

        # Ensure the model file exists before we try to load it.
        _ensure_model()

        # ── Convenience aliases for the long MediaPipe namespace paths ──

        # BaseOptions wraps low-level configuration (model path, delegate).
        BaseOptions = mp.tasks.BaseOptions

        # FaceLandmarker is the main inference class.
        FaceLandmarker = mp.tasks.vision.FaceLandmarker

        # FaceLandmarkerOptions holds all tunable hyperparameters.
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        # RunningMode.IMAGE means we feed one complete image at a time
        # (as opposed to VIDEO or LIVE_STREAM modes which add timestamps).
        VisionRunningMode = mp.tasks.vision.RunningMode

        # ── Build the options object ──────────────────────────────────────────

        options = FaceLandmarkerOptions(
            # Tell MediaPipe where the model file is on disk.
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),

            # IMAGE mode: each call to detect() is independent.
            running_mode=VisionRunningMode.IMAGE,

            # Detect at most 2 faces per frame (primary student + 1 intruder).
            num_faces=2,

            # Minimum confidence required to report a detected face.
            min_face_detection_confidence=0.5,

            # Minimum confidence that a face is present (vs false positive).
            min_face_presence_confidence=0.5,

            # Minimum confidence for tracking consistency between frames.
            min_tracking_confidence=0.5,

            # We don't need blend-shape scores (for facial expressions).
            output_face_blendshapes=False,

            # We don't need the 4×4 transformation matrix for each face.
            output_facial_transformation_matrixes=False
        )

        # Create and store the landmarker instance.
        # This loads the model into memory (expensive, done only once).
        self.landmarker = FaceLandmarker.create_from_options(options)

    # ─── Face Detection ───────────────────────────────────────────────────────

    def detect_faces(self, bgr_frame):
        """
        Run face landmark detection on a single BGR webcam frame.

        Parameters
        ----------
        bgr_frame : np.ndarray  – raw BGR frame from cv2.VideoCapture.

        Returns
        -------
        list of dicts, one per detected face:
            {
              'bbox'      : (x, y, w, h) bounding box in pixels,
              'landmarks' : original MediaPipe landmark list,
              'coords'    : (N, 2) NumPy array of (x, y) pixel positions,
              'score'     : detection confidence (1.0 – always set for Tasks API)
            }
        """

        # MediaPipe Tasks API requires a mp.Image object (not a plain NumPy array).
        # We must also convert BGR → RGB because MediaPipe expects SRGB (RGB) input.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        )

        # Run inference.  result.face_landmarks is a list of landmark sets,
        # one per detected face.  Each landmark set has 468 landmarks.
        result = self.landmarker.detect(mp_image)

        # Accumulate results into a plain Python list.
        faces = []

        if result.face_landmarks:
            # Get the frame dimensions for converting normalised coordinates.
            # Landmark x/y values are in the range [0.0, 1.0] (relative to frame).
            h, w = bgr_frame.shape[:2]

            # Process each detected face separately.
            for face_landmarks in result.face_landmarks:

                # ── Convert normalised landmarks → pixel coordinates ───────────

                # Build a NumPy array of shape (468, 2).
                # lm.x * w → pixel X;  lm.y * h → pixel Y.
                coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])

                # Find the bounding box by taking the minimum and maximum
                # X and Y values across all 468 landmarks.
                xmin, ymin = np.min(coords, axis=0)  # top-left corner
                xmax, ymax = np.max(coords, axis=0)  # bottom-right corner

                # Convert to integer (x, y, width, height) tuple.
                bbox = (int(xmin), int(ymin),
                        int(xmax - xmin), int(ymax - ymin))

                # Append a structured dict for this face to the results list.
                faces.append({
                    'bbox'      : bbox,
                    'landmarks' : face_landmarks,  # raw MediaPipe landmarks
                    'coords'    : coords,           # pixel-space (468, 2) array
                    'score'     : 1.0               # Tasks API doesn't expose score
                })

        return faces

    # ─── Primary Face Selection ───────────────────────────────────────────────

    def get_primary_face(self, faces):
        """
        Identify the primary (exam-taker) face – the largest face by area.

        A larger bounding box generally means the face is closer to the camera,
        which is the most reliable heuristic when at most one exam-taker is
        expected in the frame.

        Parameters
        ----------
        faces : list – output of detect_faces().

        Returns
        -------
        The face dict with the largest bbox area, or None if faces is empty.
        """

        if faces:
            # bbox[2] = width, bbox[3] = height.
            # Multiplying them gives the bounding-box area.
            # max() with this key returns the face with the largest area.
            return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])

        # No faces detected → return None.
        return None

    # ─── Eye Region Extraction ────────────────────────────────────────────────

    def get_eye_rois(self, bgr_frame, primary_face):
        """
        Crop rectangular Regions of Interest (ROIs) around each eye.

        Uses specific landmark indices (same numbering as the 468-point
        Face Mesh) to locate the eye centres, then pads them into boxes
        proportional to the face width.

        Parameters
        ----------
        bgr_frame    : np.ndarray – the full BGR frame.
        primary_face : dict       – face dict from detect_faces().

        Returns
        -------
        list of dicts, one per eye found:
            {
              'roi'    : np.ndarray – cropped BGR eye image,
              'coords' : (x, y, w, h) – position of the crop in the frame
            }
        """

        # Frame dimensions used for boundary clamping.
        h, w = bgr_frame.shape[:2]

        # Pixel-space (468, 2) landmark array for the primary face.
        coords = primary_face['coords']

        # ── Landmark indices for each eye ─────────────────────────────────────
        # These 16-point index sets outline the eyelid contour in MediaPipe's
        # 468-landmark face mesh (identical numbering to Face Mesh).

        # Indices that form the LEFT eye eyelid (from the subject's perspective).
        LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]

        # Indices that form the RIGHT eye eyelid.
        RIGHT_EYE = [33,  7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

        # Compute the geometric centre of each eye by averaging the
        # (x, y) positions of its landmark ring.
        # np.mean(…, axis=0) averages over rows → returns [mean_x, mean_y].
        eye_centers = [
            np.mean(coords[LEFT_EYE],  axis=0),  # [cx_left,  cy_left]
            np.mean(coords[RIGHT_EYE], axis=0),  # [cx_right, cy_right]
        ]

        # Determine the crop box size as a fraction of the face width.
        # bbox[2] is the face bounding-box width.
        face_width = primary_face['bbox'][2]

        # Box half-size: 12% of the face width.
        # A wider face → larger eye crop, keeping the eye nicely framed.
        box_size = int(face_width * 0.12)

        rois = []
        for center in eye_centers:
            # Integer pixel coordinates of the eye centre.
            ex, ey = int(center[0]), int(center[1])

            # Compute the crop boundaries.
            # The box extends box_size pixels left/right and box_size/1.5
            # pixels up/down (eyes are wider than tall, so less vertical pad).
            x1, y1 = ex - box_size,        ey - int(box_size / 1.5)
            x2, y2 = ex + box_size,        ey + int(box_size / 1.5)

            # Clamp to frame boundaries to avoid NumPy slice errors.
            # max(0, …) prevents negative indices.
            # min(w, …) / min(h, …) prevents indices beyond the frame edge.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Only add the ROI if the crop has positive area (sanity check).
            if x2 > x1 and y2 > y1:
                # Slice the eye region out of the full BGR frame.
                # NumPy slicing: [row_start:row_end, col_start:col_end]
                eye_roi = bgr_frame[y1:y2, x1:x2]

                rois.append({
                    'roi'    : eye_roi,
                    'coords' : (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
                })

        return rois

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def __del__(self):
        """Release MediaPipe resources when the detector is garbage-collected."""

        # hasattr guard: if __init__ failed before self.landmarker was created
        # (e.g. model download error), __del__ would raise AttributeError.
        if hasattr(self, 'landmarker'):
            # close() releases the model from memory and any background threads
            # MediaPipe may have spawned.
            self.landmarker.close()
