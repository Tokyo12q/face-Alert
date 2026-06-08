import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")

def _ensure_model():

    if not os.path.exists(_MODEL_PATH):

        os.makedirs(_MODEL_DIR, exist_ok=True)

        print(f"[detector] Downloading face landmarker model to {_MODEL_PATH} ...")

        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

        print("[detector] Model downloaded successfully.")

class ProctorDetector:

    def __init__(self):

        _ensure_model()

        BaseOptions = mp.tasks.BaseOptions

        FaceLandmarker = mp.tasks.vision.FaceLandmarker

        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),

            running_mode=VisionRunningMode.IMAGE,

            num_faces=2,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect_faces(self, bgr_frame):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        )

        result = self.landmarker.detect(mp_image)

        faces = []

        if result.face_landmarks:
            h, w = bgr_frame.shape[:2]

            for face_landmarks in result.face_landmarks:

                coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])

                xmin, ymin = np.min(coords, axis=0)
                xmax, ymax = np.max(coords, axis=0)

                bbox = (int(xmin), int(ymin),
                        int(xmax - xmin), int(ymax - ymin))

                faces.append({
                    'bbox'      : bbox,
                    'landmarks' : face_landmarks,
                    'coords'    : coords,
                    'score'     : 1.0
                })

        return faces

    def get_primary_face(self, faces):

        if faces:
            return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])

        return None

    def get_eye_rois(self, bgr_frame, primary_face):

        h, w = bgr_frame.shape[:2]

        coords = primary_face['coords']

        LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]

        RIGHT_EYE = [33,  7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

        eye_centers = [
            np.mean(coords[LEFT_EYE],  axis=0),
            np.mean(coords[RIGHT_EYE], axis=0),
        ]

        face_width = primary_face['bbox'][2]

        box_size = int(face_width * 0.12)

        rois = []
        for center in eye_centers:
            ex, ey = int(center[0]), int(center[1])

            x1, y1 = ex - box_size,        ey - int(box_size / 1.5)
            x2, y2 = ex + box_size,        ey + int(box_size / 1.5)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                eye_roi = bgr_frame[y1:y2, x1:x2]

                rois.append({
                    'roi'    : eye_roi,
                    'coords' : (x1, y1, x2 - x1, y2 - y1)
                })

        return rois

    def __del__(self):

        if hasattr(self, 'landmarker'):
            self.landmarker.close()
