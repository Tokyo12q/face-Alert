import cv2
import numpy as np

class ProctorProcessor:

    @staticmethod
    def sharpen(image):

        kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])

        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def apply_blur(image):

        return cv2.GaussianBlur(image, (7, 7), 0)

    @staticmethod
    def adaptive_threshold(image):

        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    @staticmethod
    def apply_clahe(image):

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        return clahe.apply(image)

    @staticmethod
    def process_eye_region(eye_roi):

        enhanced = ProctorProcessor.apply_clahe(eye_roi)

        sharpened = ProctorProcessor.sharpen(enhanced)

        blurred = ProctorProcessor.apply_blur(sharpened)

        thresholded = ProctorProcessor.adaptive_threshold(blurred)

        return thresholded
