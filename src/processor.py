# ─────────────────────────────────────────────────────────────────────────────
# src/processor.py  –  Image Processing Pipeline for Eye Regions
# Purpose : Provides a series of static image-processing methods that transform
#           a raw grayscale eye crop into a clean binary (black/white) image
#           suitable for pupil detection via contour analysis.
# ─────────────────────────────────────────────────────────────────────────────

import cv2    # OpenCV – computer-vision library used for all image operations.
import numpy as np  # NumPy – used here to define the sharpening kernel matrix.


class ProctorProcessor:
    """A collection of static image-processing helpers (no instance needed)."""

    # ── Step helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def sharpen(image):
        """Enhance edges by applying a sharpening convolution kernel."""

        # A 3×3 sharpening kernel: the centre pixel gets weight +9 while its
        # 8 neighbours each get weight -1.  Convolving with this kernel makes
        # edges appear crisper and increases local contrast.
        kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])

        # cv2.filter2D applies the kernel to every pixel of 'image'.
        # -1 means the output depth matches the input depth (e.g. uint8).
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def apply_blur(image):
        """Smooth the image with a Gaussian blur to reduce noise."""

        # GaussianBlur(src, ksize, sigmaX)
        # ksize (7, 7) – 7×7 pixel neighbourhood is large enough to suppress
        #                high-frequency noise without over-blurring the pupil.
        # sigmaX = 0   – OpenCV calculates sigma automatically from ksize.
        return cv2.GaussianBlur(image, (7, 7), 0)

    @staticmethod
    def adaptive_threshold(image):
        """Binarise the image using locally adaptive thresholding."""

        # adaptiveThreshold computes a different threshold for each small
        # region of the image, which handles uneven lighting much better than
        # a single global threshold.
        #
        # Arguments:
        #   image           – grayscale source image
        #   255             – value assigned to pixels above the threshold
        #   ADAPTIVE_THRESH_GAUSSIAN_C – threshold = weighted mean of
        #                    neighbourhood pixels (Gaussian weights)
        #   THRESH_BINARY   – pixels > threshold → 255, else → 0
        #   11              – neighbourhood block size (must be odd)
        #   2               – constant subtracted from the computed mean
        #                     (fine-tunes sensitivity)
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    @staticmethod
    def apply_clahe(image):
        """Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE)."""

        # CLAHE improves local contrast, making dark pupils stand out even
        # when the surrounding region is poorly lit.
        #
        # clipLimit=2.0       – caps the contrast amplification to prevent
        #                       noise from being amplified too much.
        # tileGridSize=(8, 8) – image is divided into 8×8 tiles; each tile
        #                       gets its own histogram equalisation.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Apply CLAHE to the grayscale image and return the result.
        return clahe.apply(image)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def process_eye_region(eye_roi):
        """
        Run the full 4-step processing pipeline on a grayscale eye crop.

        Input  : eye_roi – a grayscale (single-channel) NumPy array of the
                           cropped eye region captured from the webcam.
        Output : A binary (black/white) image where the dark pupil appears as
                 a large bright blob, ready for contour-based pupil detection.
        """

        # Step 0 – Enhancement (CLAHE)
        # Equalise local contrast so the pupil is visible regardless of
        # ambient lighting conditions.
        enhanced = ProctorProcessor.apply_clahe(eye_roi)

        # Step 1 – Sharpening
        # Accentuate the boundary between the pupil and the iris/sclera.
        sharpened = ProctorProcessor.sharpen(enhanced)

        # Step 2 – Gaussian Blur
        # Smooth out salt-and-pepper noise introduced by sharpening so the
        # subsequent thresholding step produces clean blobs.
        blurred = ProctorProcessor.apply_blur(sharpened)

        # Step 3 – Adaptive Thresholding (Segmentation)
        # Convert the grayscale image to a binary black/white image.
        # The pupil (dark region) will become a well-defined dark area
        # surrounded by a white iris/sclera.
        # Note: Input eye_roi should be grayscale (single channel).
        thresholded = ProctorProcessor.adaptive_threshold(blurred)

        return thresholded
