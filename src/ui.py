# ─────────────────────────────────────────────────────────────────────────────
# src/ui.py  –  Visual Overlay & Frame Conversion Helpers
# Purpose : Pure drawing/rendering utilities.  This class has NO state; every
#           method is @staticmethod so it can be called without creating an
#           object.  It handles three jobs:
#             1. convert_frame  – turn an OpenCV frame into a Tkinter image.
#             2. draw_overlays  – paint face boxes, landmarks & status text.
#             3. create_hud     – combine the live feed + eye-zoom side-by-side.
# ─────────────────────────────────────────────────────────────────────────────

import cv2    # OpenCV – drawing rectangles, text, circles, colour conversion.
import numpy as np  # NumPy – used for shape queries on image arrays.

# PIL (Pillow) – Python Imaging Library.
# Image    : Opens/creates image objects.
# ImageTk  : Converts a PIL Image to a Tkinter-compatible PhotoImage.
from PIL import Image, ImageTk


class ProctorUI:
    """Stateless helper class with three rendering utility methods."""

    # ─── 1. Frame Conversion ──────────────────────────────────────────────────

    @staticmethod
    def convert_frame(cv_frame, width, height):
        """
        Convert an OpenCV BGR NumPy array to a Tkinter PhotoImage.

        OpenCV uses BGR colour order and stores pixels as NumPy arrays.
        Tkinter (and PIL) expect RGB order.  This method bridges that gap
        and also resizes the frame to fit the GUI widget dimensions.

        Parameters
        ----------
        cv_frame : np.ndarray  – the raw BGR frame from the webcam.
        width    : int         – target display width in pixels.
        height   : int         – target display height in pixels.

        Returns
        -------
        ImageTk.PhotoImage that can be assigned to a CTkLabel widget.
        """

        # Make a copy so the original frame is not altered by later operations.
        # (NumPy arrays are passed by reference; without this, changes here
        # would corrupt the frame used elsewhere in the pipeline.)
        cv_frame = cv_frame.copy()

        # Ensure every pixel value is an unsigned 8-bit integer (0-255).
        # Some processing steps can produce float32 values; astype(np.uint8)
        # clamps and converts them back to the standard image format.
        cv_frame = cv_frame.astype(np.uint8)

        # Convert colour order from BGR (OpenCV default) to RGB (PIL/Tkinter).
        # cv2.COLOR_BGR2RGB swaps the first and third colour channels.
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

        # Wrap the NumPy array in a PIL Image object so we can use PIL methods.
        # Image.fromarray() reads the array as an RGB image automatically.
        pil_image = Image.fromarray(rgb_frame)

        # Resize the image to the requested (width, height) dimensions.
        # Image.LANCZOS is a high-quality downsampling filter that avoids the
        # blocky artefacts of simpler methods like nearest-neighbour.
        pil_image = pil_image.resize((width, height), Image.LANCZOS)

        # Convert the PIL Image into a Tkinter-compatible PhotoImage and return.
        # Tkinter can only display PhotoImage objects inside Label widgets.
        return ImageTk.PhotoImage(image=pil_image)

    # ─── 2. Drawing Overlays ──────────────────────────────────────────────────

    @staticmethod
    def draw_overlays(frame, faces, primary_face, eye_coords, status):
        """
        Draw all visual annotations directly onto 'frame' (in-place).

        Annotations include:
          • Coloured bounding boxes around every detected face.
          • "UNAUTHORIZED PERSON" label on non-primary faces.
          • Sparse 3-D landmark mesh on the primary (exam-taker) face.
          • Highlighted eye corner landmarks.
          • Eye region bounding boxes (from get_eye_rois).
          • Status text in the top-left corner.

        Parameters
        ----------
        frame        : np.ndarray  – BGR frame to draw on.
        faces        : list        – list of face dicts from detect_faces().
        primary_face : dict|None   – the largest / most central face dict.
        eye_coords   : list        – list of (x, y, w, h) eye bounding boxes.
        status       : str         – current status string (e.g. "NORMAL …").

        Returns
        -------
        The annotated frame (same object, mutated in-place).
        """

        # ── Choose overlay colour based on the current status string ──────────

        # If the status contains "VIOLATION" or "SUSPICION" → red.
        # OpenCV uses BGR colour order, so (0, 0, 255) = Red.
        if "VIOLATION" in status or "SUSPICION" in status:
            color = (0, 0, 255)   # Red  – alert state

        # If calibration is in progress → yellow.
        # (0, 255, 255) in BGR = Yellow.
        elif "AUTO-CALIBRATING" in status:
            color = (0, 255, 255) # Yellow – neutral/calibrating state

        # Otherwise (normal behaviour) → green.
        # (0, 255, 0) in BGR = Green.
        else:
            color = (0, 255, 0)   # Green – everything is fine

        # ── Draw bounding boxes around every detected face ────────────────────

        # Iterate over every face found by the detector in this frame.
        for face in faces:

            # Unpack the bounding-box tuple: top-left corner (fx, fy) and
            # dimensions (fw = width, fh = height).
            fx, fy, fw, fh = face['bbox']

            # Determine whether this face is the primary (exam-taker) face.
            # primary_face is None if no face is detected at all.
            # We compare bounding boxes (tuples) for identity.
            is_primary = (primary_face is not None and
                          face['bbox'] == primary_face['bbox'])

            # Primary face → use the status colour (green/yellow/red).
            # Non-primary (stranger) → magenta (255, 0, 255) in BGR.
            box_color = color if is_primary else (255, 0, 255)

            # Draw a 2-pixel-thick coloured rectangle around the face.
            # cv2.rectangle(img, top_left, bottom_right, colour, thickness)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

            # ── Non-primary face: add warning label ───────────────────────────
            if not is_primary:
                # Print "UNAUTHORIZED PERSON" 10 pixels above the box.
                # cv2.putText(img, text, origin, font, scale, colour, thickness)
                cv2.putText(frame, "UNAUTHORIZED PERSON", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ── Primary face: draw landmark mesh + eye highlights ─────────────
            else:
                # Draw a sparse subset of the 468 face-mesh landmarks as tiny
                # dots so the viewer can see the 3-D tracking at work.
                # face['coords'] is a (468, 2) array of (x, y) pixel positions.
                for i, lm_pixel in enumerate(face['coords']):

                    # Only draw every 10th landmark (i % 10 == 0) to keep
                    # the mesh readable without cluttering the frame.
                    if i % 10 == 0:
                        # cv2.circle(img, centre, radius, colour, thickness)
                        # radius=1, thickness=-1 → filled dot 1 px in diameter.
                        # Colour (0, 255, 255) = Cyan in BGR.
                        cv2.circle(frame,
                                   (int(lm_pixel[0]), int(lm_pixel[1])),
                                   1, (0, 255, 255), -1)

                # Highlight the four eye-corner landmark indices specifically.
                # [33, 133] are the inner/outer corners of the RIGHT eye.
                # [362, 263] are the inner/outer corners of the LEFT eye.
                for eye_indices in [[33, 133], [362, 263]]:
                    for idx in eye_indices:
                        # Retrieve the (x, y) pixel position of this landmark.
                        pt = face['coords'][idx]

                        # Draw a slightly larger filled circle (radius=2) in
                        # yellow (255, 255, 0) BGR to make eye corners obvious.
                        cv2.circle(frame,
                                   (int(pt[0]), int(pt[1])),
                                   2, (255, 255, 0), -1)

        # ── Draw eye-region bounding boxes ────────────────────────────────────

        # eye_coords is a list of (ex, ey, ew, eh) tuples from get_eye_rois().
        # Draw a yellow (255, 255, 0) rectangle around each cropped eye area.
        for (ex, ey, ew, eh) in eye_coords:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh),
                          (255, 255, 0), 2)

        # ── Draw status text ──────────────────────────────────────────────────

        # Print the current status string at position (30, 50) – near the
        # top-left corner of the frame.  Font scale 0.8, thickness 2.
        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Return the annotated frame (useful for chaining).
        return frame

    # ─── 3. HUD Assembly ──────────────────────────────────────────────────────

    @staticmethod
    def create_hud(main_frame, binary_eye_zoom):
        """
        Create a side-by-side HUD by concatenating the live feed (left)
        with the binary eye-tracking zoom view (right).

        Parameters
        ----------
        main_frame      : np.ndarray – annotated BGR main camera frame.
        binary_eye_zoom : np.ndarray – binary (grayscale) zoomed eye image.

        Returns
        -------
        hud : np.ndarray – horizontally concatenated BGR image (both panels).
        """

        # Read the height and width of the main frame.
        # shape returns (height, width, channels); [:2] drops the channel count.
        h, w = main_frame.shape[:2]

        # We want the eye panel to have the same height as the main frame.
        zoom_h = h

        # Calculate the width that preserves the eye panel's aspect ratio
        # when it is scaled to height zoom_h.
        # (zoom_h / binary_eye_zoom.shape[0]) is the scaling factor.
        zoom_w = int(binary_eye_zoom.shape[1] * (zoom_h / binary_eye_zoom.shape[0]))

        # Resize the binary eye panel to (zoom_w, zoom_h).
        binary_zoom_resized = cv2.resize(binary_eye_zoom, (zoom_w, zoom_h))

        # cv2.hconcat requires both images to have the same number of channels.
        # A grayscale image has shape (H, W) – only 2 dimensions.
        # len(shape) == 2 means it IS grayscale (no channel dimension).
        if len(binary_zoom_resized.shape) == 2:
            # Convert grayscale → BGR by replicating the single channel 3 times
            # so that cv2.hconcat can join it with the colour main_frame.
            binary_zoom_resized = cv2.cvtColor(binary_zoom_resized,
                                               cv2.COLOR_GRAY2BGR)

        # ── Add panel labels ──────────────────────────────────────────────────

        # Print "LIVE FEED" in white at the top-left of the main camera panel.
        cv2.putText(main_frame, "LIVE FEED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Print "EYE TRACKING ZOOM" at the top-left of the eye panel.
        cv2.putText(binary_zoom_resized, "EYE TRACKING ZOOM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── Concatenate panels side by side ───────────────────────────────────

        # cv2.hconcat([img1, img2]) creates a single wide image by placing
        # img1 on the left and img2 on the right (horizontal concatenation).
        hud = cv2.hconcat([main_frame, binary_zoom_resized])

        return hud
