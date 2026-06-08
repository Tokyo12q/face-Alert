import cv2
import numpy as np

from PIL import Image, ImageTk

class ProctorUI:

    @staticmethod
    def convert_frame(cv_frame, width, height):

        cv_frame = cv_frame.copy()

        cv_frame = cv_frame.astype(np.uint8)

        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_frame)

        pil_image = pil_image.resize((width, height), Image.LANCZOS)

        return ImageTk.PhotoImage(image=pil_image)

    @staticmethod
    def draw_overlays(frame, faces, primary_face, eye_coords, status):

        if "VIOLATION" in status or "SUSPICION" in status:
            color = (0, 0, 255)

        elif "AUTO-CALIBRATING" in status:
            color = (0, 255, 255)

        else:
            color = (0, 255, 0)

        for face in faces:

            fx, fy, fw, fh = face['bbox']

            is_primary = (primary_face is not None and
                          face['bbox'] == primary_face['bbox'])

            box_color = color if is_primary else (255, 0, 255)

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

            if not is_primary:
                cv2.putText(frame, "UNAUTHORIZED PERSON", (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            else:
                for i, lm_pixel in enumerate(face['coords']):

                    if i % 10 == 0:
                        cv2.circle(frame,
                                   (int(lm_pixel[0]), int(lm_pixel[1])),
                                   1, (0, 255, 255), -1)

                for eye_indices in [[33, 133], [362, 263]]:
                    for idx in eye_indices:
                        pt = face['coords'][idx]

                        cv2.circle(frame,
                                   (int(pt[0]), int(pt[1])),
                                   2, (255, 255, 0), -1)

        for (ex, ey, ew, eh) in eye_coords:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh),
                          (255, 255, 0), 2)

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    @staticmethod
    def create_hud(main_frame, binary_eye_zoom):

        h, w = main_frame.shape[:2]

        zoom_h = h

        zoom_w = int(binary_eye_zoom.shape[1] * (zoom_h / binary_eye_zoom.shape[0]))

        binary_zoom_resized = cv2.resize(binary_eye_zoom, (zoom_w, zoom_h))

        if len(binary_zoom_resized.shape) == 2:
            binary_zoom_resized = cv2.cvtColor(binary_zoom_resized,
                                               cv2.COLOR_GRAY2BGR)

        cv2.putText(main_frame, "LIVE FEED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(binary_zoom_resized, "EYE TRACKING ZOOM", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        hud = cv2.hconcat([main_frame, binary_zoom_resized])

        return hud
