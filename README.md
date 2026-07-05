# 🚨 face-Alert: Smart Proctoring System

`face-Alert` is a state-of-the-art, desktop-based **Smart Proctoring System** designed to monitor user integrity in real-time during online examinations, remote interviews, or secure sessions. Built with **Python**, **OpenCV**, **MediaPipe**, and a modern **CustomTkinter** graphical interface, the application tracks gaze, detects multiple individuals, logs violations, and keeps a persistent integrity score.

---

## 🌟 Key Features

*   **🤖 AI-Powered Face Detection:** Uses Google's MediaPipe Face Landmarker model to detect faces and extract high-fidelity landmarks in real time.
*   **👁️ Precise Gaze Tracking:** Monitors pupil movement relative to a calibrated baseline to detect if the user is looking away/aside.
*   **📊 Dynamic Integrity Scoring:** Starts with an integrity score of `100` and automatically penalizes scores if suspicious behavior (e.g., looking aside, face missing) persists.
*   **👥 Multi-Person Detection:** Triggers immediate violations if unauthorized people enter the camera view.
*   **🔄 Auto & Manual Calibration:** Calibrates baseline eye position automatically upon session start or manually on-demand.
*   **📸 Incident Screenshot Capture:** Automatically captures and saves frame screenshots when violations occur, stored securely in an `evidence/` directory.
*   **🗄️ SQLite Logging:** Persists all proctoring logs and session details (timestamps, event types, descriptions, integrity scores, screenshot paths) in a local database (`proctoring_log.db`).
*   **🔊 Audio Warnings:** Provides immediate feedback using audio beeps when a user violates rules.
*   **🎨 Premium Dark GUI:** Features a responsive, modern dashboard built with CustomTkinter incorporating custom gauges, real-time status indicators, and live binary eye-tracking zoom view.

---

## 🛠️ Technologies Used

*   **Programming Language:** Python 3
*   **GUI Framework:** CustomTkinter (Modern Tkinter fork)
*   **Computer Vision & AI:** OpenCV (`opencv-python`), MediaPipe (`mediapipe`)
*   **Database:** SQLite 3
*   **Utilities:** NumPy, Pillow (PIL), Winsound

---

## 📂 Project Structure

```text
Final Project/
├── main.py                  # Application entry point
├── download_models.py       # Helper script to download MediaPipe models
├── requirements.txt         # Project dependencies
├── proctoring_log.db        # SQLite database containing logs (auto-generated)
├── evidence/                # Directory containing violation screenshots (auto-generated)
├── models/                  # Directory containing AI model weights
│   └── face_landmarker.task
└── src/                     # Core application source code
    ├── __init__.py
    ├── detector.py          # MediaPipe face & eye ROI detection logic
    ├── processor.py         # Image preprocessing for eye regions (CLAHE, blur, adaptive thresholding)
    ├── tracker.py           # Gaze calibration, deviation checks, and integrity scoring
    ├── ui.py                # Drawing visual overlays (boxes, status, HUD)
    ├── notifier.py          # Audio alerts and warnings
    ├── recorder.py          # SQLite database connection and screenshot capture logic
    └── gui.py               # CustomTkinter dashboard layout and main application loop
```

---

## 📦 Installation & Setup

Follow these steps to run `face-Alert` locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Tokyo12q/face-Alert.git
cd face-Alert
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed. Install the required python packages:
```bash
pip install -r requirements.txt
```

*Note: The primary dependencies include `opencv-python`, `mediapipe`, `customtkinter`, `numpy`, and `pillow`.*

### 3. Download the AI Model
The app automatically downloads the MediaPipe Face Landmarker model on first run, but you can download it manually:
```bash
python download_models.py
```

---

## 🖥️ How to Run

Run the main script to start the dashboard:
```bash
python main.py
```

### How to Use:
1. Click **Start Session** to turn on the camera and initialize the system.
2. The system will start in **Auto-Calibrating** mode. Look directly at the screen for 5 seconds to calibrate the gaze baseline.
3. Once calibrated, the system status changes to **NORMAL** and your integrity score begins monitoring.
4. If you look away/aside for more than 3 seconds, a suspicion warning will be triggered, and persistent violations will decrease your integrity score.
5. If another person enters the frame, a **MULTI-PERSON** violation is instantly logged.
6. Click **Recalibrate Gaze** at any time if your baseline shifts.
7. Click **End Session** to close the camera feed and view the session summary in the terminal.

---

## 🗄️ Database & Event Logs

All event statistics are saved inside `proctoring_log.db` in the `events` table:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER (PK) | Auto-incrementing primary key |
| `timestamp` | TEXT | Human-readable date and time of the event |
| `event_type` | TEXT | Classification (e.g., `VIOLATION`) |
| `description` | TEXT | Description of the action (e.g., `Face missing`, `Multi-face`, `Look aside`) |
| `integrity_score` | REAL | The user's integrity score at the moment of the event |
| `screenshot_path` | TEXT | File path to the captured evidence image (if applicable) |

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.