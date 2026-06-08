import sqlite3

import cv2

import os

import time

import shutil

from datetime import datetime

class ProctorRecorder:

    def __init__(self, db_path="proctoring_log.db", evidence_dir="evidence"):

        self.evidence_dir = evidence_dir

        self._clear_evidence()

        if not os.path.exists(evidence_dir):
            os.makedirs(evidence_dir)

        self.conn = sqlite3.connect(db_path)

        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,      -- human-readable date/time string
                event_type      TEXT,      -- e.g. "VIOLATION"
                description     TEXT,      -- e.g. "Face missing"
                integrity_score REAL,      -- score at the time of the event
                screenshot_path TEXT       -- path to the saved JPEG, or NULL
            )
        ''')

        self.conn.commit()

        self.last_screenshot_time = 0

        self.screenshot_cooldown = 5

    def _clear_evidence(self):

        if os.path.exists(self.evidence_dir):
            print(f"[recorder] Clearing old evidence from {self.evidence_dir}...")

            for filename in os.listdir(self.evidence_dir):

                file_path = os.path.join(self.evidence_dir, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def log_event(self, event_type, description, integrity_score, frame=None):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        screenshot_path = None

        if frame is not None and "VIOLATION" in event_type:
            if time.time() - self.last_screenshot_time > self.screenshot_cooldown:

                filename = f"violation_{datetime.now().strftime('%H%M%S')}.jpg"

                screenshot_path = os.path.join(self.evidence_dir, filename)

                cv2.imwrite(screenshot_path, frame)

                self.last_screenshot_time = time.time()

        self.cursor.execute('''
            INSERT INTO events
                (timestamp, event_type, description, integrity_score, screenshot_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, event_type, description, integrity_score, screenshot_path))

        self.conn.commit()

    def generate_summary(self):

        self.cursor.execute(
            'SELECT COUNT(*) FROM events WHERE event_type LIKE "%VIOLATION%"'
        )
        violations = self.cursor.fetchone()[0]

        self.cursor.execute('SELECT MIN(integrity_score) FROM events')
        min_score = self.cursor.fetchone()[0]

        print("\n--- PROCTORING SESSION SUMMARY ---")
        print(f"Total Violations: {violations}")

        print(f"Lowest Integrity Score: {int(min_score if min_score is not None else 100)}")

        print(f"Evidence saved in: {os.path.abspath(self.evidence_dir)}")
        print("----------------------------------\n")

    def __del__(self):
        self.conn.close()
