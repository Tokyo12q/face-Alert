# ─────────────────────────────────────────────────────────────────────────────
# src/recorder.py  –  Event Logger & Evidence Collector
# Purpose : Persists all proctoring violations to a SQLite database and
#           saves screenshot images as evidence.  At the end of a session,
#           it prints a summary report to the console.
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3   # Built-in Python module for working with SQLite databases.
                 # SQLite stores the entire database as a single .db file –
                 # no separate database server required.

import cv2       # OpenCV – used to write JPEG screenshots to disk.

import os        # Built-in – file/directory existence checks and path building.

import time      # Built-in – wall-clock timestamps for screenshot cooldown.

import shutil    # Built-in – high-level file operations (recursive folder delete).

from datetime import datetime  # Built-in – human-readable timestamp formatting.


class ProctorRecorder:
    """
    Handles all persistence for the proctoring session:
      • Creates/clears the evidence screenshot folder on startup.
      • Creates a SQLite database with an 'events' table.
      • log_event()       – inserts a row and optionally saves a screenshot.
      • generate_summary() – queries the DB and prints session statistics.
    """

    def __init__(self, db_path="proctoring_log.db", evidence_dir="evidence"):
        """
        Parameters
        ----------
        db_path      : str – path to the SQLite database file.
        evidence_dir : str – folder where screenshot images are saved.
        """

        # Store the evidence folder path for use by other methods.
        self.evidence_dir = evidence_dir

        # Wipe all screenshots from the previous session so old evidence
        # does not mix with the new session.
        self._clear_evidence()

        # If the evidence folder was deleted by _clear_evidence (or never
        # existed), recreate it now so cv2.imwrite() can save files into it.
        if not os.path.exists(evidence_dir):
            os.makedirs(evidence_dir)

        # Open (or create) the SQLite database file at db_path.
        # sqlite3.connect() returns a Connection object.
        self.conn = sqlite3.connect(db_path)

        # Create a Cursor object used to execute SQL statements.
        self.cursor = self.conn.cursor()

        # Execute a CREATE TABLE statement.
        # "CREATE TABLE IF NOT EXISTS" is safe to run on every startup:
        # it only creates the table if it does not already exist.
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

        # Flush the CREATE TABLE change to disk.
        self.conn.commit()

        # Timestamp of the last screenshot to enforce a cooldown between saves.
        self.last_screenshot_time = 0

        # Minimum gap (in seconds) between consecutive screenshots.
        # Prevents the disk from filling up with near-duplicate images.
        self.screenshot_cooldown = 5  # seconds

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _clear_evidence(self):
        """Delete all files (and sub-folders) inside the evidence directory."""

        # Only attempt deletion if the folder actually exists.
        if os.path.exists(self.evidence_dir):
            print(f"[recorder] Clearing old evidence from {self.evidence_dir}...")

            # Iterate over every item (file or folder) in the directory.
            for filename in os.listdir(self.evidence_dir):

                # Build the full path to this item.
                file_path = os.path.join(self.evidence_dir, filename)

                try:
                    # os.path.isfile – regular file.
                    # os.path.islink – symbolic link (treat same as file).
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        # os.unlink() deletes a single file or symlink.
                        os.unlink(file_path)

                    elif os.path.isdir(file_path):
                        # shutil.rmtree() recursively deletes a folder and all
                        # of its contents (like "rm -rf" on Linux).
                        shutil.rmtree(file_path)

                except Exception as e:
                    # If deletion fails (e.g. file locked), print a warning
                    # but continue clearing the rest of the folder.
                    print(f'Failed to delete {file_path}. Reason: {e}')

    # ─── Public API ───────────────────────────────────────────────────────────

    def log_event(self, event_type, description, integrity_score, frame=None):
        """
        Record a proctoring event in the database and optionally save a
        screenshot.

        Parameters
        ----------
        event_type      : str        – category label, e.g. "VIOLATION".
        description     : str        – detail message, e.g. "Multi-face".
        integrity_score : float      – current integrity score (0-100).
        frame           : np.ndarray – current webcam frame (BGR), or None.
        """

        # Format the current date and time as a readable string.
        # strftime format: "YYYY-MM-DD HH:MM:SS"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Assume no screenshot unless we decide to take one below.
        screenshot_path = None

        # Save a screenshot only if:
        #   1. A frame was provided.
        #   2. The event is a VIOLATION (not just a suspicion).
        #   3. Enough time has passed since the last screenshot (cooldown).
        if frame is not None and "VIOLATION" in event_type:
            if time.time() - self.last_screenshot_time > self.screenshot_cooldown:

                # Build a unique filename using the current time (HHMMSS).
                filename = f"violation_{datetime.now().strftime('%H%M%S')}.jpg"

                # Full path where the JPEG will be saved.
                screenshot_path = os.path.join(self.evidence_dir, filename)

                # Write the BGR frame to disk as a JPEG image.
                cv2.imwrite(screenshot_path, frame)

                # Update the cooldown timestamp.
                self.last_screenshot_time = time.time()

        # Insert a new row into the 'events' table.
        # The '?' placeholders are safely filled in by SQLite (prevents SQL injection).
        self.cursor.execute('''
            INSERT INTO events
                (timestamp, event_type, description, integrity_score, screenshot_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, event_type, description, integrity_score, screenshot_path))

        # Flush the INSERT to disk immediately so data is not lost on crash.
        self.conn.commit()

    def generate_summary(self):
        """
        Query the database and print a summary of the proctoring session
        to the console.  Called when the operator clicks "Stop System".
        """

        # Count the total number of VIOLATION events in the database.
        # LIKE "%VIOLATION%" matches any event_type containing "VIOLATION".
        self.cursor.execute(
            'SELECT COUNT(*) FROM events WHERE event_type LIKE "%VIOLATION%"'
        )
        # fetchone() returns a single row as a tuple; [0] extracts the count.
        violations = self.cursor.fetchone()[0]

        # Find the lowest integrity score recorded during the session.
        # MIN() returns NULL if the table is empty, which we handle below.
        self.cursor.execute('SELECT MIN(integrity_score) FROM events')
        min_score = self.cursor.fetchone()[0]

        # Print the summary to the console.
        print("\n--- PROCTORING SESSION SUMMARY ---")
        print(f"Total Violations: {violations}")

        # If min_score is None (no events were logged), default to 100.
        print(f"Lowest Integrity Score: {int(min_score if min_score is not None else 100)}")

        # Show the absolute path to the evidence folder.
        print(f"Evidence saved in: {os.path.abspath(self.evidence_dir)}")
        print("----------------------------------\n")

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def __del__(self):
        """Close the database connection when this object is garbage-collected."""
        # Closing the connection flushes any pending writes and releases the
        # file lock on the .db file so other processes can access it.
        self.conn.close()
