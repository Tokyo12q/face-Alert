# ─────────────────────────────────────────────────────────────────────────────
# src/notifier.py  –  Alert Sound Manager
# Purpose : Plays a short audible beep whenever a proctoring violation is
#           detected.  A cooldown timer prevents the beep from firing so
#           rapidly that it becomes annoying or blocks the main thread.
# ─────────────────────────────────────────────────────────────────────────────

import winsound  # Windows-only built-in module that can produce system beeps
                 # using the PC speaker or the default audio device.

import time      # Built-in module for getting the current timestamp so we can
                 # measure how long it has been since the last beep.


class ProctorNotifier:
    def __init__(self, cooldown_seconds=2):
        """
        Initialise the notifier.

        cooldown_seconds : Minimum gap (in seconds) between two consecutive
                           beeps.  Defaults to 2 s to avoid sound spam.
        """

        # Store how many seconds must pass before another beep is allowed.
        self.cooldown_seconds = cooldown_seconds

        # Record the timestamp of the last beep.  Initialising to 0 means
        # the very first call to play_alert_sound() will always be allowed
        # (because any real time.time() value is > 0).
        self.last_played_time = 0

    def play_alert_sound(self):
        """Plays a system beep if the cooldown has passed."""

        # Get the current time as a Unix timestamp (seconds since 1970-01-01).
        current_time = time.time()

        # Only beep if enough time has elapsed since the previous beep.
        # (current_time - self.last_played_time) is the number of seconds
        # that have passed since the last beep.
        if current_time - self.last_played_time > self.cooldown_seconds:

            # winsound.Beep(frequency_hz, duration_ms)
            # 1000 Hz tone for 500 ms – a short, clearly audible alert.
            winsound.Beep(1000, 500)

            # Update the timestamp so the next beep won't fire for at least
            # cooldown_seconds seconds.
            self.last_played_time = current_time
