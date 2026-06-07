import winsound
import time

class ProctorNotifier:
    def __init__(self, cooldown_seconds=2):
        self.cooldown_seconds = cooldown_seconds
        self.last_played_time = 0

    def play_alert_sound(self):
        """Plays a system beep if the cooldown has passed."""
        current_time = time.time()
        if current_time - self.last_played_time > self.cooldown_seconds:
            # frequency: 1000Hz, duration: 500ms
            winsound.Beep(1000, 500)
            self.last_played_time = current_time
