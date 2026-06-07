# ─────────────────────────────────────────────────────────────────────────────
# src/main.py  –  Application Entry Point
# Purpose : The single file you run to launch the Smart Proctoring System.
#            It imports the main GUI window class and starts the Tkinter event
#            loop so the application window appears on screen.
# ─────────────────────────────────────────────────────────────────────────────

# Import the ProctorDashboard class from gui.py (same src/ package).
# ProctorDashboard is the main application window (inherits from CTk/Tk).
from src.gui import ProctorDashboard


def main():
    """Create the application window and start the GUI event loop."""

    # Instantiate the ProctorDashboard window.
    # This triggers __init__() inside gui.py which sets up the camera,
    # detectors, and builds all the UI widgets.
    app = ProctorDashboard()

    # Register a callback for when the user clicks the window's × (close) button.
    # "WM_DELETE_WINDOW" is the Tkinter protocol name for that action.
    # app.on_closing() will safely stop the camera and release resources before
    # the window is destroyed.
    app.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the Tkinter main event loop.
    # mainloop() blocks here and keeps the window alive, processing user events
    # (button clicks, key presses, frame updates) until the window is closed.
    app.mainloop()


# Only run main() when this script is executed directly.
# If another file ever imports main.py, this block is skipped.
if __name__ == "__main__":
    main()
