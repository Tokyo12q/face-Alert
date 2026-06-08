from src.gui import ProctorDashboard

def main():

    app = ProctorDashboard()

    app.protocol("WM_DELETE_WINDOW", app.on_closing)

    app.mainloop()

if __name__ == "__main__":
    main()
