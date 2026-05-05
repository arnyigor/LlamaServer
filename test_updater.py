"""Тестовый скрипт для проверки updater без GUI."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.threads import LlamaCppUpdater
from PySide6.QtCore import QCoreApplication, QTimer


def test_updater():
    app = QCoreApplication(sys.argv)

    test_path = r"C:\path\to\llama-server.exe"

    print(f"Testing updater with path: {test_path}")
    print(f"File exists: {os.path.exists(test_path)}")

    if not os.path.exists(test_path):
        print("ERROR: File does not exist!")
        return

    updater = LlamaCppUpdater(test_path)

    def on_progress(text):
        print(f"PROGRESS: {text}")

    def on_percent(value):
        print(f"PERCENT: {value}%")

    def on_completed(changed, message):
        print(f"COMPLETED: changed={changed}, message={message}")
        app.quit()

    updater.progress.connect(on_progress)
    updater.percent.connect(on_percent)
    updater.completed.connect(on_completed)

    print("Starting updater...")
    updater.start()

    QTimer.singleShot(
        30000,
        lambda: (
            print("WARNING: Timeout after 30s") if updater.isRunning() else None,
            updater.requestInterruption() if updater.isRunning() else None,
            updater.wait(5000) if updater.isRunning() else None,
            app.quit(),
        ),
    )

    print("Test finished")


if __name__ == "__main__":
    test_updater()
