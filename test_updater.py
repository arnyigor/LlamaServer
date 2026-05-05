"""Тестовый скрипт для проверки updater без GUI."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.threads import LlamaCppUpdater
from PySide6.QtCore import QCoreApplication, QThread
import time


def test_updater():
    app = QCoreApplication(sys.argv)

    # Тестовый путь (замените на реальный)
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

    # Ждем завершения или таймаута
    timer = QThread()
    time.sleep(30)  # Ждем 30 секунд

    if updater.isRunning():
        print("WARNING: Updater still running after 30s")
        updater.requestInterruption()
        updater.wait(5000)

    print("Test finished")


if __name__ == "__main__":
    test_updater()
