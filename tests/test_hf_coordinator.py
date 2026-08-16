"""Тесты HfDownloadCoordinator: реестр задач и жизненный цикл воркеров."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import QObject, Signal

from src.services.hf_download_coordinator import HfDownloadCoordinator


class FakeSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class FakeWorker(QObject):
    instances = []

    def __init__(self, repo_id, files, model_dir):
        super().__init__()
        self.repo_id = repo_id
        self.files = files
        self.model_dir = model_dir
        self.progress = FakeSignal()
        self.percent = FakeSignal()
        self.completed = FakeSignal()
        self.finished = FakeSignal()
        self.running = False
        self.paused = False
        self.cancelled = False
        FakeWorker.instances.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def pause(self):
        self.paused = True

    def cancel_and_delete(self):
        self.cancelled = True
        self.running = False


def _file(name="model.gguf", size=1000):
    return {"name": name, "rfilename": name, "size": size}


class TestHfDownloadCoordinator(unittest.TestCase):
    def setUp(self):
        FakeWorker.instances = []
        self.coordinator = HfDownloadCoordinator(worker_factory=FakeWorker)
        self.rows = []

    def _ensure_row(self):
        self.rows.append(len(self.rows))
        return self.rows[-1]

    def test_task_key_uses_rfilename(self):
        key = HfDownloadCoordinator.task_key("author/model", _file("a.gguf"))
        self.assertEqual(key, "author/model::a.gguf")

    def test_start_registers_task_and_wires_worker(self):
        emitted = []
        self.coordinator.task_changed.connect(lambda key: emitted.append(key))
        key = self.coordinator.start("r/m", _file("a.gguf"), "/tmp", self._ensure_row)
        self.assertEqual(len(FakeWorker.instances), 1)
        worker = FakeWorker.instances[0]
        self.assertTrue(worker.running)
        task = self.coordinator.task(key)
        self.assertEqual(task["status"], "starting")
        self.assertEqual(task["row"], 0)

        worker.progress.emit("50 MiB / 100 MiB")
        self.assertEqual(task["status"], "downloading")
        self.assertEqual(task["message"], "50 MiB / 100 MiB")
        self.assertEqual(emitted, [key])

    def test_percent_updates_value_and_emits(self):
        percents = []
        self.coordinator.percent_changed.connect(lambda key: percents.append(key))
        key = self.coordinator.start("r/m", _file(), "/tmp", self._ensure_row)
        FakeWorker.instances[0].percent.emit(42)
        self.assertEqual(self.coordinator.task(key)["percent"], 42)
        self.assertEqual(percents, [key])

    def test_completed_sets_status_and_percent(self):
        results = []
        self.coordinator.task_completed.connect(
            lambda key, ok, msg: results.append((key, ok, msg))
        )
        key = self.coordinator.start("r/m", _file(), "/tmp", self._ensure_row)
        FakeWorker.instances[0].completed.emit(True, "done")
        task = self.coordinator.task(key)
        self.assertEqual(task["status"], "complete")
        self.assertEqual(task["percent"], 100)
        self.assertEqual(results, [(key, True, "done")])

    def test_is_running_and_running_keys(self):
        key = self.coordinator.start("r/m", _file(), "/tmp", self._ensure_row)
        self.assertTrue(self.coordinator.is_running(key))
        self.assertEqual(self.coordinator.running_keys([key, "missing"]), [key])
        FakeWorker.instances[0].running = False
        self.assertFalse(self.coordinator.is_running(key))

    def test_active_filters_by_repo(self):
        self.coordinator.start("r/one", _file("a.gguf"), "/tmp", self._ensure_row)
        self.coordinator.start("r/two", _file("b.gguf"), "/tmp", self._ensure_row)
        self.assertEqual(len(self.coordinator.active()), 2)
        self.assertEqual(len(self.coordinator.active("r/one")), 1)
        self.assertEqual(self.coordinator.active("r/one")[0]["repo_id"], "r/one")

    def test_pause_counts_only_running(self):
        key1 = self.coordinator.start("r/m", _file("a.gguf"), "/tmp", self._ensure_row)
        key2 = self.coordinator.start("r/m", _file("b.gguf"), "/tmp", self._ensure_row)
        FakeWorker.instances[1].running = False
        paused = self.coordinator.pause([key1, key2])
        self.assertEqual(paused, 1)
        self.assertTrue(FakeWorker.instances[0].paused)
        self.assertFalse(FakeWorker.instances[1].paused)
        self.assertEqual(self.coordinator.task(key1)["status"], "pausing")

    def test_cancel_and_delete_only_running(self):
        key = self.coordinator.start("r/m", _file(), "/tmp", self._ensure_row)
        FakeWorker.instances[0].running = False
        self.assertEqual(self.coordinator.cancel_and_delete([key]), 0)
        FakeWorker.instances[0].running = True
        self.assertEqual(self.coordinator.cancel_and_delete([key]), 1)
        self.assertTrue(FakeWorker.instances[0].cancelled)
        self.assertEqual(self.coordinator.task(key)["status"], "cancelling")

    def test_upsert_partial_computes_percent(self):
        partial = {"partial_size": 500, "partial_size_text": "500 B",
                   "partial_path": "/tmp/a.part"}
        key = self.coordinator.upsert_partial(
            "r/m", _file(size=1000), partial, "/tmp", self._ensure_row
        )
        self.assertEqual(key, "r/m::model.gguf")
        task = self.coordinator.task(key)
        self.assertEqual(task["percent"], 50)
        self.assertEqual(task["status"], "paused / resumable")
        self.assertIn("500 B", task["message"])

    def test_upsert_partial_skips_running_task_and_reuses_row(self):
        partial = {"partial_size": 10, "partial_path": "/tmp/a.part"}
        key = self.coordinator.start("r/m", _file(), "/tmp", self._ensure_row)
        self.assertIsNone(
            self.coordinator.upsert_partial("r/m", _file(), partial, "/tmp",
                                            self._ensure_row)
        )
        # После остановки — .part переиспользует существующую строку.
        FakeWorker.instances[0].running = False
        self.assertEqual(self.rows, [0])  # ensure_row ещё не вызывался повторно
        self.coordinator.upsert_partial("r/m", _file(), partial, "/tmp",
                                        self._ensure_row)
        self.assertEqual(self.rows, [0])  # новая строка не создавалась
        self.assertEqual(self.coordinator.task(key)["row"], 0)

    def test_mark_partial_deleted(self):
        partial = {"partial_size": 10, "partial_path": "/tmp/a.part"}
        key = self.coordinator.upsert_partial("r/m", _file(), partial, "/tmp",
                                              self._ensure_row)
        self.coordinator.mark_partial_deleted(key)
        task = self.coordinator.task(key)
        self.assertEqual(task["status"], "cancelled")
        self.assertEqual(task["percent"], 0)
        self.assertEqual(task["message"], "Partial .part deleted")


if __name__ == "__main__":
    unittest.main()
