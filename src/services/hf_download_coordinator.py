"""Координатор загрузок Hugging Face: реестр задач + жизненный цикл воркеров.

Держит словарь задач (worker/статус/процент/сообщение) и управляет
HfModelDownloader-потоками; таблицу загрузок и диалоги рендерит LlamaGUI,
подключённая к сигналам. В тестах воркеры подменяются через worker_factory.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from src.services.hf_downloader import HfModelDownloader, format_bytes


class HfDownloadCoordinator(QObject):
    """Управление параллельными HF-загрузками без знания про виджеты."""

    # Состояние задачи изменилось — перерисовать строку загрузки.
    task_changed = Signal(str)
    # Изменился процент (кроме перерисовки нужен refresh сводки).
    percent_changed = Signal(str)
    # Воркер завершился с результатом: (ключ, ok, сообщение).
    task_completed = Signal(str, bool, str)
    # QThread.finished: нужно обновить списки локальных файлов.
    task_finished = Signal(str)

    def __init__(self, worker_factory=None):
        super().__init__()
        # None → берём HfModelDownloader лениво в start(), чтобы patch()
        # глобального символа модуля действовал и после создания GUI.
        self._worker_factory = worker_factory
        self._tasks: Dict[str, Dict[str, Any]] = {}

    # -- Реестр задач ------------------------------------------------------

    @staticmethod
    def task_key(repo_id, file_info) -> str:
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        return f"{repo_id}::{filename}"

    def tasks(self) -> Dict[str, Dict[str, Any]]:
        return self._tasks

    def task(self, task_key: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_key)

    def is_running(self, task_key: str) -> bool:
        worker = (self._tasks.get(task_key) or {}).get("worker")
        return bool(worker and worker.isRunning())

    def running_keys(self, task_keys: List[str]) -> List[str]:
        return [key for key in task_keys if self.is_running(key)]

    def active(self, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
        active = []
        for task in self._tasks.values():
            worker = task.get("worker")
            if worker and worker.isRunning() and (
                repo_id is None or task.get("repo_id") == repo_id
            ):
                active.append(task)
        return active

    # -- Запуск --------------------------------------------------------------

    def start(
        self,
        repo_id: str,
        file_info: Dict[str, Any],
        model_dir: str,
        ensure_row: Callable[[], int],
    ) -> str:
        """Запустить загрузку; ensure_row возвращает строку таблицы (старую/новую)."""
        task_key = self.task_key(repo_id, file_info)
        previous = self._tasks.get(task_key, {})
        row = previous.get("row")
        if row is None:
            row = ensure_row()

        worker = (self._worker_factory or HfModelDownloader)(repo_id, [file_info], model_dir)
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        self._tasks[task_key] = {
            "worker": worker,
            "repo_id": repo_id,
            "file_info": file_info,
            "model_dir": model_dir,
            "name": filename,
            "percent": 0,
            "status": "starting",
            "message": "",
            "row": row,
            "item": None,
        }
        worker.progress.connect(
            lambda message, key=task_key: self._on_progress(key, message)
        )
        worker.percent.connect(
            lambda percent, key=task_key: self._on_percent(key, percent)
        )
        worker.completed.connect(
            lambda ok, message, key=task_key: self._on_completed(key, ok, message)
        )
        worker.finished.connect(lambda key=task_key: self.task_finished.emit(key))
        worker.start()
        return task_key

    def upsert_partial(
        self,
        repo_id: str,
        file_info: Dict[str, Any],
        partial: Dict[str, Any],
        model_dir: str,
        ensure_row: Callable[[], int],
    ) -> Optional[str]:
        """Зарегистрировать сохранённый .part ещё до скана репозитория.

        Возвращает ключ задачи или None, если задача уже скачивается.
        """
        if not repo_id or not file_info or not partial:
            return None
        task_key = self.task_key(repo_id, file_info)
        if self.is_running(task_key):
            return None

        previous = self._tasks.get(task_key, {})
        row = previous.get("row")
        if row is None:
            row = ensure_row()

        saved = int(partial.get("partial_size") or 0)
        expected = int(file_info.get("size") or partial.get("expected_size") or 0)
        percent = min(99, int(saved * 100 / expected)) if expected else None
        saved_text = partial.get("partial_size_text") or format_bytes(saved)
        total_text = format_bytes(expected) if expected else "size pending"
        filename = str(file_info.get("rfilename") or file_info.get("name") or "")
        self._tasks[task_key] = {
            "worker": None,
            "repo_id": repo_id,
            "file_info": dict(file_info),
            "model_dir": model_dir,
            "name": f"{repo_id} / {filename}",
            "percent": percent,
            "status": "paused / resumable",
            "message": (
                f"Saved: {saved_text} / {total_text}\n"
                f"{partial.get('partial_path') or ''}"
            ),
            "row": row,
            "item": None,
        }
        return task_key

    # -- Управление ------------------------------------------------------------

    def pause(self, task_keys: List[str]) -> int:
        """Поставить на паузу выбранные работающие задачи. Возвращает число."""
        paused = 0
        for key in task_keys:
            task = self._tasks.get(key, {})
            worker = task.get("worker")
            if worker and worker.isRunning():
                task["status"] = "pausing"
                worker.pause()
                self.task_changed.emit(key)
                paused += 1
        return paused

    def cancel_and_delete(self, task_keys: List[str]) -> int:
        """Прервать работающие задачи с удалением .part (диалог — на стороне GUI)."""
        count = 0
        for key in task_keys:
            task = self._tasks.get(key)
            if not task:
                continue
            worker = task.get("worker")
            if not (worker and worker.isRunning()):
                continue
            task["status"] = "cancelling"
            worker.cancel_and_delete()
            self.task_changed.emit(key)
            count += 1
        return count

    def mark_partial_deleted(self, task_key: str) -> None:
        task = self._tasks.get(task_key)
        if task is None:
            return
        task["status"] = "cancelled"
        task["percent"] = 0
        task["message"] = "Partial .part deleted"
        self.task_changed.emit(task_key)

    # -- Колбэки воркеров ----------------------------------------------------

    def _on_progress(self, task_key: str, message: str):
        task = self._tasks.get(task_key)
        if not task:
            return
        task["status"] = "downloading"
        task["message"] = message
        self.task_changed.emit(task_key)

    def _on_percent(self, task_key: str, percent):
        task = self._tasks.get(task_key)
        if not task:
            return
        task["percent"] = int(percent)
        self.task_changed.emit(task_key)
        self.percent_changed.emit(task_key)

    def _on_completed(self, task_key: str, ok: bool, message: str):
        task = self._tasks.get(task_key)
        if task:
            task["status"] = "complete" if ok else "stopped"
            task["message"] = message
            if ok:
                task["percent"] = 100
            self.task_changed.emit(task_key)
        self.task_completed.emit(task_key, ok, message)
