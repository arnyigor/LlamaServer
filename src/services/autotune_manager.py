"""QThread-оркестратор AutoTune поверх движка llama_autotuner."""

from __future__ import annotations

import threading
from typing import List

from PySide6.QtCore import QThread, Signal

from llama_autotuner.models import CandidateResult
from llama_autotuner.session import (
    AutotuneSessionError,
    SessionConfig,
    SessionResult,
    run_session,
)


class AutoTuneManager(QThread):
    """Драйвит один автотюн-сеанс движка llama_autotuner в фоновом потоке."""

    log = Signal(str, str)  # (message, level)
    result_ready = Signal(object)  # CandidateResult, эмитится после каждого кандидата
    progress = Signal(int, int)  # (completed_runs, max_runs)
    session_finished = Signal(object)  # SessionResult
    session_failed = Signal(str)  # текст ошибки — ни один кандидат не запускался

    def __init__(self, config: SessionConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.results: List[CandidateResult] = []
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self.requestInterruption()
        self._cancel_event.set()

    def _emit_progress(self, message: str) -> None:
        self.log.emit(message, "info")

    def _emit_result(self, result: CandidateResult) -> None:
        self.results.append(result)
        self.result_ready.emit(result)
        self.progress.emit(len(self.results), self.config.max_runs or 0)

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            session_result: SessionResult = run_session(
                self.config,
                progress=self._emit_progress,
                on_result=self._emit_result,
                cancel_event=self._cancel_event,
            )
        except AutotuneSessionError as exc:
            self.session_failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive, unexpected engine crash
            self.session_failed.emit(f"Autotune session crashed: {exc}")
            return
        self.session_finished.emit(session_result)
