"""Координация запуска llama-server: отложенные рестарты и env.

ServerLaunchController держит состояние "рестарт запрошен, ждём остановки
текущего сервера" и флаг "запущенному серверу нужны новые параметры".
Сам запуск/остановку процессов и UI выполняет LlamaGUI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QObject, Signal

LaunchTuple = Tuple[str, list, Optional[Dict[str, str]]]


class ServerLaunchController(QObject):
    pending_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self._restart_pending = False
        self._pending_restart_launch: Optional[LaunchTuple] = None
        self._restart_needed = False

    # -- Отложенный рестарт -------------------------------------------------

    @property
    def is_pending(self) -> bool:
        return self._restart_pending

    def request_restart(self, launch: LaunchTuple) -> None:
        """Запомнить запуск и ждать остановки текущего сервера."""
        self._pending_restart_launch = launch
        self._restart_pending = True
        self.pending_changed.emit(True)

    def cancel_pending(self) -> bool:
        """Отменить отложенный рестарт. True — если он был запрошен."""
        had = self._restart_pending
        self._restart_pending = False
        self._pending_restart_launch = None
        if had:
            self.pending_changed.emit(False)
        return had

    def poll_pending(self, server_running: bool) -> Tuple[bool, Optional[LaunchTuple]]:
        """Опросить отложенный рестарт.

        Возвращает (had_pending, launch):
          (False, None)  — рестарт не был запрошен;
          (True, None)   — сервер ещё работает, нужно опросить позже;
          (True, launch) — сервер остановился, запускай launch.
        """
        if not self._restart_pending:
            return False, None
        if server_running:
            return True, None
        launch = self._pending_restart_launch
        self._restart_pending = False
        self._pending_restart_launch = None
        self.pending_changed.emit(False)
        return True, launch

    # -- Индикатор "нужен рестарт" -------------------------------------------

    @property
    def restart_needed(self) -> bool:
        return self._restart_needed

    def mark_restart_needed(self) -> None:
        self._restart_needed = True

    def clear_restart_needed(self) -> None:
        self._restart_needed = False

    # -- Окружение запуска -----------------------------------------------------

    @staticmethod
    def env_from_settings(settings: Any) -> Dict[str, str]:
        env: Dict[str, str] = {}
        cuda_visible = str(
            getattr(settings, "cuda_visible_devices", "") or ""
        ).strip()
        cuda_loading = str(
            getattr(settings, "cuda_module_loading", "") or ""
        ).strip()
        if cuda_visible:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        if cuda_loading:
            env["CUDA_MODULE_LOADING"] = cuda_loading
        return env
