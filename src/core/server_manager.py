"""Управление процессами llama-server и llama-bench."""

import os
import subprocess
import sys
import time
from collections import deque

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from src.core.constants import KILL_TIMEOUT_SERVER, KILL_TIMEOUT_BENCHMARK
from src.utils.subprocess_utils import no_console_kwargs


class ServerManager(QObject):
    log_received = Signal(str, str)  # text, level
    state_changed = Signal(bool)  # is_busy
    bench_finished = Signal(int)  # exit_code
    server_stopped = Signal()  # server stopped

    def __init__(self):
        super().__init__()
        self.server_proc = QProcess()
        self.bench_proc = QProcess()
        self.server_stop_requested = False
        self.bench_stop_requested = False
        self.server_recent_output = deque(maxlen=300)
        self.server_last_exit_code = 0
        self.server_last_crash_exit = False
        self.server_last_stop_requested = False
        self.server_last_process_error = ""
        self.server_last_runtime_seconds = 0.0
        self._server_started_at = 0.0
        self._server_stop_notified = False

        self.server_proc.readyReadStandardOutput.connect(self._srv_stdout)
        self.server_proc.readyReadStandardError.connect(self._srv_stderr)
        self.server_proc.stateChanged.connect(self._srv_state)
        self.server_proc.finished.connect(self._srv_finished)
        self.server_proc.errorOccurred.connect(self._srv_process_error)
        self.bench_proc.readyReadStandardOutput.connect(self._bench_stdout)
        self.bench_proc.readyReadStandardError.connect(self._bench_stderr)
        self.bench_proc.finished.connect(self._bench_finished)

    def _emit(self, text, level="info"):
        if text.strip():
            self.log_received.emit(text, level)

    def _srv_stdout(self):
        data = (
            self.server_proc.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self._remember_server_output(data)
        self._emit(data, "info")

    def _srv_stderr(self):
        data = (
            self.server_proc.readAllStandardError()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self._remember_server_output(data)
        # llama.cpp writes much of its normal status output to stderr. Let the
        # log manager color only lines that actually contain an error marker.
        self._emit(data, "info")

    def _remember_server_output(self, text: str):
        for line in str(text or "").splitlines():
            line = line.rstrip()
            if line:
                self.server_recent_output.append(line)

    def recent_server_output(self) -> str:
        return "\n".join(self.server_recent_output)

    def _bench_stdout(self):
        data = (
            self.bench_proc.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self._emit(data, "bench")

    def _bench_stderr(self):
        data = (
            self.bench_proc.readAllStandardError()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self._emit(data, "error")

    def _srv_state(self, state):
        if state == QProcess.ProcessState.NotRunning:
            self.state_changed.emit(False)

    def _srv_finished(self, code, exit_status):
        # Drain bytes that arrived together with process termination before the
        # diagnostic report snapshots the ring buffer.
        self._srv_stdout()
        self._srv_stderr()
        self.server_last_exit_code = int(code)
        status_value = int(getattr(exit_status, "value", exit_status))
        self.server_last_crash_exit = status_value != 0
        self._notify_server_stopped()

    def _srv_process_error(self, process_error):
        error_value = int(getattr(process_error, "value", process_error))
        descriptions = {
            0: "не удалось запустить процесс",
            1: "процесс аварийно завершился",
            2: "истекло время ожидания процесса",
            3: "ошибка чтения из процесса",
            4: "ошибка записи в процесс",
            5: "неизвестная ошибка QProcess",
        }
        self.server_last_process_error = descriptions.get(
            error_value, f"ошибка QProcess {error_value}"
        )
        self._emit(
            f"❌ Ошибка процесса llama-server: {self.server_last_process_error}",
            "error",
        )
        if error_value == 0:
            self.server_last_exit_code = int(self.server_proc.exitCode())
            QTimer.singleShot(0, self._notify_server_stopped)

    def _notify_server_stopped(self):
        if self._server_stop_notified:
            return
        self._server_stop_notified = True
        self.server_last_stop_requested = bool(self.server_stop_requested)
        self.server_last_runtime_seconds = max(
            0.0, time.monotonic() - self._server_started_at
        ) if self._server_started_at else 0.0
        if self.server_stop_requested and not self.server_last_crash_exit:
            self._emit("⏹ Сервер остановлен")
        else:
            marker = "аварийно" if self.server_last_crash_exit else "неожиданно"
            self._emit(
                f"⏹ Сервер остановлен {marker} (код: {self.server_last_exit_code})",
                "error",
            )
        self.server_stop_requested = False
        self.server_stopped.emit()

    def _bench_finished(self, code):
        self.state_changed.emit(False)
        if self.bench_stop_requested:
            self._emit("⏹ Тестирование остановлено")
        elif code == 0:
            self._emit("✅ Тестирование завершено успешно")
        else:
            self._emit(f"❌ Ошибка тестирования (код: {code})", "error")
        self.bench_stop_requested = False
        self.bench_finished.emit(code)

    def _prepare_process_environment(
        self, proc: QProcess, exe: str, env: dict | None = None
    ):
        exe_dir = os.path.dirname(os.path.abspath(exe)) if exe else ""
        if exe_dir:
            proc.setWorkingDirectory(exe_dir)

        process_env = QProcessEnvironment.systemEnvironment()
        path_key = "Path" if sys.platform.startswith("win") else "PATH"
        alt_path_key = "PATH" if path_key == "Path" else "Path"
        current_path = process_env.value(path_key) or process_env.value(alt_path_key)
        if exe_dir and exe_dir not in current_path.split(os.pathsep):
            process_env.insert(
                path_key,
                exe_dir + (os.pathsep + current_path if current_path else ""),
            )
        for key, value in (env or {}).items():
            if key and value is not None and str(value).strip() != "":
                process_env.insert(str(key), str(value))
        proc.setProcessEnvironment(process_env)

    def start_server(self, exe: str, args: list, env: dict | None = None):
        self.server_stop_requested = False
        self.server_recent_output.clear()
        self.server_last_exit_code = 0
        self.server_last_crash_exit = False
        self.server_last_stop_requested = False
        self.server_last_process_error = ""
        self.server_last_runtime_seconds = 0.0
        self._server_started_at = time.monotonic()
        self._server_stop_notified = False
        self._prepare_process_environment(self.server_proc, exe, env)
        self.server_proc.start(exe, args)
        self.state_changed.emit(True)

    def stop_server(self):
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            if self.server_stop_requested:
                self._emit("⛔ Stop pressed again: forcing llama-server shutdown...")
                self.force_stop_server()
                return
            self.server_stop_requested = True
            self._emit("⏹ Остановка сервера...")
            self.server_proc.terminate()
            QTimer.singleShot(KILL_TIMEOUT_SERVER, self._kill_server_if_needed)

    def force_stop_server(self):
        if self.server_proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.server_stop_requested = True
        self._emit("⛔ Force stop: killing llama-server process tree to free RAM/VRAM...")
        pid = int(self.server_proc.processId() or 0)
        if pid and sys.platform.startswith("win"):
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=8,
                    check=False,
                    **no_console_kwargs(),
                )
            except Exception:
                pass
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            self.server_proc.kill()
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            self.server_proc.waitForFinished(2000)

    def _kill_server_if_needed(self):
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            self._emit("⚠️ Сервер не завершился штатно, принудительная остановка")
            self.force_stop_server()

    def start_bench(self, exe: str, args: list, env: dict | None = None):
        self.bench_stop_requested = False
        self._prepare_process_environment(self.bench_proc, exe, env)
        self.bench_proc.start(exe, args)
        self.state_changed.emit(True)

    def stop_bench(self):
        if self.bench_proc.state() != QProcess.ProcessState.NotRunning:
            self.bench_stop_requested = True
            self._emit("⏹ Остановка benchmark...")
            self.bench_proc.terminate()
            QTimer.singleShot(KILL_TIMEOUT_BENCHMARK, self._kill_bench_if_needed)

    def _kill_bench_if_needed(self):
        if self.bench_proc.state() != QProcess.ProcessState.NotRunning:
            self._emit("⚠️ Benchmark не завершился штатно, принудительная остановка")
            self.bench_proc.kill()

    def is_server_running(self):
        return self.server_proc.state() != QProcess.ProcessState.NotRunning

    def is_bench_running(self):
        return self.bench_proc.state() != QProcess.ProcessState.NotRunning

    def terminate_all(self):
        if self.is_server_running():
            self.server_stop_requested = True
            self.server_proc.terminate()
            if not self.server_proc.waitForFinished(2000):
                self.server_proc.kill()
        if self.is_bench_running():
            self.bench_proc.terminate()
            if not self.bench_proc.waitForFinished(2000):
                self.bench_proc.kill()
