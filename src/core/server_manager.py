"""Управление процессами llama-server и llama-bench."""

import os
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from src.core.constants import KILL_TIMEOUT_SERVER, KILL_TIMEOUT_BENCHMARK


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

        self.server_proc.readyReadStandardOutput.connect(self._srv_stdout)
        self.server_proc.readyReadStandardError.connect(self._srv_stderr)
        self.server_proc.stateChanged.connect(self._srv_state)
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
        self._emit(data, "info")

    def _srv_stderr(self):
        data = (
            self.server_proc.readAllStandardError()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self._emit(data, "error")

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
            if self.server_stop_requested:
                self._emit("⏹ Сервер остановлен")
            else:
                self._emit(f"⏹ Сервер остановлен (код: {self.server_proc.exitCode()})")
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
        self._prepare_process_environment(self.server_proc, exe, env)
        self.server_proc.start(exe, args)
        self.state_changed.emit(True)

    def stop_server(self):
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            if self.server_stop_requested:
                return
            self.server_stop_requested = True
            self._emit("⏹ Остановка сервера...")
            self.server_proc.terminate()
            QTimer.singleShot(KILL_TIMEOUT_SERVER, self._kill_server_if_needed)

    def force_stop_server(self):
        if self.server_proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.server_stop_requested = True
        self._emit("⛔ Force stop: принудительная остановка llama-server...")
        pid = int(self.server_proc.processId() or 0)
        if pid and sys.platform.startswith("win"):
            QProcess.execute("taskkill", ["/PID", str(pid), "/T", "/F"])
        if self.server_proc.state() != QProcess.ProcessState.NotRunning:
            self.server_proc.kill()

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
            self.server_proc.terminate()
            if not self.server_proc.waitForFinished(2000):
                self.server_proc.kill()
        if self.is_bench_running():
            self.bench_proc.terminate()
            if not self.bench_proc.waitForFinished(2000):
                self.bench_proc.kill()
