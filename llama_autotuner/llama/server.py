from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psutil

from llama_autotuner.llama.api import get_health
from llama_autotuner.subprocess_util import no_console_kwargs


_DRAFT_STATS_RE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated\),\s*mean len\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_draft_stats_line(line: str) -> dict[str, float | int] | None:
    m = _DRAFT_STATS_RE.search(line)
    if not m:
        return None
    return {
        "acceptance": float(m.group(1)),
        "accepted": int(m.group(2)),
        "generated": int(m.group(3)),
        "mean_len": float(m.group(4)),
    }


class ServerStartError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID plus creation time, so delayed cleanup cannot hit a reused PID."""

    pid: int
    create_time: float | None = None


@dataclass(slots=True)
class StartupCleanupResult:
    """Auditable result of the bounded stale-server cleanup pass."""

    registered_seen: int = 0
    legacy_seen: int = 0
    stopped_pids: list[int] = field(default_factory=list)
    active_owner_pids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def stopped(self) -> int:
        return len(self.stopped_pids)


def _safe_create_time(process) -> float | None:
    try:
        return float(process.create_time())
    except (AttributeError, psutil.Error, OSError, PermissionError, ValueError):
        return None


def _identity_for_process(process) -> ProcessIdentity:
    return ProcessIdentity(int(process.pid), _safe_create_time(process))


def _identity_matches(process, identity: ProcessIdentity) -> bool:
    if int(process.pid) != int(identity.pid):
        return False
    actual = _safe_create_time(process)
    if identity.create_time is None or actual is None:
        return True
    return abs(actual - float(identity.create_time)) < 1.0


def _process_is_running(process) -> bool:
    try:
        if not process.is_running():
            return False
        zombie = getattr(psutil, "STATUS_ZOMBIE", None)
        if zombie is None or not hasattr(process, "status"):
            return True
        return process.status() != zombie
    except (psutil.Error, OSError, PermissionError):
        return False


def capture_process_tree(pid: int) -> list[ProcessIdentity]:
    """Snapshot descendants before signalling their parent.

    Windows reparents surviving children as soon as the Python tuner exits. Capturing the
    identities first lets the GUI finish cleaning a leaked ``llama-server`` even after the
    parent PID has disappeared.
    """
    try:
        root = psutil.Process(int(pid))
    except (psutil.Error, OSError, PermissionError, ValueError):
        return []
    try:
        descendants = list(root.children(recursive=True))
    except (AttributeError, psutil.Error, OSError, PermissionError):
        descendants = []
    # Children first is important when a parent owns pipes or a console process group.
    return [_identity_for_process(p) for p in reversed(descendants)] + [_identity_for_process(root)]


def _live_processes(identities: list[ProcessIdentity]):
    live = []
    seen: set[int] = set()
    for identity in identities:
        if identity.pid in seen:
            continue
        seen.add(identity.pid)
        try:
            process = psutil.Process(identity.pid)
            if _identity_matches(process, identity) and _process_is_running(process):
                live.append(process)
        except (psutil.Error, OSError, PermissionError, ValueError):
            continue
    return live


def terminate_captured_processes(
    identities: list[ProcessIdentity], *, terminate_timeout: float = 3.0,
    kill_timeout: float = 5.0,
) -> list[int]:
    """Terminate a previously captured process set and return PIDs still alive.

    The helper validates process creation times before every action. It is used by both the
    CLI runner and the GUI's hard-stop safety net.
    """
    live = _live_processes(identities)
    for process in live:
        try:
            process.terminate()
        except (psutil.Error, OSError, PermissionError):
            pass
    deadline = time.monotonic() + max(0.0, terminate_timeout)
    while time.monotonic() < deadline and _live_processes(identities):
        time.sleep(0.05)

    remaining = _live_processes(identities)
    if os.name == "nt":
        # taskkill /T also catches descendants created after the initial snapshot.
        for process in remaining:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=8, check=False,
                    **no_console_kwargs(),
                )
            except (OSError, subprocess.SubprocessError):
                pass
    else:
        for process in remaining:
            try:
                process.kill()
            except (psutil.Error, OSError, PermissionError):
                pass

    deadline = time.monotonic() + max(0.0, kill_timeout)
    while time.monotonic() < deadline:
        remaining = _live_processes(identities)
        if not remaining:
            return []
        time.sleep(0.05)
    return [int(process.pid) for process in _live_processes(identities)]


def terminate_process_tree(pid: int, *, terminate_timeout: float = 3.0,
                           kill_timeout: float = 5.0) -> list[int]:
    return terminate_captured_processes(
        capture_process_tree(pid), terminate_timeout=terminate_timeout, kill_timeout=kill_timeout,
    )


def _normal_path(value: str | os.PathLike[str] | None) -> str:
    if not value:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.fspath(value)))
    except (OSError, TypeError, ValueError):
        return os.path.normcase(str(value))


def _process_executable(process) -> str:
    try:
        value = process.exe()
        if value:
            return _normal_path(value)
    except (AttributeError, psutil.Error, OSError, PermissionError):
        pass
    try:
        command = process.cmdline()
        return _normal_path(command[0]) if command else ""
    except (AttributeError, psutil.Error, OSError, PermissionError):
        return ""


def _owner_is_alive(pid: int | None, create_time: float | None) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
    except (psutil.Error, OSError, PermissionError, ValueError):
        return False
    return _identity_matches(process, ProcessIdentity(int(pid), create_time)) and _process_is_running(process)


def _looks_like_autotuner_server(command: list[str]) -> bool:
    """Recognize pre-registry v0.7.1 candidate commands conservatively."""
    flags = set(command[1:])
    model_flag = "-m" in flags or "--model" in flags
    required = {"--host", "--port", "-ngl", "-c", "-ctk", "-ctv", "--metrics"}
    return model_flag and required.issubset(flags)


def _has_live_original_parent(process) -> bool:
    try:
        parent_pid = int(process.ppid())
        child_started = _safe_create_time(process)
        if parent_pid <= 0:
            return False
        parent = psutil.Process(parent_pid)
        parent_started = _safe_create_time(parent)
        # A newer process at the same parent PID means that the original parent exited and its
        # PID was reused. Treat the child as orphaned in that case.
        if child_started is not None and parent_started is not None and parent_started > child_started + 1.0:
            return False
        return _process_is_running(parent)
    except (AttributeError, psutil.Error, OSError, PermissionError, ValueError):
        return False


def cleanup_stale_servers(
    registry_dir: Path, server_executable: str | None = None, *,
    include_legacy_orphans: bool = False,
) -> StartupCleanupResult:
    """Stop stale autotuner-owned llama servers before GPU preflight.

    Registered processes are killed only when their recorded owner is gone and PID creation time
    plus executable still match. ``include_legacy_orphans`` is the GUI upgrade path for v0.7.1:
    it additionally finds orphaned commands produced by this tuner's exact flag shape and exact
    selected ``llama-server`` executable. Active/manual servers with a live parent are left alone.
    """
    result = StartupCleanupResult()
    registry_dir = Path(registry_dir)
    known_pids: set[int] = set()
    if registry_dir.exists():
        for lease in sorted(registry_dir.glob("server-*.json")):
            result.registered_seen += 1
            try:
                data = json.loads(lease.read_text(encoding="utf-8"))
                pid = int(data["pid"])
                identity = ProcessIdentity(pid, data.get("process_create_time"))
                known_pids.add(pid)
                if _owner_is_alive(data.get("owner_pid"), data.get("owner_create_time")):
                    result.active_owner_pids.append(pid)
                    continue
                try:
                    process = psutil.Process(pid)
                except (psutil.Error, OSError, PermissionError, ValueError):
                    lease.unlink(missing_ok=True)
                    continue
                recorded_exe = _normal_path(data.get("executable"))
                if (not _identity_matches(process, identity)
                        or (recorded_exe and _process_executable(process) != recorded_exe)):
                    lease.unlink(missing_ok=True)
                    continue
                remaining = terminate_process_tree(pid)
                if remaining:
                    result.errors.append(f"registered PID {pid} survived cleanup: {remaining}")
                else:
                    result.stopped_pids.append(pid)
                    lease.unlink(missing_ok=True)
            except Exception as exc:
                result.errors.append(f"{lease.name}: {exc}")

    wanted_exe = _normal_path(server_executable)
    process_iter = getattr(psutil, "process_iter", None)
    if include_legacy_orphans and wanted_exe and process_iter is not None:
        try:
            processes = list(process_iter())
        except (psutil.Error, OSError, PermissionError):
            processes = []
        for process in processes:
            try:
                pid = int(process.pid)
                if pid in known_pids or pid == os.getpid() or _process_executable(process) != wanted_exe:
                    continue
                command = list(process.cmdline())
                if not _looks_like_autotuner_server(command) or _has_live_original_parent(process):
                    continue
                result.legacy_seen += 1
                remaining = terminate_process_tree(pid)
                if remaining:
                    result.errors.append(f"legacy PID {pid} survived cleanup: {remaining}")
                else:
                    result.stopped_pids.append(pid)
            except (AttributeError, psutil.Error, OSError, PermissionError, ValueError) as exc:
                result.errors.append(f"legacy process inspection failed: {exc}")
    return result


def _process_listens_on_port(pid: int, port: int) -> bool | None:
    """Return whether *pid* owns a LISTEN socket on localhost:*port*.

    ``None`` means the OS/psutil did not allow socket ownership inspection. In that case
    ``wait_ready`` falls back to requiring log progress from the child before accepting /health.
    """
    try:
        proc = psutil.Process(pid)
        for conn in proc.net_connections(kind="tcp"):
            if not conn.laddr:
                continue
            if int(conn.laddr.port) == int(port) and conn.status == psutil.CONN_LISTEN:
                return True
        return False
    except (psutil.Error, OSError, PermissionError):
        return None


def _process_startup_activity(pid: int) -> tuple[int, int, int] | None:
    """Return a coarse monotonic-ish process activity signature during model load.

    Large mmap-backed split GGUFs can spend minutes loading without printing another log
    line.  RSS, accumulated CPU time or I/O still changes, which is progress and must not
    be classified as a startup stall.  Coarse buckets avoid extending the timer because
    of tiny accounting jitter.
    """
    try:
        proc = psutil.Process(pid)
        rss_bucket = int(proc.memory_info().rss // (4 * 1024 * 1024))
        cpu = proc.cpu_times()
        cpu_bucket = int((float(cpu.user) + float(cpu.system)) * 10.0)
        try:
            io = proc.io_counters()
            io_bucket = int((int(io.read_bytes) + int(io.write_bytes)) // (1024 * 1024))
        except (AttributeError, psutil.Error, OSError, PermissionError):
            io_bucket = 0
        return rss_bucket, cpu_bucket, io_bucket
    except (AttributeError, psutil.Error, OSError, PermissionError):
        return None


class ServerRunner:
    def __init__(self, command: list[str], log_file: Path | None = None,
                 lease_dir: Path | None = None) -> None:
        self.command = command
        self.log_file = log_file
        self.lease_dir = Path(lease_dir) if lease_dir is not None else None
        self._lease_path: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.lines: deque[str] = deque(maxlen=500)
        self._reader_thread: threading.Thread | None = None
        self._line_count = 0
        self._draft_stats: list[tuple[int, float, int, int, float]] = []
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, creationflags=flags, start_new_session=(os.name != "nt"),
        )
        self._write_lease()
        def reader() -> None:
            assert self.process and self.process.stdout
            fh = self.log_file.open("a", encoding="utf-8") if self.log_file else None
            try:
                for line in self.process.stdout:
                    clean = line.rstrip()
                    self.lines.append(clean)
                    with self._stats_lock:
                        self._line_count += 1
                        line_no = self._line_count
                        parsed = parse_draft_stats_line(clean)
                        if parsed:
                            self._draft_stats.append((
                                line_no, float(parsed["acceptance"]), int(parsed["accepted"]),
                                int(parsed["generated"]), float(parsed["mean_len"])
                            ))
                    if fh:
                        fh.write(line); fh.flush()
            finally:
                if fh: fh.close()
        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def _write_lease(self) -> None:
        if self.lease_dir is None or self.process is None:
            return
        try:
            self.lease_dir.mkdir(parents=True, exist_ok=True)
            owner = psutil.Process(os.getpid())
            child = psutil.Process(self.process.pid)
            payload = {
                "schema": 1,
                "pid": int(self.process.pid),
                "process_create_time": _safe_create_time(child),
                "owner_pid": os.getpid(),
                "owner_create_time": _safe_create_time(owner),
                "executable": _normal_path(self.command[0] if self.command else None),
                "command": list(self.command),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            target = self.lease_dir / f"server-{self.process.pid}.json"
            temporary = self.lease_dir / f".server-{self.process.pid}-{os.getpid()}.tmp"
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, target)
            self._lease_path = target
        except (OSError, psutil.Error, ValueError, TypeError):
            # The GUI's captured-tree fallback still protects Stop even if a read-only output
            # location prevents persistent recovery metadata.
            self._lease_path = None

    def _remove_lease(self) -> None:
        if self._lease_path is None:
            return
        try:
            self._lease_path.unlink(missing_ok=True)
        except OSError:
            return
        self._lease_path = None

    def wait_ready(self, port: int, hard_timeout: float, stall_timeout: float = 30.0) -> float:
        start = time.monotonic(); last_progress = start; last_mark = self.log_mark()
        last_activity = _process_startup_activity(self.process.pid) if self.process else None
        url = f"http://127.0.0.1:{port}"
        while True:
            if not self.process:
                raise ServerStartError("Server not started")
            rc = self.process.poll()
            if rc is not None:
                raise ServerStartError(f"llama-server exited with code {rc}: {self.tail(40)}")
            status, body = get_health(url)
            if status == 200 and body.get("status") == "ok":
                # Never accept a healthy *old* llama-server that happens to occupy the same port.
                # Prefer OS socket ownership; if unavailable, at least require this child to have
                # emitted startup output before trusting /health.
                owns = _process_listens_on_port(self.process.pid, port)
                if owns is True or (owns is None and self.log_mark() > 0):
                    return time.monotonic() - start
            now = time.monotonic()
            cur_mark = self.log_mark()
            activity = _process_startup_activity(self.process.pid)
            if cur_mark != last_mark or (activity is not None and activity != last_activity):
                last_progress = now
                last_mark = cur_mark
                last_activity = activity
            if now - start > hard_timeout:
                raise ServerStartError(f"Startup hard timeout after {hard_timeout:.0f}s")
            if now - last_progress > stall_timeout:
                raise ServerStartError(
                    f"Startup stalled: no server-log or process activity for {stall_timeout:.0f}s"
                )
            time.sleep(0.4)

    def tail(self, n: int = 30) -> str:
        return "\n".join(list(self.lines)[-n:])

    def log_mark(self) -> int:
        """Return a monotonically increasing server-log line marker."""
        with self._stats_lock:
            return self._line_count

    def draft_stats_since(self, mark: int) -> list[dict[str, float | int]]:
        """Return exact speculative stats printed by llama-server after *mark*.

        Current llama.cpp HTTP timings expose draft_n / draft_n_accepted but not the verification-step
        count required to reconstruct `mean len`. The server log does print the exact value, so the
        stability benchmark associates that line with each request instead of inventing a proxy.
        """
        with self._stats_lock:
            rows = [x for x in self._draft_stats if x[0] > mark]
        return [
            {
                "line": line_no,
                "acceptance": acceptance,
                "accepted": accepted,
                "generated": generated,
                "mean_len": mean_len,
            }
            for line_no, acceptance, accepted, generated, mean_len in rows
        ]

    def stop(self, graceful_timeout: float = 5.0) -> None:
        p = self.process
        if not p:
            return
        if p.poll() is not None:
            self._remove_lease()
            return
        captured = capture_process_tree(p.pid)
        try:
            if os.name == "nt":
                try:
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    p.terminate()
            else:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    p.terminate()
            p.wait(timeout=graceful_timeout)
            self._remove_lease()
            return
        except subprocess.TimeoutExpired:
            pass
        terminate_captured_processes(captured, terminate_timeout=1.0, kill_timeout=5.0)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return
        self._remove_lease()
