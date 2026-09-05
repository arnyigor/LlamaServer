"""Human-readable failure analysis and persistent crash reports."""

from __future__ import annotations

import os
import platform
import re
import sys
import faulthandler
from datetime import datetime
from pathlib import Path
from typing import Iterable


_CAUSES = [
    (
        re.compile(r"out of memory|cuda.*alloc|failed to allocate|not enough memory", re.I),
        "Out of memory (VRAM or RAM)",
        "Reduce Context Size, Batch/UBatch or GPU layers; close other GPU applications.",
    ),
    (
        re.compile(r"address already in use|failed to bind|bind.*failed|port.*in use", re.I),
        "Server port is already in use",
        "Choose another port or stop the other llama-server process.",
    ),
    (
        re.compile(r"unknown argument|invalid argument|unrecognized option|invalid value", re.I),
        "The current llama.cpp build does not support one of the parameters",
        "Check the Args line in the report, remove the unsupported parameter or update llama.cpp.",
    ),
    (
        re.compile(r"failed to load model|invalid gguf|gguf.*(error|invalid)|model.*corrupt", re.I),
        "Failed to read the GGUF model",
        "Make sure the file is fully downloaded, readable and supported by this llama.cpp build.",
    ),
    (
        re.compile(r"failed to load draft|model-draft|mtp input row width|mtp.*assert", re.I),
        "MTP/draft model error",
        "Remove the incompatible draft or disable MTP speculative decoding for this model.",
    ),
    (
        re.compile(
            r"(?:chat template|jinja|template).*(?:error|failed|invalid)"
            r"|(?:error|failed|invalid).*(?:chat template|jinja)",
            re.I,
        ),
        "Chat template / Jinja error",
        "Check the template file and Jinja support in the selected llama.cpp build.",
    ),
    (
        re.compile(r"cudart|cublas|cuda.*dll|no cuda-capable|cuda driver|driver version", re.I),
        "CUDA or NVIDIA driver error",
        "Check the selected CUDA build, the NVIDIA driver and the DLLs next to llama-server.exe.",
    ),
    (
        re.compile(r"access is denied|permission denied|отказано в доступе", re.I),
        "No access to the file or folder",
        "Check permissions, antivirus, and whether another process holds the file.",
    ),
    (
        re.compile(r"context.*(too large|exceed|failed)|kv cache.*(failed|too large)", re.I),
        "Context or KV cache is too large",
        "Reduce Context Size/Parallel slots or use more compact KV cache types.",
    ),
    (
        re.compile(r"assert|access violation|segmentation fault|fatal error|stack overflow", re.I),
        "Crash inside llama.cpp",
        "Save the report, try another llama.cpp build and check model/parameter compatibility.",
    ),
]

_WINDOWS_EXIT_CODES = {
    -1073741819: "access violation (0xC0000005)",
    3221225477: "access violation (0xC0000005)",
    -1073740791: "stack corruption (0xC0000409)",
    3221226505: "stack corruption (0xC0000409)",
    -1073741571: "stack overflow (0xC00000FD)",
    3221225725: "stack overflow (0xC00000FD)",
    -1073740940: "heap corruption (0xC0000374)",
    3221226356: "heap corruption (0xC0000374)",
    # Very common on CPUs without AVX2/AVX512: the selected llama.cpp build
    # was compiled assuming an instruction set the CPU doesn't have.
    -1073741795: "illegal instruction (0xC000001D) — CPU likely lacks the AVX2/AVX512 instructions this build requires",
    3221225501: "illegal instruction (0xC000001D) — CPU likely lacks the AVX2/AVX512 instructions this build requires",
}


def diagnostics_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    path = base / "LlamaServerGUI" / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_server_failure(
    exit_code: int,
    output: str,
    *,
    crash_exit: bool = False,
    stop_requested: bool = False,
    process_error: str = "",
) -> dict | None:
    text = "\n".join(part for part in (output, process_error) if part).strip()
    cause = action = detail = ""
    for pattern, candidate_cause, candidate_action in _CAUSES:
        match = pattern.search(text)
        if match:
            cause, action = candidate_cause, candidate_action
            detail = next(
                (
                    line.strip()
                    for line in text.splitlines()
                    if pattern.search(line)
                ),
                "",
            )[:700]
            break

    exit_detail = _WINDOWS_EXIT_CODES.get(int(exit_code))
    unexpected = not stop_requested
    if not cause and exit_detail:
        cause = f"llama-server process crashed: {exit_detail}"
        action = "Try another llama.cpp build; attach the saved report when reporting the bug."
    if not cause and process_error and not crash_exit:
        cause = f"QProcess failed to start or serve llama-server: {process_error}"
        action = "Check the EXE path, permissions and DLLs of the selected CUDA build."
    if not cause and crash_exit:
        cause = "llama-server crashed without an explicit message"
        action = "Open the full report: it contains the command, exit code and last process lines."
    if not cause and unexpected:
        cause = f"llama-server exited unexpectedly with code {exit_code}"
        action = "Check the last report lines; try reducing the load or another llama.cpp build."
        detail = next(
            (line.strip() for line in reversed(text.splitlines()) if line.strip()), ""
        )[:700]
    if not cause:
        return None

    if stop_requested and (crash_exit or exit_detail):
        cause = f"Failed to unload the model: {cause}"
    return {
        "cause": cause,
        "action": action,
        "detail": detail,
        "exit_code": int(exit_code),
        "crash_exit": bool(crash_exit),
        "stop_requested": bool(stop_requested),
    }


def format_diagnostic_summary(result: dict, report_path: str = "") -> str:
    lines = [
        "❌ DIAGNOSTICS",
        f"Cause: {result.get('cause') or 'unidentified'}",
        *(
            [f"llama.cpp message: {result.get('detail')}"]
            if result.get("detail")
            else []
        ),
        f"What to do: {result.get('action') or 'open the full report'}",
        f"Exit code: {result.get('exit_code')}",
    ]
    if report_path:
        lines.append(f"Full report: {report_path}")
    return "\n".join(lines)


def write_server_report(
    result: dict,
    *,
    executable: str = "",
    args: Iterable[str] = (),
    env: dict | None = None,
    output: str = "",
    process_error: str = "",
    runtime_seconds: float = 0.0,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = diagnostics_dir() / f"llama-server-{stamp}.log"
    command = " ".join([str(executable), *(str(arg) for arg in args)]).strip()
    lines = [
        "LlamaServer GUI diagnostic report",
        f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Cause: {result.get('cause')}",
        f"Suggested action: {result.get('action')}",
        f"Exit code: {result.get('exit_code')}",
        f"Crash exit: {result.get('crash_exit')}",
        f"Stop requested: {result.get('stop_requested')}",
        f"Runtime: {runtime_seconds:.2f} s",
        f"Process error: {process_error or '-'}",
        f"OS: {platform.platform()}",
        f"Python: {sys.version}",
        f"Frozen EXE: {bool(getattr(sys, 'frozen', False))}",
        f"Command: {command}",
        "Environment: " + " ".join(f"{k}={v}" for k, v in (env or {}).items()),
        "",
        "--- Last llama-server output ---",
        output.strip() or "(no process output was captured)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", errors="replace")
    return path


def write_app_exception_report(exception_text: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = diagnostics_dir() / f"app-exception-{stamp}.log"
    path.write_text(
        "LlamaServer GUI unhandled exception\n"
        f"Time: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        f"OS: {platform.platform()}\nPython: {sys.version}\n\n{exception_text}",
        encoding="utf-8",
        errors="replace",
    )
    return path


def consume_previous_native_crash() -> Path | None:
    """Mark the newest unacknowledged native-crash trace as seen."""
    candidates = sorted(
        (
            path
            for path in diagnostics_dir().glob("app-native-*.log")
            if not path.stem.endswith("-seen")
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            if path.stat().st_size <= 0:
                path.unlink(missing_ok=True)
                continue
            seen = path.with_name(path.stem + "-seen.log")
            path.replace(seen)
            return seen
        except OSError:
            continue
    return None


def start_native_crash_capture():
    """Keep a faulthandler file open; a clean session removes its empty file."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = diagnostics_dir() / f"app-native-{stamp}.log"
    stream = path.open("w", encoding="utf-8")
    try:
        faulthandler.enable(file=stream, all_threads=True)
    except (RuntimeError, OSError):
        stream.close()
        path.unlink(missing_ok=True)
        return None, None
    return path, stream


def finish_native_crash_capture(path, stream) -> None:
    if stream is None:
        return
    try:
        faulthandler.disable()
    except RuntimeError:
        pass
    try:
        stream.flush()
        stream.close()
        if path and Path(path).exists() and Path(path).stat().st_size <= 0:
            Path(path).unlink(missing_ok=True)
    except OSError:
        pass
