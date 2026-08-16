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
        "Недостаточно памяти (VRAM или RAM)",
        "Уменьшите Context Size, Batch/UBatch или GPU layers; закройте другие GPU-приложения.",
    ),
    (
        re.compile(r"address already in use|failed to bind|bind.*failed|port.*in use", re.I),
        "Порт сервера уже занят",
        "Выберите другой порт или завершите другой процесс llama-server.",
    ),
    (
        re.compile(r"unknown argument|invalid argument|unrecognized option|invalid value", re.I),
        "Текущая сборка llama.cpp не поддерживает один из параметров",
        "Проверьте строку Args в отчёте, уберите несовместимый параметр или обновите llama.cpp.",
    ),
    (
        re.compile(r"failed to load model|invalid gguf|gguf.*(error|invalid)|model.*corrupt", re.I),
        "Не удалось прочитать GGUF-модель",
        "Проверьте, что файл скачан полностью, доступен и поддерживается этой версией llama.cpp.",
    ),
    (
        re.compile(r"failed to load draft|model-draft|mtp input row width|mtp.*assert", re.I),
        "Ошибка MTP/draft-модели",
        "Уберите несовместимый draft или отключите MTP speculative для этой модели.",
    ),
    (
        re.compile(
            r"(?:chat template|jinja|template).*(?:error|failed|invalid)"
            r"|(?:error|failed|invalid).*(?:chat template|jinja)",
            re.I,
        ),
        "Ошибка chat template / Jinja",
        "Проверьте файл шаблона и поддержку Jinja выбранной сборкой llama.cpp.",
    ),
    (
        re.compile(r"cudart|cublas|cuda.*dll|no cuda-capable|cuda driver|driver version", re.I),
        "Ошибка CUDA или драйвера NVIDIA",
        "Проверьте выбранную CUDA-сборку, драйвер NVIDIA и наличие DLL рядом с llama-server.exe.",
    ),
    (
        re.compile(r"access is denied|permission denied|отказано в доступе", re.I),
        "Нет доступа к файлу или папке",
        "Проверьте права, антивирус и не удерживается ли файл другим процессом.",
    ),
    (
        re.compile(r"context.*(too large|exceed|failed)|kv cache.*(failed|too large)", re.I),
        "Слишком большой контекст или KV cache",
        "Уменьшите Context Size/Parallel slots либо используйте более компактные типы KV cache.",
    ),
    (
        re.compile(r"assert|access violation|segmentation fault|fatal error|stack overflow", re.I),
        "Аварийное завершение внутри llama.cpp",
        "Сохраните отчёт, попробуйте другую сборку llama.cpp и проверьте совместимость модели и параметров.",
    ),
]

_WINDOWS_EXIT_CODES = {
    -1073741819: "нарушение доступа к памяти (0xC0000005)",
    3221225477: "нарушение доступа к памяти (0xC0000005)",
    -1073740791: "повреждение стека (0xC0000409)",
    3221226505: "повреждение стека (0xC0000409)",
    -1073741571: "переполнение стека (0xC00000FD)",
    3221225725: "переполнение стека (0xC00000FD)",
    -1073740940: "повреждение heap (0xC0000374)",
    3221226356: "повреждение heap (0xC0000374)",
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
        cause = f"Процесс llama-server аварийно завершился: {exit_detail}"
        action = "Попробуйте другую сборку llama.cpp; приложите сохранённый отчёт при сообщении об ошибке."
    if not cause and process_error and not crash_exit:
        cause = f"QProcess не смог запустить или обслужить llama-server: {process_error}"
        action = "Проверьте путь к EXE, права доступа и DLL выбранной CUDA-сборки."
    if not cause and crash_exit:
        cause = "llama-server завершился аварийно без явного сообщения"
        action = "Откройте полный отчёт: в нём сохранены команда, код завершения и последние строки процесса."
    if not cause and unexpected:
        cause = f"llama-server неожиданно завершился с кодом {exit_code}"
        action = "Проверьте последние строки отчёта; попробуйте уменьшить нагрузку или другую сборку llama.cpp."
        detail = next(
            (line.strip() for line in reversed(text.splitlines()) if line.strip()), ""
        )[:700]
    if not cause:
        return None

    if stop_requested and (crash_exit or exit_detail):
        cause = f"Сбой при выгрузке модели: {cause}"
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
        "❌ ДИАГНОСТИКА",
        f"Причина: {result.get('cause') or 'не определена'}",
        *(
            [f"Сообщение llama.cpp: {result.get('detail')}"]
            if result.get("detail")
            else []
        ),
        f"Что сделать: {result.get('action') or 'откройте полный отчёт'}",
        f"Код завершения: {result.get('exit_code')}",
    ]
    if report_path:
        lines.append(f"Полный отчёт: {report_path}")
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
