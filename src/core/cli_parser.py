"""Parser for editable llama-server CLI commands."""

from __future__ import annotations

import ctypes
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.core.param_registry import (
    FLAG_TO_SPEC,
    NEG_FLAG_TO_SPEC,
    OPTIONAL_VALUE_FLAGS as _OPTIONAL_VALUE_FLAGS,
    VALUE_FLAGS as _VALUE_FLAGS,
)


@dataclass
class ParsedCliCommand:
    executable: str = ""
    model_path: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    extra_args: str = ""
    warnings: list[str] = field(default_factory=list)


def _normalize_command_line_text(command: str) -> str:
    """Normalize copied CLI text before tokenization.

    Users often paste multi-line Windows commands where `^` is only a cmd.exe
    line-continuation marker. CommandLineToArgvW treats it as a literal token,
    so remove it before parsing.
    """
    lines = str(command or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    parts: list[str] = []
    current = ""
    saw_args_prefix = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("args:"):
            line = line.split(":", 1)[1].strip()
            saw_args_prefix = True
        elif saw_args_prefix and not line.startswith(("-", "^", "`", '"', "'")):
            break

        continuation = False
        while line in {"^", "`"}:
            line = ""
            continuation = True
            break
        if line.endswith("^") and not line.endswith("^^"):
            line = line[:-1].rstrip()
            continuation = True
        elif line.endswith("`") and not line.endswith("``"):
            line = line[:-1].rstrip()
            continuation = True

        if line:
            current = f"{current} {line}".strip()
        if not continuation and current:
            parts.append(current)
            current = ""

    if current:
        parts.append(current)
    return " ".join(parts).strip()


def split_command_line(command: str) -> list[str]:
    text = _normalize_command_line_text(command)
    if not text:
        return []
    if os.name == "nt":
        argc = ctypes.c_int()
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        argv = shell32.CommandLineToArgvW(ctypes.c_wchar_p(text), ctypes.byref(argc))
        if argv:
            try:
                return [
                    arg
                    for arg in (argv[i] for i in range(argc.value))
                    if str(arg).strip() not in {"^", "`"}
                ]
            finally:
                ctypes.windll.kernel32.LocalFree(argv)  # type: ignore[attr-defined]
    return [arg for arg in shlex.split(text) if str(arg).strip() not in {"^", "`"}]


def _quote_args(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _base_flag(arg: str) -> str:
    return arg.split("=", 1)[0] if str(arg).startswith("-") else str(arg)


def _inline_value(arg: str) -> tuple[str, str | None]:
    if str(arg).startswith("-") and "=" in arg:
        base, value = arg.split("=", 1)
        return base, value
    return str(arg), None


def _is_value_token(arg: str) -> bool:
    if not str(arg).startswith("-"):
        return True
    try:
        float(str(arg))
        return True
    except ValueError:
        return False


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _set_gpu_layers(settings: dict[str, Any], value: str) -> None:
    text = str(value).strip().lower()
    settings["gpu_auto"] = text == "auto"
    settings["gpu_layers_all"] = text == "all"
    if not settings["gpu_auto"] and not settings["gpu_layers_all"]:
        settings["gpu_layers"] = int(value)


# Наборы value/optional-value флагов генерируются из реестра параметров;
# паритет со старыми ручными таблицами фиксирует tests/test_param_registry.py.

# Спец-обработчики для cli_kind == "special": флаги с составной логикой
# (несколько полей или условное применение). (result, settings, extra, value).
def _sp_model_path(result, settings, extra, value):
    result.model_path = str(value)


def _sp_gpu_layers(result, settings, extra, value):
    _set_gpu_layers(settings, str(value))


def _sp_fit(result, settings, extra, value):
    settings["fit_off"] = str(value).strip().lower() == "off"


def _sp_mmproj(result, settings, extra, value):
    settings["use_mmproj"] = True
    settings["mmproj_path"] = str(value)


def _sp_model_draft(result, settings, extra, value):
    settings["speculative_mtp"] = True
    settings["spec_draft_model_path"] = str(value)


def _sp_chat_template(result, settings, extra, value):
    settings["use_chat_template"] = True
    settings["chat_template_file"] = str(value)


def _sp_spec_type(result, settings, extra, value):
    if str(value).strip().lower() == "draft-mtp":
        settings["speculative_mtp"] = True
    else:
        extra.extend(["--spec-type", str(value)])


def _sp_reasoning_preserve(result, settings, extra, value):
    settings["reasoning_preserve"] = "preserve"


def _sp_no_reasoning_preserve(result, settings, extra, value):
    settings["reasoning_preserve"] = "no-preserve"


_SPECIAL_HANDLERS: dict[str, Callable[[Any, dict[str, Any], list[str], "str | None"], None]] = {
    "-m": _sp_model_path,
    "--model": _sp_model_path,
    "-ngl": _sp_gpu_layers,
    "--n-gpu-layers": _sp_gpu_layers,
    "--fit": _sp_fit,
    "-mm": _sp_mmproj,
    "--mmproj": _sp_mmproj,
    "-md": _sp_model_draft,
    "--model-draft": _sp_model_draft,
    "--chat-template-file": _sp_chat_template,
    "--spec-type": _sp_spec_type,
    "--reasoning-preserve": _sp_reasoning_preserve,
    "--no-reasoning-preserve": _sp_no_reasoning_preserve,
}

# Bool-специальные флаги: не потребляют значение (в отличие от остальных
# special-флагов, которые все value-consuming). Они попадают в VALUE_FLAGS
# как cli_kind="special", но парсер не должен требовать для них значение.
_SPECIAL_NO_VALUE_FLAGS = {"--reasoning-preserve", "--no-reasoning-preserve"}


def _looks_like_executable(token: str) -> bool:
    name = Path(token).name.lower()
    return name.endswith(".exe") or name in {"llama-server", "llama-server.exe"}


def _cuda_version_from_path(path: str) -> str:
    for part in Path(path).parts:
        lower = part.lower()
        if "cuda-13" in lower:
            return "13"
        if "cuda-12" in lower:
            return "12"
    return ""


def parse_llama_server_command(command: str) -> ParsedCliCommand:
    tokens = split_command_line(command)
    result = ParsedCliCommand()
    if not tokens:
        result.warnings.append("CLI command is empty")
        return result

    extra: list[str] = []
    i = 0
    if tokens and not tokens[0].startswith("-") and _looks_like_executable(tokens[0]):
        result.executable = tokens[0]
        cuda_version = _cuda_version_from_path(tokens[0])
        if cuda_version:
            result.settings["cuda_version"] = cuda_version
        i = 1

    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("-"):
            extra.append(token)
            i += 1
            continue

        flag, inline_value = _inline_value(token)
        value = inline_value
        consumed_value = False

        if flag in _VALUE_FLAGS and flag not in _SPECIAL_NO_VALUE_FLAGS:
            if value is None and i + 1 < len(tokens) and _is_value_token(tokens[i + 1]):
                value = tokens[i + 1]
                consumed_value = True
            if value is None:
                result.warnings.append(f"{flag} requires a value")
                extra.append(token)
                i += 1
                continue
        elif flag in _OPTIONAL_VALUE_FLAGS:
            if value is None and i + 1 < len(tokens) and _is_value_token(tokens[i + 1]):
                value = tokens[i + 1]
                consumed_value = True
        else:
            value = inline_value

        try:
            spec = FLAG_TO_SPEC.get(flag)
            neg_spec = NEG_FLAG_TO_SPEC.get(flag)
            if neg_spec is not None and spec is None:
                # --no-*: выключить булево поле.
                result.settings[neg_spec.name] = False
            elif spec is None:
                # Неизвестный реестру флаг остаётся в extra_args.
                if inline_value is not None:
                    extra.append(token)
                elif i + 1 < len(tokens) and _is_value_token(tokens[i + 1]):
                    extra.extend([token, tokens[i + 1]])
                    consumed_value = True
                else:
                    extra.append(token)
            elif spec.cli_kind == "value":
                converter = spec.field_type or str
                result.settings[spec.name] = converter(value)
            elif spec.cli_kind == "bool":
                result.settings[spec.name] = True
            elif spec.cli_kind == "false":
                result.settings[spec.name] = False
            elif spec.cli_kind == "bool_value":
                result.settings[spec.name] = _as_bool(value, default=True)
            elif spec.cli_kind == "ignore":
                pass
            else:  # special: составная логика по конкретному флагу
                _SPECIAL_HANDLERS[flag](result, result.settings, extra, value)
        except (TypeError, ValueError):
            result.warnings.append(f"{flag} has invalid value: {value}")
            if inline_value is not None:
                extra.append(token)
            elif value is not None:
                extra.extend([token, str(value)])
            else:
                extra.append(token)

        i += 2 if consumed_value else 1

    result.extra_args = _quote_args(extra)
    return result
