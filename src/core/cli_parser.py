"""Parser for editable llama-server CLI commands."""

from __future__ import annotations

import ctypes
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedCliCommand:
    executable: str = ""
    model_path: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    extra_args: str = ""
    warnings: list[str] = field(default_factory=list)


def split_command_line(command: str) -> list[str]:
    text = str(command or "").strip()
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
                return [argv[i] for i in range(argc.value)]
            finally:
                ctypes.windll.kernel32.LocalFree(argv)  # type: ignore[attr-defined]
    return shlex.split(text)


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


_VALUE_FLAGS = {
    "-m",
    "--model",
    "--host",
    "--port",
    "--device",
    "--spec-draft-device",
    "--split-mode",
    "--main-gpu",
    "-ngl",
    "--n-gpu-layers",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-c",
    "--ctx-size",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-np",
    "--parallel",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-ncmoe",
    "--n-cpu-moe",
    "--fit",
    "-rea",
    "--reasoning",
    "--ctx-checkpoints",
    "--cache-ram",
    "--temp",
    "--repeat-penalty",
    "-mm",
    "--mmproj",
    "-md",
    "--model-draft",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-ngl",
    "-ngld",
    "--chat-template-file",
}

_OPTIONAL_VALUE_FLAGS = {
    "-fa",
    "--flash-attn",
}


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

        if flag in _VALUE_FLAGS:
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
            if flag in {"-m", "--model"}:
                result.model_path = str(value)
            elif flag == "--host":
                result.settings["host"] = str(value)
            elif flag == "--port":
                result.settings["port"] = int(value)
            elif flag == "--device":
                result.settings["cuda_device"] = str(value)
            elif flag == "--spec-draft-device":
                result.settings["spec_draft_device"] = str(value)
            elif flag == "--split-mode":
                result.settings["split_mode"] = str(value)
            elif flag == "--main-gpu":
                result.settings["main_gpu"] = int(value)
            elif flag in {"-ngl", "--n-gpu-layers"}:
                _set_gpu_layers(result.settings, str(value))
            elif flag in {"-t", "--threads"}:
                result.settings["threads"] = int(value)
            elif flag in {"-tb", "--threads-batch"}:
                result.settings["threads_batch"] = int(value)
            elif flag in {"-c", "--ctx-size"}:
                result.settings["ctx_size"] = int(value)
            elif flag in {"-b", "--batch-size"}:
                result.settings["batch_size"] = int(value)
            elif flag in {"-ub", "--ubatch-size"}:
                result.settings["ubatch_size"] = int(value)
            elif flag in {"-np", "--parallel"}:
                result.settings["parallel_slots"] = int(value)
            elif flag in {"-ctk", "--cache-type-k"}:
                result.settings["cache_type_k"] = str(value)
            elif flag in {"-ctv", "--cache-type-v"}:
                result.settings["cache_type_v"] = str(value)
            elif flag in {"-ncmoe", "--n-cpu-moe"}:
                result.settings["cpu_moe_layers"] = int(value)
            elif flag == "--fit":
                result.settings["fit_off"] = str(value).strip().lower() == "off"
            elif flag in {"-rea", "--reasoning"}:
                result.settings["reasoning_mode"] = str(value)
            elif flag == "--ctx-checkpoints":
                result.settings["ctx_checkpoints"] = int(value)
            elif flag == "--cache-ram":
                result.settings["cache_ram"] = int(value)
            elif flag == "--temp":
                result.settings["temperature"] = float(value)
            elif flag == "--repeat-penalty":
                result.settings["repeat_penalty"] = float(value)
            elif flag in {"-fa", "--flash-attn"}:
                result.settings["flash_attn"] = _as_bool(value, default=True)
            elif flag in {"-mm", "--mmproj"}:
                result.settings["use_mmproj"] = True
                result.settings["mmproj_path"] = str(value)
            elif flag in {"-md", "--model-draft"}:
                result.settings["speculative_mtp"] = True
                result.settings["spec_draft_model_path"] = str(value)
            elif flag == "--spec-type":
                if str(value).strip().lower() == "draft-mtp":
                    result.settings["speculative_mtp"] = True
                else:
                    extra.extend([flag, str(value)])
            elif flag == "--spec-draft-n-max":
                result.settings["spec_draft_n_max"] = int(value)
            elif flag in {"--spec-draft-ngl", "-ngld"}:
                result.settings["spec_draft_gpu_layers"] = str(value)
            elif flag == "--chat-template-file":
                result.settings["use_chat_template"] = True
                result.settings["chat_template_file"] = str(value)
            elif flag in {"--kv-unified", "-kvu"}:
                result.settings["kv_unified"] = True
            elif flag == "--no-mmproj":
                result.settings["use_mmproj"] = False
            elif flag == "--no-mmproj-offload":
                result.settings["mmproj_offload"] = False
            elif flag == "--mmap":
                result.settings["use_mmap"] = True
            elif flag == "--no-mmap":
                result.settings["use_mmap"] = False
            elif flag == "--mlock":
                result.settings["use_mlock"] = True
            elif flag == "--verbose":
                result.settings["verbose"] = True
            elif flag == "--log-timestamps":
                result.settings["log_timestamps"] = True
            elif flag == "--no-cont-batching":
                result.settings["cont_batching"] = False
            elif flag == "--no-cache-prompt":
                result.settings["cache_prompt"] = False
            elif flag == "--context-shift":
                result.settings["context_shift"] = True
            elif flag == "--no-webui":
                result.settings["no_webui"] = True
            elif flag == "--jinja":
                result.settings["jinja"] = True
            elif flag == "--metrics":
                pass
            else:
                if inline_value is not None:
                    extra.append(token)
                elif i + 1 < len(tokens) and _is_value_token(tokens[i + 1]):
                    extra.extend([token, tokens[i + 1]])
                    consumed_value = True
                else:
                    extra.append(token)
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
