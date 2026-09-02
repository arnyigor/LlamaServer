"""Сборщик CLI-аргументов для llama-server и llama-bench."""

import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.constants import (
    SAMPLING_AUTO_FLOAT,
    SAMPLING_AUTO_INT,
    SAMPLING_LAST_N_AUTO,
    SAMPLING_PENALTY_AUTO,
    SAMPLING_SEED_AUTO,
)
from src.core.param_registry import (
    FILTER_BOOL_FLAGS,
    FILTER_VALUE_FLAGS,
    MANAGED_FIELD_NAMES,
)
from src.utils.file_utils import validate_path


def validate_extra_args(args: List[str], model_dir: str) -> List[str]:
    """Валидация дополнительных аргументов. Возвращает список ошибок."""
    invalid = []
    i = 0
    base_dir = Path(model_dir or ".").resolve()
    path_flags = {
        "--grammar-file",
        "--api-key-file",
        "--lora",
        "--lora-scaled",
        "--mmproj",
        "--chat-template-file",
        "-md",
        "--model-draft",
    }

    while i < len(args):
        arg = args[i]
        if not arg.startswith("-"):
            i += 1
            continue

        if "=" in arg:
            base_arg, inline_value = arg.split("=", 1)
            has_inline = True
        else:
            base_arg, inline_value = arg, None
            has_inline = False

        def get_value() -> tuple:
            if has_inline:
                return inline_value, False
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                return args[i + 1], True
            return None, False

        if base_arg in path_flags:
            value, consumed = get_value()
            if value is None:
                invalid.append(f"{base_arg} (требуется значение)")
            else:
                if consumed:
                    i += 1
                try:
                    validate_path(value, base_dir=base_dir)
                except ValueError as e:
                    invalid.append(f"{base_arg} {value} (недопустимый путь: {e})")

        elif base_arg == "--host":
            value, consumed = get_value()
            if value is None:
                invalid.append(f"{base_arg} (требуется значение)")
            else:
                if consumed:
                    i += 1
                if value in ("0.0.0.0", "::", ""):
                    invalid.append(
                        f"{base_arg} {value} (запрещено: открывает сервер всем интерфейсам)"
                    )
        i += 1
    return invalid


def _flag_base(arg: str) -> str:
    return arg.split("=", 1)[0] if arg.startswith("-") else arg


def _is_value_token(arg: str) -> bool:
    if not str(arg).startswith("-"):
        return True
    try:
        float(str(arg))
        return True
    except ValueError:
        return False


def _filter_duplicate_extra_args(
    extra: List[str], existing_args: List[str]
) -> List[str]:
    """Удаляет из extra_args флаги, которыми уже управляют UI/AutoTune.

    Это предотвращает CLI вроде `--ctx-checkpoints 0 ... --ctx-checkpoints 0`.
    Если флаг не был добавлен UI, он остаётся валидным extra arg.
    """
    # Классификация флагов (потребляет значение / автономный переключатель)
    # генерируется из реестра параметров; паритет со старыми ручными наборами
    # фиксирует tests/test_param_registry.py.
    flags_with_values = FILTER_VALUE_FLAGS
    bool_flags = FILTER_BOOL_FLAGS
    managed = {_flag_base(a) for a in existing_args if str(a).startswith("-")}
    filtered: List[str] = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        base = _flag_base(arg)
        if base in managed and base in flags_with_values:
            if "=" not in arg and i + 1 < len(extra) and _is_value_token(extra[i + 1]):
                i += 2
            else:
                i += 1
            continue
        if base in managed and base in bool_flags:
            i += 1
            continue
        filtered.append(arg)
        i += 1
    return filtered


def _split_extra_groups(tokens: List[str]) -> List[List[str]]:
    """Разбивает токены на группы «флаг + опциональное значение».

    Позиционные (не-флаговые) токены остаются одиночными группами.
    Значение потребляется, если оно inline через "=" либо следующий токен —
    не флаг (отрицательные числа считаются значениями, см. _is_value_token).
    """
    groups: List[List[str]] = []
    i = 0
    while i < len(tokens):
        arg = tokens[i]
        if not str(arg).startswith("-"):
            groups.append([arg])
            i += 1
            continue
        group = [arg]
        if "=" not in arg and i + 1 < len(tokens) and _is_value_token(tokens[i + 1]):
            group.append(tokens[i + 1])
            i += 2
        else:
            i += 1
        groups.append(group)
    return groups


def merge_extra_args(existing: str, incoming: str) -> str:
    """Мержит incoming extra-флаги поверх existing (импорт побеждает).

    - Флаги existing, которых нет в incoming, сохраняются.
    - Одинаковые флаги (по базовому имени до "=") заменяются значениями из incoming.
    - Новые флаги добавляются в конец в порядке incoming.

    Используется при импорте/применении CLI-команды, чтобы пользовательские
    extra-параметры не стирались при отсутствии unknown-флагов в команде.
    """

    def tokenize(text: str) -> List[str]:
        text = str(text or "").strip()
        if not text:
            return []
        try:
            return shlex.split(text)
        except ValueError:
            return text.split()

    existing_tokens = tokenize(existing)
    incoming_tokens = tokenize(incoming)
    quote_all = lambda toks: " ".join(shlex.quote(t) for t in toks)  # noqa: E731

    if not incoming_tokens:
        return quote_all(existing_tokens)
    if not existing_tokens:
        return quote_all(incoming_tokens)

    existing_groups = _split_extra_groups(existing_tokens)
    incoming_groups = _split_extra_groups(incoming_tokens)

    incoming_bases = {
        _flag_base(g[0]) for g in incoming_groups if str(g[0]).startswith("-")
    }

    merged: List[List[str]] = [
        g
        for g in existing_groups
        if not (str(g[0]).startswith("-") and _flag_base(g[0]) in incoming_bases)
    ]
    merged.extend(incoming_groups)

    tokens: List[str] = []
    for group in merged:
        tokens.extend(group)
    return quote_all(tokens)


def build_args(
    cfg: Any, model_path: str, for_benchmark: bool = False
) -> Optional[List[str]]:
    """Сборка аргументов. cfg - объект AppSettings или аналогичный интерфейс."""
    if not model_path:
        return None

    args = ["-m", model_path]
    gpu_layers_all = bool(getattr(cfg, "gpu_layers_all", False))
    if for_benchmark:
        gpu_val = "99" if cfg.gpu_auto or gpu_layers_all else str(cfg.gpu_layers)
    else:
        if gpu_layers_all:
            gpu_val = "all"
        else:
            gpu_val = "auto" if cfg.gpu_auto else str(cfg.gpu_layers)

    if for_benchmark:
        args += [
            "-p",
            str(cfg.bench_prompt),
            "-n",
            str(cfg.bench_gen),
            "-r",
            "1",
            "-ngl",
            gpu_val,
        ]
        args += ["-fa", "on" if cfg.flash_attn else "off"]
        args += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v]
        bs = int(cfg.batch_size if cfg.batch_size and cfg.batch_size > 0 else 512)
        ub = int(
            cfg.ubatch_size if cfg.ubatch_size and cfg.ubatch_size > 0 else min(bs, 512)
        )
        args += ["-b", str(bs), "-ub", str(min(ub, bs))]
        if cfg.threads > 0:
            args += ["-t", str(cfg.threads)]
        # Current llama-bench builds do not expose -tb/--threads-batch.
        if "verbose" in MANAGED_FIELD_NAMES and getattr(cfg, "verbose", False):
            args.append("-v")
        if cfg.cpu_moe_layers >= 0:
            args += ["-ncmoe", str(cfg.cpu_moe_layers)]
    else:
        args += ["--host", str(getattr(cfg, "host", "127.0.0.1") or "127.0.0.1")]
        args += ["--port", str(cfg.port)]
        if "cuda_device" in MANAGED_FIELD_NAMES:
            device = str(getattr(cfg, "cuda_device", "") or "").strip()
            if device:
                args += ["--device", device]
        if "split_mode" in MANAGED_FIELD_NAMES:
            split_mode = str(getattr(cfg, "split_mode", "") or "").strip()
            if split_mode:
                args += ["--split-mode", split_mode]
        if "main_gpu" in MANAGED_FIELD_NAMES:
            main_gpu_raw = getattr(cfg, "main_gpu", -1)
            main_gpu = (
                -1 if main_gpu_raw is None or main_gpu_raw == "" else int(main_gpu_raw)
            )
            if main_gpu >= 0:
                args += ["--main-gpu", str(main_gpu)]
        args += ["-ngl", gpu_val, "-t", str(cfg.threads)]
        if cfg.ctx_size >= 0:
            args += ["-c", str(cfg.ctx_size)]
        if cfg.threads_batch > 0:
            args += ["-tb", str(cfg.threads_batch)]
        if cfg.batch_size >= 0:
            bs = cfg.batch_size
            ub = cfg.ubatch_size if cfg.ubatch_size >= 0 else bs
            args += ["-b", str(bs), "-ub", str(min(ub, bs))]
        if cfg.parallel_slots >= 0:
            args += ["-np", str(cfg.parallel_slots)]
        if "kv_unified" in MANAGED_FIELD_NAMES and getattr(cfg, "kv_unified", False):
            args.append("--kv-unified")
        args += [
            "-ctk",
            cfg.cache_type_k,
            "-ctv",
            cfg.cache_type_v,
        ]
        if cfg.cpu_moe_layers >= 0:
            args += ["-ncmoe", str(cfg.cpu_moe_layers)]
        if cfg.fit_off:
            args += ["--fit", "off"]
        thinking = getattr(cfg, "enable_thinking", "off")
        if thinking is True:
            thinking = "true"
        elif thinking is False or thinking is None:
            thinking = "off"
        else:
            thinking = str(thinking).strip().lower()
        reasoning_mode = (
            str(getattr(cfg, "reasoning_mode", "auto") or "auto").strip().lower()
        )
        if thinking == "false":
            reasoning_mode = "off"
        elif thinking == "true" and reasoning_mode == "auto":
            reasoning_mode = "on"
        if reasoning_mode != "auto":
            # Latest llama.cpp uses --reasoning on/off for thinking templates;
            # --chat-template-kwargs enable_thinking is now deprecated.
            args += ["--reasoning", reasoning_mode]

        # --- Reasoning controls (effort / preserve / budget) ---
        # Эмитятся только когда reasoning активен (on/auto). При явном
        # --reasoning off (в т.ч. когда enable_thinking=false принудительно
        # ставит off) эти суб-параметры бессмысленны и не добавляются, чтобы
        # не генерировать конфликтующие/лишние флаги. Сами значения остаются
        # в настройках и вернутся в CLI при возврате reasoning в on/auto.
        if reasoning_mode != "off":
            effort = str(getattr(cfg, "reasoning_effort", "") or "").strip()
            if effort:
                args += ["--reasoning-effort", effort]
            preserve = (
                str(getattr(cfg, "reasoning_preserve", "off") or "off").strip().lower()
            )
            if preserve == "preserve":
                args.append("--reasoning-preserve")
            elif preserve == "no-preserve":
                args.append("--no-reasoning-preserve")
            budget = int(getattr(cfg, "reasoning_budget", 0) or 0)
            if budget > 0:
                args += ["--reasoning-budget", str(budget)]
            budget_msg = str(getattr(cfg, "reasoning_budget_message", "") or "").strip()
            if budget_msg:
                args += ["--reasoning-budget-message", budget_msg]
        if "ctx_checkpoints" in MANAGED_FIELD_NAMES and cfg.ctx_checkpoints >= 0:
            args += ["--ctx-checkpoints", str(cfg.ctx_checkpoints)]
        if "cache_ram" in MANAGED_FIELD_NAMES and cfg.cache_ram >= -1:
            args += ["--cache-ram", str(cfg.cache_ram)]
        # Третий элемент — sentinel: значение не выше него означает "auto",
        # флаг не передаётся (см. constants.py).
        sampling_values = (
            ("--temp", "temperature", SAMPLING_AUTO_FLOAT),
            ("--top-k", "top_k", SAMPLING_AUTO_INT),
            ("--top-p", "top_p", SAMPLING_AUTO_FLOAT),
            ("--min-p", "min_p", SAMPLING_AUTO_FLOAT),
            ("--typical", "typical_p", SAMPLING_AUTO_FLOAT),
            ("--repeat-penalty", "repeat_penalty", SAMPLING_AUTO_FLOAT),
            ("--repeat-last-n", "repeat_last_n", SAMPLING_LAST_N_AUTO),
            ("--presence-penalty", "presence_penalty", SAMPLING_PENALTY_AUTO),
            ("--frequency-penalty", "frequency_penalty", SAMPLING_PENALTY_AUTO),
            ("--seed", "seed", SAMPLING_SEED_AUTO),
        )
        for flag, field_name, auto_value in sampling_values:
            value = getattr(cfg, field_name, auto_value)
            if value > auto_value:
                args += [flag, str(value)]
        if cfg.flash_attn:
            args += ["--flash-attn", "on"]
        if getattr(cfg, "speculative_mtp", False):
            # Keep MTP CLI minimal and close to llama.cpp/Unsloth examples.
            draft_model = str(getattr(cfg, "spec_draft_model_path", "") or "").strip()
            if draft_model:
                args += ["--model-draft", draft_model]
            args += [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(max(1, int(getattr(cfg, "spec_draft_n_max", 8) or 8))),
                "--spec-draft-p-min",
                str(min(1.0, max(0.0, float(getattr(cfg, "spec_draft_p_min", 0.8))))),
            ]
            draft_ngl = str(getattr(cfg, "spec_draft_gpu_layers", "") or "").strip()
            if draft_ngl:
                args += ["--spec-draft-ngl", draft_ngl]
            draft_device = str(getattr(cfg, "spec_draft_device", "") or "").strip()
            if draft_device:
                args += ["--spec-draft-device", draft_device]

        # mmproj logic expects model_info dict, passed separately if needed
        # handled in UI layer or passed via kwargs if necessary.
        # For purity, we assume mmproj flags are pre-resolved or passed in cfg.
        if hasattr(cfg, "mmproj_path") and cfg.mmproj_path:
            if cfg.use_mmproj:
                args += ["-mm", cfg.mmproj_path]
            if not cfg.mmproj_offload:
                args.append("--no-mmproj-offload")
        if not cfg.use_mmproj:
            args.append("--no-mmproj")

        if "use_mlock" in MANAGED_FIELD_NAMES and cfg.use_mlock:
            args.append("--mlock")
        if "verbose" in MANAGED_FIELD_NAMES and cfg.verbose:
            args.append("--verbose")
        if "log_timestamps" in MANAGED_FIELD_NAMES and cfg.log_timestamps:
            args.append("--log-timestamps")
        if "context_shift" in MANAGED_FIELD_NAMES and cfg.context_shift:
            args.append("--context-shift")
        if "no_webui" in MANAGED_FIELD_NAMES and cfg.no_webui:
            args.append("--no-webui")
        if cfg.jinja:
            args.append("--jinja")
        if "use_chat_template" in MANAGED_FIELD_NAMES:
            chat_template = str(getattr(cfg, "chat_template_file", "") or "").strip()
            if getattr(cfg, "use_chat_template", False) and chat_template:
                args += ["--chat-template-file", chat_template]
        args.append("--metrics")

    if cfg.extra_args.strip():
        try:
            extra = shlex.split(cfg.extra_args)
            errs = validate_extra_args(extra, cfg.model_dir)
            if errs:
                raise ValueError("; ".join(errs))
            managed_args = args
            if getattr(cfg, "speculative_mtp", False):
                managed_args = args + [
                    "--model-draft",
                    "--spec-type",
                    "--spec-draft-n-max",
                    "--spec-draft-p-min",
                    "--spec-draft-ngl",
                    "--spec-draft-device",
                ]
            args.extend(_filter_duplicate_extra_args(extra, managed_args))
        except ValueError as e:
            raise ValueError(f"Ошибка доп. параметров: {e}")
    return args
