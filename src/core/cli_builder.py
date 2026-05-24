"""Сборщик CLI-аргументов для llama-server и llama-bench."""

import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

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


def _filter_duplicate_extra_args(
    extra: List[str], existing_args: List[str]
) -> List[str]:
    """Удаляет из extra_args флаги, которыми уже управляют UI/AutoTune.

    Это предотвращает CLI вроде `--ctx-checkpoints 0 ... --ctx-checkpoints 0`.
    Если флаг не был добавлен UI, он остаётся валидным extra arg.
    """
    flags_with_values = {
        "-m",
        "--model",
        "--port",
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
        "--flash-attn",
        "-fa",
        "-mm",
        "--mmproj",
    }
    bool_flags = {
        "--no-mmproj",
        "--no-mmproj-offload",
        "--mmap",
        "--no-mmap",
        "--mlock",
        "--verbose",
        "--log-timestamps",
        "--no-cont-batching",
        "--no-cache-prompt",
        "--context-shift",
        "--no-webui",
        "--jinja",
    }
    managed = {_flag_base(a) for a in existing_args if str(a).startswith("-")}
    filtered: List[str] = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        base = _flag_base(arg)
        if base in managed and base in flags_with_values:
            if (
                "=" not in arg
                and i + 1 < len(extra)
                and not extra[i + 1].startswith("-")
            ):
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


def build_benchmark_args_from_params(
    model_path: str,
    params: Dict[str, Any],
    prompt_tokens: int = 128,
    generation_tokens: int = 256,
) -> Optional[List[str]]:
    """Сборка аргументов llama-bench для одного AutoTune-кандидата."""
    if not model_path:
        return None

    args = ["-m", model_path, "-p", str(prompt_tokens), "-n", str(generation_tokens)]

    # Важно: llama-bench CLI отличается от llama-server.
    # В актуальных сборках llama-bench нет -c/-np/--ctx-checkpoints/--cache-ram/--no-mmproj.
    # Context Size остаётся частью AutoTune-плана/пресета, но в llama-bench проверяется через
    # выбранные prompt/gen размеры; полный server-context тест будет отдельным server-engine этапом.
    ngl = params.get("ngl", "auto")
    args += ["-ngl", "99" if str(ngl).lower() == "auto" else str(ngl)]

    flash_attn = bool(params.get("flash_attn", True))
    args += ["-fa", "1" if flash_attn else "0"]

    args += [
        "-ctk",
        str(params.get("cache_type_k", "q8_0")),
        "-ctv",
        str(params.get("cache_type_v", "q8_0")),
    ]

    batch_size = int(params.get("batch_size") or 512)
    ubatch_size = int(params.get("ubatch_size") or min(batch_size, 512))
    args += ["-b", str(batch_size), "-ub", str(min(ubatch_size, batch_size))]

    threads = int(params.get("threads") or 0)
    if threads > 0:
        args += ["-t", str(threads)]

    threads_batch = int(params.get("threads_batch") or 0)
    if threads_batch > 0:
        args += ["-tb", str(threads_batch)]

    ncmoe = int(params.get("ncmoe", -1))
    if ncmoe >= 0:
        args += ["-ncmoe", str(ncmoe)]

    return args


def build_args(
    cfg: Any, model_path: str, for_benchmark: bool = False
) -> Optional[List[str]]:
    """Сборка аргументов. cfg - объект AppSettings или аналогичный интерфейс."""
    if not model_path:
        return None

    args = ["-m", model_path]
    if for_benchmark:
        gpu_val = "99" if cfg.gpu_auto else str(cfg.gpu_layers)
    else:
        gpu_val = "auto" if cfg.gpu_auto else str(cfg.gpu_layers)

    if for_benchmark:
        args += ["-p", str(cfg.bench_prompt), "-n", str(cfg.bench_gen), "-ngl", gpu_val]
        args += ["-fa", "1" if cfg.flash_attn else "0"]
        args += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v]
        bs = int(cfg.batch_size if cfg.batch_size and cfg.batch_size > 0 else 512)
        ub = int(cfg.ubatch_size if cfg.ubatch_size and cfg.ubatch_size > 0 else min(bs, 512))
        args += ["-b", str(bs), "-ub", str(min(ub, bs))]
        if cfg.threads > 0:
            args += ["-t", str(cfg.threads)]
        if cfg.threads_batch > 0:
            args += ["-tb", str(cfg.threads_batch)]
        if cfg.cpu_moe_layers >= 0:
            args += ["-ncmoe", str(cfg.cpu_moe_layers)]
    else:
        args += [
            "--port",
            str(cfg.port),
            "-ngl",
            gpu_val,
            "-t",
            str(cfg.threads),
        ]
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
        if cfg.reasoning_mode != "auto":
            args += ["-rea", cfg.reasoning_mode]
        thinking = getattr(cfg, "enable_thinking", "off")
        if thinking is True:
            thinking = "true"
        elif thinking is False or thinking is None:
            thinking = "off"
        else:
            thinking = str(thinking).strip().lower()
        if thinking in {"false", "true"}:
            args += [
                "--chat-template-kwargs",
                f'{{"enable_thinking":{thinking}}}',
            ]
        if cfg.ctx_checkpoints >= 0:
            args += ["--ctx-checkpoints", str(cfg.ctx_checkpoints)]
        if cfg.cache_ram >= -1:
            args += ["--cache-ram", str(cfg.cache_ram)]
        if cfg.temperature >= 0:
            args += ["--temp", str(cfg.temperature)]
        if cfg.repeat_penalty >= 0:
            args += ["--repeat-penalty", str(cfg.repeat_penalty)]
        if cfg.flash_attn:
            args += ["--flash-attn", "on"]

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

        if cfg.use_mmap:
            args.append("--mmap")
        else:
            args.append("--no-mmap")
        if cfg.use_mlock:
            args.append("--mlock")
        if cfg.verbose:
            args.append("--verbose")
        if cfg.log_timestamps:
            args.append("--log-timestamps")
        if not cfg.cont_batching:
            args.append("--no-cont-batching")
        if not cfg.cache_prompt:
            args.append("--no-cache-prompt")
        if cfg.context_shift:
            args.append("--context-shift")
        if cfg.no_webui:
            args.append("--no-webui")
        if cfg.jinja:
            args.append("--jinja")

    if cfg.extra_args.strip():
        try:
            extra = shlex.split(cfg.extra_args)
            errs = validate_extra_args(extra, cfg.model_dir)
            if errs:
                raise ValueError("; ".join(errs))
            args.extend(_filter_duplicate_extra_args(extra, args))
        except ValueError as e:
            raise ValueError(f"Ошибка доп. параметров: {e}")
    return args
