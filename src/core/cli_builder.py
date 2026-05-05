"""Сборщик CLI-аргументов для llama-server и llama-bench."""
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.constants import LLAMA_ALLOWED_FLAGS
from src.utils.file_utils import validate_path

def validate_extra_args(args: List[str], model_dir: str) -> List[str]:
    """Валидация дополнительных аргументов. Возвращает список ошибок."""
    invalid = []
    i = 0
    base_dir = Path(model_dir or ".").resolve()
    path_flags = {"--grammar-file", "--api-key-file", "--lora", "--lora-scaled", "--mmproj", "--chat-template-file"}

    while i < len(args):
        arg = args[i]
        if not arg.startswith("-"):
            i += 1
            continue

        base_arg = arg.split("=")[0] if "=" in arg else arg
        if base_arg not in LLAMA_ALLOWED_FLAGS:
            invalid.append(f"{arg} (неизвестный флаг)")
            i += 1
            continue

        if base_arg in path_flags:
            value = arg.split("=", 1)[1] if "=" in arg else (args[i+1] if i+1 < len(args) else None)
            if value:
                if "=" not in arg: i += 1
                try:
                    validate_path(value, base_dir=base_dir)
                except ValueError as e:
                    invalid.append(f"{arg} {value} (недопустимый путь: {e})")
            else:
                invalid.append(f"{arg} (требуется значение)")

        if base_arg == "--host":
            value = arg.split("=", 1)[1] if "=" in arg else (args[i+1] if i+1 < len(args) else None)
            if value:
                if "=" not in arg: i += 1
                if value in ("0.0.0.0", "::", ""):
                    invalid.append(f"{arg} {value} (запрещено: открывает сервер всем интерфейсам)")
            else:
                invalid.append(f"{arg} (требуется значение)")
        i += 1
    return invalid

def build_args(cfg: Any, model_path: str, for_benchmark: bool = False) -> Optional[List[str]]:
    """Сборка аргументов. cfg - объект AppSettings или аналогичный интерфейс."""
    if not model_path:
        return None

    args = ["-m", model_path]
    gpu_val = "99" if for_benchmark else "auto" if cfg.gpu_auto else str(cfg.gpu_layers)

    if for_benchmark:
        args += ["-p", str(cfg.bench_prompt), "-n", str(cfg.bench_gen), "-ngl", gpu_val]
        if cfg.flash_attn: args += ["-fa", "1"]
        args += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v]
        args += ["-b", str(cfg.batch_size), "-ub", str(min(cfg.ubatch_size, cfg.batch_size))]
    else:
        args += ["--port", str(cfg.port), "-ngl", gpu_val, "-c", str(cfg.ctx_size), "-t", str(cfg.threads)]
        if cfg.threads_batch > 0: args += ["-tb", str(cfg.threads_batch)]
        args += ["-b", str(cfg.batch_size), "-ub", str(min(cfg.ubatch_size, cfg.batch_size))]
        args += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v, "-np", str(cfg.parallel_slots)]
        if cfg.cpu_moe_layers > 0: args += ["-ncmoe", str(cfg.cpu_moe_layers)]
        if cfg.fit_off: args += ["--fit", "off"]
        if cfg.reasoning_mode != "auto": args += ["-rea", cfg.reasoning_mode]
        if cfg.ctx_checkpoints >= 0: args += ["--ctx-checkpoints", str(cfg.ctx_checkpoints)]
        if cfg.cache_ram >= -1: args += ["--cache-ram", str(cfg.cache_ram)]
        args += ["--temp", str(cfg.temperature), "--repeat-penalty", str(cfg.repeat_penalty)]
        if cfg.flash_attn: args += ["--flash-attn", "on"]

        # mmproj logic expects model_info dict, passed separately if needed
        # handled in UI layer or passed via kwargs if necessary.
        # For purity, we assume mmproj flags are pre-resolved or passed in cfg.
        if hasattr(cfg, "mmproj_path") and cfg.mmproj_path:
            if cfg.use_mmproj: args += ["-mm", cfg.mmproj_path]
            if not cfg.mmproj_offload: args.append("--no-mmproj-offload")
        elif not cfg.use_mmproj:
            args.append("--no-mmproj")

        if cfg.use_mmap: args.append("--mmap")
        else: args.append("--no-mmap")
        if cfg.use_mlock: args.append("--mlock")
        if cfg.verbose: args.append("--verbose")
        if cfg.log_timestamps: args.append("--log-timestamps")
        if not cfg.cont_batching: args.append("--no-cont-batching")
        if not cfg.cache_prompt: args.append("--no-cache-prompt")
        if cfg.context_shift: args.append("--context-shift")
        if cfg.no_webui: args.append("--no-webui")

    if cfg.extra_args.strip():
        try:
            extra = shlex.split(cfg.extra_args)
            errs = validate_extra_args(extra, cfg.model_dir)
            if errs:
                raise ValueError("; ".join(errs))
            args.extend(extra)
        except ValueError as e:
            raise ValueError(f"Ошибка доп. параметров: {e}")
    return args