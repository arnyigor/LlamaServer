# src/core/param_registry.py
"""Единый реестр параметров llama-server / llama-bench.

Единственный источник правды о том, как поле AppSettings связано с
виджетом UI и CLI-флагами. Все производные таблицы (маппинг виджетов,
managed extra flags, наборы флагов парсера/билдера) генерируются отсюда,
чтобы новый флаг llama.cpp добавлялся одной строкой в PARAM_REGISTRY,
а не правками в 4 файлах.

cli_kind:
    "none"       — параметр без CLI-флага (только UI/настройки);
    "value"      — флаг со значением, приводится к field_type;
    "bool"       — флаг-переключатель, устанавливает True;
    "false"      — флаг-переключатель, устанавливает False (--no-*);
    "bool_value" — значение опционально ("-fa on" / "-fa");
    "special"    — составная логика парсинга/сборки (см. обработчики);
    "ignore"     — известный флаг, который парсер молча пропускает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple, Type


@dataclass(frozen=True)
class ParamSpec:
    name: str  # поле AppSettings
    widget_attr: str | None = None  # атрибут MainWindowUI (None — без виджета)
    cli_flags: Tuple[str, ...] = ()  # все написания флага на CLI
    cli_kind: str = "none"
    field_type: Type[Any] | None = None
    # Противоположная полярность для булевых полей (например --no-mmap).
    cli_neg_flags: Tuple[str, ...] = ()
    managed: bool = True  # вырезается из extra_args пользователя
    server_only: bool = False  # не передаётся в llama-bench
    benchmark_flag: str | None = None  # отличающееся написание для llama-bench


# Порядок повторяет _FIELD_WIDGET_MAP из config.py (до миграции) для
# простоты сверки. Поля model_path/mmproj_path — runtime-поля без виджета,
# нужны парсеру CLI и потому тоже описаны здесь.
PARAM_REGISTRY: Tuple[ParamSpec, ...] = (
    ParamSpec("exe", "exe_path"),
    ParamSpec("bench", "bench_path"),
    ParamSpec("model_dir", "model_dir"),
    ParamSpec("opencode_config", "opencode_config_path"),
    ParamSpec("pi_config", "pi_config_path"),
    ParamSpec("bench_prompt", "bench_prompt"),
    ParamSpec("bench_gen", "bench_gen"),
    ParamSpec("auto_params", "auto_params"),
    ParamSpec("use_mmproj", "use_mmproj", ("--no-mmproj",), "false", bool),
    ParamSpec(
        "mmproj_offload", "mmproj_offload", ("--no-mmproj-offload",), "false", bool
    ),
    # gpu_auto/gpu_layers_all выставляются спец-обработчиком -ngl.
    ParamSpec("gpu_auto", "gpu_auto"),
    ParamSpec("gpu_layers", "gpu_layers", ("-ngl", "--n-gpu-layers"), "special", int),
    ParamSpec("gpu_layers_all", "gpu_layers_all"),
    ParamSpec(
        "cpu_moe_layers", "cpu_moe_layers", ("-ncmoe", "--n-cpu-moe"), "value", int
    ),
    ParamSpec("ctx_size", "ctx_size", ("-c", "--ctx-size"), "value", int),
    ParamSpec("threads", "threads", ("-t", "--threads"), "value", int),
    ParamSpec(
        "threads_batch",
        "threads_batch",
        ("-tb", "--threads-batch"),
        "value",
        int,
        server_only=True,
    ),
    ParamSpec("port", "port", ("--port",), "value", int),
    ParamSpec("host", "host", ("--host",), "value", str),
    # --device/--split-mode/--main-gpu исторически не вырезаются из
    # extra_args (см. комментарий в MANAGED_EXTRA_FLAGS).
    ParamSpec("cuda_device", "cuda_device", ("--device",), "value", str, managed=False),
    ParamSpec(
        "spec_draft_device", "spec_draft_device", ("--spec-draft-device",), "value", str
    ),
    ParamSpec(
        "split_mode", "split_mode", ("--split-mode",), "value", str, managed=False
    ),
    ParamSpec("main_gpu", "main_gpu", ("--main-gpu",), "value", int, managed=False),
    # EXTRA-параметры (managed=False): живут verbatim в extra_args, не
    # управляются builder'ом/виджетами. Флаги задаются явно в
    # _extra_flag_tokens (config.py), т.к. в реестре cli_flags пусты.
    ParamSpec("cuda_visible_devices", "cuda_visible_devices", managed=False),
    ParamSpec("cuda_module_loading", "cuda_module_loading", managed=False),
    ParamSpec(
        "temperature", "temperature", ("--temp", "--temperature"), "value", float
    ),
    ParamSpec("top_k", "top_k", ("--top-k",), "value", int),
    ParamSpec("top_p", "top_p", ("--top-p",), "value", float),
    ParamSpec("min_p", "min_p", ("--min-p",), "value", float),
    ParamSpec("typical_p", "typical_p", ("--typical", "--typical-p"), "value", float),
    ParamSpec(
        "repeat_penalty", "repeat_penalty", ("--repeat-penalty",), "value", float
    ),
    ParamSpec("repeat_last_n", "repeat_last_n", ("--repeat-last-n",), "value", int),
    ParamSpec(
        "presence_penalty", "presence_penalty", ("--presence-penalty",), "value", float
    ),
    ParamSpec(
        "frequency_penalty",
        "frequency_penalty",
        ("--frequency-penalty",),
        "value",
        float,
    ),
    ParamSpec("seed", "seed", ("-s", "--seed"), "value", int),
    ParamSpec(
        "flash_attn",
        "flash_attn",
        ("-fa", "--flash-attn"),
        "bool_value",
        bool,
        benchmark_flag="-fa",
    ),
    ParamSpec("fit_off", "fit_off", ("--fit",), "special", bool),
    ParamSpec(
        "reasoning_mode", "reasoning_mode", ("-rea", "--reasoning"), "value", str
    ),
    ParamSpec(
        "reasoning_effort", "reasoning_effort", ("--reasoning-effort",), "value", str
    ),
    ParamSpec(
        "reasoning_preserve",
        "reasoning_preserve",
        ("--reasoning-preserve", "--no-reasoning-preserve"),
        "special",
        str,
    ),
    ParamSpec(
        "reasoning_budget", "reasoning_budget", ("--reasoning-budget",), "value", int
    ),
    ParamSpec(
        "reasoning_budget_message",
        "reasoning_budget_message",
        ("--reasoning-budget-message",),
        "value",
        str,
    ),
    # UI-галочка и серверная эмиссия убраны; поле оставлено для обратной
    # совместимости/автотьюна. Флаг не managed -> не вырезается из extra params
    # (пользователь может добавить --load-mode mmap вручную).
    ParamSpec("use_mmap", None, (), "bool", bool, managed=False),
    # Ниже — EXTRA-параметры: managed=False. Ими управляет пользователь
    # через свободное поле extra_args; builder их не эмитит, а
    # _sanitize_extra_args/_filter_duplicate_extra_args их не вырезают.
    ParamSpec("use_mlock", "use_mlock", ("--mlock",), "bool", bool, managed=False),
    ParamSpec("verbose", "verbose", ("--verbose",), "bool", bool, managed=False),
    ParamSpec(
        "log_timestamps",
        "log_timestamps",
        ("--log-timestamps",),
        "bool",
        bool,
        managed=False,
    ),
    ParamSpec("cache_type_k", "cache_type_k", ("-ctk", "--cache-type-k"), "value", str),
    ParamSpec("cache_type_v", "cache_type_v", ("-ctv", "--cache-type-v"), "value", str),
    ParamSpec("batch_size", "batch_size", ("-b", "--batch-size"), "value", int),
    ParamSpec("ubatch_size", "ubatch_size", ("-ub", "--ubatch-size"), "value", int),
    ParamSpec("parallel_slots", "parallel_slots", ("-np", "--parallel"), "value", int),
    ParamSpec(
        "kv_unified",
        "kv_unified",
        ("--kv-unified", "-kvu"),
        "bool",
        bool,
        managed=False,
    ),
    # --spec-type draft-mtp включает speculative_mtp; -md дополнительно
    # заполняет spec_draft_model_path — см. спец-обработчики cli_parser.
    ParamSpec("speculative_mtp", "speculative_mtp", ("--spec-type",), "special", bool),
    ParamSpec(
        "spec_draft_model_path",
        "spec_draft_model_path",
        ("-md", "--model-draft"),
        "special",
        str,
    ),
    ParamSpec(
        "spec_draft_n_max", "spec_draft_n_max", ("--spec-draft-n-max",), "value", int
    ),
    ParamSpec(
        "spec_draft_p_min", "spec_draft_p_min", ("--spec-draft-p-min",), "value", float
    ),
    ParamSpec(
        "spec_draft_gpu_layers",
        "spec_draft_gpu_layers",
        ("--spec-draft-ngl", "-ngld"),
        "value",
        str,
    ),
    ParamSpec(
        "ctx_checkpoints",
        "ctx_checkpoints",
        ("--ctx-checkpoints",),
        "value",
        int,
        managed=False,
    ),
    ParamSpec("cache_ram", "cache_ram", ("--cache-ram",), "value", int, managed=False),
    # UI-галочка и серверная эмиссия --no-cont-batching убраны; поле оставлено.
    # Флаг не managed -> не вырезается из extra params.
    ParamSpec("cont_batching", None, (), "bool", bool, managed=False),
    # UI-галочка и серверная эмиссия убраны; поле оставлено. Флаг не managed ->
    # пользователь может добавить --cache-prompt вручную через extra params.
    ParamSpec("cache_prompt", None, (), "bool", bool, managed=False),
    ParamSpec(
        "context_shift",
        "context_shift",
        ("--context-shift",),
        "bool",
        bool,
        managed=False,
    ),
    ParamSpec("no_webui", "no_webui", ("--no-webui",), "bool", bool, managed=False),
    ParamSpec("jinja", "jinja", ("--jinja",), "bool", bool),
    ParamSpec(
        "use_chat_template",
        "use_chat_template",
        ("--chat-template-file",),
        "special",
        bool,
        managed=False,
    ),
    ParamSpec("chat_template_file", "chat_template_file"),
    ParamSpec("extra_args", "extra_args"),
    ParamSpec("enable_thinking", "enable_thinking"),
    ParamSpec("cuda_version", "cuda_version_combo"),
    ParamSpec("hf_repo", "hf_repo"),
    ParamSpec("hf_quant_filter", "hf_quant_filter"),
    ParamSpec("hf_include_mmproj", "hf_include_mmproj"),
    # Runtime-поля без виджета: путь модели и mmproj-файл приходят из CLI.
    ParamSpec("model_path", None, ("-m", "--model"), "special", str),
    ParamSpec("mmproj_path", None, ("-mm", "--mmproj"), "special", str),
    # Известный llama-server флаг, GUI всегда добавляет его сам.
    ParamSpec("metrics", None, ("--metrics",), "ignore", bool),
)

# Поля, попадающие в _FIELD_WIDGET_MAP (config.py).
FIELD_WIDGET_MAP: Dict[str, str] = {
    spec.name: spec.widget_attr for spec in PARAM_REGISTRY if spec.widget_attr
}

# Флаги, которыми управляет UI/AutoTune: вырезаются из extra_args при
# загрузке настроек (_sanitize_extra_args). --device/--split-mode/--main-gpu
# намеренно исключены (managed=False): builder добавляет их только когда
# соответствующее поле UI непустое, поэтому вырезание молча теряло бы
# пользовательское значение.
MANAGED_EXTRA_FLAGS: FrozenSet[str] = frozenset(
    flag
    for spec in PARAM_REGISTRY
    if spec.managed
    for flag in (spec.cli_flags + spec.cli_neg_flags)
)

# Имена полей, которыми управляет UI/AutoTune. Используется в cli_builder,
# чтобы не эмитить флаги EXTRA-параметров (managed=False): те живут verbatim
# в extra_args и не перезаписываются builder'ом.
MANAGED_FIELD_NAMES: FrozenSet[str] = frozenset(
    spec.name for spec in PARAM_REGISTRY if spec.managed
)

# Sampling-поля для миграции старых флагов из extra_args (config.py).
_SAMPLING_FIELD_NAMES = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)

SAMPLING_EXTRA_FIELDS: Dict[str, Tuple[str, Type[Any]]] = {
    flag: (spec.name, field_type)
    for spec in PARAM_REGISTRY
    if spec.name in _SAMPLING_FIELD_NAMES
    and spec.cli_kind == "value"
    # value-спеки всегда объявлены с field_type (проверяет реестровый тест)
    and (field_type := spec.field_type) is not None
    for flag in spec.cli_flags
}

# --- Производные таблицы для cli_parser -------------------------------------

# Флаги, требующие значение (не считая опциональных).
VALUE_FLAGS: FrozenSet[str] = frozenset(
    flag
    for spec in PARAM_REGISTRY
    if spec.cli_kind in ("value", "special")
    for flag in spec.cli_flags
)

# Флаги, у которых значение опционально.
OPTIONAL_VALUE_FLAGS: FrozenSet[str] = frozenset(
    flag
    for spec in PARAM_REGISTRY
    if spec.cli_kind == "bool_value"
    for flag in spec.cli_flags
)

# flag -> ParamSpec для диспетчеризации парсера (без neg-флагов).
FLAG_TO_SPEC: Dict[str, ParamSpec] = {
    flag: spec for spec in PARAM_REGISTRY for flag in spec.cli_flags
}

# neg-флаг -> ParamSpec (полярность False).
NEG_FLAG_TO_SPEC: Dict[str, ParamSpec] = {
    flag: spec for spec in PARAM_REGISTRY for flag in spec.cli_neg_flags
}

# --- Производные таблицы для cli_builder._filter_duplicate_extra_args -------

# Значения с consumed-value семантикой, которых нет в реестре
# (draft-KV-квантование и chat-template-kwargs в UI не управляются).
_EXTRA_FILTER_VALUE_FLAGS = frozenset(
    {
        "--chat-template-kwargs",
        "--spec-draft-n-min",
        "--spec-draft-type-k",
        "-ctkd",
        "--spec-draft-type-v",
        "-ctvd",
    }
)

# Флаги, потребляющие следующий токен-значение.
FILTER_VALUE_FLAGS: FrozenSet[str] = (
    frozenset(
        flag
        for spec in PARAM_REGISTRY
        if spec.cli_kind in ("value", "special", "bool_value")
        for flag in (spec.cli_flags + spec.cli_neg_flags)
    )
    | _EXTRA_FILTER_VALUE_FLAGS
)

# Автономные флаги-переключатели.
FILTER_BOOL_FLAGS: FrozenSet[str] = frozenset(
    flag
    for spec in PARAM_REGISTRY
    if spec.cli_kind in ("bool", "false", "ignore")
    for flag in (spec.cli_flags + spec.cli_neg_flags)
)

__all__ = [
    "ParamSpec",
    "PARAM_REGISTRY",
    "FIELD_WIDGET_MAP",
    "MANAGED_EXTRA_FLAGS",
    "SAMPLING_EXTRA_FIELDS",
    "VALUE_FLAGS",
    "OPTIONAL_VALUE_FLAGS",
    "FLAG_TO_SPEC",
    "NEG_FLAG_TO_SPEC",
    "FILTER_VALUE_FLAGS",
    "FILTER_BOOL_FLAGS",
]
