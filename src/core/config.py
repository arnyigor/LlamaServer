# src/core/config.py
"""Менеджер конфигурации с автоматическим маппингом виджетов."""

import json
import os
import hashlib
import shlex
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Type
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
)

from src.utils.file_utils import write_json_file_safely


@dataclass
class AppSettings:
    exe: str = ""
    bench: str = ""
    model_dir: str = ""
    opencode_config: str = ""
    pi_config: str = ""
    claude_config: str = field(
        default_factory=lambda: str(Path.home() / ".claude" / "settings.json")
    )
    integration_target: str = "opencode"
    bench_prompt: int = 128
    bench_gen: int = 256
    auto_params: bool = True
    use_mmproj: bool = True
    mmproj_offload: bool = True
    mmproj_path: str = ""
    last_model_path: str = ""
    model_cache: list = field(default_factory=list)
    temperature: float = -1.0
    top_k: int = -1
    top_p: float = -1.0
    min_p: float = -1.0
    typical_p: float = -1.0
    repeat_penalty: float = -1.0
    repeat_last_n: int = -2
    presence_penalty: float = -3.0
    frequency_penalty: float = -3.0
    seed: int = -2
    gpu_auto: bool = True
    gpu_layers: int = 33
    gpu_layers_all: bool = False
    cpu_moe_layers: int = -1
    ctx_size: int = -1
    threads: int = 4
    threads_batch: int = 0
    port: int = 8080
    host: str = "127.0.0.1"
    cuda_device: str = ""
    spec_draft_device: str = ""
    split_mode: str = ""
    main_gpu: int = -1
    cuda_visible_devices: str = ""
    cuda_module_loading: str = "LAZY"
    flash_attn: bool = True
    fit_off: bool = True
    reasoning_mode: str = "off"
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False
    log_timestamps: bool = False
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    batch_size: int = -1
    ubatch_size: int = -1
    parallel_slots: int = -1
    kv_unified: bool = False
    speculative_mtp: bool = False
    spec_draft_model_path: str = ""
    spec_draft_auto_disabled_models: list = field(default_factory=list)
    spec_draft_manual_paths: dict = field(default_factory=dict)
    spec_draft_n_max: int = 8
    spec_draft_p_min: float = 0.8
    spec_draft_gpu_layers: str = "all"
    ctx_checkpoints: int = -1
    cache_ram: int = -2
    cont_batching: bool = True
    cache_prompt: bool = True
    context_shift: bool = False
    no_webui: bool = False
    jinja: bool = False
    use_chat_template: bool = False
    chat_template_file: str = ""
    extra_args: str = ""
    enable_thinking: str = "off"
    cuda_version: str = "12"
    hf_repo: str = ""
    hf_quant_filter: str = "Q3-BF16"
    hf_include_mmproj: bool = True


# Явная таблица маппинга: поле -> атрибут виджета в UI
# Формат: "settings_field": "ui_widget_attr"
_FIELD_WIDGET_MAP: Dict[str, str] = {
    "exe": "exe_path",
    "bench": "bench_path",
    "model_dir": "model_dir",
    "opencode_config": "opencode_config_path",
    "pi_config": "pi_config_path",
    "claude_config": "claude_config_path",
    "bench_prompt": "bench_prompt",
    "bench_gen": "bench_gen",
    "auto_params": "auto_params",
    "use_mmproj": "use_mmproj",
    "mmproj_offload": "mmproj_offload",
    "gpu_auto": "gpu_auto",
    "gpu_layers": "gpu_layers",
    "gpu_layers_all": "gpu_layers_all",
    "cpu_moe_layers": "cpu_moe_layers",
    "ctx_size": "ctx_size",
    "threads": "threads",
    "threads_batch": "threads_batch",
    "port": "port",
    "host": "host",
    "cuda_device": "cuda_device",
    "spec_draft_device": "spec_draft_device",
    "split_mode": "split_mode",
    "main_gpu": "main_gpu",
    "cuda_visible_devices": "cuda_visible_devices",
    "cuda_module_loading": "cuda_module_loading",
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "typical_p": "typical_p",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "flash_attn": "flash_attn",
    "fit_off": "fit_off",
    "reasoning_mode": "reasoning_mode",
    "use_mmap": "use_mmap",
    "use_mlock": "use_mlock",
    "verbose": "verbose",
    "log_timestamps": "log_timestamps",
    "cache_type_k": "cache_type_k",
    "cache_type_v": "cache_type_v",
    "batch_size": "batch_size",
    "ubatch_size": "ubatch_size",
    "parallel_slots": "parallel_slots",
    "kv_unified": "kv_unified",
    "speculative_mtp": "speculative_mtp",
    "spec_draft_model_path": "spec_draft_model_path",
    "spec_draft_n_max": "spec_draft_n_max",
    "spec_draft_p_min": "spec_draft_p_min",
    "spec_draft_gpu_layers": "spec_draft_gpu_layers",
    "ctx_checkpoints": "ctx_checkpoints",
    "cache_ram": "cache_ram",
    "cont_batching": "cont_batching",
    "cache_prompt": "cache_prompt",
    "context_shift": "context_shift",
    "no_webui": "no_webui",
    "jinja": "jinja",
    "use_chat_template": "use_chat_template",
    "chat_template_file": "chat_template_file",
    "extra_args": "extra_args",
    "enable_thinking": "enable_thinking",
    "cuda_version": "cuda_version_combo",
    "hf_repo": "hf_repo",
    "hf_quant_filter": "hf_quant_filter",
    "hf_include_mmproj": "hf_include_mmproj",
}

_PERF_PRESETS_ROOT = "__perf_presets__"
_PERF_DEFAULT_PRESET_NAME = "default"

_PERF_PRESET_FIELDS = (
    "gpu_auto",
    "gpu_layers",
    "gpu_layers_all",
    "cpu_moe_layers",
    "ctx_size",
    "threads",
    "threads_batch",
    "cache_type_k",
    "cache_type_v",
    "batch_size",
    "ubatch_size",
    "parallel_slots",
    "kv_unified",
    "host",
    "cuda_device",
    "spec_draft_device",
    "split_mode",
    "main_gpu",
    "cuda_visible_devices",
    "cuda_module_loading",
    "speculative_mtp",
    "spec_draft_model_path",
    "spec_draft_n_max",
    "spec_draft_p_min",
    "spec_draft_gpu_layers",
    "flash_attn",
    "fit_off",
    "reasoning_mode",
    "ctx_checkpoints",
    "cache_ram",
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
    "use_mmap",
    "use_mlock",
    "verbose",
    "log_timestamps",
    "cont_batching",
    "cache_prompt",
    "context_shift",
    "no_webui",
    "jinja",
    "use_mmproj",
    "mmproj_offload",
    "extra_args",
    "enable_thinking",
)

_AUTOTUNE_PARAM_TO_SETTING = {
    "ngl": "gpu_layers",
    "gpu_layers_all": "gpu_layers_all",
    "batch_size": "batch_size",
    "ubatch_size": "ubatch_size",
    "cache_type_k": "cache_type_k",
    "cache_type_v": "cache_type_v",
    "threads": "threads",
    "threads_batch": "threads_batch",
    "parallel_slots": "parallel_slots",
    "kv_unified": "kv_unified",
    "speculative_mtp": "speculative_mtp",
    "spec_draft_model_path": "spec_draft_model_path",
    "spec_draft_n_max": "spec_draft_n_max",
    "spec_draft_p_min": "spec_draft_p_min",
    "spec_draft_gpu_layers": "spec_draft_gpu_layers",
    "flash_attn": "flash_attn",
    "fit_off": "fit_off",
    "cache_prompt": "cache_prompt",
    "ctx_checkpoints": "ctx_checkpoints",
    "cache_ram": "cache_ram",
    "use_mmproj": "use_mmproj",
}

_MANAGED_EXTRA_FLAGS = {
    "-m",
    "--model",
    "--port",
    "--host",
    "-ngl",
    "--n-gpu-layers",
    "-c",
    "--ctx-size",
    "--ctx-checkpoints",
    "--cache-ram",
    "--kv-unified",
    "-kvu",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-p-min",
    "--spec-draft-ngl",
    "--spec-draft-device",
    "-md",
    "--model-draft",
    "--jinja",
    "--chat-template-file",
    "--no-cache-prompt",
    "--flash-attn",
    "-fa",
    "--fit",
    "-rea",
    "--reasoning",
    "--temp",
    "--temperature",
    "--top-k",
    "--top-p",
    "--min-p",
    "--typical",
    "--typical-p",
    "--repeat-penalty",
    "--repeat-last-n",
    "--presence-penalty",
    "--frequency-penalty",
    "-s",
    "--seed",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-np",
    "--parallel",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-ncmoe",
    "--n-cpu-moe",
    "-mm",
    "--mmproj",
    "--no-mmproj",
    "--no-mmproj-offload",
    "--mmap",
    "--no-mmap",
    "--mlock",
    "--verbose",
    "--log-timestamps",
    "--no-cont-batching",
    "--context-shift",
    "--no-webui",
}

_SAMPLING_EXTRA_FIELDS = {
    "--temp": ("temperature", float),
    "--temperature": ("temperature", float),
    "--top-k": ("top_k", int),
    "--top-p": ("top_p", float),
    "--min-p": ("min_p", float),
    "--typical": ("typical_p", float),
    "--typical-p": ("typical_p", float),
    "--repeat-penalty": ("repeat_penalty", float),
    "--repeat-last-n": ("repeat_last_n", int),
    "--presence-penalty": ("presence_penalty", float),
    "--frequency-penalty": ("frequency_penalty", float),
    "-s": ("seed", int),
    "--seed": ("seed", int),
}


def _is_extra_value_token(arg: str) -> bool:
    if not str(arg).startswith("-"):
        return True
    try:
        float(str(arg))
        return True
    except ValueError:
        return False


def _sanitize_extra_args(value: Any) -> str:
    """Убирает из extra_args флаги, которыми управляют UI/AutoTune."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except ValueError:
        return text

    result = []
    i = 0
    while i < len(parts):
        arg = parts[i]
        base = arg.split("=", 1)[0]
        if base in _MANAGED_EXTRA_FLAGS:
            if (
                "=" not in arg
                and i + 1 < len(parts)
                and _is_extra_value_token(parts[i + 1])
            ):
                i += 2
            else:
                i += 1
            continue
        result.append(arg)
        i += 1
    return " ".join(shlex.quote(p) for p in result)


def _extract_sampling_extra_args(value: Any) -> tuple[str, Dict[str, Any]]:
    """Мигрирует старые sampling-флаги из Extra params в отдельные поля."""
    text = str(value or "").strip()
    if not text:
        return "", {}
    try:
        parts = shlex.split(text)
    except ValueError:
        return text, {}

    remaining = []
    extracted: Dict[str, Any] = {}
    i = 0
    while i < len(parts):
        token = parts[i]
        base, separator, inline_value = token.partition("=")
        spec = _SAMPLING_EXTRA_FIELDS.get(base)
        if spec is None:
            remaining.append(token)
            i += 1
            continue

        raw_value = inline_value if separator else None
        consumed = False
        if raw_value is None and i + 1 < len(parts) and _is_extra_value_token(parts[i + 1]):
            raw_value = parts[i + 1]
            consumed = True
        if raw_value is None:
            remaining.append(token)
            i += 1
            continue

        field_name, converter = spec
        try:
            extracted[field_name] = converter(raw_value)
        except (TypeError, ValueError):
            remaining.append(token)
            if consumed:
                remaining.append(parts[i + 1])
        i += 2 if consumed else 1

    return " ".join(shlex.quote(p) for p in remaining), extracted


def _normalize_perf_preset_name(preset_name: Optional[str]) -> str:
    text = str(preset_name or "").strip()
    return text or _PERF_DEFAULT_PRESET_NAME


def _perf_params_from_settings(settings: AppSettings) -> Dict[str, Any]:
    return {
        field_name: getattr(settings, field_name)
        for field_name in _PERF_PRESET_FIELDS
        if hasattr(settings, field_name)
    }


def _apply_autotune_params_to_perf_params(
    params: Dict[str, Any], autotune_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not autotune_params:
        return params

    merged = dict(params)
    ngl = autotune_params.get("ngl")
    if ngl is not None:
        ngl_text = str(ngl).strip().lower()
        is_auto = ngl_text == "auto"
        is_all = ngl_text == "all"
        merged["gpu_auto"] = is_auto
        merged["gpu_layers_all"] = is_all
        if not is_auto and not is_all:
            merged["gpu_layers"] = int(ngl)

    if "ncmoe" in autotune_params:
        merged["cpu_moe_layers"] = int(autotune_params["ncmoe"])

    if "ctx_size" in autotune_params:
        merged["ctx_size"] = int(autotune_params["ctx_size"])

    for source_key, setting_key in _AUTOTUNE_PARAM_TO_SETTING.items():
        if source_key == "ngl":
            # ngl already needs special gpu_auto/gpu_layers handling above.
            # Do not copy literal "auto" into gpu_layers.
            continue
        if source_key in autotune_params:
            merged[setting_key] = autotune_params[source_key]

    # AutoTune не тестирует sampling/reasoning/mmproj-extra args. Но managed extra
    # flags от старых ручных запусков могут переопределить best preset при старте.
    merged["extra_args"] = _sanitize_extra_args(merged.get("extra_args", ""))
    return merged


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off", "disabled", "none", ""}:
        return False
    return True


def _normalize_perf_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params)
    remaining_extra, migrated_sampling = _extract_sampling_extra_args(
        normalized.get("extra_args", "")
    )
    for key, value in migrated_sampling.items():
        normalized.setdefault(key, value)
    normalized["extra_args"] = remaining_extra
    bool_fields = {
        "gpu_auto",
        "gpu_layers_all",
        "flash_attn",
        "fit_off",
        "kv_unified",
        "speculative_mtp",
        "use_mmap",
        "use_mlock",
        "verbose",
        "log_timestamps",
        "cont_batching",
        "cache_prompt",
        "context_shift",
        "no_webui",
        "jinja",
        "use_mmproj",
        "mmproj_offload",
    }
    int_fields = {
        "gpu_layers",
        "cpu_moe_layers",
        "ctx_size",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "parallel_slots",
        "spec_draft_n_max",
        "main_gpu",
        "ctx_checkpoints",
        "cache_ram",
        "top_k",
        "repeat_last_n",
        "seed",
    }
    float_fields = {
        "spec_draft_p_min",
        "temperature",
        "top_p",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    }
    for key in bool_fields & normalized.keys():
        normalized[key] = _coerce_bool(normalized[key])
    for key in int_fields & normalized.keys():
        try:
            normalized[key] = int(normalized[key])
        except (TypeError, ValueError):
            if (
                key == "gpu_layers"
                and str(normalized.get(key, "")).strip().lower() == "auto"
            ):
                normalized["gpu_auto"] = True
                normalized[key] = 99
            else:
                normalized.pop(key, None)
    for key in float_fields & normalized.keys():
        try:
            normalized[key] = float(normalized[key])
        except (TypeError, ValueError):
            normalized.pop(key, None)
    if "enable_thinking" in normalized:
        normalized["enable_thinking"] = _normalize_enable_thinking(
            normalized["enable_thinking"]
        )
    if "extra_args" in normalized:
        normalized["extra_args"] = _sanitize_extra_args(normalized["extra_args"])
    return normalized


def _normalize_enable_thinking(value: Any) -> str:
    """Нормализует Thinking в одно из значений ComboBox: off/false/true."""
    if value is True:
        return "true"
    if value is False or value is None:
        return "off"

    text = str(value).strip().lower()
    if text in {"off", "false", "true"}:
        return text
    return "off"


def _widget_get(widget: Any) -> Any:
    """Универсальное чтение значения виджета."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox,)):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        # Если у комбобокса есть userData — берём его
        data = widget.currentData()
        return data if data is not None else widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    raise TypeError(f"Unsupported widget type: {type(widget)}")


def _widget_set(widget: Any, value: Any) -> None:
    """Универсальная установка значения виджета."""
    if isinstance(widget, QCheckBox):
        widget.setChecked(_coerce_bool(value))
    elif isinstance(widget, (QSpinBox,)):
        widget.setValue(int(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QComboBox):
        # Сначала пробуем по userData, затем по тексту
        idx = widget.findData(value)
        if idx < 0:
            idx = widget.findText(str(value))
        if idx >= 0:
            widget.setCurrentIndex(idx)
        # else: не устанавливаем невалидное значение
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value) if value is not None else "")
    else:
        raise TypeError(f"Unsupported widget type: {type(widget)}")


class ConfigManager:
    def __init__(
        self,
        settings_path: str = "settings.json",
        profiles_path: str = "profiles.json",
    ):
        self.settings_path = settings_path
        self.profiles_path = profiles_path
        self.settings = AppSettings()
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        """Загрузка настроек и профилей с мягкой обработкой ошибок."""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Применяем только известные поля — защита от устаревших ключей
                valid_fields = {f.name for f in fields(AppSettings)}
                for k, v in data.items():
                    if k in valid_fields:
                        try:
                            if k == "enable_thinking":
                                v = _normalize_enable_thinking(v)
                            setattr(self.settings, k, v)
                        except (TypeError, ValueError):
                            pass  # Игнорируем невалидные значения
                remaining_extra, migrated_sampling = _extract_sampling_extra_args(
                    data.get("extra_args", "")
                )
                for key, value in migrated_sampling.items():
                    if key not in data:
                        setattr(self.settings, key, value)
                self.settings.extra_args = remaining_extra
            except (json.JSONDecodeError, OSError):
                pass  # Используем дефолтные настройки

        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.profiles = {}

    def save(self) -> None:
        write_json_file_safely(self.settings_path, asdict(self.settings))

    def save_profiles(self) -> None:
        write_json_file_safely(self.profiles_path, self.profiles)

    def apply_to_ui(self, ui: Any) -> None:
        """Автоматический маппинг настроек → виджеты UI."""
        s = self.settings
        for field_name, widget_attr in _FIELD_WIDGET_MAP.items():
            widget = getattr(ui, widget_attr, None)
            if widget is None:
                continue
            value = getattr(s, field_name, None)
            if value is None:
                continue
            try:
                _widget_set(widget, value)
            except (TypeError, ValueError):
                pass

        # Специальные случаи, не покрываемые универсальным маппингом
        idx = ui.integration_target.findData(s.integration_target)
        if idx >= 0:
            ui.integration_target.setCurrentIndex(idx)

    def read_from_ui(self, ui: Any) -> None:
        """Автоматический маппинг виджеты UI → настройки."""
        s = self.settings
        for field_name, widget_attr in _FIELD_WIDGET_MAP.items():
            widget = getattr(ui, widget_attr, None)
            if widget is None:
                continue
            try:
                value = _widget_get(widget)
                # Приводим к объявленному типу поля, а не к текущему типу значения.
                # Это важно для миграции старых settings.json, где enable_thinking
                # мог быть bool: bool("off") == True ломал CLI Preview.
                field_def = next(
                    (f for f in fields(AppSettings) if f.name == field_name), None
                )
                field_type = (
                    field_def.type
                    if field_def is not None
                    else type(getattr(s, field_name))
                )
                if field_name == "extra_args":
                    value = _sanitize_extra_args(value)
                if field_type is bool:
                    setattr(s, field_name, _coerce_bool(value))
                elif field_type is int:
                    setattr(s, field_name, int(value))
                elif field_type is float:
                    setattr(s, field_name, float(value))
                elif field_type is str:
                    setattr(s, field_name, str(value))
                else:
                    setattr(s, field_name, value)
            except (TypeError, ValueError, AttributeError):
                pass

        # Специальные случаи
        s.integration_target = ui.current_config_target()
        s.last_model_path = ui.model_combo.currentData() or s.last_model_path

    def apply_values_to_ui(self, ui: Any, values: Dict[str, Any]) -> None:
        """Apply parsed setting values to AppSettings and matching UI widgets."""
        valid_fields = {f.name: f for f in fields(AppSettings)}
        for field_name, raw_value in dict(values or {}).items():
            field_def = valid_fields.get(field_name)
            if field_def is None:
                continue

            value = raw_value
            try:
                if field_name == "extra_args":
                    value = _sanitize_extra_args(value)
                if field_def.type is bool:
                    value = _coerce_bool(value)
                elif field_def.type is int:
                    value = int(value)
                elif field_def.type is float:
                    value = float(value)
                elif field_name == "enable_thinking":
                    value = _normalize_enable_thinking(value)
                elif field_def.type is str:
                    value = str(value)
            except (TypeError, ValueError):
                continue

            setattr(self.settings, field_name, value)
            widget_attr = _FIELD_WIDGET_MAP.get(field_name)
            widget = getattr(ui, widget_attr, None) if widget_attr else None
            if widget is not None:
                try:
                    _widget_set(widget, value)
                except (TypeError, ValueError):
                    pass

    def save_profile(self, name: str, ui: Any) -> None:
        """Сохранение текущих настроек как профиля."""
        self.read_from_ui(ui)
        profile_data = asdict(self.settings)
        # Исключаем системные поля из профиля
        for key in (
            "model_cache",
            "last_model_path",
            "exe",
            "bench",
            "model_dir",
            "spec_draft_auto_disabled_models",
            "spec_draft_manual_paths",
        ):
            profile_data.pop(key, None)
        self.profiles[name] = profile_data
        self.save_profiles()

    def load_profile(self, name: str, ui: Any) -> bool:
        """Загрузка профиля в UI. Возвращает True при успехе."""
        profile = self.profiles.get(name)
        if not profile:
            return False
        profile = dict(profile)
        remaining_extra, migrated_sampling = _extract_sampling_extra_args(
            profile.get("extra_args", "")
        )
        profile["extra_args"] = remaining_extra
        for key, value in migrated_sampling.items():
            profile.setdefault(key, value)
        valid_fields = {f.name for f in fields(AppSettings)}
        for k, v in profile.items():
            if k in valid_fields:
                try:
                    if k == "enable_thinking":
                        v = _normalize_enable_thinking(v)
                    setattr(self.settings, k, v)
                except (TypeError, ValueError):
                    pass
        self.apply_to_ui(ui)
        return True

    def _perf_model_digest(self, model_path: str) -> str:
        normalized = os.path.normcase(os.path.abspath(str(model_path).strip()))
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[
            :16
        ]

    def _perf_preset_key(self, model_path: str, ctx_size: int) -> str:
        digest = self._perf_model_digest(model_path)
        return f"{digest}::ctx={int(ctx_size)}"

    def _perf_named_preset_key(self, model_path: str, preset_name: str) -> str:
        digest = self._perf_model_digest(model_path)
        name_digest = hashlib.sha1(
            preset_name.casefold().encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return f"{digest}::name={name_digest}"

    def list_perf_preset_names(self, model_path: str) -> List[str]:
        if not model_path:
            return [_PERF_DEFAULT_PRESET_NAME]

        root = self.profiles.get(_PERF_PRESETS_ROOT, {})
        if not isinstance(root, dict):
            return [_PERF_DEFAULT_PRESET_NAME]

        digest = self._perf_model_digest(model_path)
        names = {_PERF_DEFAULT_PRESET_NAME}
        for key, preset_obj in root.items():
            if not str(key).startswith(f"{digest}::"):
                continue
            if not isinstance(preset_obj, dict):
                continue
            preset_name = _normalize_perf_preset_name(preset_obj.get("preset_name"))
            if "::name=" in str(key):
                names.add(preset_name)

        return [_PERF_DEFAULT_PRESET_NAME] + sorted(
            name for name in names if name != _PERF_DEFAULT_PRESET_NAME
        )

    def save_perf_preset(
        self,
        model_path: str,
        ctx_size: int,
        ui: Any,
        metadata: Optional[Dict[str, Any]] = None,
        autotune_params: Optional[Dict[str, Any]] = None,
        preset_name: Optional[str] = None,
    ) -> None:
        """
        Сохраняет параметры производительности для пары:
        конкретная GGUF-модель + конкретный context size.
        """
        if not model_path:
            raise ValueError("Model not selected")

        if ctx_size <= 0:
            raise ValueError("Preset requires specific Context Size, not auto")

        self.read_from_ui(ui)

        params = _perf_params_from_settings(self.settings)
        params = _apply_autotune_params_to_perf_params(params, autotune_params)
        params = _normalize_perf_param_types(params)

        params["ctx_size"] = int(ctx_size)

        root = self.profiles.setdefault(_PERF_PRESETS_ROOT, {})
        if not isinstance(root, dict):
            root = {}
            self.profiles[_PERF_PRESETS_ROOT] = root

        preset_name = _normalize_perf_preset_name(preset_name)
        if preset_name == _PERF_DEFAULT_PRESET_NAME:
            key = self._perf_preset_key(model_path, ctx_size)
        else:
            key = self._perf_named_preset_key(model_path, preset_name)

        preset = {
            "preset_name": preset_name,
            "model_path": str(model_path),
            "model_name": Path(model_path).name,
            "ctx_size": int(ctx_size),
            "params": params,
        }
        if metadata:
            preset["benchmark"] = metadata

        root[key] = preset

        self.save_profiles()

    def load_perf_preset(
        self,
        model_path: str,
        ctx_size: int,
        ui: Any,
        preset_name: Optional[str] = None,
    ) -> bool:
        """
        Загружает только параметры производительности.
        Не трогает глобальные настройки: exe, bench, model_dir, port, интеграции.
        """
        if not model_path:
            return False

        preset_name = _normalize_perf_preset_name(preset_name)
        if preset_name == _PERF_DEFAULT_PRESET_NAME and ctx_size <= 0:
            return False

        root = self.profiles.get(_PERF_PRESETS_ROOT, {})
        if not isinstance(root, dict):
            return False

        if preset_name == _PERF_DEFAULT_PRESET_NAME:
            key = self._perf_preset_key(model_path, ctx_size)
            preset_obj = root.get(key)
        else:
            key = self._perf_named_preset_key(model_path, preset_name)
            preset_obj = root.get(key)

        # Backward compatibility со старым форматом perf_<model_name>_<ctx>
        if preset_name == _PERF_DEFAULT_PRESET_NAME and not preset_obj:
            legacy_key = f"perf_{Path(model_path).name}_{ctx_size}"
            legacy_preset = self.profiles.get(legacy_key)
            if isinstance(legacy_preset, dict):
                preset_obj = {"params": legacy_preset}

        if not isinstance(preset_obj, dict):
            return False

        params = preset_obj.get("params", {})
        if not isinstance(params, dict):
            return False

        params = dict(params)
        if preset_name == _PERF_DEFAULT_PRESET_NAME:
            params["ctx_size"] = int(ctx_size)
        else:
            params["ctx_size"] = int(
                params.get("ctx_size", preset_obj.get("ctx_size", ctx_size))
            )
        params = _normalize_perf_param_types(params)

        for field_name, value in params.items():
            if not hasattr(self.settings, field_name):
                continue

            try:
                if field_name == "enable_thinking":
                    value = _normalize_enable_thinking(value)
                setattr(self.settings, field_name, value)
            except (TypeError, ValueError):
                continue

            widget_attr = _FIELD_WIDGET_MAP.get(field_name)
            if not widget_attr:
                continue

            widget = getattr(ui, widget_attr, None)
            if widget is None:
                continue

            try:
                _widget_set(widget, value)
            except (TypeError, ValueError):
                pass

        return True
