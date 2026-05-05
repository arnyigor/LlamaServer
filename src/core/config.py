# src/core/config.py
"""Менеджер конфигурации с автоматическим маппингом виджетов."""

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, Optional, Type
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
    repeat_penalty: float = -1.0
    gpu_auto: bool = True
    gpu_layers: int = 33
    cpu_moe_layers: int = -1
    ctx_size: int = -1
    threads: int = 4
    threads_batch: int = 0
    port: int = 8080
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
    ctx_checkpoints: int = -1
    cache_ram: int = -2
    cont_batching: bool = True
    cache_prompt: bool = True
    context_shift: bool = False
    no_webui: bool = False
    jinja: bool = False
    extra_args: str = ""


# Явная таблица маппинга: поле -> атрибут виджета в UI
# Формат: "settings_field": "ui_widget_attr"
_FIELD_WIDGET_MAP: Dict[str, str] = {
    "exe": "exe_path",
    "bench": "bench_path",
    "model_dir": "model_dir",
    "opencode_config": "opencode_config_path",
    "pi_config": "pi_config_path",
    "bench_prompt": "bench_prompt",
    "bench_gen": "bench_gen",
    "auto_params": "auto_params",
    "use_mmproj": "use_mmproj",
    "mmproj_offload": "mmproj_offload",
    "gpu_auto": "gpu_auto",
    "gpu_layers": "gpu_layers",
    "cpu_moe_layers": "cpu_moe_layers",
    "ctx_size": "ctx_size",
    "threads": "threads",
    "threads_batch": "threads_batch",
    "port": "port",
    "temperature": "temperature",
    "repeat_penalty": "repeat_penalty",
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
    "ctx_checkpoints": "ctx_checkpoints",
    "cache_ram": "cache_ram",
    "cont_batching": "cont_batching",
    "cache_prompt": "cache_prompt",
    "context_shift": "context_shift",
    "no_webui": "no_webui",
    "jinja": "jinja",
    "extra_args": "extra_args",
}


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
        widget.setChecked(bool(value))
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
        else:
            widget.setCurrentText(str(value))
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
                            setattr(self.settings, k, v)
                        except (TypeError, ValueError):
                            pass  # Игнорируем невалидные значения
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
                # Приводим к типу поля
                field_type = type(getattr(s, field_name))
                if field_type is bool:
                    setattr(s, field_name, bool(value))
                elif field_type is int:
                    setattr(s, field_name, int(value))
                elif field_type is float:
                    setattr(s, field_name, float(value))
                else:
                    setattr(s, field_name, value)
            except (TypeError, ValueError, AttributeError):
                pass

        # Специальные случаи
        s.integration_target = ui.current_config_target()
        s.last_model_path = ui.model_combo.currentData() or s.last_model_path

    def save_profile(self, name: str, ui: Any) -> None:
        """Сохранение текущих настроек как профиля."""
        self.read_from_ui(ui)
        profile_data = asdict(self.settings)
        # Исключаем системные поля из профиля
        for key in ("model_cache", "last_model_path", "exe", "bench", "model_dir"):
            profile_data.pop(key, None)
        self.profiles[name] = profile_data
        self.save_profiles()

    def load_profile(self, name: str, ui: Any) -> bool:
        """Загрузка профиля в UI. Возвращает True при успехе."""
        profile = self.profiles.get(name)
        if not profile:
            return False
        valid_fields = {f.name for f in fields(AppSettings)}
        for k, v in profile.items():
            if k in valid_fields:
                try:
                    setattr(self.settings, k, v)
                except (TypeError, ValueError):
                    pass
        self.apply_to_ui(ui)
        return True
