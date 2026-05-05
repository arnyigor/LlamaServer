"""Менеджер конфигурации и профилей."""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from pathlib import Path

from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit

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
    last_model_path: str = ""
    model_cache: list = field(default_factory=list)
    temperature: float = 0.7
    repeat_penalty: float = 1.1
    gpu_auto: bool = True
    gpu_layers: int = 33
    cpu_moe_layers: int = 0
    ctx_size: int = 4096
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
    batch_size: int = 2048
    ubatch_size: int = 2048
    parallel_slots: int = 1
    ctx_checkpoints: int = -1
    cache_ram: int = -2
    cont_batching: bool = True
    cache_prompt: bool = True
    context_shift: bool = False
    no_webui: bool = False
    extra_args: str = ""

class ConfigManager:
    def __init__(self, settings_path: str = "settings.json", profiles_path: str = "profiles.json"):
        self.settings_path = settings_path
        self.profiles_path = profiles_path
        self.settings = AppSettings()
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.settings, k):
                        setattr(self.settings, k, v)
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)

    def save(self) -> None:
        write_json_file_safely(self.settings_path, asdict(self.settings))

    def save_profiles(self) -> None:
        write_json_file_safely(self.profiles_path, self.profiles)

    def apply_to_ui(self, ui: Any) -> None:
        """Автоматический маппинг настроек на виджеты UI."""
        s = self.settings
        ui.exe_path.setText(s.exe)
        ui.bench_path.setText(s.bench)
        ui.model_dir.setText(s.model_dir)
        ui.opencode_config_path.setText(s.opencode_config)
        ui.pi_config_path.setText(s.pi_config)
        ui.bench_prompt.setValue(s.bench_prompt)
        ui.bench_gen.setValue(s.bench_gen)
        ui.auto_params.setChecked(s.auto_params)
        ui.use_mmproj.setChecked(s.use_mmproj)
        ui.mmproj_offload.setChecked(s.mmproj_offload)
        ui.gpu_auto.setChecked(s.gpu_auto)
        ui.gpu_layers.setValue(s.gpu_layers)
        ui.cpu_moe_layers.setValue(s.cpu_moe_layers)
        ui.ctx_size.setValue(s.ctx_size)
        ui.threads.setValue(s.threads)
        ui.threads_batch.setValue(s.threads_batch)
        ui.port.setValue(s.port)
        ui.temperature.setValue(s.temperature)
        ui.repeat_penalty.setValue(s.repeat_penalty)
        ui.flash_attn.setChecked(s.flash_attn)
        ui.fit_off.setChecked(s.fit_off)
        ui.reasoning_mode.setCurrentText(s.reasoning_mode)
        ui.use_mmap.setChecked(s.use_mmap)
        ui.use_mlock.setChecked(s.use_mlock)
        ui.verbose.setChecked(s.verbose)
        ui.log_timestamps.setChecked(s.log_timestamps)
        ui.cache_type_k.setCurrentText(s.cache_type_k)
        ui.cache_type_v.setCurrentText(s.cache_type_v)
        ui.batch_size.setValue(s.batch_size)
        ui.ubatch_size.setValue(s.ubatch_size)
        ui.parallel_slots.setValue(s.parallel_slots)
        ui.ctx_checkpoints.setValue(s.ctx_checkpoints)
        ui.cache_ram.setValue(s.cache_ram)
        ui.cont_batching.setChecked(s.cont_batching)
        ui.cache_prompt.setChecked(s.cache_prompt)
        ui.context_shift.setChecked(s.context_shift)
        ui.no_webui.setChecked(s.no_webui)
        ui.extra_args.setText(s.extra_args)

        idx = ui.integration_target.findData(s.integration_target)
        if idx >= 0:
            ui.integration_target.setCurrentIndex(idx)

    def read_from_ui(self, ui: Any) -> None:
        """Считывает текущие значения виджетов в dataclass."""
        s = self.settings
        s.exe = ui.exe_path.text()
        s.bench = ui.bench_path.text()
        s.model_dir = ui.model_dir.text()
        s.opencode_config = ui.opencode_config_path.text()
        s.pi_config = ui.pi_config_path.text()
        s.integration_target = ui.current_config_target()
        s.bench_prompt = ui.bench_prompt.value()
        s.bench_gen = ui.bench_gen.value()
        s.auto_params = ui.auto_params.isChecked()
        s.use_mmproj = ui.use_mmproj.isChecked()
        s.mmproj_offload = ui.mmproj_offload.isChecked()
        s.last_model_path = ui.model_combo.currentData() or s.last_model_path
        s.temperature = ui.temperature.value()
        s.repeat_penalty = ui.repeat_penalty.value()
        s.gpu_auto = ui.gpu_auto.isChecked()
        s.gpu_layers = ui.gpu_layers.value()
        s.cpu_moe_layers = ui.cpu_moe_layers.value()
        s.ctx_size = ui.ctx_size.value()
        s.threads = ui.threads.value()
        s.threads_batch = ui.threads_batch.value()
        s.port = ui.port.value()
        s.flash_attn = ui.flash_attn.isChecked()
        s.fit_off = ui.fit_off.isChecked()
        s.reasoning_mode = ui.reasoning_mode.currentText()
        s.use_mmap = ui.use_mmap.isChecked()
        s.use_mlock = ui.use_mlock.isChecked()
        s.verbose = ui.verbose.isChecked()
        s.log_timestamps = ui.log_timestamps.isChecked()
        s.cache_type_k = ui.cache_type_k.currentText()
        s.cache_type_v = ui.cache_type_v.currentText()
        s.batch_size = ui.batch_size.value()
        s.ubatch_size = ui.ubatch_size.value()
        s.parallel_slots = ui.parallel_slots.value()
        s.ctx_checkpoints = ui.ctx_checkpoints.value()
        s.cache_ram = ui.cache_ram.value()
        s.cont_batching = ui.cont_batching.isChecked()
        s.cache_prompt = ui.cache_prompt.isChecked()
        s.context_shift = ui.context_shift.isChecked()
        s.no_webui = ui.no_webui.isChecked()
        s.extra_args = ui.extra_args.text()