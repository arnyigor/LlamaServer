# tests/test_param_registry.py
"""Паритет-тесты реестра параметров против дословных старых литералов.

Литералы ниже скопированы из config.py/cli_parser.py/cli_builder.py до
миграции на PARAM_REGISTRY. Пока тест зелёный — реестр покрывает 100%
поведения старых ручных таблиц. При добавлении нового флага llama.cpp
обновлять PARAM_REGISTRY и этот файл одновременно.
"""

import pytest

from src.core.param_registry import (
    FIELD_WIDGET_MAP,
    FILTER_BOOL_FLAGS,
    FILTER_VALUE_FLAGS,
    FLAG_TO_SPEC,
    MANAGED_EXTRA_FLAGS,
    OPTIONAL_VALUE_FLAGS,
    PARAM_REGISTRY,
    SAMPLING_EXTRA_FIELDS,
    VALUE_FLAGS,
    NEG_FLAG_TO_SPEC,
)

# --- Старый _FIELD_WIDGET_MAP из config.py -----------------------------------

_OLD_FIELD_WIDGET_MAP = {
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
    "reasoning_effort": "reasoning_effort",
    "reasoning_preserve": "reasoning_preserve",
    "reasoning_budget": "reasoning_budget",
    "reasoning_budget_message": "reasoning_budget_message",
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

# --- Старый _MANAGED_EXTRA_FLAGS из config.py --------------------------------

_OLD_MANAGED_EXTRA_FLAGS = {
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
    "--flash-attn",
    "-fa",
    "--fit",
    "-rea",
    "--reasoning",
    "--reasoning-effort",
    "--reasoning-preserve",
    "--no-reasoning-preserve",
    "--reasoning-budget",
    "--reasoning-budget-message",
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
    "--mlock",
    "--verbose",
    "--log-timestamps",
    "--context-shift",
    "--no-webui",
}

# --- Старый _SAMPLING_EXTRA_FIELDS из config.py ------------------------------

_OLD_SAMPLING_EXTRA_FIELDS = {
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

# --- Старый _VALUE_FLAGS из cli_parser.py ------------------------------------

_OLD_VALUE_FLAGS = {
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
    "--reasoning-effort",
    "--reasoning-preserve",
    "--no-reasoning-preserve",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--ctx-checkpoints",
    "--cache-ram",
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
    "-mm",
    "--mmproj",
    "-md",
    "--model-draft",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-p-min",
    "--spec-draft-ngl",
    "-ngld",
    "--chat-template-file",
}

# --- Старые наборы _filter_duplicate_extra_args из cli_builder.py ------------

_OLD_FILTER_VALUE_FLAGS = {
    "-m",
    "--model",
    "--port",
    "--host",
    "--device",
    "--spec-draft-device",
    "-md",
    "--model-draft",
    "--chat-template-kwargs",
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
    "--reasoning-effort",
    "--reasoning-preserve",
    "--no-reasoning-preserve",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--ctx-checkpoints",
    "--cache-ram",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-n-min",
    "--spec-draft-p-min",
    "--spec-draft-ngl",
    "-ngld",
    "--spec-draft-type-k",
    "-ctkd",
    "--spec-draft-type-v",
    "-ctvd",
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
    "--flash-attn",
    "-fa",
    "-mm",
    "--mmproj",
    "--chat-template-file",
}

_OLD_FILTER_BOOL_FLAGS = {
    "--no-mmproj",
    "--no-mmproj-offload",
    "--mlock",
    "--verbose",
    "--log-timestamps",
    "--context-shift",
    "--no-webui",
    "--jinja",
    "--kv-unified",
    "-kvu",
    "--metrics",
}


class TestParamRegistryParity:
    def test_field_widget_map_matches_old(self):
        assert FIELD_WIDGET_MAP == _OLD_FIELD_WIDGET_MAP

    def test_managed_extra_flags_match_old(self):
        # Реестр обязан покрывать все старые флаги...
        assert _OLD_MANAGED_EXTRA_FLAGS <= set(MANAGED_EXTRA_FLAGS)
        # ...плюс два осознанных исправления рассинхрона старого кода:
        # -ngld — точный алиас --spec-draft-ngl (раньше не вырезался и
        # дублировал флаг), --metrics всегда добавляет builder.
        assert set(MANAGED_EXTRA_FLAGS) - _OLD_MANAGED_EXTRA_FLAGS == {
            "-ngld",
            "--metrics",
            "--reasoning-effort",
            "--reasoning-preserve",
            "--no-reasoning-preserve",
            "--reasoning-budget",
            "--reasoning-budget-message",
            # Спеки use_mmap/cont_batching/cache_prompt нейтрализованы
            # (managed=False, без cli_flags), поэтому их флаги больше не
            # управляются реестром и не вырезаются из extra_params.
            "--mmap",
            "--no-mmap",
            "--no-cont-batching",
            "--no-cache-prompt",
        }

    def test_sampling_extra_fields_match_old(self):
        assert SAMPLING_EXTRA_FIELDS == _OLD_SAMPLING_EXTRA_FIELDS

    def test_value_flags_match_old(self):
        assert set(VALUE_FLAGS) == _OLD_VALUE_FLAGS
        assert set(OPTIONAL_VALUE_FLAGS) == {"-fa", "--flash-attn"}

    def test_filter_flags_match_old(self):
        assert set(FILTER_VALUE_FLAGS) == _OLD_FILTER_VALUE_FLAGS
        assert set(FILTER_BOOL_FLAGS) == _OLD_FILTER_BOOL_FLAGS


class TestParamRegistryCoherence:
    def test_names_are_unique(self):
        names = [spec.name for spec in PARAM_REGISTRY]
        assert len(names) == len(set(names))

    def test_flags_are_unique(self):
        all_flags = [f for spec in PARAM_REGISTRY for f in spec.cli_flags]
        assert len(all_flags) == len(set(all_flags))
        neg_flags = [f for spec in PARAM_REGISTRY for f in spec.cli_neg_flags]
        assert len(neg_flags) == len(set(neg_flags))
        assert not set(all_flags) & set(neg_flags)

    def test_value_specs_have_field_type(self):
        for spec in PARAM_REGISTRY:
            if spec.cli_kind in ("value", "bool_value"):
                assert spec.field_type is not None, spec.name

    def test_flag_to_spec_covers_all_flags(self):
        for spec in PARAM_REGISTRY:
            for flag in spec.cli_flags:
                assert FLAG_TO_SPEC[flag] is spec
        for spec in PARAM_REGISTRY:
            for flag in spec.cli_neg_flags:
                assert NEG_FLAG_TO_SPEC[flag] is spec

    def test_widget_specs_cover_settings_fields(self):
        # Все поля, у которых был виджет до миграции, описаны в реестре.
        assert set(FIELD_WIDGET_MAP) == set(_OLD_FIELD_WIDGET_MAP)
