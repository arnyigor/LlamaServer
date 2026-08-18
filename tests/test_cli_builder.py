"""Тесты для src/core/cli_builder.py"""

import sys
import os
import unittest
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cli_builder import (
    build_args,
    build_benchmark_args_from_params,
    validate_extra_args,
)


@dataclass
class MockConfig:
    exe: str = "llama-server.exe"
    bench: str = "llama-bench.exe"
    model_dir: str = "/models"
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
    ctx_checkpoints: int = -1
    cache_ram: int = -2
    cont_batching: bool = True
    cache_prompt: bool = True
    context_shift: bool = False
    no_webui: bool = False
    extra_args: str = ""
    jinja: bool = False
    use_chat_template: bool = False
    chat_template_file: str = ""
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
    use_mmproj: bool = True
    mmproj_offload: bool = True
    enable_thinking: str = "off"
    speculative_mtp: bool = False
    spec_draft_model_path: str = ""
    spec_draft_n_max: int = 3
    spec_draft_p_min: float = 0.0
    spec_draft_gpu_layers: str = "all"
    mmproj_path: str = ""
    bench_prompt: int = 128
    bench_gen: int = 256


class TestBuildArgs(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        self.model = "/models/test.gguf"

    def test_basic_server_args(self):
        self.cfg.ctx_size = 4096
        args = build_args(self.cfg, self.model)
        self.assertIn("-m", args)
        self.assertIn(self.model, args)
        self.assertIn("--port", args)
        self.assertIn("8080", args)
        self.assertIn("-c", args)
        self.assertIn("4096", args)

    def test_basic_benchmark_args(self):
        args = build_args(self.cfg, self.model, for_benchmark=True)
        self.assertIn("-p", args)
        self.assertIn("128", args)
        self.assertIn("-n", args)
        self.assertIn("256", args)
        self.assertNotIn("--port", args)

    def test_gpu_auto_vs_manual(self):
        self.cfg.gpu_auto = True
        args_auto = build_args(self.cfg, self.model)
        self.assertIn("auto", args_auto)

        self.cfg.gpu_auto = False
        self.cfg.gpu_layers = 45
        args_manual = build_args(self.cfg, self.model)
        self.assertIn("45", args_manual)
        self.assertNotIn("auto", args_manual)

    def test_bench_gpu_auto_forces_99(self):
        """При gpu_auto=True llama-bench получает -ngl 99 (full offload)."""
        self.cfg.gpu_auto = True
        args = build_args(self.cfg, self.model, for_benchmark=True)
        idx = args.index("-ngl")
        self.assertEqual(args[idx + 1], "99")

    def test_bench_gpu_manual_preserves_value(self):
        """При gpu_auto=False llama-bench получает точное значение gpu_layers."""
        self.cfg.gpu_auto = False
        self.cfg.gpu_layers = 10
        args = build_args(self.cfg, self.model, for_benchmark=True)
        idx = args.index("-ngl")
        self.assertEqual(args[idx + 1], "10")

    def test_bench_flash_attn_uses_on_off(self):
        """Современный llama-bench принимает -fa on/off, а не 1/0."""
        self.cfg.flash_attn = True
        args = build_args(self.cfg, self.model, for_benchmark=True)
        self.assertEqual(args[args.index("-fa") + 1], "on")

        self.cfg.flash_attn = False
        args = build_args(self.cfg, self.model, for_benchmark=True)
        self.assertEqual(args[args.index("-fa") + 1], "off")

        args = build_benchmark_args_from_params(self.model, {"flash_attn": True})
        self.assertEqual(args[args.index("-fa") + 1], "on")

    def test_mmproj_handling(self):
        self.cfg.mmproj_path = "/models/mmproj.gguf"
        self.cfg.use_mmproj = True
        self.cfg.mmproj_offload = True
        args = build_args(self.cfg, self.model)
        self.assertIn("-mm", args)
        self.assertIn("/models/mmproj.gguf", args)
        self.assertNotIn("--no-mmproj-offload", args)

        self.cfg.mmproj_offload = False
        args = build_args(self.cfg, self.model)
        self.assertIn("--no-mmproj-offload", args)

        self.cfg.use_mmproj = False
        args = build_args(self.cfg, self.model)
        self.assertIn("--no-mmproj", args)
        self.assertNotIn("-mm", args)

    def test_reasoning_mode(self):
        self.cfg.reasoning_mode = "auto"
        args = build_args(self.cfg, self.model)
        self.assertNotIn("-rea", args)

        self.cfg.reasoning_mode = "on"
        args = build_args(self.cfg, self.model)
        self.assertIn("--reasoning", args)
        self.assertIn("on", args)
        self.assertNotIn("--chat-template-kwargs", args)

    def test_reasoning_controls_emitted_conditionally(self):
        # По умолчанию reasoning-контролы не добавляют флагов.
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--reasoning-effort", args)
        self.assertNotIn("--reasoning-preserve", args)
        self.assertNotIn("--no-reasoning-preserve", args)
        self.assertNotIn("--reasoning-budget", args)
        self.assertNotIn("--reasoning-budget-message", args)

        # Заданные значения эмитятся, только когда reasoning активен
        # (on/auto). Фикстура по умолчанию reasoning_mode="off".
        self.cfg.reasoning_mode = "auto"
        self.cfg.reasoning_effort = "low"
        self.cfg.reasoning_preserve = "preserve"
        self.cfg.reasoning_budget = 512
        self.cfg.reasoning_budget_message = "keep thinking"
        args = build_args(self.cfg, self.model)
        self.assertEqual(args[args.index("--reasoning-effort") + 1], "low")
        self.assertIn("--reasoning-preserve", args)
        self.assertNotIn("--no-reasoning-preserve", args)
        self.assertEqual(args[args.index("--reasoning-budget") + 1], "512")
        self.assertEqual(
            args[args.index("--reasoning-budget-message") + 1], "keep thinking"
        )

        # Вариант no-preserve.
        self.cfg.reasoning_preserve = "no-preserve"
        args = build_args(self.cfg, self.model)
        self.assertIn("--no-reasoning-preserve", args)
        self.assertNotIn("--reasoning-preserve", args)

        # Пустые значения/0 подавляют флаги.
        self.cfg.reasoning_effort = ""
        self.cfg.reasoning_budget = 0
        self.cfg.reasoning_budget_message = ""
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--reasoning-effort", args)
        self.assertNotIn("--reasoning-budget", args)
        self.assertNotIn("--reasoning-budget-message", args)

    def test_reasoning_controls_suppressed_when_reasoning_off(self):
        # При явном --reasoning off суб-параметры не эмитятся, даже если заданы.
        self.cfg.reasoning_mode = "off"
        self.cfg.reasoning_effort = "xhigh"
        self.cfg.reasoning_preserve = "preserve"
        self.cfg.reasoning_budget = 256
        self.cfg.reasoning_budget_message = "budget hit"
        args = build_args(self.cfg, self.model)
        self.assertIn("--reasoning", args)
        self.assertIn("off", args)
        self.assertNotIn("--reasoning-effort", args)
        self.assertNotIn("--reasoning-preserve", args)
        self.assertNotIn("--no-reasoning-preserve", args)
        self.assertNotIn("--reasoning-budget", args)
        self.assertNotIn("--reasoning-budget-message", args)

        # То же самое при enable_thinking=false (принудительно off).
        self.cfg.reasoning_mode = "auto"
        self.cfg.enable_thinking = "false"
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--reasoning-effort", args)
        self.assertNotIn("--reasoning-budget", args)
        self.assertNotIn("--reasoning-budget-message", args)

    def test_enable_thinking_modes(self):
        self.cfg.enable_thinking = "off"
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-kwargs", args)

        self.cfg.enable_thinking = "false"
        args = build_args(self.cfg, self.model)
        idx = args.index("--reasoning")
        self.assertEqual(args[idx + 1], "off")
        self.assertNotIn("--chat-template-kwargs", args)

        self.cfg.reasoning_mode = "auto"
        self.cfg.enable_thinking = "true"
        args = build_args(self.cfg, self.model)
        idx = args.index("--reasoning")
        self.assertEqual(args[idx + 1], "on")
        self.assertNotIn("--chat-template-kwargs", args)

    def test_enable_thinking_legacy_bool(self):
        self.cfg.reasoning_mode = "auto"
        self.cfg.enable_thinking = True
        args = build_args(self.cfg, self.model)
        idx = args.index("--reasoning")
        self.assertEqual(args[idx + 1], "on")

        self.cfg.enable_thinking = False
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-kwargs", args)

    def test_mtp_cuda_args(self):
        self.cfg.gpu_auto = False
        self.cfg.gpu_layers_all = True
        self.cfg.ctx_size = 65536
        self.cfg.cuda_device = "CUDA0"
        self.cfg.spec_draft_device = "CUDA0"
        self.cfg.split_mode = "none"
        self.cfg.main_gpu = 0
        self.cfg.cache_type_k = "q8_0"
        self.cfg.cache_type_v = "q8_0"
        self.cfg.speculative_mtp = True
        self.cfg.spec_draft_model_path = "/models/test-mtp-draft.gguf"
        self.cfg.spec_draft_n_max = 2
        self.cfg.spec_draft_p_min = 0.8
        self.cfg.spec_draft_gpu_layers = "all"
        self.cfg.use_mmproj = False
        self.cfg.jinja = True

        args = build_args(self.cfg, self.model)

        self.assertEqual(args[args.index("-ngl") + 1], "all")
        self.assertIn("--device", args)
        self.assertIn("CUDA0", args)
        self.assertEqual(args[args.index("--spec-draft-ngl") + 1], "all")
        self.assertNotIn("--spec-draft-type-k", args)
        self.assertNotIn("--spec-draft-type-v", args)
        self.assertEqual(args[args.index("--spec-draft-device") + 1], "CUDA0")
        self.assertEqual(
            args[args.index("--model-draft") + 1], "/models/test-mtp-draft.gguf"
        )
        self.assertEqual(args[args.index("--spec-draft-p-min") + 1], "0.8")
        self.assertIn("--split-mode", args)
        self.assertIn("none", args)
        self.assertIn("--main-gpu", args)
        self.assertIn("--jinja", args)
        self.assertIn("--no-mmproj", args)

    def test_mtp_optional_draft_flags_omitted_when_empty(self):
        self.cfg.speculative_mtp = True
        self.cfg.spec_draft_gpu_layers = ""
        self.cfg.spec_draft_device = ""

        args = build_args(self.cfg, self.model)

        self.assertIn("--spec-type", args)
        self.assertNotIn("--spec-draft-ngl", args)
        self.assertNotIn("--spec-draft-device", args)

    def test_flash_attn_and_fit_off(self):
        self.cfg.flash_attn = False
        self.cfg.fit_off = False
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--flash-attn", args)
        self.assertNotIn("--fit", args)

    def test_extra_args_valid(self):
        self.cfg.extra_args = "--top-p 0.9 --seed 42"
        args = build_args(self.cfg, self.model)
        self.assertIn("--top-p", args)
        self.assertIn("0.9", args)
        self.assertIn("--seed", args)
        self.assertIn("42", args)

    def test_sampling_args(self):
        self.cfg.temperature = 0.7
        self.cfg.top_k = 20
        self.cfg.top_p = 0.9
        self.cfg.min_p = 0.05
        self.cfg.typical_p = 0.95
        self.cfg.repeat_penalty = 1.1
        self.cfg.repeat_last_n = 128
        self.cfg.presence_penalty = 0.2
        self.cfg.frequency_penalty = 0.1
        self.cfg.seed = 42

        args = build_args(self.cfg, self.model)

        expected = {
            "--temp": "0.7",
            "--top-k": "20",
            "--top-p": "0.9",
            "--min-p": "0.05",
            "--typical": "0.95",
            "--repeat-penalty": "1.1",
            "--repeat-last-n": "128",
            "--presence-penalty": "0.2",
            "--frequency-penalty": "0.1",
            "--seed": "42",
        }
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_sampling_auto_values_are_omitted(self):
        args = build_args(self.cfg, self.model)
        for flag in (
            "--top-k",
            "--top-p",
            "--min-p",
            "--typical",
            "--repeat-last-n",
            "--presence-penalty",
            "--frequency-penalty",
            "--seed",
        ):
            self.assertNotIn(flag, args)

    def test_extra_args_managed_duplicates_are_filtered(self):
        self.cfg.ctx_size = 131072
        self.cfg.ctx_checkpoints = 0
        self.cfg.cache_ram = 0
        self.cfg.jinja = True
        self.cfg.extra_args = "--ctx-checkpoints 0 --cache-ram=0 --jinja --top-p 0.9"

        args = build_args(self.cfg, self.model)

        self.assertEqual(args.count("--ctx-checkpoints"), 1)
        self.assertEqual(args.count("--cache-ram"), 1)
        self.assertEqual(args.count("--jinja"), 1)
        self.assertIn("--top-p", args)
        self.assertIn("0.9", args)

    def test_extra_args_managed_negative_value_is_filtered(self):
        self.cfg.speculative_mtp = True
        self.cfg.extra_args = "--spec-draft-ngl -1 --top-p 0.9"

        args = build_args(self.cfg, self.model)

        self.assertNotIn("-1", args)
        self.assertIn("--top-p", args)
        self.assertIn("0.9", args)

    def test_managed_p_min_overrides_duplicate_extra_and_unmanaged_n_min_is_preserved(self):
        self.cfg.speculative_mtp = True
        self.cfg.spec_draft_p_min = 0.8
        self.cfg.extra_args = "--spec-draft-n-min 1 --spec-draft-p-min 0.5"

        args = build_args(self.cfg, self.model)

        self.assertIn("--spec-draft-n-min", args)
        self.assertIn("1", args)
        self.assertIn("--spec-draft-p-min", args)
        self.assertIn("0.8", args)
        self.assertNotIn("0.5", args)

    def test_extra_args_unmanaged_jinja_still_allowed(self):
        self.cfg.jinja = False
        self.cfg.extra_args = "--jinja"

        args = build_args(self.cfg, self.model)

        self.assertEqual(args.count("--jinja"), 1)

    def test_empty_model_returns_none(self):
        self.assertIsNone(build_args(self.cfg, ""))
        self.assertIsNone(build_args(self.cfg, None))

    def test_server_args_include_metrics(self):
        """llama-server всегда получает --metrics (нужен GUI-счётчикам)."""
        args = build_args(self.cfg, self.model)
        self.assertIn("--metrics", args)

    def test_benchmark_args_exclude_metrics(self):
        """llama-bench не получает --metrics."""
        args = build_args(self.cfg, self.model, for_benchmark=True)
        self.assertNotIn("--metrics", args)

    def test_extra_metrics_duplicate_is_filtered(self):
        """--metrics из Extra args не дублируется."""
        self.cfg.extra_args = "--metrics"
        args = build_args(self.cfg, self.model)
        self.assertEqual(args.count("--metrics"), 1)

    def test_chat_template_enabled_adds_flag(self):
        self.cfg.use_chat_template = True
        self.cfg.chat_template_file = "G:/AIModels/lmstudio/qwen3_claude_relaxed.jinja"
        args = build_args(self.cfg, self.model)
        idx = args.index("--chat-template-file")
        self.assertEqual(
            args[idx + 1], "G:/AIModels/lmstudio/qwen3_claude_relaxed.jinja"
        )

    def test_chat_template_disabled_omits_flag(self):
        self.cfg.use_chat_template = False
        self.cfg.chat_template_file = "G:/AIModels/lmstudio/qwen3_claude_relaxed.jinja"
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-file", args)

    def test_chat_template_empty_path_omits_flag(self):
        self.cfg.use_chat_template = True
        self.cfg.chat_template_file = ""
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-file", args)

    def test_chat_template_omitted_for_benchmark(self):
        self.cfg.use_chat_template = True
        self.cfg.chat_template_file = "G:/AIModels/lmstudio/qwen3_claude_relaxed.jinja"
        args = build_args(self.cfg, self.model, for_benchmark=True)
        self.assertNotIn("--chat-template-file", args)

    def test_chat_template_duplicate_from_extra_args_is_filtered(self):
        """Флаг из Extra args не дублируется, когда управляется UI."""
        self.cfg.use_chat_template = True
        self.cfg.chat_template_file = "G:/templates/custom.jinja"
        self.cfg.extra_args = "--chat-template-file /models/custom.jinja --top-p 0.9"
        args = build_args(self.cfg, self.model)
        self.assertEqual(args.count("--chat-template-file"), 1)
        self.assertIn("--top-p", args)
        self.assertIn("0.9", args)


class TestValidateExtraArgs(unittest.TestCase):
    def test_unknown_flag_allowed(self):
        errs = validate_extra_args(["--unknown-flag", "val"], "/models")
        self.assertEqual(len(errs), 0)

    def test_valid_path_flag(self):
        with patch("src.core.cli_builder.validate_path") as mock_vp:
            mock_vp.return_value = Path("/models/grammar.json")
            errs = validate_extra_args(["--grammar-file", "grammar.json"], "/models")
            self.assertEqual(len(errs), 0)
            mock_vp.assert_called_once()

    def test_path_traversal_blocked(self):
        with patch("src.core.cli_builder.validate_path") as mock_vp:
            mock_vp.side_effect = ValueError("Path must be inside...")
            errs = validate_extra_args(["--lora", "../../../evil.bin"], "/models")
            self.assertTrue(any("недопустимый путь" in e for e in errs))

    def test_forbidden_host(self):
        errs = validate_extra_args(["--host", "0.0.0.0"], "/models")
        self.assertTrue(any("запрещено" in e for e in errs))

        errs = validate_extra_args(["--host=::"], "/models")
        self.assertTrue(any("запрещено" in e for e in errs))

    def test_valid_host(self):
        errs = validate_extra_args(["--host", "127.0.0.1"], "/models")
        self.assertEqual(len(errs), 0)


if __name__ == "__main__":
    unittest.main()
