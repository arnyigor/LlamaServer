"""Тесты для src/core/cli_builder.py"""

import sys
import os
import unittest
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cli_builder import build_args, validate_extra_args


@dataclass
class MockConfig:
    exe: str = "llama-server.exe"
    bench: str = "llama-bench.exe"
    model_dir: str = "/models"
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
    extra_args: str = ""
    jinja: bool = False
    temperature: float = -1.0
    repeat_penalty: float = -1.0
    use_mmproj: bool = True
    mmproj_offload: bool = True
    enable_thinking: str = "off"
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

    def test_bench_gpu_forces_99(self):
        self.cfg.gpu_auto = False
        self.cfg.gpu_layers = 10
        args = build_args(self.cfg, self.model, for_benchmark=True)
        idx = args.index("-ngl")
        self.assertEqual(args[idx + 1], "99")

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
        self.assertIn("-rea", args)
        self.assertIn("on", args)

    def test_enable_thinking_modes(self):
        self.cfg.enable_thinking = "off"
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-kwargs", args)

        self.cfg.enable_thinking = "false"
        args = build_args(self.cfg, self.model)
        idx = args.index("--chat-template-kwargs")
        self.assertEqual(args[idx + 1], '{"enable_thinking":false}')

        self.cfg.enable_thinking = "true"
        args = build_args(self.cfg, self.model)
        idx = args.index("--chat-template-kwargs")
        self.assertEqual(args[idx + 1], '{"enable_thinking":true}')

    def test_enable_thinking_legacy_bool(self):
        self.cfg.enable_thinking = True
        args = build_args(self.cfg, self.model)
        idx = args.index("--chat-template-kwargs")
        self.assertEqual(args[idx + 1], '{"enable_thinking":true}')

        self.cfg.enable_thinking = False
        args = build_args(self.cfg, self.model)
        self.assertNotIn("--chat-template-kwargs", args)

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

    def test_empty_model_returns_none(self):
        self.assertIsNone(build_args(self.cfg, ""))
        self.assertIsNone(build_args(self.cfg, None))


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
