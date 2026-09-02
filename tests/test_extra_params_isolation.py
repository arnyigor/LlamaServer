import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Headless stub: config.py imports PySide6.QtWidgets at module top. In this
# test environment Qt is not installed, so we inject a minimal fake module
# providing just the names config.py imports. The classes are never
# instantiated during these pure-logic tests.
if "PySide6" not in sys.modules:
    _pyside = types.ModuleType("PySide6")
    _qt = types.ModuleType("PySide6.QtWidgets")
    for _cls in ("QCheckBox", "QComboBox", "QSpinBox", "QDoubleSpinBox", "QLineEdit"):
        setattr(_qt, _cls, object)
    _pyside.QtWidgets = _qt
    sys.modules["PySide6"] = _pyside
    sys.modules["PySide6.QtWidgets"] = _qt

from src.core.constants import AUTO_SENTINEL, SERVER_DEFAULT_SENTINEL  # noqa: E402
from src.core.config import (  # noqa: E402
    AppSettings,
    migrate_extra_fields_to_extra_args,
    _sanitize_extra_args,
)
from src.core.cli_builder import build_args  # noqa: E402
from src.core.cli_parser import parse_llama_server_command  # noqa: E402


# Все CLI-токены, соответствующие EXTRA-полям (managed=False). Ни один из них
# не должен появляться в выводе build_args и все они должны сохраняться
# verbatim в extra_args (критическое требование: EXTRA не перезаписывают команду).
EXTRA_FLAGS = {
    "--mlock",
    "--verbose",
    "--log-timestamps",
    "--kv-unified",
    "--context-shift",
    "--no-webui",
    "--ctx-checkpoints",
    "--cache-ram",
    "--split-mode",
    "--main-gpu",
    "--device",
    "--chat-template-file",
    "--no-cont-batching",
    "--no-cache-prompt",
    "--no-mmap",
    "--cuda-visible-devices",
    "--cuda-module-loading",
}


def _settings_with_extra_non_default() -> AppSettings:
    s = AppSettings()
    s.ctx_checkpoints = 8
    s.cache_ram = 0
    s.split_mode = "none"
    s.main_gpu = 1
    s.cuda_device = "CUDA0"
    s.use_mlock = True
    s.verbose = True
    s.log_timestamps = True
    s.context_shift = True
    s.no_webui = True
    s.use_chat_template = True
    s.chat_template_file = "t.jinja"
    s.cuda_visible_devices = "0"
    s.cuda_module_loading = "EAGER"
    s.kv_unified = True
    s.cont_batching = False
    s.cache_prompt = False
    s.use_mmap = False
    return s


class TestExtraParamsBuildArgsInertness(unittest.TestCase):
    """🔴 HARD: EXTRA-поля не должны эмититься build_args ни при каких значениях."""

    def test_build_args_emits_no_extra_flags(self):
        s = _settings_with_extra_non_default()
        args = build_args(s, "model.gguf")
        self.assertIsNotNone(args)
        for flag in EXTRA_FLAGS:
            self.assertNotIn(flag, args, f"EXTRA-флаг {flag} просочился в build_args")

    def test_build_args_still_emits_managed_flags(self):
        s = _settings_with_extra_non_default()
        args = build_args(s, "model.gguf")
        # MAIN-параметры по-прежнему управляются builder'ом.
        self.assertIn("-m", args)
        self.assertIn("model.gguf", args)
        self.assertIn("--flash-attn", args)
        self.assertIn("-ngl", args)


class TestExtraParamsParser(unittest.TestCase):
    """Parse-back: EXTRA-флаги уходят в extra_args, а не в settings."""

    def test_extra_flags_go_to_extra_args_not_settings(self):
        parsed = parse_llama_server_command(
            "llama-server.exe -m m.gguf --mlock --ctx-checkpoints 8 "
            "--split-mode none --device CUDA0 --no-mmap --verbose"
        )
        ea = parsed.extra_args
        for flag in (
            "--mlock",
            "--ctx-checkpoints",
            "--split-mode",
            "--device",
            "--no-mmap",
            "--verbose",
        ):
            self.assertIn(flag, ea, f"{flag} должен остаться в extra_args")
        for field in (
            "mlock",
            "ctx_checkpoints",
            "split_mode",
            "cuda_device",
            "use_mmap",
            "verbose",
        ):
            self.assertNotIn(field, parsed.settings)

    def test_managed_flag_still_goes_to_settings(self):
        parsed = parse_llama_server_command(
            "llama-server.exe -m m.gguf --flash-attn on --mlock"
        )
        self.assertTrue(parsed.settings["flash_attn"])
        self.assertIn("--mlock", parsed.extra_args)
        self.assertNotIn("flash_attn", parsed.extra_args)


class TestExtraParamsMigration(unittest.TestCase):
    """Старые значения EXTRA-полей переносятся в extra_args и сбрасываются."""

    def test_migrate_moves_values_and_resets_fields(self):
        s = _settings_with_extra_non_default()
        s.extra_args = ""
        migrate_extra_fields_to_extra_args(s)
        ea = s.extra_args

        self.assertIn("--ctx-checkpoints 8", ea)
        self.assertIn("--cache-ram 0", ea)
        self.assertIn("--split-mode none", ea)
        self.assertIn("--main-gpu 1", ea)
        self.assertIn("--device CUDA0", ea)
        self.assertIn("--mlock", ea)
        self.assertIn("--verbose", ea)
        self.assertIn("--log-timestamps", ea)
        self.assertIn("--context-shift", ea)
        self.assertIn("--no-webui", ea)
        self.assertIn("--chat-template-file t.jinja", ea)
        self.assertIn("--cuda-visible-devices 0", ea)
        self.assertIn("--cuda-module-loading EAGER", ea)
        self.assertIn("--kv-unified", ea)
        self.assertIn("--no-cont-batching", ea)
        self.assertIn("--no-cache-prompt", ea)
        self.assertIn("--no-mmap", ea)

        # Поля сброшены в default.
        self.assertEqual(s.ctx_checkpoints, AUTO_SENTINEL)
        self.assertEqual(s.cache_ram, SERVER_DEFAULT_SENTINEL)
        self.assertEqual(s.split_mode, "")
        self.assertEqual(s.main_gpu, AUTO_SENTINEL)
        self.assertEqual(s.cuda_device, "")
        self.assertFalse(s.use_mlock)
        self.assertFalse(s.verbose)
        self.assertFalse(s.log_timestamps)
        self.assertFalse(s.context_shift)
        self.assertFalse(s.no_webui)
        self.assertFalse(s.use_chat_template)
        self.assertEqual(s.chat_template_file, "")
        self.assertEqual(s.cuda_visible_devices, "")
        self.assertEqual(s.cuda_module_loading, "LAZY")
        self.assertFalse(s.kv_unified)
        self.assertTrue(s.cont_batching)
        self.assertTrue(s.cache_prompt)
        self.assertTrue(s.use_mmap)

    def test_migrate_is_idempotent(self):
        s = AppSettings()
        s.ctx_checkpoints = 8
        s.use_mlock = True
        s.extra_args = ""
        migrate_extra_fields_to_extra_args(s)
        first = s.extra_args
        # Повторный прогон: поля уже сброшены в default → ничего не добавляется.
        migrate_extra_fields_to_extra_args(s)
        self.assertEqual(s.extra_args, first)
        self.assertIn("--ctx-checkpoints 8", s.extra_args)
        self.assertIn("--mlock", s.extra_args)

    def test_migrate_does_not_duplicate_existing_extra_flag(self):
        s = AppSettings()
        s.use_mlock = True
        s.extra_args = "--mlock"
        migrate_extra_fields_to_extra_args(s)
        self.assertEqual(s.extra_args.count("--mlock"), 1)
        self.assertFalse(s.use_mlock)


class TestSanitizeExtraArgsKeepsExtra(unittest.TestCase):
    """_sanitize_extra_args не должен вырезать EXTRA-флаги (managed=False)."""

    def test_extra_flags_survive_sanitize(self):
        sanitized = _sanitize_extra_args(
            "--mlock --ctx-checkpoints 8 --split-mode none --device CUDA0"
        )
        for flag in ("--mlock", "--ctx-checkpoints", "--split-mode", "--device"):
            self.assertIn(flag, sanitized)

    def test_managed_flag_still_stripped_by_sanitize(self):
        # --flash-attn управляется UI (managed=True) → вырезается.
        sanitized = _sanitize_extra_args("--mlock --flash-attn on")
        self.assertIn("--mlock", sanitized)
        self.assertNotIn("--flash-attn", sanitized)


if __name__ == "__main__":
    unittest.main()
