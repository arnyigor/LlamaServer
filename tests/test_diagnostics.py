import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.diagnostics import (
    analyze_server_failure,
    format_diagnostic_summary,
    write_server_report,
)


class TestServerDiagnostics(unittest.TestCase):
    def test_reports_vram_oom_with_action(self):
        result = analyze_server_failure(
            1,
            "CUDA error: out of memory while allocating KV cache",
            crash_exit=False,
            stop_requested=False,
        )

        self.assertIn("Out of memory", result["cause"])
        self.assertIn("Context Size", result["action"])

    def test_reports_native_crash_during_model_unload(self):
        result = analyze_server_failure(
            -1073741819,
            "llama_model_unload: model unloaded",
            crash_exit=True,
            stop_requested=True,
        )

        self.assertIn("Failed to unload the model", result["cause"])
        self.assertIn("0xC0000005", result["cause"])

    def test_normal_requested_stop_has_no_failure(self):
        result = analyze_server_failure(
            0,
            "server stopped",
            crash_exit=False,
            stop_requested=True,
        )
        self.assertIsNone(result)

    def test_chat_template_information_is_not_reported_as_jinja_error(self):
        result = analyze_server_failure(
            1,
            "chat template supports preserving reasoning, consider enabling it",
            crash_exit=False,
            stop_requested=True,
        )
        self.assertIsNone(result)

    def test_writes_command_and_recent_output_to_report(self):
        result = analyze_server_failure(
            1,
            "unknown argument: --future-flag",
            stop_requested=False,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.core.diagnostics.diagnostics_dir", return_value=Path(tmp)
        ):
            path = write_server_report(
                result,
                executable="llama-server.exe",
                args=["-m", "model.gguf", "--future-flag"],
                output="unknown argument: --future-flag",
            )
            report = path.read_text(encoding="utf-8")

        self.assertIn("llama-server.exe -m model.gguf --future-flag", report)
        self.assertIn("unknown argument", report)
        self.assertIn("Full report", format_diagnostic_summary(result, str(path)))


if __name__ == "__main__":
    unittest.main()
