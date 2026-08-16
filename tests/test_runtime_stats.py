import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.runtime_stats import format_runtime_stats_markdown


class TestRuntimeStatsMarkdown(unittest.TestCase):
    def test_formats_snapshot_as_markdown_report(self):
        text = format_runtime_stats_markdown(
            {
                "exported_at": "2026-08-16T12:34:56+0500",
                "server": {
                    "running": True,
                    "base_url": "http://127.0.0.1:8080/v1",
                },
                "model": {
                    "path": "G:/AIModels/model.gguf",
                    "id": "model.gguf",
                },
                "tokens": {
                    "total": 100,
                    "task": 90,
                    "prompt": 30,
                    "generated": 70,
                    "request_prompt": 10,
                    "request_generated": 20,
                    "saved_last": 40,
                    "saved_total": 50,
                },
                "time_seconds": {
                    "active_total": 125,
                    "active_prompt": 5,
                    "active_generated": 120,
                    "current_total": 61,
                    "current_prompt": 1,
                    "current_generated": 60,
                },
            }
        )

        self.assertIn("# LlamaServer Runtime Stats", text)
        self.assertIn("- Running: yes", text)
        self.assertIn("| Total | 100 |", text)
        self.assertIn("| Active total | 125.000 | 2:05 |", text)

    def test_missing_sections_use_defaults(self):
        text = format_runtime_stats_markdown({})

        self.assertIn("- Exported: -", text)
        self.assertIn("- Running: no", text)
        self.assertIn("| Total | 0 |", text)
        self.assertIn("| Current generated | 0.000 | 0:00 |", text)
