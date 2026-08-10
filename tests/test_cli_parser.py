import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cli_parser import parse_llama_server_command


class TestCliParser(unittest.TestCase):
    def test_parses_known_ui_flags_and_keeps_unknown_extra(self):
        parsed = parse_llama_server_command(
            r'llama-server.exe -m G:\Models\model.gguf --host 127.0.0.1 '
            r'--port 8081 -c 32768 -ngl all -ncmoe 12 -ctk q8_0 -ctv q4_0 '
            r'--flash-attn off --top-p 0.9 --min-p 0.05 --metrics'
        )

        self.assertEqual(parsed.model_path, r"G:\Models\model.gguf")
        self.assertEqual(parsed.settings["host"], "127.0.0.1")
        self.assertEqual(parsed.settings["port"], 8081)
        self.assertEqual(parsed.settings["ctx_size"], 32768)
        self.assertTrue(parsed.settings["gpu_layers_all"])
        self.assertFalse(parsed.settings["gpu_auto"])
        self.assertEqual(parsed.settings["cpu_moe_layers"], 12)
        self.assertEqual(parsed.settings["cache_type_k"], "q8_0")
        self.assertEqual(parsed.settings["cache_type_v"], "q4_0")
        self.assertFalse(parsed.settings["flash_attn"])
        self.assertEqual(parsed.extra_args, "--top-p 0.9 --min-p 0.05")

    def test_parses_quoted_windows_paths(self):
        parsed = parse_llama_server_command(
            r'"G:\AI Models\llamacpp\llama-win-cuda-13.3-x64\llama-server.exe" '
            r'-m "G:\Models\My Model\model q4.gguf" --temp 0.7'
        )

        self.assertEqual(
            parsed.executable,
            r"G:\AI Models\llamacpp\llama-win-cuda-13.3-x64\llama-server.exe",
        )
        self.assertEqual(parsed.settings["cuda_version"], "13")
        self.assertEqual(parsed.model_path, r"G:\Models\My Model\model q4.gguf")
        self.assertEqual(parsed.settings["temperature"], 0.7)

    def test_keeps_unmanaged_speculative_flags_in_extra(self):
        parsed = parse_llama_server_command(
            "llama-server.exe -m model.gguf --spec-type draft-mtp "
            "--spec-draft-n-min 1 --spec-draft-p-min 0.5"
        )

        self.assertTrue(parsed.settings["speculative_mtp"])
        self.assertEqual(
            parsed.extra_args, "--spec-draft-n-min 1 --spec-draft-p-min 0.5"
        )


if __name__ == "__main__":
    unittest.main()
