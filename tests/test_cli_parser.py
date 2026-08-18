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
        self.assertEqual(parsed.settings["top_p"], 0.9)
        self.assertEqual(parsed.settings["min_p"], 0.05)
        self.assertEqual(parsed.extra_args, "")

    def test_parses_all_sampling_fields(self):
        parsed = parse_llama_server_command(
            "llama-server -m model.gguf --top-k 20 --top-p 0.9 --min-p 0.05 "
            "--typical-p 0.95 --repeat-last-n 128 --repeat-penalty 1.1 "
            "--presence-penalty -0.2 --frequency-penalty 0.1 --seed 42"
        )

        self.assertEqual(parsed.settings["top_k"], 20)
        self.assertEqual(parsed.settings["top_p"], 0.9)
        self.assertEqual(parsed.settings["min_p"], 0.05)
        self.assertEqual(parsed.settings["typical_p"], 0.95)
        self.assertEqual(parsed.settings["repeat_last_n"], 128)
        self.assertEqual(parsed.settings["repeat_penalty"], 1.1)
        self.assertEqual(parsed.settings["presence_penalty"], -0.2)
        self.assertEqual(parsed.settings["frequency_penalty"], 0.1)
        self.assertEqual(parsed.settings["seed"], 42)
        self.assertEqual(parsed.extra_args, "")

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

    def test_parses_managed_p_min_and_keeps_unmanaged_speculative_flags_in_extra(self):
        parsed = parse_llama_server_command(
            "llama-server.exe -m model.gguf --spec-type draft-mtp "
            "--spec-draft-n-min 1 --spec-draft-p-min 0.5"
        )

        self.assertTrue(parsed.settings["speculative_mtp"])
        self.assertEqual(parsed.settings["spec_draft_p_min"], 0.5)
        self.assertEqual(parsed.extra_args, "--spec-draft-n-min 1")

    def test_parses_multiline_args_log_with_cmd_carets(self):
        parsed = parse_llama_server_command(
            "Args: -m G:\\Models\\model.gguf ^\n"
            "  --host 127.0.0.1 --port 8080 ^\n"
            "  --device CUDA0 --split-mode none --main-gpu 0 ^\n"
            "  -ngl all -c 65536 --spec-type draft-mtp ^\n"
            "  --spec-draft-n-max 8 --spec-draft-p-min 0.8 ^\n"
            "  --spec-draft-ngl all --spec-draft-device CUDA0 ^\n"
            "  --spec-draft-n-min 0\n"
            "Env: CUDA_VISIBLE_DEVICES=0"
        )

        self.assertEqual(parsed.model_path, "G:\\Models\\model.gguf")
        self.assertEqual(parsed.settings["host"], "127.0.0.1")
        self.assertEqual(parsed.settings["port"], 8080)
        self.assertEqual(parsed.settings["cuda_device"], "CUDA0")
        self.assertEqual(parsed.settings["split_mode"], "none")
        self.assertEqual(parsed.settings["main_gpu"], 0)
        self.assertTrue(parsed.settings["gpu_layers_all"])
        self.assertEqual(parsed.settings["ctx_size"], 65536)
        self.assertTrue(parsed.settings["speculative_mtp"])
        self.assertEqual(parsed.settings["spec_draft_n_max"], 8)
        self.assertEqual(parsed.settings["spec_draft_p_min"], 0.8)
        self.assertEqual(parsed.settings["spec_draft_gpu_layers"], "all")
        self.assertEqual(parsed.settings["spec_draft_device"], "CUDA0")
        self.assertEqual(parsed.extra_args, "--spec-draft-n-min 0")
        self.assertNotIn("^", parsed.extra_args)


    def test_parses_reasoning_controls(self):
        parsed = parse_llama_server_command(
            "llama-server -m model.gguf --reasoning-effort xhigh "
            "--reasoning-preserve --reasoning-budget 256 "
            '--reasoning-budget-message "budget hit"'
        )

        self.assertEqual(parsed.settings["reasoning_effort"], "xhigh")
        self.assertEqual(parsed.settings["reasoning_preserve"], "preserve")
        self.assertEqual(parsed.settings["reasoning_budget"], 256)
        self.assertEqual(parsed.settings["reasoning_budget_message"], "budget hit")
        self.assertEqual(parsed.extra_args, "")

    def test_parses_reasoning_no_preserve(self):
        parsed = parse_llama_server_command(
            "llama-server -m model.gguf --no-reasoning-preserve"
        )
        self.assertEqual(parsed.settings["reasoning_preserve"], "no-preserve")

    def test_parses_reasoning_effort_only(self):
        parsed = parse_llama_server_command(
            "llama-server -m model.gguf --reasoning-effort medium"
        )
        self.assertEqual(parsed.settings["reasoning_effort"], "medium")
        self.assertEqual(parsed.settings.get("reasoning_preserve", "off"), "off")


if __name__ == "__main__":
    unittest.main()
