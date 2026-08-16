"""Тесты MVP AutoTune benchmark."""

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate, BenchmarkResult
from src.core.benchmark_plan import build_autotune_plan
from src.core.benchmark_scorer import score_result
from src.core.cli_builder import build_benchmark_args_from_params
from src.services.autotune_manager import AutoTuneManager
from src.services.benchmark_runner import BenchmarkRunner, parse_llama_bench_output
from src.services.report_writer import (
    write_best,
    write_json_report,
    write_markdown_report,
    write_plan,
)


@dataclass
class AutoTuneSettingsStub:
    gpu_auto: bool = True
    gpu_layers: int = 33
    cpu_moe_layers: int = 0
    ctx_size: int = 32768
    threads: int = 8
    threads_batch: int = 0
    flash_attn: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    batch_size: int = -1
    ubatch_size: int = -1
    parallel_slots: int = -1
    ctx_checkpoints: int = 0
    cache_ram: int = 0
    use_mmproj: bool = False


class TestAutoTuneCliBuilder(unittest.TestCase):
    def test_build_benchmark_args_from_params_preserves_zero_values(self):
        params = {
            "ngl": "auto",
            "ctx_size": 32768,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "q8_0",
            "cache_type_v": "q4_0",
            "threads": 12,
            "threads_batch": 0,
            "parallel_slots": 1,
            "flash_attn": False,
            "ncmoe": 0,
            "ctx_checkpoints": 0,
            "cache_ram": 0,
            "use_mmproj": False,
        }

        args = build_benchmark_args_from_params("model.gguf", params, 128, 256)

        self.assertEqual(args[args.index("-ngl") + 1], "99")
        self.assertEqual(args[args.index("-fa") + 1], "off")
        self.assertEqual(args[args.index("-ub") + 1], "512")
        self.assertEqual(args[args.index("-ncmoe") + 1], "0")
        # Эти параметры есть у llama-server, но отсутствуют в актуальном llama-bench.
        # Они остаются в плане/пресете и не должны ломать запуск benchmark.
        self.assertNotIn("-c", args)
        self.assertNotIn("-np", args)
        self.assertNotIn("-tb", args)
        self.assertNotIn("--ctx-checkpoints", args)
        self.assertNotIn("--cache-ram", args)
        self.assertNotIn("--no-mmproj", args)


class TestAutoTunePlan(unittest.TestCase):
    def test_quick_plan_has_baseline_unique_candidates_and_limits(self):
        settings = AutoTuneSettingsStub()
        plan = build_autotune_plan(
            settings,
            "G:/models/model.gguf",
            {"expert_count": 8, "recommended_ctx": 16384},
            mode="quick",
            target="balanced",
            max_runs=20,
            time_budget_sec=600,
        )

        self.assertEqual(plan.ctx_size, 32768)
        self.assertEqual(plan.mode, "quick")
        self.assertLessEqual(len(plan.candidates), 20)
        self.assertGreaterEqual(len(plan.candidates), 7)
        self.assertEqual(plan.candidates[0].id, "run_001")
        self.assertEqual(plan.candidates[0].stage, "baseline")
        self.assertEqual(plan.candidates[0].params["cache_type_k"], "q8_0")
        self.assertEqual(plan.candidates[0].params["cache_type_v"], "q8_0")
        self.assertEqual(plan.candidates[0].params["ngl"], 99)
        self.assertEqual(plan.candidates[0].params["ncmoe"], -1)
        self.assertEqual(plan.candidates[0].params["parallel_slots"], 1)
        self.assertTrue(plan.candidates[0].params["fit_off"])
        self.assertFalse(plan.candidates[0].params["cache_prompt"])
        self.assertEqual(plan.candidates[0].params["ctx_checkpoints"], 0)
        self.assertEqual(plan.candidates[0].params["cache_ram"], 0)

        normalized = [
            tuple(sorted((k, str(v)) for k, v in c.params.items()))
            for c in plan.candidates
        ]
        self.assertEqual(len(normalized), len(set(normalized)))
        for candidate in plan.candidates:
            self.assertLessEqual(
                candidate.params["ubatch_size"], candidate.params["batch_size"]
            )

    def test_low_vram_plan_prioritizes_quantized_kv(self):
        settings = AutoTuneSettingsStub(ctx_size=-1)
        plan = build_autotune_plan(
            settings,
            "model.gguf",
            {"context_length": 131072},
            mode="quick",
            target="low_vram",
            max_runs=5,
        )

        self.assertEqual(plan.ctx_size, 32768)
        kv_pairs = [
            (c.params["cache_type_k"], c.params["cache_type_v"])
            for c in plan.candidates
        ]
        self.assertIn(("q4_0", "q4_0"), kv_pairs)

    def test_dense_huge_context_plan_does_not_use_moe_and_starts_memory_safe(self):
        settings = AutoTuneSettingsStub(
            ctx_size=131072, cpu_moe_layers=99, ubatch_size=-1
        )
        plan = build_autotune_plan(
            settings,
            "dense.gguf",
            {
                "architecture": "llama",
                "expert_count": 0,
                "block_count": 40,
                "head_count": 40,
                "embedding_length": 5120,
                "size_gib": 20.0,
                "context_length": 131072,
            },
            mode="quick",
            target="balanced",
            max_runs=10,
        )

        self.assertEqual(plan.ctx_size, 131072)
        self.assertEqual(plan.candidates[0].params["model_type"], "dense")
        self.assertEqual(plan.candidates[0].params["ngl"], 40)
        self.assertEqual(plan.candidates[0].params["cache_type_k"], "q4_0")
        self.assertEqual(plan.candidates[0].params["cache_type_v"], "q4_0")
        self.assertEqual(plan.candidates[0].params["ubatch_size"], 256)
        self.assertEqual(plan.candidates[0].params["ncmoe"], -1)
        self.assertEqual(plan.candidates[0].params["ctx_checkpoints"], 0)
        self.assertEqual(plan.candidates[0].params["cache_ram"], 0)
        self.assertFalse(any(c.stage == "moe" for c in plan.candidates))
        self.assertTrue(all(c.params["ncmoe"] == -1 for c in plan.candidates))

    def test_moe_regular_context_does_not_apply_stale_server_params_to_all_candidates(
        self,
    ):
        settings = AutoTuneSettingsStub(
            ctx_size=32768, cpu_moe_layers=8, parallel_slots=2
        )
        plan = build_autotune_plan(
            settings,
            "moe.gguf",
            {
                "architecture": "gemma4",
                "expert_count": 128,
                "expert_used": 0,
                "block_count": 30,
                "head_count": 16,
                "embedding_length": 2816,
                "size_gib": 12.7,
            },
            mode="quick",
            target="balanced",
            max_runs=20,
        )

        self.assertEqual(plan.candidates[0].params["model_type"], "moe")
        self.assertEqual(plan.candidates[0].params["ngl"], 30)
        self.assertEqual(plan.candidates[0].params["ncmoe"], -1)
        self.assertEqual(plan.candidates[0].params["parallel_slots"], 1)
        self.assertTrue(plan.candidates[0].params["fit_off"])
        early_stages = {c.stage for c in plan.candidates[:5]}
        self.assertTrue({"kv", "batch"} & early_stages)
        self.assertFalse(all(c.stage == "moe_vram" for c in plan.candidates[1:5]))
        moe_vram = [c for c in plan.candidates if c.stage == "moe_vram"]
        self.assertGreaterEqual(len(moe_vram), 1)
        self.assertTrue(all(c.params["parallel_slots"] == 1 for c in plan.candidates))
        self.assertFalse(any(c.stage == "moe" for c in plan.candidates))

    def test_mtp_moe_plan_uses_real_ngl_speculative_and_ncmoe_candidates(self):
        settings = AutoTuneSettingsStub(
            ctx_size=32768, cpu_moe_layers=8, parallel_slots=1
        )
        plan = build_autotune_plan(
            settings,
            "G:/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            {
                "path": "G:/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                "architecture": "qwen35moe",
                "expert_count": 256,
                "expert_used": 0,
                "block_count": 41,
                "head_count": 16,
                "embedding_length": 2048,
                "size_gib": 21.28,
            },
            mode="quick",
            target="balanced",
            max_runs=20,
        )

        self.assertEqual(plan.candidates[0].params["ngl"], 41)
        self.assertEqual(plan.candidates[0].params["parallel_slots"], 1)
        self.assertFalse(plan.candidates[0].params["kv_unified"])
        self.assertTrue(plan.candidates[0].params["speculative_mtp"])
        self.assertTrue(plan.candidates[0].params["gpu_layers_all"])
        self.assertEqual(plan.candidates[0].params["spec_draft_n_max"], 8)
        self.assertEqual(plan.candidates[0].params["spec_draft_p_min"], 0.8)
        self.assertEqual(plan.candidates[0].params["spec_draft_gpu_layers"], "all")
        self.assertEqual(plan.candidates[0].params["cache_type_k"], "q8_0")
        self.assertEqual(plan.candidates[0].params["cache_type_v"], "q8_0")
        moe_candidates = [c for c in plan.candidates if c.stage in {"moe", "moe_vram"}]
        self.assertGreaterEqual(len(moe_candidates), 1)
        self.assertEqual(plan.candidates[0].params["ncmoe"], 8)
        self.assertTrue(
            all(c.params["ncmoe"] <= c.params["ngl"] for c in moe_candidates)
        )

    def test_smart_auto_plan_records_constraints_and_filters_blocked_vram_candidates(self):
        settings = AutoTuneSettingsStub(ctx_size=131072, ubatch_size=-1)
        model_info = {
            "architecture": "llama",
            "expert_count": 0,
            "block_count": 40,
            "head_count": 40,
            "embedding_length": 5120,
            "size_gib": 20.0,
            "context_length": 131072,
        }

        with patch("src.core.benchmark_plan._detect_total_vram_gib", return_value=16.0):
            plan = build_autotune_plan(
                settings,
                "dense.gguf",
                model_info,
                mode="smart",
                target="auto",
                max_runs=None,
                time_budget_sec=None,
            )

        self.assertEqual(plan.mode, "smart")
        self.assertEqual(plan.target, "low_vram")
        self.assertLessEqual(len(plan.candidates), 10)
        self.assertEqual(plan.constraints["gpu_vram_gib"], 16.0)
        self.assertEqual(plan.constraints["selected_target"], "low_vram")
        self.assertTrue(plan.constraints["notes"])
        self.assertTrue(all("_estimated_vram_gib" in c.params for c in plan.candidates))
        self.assertFalse(any(c.params.get("_risk") == "blocked" for c in plan.candidates))

    def test_autotune_metadata_is_ignored_by_llama_bench_cli(self):
        params = {
            "ngl": 40,
            "batch_size": 512,
            "ubatch_size": 256,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "threads": 12,
            "flash_attn": True,
            "_estimated_vram_gib": 15.5,
            "_risk": "high",
        }

        args = build_benchmark_args_from_params("model.gguf", params, 128, 256)

        self.assertNotIn("_estimated_vram_gib", args)
        self.assertNotIn("_risk", args)

    def test_moe_huge_context_quick_keeps_moe_candidates_limited(self):
        settings = AutoTuneSettingsStub(ctx_size=131072, cpu_moe_layers=8)
        plan = build_autotune_plan(
            settings,
            "moe.gguf",
            {
                "architecture": "gemma4",
                "expert_count": 128,
                "expert_used": 0,
                "block_count": 30,
                "head_count": 16,
                "embedding_length": 2816,
                "size_gib": 12.7,
            },
            mode="quick",
            target="balanced",
            max_runs=20,
        )

        moe_candidates = [c for c in plan.candidates if c.stage in {"moe", "moe_vram"}]
        self.assertGreaterEqual(len(moe_candidates), 1)
        self.assertLessEqual(len(moe_candidates), 8)
        self.assertTrue(all(c.params["ctx_checkpoints"] == 0 for c in plan.candidates))
        self.assertTrue(all(c.params["cache_ram"] == 0 for c in plan.candidates))
        self.assertTrue(all(c.params["ngl"] <= 30 for c in plan.candidates))
        self.assertTrue(
            all(c.params["ncmoe"] <= c.params["ngl"] for c in moe_candidates)
        )


class TestAutoTuneScoring(unittest.TestCase):
    def _result(self, status="success"):
        result = BenchmarkResult(
            candidate_id="run_001",
            status=status,
            prompt_tok_s=1800.0,
            generation_tok_s=32.0,
            load_time_sec=10.0,
            vram_used_mib=12000.0,
            ram_used_mib=8000.0,
        )
        result.metrics.prompt_tok_s = result.prompt_tok_s
        result.metrics.generation_tok_s = result.generation_tok_s
        result.metrics.load_time_sec = result.load_time_sec
        result.metrics.vram_used_mib = result.vram_used_mib
        result.metrics.vram_free_mib = 2048.0
        return result

    def test_failed_result_scores_zero(self):
        result = self._result(status="failed_oom")
        self.assertEqual(score_result(result, {}, "balanced"), 0.0)
        self.assertEqual(result.score, 0.0)

    def test_quality_kv_penalizes_aggressive_kv_quantization(self):
        good = self._result()
        aggressive = self._result()

        good_score = score_result(
            good, {"cache_type_k": "f16", "cache_type_v": "f16"}, "quality_kv"
        )
        aggressive_score = score_result(
            aggressive,
            {"cache_type_k": "q4_0", "cache_type_v": "q4_0"},
            "quality_kv",
        )

        self.assertGreater(good_score, aggressive_score)

    def test_huge_context_balanced_penalizes_f16_microbench_winner(self):
        f16 = self._result()
        q4 = self._result()
        f16_score = score_result(
            f16,
            {
                "ctx_size": 131072,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "ncmoe": -1,
                "model_type": "dense",
                "cache_ram": 0,
                "ctx_checkpoints": 0,
            },
            "balanced",
        )
        q4_score = score_result(
            q4,
            {
                "ctx_size": 131072,
                "cache_type_k": "q4_0",
                "cache_type_v": "q4_0",
                "ncmoe": -1,
                "model_type": "dense",
                "cache_ram": 0,
                "ctx_checkpoints": 0,
            },
            "balanced",
        )

        self.assertGreater(q4_score, f16_score)

    def test_huge_context_moe_balanced_prefers_q8_over_q4_when_speeds_are_close(self):
        q8 = self._result()
        q4 = self._result()
        q8_score = score_result(
            q8,
            {
                "ctx_size": 131072,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "ncmoe": -1,
                "model_type": "moe",
                "cache_ram": 0,
                "ctx_checkpoints": 0,
            },
            "balanced",
        )
        q4_score = score_result(
            q4,
            {
                "ctx_size": 131072,
                "cache_type_k": "q4_0",
                "cache_type_v": "q4_0",
                "ncmoe": -1,
                "model_type": "moe",
                "cache_ram": 0,
                "ctx_checkpoints": 0,
            },
            "balanced",
        )

        self.assertGreater(q8_score, q4_score)

    def test_balanced_score_penalizes_high_resource_risk(self):
        low_risk = self._result()
        high_risk = self._result()

        low_score = score_result(
            low_risk,
            {"_risk": "low", "_vram_pct": 70.0, "ctx_size": 32768},
            "balanced",
        )
        high_score = score_result(
            high_risk,
            {"_risk": "high", "_vram_pct": 99.0, "ctx_size": 32768},
            "balanced",
        )

        self.assertGreater(low_score, high_score)


class TestBenchmarkRunnerParsing(unittest.TestCase):
    def test_parse_llama_bench_markdown_output(self):
        output = """
        | model | test | t/s |
        | llama | pp 128 | 1820.5 ± 10.0 |
        | llama | tg 256 | 31.8 ± 0.2 |
        load time = 18400 ms
        VRAM used: 14.5 GiB
        RAM used: 9100 MiB
        """

        metrics = parse_llama_bench_output(output)

        self.assertEqual(metrics.prompt_tok_s, 1820.5)
        self.assertEqual(metrics.generation_tok_s, 31.8)
        self.assertAlmostEqual(metrics.load_time_sec, 18.4)
        self.assertAlmostEqual(metrics.vram_used_mib, 14.5 * 1024)
        self.assertEqual(metrics.ram_used_mib, 9100)

    def test_runner_marks_success_and_writes_log(self):
        class DummyProcess:
            returncode = 0

            def communicate(self, timeout=None):
                return "| x | pp 128 | 100.0 ± 1 |\n| x | tg 256 | 20.0 ± 1 |", None

            def poll(self):
                return 0

        candidate = BenchmarkCandidate(
            "run_001", {"ctx_size": 8192}, "test", "baseline"
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "src.services.benchmark_runner.subprocess.Popen",
                return_value=DummyProcess(),
            ),
        ):
            runner = BenchmarkRunner("llama-bench.exe", "model.gguf", tmpdir)
            result = runner.run(candidate, 128, 256, 60)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.prompt_tok_s, 100.0)
            self.assertEqual(result.generation_tok_s, 20.0)
            self.assertTrue(Path(result.log_path).exists())

    def test_runner_detects_oom(self):
        class DummyProcess:
            returncode = 1

            def communicate(self, timeout=None):
                return "CUDA error: out of memory while allocating buffer", None

            def poll(self):
                return 0

        candidate = BenchmarkCandidate("run_002", {"ctx_size": 131072}, "oom", "memory")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "src.services.benchmark_runner.subprocess.Popen",
                return_value=DummyProcess(),
            ),
        ):
            runner = BenchmarkRunner("llama-bench.exe", "model.gguf", tmpdir)
            result = runner.run(candidate, 128, 256, 60)

            self.assertEqual(result.status, "failed_oom")
            self.assertEqual(result.error, "Out of memory")


class TestAutoTuneManager(unittest.TestCase):
    def test_repeat_top_runs_verification_candidates(self):
        class FakeRunner:
            seen: list[str] = []

            def __init__(self, *_args, **_kwargs):
                pass

            def cancel(self):
                pass

            def run(
                self,
                candidate,
                prompt_tokens=128,
                generation_tokens=256,
                timeout_sec=300,
                log_callback=None,
            ):
                self.seen.append(candidate.id)
                tg_by_id = {
                    "run_001": 20.0,
                    "run_002": 26.0,
                    "verify_001_run_002": 25.5,
                    "verify_002_run_001": 20.5,
                }
                result = BenchmarkResult(
                    candidate_id=candidate.id,
                    status="success",
                    prompt_tok_s=1000.0,
                    generation_tok_s=tg_by_id.get(candidate.id, 22.0),
                    load_time_sec=2.0,
                )
                result.metrics.prompt_tok_s = result.prompt_tok_s
                result.metrics.generation_tok_s = result.generation_tok_s
                return result

        candidates = [
            BenchmarkCandidate(
                "run_001",
                {"ctx_size": 8192, "batch_size": 512, "ubatch_size": 256},
                "baseline",
                "baseline",
            ),
            BenchmarkCandidate(
                "run_002",
                {"ctx_size": 8192, "batch_size": 1024, "ubatch_size": 512},
                "faster batch",
                "batch",
            ),
        ]
        plan = AutoTunePlan(
            model_path="G:/models/model.gguf",
            ctx_size=8192,
            mode="smart",
            target="balanced",
            engine="llama-bench",
            time_budget_sec=900,
            max_runs=2,
            repeat_top=2,
            candidates=candidates,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("src.services.autotune_manager.BenchmarkRunner", FakeRunner),
        ):
            manager = AutoTuneManager(
                "llama-bench.exe",
                plan,
                per_run_timeout_sec=120,
                output_root=tmpdir,
            )
            manager.run()

        verify_ids = [c.id for c in plan.candidates if c.stage == "verify"]
        self.assertEqual(verify_ids, ["verify_001_run_002", "verify_002_run_001"])
        self.assertEqual([r.candidate_id for r in manager.results], [
            "run_001",
            "run_002",
            "verify_001_run_002",
            "verify_002_run_001",
        ])
        self.assertEqual(manager.best_result.candidate_id, "verify_001_run_002")


class TestAutoTuneReports(unittest.TestCase):
    def test_report_writer_creates_json_markdown_and_best_files(self):
        candidate = BenchmarkCandidate(
            "run_001",
            {"ctx_size": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            "baseline",
            "baseline",
        )
        plan = AutoTunePlan(
            model_path="G:/models/model.gguf",
            ctx_size=8192,
            mode="quick",
            target="balanced",
            engine="llama-bench",
            time_budget_sec=900,
            max_runs=1,
            repeat_top=1,
            candidates=[candidate],
        )
        result = BenchmarkResult(
            candidate_id="run_001",
            status="success",
            score=91.4,
            prompt_tok_s=1000.0,
            generation_tok_s=25.0,
            command=["llama-bench.exe", "-m", "model.gguf"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = write_plan(tmpdir, plan)
            json_path = write_json_report(
                tmpdir, plan, {"architecture": "gemma", "quant": "Q4"}, [result], result
            )
            report_path = write_markdown_report(tmpdir, plan, {}, [result], result)
            best_path = write_best(tmpdir, result, candidate.params)

            for path in (plan_path, json_path, report_path, best_path):
                self.assertTrue(Path(path).exists())

            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["best_run_id"], "run_001")
            self.assertEqual(payload["runs"][0]["params"]["ctx_size"], 8192)
            self.assertIn(
                "AutoTune Report", Path(report_path).read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
