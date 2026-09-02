"""Тесты новой интеграции AutoTune (llama_autotuner engine)."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from llama_autotuner.models import (
    BenchmarkMetrics,
    Candidate,
    CandidateResult,
    LaunchProfile,
    RunStatus,
)
from llama_autotuner.session import (
    AutotuneSessionError,
    SessionConfig,
    SessionResult,
    resolve_workload_profile,
)
from src.core.config import candidate_to_settings_values
from src.services.autotune_manager import AutoTuneManager

_APP = QApplication.instance() or QApplication([])


class TestResolveWorkloadProfile(unittest.TestCase):
    def test_explicit_profile_passthrough(self):
        self.assertEqual(resolve_workload_profile("chat", 999999), "chat")

    def test_auto_short_context_is_chat(self):
        self.assertEqual(resolve_workload_profile("auto", 8192), "chat")

    def test_auto_medium_context_is_agent(self):
        self.assertEqual(resolve_workload_profile("auto", 65536), "agent")

    def test_auto_long_context_is_long_context(self):
        self.assertEqual(resolve_workload_profile("auto", 131072), "long-context")


class TestCandidateToSettingsValues(unittest.TestCase):
    def _candidate(self, **overrides):
        base = dict(ctx=32768, ngl="all", ncmoe=None, kv_k="q8_0", kv_v="q8_0", mtp=False)
        base.update(overrides)
        return Candidate(**base)

    def test_ngl_all_sets_gpu_auto(self):
        values = candidate_to_settings_values(self._candidate(ngl="all"))
        self.assertTrue(values["gpu_auto"])
        self.assertNotIn("gpu_layers", values)

    def test_ngl_numeric_sets_gpu_layers(self):
        values = candidate_to_settings_values(self._candidate(ngl=22))
        self.assertFalse(values["gpu_auto"])
        self.assertEqual(values["gpu_layers"], 22)

    def test_ncmoe_none_is_omitted(self):
        values = candidate_to_settings_values(self._candidate(ncmoe=None))
        self.assertNotIn("cpu_moe_layers", values)

    def test_ncmoe_set_is_included(self):
        values = candidate_to_settings_values(self._candidate(ncmoe=12))
        self.assertEqual(values["cpu_moe_layers"], 12)

    def test_mtp_off_omits_draft_fields(self):
        values = candidate_to_settings_values(self._candidate(mtp=False))
        self.assertFalse(values["speculative_mtp"])
        self.assertNotIn("spec_draft_n_max", values)
        self.assertNotIn("spec_draft_p_min", values)

    def test_mtp_on_includes_draft_fields(self):
        values = candidate_to_settings_values(
            self._candidate(mtp=True, mtp_n_max=6, mtp_p_min=0.7)
        )
        self.assertTrue(values["speculative_mtp"])
        self.assertEqual(values["spec_draft_n_max"], 6)
        self.assertEqual(values["spec_draft_p_min"], 0.7)

    def test_common_fields_mapped(self):
        values = candidate_to_settings_values(
            self._candidate(ctx=16384, kv_k="q4_0", kv_v="f16")
        )
        self.assertEqual(values["ctx_size"], 16384)
        self.assertEqual(values["cache_type_k"], "q4_0")
        self.assertEqual(values["cache_type_v"], "f16")

    def test_extra_args_not_included(self):
        values = candidate_to_settings_values(
            self._candidate(extra_args=["--metrics"])
        )
        self.assertNotIn("extra_args", values)


def _make_result(status=RunStatus.PASS, reason="ok"):
    candidate = Candidate(ctx=32768, ngl="all", kv_k="q8_0", kv_v="q8_0")
    metrics = BenchmarkMetrics(pp_tps=1000.0, tg_tps=40.0, vram_peak_mb=8000)
    return CandidateResult(
        candidate=candidate, status=status, reason=reason, metrics=metrics, score=0.9
    )


class TestAutoTuneManager(unittest.TestCase):
    def setUp(self):
        self.config = SessionConfig(server_exe="llama-server.exe", model_path="model.gguf")

    def test_run_success_emits_results_progress_and_finished(self):
        result = _make_result()
        session_result = SessionResult(
            status="COMPLETED",
            stop_reason="RUNNING",
            profiles=[],
            results=[result],
            target=None,
            model=None,
            hardware=None,
            elapsed_seconds=12.5,
            output_dir="autotune_runs/test",
        )

        def fake_run_session(config, progress, on_result, cancel_event=None):
            progress("hello")
            on_result(result)
            return session_result

        manager = AutoTuneManager(self.config)
        logs = []
        results = []
        progresses = []
        finished = []
        manager.log.connect(lambda text, level: logs.append((text, level)))
        manager.result_ready.connect(lambda r: results.append(r))
        manager.progress.connect(lambda done, total: progresses.append((done, total)))
        manager.session_finished.connect(lambda sr: finished.append(sr))

        with patch(
            "src.services.autotune_manager.run_session", side_effect=fake_run_session
        ):
            manager.run()

        self.assertEqual(logs, [("hello", "info")])
        self.assertEqual(results, [result])
        self.assertEqual(progresses, [(1, 0)])
        self.assertEqual(finished, [session_result])
        self.assertEqual(manager.results, [result])

    def test_run_failure_emits_session_failed(self):
        def fake_run_session(config, progress, on_result, cancel_event=None):
            raise AutotuneSessionError("no NVIDIA GPU")

        manager = AutoTuneManager(self.config)
        failures = []
        finished = []
        manager.session_failed.connect(lambda msg: failures.append(msg))
        manager.session_finished.connect(lambda sr: finished.append(sr))

        with patch(
            "src.services.autotune_manager.run_session", side_effect=fake_run_session
        ):
            manager.run()

        self.assertEqual(failures, ["no NVIDIA GPU"])
        self.assertEqual(finished, [])

    def test_cancel_sets_cancel_event(self):
        manager = AutoTuneManager(self.config)
        self.assertFalse(manager._cancel_event.is_set())
        manager.cancel()
        self.assertTrue(manager._cancel_event.is_set())


if __name__ == "__main__":
    unittest.main()
