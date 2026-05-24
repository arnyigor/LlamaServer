"""QThread-оркестратор AutoTune benchmark."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Signal

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate, BenchmarkResult
from src.core.benchmark_scorer import score_result
from src.services.benchmark_runner import BenchmarkRunner
from src.services.report_writer import (
    write_best,
    write_json_report,
    write_markdown_report,
    write_plan,
)


class AutoTuneManager(QThread):
    progress = Signal(int, int)
    log = Signal(str, str)
    run_started = Signal(object)
    run_finished = Signal(object)
    autotune_finished = Signal(object, str)

    def __init__(
        self,
        bench_exe: str,
        plan: AutoTunePlan,
        model_info: Dict[str, Any] | None = None,
        prompt_tokens: int = 128,
        generation_tokens: int = 256,
        per_run_timeout_sec: int = 300,
        output_root: str = "benchmarks",
        parent=None,
    ):
        super().__init__(parent)
        self.bench_exe = bench_exe
        self.plan = plan
        self.model_info = model_info or {}
        self.prompt_tokens = int(prompt_tokens)
        self.generation_tokens = int(generation_tokens)
        self.per_run_timeout_sec = int(per_run_timeout_sec)
        self.results: List[BenchmarkResult] = []
        self.best_result: Optional[BenchmarkResult] = None
        self.output_dir = self._make_output_dir(output_root)
        self.runner: Optional[BenchmarkRunner] = None
        self.max_failures = 12
        self.max_oom_failures = 5

    def _make_output_dir(self, output_root: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = Path(self.plan.model_path).stem[:48].replace(" ", "_")
        path = Path(output_root) / f"{timestamp}_{model_name}_{self.plan.ctx_size}ctx"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def cancel(self) -> None:
        self.requestInterruption()
        if self.runner:
            self.runner.cancel()

    def _llama_cpp_build(self) -> str:
        try:
            proc = subprocess.run(
                [self.bench_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                encoding="utf-8",
                errors="ignore",
            )
            text = (proc.stdout or proc.stderr or "").strip()
            return text.splitlines()[0] if text else "unknown"
        except Exception:
            return "unknown"

    def _emit_log(self, message: str, level: str = "info") -> None:
        self.log.emit(message, level)

    def _rank_best(self) -> Optional[BenchmarkResult]:
        successful = [r for r in self.results if r.status == "success"]
        if not successful:
            return None
        return max(
            successful, key=lambda r: (r.score, r.generation_tok_s, r.prompt_tok_s)
        )

    def _should_early_stop_after_peak(self, latest: BenchmarkResult) -> bool:
        """Улучшенный early stop с rolling window и проверкой тренда.

        Останавливает benchmark если:
        1. Последний успешный результат упал ниже порога от пика, ИЛИ
        2. Большинство последних N успешных результатов ниже порога
        """
        if not bool(getattr(self.plan, "early_stop_on_peak", False)):
            return False
        if latest.status != "success":
            return False

        successful = [r for r in self.results if r.status == "success"]
        min_successes = max(
            3, int(getattr(self.plan, "early_stop_min_successes", 3) or 3)
        )
        if len(successful) < min_successes:
            return False

        # Rolling window последних успешных
        window_size = max(2, min(5, len(successful)))
        window = successful[-window_size:]

        best = max(
            successful, key=lambda r: (r.generation_tok_s, r.prompt_tok_s, r.score)
        )
        if best.candidate_id == latest.candidate_id:
            return False  # Новый пик — не останавливаем

        drop_pct = max(
            0.0, float(getattr(self.plan, "early_stop_drop_pct", 3.0) or 3.0)
        )
        threshold = best.generation_tok_s * (1.0 - drop_pct / 100.0)

        # 1) Последний упал ниже порога
        if latest.generation_tok_s < threshold:
            return True

        # 2) Большинство в rolling window ниже порога
        below_threshold = sum(1 for r in window if r.generation_tok_s < threshold)
        majority = window_size // 2 + 1
        if below_threshold >= majority:
            return True

        return False

    def _write_reports(self) -> None:
        by_id = {c.id: c for c in self.plan.candidates}
        write_plan(self.output_dir, self.plan)
        write_json_report(
            self.output_dir,
            self.plan,
            self.model_info,
            self.results,
            self.best_result,
            llama_cpp_build=self._llama_cpp_build(),
        )
        write_markdown_report(
            self.output_dir, self.plan, self.model_info, self.results, self.best_result
        )
        params = (
            by_id.get(self.best_result.candidate_id).params
            if self.best_result and by_id.get(self.best_result.candidate_id)
            else {}
        )
        write_best(self.output_dir, self.best_result, params)

    def run(self) -> None:
        self.runner = BenchmarkRunner(
            self.bench_exe, self.plan.model_path, self.output_dir
        )
        start_mono = time.monotonic()
        failures = 0
        oom_failures = 0
        total = len(self.plan.candidates)
        self._emit_log(
            f"AutoTune started: {total} candidates, output: {self.output_dir}", "info"
        )

        for index, candidate in enumerate(self.plan.candidates, start=1):
            if self.isInterruptionRequested():
                break
            if time.monotonic() - start_mono > self.plan.time_budget_sec:
                self._emit_log("AutoTune time budget reached", "warn")
                break
            if failures >= self.max_failures or oom_failures >= self.max_oom_failures:
                self._emit_log("AutoTune stopped: too many failed candidates", "warn")
                break

            self.progress.emit(index - 1, total)
            self.run_started.emit(candidate)
            self._emit_log(
                f"[{index}/{total}] {candidate.id}: {candidate.reason}", "bench"
            )

            result = self.runner.run(
                candidate,
                prompt_tokens=self.prompt_tokens,
                generation_tokens=self.generation_tokens,
                timeout_sec=self.per_run_timeout_sec,
                log_callback=lambda msg: self._emit_log(msg, "bench"),
            )
            score_result(result, candidate.params, self.plan.target)
            self.results.append(result)
            self.run_finished.emit(result)

            if result.status != "success":
                failures += 1
                if result.status == "failed_oom":
                    oom_failures += 1
            else:
                failures = 0

            self.best_result = self._rank_best()
            self.progress.emit(index, total)

            if self._should_early_stop_after_peak(result):
                best = self.best_result
                message = "AutoTune early stop: latest successful run is slower than the current peak"
                if best:
                    message += f"; best={best.candidate_id} TG={best.generation_tok_s:.1f} tok/s"
                self._emit_log(message, "info")
                break

        if self.isInterruptionRequested():
            self._emit_log("AutoTune cancelled", "warn")

        self.best_result = self._rank_best()
        self._write_reports()
        self.runner = None
        self.autotune_finished.emit(self.best_result, self.output_dir)
