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
from src.core.benchmark_plan import EARLY_STOP_DROP_PCT_DEFAULT
from src.core.benchmark_scorer import score_result
from src.services.benchmark_runner import BenchmarkRunner

# --- Пороги замены baseline ---------------------------------------------------
# Single-repetition llama-bench has noise. Do not replace the current baseline
# unless the candidate is clearly better (>= 3% score or >= 5% generation),
# otherwise AutoTune can apply a tiny/noisy +1-2% result that feels slower
# in real server use.
MIN_SCORE_GAIN_TO_REPLACE = 0.03
MIN_TG_GAIN_TO_REPLACE = 0.05

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
        # Large GGUFs can spend tens of seconds just loading before TG starts.
        # A too-low timeout produces false failures, so keep a safe floor.
        self.per_run_timeout_sec = max(120, int(per_run_timeout_sec))
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

    def _candidate_by_id(self) -> Dict[str, BenchmarkCandidate]:
        return {c.id: c for c in self.plan.candidates}

    def _candidate_signature(self, candidate: BenchmarkCandidate) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (k, str(v))
                for k, v in candidate.params.items()
                if not str(k).startswith("_")
            )
        )

    def _rank_best(self) -> Optional[BenchmarkResult]:
        successful = [r for r in self.results if r.status == "success"]
        if not successful:
            return None
        by_id = self._candidate_by_id()
        if any((by_id.get(r.candidate_id) and by_id[r.candidate_id].stage == "verify") for r in successful):
            groups: Dict[tuple[tuple[str, str], ...], List[BenchmarkResult]] = {}
            for result in successful:
                candidate = by_id.get(result.candidate_id)
                if not candidate:
                    continue
                groups.setdefault(self._candidate_signature(candidate), []).append(result)

            ranked_groups = []
            for signature, results in groups.items():
                count = len(results)
                avg_score = sum(r.score for r in results) / count
                avg_tg = sum(r.generation_tok_s for r in results) / count
                avg_pp = sum(r.prompt_tok_s for r in results) / count
                representative = next(
                    (
                        r
                        for r in reversed(results)
                        if by_id.get(r.candidate_id)
                        and by_id[r.candidate_id].stage == "verify"
                    ),
                    max(results, key=lambda r: (r.score, r.generation_tok_s)),
                )
                ranked_groups.append(
                    (avg_score, avg_tg, avg_pp, count, representative, signature)
                )

            best_group = max(ranked_groups, key=lambda item: item[:4])
            baseline = next((r for r in successful if r.candidate_id == "run_001"), None)
            if baseline:
                baseline_candidate = by_id.get(baseline.candidate_id)
                baseline_signature = (
                    self._candidate_signature(baseline_candidate)
                    if baseline_candidate
                    else None
                )
                baseline_group = next(
                    (g for g in ranked_groups if g[5] == baseline_signature),
                    None,
                )
                if baseline_group and best_group[5] != baseline_signature:
                    score_gain = (best_group[0] - baseline_group[0]) / max(
                        baseline_group[0], 1.0
                    )
                    tg_gain = (best_group[1] - baseline_group[1]) / max(
                        baseline_group[1], 1.0
                    )
                    if (
                        score_gain < MIN_SCORE_GAIN_TO_REPLACE
                        and tg_gain < MIN_TG_GAIN_TO_REPLACE
                    ):
                        return baseline_group[4]
            return best_group[4]

        best = max(
            successful, key=lambda r: (r.score, r.generation_tok_s, r.prompt_tok_s)
        )
        baseline = next((r for r in successful if r.candidate_id == "run_001"), None)
        if not baseline or best.candidate_id == baseline.candidate_id:
            return best

        # Single-repetition llama-bench has noise. Do not replace the current
        # baseline unless the candidate is clearly better, otherwise AutoTune can
        # apply a tiny/noisy +1-2% result that feels slower in real server use.
        score_gain = (best.score - baseline.score) / max(baseline.score, 1.0)
        tg_gain = (best.generation_tok_s - baseline.generation_tok_s) / max(
            baseline.generation_tok_s, 1.0
        )
        if (
            score_gain < MIN_SCORE_GAIN_TO_REPLACE
            and tg_gain < MIN_TG_GAIN_TO_REPLACE
        ):
            return baseline
        return best

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
            0.0,
            float(
                getattr(
                    self.plan, "early_stop_drop_pct", EARLY_STOP_DROP_PCT_DEFAULT
                )
                or EARLY_STOP_DROP_PCT_DEFAULT
            ),
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

    def _run_candidate(
        self,
        candidate: BenchmarkCandidate,
        index: int,
        total: int,
    ) -> BenchmarkResult:
        self.progress.emit(index - 1, total)
        self.run_started.emit(candidate)
        self._emit_log(f"[{index}/{total}] {candidate.id}: {candidate.reason}", "bench")

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
        self.best_result = self._rank_best()
        self.progress.emit(index, total)
        return result

    def _successful_ranked_results(self) -> List[BenchmarkResult]:
        successful = [r for r in self.results if r.status == "success"]
        return sorted(
            successful,
            key=lambda r: (r.score, r.generation_tok_s, r.prompt_tok_s),
            reverse=True,
        )

    def _verification_candidates(self, count: int) -> List[BenchmarkCandidate]:
        by_id = self._candidate_by_id()
        chosen: List[BenchmarkCandidate] = []
        seen_params: set[tuple[tuple[str, str], ...]] = set()
        for result in self._successful_ranked_results():
            source = by_id.get(result.candidate_id)
            if not source or source.stage == "verify":
                continue
            signature = self._candidate_signature(source)
            if signature in seen_params:
                continue
            seen_params.add(signature)
            verify_id = f"verify_{len(chosen) + 1:03d}_{source.id}"
            chosen.append(
                BenchmarkCandidate(
                    verify_id,
                    dict(source.params),
                    f"verify {source.id}: {source.reason}",
                    "verify",
                )
            )
            if len(chosen) >= count:
                break
        return chosen

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

        executed = 0
        for index, candidate in enumerate(list(self.plan.candidates), start=1):
            if self.isInterruptionRequested():
                break
            if time.monotonic() - start_mono > self.plan.time_budget_sec:
                self._emit_log("AutoTune time budget reached", "warn")
                break
            if failures >= self.max_failures or oom_failures >= self.max_oom_failures:
                self._emit_log("AutoTune stopped: too many failed candidates", "warn")
                break

            result = self._run_candidate(candidate, index, total)
            executed = index

            if result.status != "success":
                failures += 1
                if result.status == "failed_oom":
                    oom_failures += 1
            else:
                failures = 0

            if self._should_early_stop_after_peak(result):
                best = self.best_result
                message = "AutoTune early stop: latest successful run is slower than the current peak"
                if best:
                    message += f"; best={best.candidate_id} TG={best.generation_tok_s:.1f} tok/s"
                self._emit_log(message, "info")
                break

        if not self.isInterruptionRequested():
            repeat_count = int(getattr(self.plan, "repeat_top", 1) or 1)
            if repeat_count > 1:
                verify_candidates = self._verification_candidates(repeat_count)
                if verify_candidates:
                    self.plan.candidates.extend(verify_candidates)
                    total = executed + len(verify_candidates)
                    self._emit_log(
                        f"AutoTune verification: repeating top {len(verify_candidates)} candidate(s)",
                        "info",
                    )
                    for candidate in verify_candidates:
                        if self.isInterruptionRequested():
                            break
                        if time.monotonic() - start_mono > self.plan.time_budget_sec:
                            self._emit_log("AutoTune time budget reached", "warn")
                            break
                        executed += 1
                        self._run_candidate(candidate, executed, total)

        if self.isInterruptionRequested():
            self._emit_log("AutoTune cancelled", "warn")

        self.best_result = self._rank_best()
        self._write_reports()
        self.runner = None
        self.autotune_finished.emit(self.best_result, self.output_dir)
