"""Синхронный запуск одного AutoTune benchmark-кандидата."""

from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.core.benchmark_models import BenchmarkCandidate, BenchmarkMetrics, BenchmarkResult
from src.core.cli_builder import build_benchmark_args_from_params


_OOM_RE = re.compile(r"(out of memory|cuda.*oom|failed to allocate|vk::outofdevicememory|not enough memory)", re.I)
_INVALID_ARGS_RE = re.compile(r"(unknown argument|invalid argument|error:.*argument|usage:)", re.I)
_PP_RE = re.compile(r"\bpp\s*\d*[^\n|]*[|:\s]+([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-|tok/s|t/s)", re.I)
_TG_RE = re.compile(r"\btg\s*\d*[^\n|]*[|:\s]+([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-|tok/s|t/s)", re.I)
_SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens?/s|tok/s|t/s)", re.I)
_LOAD_RE = re.compile(r"load\s+time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|sec)?", re.I)
_VRAM_RE = re.compile(r"(?:VRAM|GPU memory)[^\n:]*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|MB|GB)", re.I)
_RAM_RE = re.compile(r"(?:RAM|system memory)[^\n:]*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|MB|GB)", re.I)


def _unit_to_mib(value: float, unit: str) -> float:
    unit = (unit or "MiB").lower()
    if unit in {"gib", "gb"}:
        return value * 1024.0
    return value


def parse_llama_bench_output(text: str) -> BenchmarkMetrics:
    """Извлекает основные метрики из stdout/stderr llama-bench."""
    metrics = BenchmarkMetrics()
    pp_values = []
    tg_values = []

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        pp_match = _PP_RE.search(line_clean)
        if pp_match:
            pp_values.append(float(pp_match.group(1)))
        elif "pp" in line_clean.lower():
            values = _SPEED_RE.findall(line_clean)
            if values:
                pp_values.append(float(values[-1]))

        tg_match = _TG_RE.search(line_clean)
        if tg_match:
            tg_values.append(float(tg_match.group(1)))
        elif "tg" in line_clean.lower():
            values = _SPEED_RE.findall(line_clean)
            if values:
                tg_values.append(float(values[-1]))

        load_match = _LOAD_RE.search(line_clean)
        if load_match:
            value = float(load_match.group(1))
            unit = (load_match.group(2) or "s").lower()
            metrics.load_time_sec = value / 1000.0 if unit == "ms" else value

        lower_line = line_clean.lower()
        vram_match = _VRAM_RE.search(line_clean)
        if vram_match and "total vram" not in lower_line and ("used" in lower_line or "buffer size" in lower_line):
            metrics.vram_used_mib = max(metrics.vram_used_mib, _unit_to_mib(float(vram_match.group(1)), vram_match.group(2)))

        ram_match = _RAM_RE.search(line_clean)
        if ram_match and "vram" not in lower_line and ("used" in lower_line or "buffer size" in lower_line):
            metrics.ram_used_mib = max(metrics.ram_used_mib, _unit_to_mib(float(ram_match.group(1)), ram_match.group(2)))

    if pp_values:
        metrics.prompt_tok_s = max(pp_values)
    if tg_values:
        metrics.generation_tok_s = max(tg_values)
    return metrics


class BenchmarkRunner:
    """Запускает один llama-bench и пишет лог прогона."""

    def __init__(self, bench_exe: str, model_path: str, output_dir: str):
        self.bench_exe = bench_exe
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._process: Optional[subprocess.Popen] = None

    def cancel(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass

    def run(
        self,
        candidate: BenchmarkCandidate,
        prompt_tokens: int,
        generation_tokens: int,
        timeout_sec: int,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> BenchmarkResult:
        started = datetime.now()
        log_path = self.logs_dir / f"{candidate.id}.log"
        args = build_benchmark_args_from_params(
            self.model_path,
            candidate.params,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        ) or []
        command = [self.bench_exe] + args
        result = BenchmarkResult(
            candidate_id=candidate.id,
            status="running",
            command=command,
            log_path=str(log_path),
            started_at=started.isoformat(timespec="seconds"),
        )

        text = ""
        start_mono = time.monotonic()
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                cwd=os.path.dirname(self.bench_exe) or None,
            )
            try:
                text, _ = self._process.communicate(timeout=max(5, int(timeout_sec)))
            except subprocess.TimeoutExpired:
                self.cancel()
                try:
                    text, _ = self._process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    text, _ = self._process.communicate(timeout=5)
                result.status = "failed_timeout"
                result.error = f"Timeout after {timeout_sec}s"

            result.exit_code = self._process.returncode
        except OSError as exc:
            result.status = "failed_crash"
            result.error = str(exc)
            text = str(exc)
        finally:
            self._process = None

        result.duration_sec = round(time.monotonic() - start_mono, 3)
        result.ended_at = datetime.now().isoformat(timespec="seconds")

        metrics = parse_llama_bench_output(text)
        metrics.prompt_tokens = int(prompt_tokens)
        metrics.generation_tokens = int(generation_tokens)
        result.metrics = metrics
        result.prompt_tok_s = metrics.prompt_tok_s
        result.generation_tok_s = metrics.generation_tok_s
        result.load_time_sec = metrics.load_time_sec
        result.vram_used_mib = metrics.vram_used_mib
        result.ram_used_mib = metrics.ram_used_mib

        if result.status == "running":
            if result.exit_code == 0 and (metrics.prompt_tok_s > 0 or metrics.generation_tok_s > 0):
                result.status = "success"
            elif _OOM_RE.search(text):
                result.status = "failed_oom"
                result.error = "Out of memory"
            elif _INVALID_ARGS_RE.search(text):
                result.status = "failed_invalid_args"
                result.error = "Invalid llama-bench arguments"
            else:
                result.status = "failed_crash"
                result.error = f"Exit code {result.exit_code}"

        log_header = "\n".join(
            [
                f"# {candidate.id}",
                f"status: {result.status}",
                f"reason: {candidate.reason}",
                f"command: {' '.join(command)}",
                "",
            ]
        )
        log_path.write_text(log_header + (text or ""), encoding="utf-8", errors="ignore")

        if log_callback:
            summary = (
                f"{candidate.id}: {result.status}; "
                f"PP={result.prompt_tok_s:.1f} tok/s; TG={result.generation_tok_s:.1f} tok/s"
            )
            if result.error:
                summary += f"; {result.error}"
            log_callback(summary)

        return result
