"""Модели данных для AutoTune benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


BENCHMARK_STATUSES = (
    "pending",
    "running",
    "success",
    "failed_oom",
    "failed_crash",
    "failed_timeout",
    "failed_invalid_args",
    "failed_server_not_ready",
    "skipped",
    "cancelled",
)


@dataclass
class BenchmarkCandidate:
    id: str
    params: Dict[str, Any]
    reason: str
    stage: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    prompt_tok_s: float = 0.0
    generation_tok_s: float = 0.0
    load_time_sec: float = 0.0
    vram_used_mib: float = 0.0
    ram_used_mib: float = 0.0
    vram_free_mib: float = 0.0
    ram_free_mib: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    candidate_id: str
    status: str = "pending"
    score: float = 0.0
    prompt_tok_s: float = 0.0
    generation_tok_s: float = 0.0
    load_time_sec: float = 0.0
    vram_used_mib: float = 0.0
    ram_used_mib: float = 0.0
    error: str = ""
    command: List[str] = field(default_factory=list)
    log_path: str = ""
    exit_code: Optional[int] = None
    duration_sec: float = 0.0
    started_at: str = ""
    ended_at: str = ""
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class AutoTunePlan:
    model_path: str
    ctx_size: int
    mode: str
    target: str
    engine: str
    time_budget_sec: int
    max_runs: int
    repeat_top: int
    candidates: List[BenchmarkCandidate]
    early_stop_on_peak: bool = False
    early_stop_min_successes: int = 3
    early_stop_drop_pct: float = 3.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "ctx_size": self.ctx_size,
            "mode": self.mode,
            "target": self.target,
            "engine": self.engine,
            "time_budget_sec": self.time_budget_sec,
            "max_runs": self.max_runs,
            "repeat_top": self.repeat_top,
            "early_stop_on_peak": self.early_stop_on_peak,
            "early_stop_min_successes": self.early_stop_min_successes,
            "early_stop_drop_pct": self.early_stop_drop_pct,
            "constraints": self.constraints,
            "created_at": self.created_at,
            "candidates": [c.to_dict() for c in self.candidates],
        }
