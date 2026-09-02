from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ModelKind(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    PASS = "PASS"
    PASS_DEGRADED = "PASS_DEGRADED"
    EARLY_REJECT = "EARLY_REJECT"
    FAILED = "FAILED"
    INVALID_ENVIRONMENT = "INVALID_ENVIRONMENT"
    FATAL = "FATAL"


@dataclass(slots=True)
class HardwareInfo:
    gpu_name: str
    gpu_count: int
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    gpu_util_percent: float
    gpu_temp_c: float | None
    driver_version: str | None
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_total_mb: int
    ram_available_mb: int
    os_name: str


@dataclass(slots=True)
class GpuSnapshot:
    timestamp: float
    used_mb: int
    free_mb: int
    util_percent: float
    temperature_c: float | None = None
    power_w: float | None = None
    graphics_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None


@dataclass(slots=True)
class ModelInfo:
    path: str
    size_bytes: int
    architecture: str | None
    kind: ModelKind
    block_count: int | None
    context_length: int | None
    expert_count: int | None
    expert_used_count: int | None
    has_mtp: bool
    has_vision_hint: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    tensor_count: int = 0
    tensor_data_bytes: int = 0
    tensor_layout_complete: bool = False
    tensor_alignment: int = 32
    block_tensor_bytes: dict[int, int] = field(default_factory=dict)
    block_expert_bytes: dict[int, int] = field(default_factory=dict)
    non_block_tensor_bytes: int = 0
    tensor_type_histogram: dict[int, int] = field(default_factory=dict)
    # Some GGUFs (notably Qwen3.5/3.6/3.8 with integrated NextN/MTP) expose a
    # stored block_count that includes auxiliary draft blocks. Keep the parsed
    # main/draft split explicitly so Dense -ngl search does not treat every
    # stored block as a normal target-model transformer block.
    main_block_count: int | None = None
    mtp_block_count: int = 0
    # Split GGUFs are launched through shard 00001, but sizing, startup budgets and
    # placement analysis must describe the complete logical model rather than that
    # one file.  Defaults preserve positional construction used by older callers.
    split_count: int = 1
    split_parts_found: int = 1
    # Some architectures (Gemma3n-style per-layer embeddings) keep specific huge lookup
    # tensors CPU-resident by design in llama.cpp, regardless of -ngl/-ncmoe. This is the
    # subset of non_block_tensor_bytes that must never be charged against the GPU budget.
    cpu_resident_tensor_bytes: int = 0


@dataclass(slots=True)
class Candidate:
    ctx: int
    ngl: str | int = "all"
    ncmoe: int | None = None
    batch: int = 512
    ubatch: int = 256
    threads: int = 8
    threads_batch: int = 8
    kv_k: str = "q8_0"
    kv_v: str = "q8_0"
    mtp: bool = False
    mtp_n_max: int = 8
    mtp_p_min: float = 0.8
    draft_kv_k: str | None = None
    draft_kv_v: str | None = None
    vision: bool = False
    mmproj: str | None = None
    load_mode: str = "none"
    extra_args: list[str] = field(default_factory=list)

    def key(self) -> str:
        return (
            f"ctx={self.ctx};ngl={self.ngl};ncmoe={self.ncmoe};b={self.batch};ub={self.ubatch};"
            f"t={self.threads};tb={self.threads_batch};kv={self.kv_k}/{self.kv_v};"
            f"vision={int(self.vision)};mtp={int(self.mtp)}:{self.mtp_n_max}:{self.mtp_p_min}"
        )

    def short(self) -> str:
        placement = f"ncmoe={self.ncmoe}" if self.ncmoe is not None else f"ngl={self.ngl}"
        mtp = f"MTP={self.mtp_n_max}/{self.mtp_p_min:g}" if self.mtp else "MTP=off"
        vision = "Vision=on" if self.vision else "Vision=off"
        return (
            f"ctx={self.ctx} {placement} b/ub={self.batch}/{self.ubatch} "
            f"t/tb={self.threads}/{self.threads_batch} KV={self.kv_k}/{self.kv_v} {vision} {mtp}"
        )


@dataclass(slots=True)
class BenchmarkMetrics:
    benchmark_kind: str | None = None  # quick | full | validation
    prompt_tokens: int = 0  # tokens actually processed in this request (timings.prompt_n)
    cache_tokens: int = 0   # tokens reused from prompt cache (timings.cache_n)
    prompt_total_tokens: int = 0  # full request prompt size (usage.prompt_tokens or prompt_n + cache_n)
    pp_tps: float | None = None
    generated_tokens: int = 0
    tg_tps: float | None = None
    draft_n: int = 0
    draft_accepted: int = 0
    acceptance: float | None = None
    startup_seconds: float | None = None
    vram_peak_mb: int | None = None
    vram_free_min_mb: int | None = None
    vram_operating_class: str | None = None
    vram_hard_floor_mb: int | None = None
    vram_tight_floor_mb: int | None = None
    vram_operational_floor_mb: int | None = None
    vram_preferred_reserve_mb: int | None = None
    ram_peak_mb: int | None = None
    early_pp_tps: float | None = None
    final_pp_tps: float | None = None
    variance_pct: float | None = None
    long_context_tokens: int = 0
    long_context_pp_tps: float | None = None
    long_context_passed: bool = False
    # Final robustness metrics. These are intentionally separate from the short FULL decode sample:
    # an MTP candidate can have a very high peak TG yet perform poorly on less predictable text.
    stability_samples: int = 0
    stability_tg_median: float | None = None
    stability_tg_p10: float | None = None
    stability_tg_p90: float | None = None
    stability_tg_min: float | None = None
    stability_tg_max: float | None = None
    stability_acceptance_median: float | None = None
    stability_mean_draft_len_median: float | None = None
    stability_mean_draft_len_p10: float | None = None
    stability_tg_mean_len_corr: float | None = None
    stability_tg_acceptance_corr: float | None = None
    stability_variation_pct: float | None = None
    stability_passed: bool = False
    context_tg_first: float | None = None
    context_tg_last: float | None = None
    context_tg_ratio: float | None = None
    stability_workloads: list[dict[str, Any]] = field(default_factory=list)
    context_staircase: list[dict[str, Any]] = field(default_factory=list)
    # Real multimodal smoke benchmark. A Vision candidate is not validated merely by loading mmproj.
    vision_test_passed: bool | None = None
    vision_latency_seconds: float | None = None
    vision_answer: str | None = None
    vision_prompt_tokens: int = 0
    vision_generated_tokens: int = 0
    vision_transport: str | None = None
    vision_error: str | None = None


@dataclass(slots=True)
class CandidateResult:
    candidate: Candidate
    status: RunStatus
    reason: str
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    score: float | None = None
    logs_tail: str = ""
    phase: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(slots=True)
class LaunchProfile:
    name: str
    candidate: Candidate
    result: CandidateResult
    confidence: str
    rationale: str
    command: list[str]
    provisional: bool = False
