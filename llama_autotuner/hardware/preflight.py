from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from llama_autotuner.hardware.nvidia import NvidiaSmiBackend
from llama_autotuner.models import GpuSnapshot


@dataclass(slots=True)
class PreflightResult:
    state: str
    baseline_used_mb: int
    baseline_free_mb: int
    median_util: float
    samples: list[GpuSnapshot]
    message: str


def run_preflight(gpu: NvidiaSmiBackend, total_vram_mb: int, samples: int = 5, interval: float = 0.5) -> PreflightResult:
    snaps = []
    for i in range(samples):
        snaps.append(gpu.snapshot())
        if i + 1 < samples:
            time.sleep(interval)
    used = int(statistics.median(s.used_mb for s in snaps))
    free = int(statistics.median(s.free_mb for s in snaps))
    util = statistics.median(s.util_percent for s in snaps)
    pct = 100.0 * used / max(1, total_vram_mb)
    if pct < 8 and util < 5:
        state = "CLEAN"
    elif pct <= 15 and util <= 15:
        state = "MODERATE"
    else:
        state = "BUSY"
    message = (
        "Do not start heavy GPU workloads while autotuning. Browser, YouTube and IDE may stay open "
        "if their GPU usage remains stable. Games, other LLM servers, renderers and CUDA workloads should be closed."
    )
    return PreflightResult(state, used, free, float(util), snaps, message)
