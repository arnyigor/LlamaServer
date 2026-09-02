from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llama_autotuner.models import ModelKind


class VramOperatingClass(str, Enum):
    """Runtime headroom classes for measured candidates.

    ``REJECT`` is the only hard launch constraint. ``FRAGILE`` is runnable but too close to
    the hard floor to recommend after a strong workload. ``TIGHT`` is an explicitly supported
    hysteresis band: a FULL/FINAL-proven candidate may be recommended, but the optimizer must not
    grow ubatch/MTP state from it. ``OPERATIONAL`` and ``SAFE`` have progressively more reserve.
    """

    UNKNOWN = "UNKNOWN"
    REJECT = "REJECT"
    FRAGILE = "FRAGILE"
    TIGHT = "TIGHT"
    OPERATIONAL = "OPERATIONAL"
    SAFE = "SAFE"


@dataclass(frozen=True, slots=True)
class VramThresholds:
    hard_floor_mb: int
    tight_floor_mb: int
    operational_floor_mb: int
    preferred_reserve_mb: int

    def classify(self, free_mb: int | None) -> VramOperatingClass:
        if free_mb is None:
            return VramOperatingClass.UNKNOWN
        free = int(free_mb)
        if free < self.hard_floor_mb:
            return VramOperatingClass.REJECT
        if free < self.tight_floor_mb:
            return VramOperatingClass.FRAGILE
        if free < self.operational_floor_mb:
            return VramOperatingClass.TIGHT
        if free < self.preferred_reserve_mb:
            return VramOperatingClass.OPERATIONAL
        return VramOperatingClass.SAFE

    def to_dict(self, free_mb: int | None = None) -> dict:
        operating_class = self.classify(free_mb)
        return {
            "class": operating_class.value,
            "free_min_mb": free_mb,
            "hard_floor_mb": self.hard_floor_mb,
            "tight_floor_mb": self.tight_floor_mb,
            "operational_floor_mb": self.operational_floor_mb,
            "preferred_reserve_mb": self.preferred_reserve_mb,
        }


def vram_thresholds(*, absolute_floor_mb: int, preferred_reserve_mb: int,
                    search_mode: str = "normal", model_kind: ModelKind | str = ModelKind.DENSE,
                    vision: bool = False) -> VramThresholds:
    """Return one consistent set of search/recommendation headroom thresholds.

    The mode changes how aggressively QUICK may recommend a result, but DEEP exploration does not
    weaken the final recommendation floor. MoE gets a narrower operating guard because moving one
    routed-expert layer to CPU is a cheap recovery axis; its TIGHT band remains explicit.
    """
    hard = max(128, int(absolute_floor_mb))
    preferred = max(hard, int(preferred_reserve_mb))
    try:
        kind = model_kind if isinstance(model_kind, ModelKind) else ModelKind(str(model_kind))
    except ValueError:
        kind = ModelKind.UNKNOWN

    if kind == ModelKind.MOE:
        operational_extra = 160 if search_mode == "quick" else 192
        tight_base_extra = 128
    else:
        operational_extra = 192 if search_mode == "quick" else 256
        tight_base_extra = 128

    if vision:
        operational_extra += 64
        tight_base_extra += 32

    operational = min(preferred, hard + operational_extra)
    # A strong workload may vary by a few dozen MiB between samples. The 64 MiB hysteresis band
    # prevents a 36 MiB miss from discarding a 25-30% faster candidate, while FRAGILE still rejects
    # old cases such as 328 MiB against a 300 MiB hard floor.
    tight = min(operational, max(hard + tight_base_extra, operational - 64))
    return VramThresholds(
        hard_floor_mb=hard,
        tight_floor_mb=tight,
        operational_floor_mb=operational,
        preferred_reserve_mb=preferred,
    )
