from __future__ import annotations

from dataclasses import dataclass


# This order describes KV-cache numerical precision only.  It deliberately says
# nothing about the quantization of the model weights in the selected GGUF.
_KV_TYPE_LEVEL: dict[str, int] = {
    "q4_0": 10,
    "q4_1": 11,
    "q5_0": 20,
    "q5_1": 21,
    "q8_0": 30,
    "f16": 40,
    "bf16": 40,
    "f32": 50,
}


@dataclass(frozen=True, slots=True)
class KvPrecision:
    k: str
    v: str
    tier: str
    label: str
    min_level: int
    combined_level: int

    @property
    def runtime_risk(self) -> str:
        """Return the policy class used by automatic runtime selection.

        This is deliberately not a claim about semantic equivalence.  It only
        separates the empirically low-risk Q8 operating tier from Q4/mixed
        caches whose speed and task quality are substantially more
        architecture/context dependent.
        """
        if self.tier == "FP16_OR_BETTER":
            return "REFERENCE"
        if self.tier == "Q8":
            return "LOW"
        if self.tier in {"Q4", "MIXED"}:
            return "CONTEXT_SENSITIVE"
        return "UNKNOWN"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "k": self.k,
            "v": self.v,
            "tier": self.tier,
            "label": self.label,
            "min_level": self.min_level,
            "combined_level": self.combined_level,
            "runtime_risk": self.runtime_risk,
        }


def kv_type_level(value: str) -> int:
    """Return an ordering value for a llama.cpp KV-cache storage type."""
    key = str(value or "").strip().lower()
    if key in _KV_TYPE_LEVEL:
        return _KV_TYPE_LEVEL[key]
    if key.startswith(("f", "bf")):
        return 35
    if key.startswith("q8"):
        return 30
    if key.startswith("q6"):
        return 25
    if key.startswith("q5"):
        return 20
    if key.startswith("q4"):
        return 10
    return 0


def kv_precision(k: str, v: str) -> KvPrecision:
    """Describe the attention/KV-cache precision of a K/V pair.

    The weaker side is primary because an asymmetric pair is not equivalent to
    keeping both caches at the stronger type.  The sum is a deterministic
    secondary ordering for mixed pairs.
    """
    kk = str(k or "unknown").lower()
    vv = str(v or "unknown").lower()
    kl = kv_type_level(kk)
    vl = kv_type_level(vv)
    floor = min(kl, vl)
    combined = kl + vl

    if kl >= 40 and vl >= 40:
        tier = "FP16_OR_BETTER"
        label = "FP16/BF16 (maximum KV-cache precision)"
    elif kl >= 30 and vl >= 30:
        tier = "Q8"
        label = "Q8 (high KV-cache precision)"
    elif kl >= 10 and vl >= 10 and (kl < 30 or vl < 30):
        if kl != vl:
            tier = "MIXED"
            label = f"Mixed {kk}/{vv} KV-cache precision"
        else:
            tier = "Q4"
            label = "Q4 (memory-saving KV cache)"
    else:
        tier = "CUSTOM"
        label = f"Custom {kk}/{vv} KV-cache precision"
    return KvPrecision(kk, vv, tier, label, floor, combined)


def kv_precision_key(k: str, v: str) -> tuple[int, int]:
    info = kv_precision(k, v)
    return info.min_level, info.combined_level


def balanced_context_gain_required(higher_k: str, higher_v: str,
                                   lower_k: str, lower_v: str) -> float:
    """Context multiplier that justifies a lower KV tier in BALANCED mode.

    Q8 is deliberately treated as the low-risk runtime sweet spot: in BALANCED mode
    it may replace FP16 at equal context when it preserves measured performance and
    improves residency/headroom.  This is not a universal quality proof; MAX_KV_PRECISION
    and ``priority=quality`` still retain FP16/BF16.

    Entering a Q4 or mixed-Q4 family is context/task/architecture sensitive and therefore
    needs a much larger 75% context gain.  The optimizer additionally requires a long-filled
    runtime scout before Q4 can become the automatic winner.
    """
    higher = kv_precision(higher_k, higher_v)
    lower = kv_precision(lower_k, lower_v)
    if lower.min_level >= higher.min_level:
        return 1.0
    if lower.min_level >= kv_type_level("q8_0"):
        return 1.0
    return 1.75


def kv_requires_long_context_probe(k: str, v: str) -> bool:
    """Whether automatic promotion needs an occupied-context throughput probe."""
    return kv_precision(k, v).runtime_risk == "CONTEXT_SENSITIVE"


def kv_context_probe_tokens(k: str, v: str, context_size: int, *,
                            workload_profile: str = "agent",
                            search_mode: str = "normal") -> int | None:
    """Return an explicit long-filled scout target for context-sensitive KV.

    The generic solution scout is intentionally capped at 16K.  That is too short
    to reveal kernels whose dequantization/attention cost grows materially at high
    cache occupancy.  For Q4 or mixed-Q4 in a long-context workload, NORMAL fills
    half of the configured slot, capped at 110K so that 64K+ and 100K-class
    runtime cliffs are not hidden by a 6K/12K scout; DEEP may fill 75% up to 160K.  Other
    tiers keep the ordinary bounded context scout.
    """
    context_size = max(1, int(context_size))
    if workload_profile != "long-context" or search_mode == "quick":
        return None
    if not kv_requires_long_context_probe(k, v) or context_size < 32_768:
        return None
    fraction = 0.75 if search_mode == "deep" else 0.50
    cap = 160_000 if search_mode == "deep" else 110_000
    target = min(cap, max(16_000, int(context_size * fraction)))
    return min(target, max(2_048, context_size - 1_024))


def kv_degradation_ladder(k: str, v: str) -> list[tuple[str, str, str]]:
    """Return explicit lower-precision KV alternatives.

    FP16, Q8 and Q4 are the three user-facing levels.  Q8/Q4 remains available
    as an intermediate measured memory point because some placements cross a
    useful VRAM boundary there, but it is labelled as mixed rather than as a
    fourth quality tier.
    """
    start = (str(k).lower(), str(v).lower())
    candidates: list[tuple[str, str, str]] = []
    if kv_precision_key(*start) > kv_precision_key("q8_0", "q8_0"):
        candidates.append((
            "q8_0", "q8_0",
            f"K/V cache {start[0]}/{start[1]}→q8_0/q8_0 (FP16→Q8 KV-cache precision)",
        ))
    if kv_precision_key(*start) > kv_precision_key("q8_0", "q4_0"):
        candidates.append((
            "q8_0", "q4_0",
            f"K/V cache {start[0]}/{start[1]}→q8_0/q4_0 (mixed intermediate precision)",
        ))
    if kv_precision_key(*start) > kv_precision_key("q4_0", "q4_0"):
        candidates.append((
            "q4_0", "q4_0",
            f"K/V cache {start[0]}/{start[1]}→q4_0/q4_0 (Q4 memory-saving precision)",
        ))

    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = {start}
    for kk, vv, note in candidates:
        if (kk, vv) not in seen:
            seen.add((kk, vv))
            out.append((kk, vv, note))
    return out
