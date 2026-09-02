from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from llama_autotuner.models import CandidateResult, RunStatus
from llama_autotuner.tuning.kv import kv_precision_key


@dataclass(frozen=True, slots=True)
class NoisePolicy:
    """Material-difference thresholds used by the search scheduler.

    Local llama.cpp measurements are deterministic enough to spot large wins, but small differences
    are not actionable once OS/GPU clocks, prompt shape and speculative acceptance are considered.
    Confirmed measurements inside 5% are noise.  Short SCOUT/PROBE measurements use a
    conservative 10% promotion threshold; the 5-10% band must earn stronger confirmation.
    """

    decode_rel: float = 0.05
    decode_probe_rel: float = 0.10
    decode_abs_tps: float = 3.0
    prefill_rel: float = 0.10
    prefill_abs_tps: float = 120.0
    latency_rel: float = 0.08
    vram_abs_mb: int = 128


def calibrate_noise_policy(
    results: list[CandidateResult],
    base: NoisePolicy | None = None,
) -> tuple[NoisePolicy, dict]:
    """Widen decision bands when this session proves that the workload is noisy.

    The configured 5/10% bands remain *minimum* guards.  They cannot safely be
    estimated before any repeated/robust measurement exists, so early search keeps
    the configured values.  FULL/FINAL evidence may only widen them; it never makes
    the search more aggressive than the user's policy.

    Two independent signals are used:

    * heterogeneous FINAL stability samples (median/p10 and half-spread), and
    * repeated strong measurements of the exact same launch command.

    Context-staircase slowdown is intentionally excluded: it is a real scaling
    effect, not measurement noise.
    """
    base = base or NoisePolicy()
    decode_uncertainty = 0.0
    prefill_uncertainty = 0.0
    sources: list[dict] = []

    strong_by_key: dict[str, list[CandidateResult]] = {}
    for result in results:
        if result.status not in {RunStatus.PASS, RunStatus.PASS_DEGRADED}:
            continue
        metrics = result.metrics
        if metrics.stability_tg_median and metrics.stability_tg_median > 0:
            med = float(metrics.stability_tg_median)
            p10 = float(metrics.stability_tg_p10 or med)
            p90 = float(metrics.stability_tg_p90 or med)
            downside = max(0.0, (med - p10) / med)
            half_spread = max(0.0, (p90 - p10) / (2.0 * med))
            observed = max(downside, half_spread)
            decode_uncertainty = max(decode_uncertainty, observed)
            sources.append({
                "kind": "stability",
                "candidate": result.candidate.key(),
                "samples": int(metrics.stability_samples or 0),
                "decode_uncertainty_rel": observed,
            })
        if metrics.benchmark_kind in {"full", "validation"}:
            strong_by_key.setdefault(result.candidate.key(), []).append(result)

    for key, group in strong_by_key.items():
        # Two strong runs are enough to expose session drift, but use half-range so
        # one pair does not double-count the same deviation on both sides.
        tg_values = [float(r.metrics.tg_tps) for r in group if r.metrics.tg_tps and r.metrics.tg_tps > 0]
        pp_values = [float(r.metrics.pp_tps) for r in group if r.metrics.pp_tps and r.metrics.pp_tps > 0]
        if len(tg_values) >= 2:
            center = float(median(tg_values))
            observed = (max(tg_values) - min(tg_values)) / (2.0 * center) if center > 0 else 0.0
            decode_uncertainty = max(decode_uncertainty, observed)
            sources.append({
                "kind": "repeat-strong",
                "candidate": key,
                "samples": len(tg_values),
                "decode_uncertainty_rel": observed,
            })
        if len(pp_values) >= 2:
            center = float(median(pp_values))
            observed = (max(pp_values) - min(pp_values)) / (2.0 * center) if center > 0 else 0.0
            prefill_uncertainty = max(prefill_uncertainty, observed)

    # Caps prevent a single pathological run from disabling discrimination for the
    # remainder of the session.  Explicit user thresholds above the caps are kept.
    confirmed = max(base.decode_rel, min(0.20, decode_uncertainty))
    probe = max(base.decode_probe_rel, min(0.30, confirmed + 0.05))
    prefill = max(base.prefill_rel, min(0.25, prefill_uncertainty))
    latency = max(base.latency_rel, min(0.25, confirmed * 1.25))
    policy = NoisePolicy(
        decode_rel=confirmed,
        decode_probe_rel=probe,
        decode_abs_tps=base.decode_abs_tps,
        prefill_rel=prefill,
        prefill_abs_tps=base.prefill_abs_tps,
        latency_rel=latency,
        vram_abs_mb=base.vram_abs_mb,
    )
    return policy, {
        "calibrated": policy != base,
        "decode_uncertainty_rel": decode_uncertainty,
        "prefill_uncertainty_rel": prefill_uncertainty,
        "base_decode_rel": base.decode_rel,
        "effective_decode_rel": policy.decode_rel,
        "base_decode_probe_rel": base.decode_probe_rel,
        "effective_decode_probe_rel": policy.decode_probe_rel,
        "base_prefill_rel": base.prefill_rel,
        "effective_prefill_rel": policy.prefill_rel,
        "sources": sources,
    }


@dataclass(frozen=True, slots=True)
class WorkloadShape:
    prompt_tokens: int
    generated_tokens: int
    context_prefill_weight: float
    context_decode_weight: float


WORKLOAD_SHAPES: dict[str, WorkloadShape] = {
    # Chat is mostly short-context generation. Long-context staircase samples are only a weak signal.
    "chat": WorkloadShape(prompt_tokens=4_000, generated_tokens=512, context_prefill_weight=0.20, context_decode_weight=0.15),
    # Coding/agent loops repeatedly ingest tool output/files and generate while the slot is already populated.
    "agent": WorkloadShape(prompt_tokens=12_000, generated_tokens=768, context_prefill_weight=0.60, context_decode_weight=0.70),
    # Long-context usage is dominated by both fill speed and decode speed near the large-context end.
    "long-context": WorkloadShape(prompt_tokens=32_000, generated_tokens=512, context_prefill_weight=0.85, context_decode_weight=0.90),
}


def robust_tg(r: CandidateResult) -> float:
    """Decode throughput with lucky speculative peaks discounted."""
    m = r.metrics
    if m.stability_tg_median is not None:
        p10 = m.stability_tg_p10 if m.stability_tg_p10 is not None else m.stability_tg_median
        return .65 * m.stability_tg_median + .35 * p10
    return m.tg_tps or 0.0




def context_decode_tps(r: CandidateResult, profile: str = "agent") -> float:
    """Decode throughput while the context is populated, with later staircase stages weighted more.

    Speculative decoding is especially content/context sensitive. A short FULL sample can therefore
    overstate real agent speed. We aggregate staircase TG harmonically (elapsed-time semantics) and
    weight later/larger contexts more heavily.
    """
    rows = [row for row in (r.metrics.context_staircase or []) if float(row.get("tg_tps") or 0.0) > 0]
    if not rows:
        return robust_tg(r)

    weighted = 0.0
    seconds_per_token = 0.0
    for idx, row in enumerate(rows):
        tg = float(row.get("tg_tps") or 0.0)
        total_ctx = float(row.get("prompt_tokens") or row.get("total_prompt_tokens") or (idx + 1))
        # sqrt(context) prevents the last stage from completely drowning out all earlier stages,
        # while still making 24K/48K materially more important than 4K/8K.
        weight = max(1.0, total_ctx ** 0.5)
        weighted += weight
        seconds_per_token += weight / tg
    return weighted / seconds_per_token if weighted > 0 and seconds_per_token > 0 else robust_tg(r)


def profile_decode_tps(r: CandidateResult, profile: str = "agent") -> float:
    """Generation throughput appropriate for the selected workload profile.

    Before FINAL validation this falls back to the robust/short TG. Afterwards it blends robust
    heterogeneous decode with context-populated decode by elapsed time, so profile ranking is
    automatically re-grounded in the stronger final measurements.
    """
    shape = WORKLOAD_SHAPES.get(profile, WORKLOAD_SHAPES["agent"])
    short = robust_tg(r)
    contextual = context_decode_tps(r, profile)
    if short <= 0:
        return contextual
    if contextual <= 0 or not r.metrics.context_staircase:
        return short
    w = min(1.0, max(0.0, shape.context_decode_weight))
    denom = (1.0 - w) / short + w / contextual
    return 1.0 / denom if denom > 0 else 0.0


def context_fill_pp(r: CandidateResult) -> float:
    """Effective prefill throughput as the cached context grows.

    The context staircase processes deltas at successively larger total contexts. Combining those
    deltas by elapsed time (harmonic aggregation) gives a much more useful 'fill a large context'
    number than simply averaging PP samples.
    """
    rows = r.metrics.context_staircase or []
    tokens = 0.0
    seconds = 0.0
    for row in rows:
        pp = float(row.get("pp_tps") or 0.0)
        processed = float(row.get("processed_prompt_tokens") or row.get("prompt_tokens") or 0.0)
        if pp > 0 and processed > 0:
            tokens += processed
            seconds += processed / pp
    if tokens > 0 and seconds > 0:
        return tokens / seconds
    if r.metrics.long_context_pp_tps:
        return float(r.metrics.long_context_pp_tps)
    return float(r.metrics.pp_tps or 0.0)


def profile_prefill_tps(r: CandidateResult, profile: str = "agent") -> float:
    """Blend small-prompt PP and large-context PP by *time*, not arithmetic score."""
    shape = WORKLOAD_SHAPES.get(profile, WORKLOAD_SHAPES["agent"])
    short = float(r.metrics.pp_tps or 0.0)
    large = context_fill_pp(r)
    if short <= 0:
        return large
    if large <= 0:
        return short
    w = min(1.0, max(0.0, shape.context_prefill_weight))
    # Harmonic blend: for equal token shares, elapsed times add.
    denom = (1.0 - w) / short + w / large
    return 1.0 / denom if denom > 0 else 0.0


def workload_latency_seconds(r: CandidateResult, profile: str = "agent") -> float:
    """Estimated prefill + generation time for a representative interaction.

    This is intentionally dimensional (seconds), unlike an arbitrary PP/TG weighted sum.
    """
    shape = WORKLOAD_SHAPES.get(profile, WORKLOAD_SHAPES["agent"])
    pp = profile_prefill_tps(r, profile)
    tg = profile_decode_tps(r, profile)
    if pp <= 0 or tg <= 0:
        return float("inf")
    return shape.prompt_tokens / pp + shape.generated_tokens / tg


def decode_speed_envelope(r: CandidateResult, profile: str = "agent") -> dict:
    """Return the measured decode operating envelope for reports and JSON consumers.

    A single ``tg_tps`` value is only the short FULL/validation sample.  It is not a useful
    promise once the KV cache is populated, and for MTP it may also be a lucky speculative peak.
    Keep the short robustness distribution, the context curve, and the workload-weighted value
    separate while also exposing one honest validated floor/ceiling.
    """
    m = r.metrics
    curve: list[dict] = []
    for row in m.context_staircase or []:
        tg = float(row.get("tg_tps") or 0.0)
        if tg <= 0:
            continue
        tokens = int(row.get("prompt_tokens") or row.get("total_prompt_tokens")
                     or row.get("target_tokens") or 0)
        curve.append({
            "tokens": tokens,
            "target_tokens": int(row.get("target_tokens") or tokens),
            "tg_tps": tg,
        })

    short_median = m.stability_tg_median
    if short_median is None:
        short_median = m.tg_tps
    short_p10 = m.stability_tg_p10
    if short_p10 is None:
        short_p10 = short_median

    short_values = [
        float(x) for x in (
            m.tg_tps, m.stability_tg_min, m.stability_tg_max,
            m.stability_tg_p10, m.stability_tg_median, m.stability_tg_p90,
        ) if x is not None and float(x) > 0
    ]
    context_values = [float(row["tg_tps"]) for row in curve]
    validated = short_values + context_values
    context_min = min(context_values) if context_values else None
    context_max = max(context_values) if context_values else None
    first = context_values[0] if context_values else None
    last = context_values[-1] if context_values else None

    return {
        "short_sample_tps": float(m.tg_tps) if m.tg_tps is not None else None,
        "short_median_tps": float(short_median) if short_median is not None else None,
        "short_p10_tps": float(short_p10) if short_p10 is not None else None,
        "short_robust_tps": robust_tg(r) or None,
        "validated_min_tps": min(validated) if validated else None,
        "validated_max_tps": max(validated) if validated else None,
        "context_min_tps": context_min,
        "context_max_tps": context_max,
        "context_first_tps": first,
        "context_last_tps": last,
        "context_retention": (last / first) if first and last is not None else None,
        "workload_profile": profile,
        "workload_weighted_tps": profile_decode_tps(r, profile) or None,
        "context_curve": curve,
        "mtp": bool(r.candidate.mtp),
        "mtp_acceptance_median": m.stability_acceptance_median,
        "mtp_mean_draft_len_median": m.stability_mean_draft_len_median,
    }


def prefill_speed_envelope(r: CandidateResult, profile: str = "agent") -> dict:
    """Return short and context-populated prompt-processing measurements."""
    m = r.metrics
    curve: list[dict] = []
    for row in m.context_staircase or []:
        pp = float(row.get("pp_tps") or 0.0)
        if pp <= 0:
            continue
        tokens = int(row.get("prompt_tokens") or row.get("total_prompt_tokens")
                     or row.get("target_tokens") or 0)
        curve.append({
            "tokens": tokens,
            "target_tokens": int(row.get("target_tokens") or tokens),
            "processed_tokens": int(row.get("processed_prompt_tokens") or 0),
            "cached_tokens": int(row.get("cached_prompt_tokens") or 0),
            "pp_tps": pp,
        })
    values = ([float(m.pp_tps)] if m.pp_tps is not None and m.pp_tps > 0 else []) \
        + [float(row["pp_tps"]) for row in curve]
    return {
        "short_tps": float(m.pp_tps) if m.pp_tps is not None else None,
        "validated_min_tps": min(values) if values else None,
        "validated_max_tps": max(values) if values else None,
        "context_fill_tps": context_fill_pp(r) or None,
        "workload_profile": profile,
        "workload_weighted_tps": profile_prefill_tps(r, profile) or None,
        "context_curve": curve,
    }


def _threshold(a: float, b: float, rel: float, absolute: float) -> float:
    return max(float(absolute), float(rel) * max(abs(a), abs(b), 1e-9))


def decode_noise_threshold(a: float, b: float, policy: NoisePolicy = NoisePolicy(),
                           *, conservative: bool = False) -> float:
    """Scale the absolute TG guard down for slow CPU-offloaded models.

    A fixed 3 t/s floor is sensible around 40-80 t/s, where the relative threshold is already
    several t/s.  At 6-10 t/s it would call a 20-40% improvement noise and can make an oversized
    Dense search retain much more context at a visibly worse interactive speed.  Keep the
    configured relative guard and a small low-speed absolute floor, capped by the user's normal
    absolute setting.
    """
    positive = [abs(float(x)) for x in (a, b) if abs(float(x)) > 1e-9]
    # Percentage gain/loss is measured against the slower/reference-sized result. This makes
    # 43.5 -> 48.0 a genuine 10.3% scout gain instead of requiring 11.1% because 48 was used
    # as the denominator.
    scale = min(positive) if positive else 1e-9
    rel = policy.decode_probe_rel if conservative else policy.decode_rel
    # Keep a tiny instrument-resolution guard near zero, but never let the legacy 3 t/s
    # absolute setting erase a 10-20% difference on CPU-offloaded 6-10 t/s models.
    adaptive_absolute = min(float(policy.decode_abs_tps), max(0.25, 0.03 * scale))
    return max(float(rel) * scale, adaptive_absolute)


def decode_metric_relation(a: float, b: float, policy: NoisePolicy = NoisePolicy(),
                           *, conservative: bool = False) -> int:
    d = float(a) - float(b)
    threshold = decode_noise_threshold(a, b, policy, conservative=conservative)
    if d > threshold:
        return 1
    if d < -threshold:
        return -1
    return 0


def metric_relation(a: float, b: float, rel: float, absolute: float) -> int:
    """Return +1 if a is materially better/higher, -1 if worse, 0 if inside noise zone."""
    d = a - b
    t = _threshold(a, b, rel, absolute)
    if d > t:
        return 1
    if d < -t:
        return -1
    return 0


def decode_relation(a: CandidateResult, b: CandidateResult, policy: NoisePolicy = NoisePolicy()) -> int:
    strong_kinds = {"full", "validation"}
    conservative = not (
        a.metrics.benchmark_kind in strong_kinds and b.metrics.benchmark_kind in strong_kinds
        and a.candidate.ctx == b.candidate.ctx
    )
    return decode_metric_relation(robust_tg(a), robust_tg(b), policy, conservative=conservative)


def decode_requires_confirmation(a: CandidateResult, b: CandidateResult,
                                 policy: NoisePolicy = NoisePolicy()) -> bool:
    """True for a same-context 5–10% decode difference backed only by weak evidence.

    A SCOUT/PROBE difference in this band is neither confirmed noise nor a proven winner.  The
    scheduler can use this signal to spend a bounded FULL comparison instead of silently resolving
    the ambiguity by VRAM headroom.  Cross-context comparisons retain the conservative 10% rule,
    while two FULL/validation results are already authoritative at the 5% band.
    """
    if a.candidate.ctx != b.candidate.ctx:
        return False
    strong_kinds = {"full", "validation"}
    if (a.metrics.benchmark_kind in strong_kinds
            and b.metrics.benchmark_kind in strong_kinds):
        return False
    weak = decode_metric_relation(robust_tg(a), robust_tg(b), policy, conservative=True)
    confirmed_band = decode_metric_relation(robust_tg(a), robust_tg(b), policy, conservative=False)
    return weak == 0 and confirmed_band != 0


def profile_decode_relation(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                            policy: NoisePolicy = NoisePolicy()) -> int:
    strong_kinds = {"full", "validation"}
    conservative = not (
        a.metrics.benchmark_kind in strong_kinds and b.metrics.benchmark_kind in strong_kinds
        and a.candidate.ctx == b.candidate.ctx
    )
    return decode_metric_relation(
        profile_decode_tps(a, profile), profile_decode_tps(b, profile), policy,
        conservative=conservative,
    )


def prefill_relation(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                     policy: NoisePolicy = NoisePolicy()) -> int:
    return metric_relation(
        profile_prefill_tps(a, profile), profile_prefill_tps(b, profile),
        policy.prefill_rel, policy.prefill_abs_tps,
    )


def latency_relation(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                     policy: NoisePolicy = NoisePolicy()) -> int:
    """+1 means a has materially LOWER latency than b; 0 means equivalent."""
    la = workload_latency_seconds(a, profile)
    lb = workload_latency_seconds(b, profile)
    if la == float("inf") and lb == float("inf"):
        return 0
    if la == float("inf"):
        return -1
    if lb == float("inf"):
        return 1
    # Lower is better; use a relative-only threshold because latency is already an aggregate.
    threshold = policy.latency_rel * max(la, lb, 1e-9)
    if lb - la > threshold:
        return 1
    if la - lb > threshold:
        return -1
    return 0


def performance_equivalent(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                           policy: NoisePolicy = NoisePolicy()) -> bool:
    return profile_decode_relation(a, b, profile, policy) == 0 and prefill_relation(a, b, profile, policy) == 0


def noise_aware_dominates(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                          policy: NoisePolicy = NoisePolicy()) -> bool:
    """Pareto dominance where small PP/TG changes are intentionally treated as ties."""
    if a.status not in {RunStatus.PASS, RunStatus.PASS_DEGRADED}:
        return False
    if b.status not in {RunStatus.PASS, RunStatus.PASS_DEGRADED}:
        return True
    dr = profile_decode_relation(a, b, profile, policy)
    pr = prefill_relation(a, b, profile, policy)
    af = int(a.metrics.vram_free_min_mb or 0)
    bf = int(b.metrics.vram_free_min_mb or 0)
    vr = 1 if af > bf + policy.vram_abs_mb else (-1 if bf > af + policy.vram_abs_mb else 0)
    # At the same configured context, KV-cache numerical precision is a genuine Pareto axis.
    # Q4 cannot dominate Q8 merely because it leaves more unused VRAM; that headroom was bought
    # with lower attention-cache precision. Cross-context semantic trade-offs are resolved by the
    # target-fidelity policy in the optimizer/report rather than by this local dominance helper.
    kr = 0
    if a.candidate.ctx == b.candidate.ctx:
        ak = kv_precision_key(a.candidate.kv_k, a.candidate.kv_v)
        bk = kv_precision_key(b.candidate.kv_k, b.candidate.kv_v)
        kr = 1 if ak > bk else (-1 if bk > ak else 0)
    # No material regression on any axis, material win on at least one.
    return dr >= 0 and pr >= 0 and vr >= 0 and kr >= 0 and (dr > 0 or pr > 0 or vr > 0 or kr > 0)


def pareto_frontier(results: list[CandidateResult], profile: str = "agent",
                    policy: NoisePolicy = NoisePolicy()) -> list[CandidateResult]:
    good = [r for r in results if r.status in {RunStatus.PASS, RunStatus.PASS_DEGRADED}
            and r.metrics.pp_tps and robust_tg(r)]
    out: list[CandidateResult] = []
    for r in good:
        if any(other is not r and noise_aware_dominates(other, r, profile, policy) for other in good):
            continue
        out.append(r)
    return out


def prefer_candidate(a: CandidateResult, b: CandidateResult, profile: str = "agent",
                     policy: NoisePolicy = NoisePolicy()) -> CandidateResult:
    """Choose between two candidates without turning measurement noise into fake precision.

    1. Noise-aware Pareto dominance.
    2. Materially lower representative end-to-end latency.
    3. At equal context, prefer higher KV-cache precision.
    4. If performance/precision are tied, prefer more VRAM headroom.
    5. Final deterministic tie-break: smaller ubatch, then smaller batch.
    """
    if noise_aware_dominates(a, b, profile, policy):
        return a
    if noise_aware_dominates(b, a, profile, policy):
        return b
    lr = latency_relation(a, b, profile, policy)
    if lr > 0:
        return a
    if lr < 0:
        return b
    if a.candidate.ctx == b.candidate.ctx:
        ak = kv_precision_key(a.candidate.kv_k, a.candidate.kv_v)
        bk = kv_precision_key(b.candidate.kv_k, b.candidate.kv_v)
        if ak != bk:
            return a if ak > bk else b
    af = int(a.metrics.vram_free_min_mb or 0)
    bf = int(b.metrics.vram_free_min_mb or 0)
    if af != bf:
        return a if af > bf else b
    if a.candidate.ubatch != b.candidate.ubatch:
        return a if a.candidate.ubatch < b.candidate.ubatch else b
    if a.candidate.batch != b.candidate.batch:
        return a if a.candidate.batch < b.candidate.batch else b
    return a


def choose_preferred(results: list[CandidateResult], profile: str = "agent",
                     policy: NoisePolicy = NoisePolicy()) -> CandidateResult | None:
    if not results:
        return None
    winner = results[0]
    for r in results[1:]:
        winner = prefer_candidate(winner, r, profile, policy)
    return winner


def score_result(r: CandidateResult, pp_ref: float, tg_ref: float, preferred_headroom_mb: int) -> float:
    """Backward-compatible coarse score retained for external callers/tests.

    New search decisions use noise-aware latency/Pareto helpers above.
    """
    if r.status not in {RunStatus.PASS, RunStatus.PASS_DEGRADED}:
        return -1.0
    pp = (r.metrics.pp_tps or 0) / max(1.0, pp_ref)
    tg = robust_tg(r) / max(1.0, tg_ref)
    free = r.metrics.vram_free_min_mb or 0
    headroom = min(1.0, free / max(1, preferred_headroom_mb))
    return 0.40 * pp + 0.45 * tg + 0.15 * headroom


def dominates(a: CandidateResult, b: CandidateResult) -> bool:
    # Preserve the original strict helper API; search/reporting use noise_aware_dominates.
    am, bm = a.metrics, b.metrics
    av = (am.pp_tps or 0, robust_tg(a), am.vram_free_min_mb or 0)
    bv = (bm.pp_tps or 0, robust_tg(b), bm.vram_free_min_mb or 0)
    return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))
