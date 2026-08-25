from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

from llama_autotuner.benchmark.workloads import approx_prompt, context_staircase_prompts, stability_workloads
from llama_autotuner.llama.api import completion
from llama_autotuner.models import BenchmarkMetrics

ProgressFn = Callable[[str], None] | None
LogMarkFn = Callable[[], int] | None
DraftStatsFn = Callable[[int], list[dict[str, float | int]]] | None


@dataclass(slots=True)
class WorkloadResult:
    metrics: BenchmarkMetrics
    response: dict


@dataclass(slots=True)
class QuickBenchmarkResult:
    metrics: BenchmarkMetrics
    severe_regression: bool = False
    pp_ratio_to_reference: float | None = None
    tg_ratio_to_reference: float | None = None


@dataclass(slots=True)
class StagedBenchmarkResult:
    metrics: BenchmarkMetrics
    # Historical field name retained for API compatibility. This flag means a severe prefill
    # throughput cliff as context grows; it is NOT proof of VRAM exhaustion.
    memory_cliff: bool = False
    cliff_ratio: float | None = None


@dataclass(slots=True)
class LongContextResult:
    prompt_tokens: int
    pp_tps: float | None
    passed: bool
    ratio_to_reference: float | None = None


@dataclass(slots=True)
class StabilityBenchmarkResult:
    samples: list[dict]
    context_staircase: list[dict]
    tg_median: float | None
    tg_p10: float | None
    tg_p90: float | None
    tg_min: float | None
    tg_max: float | None
    acceptance_median: float | None
    mean_draft_len_median: float | None
    mean_draft_len_p10: float | None
    tg_mean_len_corr: float | None
    tg_acceptance_corr: float | None
    variation_pct: float | None
    context_tg_ratio: float | None
    passed: bool


def _emit(progress: ProgressFn, message: str) -> None:
    if progress:
        progress(message)


def _timeout(prompt_tokens: int, gen_tokens: int, pp_hint: float = 200.0, tg_hint: float = 10.0) -> float:
    expected = prompt_tokens / max(1.0, pp_hint) + gen_tokens / max(0.5, tg_hint)
    return max(30.0, min(600.0, expected * 3.0 + 15.0))


def run_completion(base_url: str, prompt: str, n_predict: int, cache_prompt: bool,
                   pp_hint: float = 200.0, tg_hint: float = 10.0, seed: int | None = 42) -> WorkloadResult:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": n_predict > 0,
        "cache_prompt": cache_prompt,
        "temperature": 0.6,
    }
    if seed is not None:
        payload["seed"] = seed
    data = completion(base_url, payload, timeout=_timeout(max(1, len(prompt)//4), n_predict, pp_hint, tg_hint))
    t = data.get("timings") or {}
    usage = data.get("usage") or {}
    processed_prompt = int(t.get("prompt_n") or data.get("tokens_evaluated") or 0)
    cache_tokens = int(t.get("cache_n") or 0)
    total_prompt = int(usage.get("prompt_tokens") or (processed_prompt + cache_tokens))
    draft_n = int(t.get("draft_n") or 0)
    accepted = int(t.get("draft_n_accepted") or 0)
    m = BenchmarkMetrics(
        prompt_tokens=processed_prompt,
        cache_tokens=cache_tokens,
        prompt_total_tokens=total_prompt,
        pp_tps=float(t["prompt_per_second"]) if t.get("prompt_per_second") is not None else None,
        generated_tokens=int(t.get("predicted_n") or data.get("tokens_predicted") or 0),
        tg_tps=float(t["predicted_per_second"]) if t.get("predicted_per_second") is not None else None,
        draft_n=draft_n, draft_accepted=accepted,
        acceptance=(accepted / draft_n) if draft_n else None,
    )
    return WorkloadResult(m, data)


def _scaled_stages(context_size: int) -> tuple[int, ...]:
    """Return useful staged prefill targets that stay below the configured slot context."""
    ceiling = max(768, min(10_000, int(context_size * 0.72)))
    if ceiling >= 10_000:
        return (2_000, 6_000, 10_000)
    if ceiling >= 6_000:
        return (2_000, 4_000, ceiling)
    if ceiling >= 3_000:
        return (1_000, 2_000, ceiling)
    return (max(512, ceiling // 3), max(768, (ceiling * 2) // 3), ceiling)


def benchmark_quick(base_url: str, context_size: int = 65_536, pp_tokens: int = 2500,
                    tg_tokens: int = 128, progress: ProgressFn = None,
                    reference_pp_tps: float | None = None, reference_tg_tps: float | None = None,
                    severe_ratio: float = 0.50) -> QuickBenchmarkResult:
    """Cheap candidate probe with a relative-performance guard.

    When a nearby known-good candidate exists, a catastrophic prefill regression is enough to
    reject this candidate without paying for decode or a later staged/full benchmark. This is
    intentionally conservative: only a >=50% collapse is rejected by default.
    """
    pp_tokens = min(pp_tokens, max(768, int(context_size * 0.60)))
    _emit(progress, "warmup: 64 generated tokens")
    run_completion(base_url, "Warm up the inference graph with a short deterministic response.", 64, False)
    _emit(progress, f"quick prefill: ~{pp_tokens} tokens")
    pp = run_completion(base_url, approx_prompt(pp_tokens), 1, False)

    pp_ratio = None
    if reference_pp_tps and pp.metrics.pp_tps:
        pp_ratio = pp.metrics.pp_tps / max(1e-9, reference_pp_tps)
        if pp_ratio < severe_ratio:
            _emit(progress,
                  f"probe guard: PP retained only {pp_ratio:.0%} of nearest good candidate; skipping decode/full validation")
            m = pp.metrics
            m.benchmark_kind = "quick"
            # The prefill request uses n_predict=1 only to force server timings. It is not a decode benchmark.
            m.generated_tokens = 0
            m.tg_tps = None
            m.draft_n = 0
            m.draft_accepted = 0
            m.acceptance = None
            m.early_pp_tps = m.pp_tps
            m.final_pp_tps = m.pp_tps
            return QuickBenchmarkResult(m, severe_regression=True, pp_ratio_to_reference=pp_ratio)

    _emit(progress, f"quick decode: {tg_tokens} tokens")
    tg = run_completion(base_url, approx_prompt(min(600, max(256, context_size // 8)), coding=True), tg_tokens, False,
                        pp_hint=pp.metrics.pp_tps or 200)
    m = tg.metrics
    m.benchmark_kind = "quick"
    m.prompt_tokens = pp.metrics.prompt_tokens
    m.pp_tps = pp.metrics.pp_tps
    m.early_pp_tps = pp.metrics.pp_tps
    m.final_pp_tps = pp.metrics.pp_tps

    tg_ratio = None
    severe = False
    if reference_tg_tps and m.tg_tps:
        tg_ratio = m.tg_tps / max(1e-9, reference_tg_tps)
        severe = tg_ratio < severe_ratio
        if severe:
            _emit(progress, f"probe guard: TG retained only {tg_ratio:.0%} of nearest good candidate")
    return QuickBenchmarkResult(m, severe_regression=severe,
                                pp_ratio_to_reference=pp_ratio, tg_ratio_to_reference=tg_ratio)




def benchmark_dense_partial_screen(base_url: str, context_size: int = 65_536,
                                   progress: ProgressFn = None,
                                   reference_pp_tps: float | None = None,
                                   reference_tg_tps: float | None = None,
                                   severe_ratio: float = 0.50,
                                   decode_tokens: int = 32) -> QuickBenchmarkResult:
    """Cheap batch/ubatch screen for slow numeric-ngl Dense candidates.

    Placement is already known at this point. Keep enough prompt tokens (2.5K) to expose ubatch
    throughput, but avoid generic 64-token warmup + 128-token decode when generation may be only
    5-15 tok/s. Winners are FULL-confirmed by the optimizer.
    """
    pp_tokens = min(2500, max(768, int(context_size * 0.60)))
    _emit(progress, "partial-Dense warmup: 8 generated tokens")
    run_completion(base_url, "Warm up briefly.", 8, False, tg_hint=8.0)
    _emit(progress, f"partial-Dense prefill: ~{pp_tokens} tokens")
    pp = run_completion(base_url, approx_prompt(pp_tokens), 1, False, pp_hint=500.0, tg_hint=8.0)
    pp_ratio = None
    if reference_pp_tps and pp.metrics.pp_tps:
        pp_ratio = pp.metrics.pp_tps / max(1e-9, reference_pp_tps)
        if pp_ratio < severe_ratio:
            _emit(progress, f"partial-Dense guard: PP retained only {pp_ratio:.0%}; skip decode")
            m = pp.metrics
            m.benchmark_kind = "quick"
            m.generated_tokens = 0
            m.tg_tps = None
            m.early_pp_tps = m.pp_tps
            m.final_pp_tps = m.pp_tps
            return QuickBenchmarkResult(m, severe_regression=True, pp_ratio_to_reference=pp_ratio)
    decode_tokens = max(16, int(decode_tokens))
    _emit(progress, f"partial-Dense decode: {decode_tokens} tokens")
    tg = run_completion(
        base_url, approx_prompt(min(600, max(256, context_size // 8)), coding=True),
        decode_tokens, False, pp_hint=pp.metrics.pp_tps or 500.0, tg_hint=reference_tg_tps or 8.0,
    )
    m = tg.metrics
    m.benchmark_kind = "quick"
    m.prompt_tokens = pp.metrics.prompt_tokens
    m.pp_tps = pp.metrics.pp_tps
    m.early_pp_tps = pp.metrics.pp_tps
    m.final_pp_tps = pp.metrics.pp_tps
    tg_ratio = None
    severe = False
    if reference_tg_tps and m.tg_tps:
        tg_ratio = m.tg_tps / max(1e-9, reference_tg_tps)
        severe = tg_ratio < severe_ratio
        if severe:
            _emit(progress, f"partial-Dense guard: TG retained only {tg_ratio:.0%}")
    return QuickBenchmarkResult(m, severe_regression=severe,
                                pp_ratio_to_reference=pp_ratio, tg_ratio_to_reference=tg_ratio)


def benchmark_recon(base_url: str, context_size: int = 65_536, progress: ProgressFn = None,
                    mtp: bool = False, tg_tokens: int | None = None,
                    slow_cpu: bool = False) -> QuickBenchmarkResult:
    """Very cheap solution-level scout.

    Reconnaissance compares *different compromises* (context/KV/placement). It should not
    spend 60-80 seconds proving that a heavily CPU-offloaded exact target is slow. Full-GPU
    candidates get a small but useful sample. Heavy Dense CPU-offload gets an even smaller
    short warmup/512-token-prefill sample. Slow decode uses 16/32/48 tokens by search mode so a
    one-token timing wobble cannot masquerade as a >10% architectural win. Any selected branch is
    still re-measured later with a FULL workload.
    """
    if slow_cpu:
        pp_tokens = min(512, max(256, int(context_size * 0.05)))
        warmup_tokens = 8
        tg_tokens = int(tg_tokens or 32)
    else:
        pp_tokens = min(1200, max(512, int(context_size * 0.20)))
        warmup_tokens = 16
        tg_tokens = int(tg_tokens or (96 if mtp else 64))
    _emit(progress, f"recon warmup: {warmup_tokens} generated tokens")
    run_completion(base_url, "Warm up briefly.", warmup_tokens, False, tg_hint=8.0)
    _emit(progress, f"recon prefill: ~{pp_tokens} tokens")
    pp = run_completion(base_url, approx_prompt(pp_tokens), 1, False, pp_hint=200.0, tg_hint=8.0)
    _emit(progress, f"recon decode: {tg_tokens} tokens")
    tg = run_completion(
        base_url, approx_prompt(min(420, max(192, context_size // 16)), coding=True),
        tg_tokens, False, pp_hint=pp.metrics.pp_tps or 200.0, tg_hint=8.0,
    )
    m = tg.metrics
    m.benchmark_kind = "recon"
    m.prompt_tokens = pp.metrics.prompt_tokens
    m.pp_tps = pp.metrics.pp_tps
    m.early_pp_tps = pp.metrics.pp_tps
    m.final_pp_tps = pp.metrics.pp_tps
    return QuickBenchmarkResult(m)


def benchmark_recon_context(base_url: str, context_size: int = 65_536,
                            progress: ProgressFn = None,
                            target_tokens: int | None = None,
                            reference_tg_tps: float | None = None,
                            severe_ratio: float = 0.60) -> QuickBenchmarkResult:
    """A medium-cost context-aware scout for the final 2 solution families.

    A 1.2K short scout cannot distinguish 48K from 96K when both have the same empty-slot
    decode speed. The ordinary probe fills a meaningful fraction of *that candidate's* context,
    capped at 16K so it stays far cheaper than FINAL. ``target_tokens`` lets the KV policy request
    one deeper occupied-cache measurement for Q4/mixed-Q4, where a short scout can hide a
    long-context dequantization/attention throughput cliff.
    """
    target = (
        min(16_000, max(6_000, int(context_size * 0.20)))
        if target_tokens is None else max(2_048, int(target_tokens))
    )
    target = min(target, max(2_048, context_size - 1_024))
    warmup_tokens = 8
    decode_tokens = 64
    _emit(progress, f"context scout warmup: {warmup_tokens} generated tokens")
    run_completion(base_url, "Warm up briefly for a context-aware scout.", warmup_tokens, False, tg_hint=10.0)
    # Very large MoE qualifications used to be one opaque 3-4 minute HTTP request. Grow a
    # shared-prefix cache in three stages instead: total prefill work remains approximately the
    # same, while logs gain real checkpoints and an obvious early scaling collapse can stop before
    # paying for the remaining 2/3. Small Dense scouts retain the old one-request path.
    if target >= 65_536:
        align = 1_024
        stage_targets = [
            max(8_192, (target // 3 // align) * align),
            max(16_384, ((target * 2) // 3 // align) * align),
            target,
        ]
        stage_targets = list(dict.fromkeys(stage_targets))
    else:
        stage_targets = [target]

    stage_metrics: list[BenchmarkMetrics] = []
    pp_hint = 1000.0
    tg_hint = reference_tg_tps or 40.0
    severe = False
    tg_ratio = None
    for index, stage_target in enumerate(stage_targets, start=1):
        stage_decode = decode_tokens if index == len(stage_targets) else 16
        label = (
            f"context scout stage {index}/{len(stage_targets)}: "
            if len(stage_targets) > 1 else "context scout: "
        )
        _emit(progress, f"{label}~{stage_target} prompt tokens + {stage_decode} decode")
        wr = run_completion(
            base_url, approx_prompt(stage_target, coding=True), stage_decode,
            len(stage_targets) > 1, pp_hint=pp_hint, tg_hint=tg_hint, seed=31415,
        )
        stage_metrics.append(wr.metrics)
        if wr.metrics.pp_tps:
            pp_hint = wr.metrics.pp_tps
        if wr.metrics.tg_tps:
            tg_hint = wr.metrics.tg_tps
        if reference_tg_tps and wr.metrics.tg_tps:
            tg_ratio = float(wr.metrics.tg_tps) / max(1e-9, float(reference_tg_tps))
            if index < len(stage_targets) and tg_ratio < severe_ratio:
                severe = True
                _emit(
                    progress,
                    f"context scout early-stop: decode retained only {tg_ratio:.0%} of the "
                    "short scout; remaining occupied-cache stages are skipped",
                )
                break

    m = stage_metrics[-1]
    early_pp_tps = stage_metrics[0].pp_tps
    final_pp_tps = stage_metrics[-1].pp_tps
    m.benchmark_kind = "recon-context"
    pp_rows = [
        (int(row.prompt_tokens or 0), float(row.pp_tps))
        for row in stage_metrics if row.pp_tps and row.prompt_tokens
    ]
    if pp_rows:
        total_processed = sum(tokens for tokens, _ in pp_rows)
        total_seconds = sum(tokens / speed for tokens, speed in pp_rows)
        if total_seconds > 0:
            m.pp_tps = total_processed / total_seconds
    m.early_pp_tps = early_pp_tps
    m.final_pp_tps = final_pp_tps
    m.long_context_tokens = int(m.prompt_total_tokens or m.prompt_tokens or 0)
    m.long_context_pp_tps = m.pp_tps
    # This is qualification evidence, not FINAL.  Keep ``long_context_passed``
    # reserved for the full robustness/staircase validator.
    m.long_context_passed = False
    return QuickBenchmarkResult(m, severe_regression=severe, tg_ratio_to_reference=tg_ratio)


def benchmark_staged(base_url: str, context_size: int = 65_536, tg_tokens: int = 512, repeats: int = 2,
                     cliff_threshold: float = 0.65, progress: ProgressFn = None,
                     warmup: bool = True) -> StagedBenchmarkResult:
    """Real server benchmark with an early memory-pressure guard.

    Prefill is measured at progressively larger prompts. The stage sizes are reduced automatically
    for small context configurations. A catastrophic drop can reject the candidate before paying
    for the generation workload. Server-reported token counts remain the ground truth.
    """
    if warmup:
        _emit(progress, "warmup: 64 generated tokens")
        run_completion(base_url, "Warm up the inference graph with a short deterministic response.", 64, False)
    pp_stages: list[tuple[int, float]] = []
    hint = 200.0
    last_prompt_n = 0
    stages = _scaled_stages(context_size)
    for target in stages:
        _emit(progress, f"staged prefill: ~{target} tokens")
        wr = run_completion(base_url, approx_prompt(target), 1, False, pp_hint=hint)
        if wr.metrics.pp_tps:
            pp_stages.append((wr.metrics.prompt_tokens, wr.metrics.pp_tps))
            hint = wr.metrics.pp_tps
        last_prompt_n = wr.metrics.prompt_tokens
        if len(pp_stages) >= 2:
            ratio = pp_stages[-1][1] / max(1e-9, pp_stages[0][1])
            if ratio < cliff_threshold:
                _emit(progress, f"prefill-performance cliff guard triggered: PP ratio={ratio:.2f}")
                return StagedBenchmarkResult(BenchmarkMetrics(
                    benchmark_kind="full",
                    prompt_tokens=last_prompt_n, pp_tps=pp_stages[-1][1],
                    early_pp_tps=pp_stages[0][1], final_pp_tps=pp_stages[-1][1],
                ), memory_cliff=True, cliff_ratio=ratio)

    tg_vals: list[float] = []
    draft_n = accepted = gen_n = 0
    for i in range(max(1, repeats)):
        _emit(progress, f"decode benchmark: {tg_tokens} tokens ({i+1}/{max(1, repeats)})")
        tg = run_completion(base_url, approx_prompt(min(700, max(300, context_size // 10)), coding=True), tg_tokens, False,
                            pp_hint=hint, tg_hint=max(tg_vals or [10]))
        if tg.metrics.tg_tps:
            tg_vals.append(tg.metrics.tg_tps)
        gen_n = tg.metrics.generated_tokens
        draft_n += tg.metrics.draft_n; accepted += tg.metrics.draft_accepted
    pp_values = [x[1] for x in pp_stages]
    m = BenchmarkMetrics(
        benchmark_kind="full",
        prompt_tokens=last_prompt_n,
        pp_tps=pp_values[-1] if pp_values else None,
        early_pp_tps=pp_values[0] if pp_values else None,
        final_pp_tps=pp_values[-1] if pp_values else None,
        generated_tokens=gen_n,
        tg_tps=statistics.median(tg_vals) if tg_vals else None,
        draft_n=draft_n, draft_accepted=accepted,
        acceptance=(accepted / draft_n) if draft_n else None,
    )
    if len(tg_vals) >= 2 and m.tg_tps:
        m.variance_pct = 100 * (max(tg_vals) - min(tg_vals)) / m.tg_tps
    return StagedBenchmarkResult(m)


def validate_long_context(base_url: str, context_size: int, reference_pp: float | None,
                          progress: ProgressFn = None, cliff_threshold: float = 0.50) -> LongContextResult:
    """Validate that a candidate remains usable at a materially longer prompt.

    For a 64K slot this targets ~32K. For smaller slots it uses ~75% of the configured context.
    This validates real prompt processing; it does not claim that every token up to n_ctx was tested.
    """
    target = min(32_000, max(1_500, int(context_size * 0.75)))
    _emit(progress, f"long-context validation: ~{target} prompt tokens")
    wr = run_completion(base_url, approx_prompt(target), 1, False, pp_hint=reference_pp or 100.0)
    pp = wr.metrics.pp_tps
    ratio = None
    passed = bool(pp and pp > 0)
    if pp and reference_pp:
        ratio = pp / max(1e-9, reference_pp)
        if ratio < cliff_threshold:
            passed = False
    return LongContextResult(wr.metrics.prompt_tokens, pp, passed, ratio)



def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    q = min(1.0, max(0.0, q))
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _exact_draft_stats(mark: int | None, draft_stats_since: DraftStatsFn) -> dict | None:
    if mark is None or draft_stats_since is None:
        return None
    # The HTTP response and stdout reader are asynchronous by a few milliseconds on Windows.
    # Briefly poll so a just-printed `mean len` line is not missed due to thread scheduling.
    deadline = time.monotonic() + 0.35
    while True:
        rows = draft_stats_since(mark)
        if rows:
            return rows[-1]
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.025)


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in pairs)
    dx = sum((x-mx)**2 for x in xs)
    dy = sum((y-my)**2 for y in ys)
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx*dy) ** 0.5


def benchmark_stability(base_url: str, context_size: int, mode: str = "normal",
                        progress: ProgressFn = None, log_mark: LogMarkFn = None,
                        draft_stats_since: DraftStatsFn = None,
                        tg_tokens: int | None = None) -> StabilityBenchmarkResult:
    """Measure decode robustness across heterogeneous text and a growing cached context.

    This benchmark deliberately runs in one llama-server process. Heterogeneous samples expose MTP
    sensitivity to token predictability; the context staircase exposes decode slowdown as the slot/KV
    working set grows. `mean len` is associated from the exact server log when available because the
    current HTTP timings do not expose speculative verification-step count.
    """
    decode_tokens = tg_tokens or ({"quick": 256, "normal": 256, "deep": 512}.get(mode, 256))
    samples: list[dict] = []
    tg_values: list[float] = []
    acc_values: list[float] = []
    mean_len_values: list[float] = []

    for idx, (name, prompt) in enumerate(stability_workloads(mode), start=1):
        _emit(progress, f"decode stability workload {idx}: {name} ({decode_tokens} generated tokens)")
        mark = log_mark() if log_mark else None
        wr = run_completion(base_url, prompt, decode_tokens, False, tg_hint=45.0, seed=1000 + idx)
        exact = _exact_draft_stats(mark, draft_stats_since)
        row = {
            "name": name,
            "prompt_tokens": wr.metrics.prompt_tokens,
            "generated_tokens": wr.metrics.generated_tokens,
            "tg_tps": wr.metrics.tg_tps,
            "acceptance": wr.metrics.acceptance,
            "mean_draft_len": exact.get("mean_len") if exact else None,
        }
        samples.append(row)
        if wr.metrics.tg_tps is not None:
            tg_values.append(wr.metrics.tg_tps)
        if wr.metrics.acceptance is not None:
            acc_values.append(wr.metrics.acceptance)
        if exact and exact.get("mean_len") is not None:
            mean_len_values.append(float(exact["mean_len"]))
        suffix = f"TG={wr.metrics.tg_tps:.1f} t/s" if wr.metrics.tg_tps is not None else "TG=n/a"
        if exact:
            suffix += f", mean-len={float(exact['mean_len']):.2f}"
        _emit(progress, f"stability sample: {name} -> {suffix}")

    # Grow a single cached slot using prompts with a shared prefix. This is intentionally after the
    # heterogeneous phase; the first staircase request establishes a new prefix, later stages extend it.
    staircase: list[dict] = []
    stair_tg: list[float] = []
    for idx, (target, prompt) in enumerate(context_staircase_prompts(context_size, mode), start=1):
        stair_decode = int(tg_tokens) if tg_tokens is not None else (128 if mode == "quick" else 256)
        _emit(progress, f"context staircase {idx}: target ~{target} prompt tokens + {stair_decode} decode")
        mark = log_mark() if log_mark else None
        wr = run_completion(base_url, prompt, stair_decode, True, pp_hint=1200.0, tg_hint=45.0, seed=2000)
        exact = _exact_draft_stats(mark, draft_stats_since)
        row = {
            "target_tokens": target,
            # `timings.prompt_n` is only the newly processed delta when cache_prompt reuses a prefix.
            # The full prompt size is usage.prompt_tokens (fallback: prompt_n + cache_n).
            "prompt_tokens": wr.metrics.prompt_total_tokens,
            "processed_prompt_tokens": wr.metrics.prompt_tokens,
            "cached_prompt_tokens": wr.metrics.cache_tokens,
            "pp_tps": wr.metrics.pp_tps,
            "tg_tps": wr.metrics.tg_tps,
            "acceptance": wr.metrics.acceptance,
            "mean_draft_len": exact.get("mean_len") if exact else None,
        }
        staircase.append(row)
        if wr.metrics.tg_tps is not None:
            stair_tg.append(wr.metrics.tg_tps)
        _emit(progress, "context stage: " +
              (f"total={wr.metrics.prompt_total_tokens} tok, processed={wr.metrics.prompt_tokens}, "
               f"cached={wr.metrics.cache_tokens}, PP={wr.metrics.pp_tps:.1f} t/s, TG={wr.metrics.tg_tps:.1f} t/s")
              if wr.metrics.pp_tps is not None and wr.metrics.tg_tps is not None else
              f"context stage total={wr.metrics.prompt_total_tokens} tok")

    median_tg = statistics.median(tg_values) if tg_values else None
    p10_tg = _percentile(tg_values, .10)
    p90_tg = _percentile(tg_values, .90)
    variation = None
    if median_tg and tg_values:
        variation = 100.0 * (max(tg_values) - min(tg_values)) / median_tg
    context_ratio = None
    if len(stair_tg) >= 2 and stair_tg[0] > 0:
        context_ratio = stair_tg[-1] / stair_tg[0]

    # Stability is a robustness signal, not a strict semantic-quality gate. Reject only catastrophic
    # behavior; normal MTP variability is preserved in p10/median and influences ranking.
    passed = bool(len(tg_values) >= max(2, len(stability_workloads(mode)) - 1))
    if median_tg and p10_tg is not None and p10_tg < median_tg * 0.45:
        passed = False
    if context_ratio is not None and context_ratio < 0.45:
        passed = False

    tg_mean_pairs = [
        (float(row["tg_tps"]), float(row["mean_draft_len"]))
        for row in samples
        if row.get("tg_tps") is not None and row.get("mean_draft_len") is not None
    ]
    tg_acc_pairs = [
        (float(row["tg_tps"]), float(row["acceptance"]))
        for row in samples
        if row.get("tg_tps") is not None and row.get("acceptance") is not None
    ]

    return StabilityBenchmarkResult(
        samples=samples,
        context_staircase=staircase,
        tg_median=median_tg,
        tg_p10=p10_tg,
        tg_p90=p90_tg,
        tg_min=min(tg_values) if tg_values else None,
        tg_max=max(tg_values) if tg_values else None,
        acceptance_median=statistics.median(acc_values) if acc_values else None,
        mean_draft_len_median=statistics.median(mean_len_values) if mean_len_values else None,
        mean_draft_len_p10=_percentile(mean_len_values, .10),
        tg_mean_len_corr=_pearson(tg_mean_pairs),
        tg_acceptance_corr=_pearson(tg_acc_pairs),
        variation_pct=variation,
        context_tg_ratio=context_ratio,
        passed=passed,
    )
