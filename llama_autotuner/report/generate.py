from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llama_autotuner.llama.command import build_server_command
from llama_autotuner.models import CandidateResult, LaunchProfile, ModelKind, RunStatus
from llama_autotuner.tuning.kv import kv_precision, kv_precision_key
from llama_autotuner.tuning.scoring import (
    NoisePolicy, calibrate_noise_policy, choose_preferred, context_fill_pp, decode_noise_threshold, decode_relation, decode_speed_envelope,
    pareto_frontier, performance_equivalent, prefill_speed_envelope, profile_decode_tps,
    profile_prefill_tps, robust_tg, workload_latency_seconds,
)
from llama_autotuner.tuning.vram import VramOperatingClass, vram_thresholds

SAFE_PERF_FLOOR_RATIO = 0.80

PROFILE_PURPOSES = {
    "OPTIMAL": "Recommended balance: lowest representative prefill + generation latency.",
    "MAX_KV_PRECISION": "Highest measured attention/KV-cache numerical precision.",
    "FASTEST": "Highest materially distinct robust generation speed.",
    "MAX_CONTEXT": "Largest successfully measured configured context.",
    "FALLBACK": "Runnable fallback when no normal recommendation was established.",
}


def _measurement_scope(results: list[CandidateResult], target_ctx: dict) -> dict:
    """Describe what the search actually measured so profile names are not overclaimed."""
    attempted = list(results)
    measured = [r for r in results if r.metrics.pp_tps and r.metrics.tg_tps]
    passed = [r for r in measured if r.status in {RunStatus.PASS, RunStatus.PASS_DEGRADED}]
    validated = [r for r in passed if r.metrics.benchmark_kind == "validation"]
    preferred = (target_ctx.get("preferred_kv_k", "f16"), target_ctx.get("preferred_kv_v", "f16"))
    requested = int(target_ctx.get("context") or 0)
    successful_families = sorted({
        (r.candidate.ctx, r.candidate.kv_k, r.candidate.kv_v, bool(r.candidate.mtp))
        for r in passed
    })
    tradeoff_measured = any(
        ctx != requested or (k, v) != preferred
        for ctx, k, v, _ in successful_families
    )
    return {
        "attempted_runs": len(attempted),
        "measured_runs": len(measured),
        "validated_runs": len(validated),
        "contexts_measured": sorted({r.candidate.ctx for r in passed}),
        "kv_families_measured": sorted({f"{r.candidate.kv_k}/{r.candidate.kv_v}" for r in passed}),
        "mtp_attempted": any(r.candidate.mtp for r in attempted),
        "mtp_text_benchmarked": any(r.candidate.mtp and r.metrics.tg_tps for r in measured),
        "tradeoff_families_measured": tradeoff_measured,
        "successful_families": [
            {"context": ctx, "kv_k": k, "kv_v": v, "mtp": mtp}
            for ctx, k, v, mtp in successful_families
        ],
    }


def _group_profiles(profiles: list[LaunchProfile]) -> list[list[LaunchProfile]]:
    """Render identical launch commands once while preserving every semantic role."""
    groups: list[list[LaunchProfile]] = []
    by_key: dict[str, list[LaunchProfile]] = {}
    for profile in profiles:
        key = profile.candidate.key()
        if key not in by_key:
            by_key[key] = []
            groups.append(by_key[key])
        by_key[key].append(profile)
    return groups


def _experimental_profile(profile: LaunchProfile) -> bool:
    """LOW-confidence scout roles are findings, not copy-as-is recommendations."""
    return (
        profile.confidence == "LOW"
        and profile.result.metrics.benchmark_kind != "validation"
    )


def _result_vram_headroom(r: CandidateResult, *, preferred_margin_mb: int,
                          absolute_floor_mb: int = 300, search_mode: str = "normal") -> dict:
    kind = ModelKind.MOE if r.candidate.ncmoe is not None else ModelKind.DENSE
    thresholds = vram_thresholds(
        absolute_floor_mb=absolute_floor_mb,
        preferred_reserve_mb=preferred_margin_mb,
        search_mode=search_mode,
        model_kind=kind,
        vision=r.candidate.vision,
    )
    return thresholds.to_dict(r.metrics.vram_free_min_mb)


_EVIDENCE_RANK = {
    "recon": 1,
    "quick": 2,
    "recon-context": 3,
    "full": 4,
    "validation": 5,
}


def _result_evidence_rank(r: CandidateResult) -> int:
    """Return the authority of one measurement, including failed strong runs.

    A successful SCOUT must not resurrect a command after the exact same command later
    proved FRAGILE/OOM during FULL or FINAL.  Benchmark kind is normally populated even
    for a failed run; phase-name fallbacks keep imported/older SQLite rows safe too.
    """
    kind = str(r.metrics.benchmark_kind or "").lower()
    if kind in _EVIDENCE_RANK:
        return _EVIDENCE_RANK[kind]
    phase = str(r.phase or "").upper()
    if "FINAL" in phase or "VALIDATION" in phase:
        return 5
    if "FULL" in phase or "CONFIRM" in phase or "BOUNDARY" in phase:
        return 4
    if "CONTEXT" in phase:
        return 3
    if "PROBE" in phase or "SCOUT" in phase or "RECON" in phase:
        return 2
    return 0


def _resolved_candidate_evidence(results: list[CandidateResult]) -> list[CandidateResult]:
    """Collapse an exact launch command to its strongest, latest evidence.

    Resolution intentionally happens *before* PASS/VRAM filtering.  Otherwise a later
    FRAGILE FULL is filtered out first and an older optimistic recon measurement leaks
    back into FASTEST/MAX_KV_PRECISION/MAX_CONTEXT.
    """
    strongest: dict[str, tuple[int, int, CandidateResult]] = {}
    for index, result in enumerate(results):
        key = result.candidate.key()
        rank = _result_evidence_rank(result)
        previous = strongest.get(key)
        if previous is None or rank > previous[0] or (rank == previous[0] and index > previous[1]):
            strongest[key] = (rank, index, result)
    return [entry[2] for entry in sorted(strongest.values(), key=lambda entry: entry[1])]


def _candidate_pool(results: list[CandidateResult], session_complete: bool,
                    preferred_margin_mb: int, absolute_floor_mb: int = 300,
                    search_mode: str = "normal", require_preferred_margin: bool = False) -> list[CandidateResult]:
    passed: list[CandidateResult] = []
    for r in _resolved_candidate_evidence(results):
        if r.status != RunStatus.PASS or not r.metrics.pp_tps or not r.metrics.tg_tps:
            continue
        if r.metrics.benchmark_kind in {"full", "validation"}:
            cls = _result_vram_headroom(
                r, preferred_margin_mb=preferred_margin_mb,
                absolute_floor_mb=absolute_floor_mb, search_mode=search_mode,
            )["class"]
            if cls in {VramOperatingClass.REJECT.value, VramOperatingClass.FRAGILE.value}:
                continue
            if require_preferred_margin and int(r.metrics.vram_free_min_mb or 0) < preferred_margin_mb:
                continue
        passed.append(r)
    if not passed:
        return []
    if session_complete:
        validated = [
            r for r in passed
            if r.metrics.benchmark_kind == "validation" and r.metrics.long_context_passed
        ]
        if validated:
            return validated
        full = [r for r in passed if r.metrics.benchmark_kind == "full"]
        return full or passed
    # Partial sessions prefer full measurements when any exist, otherwise show probe-only provisional data.
    full = [r for r in passed if r.metrics.benchmark_kind in {"full", "validation"}]
    return full or passed


def _confidence(r: CandidateResult, session_complete: bool) -> str:
    if not session_complete:
        return "LOW"
    # Loading mmproj proves runtime compatibility and accounts for its memory, but a profile that
    # promises Vision should not receive HIGH/MEDIUM capability confidence unless the deterministic
    # FINAL image request was actually recognized.  Recognition remains non-gating for search.
    if r.candidate.vision and r.metrics.vision_test_passed is not True:
        return "LOW"
    if r.metrics.long_context_passed and r.metrics.stability_samples > 0 and r.metrics.stability_passed:
        return "HIGH"
    if r.metrics.long_context_passed:
        return "MEDIUM"
    return "LOW"


def build_profiles(results: list[CandidateResult], server_exe: str, model_path: str, caps,
                   preferred_margin_mb: int, session_complete: bool = True,
                   workload_profile: str = "agent", noise_policy: NoisePolicy | None = None,
                   absolute_vram_floor_mb: int = 300,
                   search_mode: str = "normal",
                   require_preferred_margin: bool = False,
                   preferred_candidate_key: str | None = None) -> list[LaunchProfile]:
    policy, _noise_diagnostics = calibrate_noise_policy(results, noise_policy or NoisePolicy())
    good = _candidate_pool(
        results, session_complete, preferred_margin_mb,
        absolute_floor_mb=absolute_vram_floor_mb, search_mode=search_mode,
        require_preferred_margin=require_preferred_margin,
    )
    degraded = []
    for r in _resolved_candidate_evidence(results):
        if r.status != RunStatus.PASS_DEGRADED or not r.metrics.tg_tps:
            continue
        if require_preferred_margin and int(r.metrics.vram_free_min_mb or 0) < preferred_margin_mb:
            continue
        if r.metrics.benchmark_kind in {"full", "validation"}:
            cls = _result_vram_headroom(
                r, preferred_margin_mb=preferred_margin_mb,
                absolute_floor_mb=absolute_vram_floor_mb, search_mode=search_mode,
            )["class"]
            if cls in {VramOperatingClass.REJECT.value, VramOperatingClass.FRAGILE.value}:
                continue
        degraded.append(r)
    provisional = not session_complete
    if not good:
        if not degraded:
            return []
        r = choose_preferred(degraded, workload_profile, policy) or degraded[0]
        return [LaunchProfile(
            "FALLBACK", r.candidate, r, "LOW" if provisional else "MEDIUM",
            "Provisional fallback from an incomplete search." if provisional else
            "The model is technically runnable, but measured interactive throughput is below the normal threshold.",
            build_server_command(server_exe, model_path, r.candidate, caps=caps), provisional=provisional,
        )]

    frontier = pareto_frontier(good, workload_profile, policy) or good

    # OPTIMAL keeps the existing noise-aware end-to-end policy.  In an incomplete session the
    # optimizer may already have selected a measured reconnaissance winner; keep that branch as
    # provisional OPTIMAL instead of re-sorting scouts into "highest KV at shortest context".
    # A chain of pairwise noise-band ties (e.g. a long MoE ncmoe funnel) can pareto-prune the
    # optimizer's own winner out of `frontier` even though nothing in `good` truly beats it, so
    # fall back to the unpruned pool before giving up on the measured branch.
    preferred = None
    if not session_complete and preferred_candidate_key:
        preferred = next((r for r in frontier if r.candidate.key() == preferred_candidate_key), None)
        if preferred is None:
            preferred = next((r for r in good if r.candidate.key() == preferred_candidate_key), None)
    optimal = preferred or choose_preferred(frontier, workload_profile, policy) or frontier[0]

    # Semantic roles may use solution-level scouts that were intentionally not FINAL-validated in
    # NORMAL mode.  Keep only runtime-successful, non-fragile evidence and collapse duplicate runs
    # of the same command to their strongest measurement.
    broad: dict[str, CandidateResult] = {}
    for r in _resolved_candidate_evidence(results):
        if r.status != RunStatus.PASS or not r.metrics.pp_tps or not r.metrics.tg_tps:
            continue
        cls = _result_vram_headroom(
            r, preferred_margin_mb=preferred_margin_mb,
            absolute_floor_mb=absolute_vram_floor_mb, search_mode=search_mode,
        )["class"]
        if cls in {VramOperatingClass.REJECT.value, VramOperatingClass.FRAGILE.value}:
            continue
        if require_preferred_margin and int(r.metrics.vram_free_min_mb or 0) < preferred_margin_mb:
            continue
        broad[r.candidate.key()] = r
    measured = list(broad.values()) or list(good)

    # MAX_KV_PRECISION selects the numerically strongest measured KV family first, then the most
    # useful noise-aware runtime inside that family.  It never claims that the selected GGUF's
    # weight quantization changed.
    best_kv_key = max(kv_precision_key(r.candidate.kv_k, r.candidate.kv_v) for r in measured)
    kv_pool = [r for r in measured if kv_precision_key(r.candidate.kv_k, r.candidate.kv_v) == best_kv_key]
    if kv_precision_key(optimal.candidate.kv_k, optimal.candidate.kv_v) == best_kv_key:
        # Do not manufacture a second launch command merely because a shorter scout exists in
        # the same precision tier.  OPTIMAL already provides the strongest measured KV precision.
        max_kv = optimal
    else:
        kv_frontier = pareto_frontier(kv_pool, workload_profile, policy) or kv_pool
        max_kv = choose_preferred(kv_frontier, workload_profile, policy) or kv_frontier[0]
    kv_info = kv_precision(max_kv.candidate.kv_k, max_kv.candidate.kv_v)

    # FASTEST uses robust short decode for the role users expect. A weak/cross-context scout must
    # clear the conservative 10% promotion band; otherwise the stronger OPTIMAL evidence wins.
    max_tg = max(robust_tg(r) for r in measured)
    decode_tied = [
        r for r in measured
        if max_tg - robust_tg(r) <= decode_noise_threshold(robust_tg(r), max_tg, policy, conservative=True)
    ]
    fastest_probe = choose_preferred(decode_tied, workload_profile, policy) or decode_tied[0]
    if fastest_probe.candidate.key() != optimal.candidate.key() and (
        robust_tg(fastest_probe) - robust_tg(optimal)
        <= decode_noise_threshold(robust_tg(optimal), robust_tg(fastest_probe), policy, conservative=True)
    ):
        fastest = optimal
    else:
        fastest = fastest_probe

    # MAX_CONTEXT is deliberately preserved as a user-facing launch role. It may be a LOW-confidence
    # scout when NORMAL chose to spend expensive FINAL validation on a different knee; the report says so.
    max_ctx_value = max(r.candidate.ctx for r in measured)
    max_ctx_pool = [r for r in measured if r.candidate.ctx == max_ctx_value]
    max_context = choose_preferred(max_ctx_pool, "long-context", policy) or max_ctx_pool[0]

    optimal_envelope = decode_speed_envelope(optimal, workload_profile)
    optimal_why = (
        f"Lowest representative {workload_profile} prefill+generation latency on the noise-aware Pareto frontier. "
        f"Measured decode is {optimal_envelope['validated_min_tps']:.1f}–"
        f"{optimal_envelope['validated_max_tps']:.1f} t/s; effective prefill is "
        f"{profile_prefill_tps(optimal, workload_profile):.0f} t/s and workload decode is "
        f"{profile_decode_tps(optimal, workload_profile):.1f} t/s."
    )
    kv_why = (
        f"Highest measured KV-cache/attention precision: {kv_info.label} "
        f"({max_kv.candidate.kv_k}/{max_kv.candidate.kv_v}). The selected GGUF weight quantization is fixed and unchanged. "
        "Within this KV tier, the noise-aware runtime winner is used."
    )
    fastest_why = (
        f"Highest materially distinct robust decode throughput in the measured search. Same-context confirmed "
        f"differences up to {policy.decode_rel:.0%} are noise; weak or cross-context evidence must clear "
        f"{policy.decode_probe_rel:.0%}. It may ignore the soft preferred reserve, but never the hard/operational "
        "recommendation floor; strict --require-vram-margin makes the preferred reserve mandatory. "
        "Long-context speed is shown separately from the short ceiling."
    )
    max_context_why = (
        f"Largest successfully measured configured context: {max_context.candidate.ctx} tokens. "
        "When several KV/placement choices reached it, the best long-context runtime was selected. "
        "If confidence is LOW, this is a scout command rather than a FINAL-validated promise."
    )
    entries: list[tuple[str, CandidateResult, str]] = [
        ("OPTIMAL", optimal, optimal_why),
        ("MAX_KV_PRECISION", max_kv, kv_why),
        ("FASTEST", fastest, fastest_why),
    ]
    # An interrupted run has only a "largest point seen so far", not a defensible MAX_CONTEXT
    # recommendation. Completed reports always expose the fourth role requested by the UI.
    if session_complete:
        entries.append(("MAX_CONTEXT", max_context, max_context_why))

    out: list[LaunchProfile] = []
    for name, r, why in entries:
        cmd = build_server_command(server_exe, model_path, r.candidate, caps=caps)
        if provisional:
            why = "PROVISIONAL: autotune did not finish. " + why
        out.append(LaunchProfile(
            name, r.candidate, r, _confidence(r, session_complete), why, cmd, provisional=provisional
        ))
    return out


def write_reports(out_dir: Path, profiles: list[LaunchProfile], results: list[CandidateResult], context: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_ctx = context.get("target_spec") or {}
    preferred_margin = int(context.get("preferred_vram_reserve_mb", 1024))
    absolute_floor = int(context.get("absolute_vram_floor_mb", 300))
    search_mode = str(context.get("search_mode", "normal"))
    scope = _measurement_scope(results, target_ctx)
    profile_groups = _group_profiles(profiles)
    recommended_profile_groups = [
        group for group in profile_groups if not all(_experimental_profile(p) for p in group)
    ]
    experimental_profile_groups = [
        group for group in profile_groups if all(_experimental_profile(p) for p in group)
    ]
    blob = {
        "context": context,
        "runtime_tuning_scope": {
            "model_weight_quantization": "fixed by the selected GGUF; not tuned",
            "kv_cache_precision": "runtime-tuned attention cache",
            "primary_kv_levels": ["f16/f16", "q8_0/q8_0", "q4_0/q4_0"],
            "mixed_kv_points_may_be_measured": True,
            "q8_automatic_policy": (
                "low-risk runtime sweet spot; may replace FP16 in BALANCED at equal context, "
                "without claiming universal semantic equivalence"
            ),
            "q4_automatic_policy": (
                "context-sensitive; long-context workloads require an occupied-cache throughput "
                "probe before automatic promotion, and task quality remains unproven"
            ),
        },
        "measurement_scope": scope,
        "profiles": [
            {"name": p.name, "candidate": asdict(p.candidate), "result": p.result.to_dict(),
             "purpose": PROFILE_PURPOSES.get(p.name, p.rationale),
             "kv_cache_precision": kv_precision(p.candidate.kv_k, p.candidate.kv_v).to_dict(),
             "confidence": p.confidence, "rationale": p.rationale, "command": p.command,
             "provisional": p.provisional,
             "recommendation_tier": "experimental" if _experimental_profile(p) else "recommended",
             "decode_speed": decode_speed_envelope(p.result, context.get("workload_profile", "agent")),
             "vram_headroom": _result_vram_headroom(
                 p.result, preferred_margin_mb=preferred_margin,
                 absolute_floor_mb=absolute_floor, search_mode=search_mode,
             ),
             "prompt_processing_speed": prefill_speed_envelope(
                 p.result, context.get("workload_profile", "agent")
             ),
             "estimated_cycle_seconds": workload_latency_seconds(
                 p.result, context.get("workload_profile", "agent")
             )}
            for p in profiles
        ],
        "profile_groups": [
            {
                "roles": [p.name for p in group],
                "purposes": [PROFILE_PURPOSES.get(p.name, p.rationale) for p in group],
                "candidate": asdict(group[0].candidate),
                "kv_cache_precision": kv_precision(
                    group[0].candidate.kv_k, group[0].candidate.kv_v
                ).to_dict(),
                "confidence": group[0].confidence,
                "provisional": group[0].provisional,
                "benchmark_kind": group[0].result.metrics.benchmark_kind,
                "recommendation_tier": (
                    "experimental" if all(_experimental_profile(p) for p in group) else "recommended"
                ),
                "decode_speed": decode_speed_envelope(
                    group[0].result, context.get("workload_profile", "agent")
                ),
                "prompt_processing_speed": prefill_speed_envelope(
                    group[0].result, context.get("workload_profile", "agent")
                ),
                "profile_effective_prefill_tps": profile_prefill_tps(
                    group[0].result, context.get("workload_profile", "agent")
                ),
                "profile_effective_decode_tps": profile_decode_tps(
                    group[0].result, context.get("workload_profile", "agent")
                ),
                "estimated_cycle_seconds": workload_latency_seconds(
                    group[0].result, context.get("workload_profile", "agent")
                ),
                "vram_headroom": _result_vram_headroom(
                    group[0].result, preferred_margin_mb=preferred_margin,
                    absolute_floor_mb=absolute_floor, search_mode=search_mode,
                ),
            }
            for group in profile_groups
        ],
        "runs": [r.to_dict() for r in results],
        "startup_diagnosis": {
            "model_became_ready": any(r.metrics.startup_seconds is not None for r in results),
            "configuration_feasibility_established": context.get("stop_reason") != "MODEL_STARTUP_FAILED",
            "startup_failure_count": sum(
                1 for r in results if r.reason in {"FAIL_STARTUP_STALL", "FAIL_STARTUP_TIMEOUT"}
            ),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    session_status = context.get("session_status", "UNKNOWN")
    stop_reason = context.get("stop_reason", "UNKNOWN")
    model_ctx = context.get("model") or {}
    feasibility = context.get("feasibility_plan") or {}
    target_status = context.get("target_status", "UNRESOLVED")
    selected_option = context.get("selected_option")
    lines = [
        "# Llama Autotuner Report", "",
        f"Session status: **{session_status}**",
        f"Stop reason: `{stop_reason}`",
        f"Environment: **{context.get('environment_state','unknown')}**",
        f"Workload profile: **{context.get('workload_profile','agent')}**",
        f"Decode decision bands: at the same context, confirmed difference **<={context.get('decode_noise_pct',5):g}% = noise**; "
        f"the intermediate band requires confirmation; SCOUT/cross-context promotion requires "
        f"**>{context.get('decode_probe_promotion_pct',10):g}%**; "
        f"prefill noise **±{context.get('prefill_noise_pct',10):g}%**",
        f"Completed candidate runs: **{len(results)}**",
        f"Autotune elapsed: **{float(context.get('elapsed_seconds', 0.0)):.1f} s**",
        f"Model architecture: `{model_ctx.get('architecture','unknown')}`",
        f"Model kind: `{model_ctx.get('kind','unknown')}`",
        f"Selected GGUF: `{model_ctx.get('filename') or Path(str(model_ctx.get('path') or 'unknown')).name}`",
        "Model-weight quantization: **fixed by the selected GGUF; the autotuner does not change it**",
        "Runtime attention/KV-cache precision: **tuned separately (FP16 / Q8 / Q4, plus optional measured mixed points)**",
        f"Model storage: `{float(model_ctx.get('size_bytes', 0)) / (1024**3):.2f} GiB`"
        + (f" across `{model_ctx.get('split_parts_found', 1)}/{model_ctx.get('split_count', 1)}` shards"
           if int(model_ctx.get('split_count', 1) or 1) > 1 else ""),
        f"Stored/main/MTP blocks: `{model_ctx.get('block_count','unknown')} / {model_ctx.get('main_block_count','unknown')} / {model_ctx.get('mtp_block_count',0)}`",
        f"MTP detected: `{model_ctx.get('has_mtp', False)}`",
        f"Vision hint: `{model_ctx.get('has_vision_hint', False)}`",
        f"Vision benchmark mode: `{model_ctx.get('vision_benchmark_mode', 'text-only')}`",
        *([f"Upstream fit oracle: `{context.get('fit_oracle', {}).get('raw_args', 'available')}` (advisory)"] if context.get('fit_oracle') else []),
        "",
    ]

    noise_calibration = context.get("noise_calibration") or {}
    if noise_calibration.get("calibrated"):
        lines += [
            "> **Adaptive noise bands widened from session evidence.** Configured thresholds remain the minimum; "
            "robust/repeated FULL measurements increased the effective bands shown above. The tuner never narrows "
            "the user's thresholds from a small sample.",
            "",
        ]
    if context.get("require_vram_margin"):
        lines += [
            f"> **Strict production reserve enabled:** only FULL/FINAL candidates with at least "
            f"`{preferred_margin} MiB` measured free VRAM are recommendation-eligible.",
            "",
        ]

    if stop_reason == "MODEL_STARTUP_FAILED":
        lines += [
            "## Startup diagnosis", "",
            "**No configuration was declared infeasible.** The child llama-server did not become healthy "
            "after the one widened retry, so PP/TG/VRAM feasibility was never measured.", "",
            "For a split GGUF this is treated as a model/runtime bootstrap problem. The tuner intentionally "
            "stopped instead of replaying every context and KV alternative with the same unopened model.", "",
            "Check the final candidate server log for a missing shard, unsupported architecture/build, storage "
            "or RAM pressure. If the command starts manually, its observed cold-start duration is the useful "
            "diagnostic for adjusting the startup budget.", "",
        ]

    summary_profile = next((p for p in profiles if p.name == "OPTIMAL"), profiles[0] if profiles else None)
    if summary_profile is not None:
        sm = summary_profile.result.metrics
        sd = decode_speed_envelope(summary_profile.result, context.get("workload_profile", "agent"))
        sv = _result_vram_headroom(
            summary_profile.result, preferred_margin_mb=preferred_margin,
            absolute_floor_mb=absolute_floor, search_mode=search_mode,
        )
        roles = next((g for g in profile_groups if summary_profile in g), [summary_profile])
        lines += [
            "## Practical result", "",
            f"**Recommended OPTIMAL configuration:** `{summary_profile.candidate.short()}`", "",
            f"**Practical validated decode range:** `{sd['validated_min_tps']:.2f}–{sd['validated_max_tps']:.2f} t/s`, depending on occupied context."
              if sd["validated_min_tps"] is not None and sd["validated_max_tps"] is not None
              else "**Measured decode range:** not available.",
            f"**Short/empty-context ceiling:** about `{sd['short_median_tps']:.2f} t/s`. Do not use this as the expected speed after a long prompt."
              if sd["short_median_tps"] is not None else "**Short-context speed:** not measured.",
            f"**Synthetic {context.get('workload_profile','agent')} ranking score:** `{sd['workload_weighted_tps']:.2f} t/s`. "
            "This is a workload model used for ranking, not a directly measured speed at one context size."
              if sd["workload_weighted_tps"] is not None else "**Synthetic workload ranking score:** not available.",
            "",
        ]
        if sm.context_staircase:
            last_row = sm.context_staircase[-1]
            last_tokens = int(last_row.get("prompt_tokens") or last_row.get("target_tokens") or 0)
            last_tg = float(last_row.get("tg_tps") or 0.0)
            lines += [
                f"**Longest measured operating point:** `{last_tg:.2f} t/s` at `{last_tokens:,}` occupied prompt tokens.", "",
                "### Speed by occupied context", "",
                "This table is the practical speed guide. Generation slows as the slot/KV cache fills.", "",
                "| Occupied context | Prompt processing | Decode |",
                "| ---: | ---: | ---: |",
            ]
            for row in sm.context_staircase:
                occupied = int(row.get("prompt_tokens") or row.get("target_tokens") or 0)
                pp = f"{float(row.get('pp_tps')):.2f} t/s" if row.get("pp_tps") is not None else "n/a"
                tg = f"{float(row.get('tg_tps')):.2f} t/s" if row.get("tg_tps") is not None else "n/a"
                lines.append(f"| {occupied:,} tokens | {pp} | **{tg}** |")
            lines += [
                "",
                "> Do not use the short-context TG as the expected speed for a long prompt. "
                "For example, a manual request starting between two measured rows should normally land near the corresponding part of this curve.",
                "",
            ]
        free_min = int(sm.vram_free_min_mb or 0)
        preferred_free = preferred_margin
        hard_floor = absolute_floor
        lines += [
            f"**VRAM operating class:** `{sv['class']}` — sampled minimum `{free_min} MiB`; "
            f"hard/tight/operational/preferred thresholds are `{sv['hard_floor_mb']} / "
            f"{sv['tight_floor_mb']} / {sv['operational_floor_mb']} / {sv['preferred_reserve_mb']} MiB`.",
            "",
        ]
        if sv["class"] == VramOperatingClass.TIGHT.value:
            lines += [
                "**TIGHT headroom:** this configuration passed the strong workload inside the explicit "
                "64 MiB hysteresis band. Its confirmed batch/ubatch is locked; do not increase ubatch, "
                "parallel slots or MTP state without re-tuning.", "",
            ]
        elif free_min < preferred_free:
            lines += [
                f"**VRAM headroom warning:** minimum sampled free VRAM was `{free_min} MiB`; "
                f"this is above the hard `{hard_floor} MiB` floor but below the preferred "
                f"`{preferred_free} MiB` reserve.", "",
            ]
        lines += [
            f"**Profile roles sharing this exact launch command:** `{', '.join(p.name for p in roles)}`.",
            "These names describe roles within the measured search scope; they do not claim that runtime KV precision changes the selected model's weight quantization.",
            "",
            "### What was actually searched", "",
            f"- Successfully measured contexts: `{', '.join(str(x) for x in scope['contexts_measured']) or 'none'}`.",
            f"- Successfully measured KV families: `{', '.join(scope['kv_families_measured']) or 'none'}`.",
            f"- MTP attempted: `{'yes' if scope['mtp_attempted'] else 'no'}`; text throughput measured with MTP: `{'yes' if scope['mtp_text_benchmarked'] else 'no'}`.",
            f"- Alternative context/KV trade-off families measured: `{'yes' if scope['tradeoff_families_measured'] else 'no'}`.",
        ]
        if not scope["tradeoff_families_measured"]:
            lines += [
                "- `FASTEST` is limited to the successfully benchmarked exact-target family because "
                "lower-context/KV alternatives were not measured in this session.",
            ]
        lines.append("")

    if recommended_profile_groups:
        lines += [
            "## Choose a launch profile", "",
            ("Completed reports expose four roles answering different questions. "
             "They may share one command when a single measured configuration wins multiple roles."
             if session_status == "COMPLETED" else
             "These are provisional roles from completed measurements so far; MAX_CONTEXT is withheld until the search completes."), "",
            "Measured ranges and profile-effective values are intentionally separate: the range is "
            "what the listed evidence observed, while effective PP/TG and cycle are workload-model ranking values. "
            "`validation` is FINAL evidence; `recon`/`quick` rows remain LOW-confidence scouts.", "",
            "| Roles | Purpose | Evidence | Confidence | Context | KV-cache precision | Placement | Measured decode | Longest measured | Cycle | VRAM |",
            "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for group in recommended_profile_groups:
            p = group[0]
            m = p.result.metrics
            decode = decode_speed_envelope(p.result, context.get("workload_profile", "agent"))
            vram = _result_vram_headroom(
                p.result, preferred_margin_mb=preferred_margin,
                absolute_floor_mb=absolute_floor, search_mode=search_mode,
            )
            placement = f"ncmoe={p.candidate.ncmoe}" if p.candidate.ncmoe is not None else f"ngl={p.candidate.ngl}"
            kv_info = kv_precision(p.candidate.kv_k, p.candidate.kv_v)
            if decode["validated_min_tps"] is not None and decode["validated_max_tps"] is not None:
                measured_decode = f"{decode['validated_min_tps']:.1f}–{decode['validated_max_tps']:.1f} t/s"
            else:
                measured_decode = "n/a"
            if decode["context_last_tps"] is not None and decode["context_curve"]:
                longest = f"{decode['context_last_tps']:.1f} t/s @ {decode['context_curve'][-1]['tokens']}"
            else:
                longest = "not measured"
            effective_pp = profile_prefill_tps(p.result, context.get("workload_profile", "agent"))
            effective_tg = profile_decode_tps(p.result, context.get("workload_profile", "agent"))
            cycle = workload_latency_seconds(p.result, context.get("workload_profile", "agent"))
            cycle_text = f"{cycle:.2f} s" if cycle != float("inf") else "n/a"
            roles_text = " / ".join(x.name for x in group)
            purposes = "<br>".join(PROFILE_PURPOSES.get(x.name, x.rationale) for x in group)
            lines.append(
                f"| {roles_text} | {purposes} | {m.benchmark_kind or 'unknown'} | {p.confidence} | {p.candidate.ctx} | "
                f"{kv_info.label} (`{p.candidate.kv_k}/{p.candidate.kv_v}`) | {placement} | {measured_decode} | "
                f"{longest} | {cycle_text} | "
                f"{m.vram_free_min_mb if m.vram_free_min_mb is not None else 'n/a'} MiB ({vram['class']}) |"
            )
        lines.append("")

    if experimental_profile_groups:
        lines += [
            "## Experimental / scout-only profile roles", "",
            "> **Do not treat these as ready-to-run recommendations.** They are retained to show the measured "
            "frontier, but LOW-confidence commands did not pass FINAL long-context validation. Run a dedicated "
            "validation before deployment.", "",
            "| Roles | Evidence | Context | KV | Decode sample | Free VRAM |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
        for group in experimental_profile_groups:
            p = group[0]
            roles_text = " / ".join(x.name for x in group)
            lines.append(
                f"| {roles_text} | {p.result.metrics.benchmark_kind or 'unknown'} / LOW | "
                f"{p.candidate.ctx} | {p.candidate.kv_k}/{p.candidate.kv_v} | "
                f"{robust_tg(p.result):.1f} t/s | {p.result.metrics.vram_free_min_mb or 0} MiB |"
            )
        lines.append("")

    lines += [
        "## Requested target", "",
        f"- Target status: **{target_status}**",
        f"- Requested context: `{target_ctx.get('context', 'unknown')}`",
        f"- Vision requirement: `{target_ctx.get('vision', 'off')}`",
        f"- Trade-off priority: `{target_ctx.get('priority', 'balanced')}`",
        f"- Preferred KV-cache precision: `{target_ctx.get('preferred_kv_k','f16')}/{target_ctx.get('preferred_kv_v','f16')}`",
        "- Model-weight quantization: fixed by the selected GGUF (not an optimizer variable)",
        f"- Degradation policy: `{target_ctx.get('degradation_policy','report')}`",
        f"- Static resource class: `{feasibility.get('resource_class','unknown')}`",
        "",
    ]
    if selected_option:
        degradations = selected_option.get("degradation") or ["none"]
        lines += [
            "### Selected solution option", "",
            f"- Name: `{selected_option.get('name','unknown')}`",
            f"- Context: `{selected_option.get('context','unknown')}`",
            f"- KV: `{selected_option.get('kv_k','unknown')}/{selected_option.get('kv_v','unknown')}`",
            f"- Strategy: `{selected_option.get('strategy','unknown')}`",
            f"- Degradation: `{', '.join(degradations)}`",
        ]
        for note in selected_option.get("degradation_notes") or []:
            lines.append(f"- Trade-off: {note}")
        lines.append("")
    options = feasibility.get("options") or []
    if options:
        lines += [
            "## Solution envelope", "",
            "Static planning alternatives. Runtime validation is authoritative; lower KV-cache precision is never treated as a free optimization, and model weights remain unchanged.", "",
            "| Rank | Option | Context | KV | Strategy | Predicted free | Resource | Degradation |",
            "| ---: | --- | ---: | --- | --- | ---: | --- | --- |",
        ]
        for opt in options:
            deg = ", ".join(opt.get("degradation") or ["none"])
            free = opt.get("predicted_free_mb")
            free_text = f"{free} MiB" if free is not None else "unknown"
            lines.append(
                f"| {opt.get('recommended_rank','?')} | {opt.get('name','unknown')} | {opt.get('context','?')} | "
                f"{opt.get('kv_k','?')}/{opt.get('kv_v','?')} | {opt.get('strategy','?')} | {free_text} | "
                f"{opt.get('resource_class','?')} | {deg} |"
            )
        lines.append("")

    scouts = [
        r for r in results
        if r.status in {RunStatus.PASS, RunStatus.PASS_DEGRADED}
        and r.metrics.benchmark_kind in {"recon", "recon-context"}
        and r.metrics.pp_tps and r.metrics.tg_tps
    ]
    if scouts:
        pref_k = target_ctx.get("preferred_kv_k", "f16")
        pref_v = target_ctx.get("preferred_kv_v", "f16")
        requested_ctx = int(target_ctx.get("context") or 0)
        # KV-precision-first presentation: preferred KV first (largest context first), then progressively
        # more memory-saving KV trade-offs. These are intentionally labelled SCOUT because only the
        # selected branch receives deep tuning/final validation in NORMAL mode.
        def scout_key(r):
            c = r.candidate
            precision = kv_precision_key(c.kv_k, c.kv_v)
            q = (-precision[0], -precision[1])
            strength = 0 if r.metrics.benchmark_kind == "recon-context" else 1
            return (*q, -c.ctx, strength, -(r.metrics.tg_tps or 0.0))
        scouts = sorted(scouts, key=scout_key)
        exact_moe = next((r for r in scouts if r.candidate.ctx == requested_ctx
                          and (r.candidate.kv_k, r.candidate.kv_v) == (pref_k, pref_v)
                          and r.candidate.ncmoe is not None), None)
        exact_ncmoe = exact_moe.candidate.ncmoe if exact_moe is not None else None
        lines += [
            "## Measured solution scouts", "",
            "Short solution-level measurements spanning KV-cache precision/context/speed trade-offs. "
            "Q4/mixed points may receive a deeper occupied-cache scout because their long-context runtime "
            "scaling cannot be inferred from a 1.2K prompt. These remain runtime tests, not semantic-quality evals.", "",
            "| Role | Context | KV | KV runtime risk | Placement | Residency delta | Scout | Filled tokens | PP t/s | TG t/s | Free VRAM | Context retained |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        seen = set()
        for r in scouts:
            c = r.candidate
            key = c.key()
            if key in seen:
                continue
            seen.add(key)
            if c.ctx == requested_ctx and (c.kv_k, c.kv_v) == (pref_k, pref_v):
                role = "EXACT_TARGET_SCOUT"
            elif (c.kv_k, c.kv_v) == (pref_k, pref_v):
                role = "KV_PRECISION_SCOUT"
            else:
                role = "TRADEOFF_SCOUT"
            placement = f"ncmoe={c.ncmoe}" if c.ncmoe is not None else f"ngl={c.ngl}"
            if exact_ncmoe is not None and c.ncmoe is not None:
                gained = int(exact_ncmoe) - int(c.ncmoe)
                residency = f"{gained:+d} expert layers on GPU"
            else:
                residency = "n/a"
            retained = (100.0 * c.ctx / requested_ctx) if requested_ctx else 0.0
            kv_info = kv_precision(c.kv_k, c.kv_v)
            filled = int(r.metrics.long_context_tokens or r.metrics.prompt_total_tokens or r.metrics.prompt_tokens or 0)
            lines.append(
                f"| {role} | {c.ctx} | {c.kv_k}/{c.kv_v} | {kv_info.runtime_risk} | {placement} | {residency} | "
                f"{r.metrics.benchmark_kind} | {filled} | {r.metrics.pp_tps:.1f} | {r.metrics.tg_tps:.1f} | "
                f"{r.metrics.vram_free_min_mb if r.metrics.vram_free_min_mb is not None else 'n/a'} MiB | {retained:.1f}% |"
            )
        lines.append("")
    if session_status != "COMPLETED":
        lines += [
            "> **Partial report.** The search did not reach final long-context validation. "
            "Any launch profiles below are provisional and must not be treated as the final optimum.",
            "",
        ]

    for group in profile_groups:
        p = group[0]
        experimental = all(_experimental_profile(x) for x in group)
        role_names = [x.name for x in group]
        m = p.result.metrics
        decode = decode_speed_envelope(p.result, context.get('workload_profile', 'agent'))
        prefill = prefill_speed_envelope(p.result, context.get('workload_profile', 'agent'))
        vram = _result_vram_headroom(
            p.result, preferred_margin_mb=preferred_margin,
            absolute_floor_mb=absolute_floor, search_mode=search_mode,
        )
        prefix = "[EXPERIMENTAL — FINAL REQUIRED] " if experimental else ("[PROVISIONAL] " if p.provisional else "")
        heading = f"## {prefix}Configuration — {' / '.join(role_names)}"
        lines += [heading, ""]
        if experimental:
            lines += [
                "> **Scout-only diagnostic.** This command is intentionally excluded from the main recommendation "
                "table because its confidence is LOW and it has no passing FINAL evidence.", "",
            ]
        for role in group:
            lines.append(f"- **{role.name}:** {role.rationale}")
        lines += ["",
                  f"- Context: `{p.candidate.ctx}`",
                  f"- **KV-cache / attention precision:** `{kv_precision(p.candidate.kv_k, p.candidate.kv_v).label}` "
                  f"(`{p.candidate.kv_k}/{p.candidate.kv_v}`)",
                  f"- **KV automatic-policy risk:** `{kv_precision(p.candidate.kv_k, p.candidate.kv_v).runtime_risk}` "
                  "(runtime classification; not a semantic-quality guarantee)",
                  "- **Model-weight quantization:** unchanged; fixed by the selected GGUF file",
                  f"- Benchmark kind: `{m.benchmark_kind or 'unknown'}`",
                  f"- **Practical measured decode range:** `{decode['validated_min_tps']:.2f}–{decode['validated_max_tps']:.2f} t/s`"
                    if decode['validated_min_tps'] is not None and decode['validated_max_tps'] is not None else "- Validated decode envelope: n/a",
                  f"- **Short/low-occupancy ceiling (median / p10; not long-context speed):** `{decode['short_median_tps']:.2f} / {decode['short_p10_tps']:.2f} t/s`"
                    if decode['short_median_tps'] is not None and decode['short_p10_tps'] is not None else "- Typical short decode: n/a",
                  f"- **Context-staircase decode range:** `{decode['context_min_tps']:.2f}–{decode['context_max_tps']:.2f} t/s`"
                    if decode['context_min_tps'] is not None and decode['context_max_tps'] is not None else "- Context-staircase decode range: not measured",
                  f"- **Longest measured context point:** `{decode['context_last_tps']:.2f} t/s` at `{decode['context_curve'][-1]['tokens']}` tokens"
                    if decode['context_last_tps'] is not None and decode['context_curve'] else "- Longest measured context point: n/a",
                  f"- Short/low-occupancy validation TG: `{m.tg_tps:.2f} t/s` (not the expected long-context speed)" if m.tg_tps else "- Short validation TG sample: n/a",
                  f"- Measured prompt-processing range: `{prefill['validated_min_tps']:.2f}–{prefill['validated_max_tps']:.2f} t/s`"
                    if prefill['validated_min_tps'] is not None and prefill['validated_max_tps'] is not None else "- Validated prompt-processing envelope: n/a",
                  f"- Short/full PP sample: `{m.pp_tps:.2f} t/s`" if m.pp_tps else "- Short/full PP sample: n/a",
                  f"- Effective context-fill PP: `{context_fill_pp(p.result):.2f} t/s`" if context_fill_pp(p.result) else "- Effective context-fill PP: n/a",
                  f"- Synthetic profile-effective prefill: `{profile_prefill_tps(p.result, context.get('workload_profile','agent')):.2f} t/s`" if profile_prefill_tps(p.result, context.get('workload_profile','agent')) else "- Synthetic profile-effective prefill: n/a",
                  f"- Synthetic profile-effective decode: `{profile_decode_tps(p.result, context.get('workload_profile','agent')):.2f} t/s`" if profile_decode_tps(p.result, context.get('workload_profile','agent')) else "- Synthetic profile-effective decode: n/a",
                  f"- Synthetic estimated {context.get('workload_profile','agent')} prefill+generation cycle: `{workload_latency_seconds(p.result, context.get('workload_profile','agent')):.2f} s`" if workload_latency_seconds(p.result, context.get('workload_profile','agent')) != float('inf') else "- Synthetic estimated workload cycle: n/a",
                  f"- Robust TG median / p10 / p90: `{m.stability_tg_median:.2f} / {m.stability_tg_p10:.2f} / {m.stability_tg_p90:.2f} t/s`"
                    if m.stability_tg_median is not None and m.stability_tg_p10 is not None and m.stability_tg_p90 is not None else "- Robust TG: not measured",
                  f"- Robust TG range: `{m.stability_tg_min:.2f}–{m.stability_tg_max:.2f} t/s`"
                    if m.stability_tg_min is not None and m.stability_tg_max is not None else "- Robust TG range: n/a",
                  f"- Stability variation: `{m.stability_variation_pct:.1f}%`"
                    if m.stability_variation_pct is not None else "- Stability variation: n/a",
                  f"- TG ↔ mean draft len correlation: `{m.stability_tg_mean_len_corr:+.3f}`"
                    if m.stability_tg_mean_len_corr is not None else "- TG ↔ mean draft len correlation: n/a",
                  f"- TG ↔ acceptance correlation: `{m.stability_tg_acceptance_corr:+.3f}`"
                    if m.stability_tg_acceptance_corr is not None else "- TG ↔ acceptance correlation: n/a",
                  f"- Vision recognition (FINAL diagnostic, non-gating): `{'PASS' if m.vision_test_passed else 'FAIL'} ({m.vision_latency_seconds:.2f} s, answer={m.vision_answer!r})`"
                    if m.vision_test_passed is not None and m.vision_latency_seconds is not None else "- Vision recognition: not requested/measured",
                  f"- Vision diagnostic error: `{m.vision_error}`" if m.vision_error else "- Vision diagnostic error: n/a",
                  f"- VRAM free minimum (sampled): `{m.vram_free_min_mb} MiB`",
                  f"- **VRAM operating class:** `{vram['class']}` "
                  f"(hard/tight/operational/preferred: {vram['hard_floor_mb']}/"
                  f"{vram['tight_floor_mb']}/{vram['operational_floor_mb']}/"
                  f"{vram['preferred_reserve_mb']} MiB)",
                  f"- MTP acceptance (short sample): `{(m.acceptance or 0)*100:.1f}%`" if p.candidate.mtp else "- MTP: off",
                  f"- MTP stability acceptance median: `{m.stability_acceptance_median*100:.1f}%`"
                    if p.candidate.mtp and m.stability_acceptance_median is not None else "- MTP stability acceptance median: n/a",
                  f"- MTP mean draft len median / p10: `{m.stability_mean_draft_len_median:.2f} / {m.stability_mean_draft_len_p10:.2f}`"
                    if p.candidate.mtp and m.stability_mean_draft_len_median is not None and m.stability_mean_draft_len_p10 is not None else "- MTP mean draft len: n/a",
                  f"- Context staircase TG retained (last/first): `{m.context_tg_ratio*100:.1f}%`"
                    if m.context_tg_ratio is not None else "- Context staircase TG retained: not measured",
                  f"- Robustness verdict: `{'PASS' if m.stability_passed else 'WARN'}`"
                    if m.stability_samples else "- Robustness verdict: not measured",
                  f"- Long-context validation: `PASS ({m.long_context_tokens} tokens, {m.long_context_pp_tps:.2f} t/s)`"
                    if m.long_context_passed and m.long_context_pp_tps else "- Long-context validation: not completed",
                  f"- Confidence: **{p.confidence}**", "",
                  "### Diagnostic command — validate before use" if experimental else "### Ready-to-run llama.cpp command", "",
                  "This command requires a dedicated FINAL/stress validation before deployment."
                    if experimental else "Copy this command as-is. Paths containing spaces are already quoted.", "", "```powershell",
                  " ".join(f'\"{x}\"' if " " in x else x for x in p.command), "```", ""]

        if p.result.metrics.benchmark_kind == "validation":
            if p.candidate.ncmoe is not None:
                lines += [
                    "### MoE operating regime", "",
                    f"- Expert residency: `ncmoe={p.candidate.ncmoe}`; this placement is coupled to context, KV and ubatch.",
                    "- A lower-precision KV cache is a memory trade-off, not a presumed speed win; only measured PP/TG can promote it.",
                    "- MTP peak, robust short speed and long-context floor are reported separately because draft acceptance is workload-dependent.",
                    "",
                ]
            elif p.candidate.ngl != "all":
                lines += [
                    "### Model / quant fit verdict", "",
                    "- **Strongly oversized for this GPU:** the selected Dense configuration requires numeric layer offload.",
                    f"- Measured decode envelope: `{decode['validated_min_tps']:.2f}–{decode['validated_max_tps']:.2f} t/s`."
                      if decode['validated_min_tps'] is not None else "- Measured decode envelope: unavailable.",
                    "- A smaller quant that becomes fully GPU-resident is likely preferable for interactive use; validate it rather than inferring from bits-per-weight alone.",
                    "",
                ]

        if m.stability_workloads:
            lines += ["### Decode stability samples", "", "| Workload | TG t/s | Acceptance | Mean draft len |",
                      "| --- | ---: | ---: | ---: |"]
            for row in m.stability_workloads:
                tg = f"{row.get('tg_tps'):.2f}" if row.get('tg_tps') is not None else "n/a"
                acc = f"{row.get('acceptance')*100:.1f}%" if row.get('acceptance') is not None else "n/a"
                mean_len = f"{row.get('mean_draft_len'):.2f}" if row.get('mean_draft_len') is not None else "n/a"
                lines.append(f"| {row.get('name','unknown')} | {tg} | {acc} | {mean_len} |")
            lines.append("")

        if m.context_staircase:
            lines += ["### Context staircase", "",
                      "| Target | Total prompt | Processed delta | Cached | PP t/s | TG t/s | Acceptance | Mean draft len |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for row in m.context_staircase:
                pp = f"{row.get('pp_tps'):.2f}" if row.get('pp_tps') is not None else "n/a"
                tg = f"{row.get('tg_tps'):.2f}" if row.get('tg_tps') is not None else "n/a"
                acc = f"{row.get('acceptance')*100:.1f}%" if row.get('acceptance') is not None else "n/a"
                mean_len = f"{row.get('mean_draft_len'):.2f}" if row.get('mean_draft_len') is not None else "n/a"
                lines.append(
                    f"| {row.get('target_tokens',0)} | {row.get('prompt_tokens',0)} | "
                    f"{row.get('processed_prompt_tokens', row.get('prompt_tokens',0))} | "
                    f"{row.get('cached_prompt_tokens',0)} | {pp} | {tg} | {acc} | {mean_len} |"
                )
            lines.append("")

    rejected = [r for r in results if r.status not in {RunStatus.PASS, RunStatus.PASS_DEGRADED}]
    if rejected:
        lines += ["## Rejected / failed candidates", ""]
        for r in rejected:
            extra = ""
            if r.metrics.early_pp_tps and r.metrics.final_pp_tps:
                extra = f" (PP {r.metrics.early_pp_tps:.1f} → {r.metrics.final_pp_tps:.1f} t/s)"
            lines.append(f"- `{r.candidate.key()}` — **{r.status.value}** / `{r.reason}`{extra}")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
