from __future__ import annotations

import copy
import threading
import time
from math import ceil
from pathlib import Path
from statistics import median
from typing import Callable

import psutil

from llama_autotuner.benchmark.runner import (
    benchmark_dense_partial_screen, benchmark_quick, benchmark_recon, benchmark_recon_context,
    benchmark_staged, benchmark_stability, validate_long_context,
)
from llama_autotuner.benchmark.vision import benchmark_vision_recognition, bundled_vision_asset
from llama_autotuner.hardware.nvidia import NvidiaSmiBackend
from llama_autotuner.llama.command import build_server_command
from llama_autotuner.llama.server import ServerRunner, ServerStartError
from llama_autotuner.models import Candidate, CandidateResult, HardwareInfo, ModelInfo, ModelKind, RunStatus
from llama_autotuner.tuning.kv import (
    balanced_context_gain_required, kv_context_probe_tokens, kv_degradation_ladder,
    kv_precision, kv_precision_key, kv_requires_long_context_probe,
)
from llama_autotuner.tuning.target import DegradationKind, ResourceClass, SolutionOption
from llama_autotuner.tuning.scoring import (
    NoisePolicy, calibrate_noise_policy, choose_preferred, decode_noise_threshold, decode_relation, decode_requires_confirmation,
    latency_relation, noise_aware_dominates,
    pareto_frontier, performance_equivalent, prefill_relation, profile_decode_tps, profile_prefill_tps, robust_tg,
    workload_latency_seconds,
)
from llama_autotuner.tuning.static_memory import (
    StaticMemoryEstimate, dense_seed_order, estimate_candidate_free_mb, estimate_static_memory,
    format_static_estimate, moe_seed_order, model_main_block_count,
)
from llama_autotuner.tuning.vram import VramOperatingClass, VramThresholds, vram_thresholds

ProgressFn = Callable[[str], None] | None
ResultFn = Callable[[CandidateResult], None] | None


class AutotuneEngine:
    """Adaptive llama-server autotuner.

    The engine deliberately separates a cheap placement probe from full 10K-ish server validation.
    For MoE, it starts from a safe expert-offload point and moves toward more GPU-resident experts
    until it brackets the memory-pressure boundary, then refines that bracket with full workloads.
    """

    def __init__(self, server_exe: str, model: ModelInfo, hardware: HardwareInfo, caps,
                 results_dir: Path, baseline_vram_mb: int, vram_margin_mb: int = 1024,
                 port: int = 8080, max_runs: int = 40, max_minutes: int = 45,
                 progress: ProgressFn = None, on_result: ResultFn = None,
                 heartbeat_seconds: float = 5.0, search_mode: str = "normal",
                 absolute_vram_floor_mb: int = 300, severe_regression_ratio: float = 0.50,
                 base_extra_args: list[str] | None = None, safe_perf_floor_ratio: float = 0.80,
                 workload_profile: str = "agent", noise_policy: NoisePolicy | None = None,
                 min_tg_tps: float | None = None, min_pp_tps: float | None = None,
                 min_tg_is_default: bool = False,
                 selection_priority: str = "balanced",
                 require_preferred_vram_reserve: bool = False,
                 server_lease_dir: Path | None = None,
                 prior_calibration_mb: dict[str, int] | None = None) -> None:
        self.server_exe = server_exe
        self.model = model
        self.hardware = hardware
        self.caps = caps
        self.results_dir = results_dir
        self.baseline_vram_mb = int(baseline_vram_mb)
        self.initial_baseline_vram_mb = int(baseline_vram_mb)
        self.baseline_rebases: list[dict[str, int]] = []
        self.vram_margin_mb = vram_margin_mb
        self.port = port
        self.max_runs = max_runs
        # Phase 0 time-aware budget: max_runs is a soft target, not a hard ceiling, as long as
        # the remaining max_minutes budget can plausibly fit another candidate. Bounded by this
        # multiplier so a session cannot run away if per-candidate cost unexpectedly shrinks.
        self._run_budget_ceiling_multiplier = 2.0
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + max_minutes * 60
        self.max_minutes = max_minutes
        self.gpu = NvidiaSmiBackend()
        self.results: list[CandidateResult] = []
        self.progress = progress
        self.on_result = on_result
        self.heartbeat_seconds = max(2.0, heartbeat_seconds)
        self.search_mode = search_mode
        self.absolute_vram_floor_mb = max(128, absolute_vram_floor_mb)
        self.severe_regression_ratio = min(0.90, max(0.10, severe_regression_ratio))
        self.base_extra_args = list(base_extra_args or [])
        self.safe_perf_floor_ratio = min(0.95, max(0.50, safe_perf_floor_ratio))
        self.workload_profile = workload_profile if workload_profile in {"chat", "agent", "long-context"} else "agent"
        self.base_noise_policy = noise_policy or NoisePolicy()
        self.noise_policy = self.base_noise_policy
        self.noise_calibration: dict = {
            "calibrated": False,
            "base_decode_rel": self.base_noise_policy.decode_rel,
            "effective_decode_rel": self.base_noise_policy.decode_rel,
            "base_decode_probe_rel": self.base_noise_policy.decode_probe_rel,
            "effective_decode_probe_rel": self.base_noise_policy.decode_probe_rel,
            "base_prefill_rel": self.base_noise_policy.prefill_rel,
            "effective_prefill_rel": self.base_noise_policy.prefill_rel,
            "sources": [],
        }
        self.min_tg_tps = min_tg_tps if min_tg_tps is None else max(0.0, float(min_tg_tps))
        self.min_pp_tps = min_pp_tps if min_pp_tps is None else max(0.0, float(min_pp_tps))
        # True when min_tg_tps came from the workload-based default rather than an explicit
        # user --min-tg. Some optimizations (e.g. deferring a redundant MoE FULL confirmation)
        # only make sense when the user gave no target to validate against; the automatic
        # degraded-branch abandonment gate should still apply to the default.
        self._min_tg_is_default = bool(min_tg_is_default) and self.min_tg_tps is not None
        self.selection_priority = selection_priority if selection_priority in {"balanced", "context", "quality", "speed"} else "balanced"
        self.require_preferred_vram_reserve = bool(require_preferred_vram_reserve)
        self.server_lease_dir = Path(server_lease_dir) if server_lease_dir is not None else None
        # Static VRAM corrections are placement-family specific. A numeric Dense CPU-offload
        # probe can differ by multiple GiB from the full-GPU estimate and must never calibrate
        # future ngl=all/MTP predictions.
        self._static_free_corrections_mb: dict[str, int] = {}
        self._dense_numeric_correction_samples_mb: dict[str, int] = {}
        # Hardware-scoped priors from earlier sessions (any model, same GPU). Used only as a
        # fallback seed before this session has measured its own correction for a given
        # placement family; a real in-session measurement always takes precedence (see
        # ``_calibrate_from_result``, which never overwrites an already-session-calibrated key).
        self._prior_calibration_mb: dict[str, int] = dict(prior_calibration_mb or {})
        self._dense_oversized_full_penalty_mb: dict[tuple[str, str], int] = {}
        self._tight_candidate_keys: set[str] = set()
        self._mtp_overhead_calibration_done = False
        self._last_dense_mtp_outcome = "NOT_RUN"
        self._preferred_mtp_key: str | None = None
        # NORMAL MoE can retain an MTP candidate solely as a validated FASTEST alternative even
        # when its slower prefill prevents it from winning the representative workload.  Such a
        # speed-only branch must not unlock another ubatch/p-min sweep.
        self._mtp_speed_only = False
        self.phase = "INITIALIZING"
        self.completed = False
        self.stop_reason = "RUNNING"
        self._static_estimate: StaticMemoryEstimate | None = None
        self.selected_option: SolutionOption | None = None
        self.target_status: str = "UNRESOLVED"
        self.provisional_recommendation_key: str | None = None
        self._solution_options_ordered: list[SolutionOption] = []
        self._declared_target_ctx: int | None = None
        self._declared_kv: tuple[str, str] = ("f16", "f16")
        self._vision_core_notice_emitted = False
        # A repeated startup timeout for a complete split model is a model/runtime
        # bootstrap failure, not evidence that a different KV/context candidate is
        # infeasible.  Preserve that distinction so tune() does not replay every
        # semantic fallback with the same unopened model.
        self._startup_blocker: str | None = None
        self._split_startup_notice_emitted = False
        # v0.6.0: Dense models whose weights cannot become full-GPU even at tiny context need
        # a different optimizer. Phase 0 maps context × KV × numeric-ngl first and stores the
        # measured placement so later phases do not restart the old exhaustive placement walk.
        self._dense_oversized_active = False
        self._dense_oversized_evidence: dict[tuple[int, str, str, int], CandidateResult] = {}

    def _emit(self, message: str) -> None:
        if self.progress:
            self.progress(message)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def budget_reason(self) -> str | None:
        if self._startup_blocker:
            return "MODEL_STARTUP_FAILED"
        if time.monotonic() >= self.deadline:
            return "TIME_BUDGET_REACHED"
        if len(self.results) >= self.max_runs and not self._time_remaining_for_another_run():
            return "RUN_BUDGET_REACHED"
        return None

    def _time_remaining_for_another_run(self) -> bool:
        """True once ``max_runs`` is reached only if the remaining max-minutes budget can
        plausibly fit one more candidate, estimated from the average cost of runs so far, and
        the bounded run-count ceiling has not been hit. Without this, a search that is genuinely
        time-rich (e.g. MoE+MTP finishing 12/12 runs with several minutes still unused) stops on
        run count alone instead of on the budget the user actually configured."""
        if not self.results:
            return False
        if len(self.results) >= self.max_runs * self._run_budget_ceiling_multiplier:
            return False
        avg_run_seconds = self.elapsed_seconds() / len(self.results)
        remaining_seconds = self.deadline - time.monotonic()
        return remaining_seconds > avg_run_seconds

    def budget_ok(self) -> bool:
        reason = self.budget_reason()
        if reason:
            self.stop_reason = reason
            return False
        return True

    def mark_interrupted(self) -> None:
        self.completed = False
        self.stop_reason = "USER_CANCELLED"

    def _record(self, result: CandidateResult) -> CandidateResult:
        self._annotate_vram_policy(result)
        self.results.append(result)
        self._calibrate_from_result(result)
        self._refresh_noise_policy()
        if self.on_result:
            try:
                self.on_result(result)
            except Exception as exc:
                self._emit(f"WARNING: could not persist candidate immediately: {exc}")
        return result

    def _refresh_noise_policy(self) -> None:
        previous = self.noise_policy
        policy, diagnostics = calibrate_noise_policy(self.results, self.base_noise_policy)
        self.noise_policy = policy
        self.noise_calibration = diagnostics
        if (
            abs(policy.decode_rel - previous.decode_rel) >= 0.005
            or abs(policy.prefill_rel - previous.prefill_rel) >= 0.005
        ):
            self._emit(
                "  adaptive noise calibration: confirmed decode band "
                f"{previous.decode_rel:.1%}→{policy.decode_rel:.1%}, probe/cross-context "
                f"{previous.decode_probe_rel:.1%}→{policy.decode_probe_rel:.1%}, "
                f"prefill {previous.prefill_rel:.1%}→{policy.prefill_rel:.1%}."
            )

    def _static_correction_key(self, candidate: Candidate) -> str | None:
        if self.model.kind == ModelKind.DENSE:
            # MTP still uses the plain full-GPU correction for the same target placement; the
            # speculative overhead itself is estimated separately. Numeric ngl values get their
            # own keys and can never poison ngl=all.
            return "dense:all" if candidate.ngl == "all" else f"dense:ngl:{candidate.ngl}"
        if self.model.kind == ModelKind.MOE and candidate.ncmoe is not None:
            # Preserve v0.5.5 MoE behavior for now; this release intentionally focuses on Dense.
            return "moe"
        return None

    def _calibrate_from_result(self, result: CandidateResult) -> None:
        """Turn static VRAM into a placement-scoped session prior.

        Dense ``ngl=24`` and ``ngl=all`` are different execution/memory families. v0.5.5 used
        one global correction, so a heavily CPU-offloaded 262K scout could add +2-3 GiB to later
        full-GPU/MTP predictions. Keep corrections local to the exact placement family instead.
        """
        free = result.metrics.vram_free_min_mb
        if free is None:
            return

        if not result.candidate.mtp:
            key = self._static_correction_key(result.candidate)
            if key and key not in self._static_free_corrections_mb:
                predicted_value = estimate_candidate_free_mb(
                    self.model, self.hardware, self.baseline_vram_mb, result.candidate
                )
                if predicted_value is not None:
                    predicted = int(predicted_value)
                    correction = int(free - predicted)
                    self._static_free_corrections_mb[key] = correction
                    sign = "+" if correction >= 0 else ""
                    self._emit(
                        f"  static calibration [{key}]: predicted free≈{predicted} MiB, measured={free} MiB; "
                        f"session correction={sign}{correction} MiB."
                    )

                    if (self.model.kind == ModelKind.DENSE and result.candidate.ngl != "all"
                            and result.metrics.benchmark_kind in {"recon", "quick"}):
                        sample_key = result.candidate.key()
                        self._dense_numeric_correction_samples_mb.setdefault(sample_key, correction)
                        global_correction = int(round(median(self._dense_numeric_correction_samples_mb.values())))
                        self._emit(
                            f"  numeric-Dense global calibration: median correction={global_correction:+d} MiB "
                            f"from {len(self._dense_numeric_correction_samples_mb)} distinct scout(s)."
                        )

        if result.candidate.mtp and self.model.kind == ModelKind.DENSE and not self._mtp_overhead_calibration_done:
            peers = [
                r for r in self.results
                if (not r.candidate.mtp and r.candidate.ngl == "all"
                    and r.candidate.ctx == result.candidate.ctx
                    and r.candidate.ubatch == result.candidate.ubatch
                    and r.metrics.vram_free_min_mb is not None
                    and self._is_good(r))
            ]
            if peers:
                plain = max(peers, key=lambda r: r.metrics.vram_free_min_mb or -1)
                overhead = int((plain.metrics.vram_free_min_mb or 0) - free)
                self._mtp_overhead_calibration_done = True
                self._emit(
                    f"  MTP calibration: measured full-GPU speculative overhead≈{overhead} MiB "
                    f"at ubatch={result.candidate.ubatch}."
                )

    def _predicted_free_for(self, candidate: Candidate) -> int | None:
        predicted = estimate_candidate_free_mb(self.model, self.hardware, self.baseline_vram_mb, candidate)
        if predicted is None:
            return None
        key = self._static_correction_key(candidate)
        if key is not None and key in self._static_free_corrections_mb:
            predicted += self._static_free_corrections_mb[key]
        elif (self.model.kind == ModelKind.DENSE and candidate.ngl != "all"
              and self._dense_numeric_correction_samples_mb):
            predicted += int(round(median(self._dense_numeric_correction_samples_mb.values())))
        elif key is not None and key in self._prior_calibration_mb:
            # No measurement yet this session: fall back to a prior session's correction for
            # this exact GPU, still scoped to the same placement family.
            predicted += self._prior_calibration_mb[key]
        elif (self.model.kind == ModelKind.DENSE and candidate.ngl != "all"
              and "dense:numeric_median" in self._prior_calibration_mb):
            predicted += self._prior_calibration_mb["dense:numeric_median"]
        return predicted

    def exportable_calibration_mb(self) -> dict[str, int]:
        """Session corrections worth persisting per-hardware for a future session's cold start.

        Only placement-family-level keys generalize across different models on the same GPU
        (an exact numeric ``dense:ngl:N`` key does not: N is model-specific layer count)."""
        out = {k: v for k, v in self._static_free_corrections_mb.items() if k in {"dense:all", "moe"}}
        if self._dense_numeric_correction_samples_mb:
            out["dense:numeric_median"] = int(round(median(self._dense_numeric_correction_samples_mb.values())))
        return out

    def _vram_thresholds(self, candidate: Candidate) -> VramThresholds:
        return vram_thresholds(
            absolute_floor_mb=self.absolute_vram_floor_mb,
            preferred_reserve_mb=self.vram_margin_mb,
            search_mode=self.search_mode,
            model_kind=self.model.kind,
            vision=candidate.vision,
        )

    def _vram_class(self, result: CandidateResult) -> VramOperatingClass:
        return self._vram_thresholds(result.candidate).classify(result.metrics.vram_free_min_mb)

    def _annotate_vram_policy(self, result: CandidateResult) -> None:
        thresholds = self._vram_thresholds(result.candidate)
        operating_class = thresholds.classify(result.metrics.vram_free_min_mb)
        result.metrics.vram_operating_class = operating_class.value
        result.metrics.vram_hard_floor_mb = thresholds.hard_floor_mb
        result.metrics.vram_tight_floor_mb = thresholds.tight_floor_mb
        result.metrics.vram_operational_floor_mb = thresholds.operational_floor_mb
        result.metrics.vram_preferred_reserve_mb = thresholds.preferred_reserve_mb

    def _is_recommendable_full(self, result: CandidateResult | None) -> bool:
        if result is None or not self._is_good(result):
            return False
        if result.metrics.benchmark_kind not in {"full", "validation"}:
            return False
        if self._vram_class(result) in {
            VramOperatingClass.REJECT, VramOperatingClass.FRAGILE,
        }:
            return False
        if self.require_preferred_vram_reserve:
            return int(result.metrics.vram_free_min_mb or 0) >= int(self.vram_margin_mb)
        return True

    def _mark_tight(self, result: CandidateResult) -> None:
        if self._vram_class(result) == VramOperatingClass.TIGHT:
            self._tight_candidate_keys.add(result.candidate.key())

    def _is_tight(self, result: CandidateResult | None) -> bool:
        return result is not None and self._vram_class(result) == VramOperatingClass.TIGHT

    def _llama_compute_process_present(self) -> bool:
        """Best-effort guard against adopting a leaked llama-server allocation as idle."""
        target = Path(self.server_exe).name.lower()
        try:
            processes = self.gpu.compute_processes()
        except (AttributeError, OSError, RuntimeError):
            return False
        for process in processes:
            raw_name = str(process.get("name", "")).replace("\\", "/")
            name = raw_name.rsplit("/", 1)[-1].lower()
            if name == target or "llama-server" in name:
                return True
        return False

    def _adopt_idle_baseline_drift(self, samples: list, *, tolerance_mb: int) -> bool:
        """Accept a small, stable Windows/WDDM idle drift without hiding a real workload.

        Browser/desktop allocations and a released CUDA context can leave a few hundred MiB more
        than the preflight median.  v0.6.8 treated 251 MiB and 2 GiB identically, so a stable
        422→711 MiB drift aborted both MTP and FINAL after all useful tuning had completed.

        Re-basing is deliberately narrow: several stable low-utilization samples, no visible
        llama-server compute process, and at most roughly 3% of VRAM (capped at 640 MiB).  Candidate
        headroom continues to use absolute measured free VRAM, so hard/operational floors are not
        weakened.
        """
        if len(samples) < 4:
            return False
        tail = samples[-6:]
        used = [int(s.used_mb) for s in tail]
        util = [float(s.util_percent) for s in tail]
        new_baseline = int(round(median(used)))
        delta = new_baseline - int(self.baseline_vram_mb)
        drift_limit = max(384, min(640, int(round(self.hardware.vram_total_mb * 0.03))))
        if delta <= tolerance_mb or delta > drift_limit:
            return False
        if max(used) - min(used) > 96 or median(util) > 8.0:
            return False
        if self._llama_compute_process_present():
            return False

        old_baseline = int(self.baseline_vram_mb)
        self.baseline_vram_mb = new_baseline
        self.baseline_rebases.append({
            "from_mb": old_baseline,
            "to_mb": new_baseline,
            "delta_mb": delta,
        })
        self._emit(
            f"  stable idle VRAM drift accepted: baseline {old_baseline}→{new_baseline} MiB "
            f"(Δ{delta} MiB, low utilization, no llama-server process)."
        )
        return True

    def _wait_clean(self, timeout: float = 15.0, tolerance_mb: int = 250) -> bool:
        start = time.monotonic()
        end = start + timeout
        warned = False
        samples = []
        while time.monotonic() < end:
            s = self.gpu.snapshot()
            samples.append(s)
            if s.used_mb <= self.baseline_vram_mb + tolerance_mb:
                return True
            if not warned and time.monotonic() - start >= 2.0:
                self._emit(
                    f"  waiting for VRAM release: used={s.used_mb} MiB, "
                    f"baseline={self.baseline_vram_mb} MiB"
                )
                warned = True
            time.sleep(0.5)
        return self._adopt_idle_baseline_drift(samples, tolerance_mb=tolerance_mb)

    @staticmethod
    def _is_good(r: CandidateResult) -> bool:
        return r.status in {RunStatus.PASS, RunStatus.PASS_DEGRADED}

    @staticmethod
    def _robust_tg(r: CandidateResult) -> float:
        return robust_tg(r)

    def _perf_score(self, r: CandidateResult) -> float:
        """Search utility based on representative elapsed time, not an arbitrary PP/TG sum."""
        latency = workload_latency_seconds(r, self.workload_profile)
        if latency == float("inf") or latency <= 0:
            return -1.0
        return 1_000_000.0 / latency

    def _prefer(self, a: CandidateResult, b: CandidateResult) -> CandidateResult:
        return choose_preferred([a, b], self.workload_profile, self.noise_policy) or a

    def _performance_equivalent(self, a: CandidateResult, b: CandidateResult) -> bool:
        return performance_equivalent(a, b, self.workload_profile, self.noise_policy)

    def _meets_minimum_performance(self, result: CandidateResult | None) -> bool:
        if result is None or not self._is_good(result):
            return False
        tg = profile_decode_tps(result, self.workload_profile) or result.metrics.tg_tps or 0.0
        pp = profile_prefill_tps(result, self.workload_profile) or result.metrics.pp_tps or 0.0
        if self.min_tg_tps is not None and tg < self.min_tg_tps:
            return False
        if self.min_pp_tps is not None and pp < self.min_pp_tps:
            return False
        return True

    def _minimum_performance_text(self, result: CandidateResult | None) -> str:
        if result is None:
            return "no full benchmark result"
        tg = profile_decode_tps(result, self.workload_profile) or result.metrics.tg_tps or 0.0
        pp = profile_prefill_tps(result, self.workload_profile) or result.metrics.pp_tps or 0.0
        req = []
        if self.min_tg_tps is not None:
            req.append(f"TG≥{self.min_tg_tps:g}")
        if self.min_pp_tps is not None:
            req.append(f"PP≥{self.min_pp_tps:g}")
        return f"measured effective TG={tg:.1f}, PP={pp:.0f}; required " + (", ".join(req) if req else "none")

    @classmethod
    def _tg_retention(cls, result: CandidateResult, reference: CandidateResult | None) -> float | None:
        if reference is None:
            return None
        ref = cls._robust_tg(reference)
        cur = cls._robust_tg(result)
        if ref <= 0 or cur <= 0:
            return None
        return cur / ref

    def _best_exact_result(self, candidate: Candidate, full_only: bool = False) -> CandidateResult | None:
        matches = [
            r for r in self.results
            if self._is_good(r) and r.candidate.key() == candidate.key()
            and (not full_only or self._is_recommendable_full(r))
        ]
        return max(matches, key=self._perf_score) if matches else None

    def _guarded_full(self, candidate: Candidate, phase: str, reference: CandidateResult | None) -> CandidateResult:
        """Probe and, when safe, continue to FULL in the same llama-server process."""
        return self._run(candidate, quick=False, phase=phase, reference=reference, guard_probe=True)

    def _cached_for_run(self, candidate: Candidate, *, quick: bool, long_validate: bool,
                        guard_probe: bool, recon: bool, recon_context: bool = False,
                        recon_context_target: int | None = None) -> CandidateResult | None:
        """Reuse an equal-or-stronger measurement for the exact same runtime candidate.

        Recon→placement duplicated the exact same 512/256 server launch in v0.5.5. A stronger
        FULL/validation measurement can satisfy a weaker probe as well. Validation itself is never
        replaced by a weaker measurement.
        """
        if long_validate:
            allowed = {"validation"}
        elif recon_context:
            allowed = {"recon-context", "full", "validation"}
        elif guard_probe or not quick:
            allowed = {"full", "validation"}
        elif recon:
            allowed = {"recon", "recon-context", "quick", "full", "validation"}
        else:
            allowed = {"recon", "recon-context", "quick", "full", "validation"}
        rank = {"recon": 0, "recon-context": 1, "quick": 2, "full": 3, "validation": 4}
        minimum_context_evidence = (
            max(2_048, int(recon_context_target * 0.80))
            if recon_context and recon_context_target is not None else 0
        )
        matches = [
            r for r in self.results
            if self._is_good(r) and r.candidate.key() == candidate.key()
            and (r.metrics.benchmark_kind or "") in allowed
            and (
                not minimum_context_evidence
                or int(r.metrics.long_context_tokens or r.metrics.prompt_total_tokens
                       or r.metrics.prompt_tokens or 0) >= minimum_context_evidence
            )
        ]
        if not matches:
            return None
        return max(matches, key=lambda r: rank.get(r.metrics.benchmark_kind or "", -1))

    def _summary(self, r: CandidateResult) -> str:
        m = r.metrics
        parts = [r.status.value, r.reason]
        if m.pp_tps is not None:
            parts.append(f"PP={m.pp_tps:.1f} t/s")
        if m.tg_tps is not None:
            parts.append(f"TG={m.tg_tps:.1f} t/s")
        if m.acceptance is not None:
            parts.append(f"MTP={m.acceptance*100:.1f}%")
        if m.stability_tg_median is not None:
            parts.append(f"robustTG med/p10={m.stability_tg_median:.1f}/{m.stability_tg_p10:.1f}")
        if m.stability_mean_draft_len_median is not None:
            parts.append(f"mean-len-med={m.stability_mean_draft_len_median:.2f}")
        if m.context_tg_ratio is not None:
            parts.append(f"ctxTG-retained={m.context_tg_ratio:.0%}")
        if m.vision_test_passed is not None:
            state = "PASS" if m.vision_test_passed else "FAIL"
            latency = f"/{m.vision_latency_seconds:.2f}s" if m.vision_latency_seconds is not None else ""
            parts.append(f"Vision={state}{latency}")
        if m.vram_free_min_mb is not None:
            vram_class = m.vram_operating_class or self._vram_class(r).value
            parts.append(f"VRAM-free-min={m.vram_free_min_mb} MiB/{vram_class}")
        if m.ram_peak_mb is not None:
            parts.append(f"RAM-peak={m.ram_peak_mb} MiB")
        if m.long_context_tokens:
            # A recon-context run deliberately does not set the FINAL-only
            # ``long_context_passed`` bit.  Calling a successful occupied-cache
            # qualification "FAIL" in the live summary is misleading.
            if m.benchmark_kind == "recon-context":
                state = "MEASURED"
            else:
                state = "PASS" if m.long_context_passed else "FAIL"
            parts.append(f"long={m.long_context_tokens} tok/{state}")
        return " | ".join(parts)

    def _slow_dense_scout_tokens(self) -> int:
        """Decode sample used by numeric-ngl Dense SCOUT/PROBE measurements.

        QUICK remains cheap, NORMAL is long enough to cross the 10% promotion band
        reliably at single-digit tok/s, and DEEP spends extra tokens to reduce variance.
        Every promoted candidate still requires a recommendation-eligible FULL result.
        """
        return {"quick": 16, "normal": 32, "deep": 48}.get(self.search_mode, 32)

    def _kv_long_scout_target(self, candidate: Candidate) -> int | None:
        return kv_context_probe_tokens(
            candidate.kv_k, candidate.kv_v, candidate.ctx,
            workload_profile=self.workload_profile, search_mode=self.search_mode,
        )

    def _has_required_kv_runtime_evidence(self, result: CandidateResult) -> bool:
        """Whether a context-sensitive KV point has enough occupied-cache evidence.

        Q8 does not pay this extra gate.  Q4/mixed-Q4 at a long configured context may
        still be reported as MAX_CONTEXT/diagnostic evidence, but BALANCED cannot promote
        it from a 1.2K/10K-only sample.
        """
        c = result.candidate
        if (self.workload_profile != "long-context" or c.ctx < 32_768
                or not kv_requires_long_context_probe(c.kv_k, c.kv_v)):
            return True
        if result.metrics.benchmark_kind == "validation" and result.metrics.long_context_passed:
            return True
        target = kv_context_probe_tokens(
            c.kv_k, c.kv_v, c.ctx,
            workload_profile=self.workload_profile,
            # QUICK intentionally cannot manufacture high confidence from a short scout.
            search_mode="normal" if self.search_mode == "quick" else self.search_mode,
        )
        measured = int(
            result.metrics.long_context_tokens or result.metrics.prompt_total_tokens
            or result.metrics.prompt_tokens or 0
        )
        return bool(target and measured >= max(2_048, int(target * 0.80)))

    def _balanced_lower_kv_allowed(
        self,
        higher_opt: SolutionOption,
        higher_res: CandidateResult,
        lower_opt: SolutionOption,
        lower_res: CandidateResult,
    ) -> bool:
        """Apply the semantic guard before measured speed can promote lower KV.

        Q8 is allowed at equal context in BALANCED because it is the low-risk runtime
        sweet spot and often buys residency/headroom.  Q4/mixed-Q4 must both pass the
        occupied-context runtime probe and buy the configured context multiplier.  The
        speed/context priorities remain explicit escape hatches; quality remains with the
        higher tier.
        """
        required = balanced_context_gain_required(
            higher_opt.kv_k, higher_opt.kv_v, lower_opt.kv_k, lower_opt.kv_v,
        )
        if kv_requires_long_context_probe(lower_opt.kv_k, lower_opt.kv_v) \
                and not self._has_required_kv_runtime_evidence(lower_res):
            return False
        lower_floor = self._operational_vram_floor_mb(vision=lower_res.candidate.vision)
        return (
            lower_opt.context >= ceil(higher_opt.context * required)
            and (lower_res.metrics.vram_free_min_mb or 0) >= lower_floor
        )

    def _run_vision_final_diagnostic(self, base_url: str, metrics, progress: ProgressFn) -> None:
        """Record one non-gating deterministic Vision recognition check on a FINAL run."""
        try:
            vision = benchmark_vision_recognition(
                base_url,
                bundled_vision_asset(),
                use_media_path=False,
                progress=progress,
            )
            if not vision.passed and not vision.answer:
                self._emit(
                    "  Vision diagnostic returned an empty visible answer; retry once with a "
                    "larger output budget and an explicit no-thinking prompt."
                )
                retry = benchmark_vision_recognition(
                    base_url,
                    bundled_vision_asset(),
                    use_media_path=False,
                    progress=progress,
                    max_tokens=128,
                    force_no_think_prompt=True,
                )
                retry.latency_seconds += vision.latency_seconds
                vision = retry
            metrics.vision_test_passed = vision.passed
            metrics.vision_latency_seconds = vision.latency_seconds
            metrics.vision_answer = vision.answer
            metrics.vision_prompt_tokens = vision.prompt_tokens
            metrics.vision_generated_tokens = vision.generated_tokens
            metrics.vision_transport = vision.transport
            metrics.vision_error = vision.error
        except Exception as exc:
            # The diagnostic is deliberately not a search gate.  Preserve the runtime/context
            # measurement and make the unconfirmed capability explicit in report confidence.
            metrics.vision_test_passed = False
            metrics.vision_error = f"diagnostic exception: {exc}"

        if metrics.vision_test_passed:
            self._emit(
                f"  Vision recognition diagnostic: PASS in {metrics.vision_latency_seconds:.2f}s "
                f"(answer={metrics.vision_answer!r})."
            )
        else:
            self._emit(
                "  WARNING: Vision recognition diagnostic did not confirm the bundled code "
                f"({metrics.vision_error or 'wrong answer'}; answer={metrics.vision_answer!r}). "
                "The runtime candidate remains measured, but its report confidence is LOW."
            )

    def _run(self, c: Candidate, quick: bool = False, phase: str = "BENCHMARK",
             long_validate: bool = False, reference: CandidateResult | None = None,
             guard_probe: bool = False, recon: bool = False, recon_context: bool = False,
             recon_context_target: int | None = None,
             _startup_retry_attempt: int = 0) -> CandidateResult:
        if not self.budget_ok():
            return CandidateResult(c, RunStatus.FAILED, self.stop_reason, phase=phase)

        if _startup_retry_attempt == 0:
            cached = self._cached_for_run(c, quick=quick, long_validate=long_validate,
                                          guard_probe=guard_probe, recon=recon,
                                          recon_context=recon_context,
                                          recon_context_target=recon_context_target)
            if cached is not None:
                self._emit(
                    f"  reuse: {cached.metrics.benchmark_kind} measurement already exists for {c.short()} "
                    f"→ {self._summary(cached)}"
                )
                return cached

        run_no = len(self.results) + 1
        kind = "CONTEXT_SCOUT" if recon_context else ("SCOUT" if recon else ("PROBE" if quick else ("VALIDATE" if long_validate else ("GUARDED_FULL" if guard_probe else "FULL"))))
        self.phase = phase
        self._emit("")
        budget_note = " (run budget extended, time remaining)" if run_no > self.max_runs else ""
        self._emit(f"[Run {run_no}/{self.max_runs}] {phase} / {kind}{budget_note}")
        self._emit(f"  candidate: {c.short()}")

        if not self._wait_clean():
            r = CandidateResult(c, RunStatus.INVALID_ENVIRONMENT, "GPU_DID_NOT_RETURN_TO_BASELINE", phase=phase)
            self._emit(f"  result: {self._summary(r)}")
            return self._record(r)

        pre = self.gpu.snapshot()
        if pre.used_mb > self.baseline_vram_mb + 500:
            r = CandidateResult(c, RunStatus.INVALID_ENVIRONMENT, "EXTERNAL_GPU_LOAD_CHANGED", phase=phase)
            self._emit(
                f"  GPU environment changed: used={pre.used_mb} MiB vs baseline={self.baseline_vram_mb} MiB"
            )
            self._emit(f"  result: {self._summary(r)}")
            return self._record(r)

        log_path = self.results_dir / f"run_{run_no:03d}.log"
        self._emit(f"  server log: {log_path}")
        media_path = None
        try:
            cmd = build_server_command(
                self.server_exe, self.model.path, c, self.port, self.caps, media_path=media_path
            )
        except ValueError as exc:
            r = CandidateResult(c, RunStatus.FAILED, "FAIL_INVALID_ARGUMENT", logs_tail=str(exc), phase=phase)
            self._emit(f"  result: {self._summary(r)}")
            return self._record(r)

        runner = (
            ServerRunner(cmd, log_path, lease_dir=self.server_lease_dir)
            if self.server_lease_dir is not None else ServerRunner(cmd, log_path)
        )
        samples = []
        ram_samples: list[int] = []
        stop_monitor = threading.Event()
        active_step = {"text": "starting llama-server"}
        run_start = time.monotonic()

        def set_step(text: str) -> None:
            active_step["text"] = text
            self._emit(f"  -> {text}")

        def monitor() -> None:
            next_heartbeat = time.monotonic() + self.heartbeat_seconds
            while not stop_monitor.wait(0.5):
                snap = None
                try:
                    snap = self.gpu.snapshot()
                    samples.append(snap)
                except Exception:
                    pass
                try:
                    if runner.process and runner.process.poll() is None:
                        ram_samples.append(int(psutil.Process(runner.process.pid).memory_info().rss / 1024 / 1024))
                except Exception:
                    pass
                now = time.monotonic()
                if now >= next_heartbeat:
                    suffix = ""
                    if snap:
                        suffix = f" | GPU {snap.used_mb}/{self.hardware.vram_total_mb} MiB, util={snap.util_percent:.0f}%"
                    if ram_samples:
                        suffix += f" | RAM {ram_samples[-1]}/{self.hardware.ram_total_mb} MiB"
                    self._emit(
                        f"  ... {active_step['text']} | elapsed={now-run_start:.0f}s{suffix}"
                    )
                    next_heartbeat = now + self.heartbeat_seconds

        mon_thread: threading.Thread | None = None
        try:
            self._emit("  starting llama-server...")
            runner.start()
            mon_thread = threading.Thread(target=monitor, daemon=True)
            mon_thread.start()
            startup_timeout, startup_stall_timeout = self._startup_limits(_startup_retry_attempt)
            if self.model.split_count > 1 and not self._split_startup_notice_emitted:
                self._emit(
                    f"  split-GGUF startup budget: {self.model.split_parts_found}/{self.model.split_count} shards, "
                    f"total={self.model.size_bytes / (1024**3):.1f} GiB, hard={startup_timeout:.0f}s, "
                    f"idle={startup_stall_timeout:.0f}s. Process CPU/RAM/I/O activity counts as load progress."
                )
                self._split_startup_notice_emitted = True
            active_step["text"] = "loading model / waiting for /health"
            startup = runner.wait_ready(self.port, startup_timeout, stall_timeout=startup_stall_timeout)
            s0 = self.gpu.snapshot()
            samples.append(s0)
            ram_suffix = f" | RAM {ram_samples[-1]}/{self.hardware.ram_total_mb} MiB" if ram_samples else ""
            self._emit(
                f"  server ready in {startup:.1f}s | VRAM used/free={s0.used_mb}/{s0.free_mb} MiB{ram_suffix}"
            )
            # The absolute floor is a hard usability constraint, distinct from the preferred headroom reserve.
            # If loading alone consumes the floor, there is no value in executing a benchmark.
            skip_benchmark_for_vram = s0.free_mb < self.absolute_vram_floor_mb
            severe_probe = False
            safe_perf_reject = False
            cliff = False
            base = f"http://127.0.0.1:{self.port}"

            # Core-tuning policy (v0.5.4): Vision affects capability and memory through the real
            # mmproj load, but image-recognition correctness is deliberately NOT a hard candidate
            # gate. v0.5.3 made a transport/prompt-level smoke test authoritative and therefore
            # rejected every memory configuration when that one synthetic request failed. The
            # optimizer's job is to find a stable runtime first; multimodal quality validation can
            # be layered on later without corrupting context/KV/placement search.
            if c.vision and not self._vision_core_notice_emitted:
                self._emit(
                    "  Vision core mode: compatible mmproj must load successfully and its VRAM cost is measured; "
                    "image-recognition quality is not used as a tuning gate."
                )
                self._vision_core_notice_emitted = True

            # Relative cliff detection is only valid within the same execution family. Enabling MTP
            # deliberately adds a draft context and can reduce PP while increasing real decode throughput.
            # Comparing an MTP probe's PP directly with a non-MTP reference caused valid MTP candidates
            # to be rejected before decode was even measured.
            comparable_reference = reference if (reference and reference.candidate.mtp == c.mtp) else None
            reference_pp = comparable_reference.metrics.pp_tps if comparable_reference else None
            reference_tg = comparable_reference.metrics.tg_tps if comparable_reference else None

            if skip_benchmark_for_vram:
                self._emit(
                    f"  guard: only {s0.free_mb} MiB VRAM remains after load; absolute floor is "
                    f"{self.absolute_vram_floor_mb} MiB. Skipping text benchmark."
                )
                from llama_autotuner.models import BenchmarkMetrics
                metrics = BenchmarkMetrics(benchmark_kind="recon-context" if recon_context else ("recon" if recon else ("quick" if quick else "full")))
            elif recon_context:
                qb = benchmark_recon_context(
                    base, context_size=c.ctx, progress=set_step,
                    target_tokens=recon_context_target,
                    reference_tg_tps=reference_tg,
                    severe_ratio=max(0.60, self.severe_regression_ratio),
                )
                metrics = qb.metrics
                severe_probe = qb.severe_regression
            elif recon:
                # Full-GPU scouts use 64 decode tokens. Slow numeric-ngl Dense scouts use a
                # mode-scaled sample: 8 tokens was too short to distinguish noise from the >10%
                # promotion band on 5-10 t/s candidates.
                slow_cpu = self.model.kind == ModelKind.DENSE and c.ngl != "all"
                recon_tg = self._slow_dense_scout_tokens() if slow_cpu else None
                qb = benchmark_recon(
                    base, context_size=c.ctx, progress=set_step, mtp=c.mtp,
                    tg_tokens=recon_tg, slow_cpu=slow_cpu,
                )
                metrics = qb.metrics
                severe_probe = qb.severe_regression
            elif quick:
                # Numeric-ngl Dense candidates can decode at single-digit tok/s. Using the generic
                # 64-token warmup + 128-token quick probe on every ubatch wastes tens of seconds.
                # The dedicated screen keeps a meaningful 2.5K prefill sample and a mode-scaled
                # 16/32/48-token decode; any winner still earns a real FULL confirmation.
                if (self.model.kind == ModelKind.DENSE and c.ngl != "all"
                        and phase.startswith("DENSE_PARTIAL_UBATCH")):
                    qb = benchmark_dense_partial_screen(
                        base, context_size=c.ctx, progress=set_step,
                        reference_pp_tps=reference_pp, reference_tg_tps=reference_tg,
                        severe_ratio=self.severe_regression_ratio,
                        decode_tokens=self._slow_dense_scout_tokens(),
                    )
                else:
                    mtp_quick_tokens = 128 if (c.mtp and self.model.kind == ModelKind.MOE and self.search_mode != "deep") else (256 if c.mtp else 128)
                    qb = benchmark_quick(
                        base, context_size=c.ctx, progress=set_step, tg_tokens=mtp_quick_tokens,
                        reference_pp_tps=reference_pp,
                        reference_tg_tps=reference_tg,
                        severe_ratio=self.severe_regression_ratio,
                    )
                metrics = qb.metrics
                severe_probe = qb.severe_regression
            elif guard_probe:
                mtp_quick_tokens = 128 if (c.mtp and self.model.kind == ModelKind.MOE and self.search_mode != "deep") else (256 if c.mtp else 128)
                qb = benchmark_quick(
                    base, context_size=c.ctx, progress=set_step, tg_tokens=mtp_quick_tokens,
                    reference_pp_tps=reference_pp,
                    reference_tg_tps=reference_tg,
                    severe_ratio=self.severe_regression_ratio,
                )
                metrics = qb.metrics
                severe_probe = qb.severe_regression
                sg = self.gpu.snapshot()
                samples.append(sg)
                if sg.free_mb < self.absolute_vram_floor_mb:
                    skip_benchmark_for_vram = True
                    self._emit(
                        f"  guard: probe left only {sg.free_mb} MiB VRAM; FULL skipped (floor={self.absolute_vram_floor_mb} MiB)."
                    )
                elif severe_probe:
                    self._emit(f"  guard: severe relative regression detected; {phase} FULL skipped.")
                elif phase == "VRAM_HEADROOM_SEARCH" and reference is not None and metrics.tg_tps is not None:
                    ref_tg = self._robust_tg(reference)
                    retention = float(metrics.tg_tps) / max(1e-9, ref_tg) if ref_tg > 0 else 1.0
                    if retention < self.safe_perf_floor_ratio:
                        safe_perf_reject = True
                        self._emit(
                            f"  headroom early-stop: quick decode retains only {retention:.0%} of the reference "
                            f"(< {self.safe_perf_floor_ratio:.0%}); skip staged FULL validation."
                        )
                    else:
                        self._emit("  guard passed; continuing staged FULL workload without restarting llama-server.")
                        staged = benchmark_staged(
                            base, context_size=c.ctx, progress=set_step, repeats=1, warmup=False,
                            tg_tokens=(512 if self.search_mode == "deep" else 256),
                        )
                        metrics = staged.metrics
                        cliff = staged.memory_cliff
                else:
                    self._emit("  guard passed; continuing staged FULL workload without restarting llama-server.")
                    staged = benchmark_staged(
                        base, context_size=c.ctx, progress=set_step, repeats=1, warmup=False,
                        tg_tokens=(512 if self.search_mode == "deep" else 256),
                    )
                    metrics = staged.metrics
                    cliff = staged.memory_cliff
            else:
                slow_dense_validation = bool(
                    long_validate and self.model.kind == ModelKind.DENSE and c.ngl != "all"
                    and reference_tg is not None and reference_tg < 20.0
                )
                staged_tg_tokens = (
                    128 if slow_dense_validation else
                    (512 if self.search_mode == "deep" else (384 if long_validate else 256))
                )
                if slow_dense_validation:
                    self._emit(
                        f"  slow-Dense validation budget: reference TG={reference_tg:.1f} t/s; "
                        "use shorter decode samples while preserving the same context/stability checks."
                    )
                staged = benchmark_staged(
                    base, context_size=c.ctx, progress=set_step,
                    repeats=(2 if long_validate and self.search_mode == "deep" else 1),
                    tg_tokens=staged_tg_tokens,
                )
                metrics = staged.metrics
                cliff = staged.memory_cliff

            if long_validate and not cliff and not skip_benchmark_for_vram:
                metrics.benchmark_kind = "validation"
                self._emit(
                    "  -> robustness validation: heterogeneous decode workloads + cached context staircase"
                )
                slow_dense_stability_tokens = None
                if (self.model.kind == ModelKind.DENSE and c.ngl != "all"
                        and reference_tg is not None and reference_tg < 20.0):
                    slow_dense_stability_tokens = 96 if self.search_mode != "deep" else 160
                stab = benchmark_stability(
                    base, c.ctx, mode=self.search_mode, progress=set_step,
                    log_mark=runner.log_mark, draft_stats_since=runner.draft_stats_since,
                    tg_tokens=slow_dense_stability_tokens,
                )
                metrics.stability_samples = len(stab.samples)
                metrics.stability_tg_median = stab.tg_median
                metrics.stability_tg_p10 = stab.tg_p10
                metrics.stability_tg_p90 = stab.tg_p90
                metrics.stability_tg_min = stab.tg_min
                metrics.stability_tg_max = stab.tg_max
                metrics.stability_acceptance_median = stab.acceptance_median
                metrics.stability_mean_draft_len_median = stab.mean_draft_len_median
                metrics.stability_mean_draft_len_p10 = stab.mean_draft_len_p10
                metrics.stability_tg_mean_len_corr = stab.tg_mean_len_corr
                metrics.stability_tg_acceptance_corr = stab.tg_acceptance_corr
                metrics.stability_variation_pct = stab.variation_pct
                metrics.stability_passed = stab.passed
                metrics.stability_workloads = stab.samples
                metrics.context_staircase = stab.context_staircase
                metrics.context_tg_ratio = stab.context_tg_ratio
                if stab.context_staircase:
                    metrics.context_tg_first = stab.context_staircase[0].get("tg_tps")
                    metrics.context_tg_last = stab.context_staircase[-1].get("tg_tps")
                    # The final shared-prefix staircase stage replaces the old duplicate standalone 32K prefill.
                    # At ctx=65536 NORMAL/DEEP this is ~48K and is therefore a stronger long-context check.
                    last_ctx = stab.context_staircase[-1]
                    metrics.long_context_tokens = int(last_ctx.get("prompt_tokens") or 0)
                    metrics.long_context_pp_tps = last_ctx.get("pp_tps")
                    if metrics.long_context_pp_tps and metrics.pp_tps:
                        long_ratio = metrics.long_context_pp_tps / max(1e-9, metrics.pp_tps)
                        metrics.long_context_passed = long_ratio >= 0.50
                    else:
                        metrics.long_context_passed = bool(metrics.long_context_pp_tps)
                    if not metrics.long_context_passed:
                        cliff = True
                else:
                    # Defensive fallback for future custom stability modes with no staircase.
                    lc = validate_long_context(base, c.ctx, metrics.pp_tps, progress=set_step)
                    metrics.long_context_tokens = lc.prompt_tokens
                    metrics.long_context_pp_tps = lc.pp_tps
                    metrics.long_context_passed = lc.passed
                    if not lc.passed:
                        cliff = True
                if stab.tg_median is not None:
                    p10 = stab.tg_p10 if stab.tg_p10 is not None else stab.tg_median
                    self._emit(
                        f"  robustness: TG median={stab.tg_median:.1f}, p10={p10:.1f}, "
                        f"range={stab.tg_min:.1f}-{stab.tg_max:.1f} t/s"
                    )
                if stab.mean_draft_len_median is not None:
                    self._emit(
                        f"  MTP stability: mean draft len median={stab.mean_draft_len_median:.2f}, "
                        f"p10={stab.mean_draft_len_p10:.2f}"
                    )
                if stab.tg_mean_len_corr is not None:
                    self._emit(f"  diagnostic correlation: TG↔mean-draft-len r={stab.tg_mean_len_corr:+.2f}")
                if stab.tg_acceptance_corr is not None:
                    self._emit(f"  diagnostic correlation: TG↔acceptance r={stab.tg_acceptance_corr:+.2f}")
                if stab.context_tg_ratio is not None:
                    self._emit(f"  context staircase: final/first TG={stab.context_tg_ratio:.0%}")
                if not stab.passed:
                    self._emit(
                        "  WARNING: robustness benchmark detected severe decode variance/context degradation; "
                        "candidate remains runnable but will be penalized in profile ranking."
                    )

                # Vision is measured once, on FINAL candidates only.  Projector startup and VRAM
                # pressure remain the hard capability gate; this deterministic image request is a
                # non-gating diagnostic so an API/prompt transport issue cannot corrupt the whole
                # context/KV/placement search.  Reporting lowers confidence when recognition is
                # not confirmed.
                if c.vision:
                    self._run_vision_final_diagnostic(base, metrics, set_step)

            metrics.startup_seconds = startup
            s1 = self.gpu.snapshot()
            samples.append(s1)
            if samples:
                metrics.vram_peak_mb = max(s.used_mb for s in samples)
                metrics.vram_free_min_mb = min(s.free_mb for s in samples)
            if ram_samples:
                metrics.ram_peak_mb = max(ram_samples)
            else:
                try:
                    p = psutil.Process(runner.process.pid) if runner.process else None
                    metrics.ram_peak_mb = int(p.memory_info().rss / 1024 / 1024) if p else None
                except Exception:
                    pass

            below_floor = (metrics.vram_free_min_mb is not None
                           and metrics.vram_free_min_mb < self.absolute_vram_floor_mb)
            if below_floor:
                r = CandidateResult(c, RunStatus.EARLY_REJECT, "EARLY_REJECT_ABSOLUTE_VRAM_FLOOR", metrics,
                                    logs_tail=runner.tail(30), phase=phase)
            elif safe_perf_reject:
                r = CandidateResult(c, RunStatus.EARLY_REJECT, "EARLY_REJECT_SAFE_PERFORMANCE_FLOOR", metrics,
                                    logs_tail=runner.tail(30), phase=phase)
            elif severe_probe:
                r = CandidateResult(c, RunStatus.EARLY_REJECT, "EARLY_REJECT_SEVERE_PERFORMANCE_CLIFF", metrics,
                                    logs_tail=runner.tail(30), phase=phase)
            elif cliff:
                # A falling PP curve is a performance/context-scaling signal, not evidence that
                # VRAM is exhausted. Never use it as permission to move more Dense layers to CPU.
                reason = "EARLY_REJECT_LONG_CONTEXT_CLIFF" if long_validate else "EARLY_REJECT_PREFILL_PERFORMANCE_CLIFF"
                r = CandidateResult(c, RunStatus.EARLY_REJECT, reason, metrics,
                                    logs_tail=runner.tail(30), phase=phase)
            else:
                status = RunStatus.PASS
                reason = "SUCCESS"
                if metrics.tg_tps is not None and metrics.tg_tps < 2.0:
                    status = RunStatus.PASS_DEGRADED
                    reason = "TECHNICALLY_RUNNABLE_BUT_SLOW"
                r = CandidateResult(c, status, reason, metrics, logs_tail=runner.tail(30), phase=phase)
        except ServerStartError as exc:
            tail = runner.tail(50)
            text = (str(exc) + "\n" + tail).lower()
            if "out of memory" in text or "cuda error" in text or "allocation" in text:
                reason = "FAIL_OOM"
            elif "stall" in str(exc).lower():
                reason = "FAIL_STARTUP_STALL"
            else:
                reason = "FAIL_STARTUP_TIMEOUT"
            r = CandidateResult(c, RunStatus.FAILED, reason, logs_tail=tail, phase=phase)
        except Exception as exc:
            text = (str(exc) + "\n" + runner.tail(30)).lower()
            reason = "FAIL_OOM" if "out of memory" in text else str(exc)
            r = CandidateResult(c, RunStatus.FAILED, reason, logs_tail=runner.tail(30), phase=phase)
        finally:
            stop_monitor.set()
            if mon_thread:
                mon_thread.join(timeout=1)
            self._emit("  stopping server and waiting for VRAM release...")
            runner.stop()
            self._wait_clean(timeout=20)

        self._annotate_vram_policy(r)
        self._emit(f"  result: {self._summary(r)} | run-time={time.monotonic()-run_start:.1f}s")
        recorded = self._record(r)
        if (recorded.status == RunStatus.FAILED
                and recorded.reason in {"FAIL_STARTUP_STALL", "FAIL_STARTUP_TIMEOUT"}
                and _startup_retry_attempt == 0 and self.budget_ok()):
            self._emit(
                f"  transient startup failure ({recorded.reason}); retrying the exact same candidate once "
                "with a wider startup-stall window before changing context or placement."
            )
            return self._run(
                copy.deepcopy(c), quick=quick, phase=phase, long_validate=long_validate,
                reference=reference, guard_probe=guard_probe, recon=recon,
                recon_context=recon_context, recon_context_target=recon_context_target,
                _startup_retry_attempt=1,
            )
        if (recorded.status == RunStatus.FAILED
                and recorded.reason in {"FAIL_STARTUP_STALL", "FAIL_STARTUP_TIMEOUT"}
                and _startup_retry_attempt > 0 and self.model.split_count > 1):
            self._startup_blocker = recorded.reason
            self._emit(
                "  repeated split-GGUF startup failure: stop semantic context/KV fallbacks. "
                "The server never became ready, so this is a model/runtime startup diagnostic, "
                "not a measured feasibility result."
            )
        return recorded

    def _startup_limits(self, retry_attempt: int = 0) -> tuple[float, float]:
        """Return hard and inactivity limits using the complete logical GGUF size."""
        model_gib = max(0.0, self.model.size_bytes / (1024**3))
        split = max(1, int(self.model.split_count))
        hard = min(600.0, max(45.0, 30.0 + model_gib * 5.0))

        if split > 1:
            # Large split models often keep a tiny metadata-only first shard.  Their
            # total cold-load time tracks the sum of every shard, plus filesystem-open
            # overhead.  A retry widens the hard deadline once, never per solution option.
            hard = min(600.0, max(180.0, hard + (split - 1) * 15.0))
            idle = min(240.0, max(90.0, 40.0 + model_gib * 2.0))
            if retry_attempt:
                hard = min(900.0, max(hard * 1.35, hard + 90.0))
                idle = min(360.0, max(180.0, idle * 1.5))
            return hard, min(idle, hard)

        # Preserve the established single-file policy.  Process activity observed by
        # ServerRunner now also keeps the inactivity deadline alive.
        idle = min(105.0, max(40.0, 25.0 + model_gib * 2.0))
        if retry_attempt:
            idle = min(120.0, max(75.0, idle * 1.5))
        return hard, min(idle, hard)

    def _moe_coarse_values(self, seed: Candidate | None = None) -> list[int]:
        """Return safe→aggressive expert-offload probes around the real floor seed.

        -ncmoe N keeps experts from the first N target-model layers on CPU. Therefore
        larger N is safer for VRAM, smaller N is more GPU-resident and normally faster.
        Auxiliary MTP/NextN blocks are not counted as target MoE layers.
        """
        blocks = model_main_block_count(self.model)
        if (self._static_estimate is not None
                and self._static_estimate.predicted_moe_all_free_mb is not None
                and self._static_estimate.predicted_moe_all_free_mb >= self.absolute_vram_floor_mb):
            # All experts are predicted to fit. Try ncmoe=0 directly; if the static model was
            # optimistic, normal recovery will move experts back to CPU.
            return [0]

        p = None
        if seed is not None and seed.ncmoe is not None:
            p = int(seed.ncmoe)
        elif self._static_estimate is not None and self._static_estimate.predicted_moe_ncmoe is not None:
            p = int(self._static_estimate.predicted_moe_ncmoe)
        if p is None:
            step = max(2, int(round(blocks * 0.12)))
            vals = list(range(blocks, -1, -step))
            if vals[-1] != 0:
                vals.append(0)
            return sorted(set(vals), reverse=True)

        p = max(0, min(blocks, p))
        guard = max(3, int(round(blocks * 0.12)))
        values = [
            min(blocks, p + guard),
            min(blocks, p + max(2, guard // 2)),
            p,
            max(0, p - 2),
            max(0, p - 4),
            0,
        ]
        out: list[int] = []
        for v in values:
            if v not in out:
                out.append(v)
        return out

    def _moe_seed_for_operational_floor(self, candidate: Candidate) -> Candidate:
        """Use calibrated static memory to jump directly near a stable ncmoe seed.

        Once one MoE probe calibrates the session, retrying ncmoe=23→24→25 one server at a
        time is wasted work. Move the *prediction* toward the operational floor first; runtime
        still remains authoritative and may require one final recovery.
        """
        if self.model.kind != ModelKind.MOE or candidate.ncmoe is None:
            return copy.deepcopy(candidate)
        c = copy.deepcopy(candidate)
        blocks = model_main_block_count(self.model)
        floor = self._operational_vram_floor_mb(vision=c.vision)
        while c.ncmoe < blocks:
            free = self._predicted_free_for(c)
            if free is None or free >= floor:
                break
            c.ncmoe += 1
        return c

    @staticmethod
    def _interpolated_context_repair(
        *, context: int, measured_free_mb: int, target_free_mb: int,
        predicted_free_here_mb: int | None, predicted_free_lower_mb: int | None,
        lower_context: int, minimum_context: int = 16_384,
        guard_mb: int = 64, alignment: int = 1024,
    ) -> int:
        """Estimate the largest context likely to clear a measured VRAM floor.

        KV memory is locally close to linear in context.  Older repair code moved in coarse 4K
        steps and added a full extra step as a guard, so a 38 MiB miss could discard 8K context.
        Interpolate the local static slope, target a small explicit guard, align conservatively,
        and let one runtime scout prove the result.  Static memory remains an estimate, never proof.
        """
        span = max(1, int(context) - int(lower_context))
        predicted_gain = None
        if predicted_free_here_mb is not None and predicted_free_lower_mb is not None:
            predicted_gain = int(predicted_free_lower_mb) - int(predicted_free_here_mb)
        if predicted_gain is not None and predicted_gain > 0:
            mib_per_token = predicted_gain / span
        else:
            # Conservative fallback: request at least one 4K reduction when no useful slope exists.
            mib_per_token = max(1.0 / 4096.0, (target_free_mb - measured_free_mb) / 4096.0)
        needed = max(1, int(target_free_mb) + int(guard_mb) - int(measured_free_mb))
        raw_drop = int((needed / mib_per_token) + 0.999999)
        aligned_drop = max(alignment, ((raw_drop + alignment - 1) // alignment) * alignment)
        return max(int(minimum_context), int(context) - aligned_drop)

    @staticmethod
    def _runtime_bracket_context(
        *, high_context: int, high_free_mb: int,
        low_context: int, low_free_mb: int,
        target_free_mb: int, guard_mb: int = 16,
        alignment: int = 1024,
    ) -> int | None:
        """Refine upward inside one measured fragile↔stable context bracket.

        The first repair intentionally lands on the safe side.  Once both endpoints are real
        llama-server measurements, linear KV interpolation is substantially better than keeping
        the coarse repaired point.  The returned context is aligned downward and still needs one
        authoritative SCOUT/FULL measurement.
        """
        hi_ctx, lo_ctx = int(high_context), int(low_context)
        hi_free, lo_free = int(high_free_mb), int(low_free_mb)
        target = int(target_free_mb) + max(0, int(guard_mb))
        align = max(1, int(alignment))
        if hi_ctx <= lo_ctx or hi_free >= target or lo_free < target or lo_free <= hi_free:
            return None
        fraction = (lo_free - target) / (lo_free - hi_free)
        raw = lo_ctx + fraction * (hi_ctx - lo_ctx)
        aligned = int(raw // align) * align
        aligned = max(lo_ctx + align, min(hi_ctx - align, aligned))
        return aligned if lo_ctx < aligned < hi_ctx else None

    def _repair_fragile_dense_full_gpu_context(
        self, seed: Candidate, fragile_full: CandidateResult,
    ) -> Candidate | None:
        """Recover a near-miss full-GPU Dense branch by reducing context, never layers.

        Phase-0 scouts often leave a real lower-context point in the same KV/Vision family.
        When the selected upper point later misses the stronger FULL headroom floor by only
        tens of MiB, discarding the whole family can lose 40K+ useful tokens.  Interpolate the
        measured VRAM bracket, FULL-confirm once, and allow one bounded measured-slope retry.
        """
        if seed.ngl != "all" or seed.mtp or seed.ctx <= 16_384 or not self._is_good(fragile_full):
            return None
        measured_free = fragile_full.metrics.vram_free_min_mb
        if measured_free is None:
            return None
        thresholds = self._vram_thresholds(seed)
        target_free = thresholds.operational_floor_mb
        if measured_free >= target_free:
            return None

        lower = [
            r for r in self.results
            if self._is_good(r) and r.candidate.ctx < seed.ctx
            and r.candidate.ngl == "all" and not r.candidate.mtp
            and (r.candidate.kv_k, r.candidate.kv_v) == (seed.kv_k, seed.kv_v)
            and r.candidate.vision == seed.vision and r.candidate.mmproj == seed.mmproj
            and r.metrics.vram_free_min_mb is not None
            and int(r.metrics.vram_free_min_mb) >= target_free
        ]
        lower_result = max(lower, key=lambda r: r.candidate.ctx, default=None)

        repair_ctx: int | None = None
        if lower_result is not None:
            repair_ctx = self._runtime_bracket_context(
                high_context=seed.ctx,
                high_free_mb=int(measured_free),
                low_context=lower_result.candidate.ctx,
                low_free_mb=int(lower_result.metrics.vram_free_min_mb or 0),
                target_free_mb=target_free,
                guard_mb=32,
            )
        if repair_ctx is None:
            lower_ctx = max(16_384, seed.ctx - 4096)
            lower_candidate = copy.deepcopy(seed)
            lower_candidate.ctx = lower_ctx
            repair_ctx = self._interpolated_context_repair(
                context=seed.ctx,
                measured_free_mb=int(measured_free),
                target_free_mb=target_free,
                predicted_free_here_mb=self._predicted_free_for(seed),
                predicted_free_lower_mb=self._predicted_free_for(lower_candidate),
                lower_context=lower_ctx,
                guard_mb=32,
            )
        if not 16_384 <= int(repair_ctx) < seed.ctx:
            return None

        repaired = copy.deepcopy(seed)
        repaired.ctx = int(repair_ctx)
        self._emit(
            f"  full-GPU context repair: FULL left {measured_free} MiB versus the "
            f"{target_free} MiB operational target; try ctx={seed.ctx}→{repaired.ctx} "
            f"with identical {seed.kv_k}/{seed.kv_v}, Vision and ngl=all."
        )
        first = self._guarded_full(repaired, "DENSE_CONTEXT_REPAIR_CONFIRM", fragile_full)
        if self._is_recommendable_full(first):
            self._mark_tight(first)
            self._emit(f"  full-GPU context repair succeeded: {self._summary(first)}")
            return copy.deepcopy(first.candidate)

        # One measured retry is useful when static mmproj/workspace estimates missed again.
        if (lower_result is not None and self._is_good(first)
                and first.metrics.vram_free_min_mb is not None
                and first.candidate.ctx - lower_result.candidate.ctx >= 2048
                and int(first.metrics.vram_free_min_mb) < target_free
                and self.budget_ok()):
            second_ctx = self._runtime_bracket_context(
                high_context=first.candidate.ctx,
                high_free_mb=int(first.metrics.vram_free_min_mb),
                low_context=lower_result.candidate.ctx,
                low_free_mb=int(lower_result.metrics.vram_free_min_mb or 0),
                target_free_mb=target_free,
                guard_mb=32,
            )
            if second_ctx is not None and second_ctx < first.candidate.ctx:
                repaired2 = copy.deepcopy(seed)
                repaired2.ctx = second_ctx
                self._emit(
                    f"  measured-slope context repair retry: ctx={first.candidate.ctx}→{second_ctx}."
                )
                second = self._guarded_full(
                    repaired2, "DENSE_CONTEXT_REPAIR_CONFIRM_2", first,
                )
                if self._is_recommendable_full(second):
                    self._mark_tight(second)
                    self._emit(f"  measured-slope context repair succeeded: {self._summary(second)}")
                    return copy.deepcopy(second.candidate)

        self._emit("  full-GPU context repair did not clear the recommendation floor; try the next Pareto branch.")
        return None

    def _calibrated_exact_dense_scout(self, seed: Candidate,
                                      measured: CandidateResult) -> Candidate | None:
        """Use the first numeric-ngl runtime calibration for one better exact-context scout.

        The initial planner seed is deliberately conservative.  When it leaves several GiB unused,
        reporting that point as MAX_CONTEXT evidence materially understates the best exact-context
        placement.  This bounded helper proposes exactly one more aggressive numeric ngl; it never
        converts the result into a final recommendation without the normal FULL/FINAL gates.
        """
        if seed.ngl == "all" or not self._is_good(measured):
            return None
        free = measured.metrics.vram_free_min_mb
        if free is None:
            return None
        thresholds = self._vram_thresholds(seed)
        target = thresholds.operational_floor_mb + 64
        if free < target:
            return None
        start = int(seed.ngl)
        total = model_main_block_count(self.model)
        best = start
        for ngl in range(start + 1, total + 1):
            c = copy.deepcopy(seed)
            c.ngl = ngl
            predicted = self._predicted_free_for(c)
            if predicted is None:
                break
            if predicted >= target:
                best = ngl
            else:
                break
        if best <= start:
            return None
        out = copy.deepcopy(seed)
        out.ngl = best
        return out

    def _search_moe_placement(self, seed: Candidate) -> Candidate | None:
        """Locate the MoE placement knee with cheap probes first, then FULL-validate a finalist.

        v0.5.4 ran a long FULL workload as soon as headroom approached ~2 GiB. On a
        large MoE this could spend minutes validating ncmoe=29 and then ncmoe=27 before
        even learning where the actual memory/performance boundary was. Placement memory
        is allocated at server startup, so coarse/binary boundary discovery can be quick.
        """
        self.phase = "MOE_PLACEMENT"
        self._emit("\n[Phase 1] MoE placement: quick-map expert residency before expensive validation")
        self._emit(
            "  Strategy: -ncmoe N keeps experts from the first N target layers on CPU. "
            "Probe safe→aggressive with short runs, locate the hard/performance boundary, "
            "then FULL-validate only the best measured placement."
        )

        good_by_n: dict[int, CandidateResult] = {}
        bad_n: int | None = None
        last_good: CandidateResult | None = None

        # Reuse solution-recon evidence for this exact ctx/KV/Vision family. v0.5.7 could spend
        # 5 more launches mapping 32/29/27/25/26 even after Phase 0 had already established
        # ncmoe=27 PASS and 25/26 as memory failures.
        family_results = [
            r for r in self.results
            if r.candidate.ncmoe is not None and r.candidate.ctx == seed.ctx
            and r.candidate.kv_k == seed.kv_k and r.candidate.kv_v == seed.kv_v
            and r.candidate.vision == seed.vision and not r.candidate.mtp
            and r.metrics.benchmark_kind in {"recon", "quick", "recon-context"}
        ]
        for r in family_results:
            n = int(r.candidate.ncmoe)
            if self._is_good(r):
                prev = good_by_n.get(n)
                if prev is None or self._perf_score(r) > self._perf_score(prev):
                    good_by_n[n] = r
            elif self._recoverable_boundary_reason(r.reason):
                if bad_n is None or n > bad_n:
                    bad_n = n
        if good_by_n:
            last_good = good_by_n[min(good_by_n)]
            self._emit(
                "  reuse Phase-0 MoE placement evidence: "
                + ", ".join(f"ncmoe={n} PASS" for n in sorted(good_by_n))
                + (f", nearest rejected ncmoe={bad_n}." if bad_n is not None else ".")
            )

        # If Phase 0 already bracketed the aggressive side, do not re-run the coarse safe map.
        if good_by_n and bad_n is not None and bad_n < min(good_by_n):
            coarse_values = []
        else:
            coarse_values = [n for n in self._moe_coarse_values(seed) if n not in good_by_n]

        for n in coarse_values:
            if not self.budget_ok():
                return None
            c = copy.deepcopy(seed)
            c.ngl = "all"
            c.ncmoe = n
            probe = self._run(c, quick=True, phase="MOE_PLACEMENT_PROBE", reference=last_good)
            if not self._is_good(probe):
                bad_n = n
                self._emit(
                    f"  placement map: ncmoe={n} rejected ({probe.reason}); "
                    "stop the coarse walk and refine only the last safe/aggressive bracket."
                )
                break
            good_by_n[n] = probe
            last_good = probe
            self._emit(
                f"  placement map: ncmoe={n} -> PP={probe.metrics.pp_tps or 0:.1f} t/s | "
                f"TG={probe.metrics.tg_tps or 0:.1f} t/s | free={probe.metrics.vram_free_min_mb} MiB"
            )

        if not good_by_n:
            self._emit("  no usable MoE placement probe was found at this context.")
            return None

        # If the first bad point is adjacent to the most aggressive known-good point we
        # already have the boundary. Otherwise refine with QUICK probes only.
        aggressive_good_n = min(good_by_n)
        if bad_n is not None and bad_n < aggressive_good_n:
            safe_n = aggressive_good_n
            while safe_n - bad_n > 1 and self.budget_ok():
                mid = (safe_n + bad_n) // 2
                c = copy.deepcopy(seed); c.ngl = "all"; c.ncmoe = mid
                self._emit(f"  placement refine: quick-probing ncmoe={mid} between safe={safe_n} and bad={bad_n}.")
                probe = self._run(c, quick=True, phase="MOE_PLACEMENT_REFINE_PROBE", reference=good_by_n[safe_n])
                if self._is_good(probe):
                    good_by_n[mid] = probe
                    safe_n = mid
                    self._emit(
                        f"  placement refine: ncmoe={mid} is usable -> TG={probe.metrics.tg_tps or 0:.1f}, "
                        f"free={probe.metrics.vram_free_min_mb} MiB."
                    )
                else:
                    bad_n = mid
                    self._emit(f"  placement refine: ncmoe={mid} rejected ({probe.reason}).")

        probes = list(good_by_n.values())
        winner_probe = choose_preferred(probes, self.workload_profile, self.noise_policy) or probes[0]
        defer_full = bool(
            self.search_mode != "deep" and family_results and bad_n is not None
            and bad_n < int(winner_probe.candidate.ncmoe or 0)
            and (self.min_tg_tps is None or self._min_tg_is_default) and self.min_pp_tps is None
        )
        if defer_full:
            self._emit(
                f"  placement probe winner: ncmoe={winner_probe.candidate.ncmoe} -> "
                f"PP={winner_probe.metrics.pp_tps or 0:.1f} t/s | TG={winner_probe.metrics.tg_tps or 0:.1f} t/s | "
                f"free={winner_probe.metrics.vram_free_min_mb} MiB. Phase-0 already brackets the memory edge; "
                "defer expensive FULL to the joint ubatch/placement winner."
            )
            return copy.deepcopy(winner_probe.candidate)

        self._emit(
            f"  placement probe winner: ncmoe={winner_probe.candidate.ncmoe} -> "
            f"PP={winner_probe.metrics.pp_tps or 0:.1f} t/s | TG={winner_probe.metrics.tg_tps or 0:.1f} t/s | "
            f"free={winner_probe.metrics.vram_free_min_mb} MiB. FULL-validating this placement once."
        )

        # Try the measured winner first. If a long validation fails, fall back through
        # the remaining quick-good placements in noise-aware preference order. Cap the
        # expensive recovery to two FULL attempts; later SAFE search can establish a
        # more conservative profile without redoing the whole placement map.
        remaining = [r for r in probes if r is not winner_probe]
        ordered = [winner_probe]
        while remaining:
            nxt = choose_preferred(remaining, self.workload_profile, self.noise_policy)
            if nxt is None:
                break
            ordered.append(nxt)
            remaining.remove(nxt)

        for idx, probe in enumerate(ordered[:2], start=1):
            c = copy.deepcopy(probe.candidate)
            full = self._run(c, quick=False, phase="MOE_PLACEMENT_CONFIRM")
            if self._is_recommendable_full(full):
                self._mark_tight(full)
                self._emit(
                    f"  MoE placement selected for performance tuning: ncmoe={full.candidate.ncmoe} "
                    f"({self._summary(full)})"
                )
                return copy.deepcopy(full.candidate)
            if self._is_good(full):
                self._emit(
                    f"  FULL placement confirmation #{idx} is {self._vram_class(full).value} at "
                    f"{full.metrics.vram_free_min_mb} MiB; it is runnable but not recommendation-safe. "
                    "Trying the next measured placement."
                )
                continue
            self._emit(
                f"  FULL placement confirmation #{idx} failed at ncmoe={c.ncmoe} ({full.reason}); "
                "trying the next measured-safe placement."
            )

        self._emit("  no fully validated MoE placement was found at this context.")
        return None

    def _dense_coarse_values(self) -> list[str | int]:
        """Return Dense placement candidates, preferring a GGUF-derived local seed.

        If static tensor/KV analysis can estimate the useful boundary, start close to that
        boundary instead of testing CPU-only or obviously impossible extremes. The real
        llama-server probe remains authoritative and will refine/override the estimate.
        """
        smart = dense_seed_order(self.model, self._static_estimate)
        if smart:
            return smart
        total = model_main_block_count(self.model)
        fractions = (1.00, 0.92, 0.84, 0.75, 0.65, 0.55, 0.40, 0.25, 0.0)
        numeric: list[int] = []
        for fraction in fractions:
            n = int(round(total * fraction))
            n = min(total, max(0, n))
            if n not in numeric:
                numeric.append(n)
        if numeric[-1] != 0:
            numeric.append(0)
        return ["all", *numeric]

    def _search_dense_placement(self, seed: Candidate, *, preserve_full_gpu: bool = False) -> Candidate | None:
        self.phase = "DENSE_PLACEMENT"
        self._emit("\n[Phase 1] Dense placement: locating the VRAM boundary from the GPU-heavy side")
        self._emit(
            "  Strategy: try full GPU first; move layers to CPU only when VRAM/startup/long-prefill "
            "guards require it, then refine the highest stable numeric ngl."
        )

        total = model_main_block_count(self.model)
        safe_full: CandidateResult | None = None
        safe_ngl: int | None = None
        # Treat `all` as one logical step above the highest numeric transformer-layer count.
        # This lets us bracket `all` failure vs `ngl=total` success without conflating them.
        bad_ngl: int | None = None

        for placement in self._dense_coarse_values():
            if not self.budget_ok():
                return None

            c = copy.deepcopy(seed)
            c.ngl = placement
            probe = self._run(c, quick=True, phase="DENSE_PLACEMENT_PROBE")

            if not self._is_good(probe):
                if placement == "all":
                    bad_ngl = total + 1
                    label = "all"
                else:
                    bad_ngl = int(placement)
                    label = str(placement)

                if probe.status in {RunStatus.INVALID_ENVIRONMENT, RunStatus.FATAL}:
                    self._emit(
                        f"  dense placement probe ngl={label} failed for a non-placement reason "
                        f"({probe.reason}); aborting Dense placement search."
                    )
                    return None

                if not self._recoverable_boundary_reason(probe.reason):
                    self._emit(
                        f"  dense placement probe ngl={label} failed with {probe.reason}; "
                        "this is not a recognized memory/boundary failure, so the tuner will not "
                        "silently reinterpret it as 'use more CPU'."
                    )
                    return None

                if preserve_full_gpu and placement == "all":
                    self._emit(
                        f"  full-GPU solution invariant: ngl=all is not stable ({probe.reason}); "
                        "do not recover with numeric ngl; try another/lower-context solution family."
                    )
                    return None
                self._emit(
                    f"  decision: ngl={label} is too aggressive ({probe.reason}); "
                    "continue with a safer GPU-layer count."
                )
                continue

            # A short probe can pass while a 6K/10K staged prefill later crosses the VRAM
            # boundary.  Establish a FULL-stable anchor before doing binary refinement.
            self._emit(
                f"  ngl={placement} passed the cheap probe; full-validating this placement before refinement."
            )
            full = self._run(c, quick=False, phase="DENSE_PLACEMENT_BOUNDARY")
            if self._is_good(full) and not self._is_recommendable_full(full):
                if placement == "all":
                    bad_ngl = total + 1
                    label = "all"
                else:
                    bad_ngl = int(placement)
                    label = str(placement)
                thresholds = self._vram_thresholds(c)
                self._emit(
                    f"  FULL validation for ngl={label} is {self._vram_class(full).value}: "
                    f"{full.metrics.vram_free_min_mb} MiB is below the {thresholds.tight_floor_mb} MiB "
                    "recommendation threshold. Continue toward a safer placement."
                )
                if preserve_full_gpu and placement == "all":
                    return self._repair_fragile_dense_full_gpu_context(c, full)
                continue
            if not self._is_good(full):
                if placement == "all":
                    bad_ngl = total + 1
                    label = "all"
                else:
                    bad_ngl = int(placement)
                    label = str(placement)

                if full.status in {RunStatus.INVALID_ENVIRONMENT, RunStatus.FATAL}:
                    self._emit(
                        f"  FULL validation for ngl={label} failed for a non-placement reason "
                        f"({full.reason}); aborting Dense placement search."
                    )
                    return None
                if not self._recoverable_boundary_reason(full.reason):
                    self._emit(
                        f"  FULL validation for ngl={label} failed with {full.reason}; "
                        "not treating it as a VRAM boundary."
                    )
                    return None

                if preserve_full_gpu and placement == "all":
                    self._emit(
                        f"  full-GPU solution invariant: ngl=all failed FULL validation ({full.reason}); "
                        "do not convert this Dense solution into CPU layer offload. Re-plan a lower context instead."
                    )
                    return None
                self._emit(
                    f"  decision: ngl={label} passed the quick probe but failed FULL validation "
                    f"({full.reason}); continue safer."
                )
                continue

            safe_full = full
            self._mark_tight(full)

            # `all` is already the maximum possible placement.  There is nothing to refine.
            if placement == "all":
                self._emit(
                    f"  Dense full-GPU placement is stable: {safe_full.candidate.short()} "
                    f"({self._summary(safe_full)})"
                )
                return copy.deepcopy(safe_full.candidate)

            safe_ngl = int(placement)
            break

        if safe_full is None or safe_ngl is None:
            self._emit(
                "  no Dense placement survived both the quick probe and staged FULL validation, "
                "including the CPU-heavy fallback."
            )
            return None

        # We walked from aggressive -> safe, so `bad_ngl` is the closest known bad point above
        # the first FULL-stable anchor.  Refine *upward* to find the highest stable numeric ngl.
        if bad_ngl is not None and bad_ngl > safe_ngl:
            self._emit(
                f"  refining Dense bracket: safe={safe_ngl}, bad/aggressive="
                f"{'all' if bad_ngl == total + 1 else bad_ngl}"
            )
            while bad_ngl - safe_ngl > 1 and self.budget_ok():
                mid = (safe_ngl + bad_ngl) // 2
                c = copy.deepcopy(seed)
                c.ngl = mid
                self._emit(f"  refine: testing ngl={mid} with guarded FULL validation.")
                r = self._guarded_full(c, "DENSE_PLACEMENT_REFINE", reference=safe_full)

                if self._is_recommendable_full(r):
                    safe_ngl = mid
                    safe_full = r
                    self._mark_tight(r)
                    self._emit(f"  refine: ngl={mid} is stable -> move boundary toward more GPU layers.")
                    continue

                if self._is_good(r):
                    bad_ngl = mid
                    self._emit(
                        f"  refine: ngl={mid} is {self._vram_class(r).value} at "
                        f"{r.metrics.vram_free_min_mb} MiB -> keep safer side."
                    )
                    continue

                if self._recoverable_boundary_reason(r.reason):
                    bad_ngl = mid
                    self._emit(f"  refine: ngl={mid} is too aggressive ({r.reason}) -> keep safer side.")
                    continue

                self._emit(
                    f"  refine: ngl={mid} failed for a non-boundary reason ({r.reason}); "
                    "stop refinement and keep the last FULL-stable candidate."
                )
                break

        self._emit(
            f"  Dense placement selected for performance tuning: ngl={safe_full.candidate.ngl} "
            f"({self._summary(safe_full)})"
        )
        return copy.deepcopy(safe_full.candidate)

    def _confirm_dense_oversized_placement(self, seed: Candidate) -> Candidate | None:
        key = self._dense_oversized_key(seed)
        evidence = self._dense_oversized_evidence.get(key) if key is not None else None
        if evidence is None:
            return None
        pair = (seed.kv_k, seed.kv_v)
        known_full_penalty = self._dense_oversized_full_penalty_mb.get(pair)
        if known_full_penalty is not None and evidence.metrics.vram_free_min_mb is not None:
            projected_full_free = int(evidence.metrics.vram_free_min_mb) - known_full_penalty
            projected_class = self._vram_thresholds(seed).classify(projected_full_free)
            if projected_class in {VramOperatingClass.REJECT, VramOperatingClass.FRAGILE}:
                self._emit(
                    f"\n[Phase 1] Skip oversized Dense confirmation: the measured {seed.kv_k}/{seed.kv_v} "
                    f"FULL-workload penalty is {known_full_penalty} MiB, so this scout projects only "
                    f"~{projected_full_free} MiB ({projected_class.value}). Try the next measured Pareto point."
                )
                return None
        self._emit("\n[Phase 1] Oversized Dense placement confirmation")
        self._emit(
            f"  reuse Phase-0 placement evidence: {seed.short()} -> PP={evidence.metrics.pp_tps or 0:.1f} | "
            f"TG={evidence.metrics.tg_tps or 0:.1f} | free={evidence.metrics.vram_free_min_mb} MiB. "
            "FULL-confirm exactly this context/KV/ngl once; do not restart a broad layer walk."
        )
        full = self._run(copy.deepcopy(seed), quick=False, phase="DENSE_OVERSIZED_CONFIRM", reference=evidence)
        if evidence.metrics.vram_free_min_mb is not None and full.metrics.vram_free_min_mb is not None:
            penalty = max(0, int(evidence.metrics.vram_free_min_mb) - int(full.metrics.vram_free_min_mb))
            self._dense_oversized_full_penalty_mb[pair] = max(
                penalty, self._dense_oversized_full_penalty_mb.get(pair, 0)
            )
        if self._is_good(full):
            free = int(full.metrics.vram_free_min_mb or 0)
            thresholds = self._vram_thresholds(seed)
            operating_class = thresholds.classify(full.metrics.vram_free_min_mb)
            if operating_class in {VramOperatingClass.SAFE, VramOperatingClass.OPERATIONAL}:
                return copy.deepcopy(full.candidate)
            if operating_class == VramOperatingClass.TIGHT:
                self._mark_tight(full)
                self._emit(
                    f"  oversized FULL is TIGHT but validated: {free} MiB free is within the "
                    f"supported {thresholds.tight_floor_mb}–{thresholds.operational_floor_mb - 1} MiB "
                    "hysteresis band. Keep this faster placement, lock b/ub at the confirmed values, "
                    "and proceed directly toward FINAL validation."
                )
                return copy.deepcopy(full.candidate)
            self._emit(
                f"  oversized FULL is {operating_class.value}: {free} MiB free is below the "
                f"{thresholds.tight_floor_mb} MiB minimum recommendation threshold (hard floor "
                f"{thresholds.hard_floor_mb} MiB). Keep it as diagnostic evidence, but try the next "
                "measured context/KV knee instead of recommending a fragile placement."
            )
            return None
        if full.reason in {"EARLY_REJECT_ABSOLUTE_VRAM_FLOOR", "FAIL_OOM", "FAIL_CUDA_OOM"} and int(seed.ngl) > 0:
            safer = copy.deepcopy(seed); safer.ngl = max(0, int(seed.ngl) - 1)
            self._emit(
                f"  FULL crossed the memory floor; one safety retry at ngl={safer.ngl}. "
                "This is stability repair, not a new placement sweep."
            )
            repaired = self._guarded_full(safer, "DENSE_OVERSIZED_CONFIRM_REPAIR", evidence)
            if self._is_recommendable_full(repaired):
                self._mark_tight(repaired)
                return copy.deepcopy(repaired.candidate)
        else:
            self._emit(
                f"  oversized branch rejected after FULL ({full.reason}); do not move more layers to CPU to "
                "hide a prefill/decode performance cliff. Try the next measured context/KV branch."
            )
        return None

    def _search_placement(self, seed: Candidate, *, preserve_full_gpu: bool = False) -> Candidate | None:
        self._static_estimate = estimate_static_memory(
            self.model, self.hardware, self.baseline_vram_mb, self.vram_margin_mb, seed
        )
        self._emit("")
        for line in format_static_estimate(self.model, self._static_estimate):
            self._emit(line)

        if self.model.kind == ModelKind.MOE:
            values = self._moe_coarse_values(seed)
            if values:
                self._emit(f"  smart MoE probe order: {', '.join('ncmoe='+str(v) for v in values[:8])}")
            return self._search_moe_placement(seed)
        if self.model.kind == ModelKind.DENSE:
            if self._dense_oversized_active and self._dense_oversized_key(seed) in self._dense_oversized_evidence:
                return self._confirm_dense_oversized_placement(seed)
            values = self._dense_coarse_values()
            if values:
                self._emit(f"  smart Dense probe order: {', '.join('ngl='+str(v) for v in values[:8])}")
            return self._search_dense_placement(seed, preserve_full_gpu=preserve_full_gpu)
        self._emit("  model architecture is UNKNOWN; refusing to guess Dense/MoE placement strategy.")
        self.stop_reason = "UNKNOWN_MODEL_ARCHITECTURE"
        return None

    def _select_validation_candidates(self, ctx: int) -> list[Candidate]:
        """Select only materially distinct FULL candidates for expensive long validation.

        A candidate is not distinct merely because it measured 44.9 instead of 44.1 t/s. Search
        results are first reduced to a noise-aware Pareto frontier over decode, context-fill/prefill
        and VRAM headroom. Performance-equivalent candidates collapse to the safer/simpler one.
        """
        full = [
            r for r in self.results
            if self._is_recommendable_full(r) and r.candidate.ctx == ctx
            and r.metrics.pp_tps and r.metrics.tg_tps
        ]
        if not full:
            return []

        frontier = pareto_frontier(full, self.workload_profile, self.noise_policy) or full
        # Keep plain and speculative families separate until robustness validation. A materially fast
        # MTP FULL sample can noise-dominate plain on the short workload, but plain is still the
        # control needed to verify that the speculative win survives heterogeneous text.
        plain_all = [r for r in full if not r.candidate.mtp]
        mtp_all = [r for r in full if r.candidate.mtp]
        plain = pareto_frontier(plain_all, self.workload_profile, self.noise_policy) or plain_all
        mtp = pareto_frontier(mtp_all, self.workload_profile, self.noise_policy) or mtp_all
        best_plain = choose_preferred(plain, self.workload_profile, self.noise_policy) if plain else None
        best_mtp = None
        if mtp:
            if self._preferred_mtp_key:
                exact = [r for r in mtp if r.candidate.key() == self._preferred_mtp_key]
                if exact:
                    best_mtp = max(exact, key=self._perf_score)
            if best_mtp is None:
                best_mtp = choose_preferred(mtp, self.workload_profile, self.noise_policy)

        # A high-headroom plain candidate deserves a second slot only when it is materially distinct
        # from the performance winner. If performance is inside the noise zone, keep the safer one only.
        safe_plain = None
        if plain:
            performant = [
                r for r in plain
                if best_plain is None or self._tg_retention(r, best_plain) is None
                or self._tg_retention(r, best_plain) >= self.safe_perf_floor_ratio
            ]
            if performant:
                safe_plain = max(performant, key=lambda r: r.metrics.vram_free_min_mb or 0)

        ordered: list[CandidateResult] = []
        if best_plain is not None and safe_plain is not None and best_plain is not safe_plain:
            if self._performance_equivalent(best_plain, safe_plain):
                chosen = self._prefer(best_plain, safe_plain)
                self._emit(
                    "  final-prune: performance and SAFE plain candidates are inside the noise zone; "
                    f"validate only {chosen.candidate.short()}."
                )
                ordered.append(chosen)
            else:
                ordered.extend([best_plain, safe_plain])
        elif best_plain is not None:
            ordered.append(best_plain)
        elif safe_plain is not None:
            ordered.append(safe_plain)

        if best_mtp is not None:
            # MTP gets a final slot only if it remains materially different from the best plain FULL.
            material_decode = best_plain is None or decode_relation(best_mtp, best_plain, self.noise_policy) > 0
            material_latency = best_plain is None or latency_relation(
                best_mtp, best_plain, self.workload_profile, self.noise_policy,
            ) > 0
            if self._mtp_speed_only and self.search_mode != "deep":
                # A speed-only MTP point already earned an authoritative FULL.  Spending a 196K
                # staircase on it first can consume the whole NORMAL budget while the actual
                # OPTIMAL non-MTP command remains unvalidated.  Keep the FULL for FASTEST and
                # validate the end-to-end winner instead.
                self._emit(
                    "  final-prune: speed-only MTP already has FULL evidence; skip its expensive "
                    "long-context staircase in NORMAL and validate OPTIMAL non-MTP first."
                )
            elif material_latency:
                # A genuine end-to-end MTP winner may lead the final queue.
                ordered.insert(0, best_mtp)
            elif material_decode:
                # Decode-only candidates follow the primary workload winner so an interrupted
                # session still preserves the most useful validated command.
                ordered.append(best_mtp)
            else:
                self._emit("  final-prune: MTP is not materially better than plain; skip duplicate validation.")

        # Deep mode can validate more of the frontier; normal Dense is intentionally strict.
        if self.search_mode == "deep":
            for r in frontier:
                if r not in ordered:
                    ordered.append(r)
            max_final = 5
        elif self.model.kind == ModelKind.DENSE:
            # NORMAL validates the selected Dense branch once. Other solution-level compromises
            # remain visible as measured SCOUT/FULL alternatives; validating both MTP and plain
            # added another ~2.5 minutes without changing the selected branch in common runs.
            max_final = 1
        else:
            max_final = 4

        unique: list[Candidate] = []
        seen: set[str] = set()
        for r in ordered:
            key = r.candidate.key()
            if key not in seen:
                seen.add(key)
                unique.append(copy.deepcopy(r.candidate))
            if len(unique) >= max_final:
                break
        return unique

    @staticmethod
    def _evidence_rank(result: CandidateResult) -> int:
        kind = str(result.metrics.benchmark_kind or "").lower()
        ranks = {"recon": 1, "quick": 2, "recon-context": 3, "full": 4, "validation": 5}
        if kind in ranks:
            return ranks[kind]
        phase = str(result.phase or "").upper()
        if "FINAL" in phase or "VALIDATION" in phase:
            return 5
        if "FULL" in phase or "CONFIRM" in phase or "BOUNDARY" in phase:
            return 4
        if "CONTEXT" in phase:
            return 3
        if "PROBE" in phase or "SCOUT" in phase or "RECON" in phase:
            return 2
        return 0

    def _resolved_evidence(self) -> list[CandidateResult]:
        """Resolve each exact command to its strongest/latest evidence before filtering."""
        strongest: dict[str, tuple[int, int, CandidateResult]] = {}
        for index, result in enumerate(self.results):
            key = result.candidate.key()
            rank = self._evidence_rank(result)
            previous = strongest.get(key)
            if previous is None or rank > previous[0] or (rank == previous[0] and index > previous[1]):
                strongest[key] = (rank, index, result)
        return [row[2] for row in sorted(strongest.values(), key=lambda row: row[1])]

    def _option_index_for(self, candidate: Candidate) -> int:
        for index, option in enumerate(self._solution_options_ordered):
            if (
                option.context == candidate.ctx
                and (option.kv_k, option.kv_v) == (candidate.kv_k, candidate.kv_v)
                and bool(option.vision_required) == bool(candidate.vision)
            ):
                return index
        return 10_000

    def _final_fallback_evidence(
        self,
        failed: Candidate | CandidateResult,
        excluded_keys: set[str],
    ) -> list[CandidateResult]:
        """Return a bounded, evidence-ordered fallback queue after candidate-local FINAL failure."""
        failed_result = failed if isinstance(failed, CandidateResult) else None
        failed_candidate = failed_result.candidate if failed_result is not None else failed
        memory_limited = bool(
            failed_result is not None and (
                self._vram_class(failed_result) == VramOperatingClass.FRAGILE
                or self._recoverable_boundary_reason(failed_result.reason)
            )
        )
        same_semantic_full: list[CandidateResult] = []
        other_full: list[CandidateResult] = []
        scouts: list[CandidateResult] = []
        for result in self._resolved_evidence():
            if result.candidate.key() in excluded_keys or not self._is_good(result):
                continue
            if not result.metrics.pp_tps or not result.metrics.tg_tps:
                continue
            if result.metrics.benchmark_kind == "validation":
                # A validation row in this queue is known failed/non-recommendable evidence.
                continue
            if self._vram_class(result) in {VramOperatingClass.REJECT, VramOperatingClass.FRAGILE}:
                continue
            same_semantic = (
                result.candidate.ctx == failed_candidate.ctx
                and (result.candidate.kv_k, result.candidate.kv_v) == (
                    failed_candidate.kv_k, failed_candidate.kv_v,
                )
                and bool(result.candidate.vision) == bool(failed_candidate.vision)
            )
            if (memory_limited and same_semantic
                    and (result.candidate.ubatch > failed_candidate.ubatch
                         or result.candidate.batch > failed_candidate.batch)):
                # A larger workspace is not a safety fallback after FINAL already
                # downgraded the smaller command for VRAM.  Repair context first.
                continue
            if result.metrics.benchmark_kind == "full" and self._is_recommendable_full(result):
                (same_semantic_full if same_semantic else other_full).append(result)
            else:
                scouts.append(result)

        # First repair the same semantic target (e.g. plain after fragile MTP, or a
        # safer batch). Then walk disclosed solution options. Weak evidence is allowed
        # only because it will receive a fresh FULL before FINAL.
        same_semantic_full.sort(
            key=lambda r: (
                -(r.metrics.vram_free_min_mb or 0),
                r.candidate.ubatch,
                r.candidate.batch,
                workload_latency_seconds(r, self.workload_profile),
            )
        )
        other_full.sort(
            key=lambda r: (
                self._option_index_for(r.candidate),
                workload_latency_seconds(r, self.workload_profile),
                -(r.metrics.vram_free_min_mb or 0),
            )
        )
        scouts.sort(
            key=lambda r: (
                self._option_index_for(r.candidate),
                -self._evidence_rank(r),
                workload_latency_seconds(r, self.workload_profile),
                -(r.metrics.vram_free_min_mb or 0),
            )
        )
        out: list[CandidateResult] = []
        seen_semantic: set[tuple[int, str, str, bool, bool]] = set()
        for result in same_semantic_full + other_full + scouts:
            semantic = (
                result.candidate.ctx, result.candidate.kv_k, result.candidate.kv_v,
                bool(result.candidate.vision), bool(result.candidate.mtp),
            )
            if semantic in seen_semantic:
                continue
            seen_semantic.add(semantic)
            out.append(result)
        return out

    @staticmethod
    def _environmental_final_failure(result: CandidateResult) -> bool:
        return (
            result.status == RunStatus.INVALID_ENVIRONMENT
            or result.reason in {
                "GPU_DID_NOT_RETURN_TO_BASELINE", "EXTERNAL_GPU_LOAD_CHANGED",
                "MODEL_STARTUP_FAILED", "RUN_BUDGET_REACHED", "TIME_BUDGET_REACHED",
            }
        )

    def _set_selected_candidate_status(self, candidate: Candidate) -> None:
        """Update target semantics after a late FINAL fallback changes the winner."""
        matched = next((
            option for option in self._solution_options_ordered
            if option.context == candidate.ctx
            and (option.kv_k, option.kv_v) == (candidate.kv_k, candidate.kv_v)
            and bool(option.vision_required) == bool(candidate.vision)
        ), None)
        if matched is None and self._solution_options_ordered:
            compatible = [
                option for option in self._solution_options_ordered
                if (option.kv_k, option.kv_v) == (candidate.kv_k, candidate.kv_v)
                and bool(option.vision_required) == bool(candidate.vision)
            ]
            if compatible:
                source = min(compatible, key=lambda option: abs(option.context - candidate.ctx))
                matched = copy.deepcopy(source)
                matched.name = f"{source.name}_RUNTIME_REPAIRED"
                matched.context = candidate.ctx
                matched.predicted_placement = candidate.ncmoe if candidate.ncmoe is not None else candidate.ngl
                matched.predicted_free_mb = None
                matched.exact_target = candidate.ctx == (self._declared_target_ctx or candidate.ctx)
                matched.degradation_notes = list(matched.degradation_notes) + [
                    "Runtime FULL/FINAL evidence selected a repaired/fallback command that differs "
                    "from the original static seed; the measured command is authoritative."
                ]
                if candidate.ctx != (self._declared_target_ctx or candidate.ctx) \
                        and DegradationKind.CAPABILITY not in matched.degradation:
                    matched.degradation.append(DegradationKind.CAPABILITY)
        self.selected_option = matched
        declared_ctx = self._declared_target_ctx or candidate.ctx
        same_context = candidate.ctx == declared_ctx
        same_kv = (candidate.kv_k, candidate.kv_v) == self._declared_kv
        if same_context and same_kv:
            if matched is not None and DegradationKind.PERFORMANCE in matched.degradation:
                self.target_status = "SATISFIED_WITH_PERFORMANCE_TRADEOFF"
            else:
                self.target_status = "SATISFIED"
        elif same_context:
            self.target_status = "SATISFIED_WITH_KV_PRECISION_TRADEOFF"
        else:
            self.target_status = "ALTERNATIVE_CAPABILITY_REDUCED"

    def _run_final_fallbacks(
        self,
        failed: Candidate | CandidateResult,
        excluded_keys: set[str],
    ) -> tuple[list[CandidateResult], Candidate | None]:
        max_fallbacks = {"quick": 1, "normal": 2, "deep": 4}.get(self.search_mode, 2)
        passed: list[CandidateResult] = []
        selected: Candidate | None = None
        failed_result = failed if isinstance(failed, CandidateResult) else None
        failed_candidate = failed_result.candidate if failed_result is not None else failed
        fallbacks_used = 0

        # A FRAGILE Dense FINAL is the strongest possible context-boundary evidence.
        # Before changing KV precision (or retrying a larger ubatch), interpolate a
        # smaller context with identical KV/Vision/full-GPU placement and validate it.
        if (failed_result is not None and self.model.kind == ModelKind.DENSE
                and failed_candidate.ngl == "all" and not failed_candidate.mtp
                and self._is_good(failed_result)
                and self._vram_class(failed_result) == VramOperatingClass.FRAGILE
                and self.budget_ok()):
            self._emit(
                f"  FINAL fallback repair: {failed_candidate.ctx}-token command became FRAGILE; "
                "repair context inside the same KV family before crossing the Pareto frontier."
            )
            repaired = self._repair_fragile_dense_full_gpu_context(
                copy.deepcopy(failed_candidate), failed_result,
            )
            if repaired is not None and self.budget_ok():
                fallbacks_used += 1
                excluded_keys.add(repaired.key())
                repair_full = self._best_exact_result(repaired, full_only=True)
                if repair_full is not None:
                    self._emit(f"  fallback FINAL on repaired same-KV context: {repaired.short()}.")
                    final = self._run(
                        copy.deepcopy(repaired), quick=False,
                        phase="FINAL_CONTEXT_REPAIR_VALIDATION", long_validate=True,
                        reference=repair_full,
                    )
                    if self._environmental_final_failure(final):
                        self._emit(
                            "  fallback stopped: repaired FINAL environment is invalid; "
                            "do not reinterpret it as model evidence."
                        )
                        return passed, selected
                    if (self._is_recommendable_full(final)
                            and final.metrics.benchmark_kind == "validation"
                            and final.metrics.long_context_passed):
                        self._emit(
                            "  repaired same-KV FINAL passed; it replaces the fragile primary."
                        )
                        return [final], copy.deepcopy(final.candidate)
                    self._emit(
                        "  repaired same-KV FINAL did not clear the recommendation floor; "
                        "continue to the next semantic frontier point."
                    )

        # Second repair axis: the context-repair block above only covers a full-GPU (ngl="all"),
        # ctx>16384, *softly* FRAGILE FINAL. A FINAL that crosses the absolute VRAM floor outright
        # (EARLY_REJECT, e.g. from a slightly-too-optimistic placement at any context, oversized or
        # not) fell straight through to the generic frontier queue below, which can pick an
        # unrelated candidate over a single step of *more* CPU placement at the *same* context/KV --
        # even when recon already measured that safer placement with a better margin. Reuse the same
        # nearest-safer-placement machinery as live placement recovery (_safer_variants) for one
        # bounded repair attempt before falling through to the frontier queue.
        if (failed_result is not None and self.model.kind == ModelKind.DENSE
                and not failed_candidate.mtp and self._recoverable_boundary_reason(failed_result.reason)
                and self.budget_ok()):
            # max_steps=2 (not 1): for an already-numeric ngl, _safer_variants' own range excludes
            # its stop bound, so max_steps=1 only re-yields the current placement (deduped away) and
            # never actually produces a safer step. max_steps=2 reliably yields exactly one step
            # more conservative than current for both the numeric and "all" starting placements;
            # only that single first step (variants[1]) is ever used here.
            variants = self._safer_variants(failed_candidate, max_steps=2)
            safer_candidate = variants[1] if len(variants) > 1 else None
            if safer_candidate is not None and safer_candidate.key() not in excluded_keys:
                self._emit(
                    f"  FINAL fallback repair: {failed_candidate.short()} crossed the absolute VRAM "
                    f"floor; try one step more conservative placement (ngl={safer_candidate.ngl}) at "
                    "the same context/KV before crossing to a different frontier point."
                )
                fallbacks_used += 1
                excluded_keys.add(safer_candidate.key())
                repair_full = self._guarded_full(safer_candidate, "FINAL_PLACEMENT_REPAIR_CONFIRM", failed_result)
                if self._is_recommendable_full(repair_full):
                    self._mark_tight(repair_full)
                    final = self._run(
                        copy.deepcopy(repair_full.candidate), quick=False,
                        phase="FINAL_PLACEMENT_REPAIR_VALIDATION", long_validate=True,
                        reference=repair_full,
                    )
                    if self._environmental_final_failure(final):
                        self._emit(
                            "  fallback stopped: repaired-placement FINAL environment is invalid; "
                            "do not reinterpret it as model evidence."
                        )
                        return passed, selected
                    if (self._is_recommendable_full(final) and final.metrics.benchmark_kind == "validation"
                            and final.metrics.long_context_passed):
                        self._emit(
                            "  repaired same-context/KV placement FINAL passed; it replaces the failed primary."
                        )
                        return [final], copy.deepcopy(final.candidate)
                    self._emit(
                        "  repaired-placement FINAL did not clear the recommendation floor; "
                        "continue to the next semantic frontier point."
                    )

        queue = self._final_fallback_evidence(failed, excluded_keys)
        if queue:
            self._emit(
                f"  FINAL fallback ladder opened: up to {max_fallbacks - fallbacks_used} "
                "next-frontier candidate(s); "
                "environment failures never trigger semantic fallback."
            )
        for evidence in queue[:max(0, max_fallbacks - fallbacks_used)]:
            if not self.budget_ok():
                break
            candidate = copy.deepcopy(evidence.candidate)
            excluded_keys.add(candidate.key())
            full = evidence
            if evidence.metrics.benchmark_kind != "full" or not self._is_recommendable_full(evidence):
                self._emit(
                    f"  fallback FULL confirmation before FINAL: {candidate.short()} "
                    f"(source={evidence.metrics.benchmark_kind or 'unknown'})."
                )
                full = self._run(
                    candidate, quick=False, phase="FINAL_FALLBACK_CONFIRM",
                    reference=evidence,
                )
                if self._environmental_final_failure(full):
                    self._emit("  fallback stopped: GPU/runtime environment is invalid, not candidate-local.")
                    break
                if not self._is_recommendable_full(full):
                    self._emit(f"  fallback FULL rejected ({full.reason}); continue to the next frontier point.")
                    continue
                candidate = copy.deepcopy(full.candidate)
            self._emit(f"  fallback FINAL: {candidate.short()}.")
            final = self._run(
                candidate, quick=False, phase="FINAL_FALLBACK_VALIDATION",
                long_validate=True, reference=full,
            )
            if self._environmental_final_failure(final):
                self._emit("  fallback stopped: FINAL environment is invalid; do not reinterpret it as model evidence.")
                break
            if (
                self._is_recommendable_full(final)
                and final.metrics.benchmark_kind == "validation"
                and final.metrics.long_context_passed
            ):
                passed.append(final)
                selected = copy.deepcopy(final.candidate)
                self._emit("  fallback FINAL passed; this candidate replaces the downgraded primary.")
                break
            self._emit(
                f"  fallback FINAL did not earn recommendation status ({final.reason}, "
                f"VRAM={self._vram_class(final).value}); continue if budget remains."
            )
        return passed, selected

    @staticmethod
    def _recoverable_boundary_reason(reason: str) -> bool:
        # Only explicit memory evidence is a placement-recovery signal. PP/TG cliffs can come
        # from KV kernels, context scaling or CPU execution itself and must not trigger more offload.
        return reason in {
            "EARLY_REJECT_DANGEROUS_VRAM",
            "EARLY_REJECT_ABSOLUTE_VRAM_FLOOR",
            "FAIL_OOM",
        }

    def _safer_variants(self, candidate: Candidate, max_steps: int | None = None) -> list[Candidate]:
        """Return placement variants from current placement toward more CPU/RAM headroom."""
        if max_steps is None:
            max_steps = {"quick": 4, "normal": 8, "deep": 12}.get(self.search_mode, 8)
        out: list[Candidate] = []
        if self.model.kind == ModelKind.MOE and candidate.ncmoe is not None:
            blocks = max(candidate.ncmoe, model_main_block_count(self.model))
            for n in range(candidate.ncmoe, min(blocks, candidate.ncmoe + max_steps) + 1):
                c = copy.deepcopy(candidate)
                c.ncmoe = n
                out.append(c)
            return out
        if self.model.kind == ModelKind.DENSE:
            total = model_main_block_count(self.model)
            start = total if candidate.ngl == "all" else int(candidate.ngl)
            # Try the exact current form first, then progressively fewer GPU layers.
            out.append(copy.deepcopy(candidate))
            numeric_start = min(total, start)
            for n in range(numeric_start, max(-1, numeric_start - max_steps), -1):
                c = copy.deepcopy(candidate)
                c.ngl = n
                if c.key() not in {x.key() for x in out}:
                    out.append(c)
            return out
        return [copy.deepcopy(candidate)]

    def _recover_full(self, candidate: Candidate, phase: str,
                      reference: CandidateResult | None = None) -> CandidateResult | None:
        """Find the nearest safer placement that survives a guarded FULL workload.

        MoE can often trade expert residency for headroom with a modest performance cost. Dense
        models are different: moving a target layer to CPU can introduce a large per-token PCIe/CPU
        penalty. For Dense recovery we therefore refuse to accept a partial-offload candidate that
        falls below the configured performance-retention floor relative to the current reference.
        """
        best_failed_free = -1
        non_improving_failures = 0
        for idx, c in enumerate(self._safer_variants(candidate)):
            if not self.budget_ok():
                return None
            cached = self._best_exact_result(c, full_only=True)
            if cached is not None:
                self._emit(f"  reuse: already have FULL result for {c.short()} → {self._summary(cached)}")
                r = cached
            else:
                if idx:
                    if self.model.kind == ModelKind.MOE:
                        self._emit(f"  recovery: retry same settings with safer ncmoe={c.ncmoe}.")
                    else:
                        self._emit(f"  recovery: retry same settings with safer ngl={c.ngl}.")
                r = self._guarded_full(c, phase, reference)

            if self._is_recommendable_full(r):
                if self.model.kind == ModelKind.DENSE and c.ngl != candidate.ngl:
                    retention = self._tg_retention(r, reference)
                    if retention is not None and retention < self.safe_perf_floor_ratio:
                        self._emit(
                            f"  Dense recovery stopped: ngl={c.ngl} retains only {retention:.0%} "
                            f"of reference decode (< {self.safe_perf_floor_ratio:.0%}). "
                            "More CPU offload is not a useful recovery path."
                        )
                        return None
                self._mark_tight(r)
                return r

            if self._is_good(r) and r.metrics.benchmark_kind in {"full", "validation"}:
                self._emit(
                    f"  recovery: {c.short()} completed but is {self._vram_class(r).value} at "
                    f"{r.metrics.vram_free_min_mb} MiB; continue to the nearest safer placement."
                )
                continue

            if not self._recoverable_boundary_reason(r.reason):
                self._emit(f"  recovery stopped: {r.reason} is not a normal memory-boundary signal.")
                return None

            # Dense VRAM versus numeric ngl is not guaranteed to be monotonic because output/special
            # tensors and CUDA graph allocations can change placement. If several successively safer
            # numeric candidates fail without increasing measured headroom, stop wasting runs.
            if self.model.kind == ModelKind.DENSE and idx > 0:
                free = r.metrics.vram_free_min_mb
                if free is not None:
                    if free <= best_failed_free + 32:
                        non_improving_failures += 1
                    else:
                        best_failed_free = free
                        non_improving_failures = 0
                    if non_improving_failures >= 2:
                        self._emit(
                            "  Dense recovery stopped: measured VRAM headroom is not improving "
                            "monotonically as ngl decreases."
                        )
                        return None

        self._emit("  recovery exhausted without a stable placement.")
        return None

    def _find_safe_reserve_candidate(self, candidate: Candidate,
                                     reference: CandidateResult | None = None) -> CandidateResult | None:
        """Find useful headroom without burning runs on destructive Dense partial offload.

        For Dense ``ngl=all`` the first memory-saving action is to reuse already measured smaller
        batch/ubatch full-GPU configurations. If one is within normal measurement noise of the
        requested reserve, NORMAL/QUICK stop there and report its measured VRAM operating class
        instead of moving a target block to CPU. If headroom is materially short, NORMAL permits
        only one numeric-ngl experiment; DEEP retains the longer recovery walk.
        """
        self._emit(
            f"\n[Phase 4b] VRAM headroom check: preferred reserve {self.vram_margin_mb} MiB "
            f"with >= {self.safe_perf_floor_ratio:.0%} of reference decode"
        )
        perf_reference = reference or self._best_exact_result(candidate, full_only=True)
        if self.model.kind == ModelKind.DENSE and self._is_tight(perf_reference):
            self._emit(
                "  performance anchor is TIGHT and already FULL-proven. Keep its confirmed workspace "
                "locked; do not spend more launches chasing the comfort reserve. The report will expose "
                "the headroom class and operating limits explicitly."
            )
            return None

        if self.model.kind == ModelKind.DENSE and candidate.ngl == "all":
            full_gpu = [
                r for r in self.results
                if (not r.candidate.mtp and r.candidate.ngl == "all"
                    and r.candidate.ctx == candidate.ctx and self._is_good(r)
                    and r.metrics.benchmark_kind in {"full", "validation"}
                    and r.metrics.vram_free_min_mb is not None)
            ]
            performant = []
            for r in full_gpu:
                retention = self._tg_retention(r, perf_reference)
                if retention is None or retention >= self.safe_perf_floor_ratio:
                    performant.append(r)
            if performant:
                safest_full_gpu = max(
                    performant,
                    key=lambda r: ((r.metrics.vram_free_min_mb or 0), self._perf_score(r)),
                )
                free = safest_full_gpu.metrics.vram_free_min_mb or 0
                retention = self._tg_retention(safest_full_gpu, perf_reference)
                self._emit(
                    f"  safest already-measured full-GPU candidate: {safest_full_gpu.candidate.short()} → "
                    f"{free} MiB free" + (f", decode retained={retention:.0%}." if retention is not None else ".")
                )
                if free >= self.vram_margin_mb:
                    self._emit("  preferred headroom already satisfied without moving any target layer to CPU.")
                    return safest_full_gpu

                # Preferred reserve is a comfort target, not a reason to halve Dense decode speed.
                # In NORMAL/QUICK tolerate a modest shortfall (up to 256 MiB / 20%) once full-GPU
                # has been proven stable above the hard floor. DEEP may still explore the boundary.
                tolerance = (max(64, int(round(self.vram_margin_mb * 0.05)))
                             if self.search_mode == "deep" else max(256, int(round(self.vram_margin_mb * 0.20))))
                shortfall = self.vram_margin_mb - free
                operational_floor = self._operational_vram_floor_mb(vision=candidate.vision)
                if self.search_mode != "deep" and free >= operational_floor:
                    self._emit(
                        f"  full-GPU headroom {free} MiB is above the operational floor {operational_floor} MiB. "
                        "Preferred reserve is not a reason to move Dense target layers to CPU in NORMAL/QUICK; "
                        "keep the full-GPU profile and report the reserve shortfall explicitly."
                    )
                    return None
                if shortfall <= tolerance:
                    self._emit(
                        f"  reserve shortfall is only {shortfall} MiB (<= {tolerance} MiB measurement tolerance); "
                        "skip Dense partial offload. Reporting will expose the measured VRAM class and "
                        "reserve shortfall instead of inventing a separate launch role."
                    )
                    return None

            # Full-GPU is proven stable. NORMAL gets at most one partial-offload experiment; a long
            # 64->63->62 walk is disproportionate because every CPU target block sits on decode's
            # critical path. DEEP can still explore it explicitly.
            if self.search_mode != "deep":
                total = model_main_block_count(self.model)
                c = copy.deepcopy(candidate)
                c.ngl = total
                self._emit(
                    f"  full-GPU headroom is materially below target; trying exactly one partial-offload "
                    f"control point ngl={total}."
                )
                r = self._guarded_full(c, "VRAM_HEADROOM_SEARCH", perf_reference)
                if self._is_good(r):
                    retention = self._tg_retention(r, perf_reference)
                    free = r.metrics.vram_free_min_mb or 0
                    if retention is not None and retention < self.safe_perf_floor_ratio:
                        self._emit(
                            f"  Dense headroom stop: ngl={total} retains only {retention:.0%} of decode; "
                            "do not test lower ngl values."
                        )
                        return None
                    if free >= self.vram_margin_mb:
                        self._emit(
                            f"  preferred headroom satisfied at the single control point: {free} MiB free, "
                            + (f"decode retained={retention:.0%}." if retention is not None else "performance acceptable.")
                        )
                        return r
                self._emit("  one-step Dense headroom control did not justify further CPU offload; stop.")
                return None

        if (self.model.kind == ModelKind.DENSE and candidate.ngl != "all"
                and self._dense_oversized_active and self.search_mode != "deep"):
            self._emit(
                "  oversized Dense headroom check reuses the context/placement frontier; do not spend another "
                "multi-layer walk solely to chase the preferred reserve. DEEP can explore that trade-off."
            )
            return None

        if self.model.kind == ModelKind.MOE and self._is_recommendable_full(perf_reference):
            free = int(perf_reference.metrics.vram_free_min_mb or 0)
            tolerance = max(64, int(round(self.vram_margin_mb * 0.05)))
            shortfall = self.vram_margin_mb - free
            if 0 < shortfall <= tolerance:
                self._emit(
                    f"  MoE reserve shortfall is only {shortfall} MiB (<= {tolerance} MiB FULL/FINAL "
                    "hysteresis). Keep the faster confirmed expert residency; do not move another "
                    "expert layer to CPU for a comfort-threshold measurement fluctuation."
                )
                return perf_reference

        # MoE and DEEP Dense keep the adaptive safer-placement walk.
        for idx, c in enumerate(self._safer_variants(candidate, max_steps=10)):
            if not self.budget_ok():
                return None
            cached = self._best_exact_result(c, full_only=True)
            if cached is not None and self._is_good(cached):
                r = cached
                self._emit(f"  reuse: {c.short()} → {self._summary(r)}")
            else:
                if idx:
                    if self.model.kind == ModelKind.MOE:
                        self._emit(f"  VRAM headroom: trying safer ncmoe={c.ncmoe}.")
                    else:
                        self._emit(f"  VRAM headroom: trying safer ngl={c.ngl}.")
                r = self._guarded_full(c, "VRAM_HEADROOM_SEARCH", perf_reference)

            if self._is_good(r):
                free = r.metrics.vram_free_min_mb or 0
                retention = self._tg_retention(r, perf_reference)
                retention_ok = retention is None or retention >= self.safe_perf_floor_ratio
                if not retention_ok:
                    self._emit(
                        f"  headroom performance floor reached: {c.short()} retains {retention:.0%} "
                        f"of reference decode (< {self.safe_perf_floor_ratio:.0%}). Stop."
                    )
                    return None
                if free >= self.vram_margin_mb:
                    self._emit(
                        f"  preferred headroom satisfied: {free} MiB >= {self.vram_margin_mb} MiB"
                        + (f", decode retained={retention:.0%}." if retention is not None else ".")
                    )
                    return r
                self._emit(
                    f"  preferred headroom not yet met: {free} MiB < {self.vram_margin_mb} MiB"
                    + (f"; decode retained={retention:.0%}." if retention is not None else "; continue safer.")
                )
            elif not self._recoverable_boundary_reason(r.reason):
                self._emit(f"  VRAM headroom search stopped by non-boundary failure: {r.reason}")
                return None
        self._emit("  VRAM headroom search exhausted without a useful candidate meeting the preferred reserve.")
        return None

    def _ubatch_values(self, base_ubatch: int = 256) -> list[int]:
        if self.search_mode == "quick":
            vals = [512, 1024, 1536]
        elif self.search_mode == "deep":
            vals = [512, 768, 1024, 1280, 1536, 1792, 2048]
        else:
            vals = [512, 1024, 1536, 2048]
        return sorted(set(v for v in vals if v >= 256))

    def _dense_full_gpu_ubatch_search(self, base: Candidate,
                                       reference: CandidateResult | None = None) -> CandidateResult | None:
        """Adaptive, noise-aware Dense full-GPU ubatch search.

        Search the *marginal value* of larger ubatches instead of testing a fixed grid. All probe
        points use the same batch=2048 so PP is comparable. 44 vs 45 t/s is explicitly a tie under
        the default noise policy. We continue only when a larger ubatch produces a material prefill
        or end-to-end latency win without a material decode regression.
        """
        if self.model.kind == ModelKind.DENSE and self._is_tight(reference):
            self._mark_tight(reference)
            self._emit(
                "\n[Phase 2 / DENSE_FULL_GPU] Confirmed anchor is in the TIGHT VRAM band; "
                "lock its batch/ubatch and skip workspace growth."
            )
            return reference
        self._emit(
            "\n[Phase 2 / DENSE_FULL_GPU] Adaptive ubatch search: compare generation and context-fill "
            "speed separately; ignore changes inside the noise zone."
        )
        self._emit(
            f"  noise policy: same-context FULL differences <= {self.noise_policy.decode_rel:.0%} tie; "
            f"SCOUT/cross-context promotion requires > {self.noise_policy.decode_probe_rel:.0%}; "
            f"prefill ±{self.noise_policy.prefill_rel:.0%}."
        )

        # Exponential search gives a clean diminishing-returns signal. NORMAL has one look-ahead
        # after a tie so a non-linear jump at 1024 is not missed; DEEP can add a local midpoint.
        values = [256, 512] if self.search_mode == "quick" else [256, 512, 1024, 2048]
        probes: list[CandidateResult] = []
        best_probe: CandidateResult | None = None
        lookahead_used = False

        for idx, ub in enumerate(values):
            if not self.budget_ok():
                break
            c = copy.deepcopy(base)
            c.ngl = "all"
            c.ncmoe = None
            c.batch = max(2048, ub)
            c.ubatch = ub
            r = self._run(c, quick=True, phase="DENSE_UBATCH_PROBE", reference=None)
            if not self._is_good(r):
                if self._recoverable_boundary_reason(r.reason):
                    self._emit(
                        f"  ubatch={ub} crossed the full-GPU memory boundary ({r.reason}); stop larger ubatches."
                    )
                    break
                self._emit(f"  ubatch={ub} stopped by non-boundary failure {r.reason}.")
                break
            if self._vram_class(r) in {VramOperatingClass.REJECT, VramOperatingClass.FRAGILE}:
                self._emit(
                    f"  ubatch={ub} is only {self._vram_class(r).value} at "
                    f"{r.metrics.vram_free_min_mb} MiB; stop larger workspace points."
                )
                break

            probes.append(r)
            self._emit(
                f"  ubatch={ub}: PP={profile_prefill_tps(r, self.workload_profile):.1f} t/s effective-prefill, "
                f"TG={self._robust_tg(r):.1f} t/s, free={r.metrics.vram_free_min_mb or 0} MiB"
            )

            if best_probe is None:
                best_probe = r
                continue

            dr = decode_relation(r, best_probe, self.noise_policy)
            pr = prefill_relation(r, best_probe, self.workload_profile, self.noise_policy)
            lr = latency_relation(r, best_probe, self.workload_profile, self.noise_policy)
            preferred = self._prefer(r, best_probe)

            if dr < 0 and pr <= 0:
                self._emit(
                    f"  early-stop: ubatch={ub} is materially worse in decode and does not improve prefill."
                )
                break

            if preferred is r:
                if dr > 0 or pr > 0 or lr > 0:
                    self._emit(
                        f"  material gain at ubatch={ub}: decode={'better' if dr>0 else 'tie'}, "
                        f"prefill={'better' if pr>0 else ('worse' if pr<0 else 'tie')}; continue upward."
                    )
                    best_probe = r
                    continue
                # Equivalent performance but r can only win tie-breaks if it has more headroom,
                # which a larger ubatch normally does not. Keep the smaller setting.
                best_probe = preferred

            if dr == 0 and pr == 0 and lr == 0:
                if self.search_mode == "normal" and ub == 512 and not lookahead_used:
                    lookahead_used = True
                    self._emit(
                        "  ubatch=512 is inside the noise zone versus 256; take one 1024 look-ahead, "
                        "then stop unless the gain becomes material."
                    )
                    continue
                self._emit(
                    f"  diminishing returns: ubatch={ub} is performance-equivalent to the current best; stop."
                )
                break

            # A trade-off can still be worthwhile when representative prefill+generation latency
            # improves materially. Otherwise keep the lower-memory candidate and stop the ladder.
            if lr <= 0 and preferred is not r:
                self._emit(
                    f"  ubatch={ub} offers no material end-to-end latency win; stop the ladder."
                )
                break
            if preferred is r:
                best_probe = r

        if not probes or best_probe is None:
            self._emit("  no Dense full-GPU ubatch probe survived; keep the validated anchor.")
            return reference

        # Optional midpoint only in DEEP when the exponential ladder bracketed a real optimum.
        if self.search_mode == "deep" and len(probes) >= 3 and self.budget_ok():
            ordered = sorted({r.candidate.ubatch for r in probes})
            winner_ub = best_probe.candidate.ubatch
            pos = ordered.index(winner_ub)
            neighbor = None
            if pos + 1 < len(ordered):
                neighbor = ordered[pos + 1]
            elif pos > 0:
                neighbor = ordered[pos - 1]
            if neighbor is not None and abs(neighbor - winner_ub) >= 512:
                mid = (neighbor + winner_ub) // 2
                mid = max(256, (mid // 128) * 128)
                if mid not in ordered:
                    c = copy.deepcopy(base); c.ngl = "all"; c.batch = max(2048, mid); c.ubatch = mid
                    r = self._run(c, quick=True, phase="DENSE_UBATCH_REFINE_PROBE", reference=None)
                    if self._is_good(r):
                        probes.append(r)
                        best_probe = self._prefer(best_probe, r)

        # Do not FULL-confirm a batch/ubatch family that failed to show any material gain over the
        # already FULL-validated phase-1 anchor. v0.5.5 always paid another ~27 s confirmation even
        # when every quick sample was inside the noise zone or slower.
        if reference is not None and self._is_good(reference):
            dr_ref = decode_relation(best_probe, reference, self.noise_policy)
            pr_ref = prefill_relation(best_probe, reference, self.workload_profile, self.noise_policy)
            lr_ref = latency_relation(best_probe, reference, self.workload_profile, self.noise_policy)
            if dr_ref <= 0 and pr_ref <= 0 and lr_ref <= 0:
                self._emit(
                    "  no batch/ubatch probe materially improves the already FULL-validated anchor; "
                    "skip duplicate FULL confirmation."
                )
                return reference

        # One expensive confirmation only when a quick candidate actually earned it. The phase-1
        # full result remains in the pool as the low-memory anchor.
        c = copy.deepcopy(best_probe.candidate)
        self._emit(
            f"  FULL-confirming one noise-aware winner: ubatch={c.ubatch}; "
            f"estimated {self.workload_profile} cycle={workload_latency_seconds(best_probe, self.workload_profile):.2f}s."
        )
        confirmed = self._run(c, quick=False, phase="DENSE_UBATCH_CONFIRM", reference=reference)
        pool = [r for r in [reference, confirmed] if self._is_recommendable_full(r)]
        if not pool:
            return None
        best = choose_preferred(pool, self.workload_profile, self.noise_policy) or pool[0]
        if len(pool) == 2 and self._performance_equivalent(pool[0], pool[1]):
            self._emit(
                "  FULL results are performance-equivalent inside the noise zone; prefer the candidate "
                "with more VRAM headroom / lower complexity."
            )
        self._emit(
            f"  Dense full-GPU winner: {best.candidate.short()} → {self._summary(best)} | "
            f"effective prefill={profile_prefill_tps(best, self.workload_profile):.1f} t/s | "
            f"{self.workload_profile} cycle≈{workload_latency_seconds(best, self.workload_profile):.2f}s"
        )
        return best

    def _dense_partial_ubatch_search(self, base: Candidate,
                                    reference: CandidateResult | None = None) -> CandidateResult | None:
        """Tune PP after numeric Dense placement without sacrificing GPU layers for a larger ubatch.

        In oversized Dense, whole-layer residency dominates decode. NORMAL therefore keeps ngl fixed,
        screens only 512/1024, and FULL-confirms at most one ubatch that materially improves the
        representative workload. If a larger ubatch needs ngl-1, it is rejected here rather than
        exchanging decode speed for a prettier isolated PP number.
        """
        self._emit(
            "\n[Phase 2 / DENSE_PARTIAL] Placement-first ubatch tuning: keep ngl fixed; optimize prompt "
            "processing only when end-to-end latency improves without a material decode loss."
        )
        if reference is None or not self._is_good(reference):
            return reference
        if self._is_tight(reference):
            self._mark_tight(reference)
            self._emit(
                "  confirmed oversized-Dense anchor is TIGHT; lock b/ub at the FULL-proven values. "
                "Larger ubatches cannot borrow from this 64 MiB hysteresis reserve."
            )
            return reference
        screens: list[CandidateResult] = []
        for ub in ([512] if self.search_mode == "quick" else [512, 1024]):
            if not self.budget_ok() or ub <= base.ubatch:
                continue
            c = copy.deepcopy(base); c.batch = max(2048, ub); c.ubatch = ub
            r = self._run(c, quick=True, phase=f"DENSE_PARTIAL_UBATCH_{ub}", reference=reference)
            if not self._is_good(r):
                if self._recoverable_boundary_reason(r.reason):
                    self._emit(
                        f"  ubatch={ub} cannot fit at ngl={base.ngl}; stop. NORMAL will not lower ngl "
                        "just to increase batch size."
                    )
                else:
                    self._emit(f"  ubatch={ub} stopped by {r.reason}.")
                break
            if self._vram_class(r) in {VramOperatingClass.REJECT, VramOperatingClass.FRAGILE}:
                self._emit(
                    f"  ubatch={ub} SCREEN is {self._vram_class(r).value} at "
                    f"{r.metrics.vram_free_min_mb} MiB; do not promote it to FULL."
                )
                break
            screens.append(r)
            dr = decode_relation(r, reference, self.noise_policy)
            pr = prefill_relation(r, reference, self.workload_profile, self.noise_policy)
            lr = latency_relation(r, reference, self.workload_profile, self.noise_policy)
            self._emit(
                f"  ubatch={ub} SCREEN: PP={r.metrics.pp_tps or 0:.1f} | TG={r.metrics.tg_tps or 0:.1f} | "
                f"free={r.metrics.vram_free_min_mb} MiB | relations decode={dr:+d}, prefill={pr:+d}, latency={lr:+d}"
            )
            if dr < 0:
                self._emit("  material decode loss: larger ubatch does not earn another launch.")
                break
            if pr <= 0 and lr <= 0:
                self._emit("  no material PP or end-to-end gain: stop the ubatch ladder.")
                break

        if not screens:
            return reference
        candidate = choose_preferred([reference, *screens], self.workload_profile, self.noise_policy) or reference
        if candidate is reference:
            self._emit("  baseline placement remains best; no duplicate FULL confirmation.")
            return reference
        if decode_relation(candidate, reference, self.noise_policy) < 0 \
                or (latency_relation(candidate, reference, self.workload_profile, self.noise_policy) <= 0
                    and prefill_relation(candidate, reference, self.workload_profile, self.noise_policy) <= 0):
            self._emit("  larger ubatch does not materially improve the workload; keep baseline.")
            return reference
        self._emit(
            f"  one ubatch candidate earned FULL: ubatch={candidate.candidate.ubatch}, same ngl={base.ngl}."
        )
        full = self._run(copy.deepcopy(candidate.candidate), quick=False,
                         phase="DENSE_PARTIAL_UBATCH_CONFIRM", reference=reference)
        if not self._is_recommendable_full(full):
            self._emit(f"  ubatch FULL failed ({full.reason}); keep baseline placement.")
            return reference
        self._mark_tight(full)
        return choose_preferred([reference, full], self.workload_profile, self.noise_policy) or reference

    def _joint_ubatch_placement_search(self, base: Candidate, phase: str,
                                       reference: CandidateResult | None = None) -> CandidateResult | None:
        """Jointly tune ubatch and memory placement instead of treating them as independent phases."""
        if self._is_tight(reference):
            self._mark_tight(reference)
            self._emit(
                f"\n[{phase}] Confirmed anchor is TIGHT; keep its exact ubatch/placement and skip "
                "joint workspace growth in every search mode."
            )
            return reference if reference.candidate.mtp == base.mtp else None
        self._emit(
            f"\n[{phase}] Joint ubatch/placement search: larger ubatch may automatically move "
            "experts/layers back to CPU instead of terminating the search."
        )
        successes: list[CandidateResult] = []
        placement_seed = copy.deepcopy(base)
        best_score = -1.0
        for ub in self._ubatch_values(base.ubatch):
            if not self.budget_ok():
                break
            c = copy.deepcopy(placement_seed)
            c.batch = max(2048, ub)
            c.ubatch = ub
            self._emit(f"  trying ubatch={ub} with adaptive placement recovery...")
            dynamic_reference = max(successes, key=self._perf_score) if successes else reference
            r = self._recover_full(c, f"{phase}_UB{ub}", dynamic_reference)
            if r is None:
                self._emit(f"  ubatch={ub}: no stable placement found; stop searching larger ubatch values.")
                break
            successes.append(r)
            placement_seed = copy.deepcopy(r.candidate)
            score = self._perf_score(r)
            self._emit(f"  ubatch={ub}: stable as {r.candidate.short()} → {self._summary(r)}")
            if best_score > 0 and score < best_score * 0.70:
                self._emit(
                    f"  ubatch={ub}: full-workload score fell to {score/best_score:.0%} of best; "
                    "larger values are unlikely to be useful."
                )
                break
            best_score = max(best_score, score)

        if not successes:
            # Never masquerade an out-of-family reference as a successful joint-search result.
            # This was especially harmful for MTP: a failed speculative search returned the non-MTP
            # reference, after which n-max/p-min refinement silently ran with MTP disabled.
            return None
        best = max(successes, key=self._perf_score)
        self._emit(f"  joint-search winner: {best.candidate.short()} → {self._summary(best)}")
        return best

    def _moe_screen_ubatch_placement_search(self, base: Candidate, phase: str,
                                            reference: CandidateResult | None = None) -> CandidateResult | None:
        """SCREEN → REFINE → CONFIRM for MoE ubatch/expert placement.

        v0.5.8 used GUARDED_FULL for every ubatch and every recovery placement.  On a 28 GiB
        MoE that turns a four-point sweep into several minutes.  NORMAL now maps the joint
        surface with QUICK probes, refines one adjacent expert placement, then FULL-confirms
        at most two measured finalists.  DEEP keeps the older exhaustive path.
        """
        floor = self._moe_screen_vram_floor_mb(vision=base.vision)
        self._emit(
            f"\n[{phase}] MoE SCREEN → REFINE → CONFIRM: quick-map ubatch/expert residency; "
            f"recommended screen points need >= {floor} MiB headroom before a FULL is earned."
        )
        if self.search_mode == "quick":
            ub_values = [512, 1024]
        else:
            # NORMAL screens the endpoints first.  A midpoint is only earned when 2048 cannot
            # be stabilized; DEEP still owns dense ubatch grids.
            ub_values = [512, 2048]

        blocks = model_main_block_count(self.model)
        seed = copy.deepcopy(base)
        screens: list[CandidateResult] = []
        attempted: set[str] = set()

        def screen(c: Candidate, label: str) -> CandidateResult:
            key = c.key()
            cached = self._cached_for_run(c, quick=True, long_validate=False, guard_probe=False, recon=False)
            if cached is not None:
                self._emit(f"  reuse SCREEN: {c.short()} → {self._summary(cached)}")
                return cached
            self._emit(f"  {label}: {c.short()}")
            attempted.add(key)
            return self._run(c, quick=True, phase=f"{phase}_SCREEN", reference=None)

        def stable_screen_for(c: Candidate, label: str) -> CandidateResult | None:
            # One calibrated recovery is normally enough after Phase 0.  Permit a second only
            # when the first still misses the operating floor by a small amount.
            attempts = 0
            cur = copy.deepcopy(c)
            while attempts < 3 and self.budget_ok():
                r = screen(cur, label if attempts == 0 else f"{label} recovery")
                free = r.metrics.vram_free_min_mb or 0
                if self._is_good(r) and free >= floor:
                    return r
                if not (self._recoverable_boundary_reason(r.reason) or (self._is_good(r) and free < floor)):
                    return None
                if cur.ncmoe is None or cur.ncmoe >= blocks:
                    return None
                cur = copy.deepcopy(cur)
                cur.ncmoe += 1
                attempts += 1
                self._emit(
                    f"  SCREEN headroom/recovery: move one expert layer to CPU -> ncmoe={cur.ncmoe} "
                    f"(previous free={free} MiB, target>={floor})."
                )
            return None

        placement_seed = copy.deepcopy(seed)
        need_midpoint = False
        for ub in ub_values:
            if not self.budget_ok():
                break
            c = copy.deepcopy(placement_seed)
            c.batch = max(2048, ub)
            c.ubatch = ub
            r = stable_screen_for(c, f"ubatch={ub} screen")
            if r is None:
                if ub == 2048:
                    need_midpoint = True
                else:
                    self._emit(f"  ubatch={ub}: no stable SCREEN point; do not pay for FULL.")
                continue
            screens.append(r)
            placement_seed = copy.deepcopy(r.candidate)
            self._emit(
                f"  ubatch={ub} SCREEN -> PP={r.metrics.pp_tps or 0:.1f} | TG={r.metrics.tg_tps or 0:.1f} | "
                f"ncmoe={r.candidate.ncmoe} | free={r.metrics.vram_free_min_mb} MiB"
            )

        if need_midpoint and self.search_mode == "normal" and self.budget_ok():
            c = copy.deepcopy(placement_seed)
            c.batch = 2048; c.ubatch = 1536
            r = stable_screen_for(c, "ubatch=1536 midpoint screen")
            if r is not None:
                screens.append(r)

        if not screens:
            self._emit("  MoE joint SCREEN found no stable point.")
            return reference if reference is not None and self._is_good(reference) else None

        winner = choose_preferred(screens, self.workload_profile, self.noise_policy) or screens[0]

        # A safer adjacent ncmoe can occasionally be faster under heavy VRAM pressure (the real
        # Ornith log showed ncmoe=26 beating ncmoe=25 at ub=2048).  Measure exactly one neighbor
        # before confirmation instead of running a separate local-placement FULL phase.
        if winner.candidate.ncmoe is not None and winner.candidate.ncmoe < blocks and self.budget_ok():
            neighbor = copy.deepcopy(winner.candidate)
            neighbor.ncmoe += 1
            nr = stable_screen_for(neighbor, "adjacent safer-placement screen")
            if nr is not None:
                screens.append(nr)
                preferred = choose_preferred([winner, nr], self.workload_profile, self.noise_policy)
                if preferred is nr:
                    winner = nr

        # A 5–10% same-context SCREEN difference is explicitly a confirmation band.  The generic
        # conservative comparator treats it as a tie and would otherwise choose the safer ncmoe,
        # even when the more GPU-resident point has lower raw end-to-end latency.  Confirm both
        # bounded contenders once; two FULL results can resolve the band at the 5% threshold.
        gray_companion: CandidateResult | None = None
        raw_latency_winner = min(screens, key=lambda r: workload_latency_seconds(r, self.workload_profile))
        if (raw_latency_winner.candidate.key() != winner.candidate.key()
                and workload_latency_seconds(raw_latency_winner, self.workload_profile)
                    < workload_latency_seconds(winner, self.workload_profile)
                and decode_requires_confirmation(raw_latency_winner, winner, self.noise_policy)):
            gray_companion = winner
            winner = raw_latency_winner
            self._emit(
                "  same-context 5–10% confirmation band: the lower-latency SCREEN point and the "
                "safer-headroom point will each receive one FULL measurement before selection."
            )

        # Rank unique SCREEN finalists; a selected point earns one FULL.  A second FULL is allowed
        # only if the first unexpectedly fails stronger memory/context guards.
        unique: dict[str, CandidateResult] = {}
        for r in screens:
            prev = unique.get(r.candidate.key())
            if prev is None or self._perf_score(r) > self._perf_score(prev):
                unique[r.candidate.key()] = r
        remaining = list(unique.values())
        ordered: list[CandidateResult] = []
        for forced in (winner, gray_companion):
            if forced is None:
                continue
            match = next((r for r in remaining if r.candidate.key() == forced.candidate.key()), None)
            if match is not None:
                ordered.append(match)
                remaining.remove(match)
        while remaining:
            nxt = choose_preferred(remaining, self.workload_profile, self.noise_policy) or remaining[0]
            ordered.append(nxt); remaining.remove(nxt)

        confirmed: list[CandidateResult] = []
        for idx, scout in enumerate(ordered[:2], start=1):
            c = copy.deepcopy(scout.candidate)
            self._emit(
                f"  CONFIRM #{idx}: ubatch={c.ubatch}, ncmoe={c.ncmoe}; "
                f"SCREEN PP/TG={scout.metrics.pp_tps or 0:.0f}/{scout.metrics.tg_tps or 0:.1f}."
            )
            full = self._run(c, quick=False, phase=f"{phase}_CONFIRM", reference=reference)
            if self._is_recommendable_full(full):
                self._mark_tight(full)
                confirmed.append(full)
                if gray_companion is None:
                    self._emit(f"  MoE joint winner: {full.candidate.short()} → {self._summary(full)}")
                    return full
                continue
            if self._is_good(full):
                self._emit(
                    f"  confirmation #{idx} completed but is {self._vram_class(full).value} at "
                    f"{full.metrics.vram_free_min_mb} MiB; trying next measured SCREEN finalist."
                )
            else:
                self._emit(f"  confirmation #{idx} failed ({full.reason}); trying next measured SCREEN finalist.")

        if confirmed:
            chosen = choose_preferred(confirmed, self.workload_profile, self.noise_policy) or confirmed[0]
            if gray_companion is not None and len(confirmed) > 1:
                # FULL has now resolved the former 5–10% uncertainty band.  If the lower raw
                # workload-latency point also has materially faster confirmed decode, do not let
                # the generic headroom tie-break silently turn that speed difference back into
                # "noise".  VRAM headroom remains a separate measured search property.
                fastest_cycle = min(
                    confirmed,
                    key=lambda r: workload_latency_seconds(r, self.workload_profile),
                )
                if any(
                    decode_relation(fastest_cycle, other, self.noise_policy) > 0
                    for other in confirmed if other is not fastest_cycle
                ):
                    chosen = fastest_cycle
            self._emit(
                f"  MoE confirmed gray-band winner: {chosen.candidate.short()} → {self._summary(chosen)}"
            )
            return chosen

        return None

    def _moe_mtp_screen_search(self, base: Candidate, reference: CandidateResult | None,
                               *, force_expand: bool = False) -> CandidateResult | None:
        """Cheap MoE speculative-decoding gate for NORMAL/QUICK.

        MTP must first prove a material decode or representative-latency gain.  A decode-only
        SCREEN earns exactly one direct FULL confirmation so it can still become FASTEST, but it
        does not unlock ubatch/p-min refinement.  Only a representative workload-latency win (or
        explicit ``--mtp on``) earns broader expansion.  Start with n-max=4 because it is the
        cheapest memory test.
        """
        if reference is None or not self._is_good(reference):
            return None
        self._mtp_speed_only = False
        floor = self._moe_screen_vram_floor_mb(vision=base.vision)
        blocks = model_main_block_count(self.model)
        start_ub = min(512, max(256, base.ubatch))
        c = copy.deepcopy(base)
        c.mtp = True; c.mtp_n_max = 4; c.mtp_p_min = 0.8
        c.ubatch = start_ub; c.batch = max(1024, start_ub)

        self._emit(
            f"\n[Phase 5 / MOE_MTP_SCREEN] MTP must earn expansion: n-max=4/ub={start_ub} QUICK first; "
            "no n-max/ubatch/p-min sweep unless it materially beats non-MTP."
        )

        screen: CandidateResult | None = None
        for attempt in range(3):
            if not self.budget_ok():
                return None
            r = self._run(c, quick=True, phase="MOE_MTP_SCREEN", reference=None)
            free = r.metrics.vram_free_min_mb or 0
            if self._is_good(r) and free >= floor:
                screen = r
                break
            if not (self._recoverable_boundary_reason(r.reason) or (self._is_good(r) and free < floor)):
                break
            if c.ncmoe is None or c.ncmoe >= blocks:
                break
            c = copy.deepcopy(c); c.ncmoe += 1
            self._emit(f"  MTP SCREEN recovery -> ncmoe={c.ncmoe} (previous free={free} MiB).")

        if screen is None:
            self._emit("  MTP rejected at SCREEN: no stable n-max=4 point.")
            return None

        dr = decode_relation(screen, reference, self.noise_policy)
        lr = latency_relation(screen, reference, self.workload_profile, self.noise_policy)
        self._emit(
            f"  MTP SCREEN: TG={screen.metrics.tg_tps or 0:.1f} vs non-MTP {self._robust_tg(reference):.1f}, "
            f"PP={screen.metrics.pp_tps or 0:.0f}; "
            f"decision={'EARNED' if (dr > 0 or lr > 0) else 'NO MATERIAL GAIN'}."
        )
        if dr <= 0 and lr <= 0:
            self._emit("  MTP early-stop: speculative branch did not earn further launches.")
            return None

        if dr > 0 and lr <= 0 and not force_expand:
            self._emit(
                "  MTP SCREEN is decode-only: it may become FASTEST, but slower representative "
                "prefill+generation latency does not earn an ubatch/p-min sweep. FULL-confirm this "
                "exact point once."
            )
            full = self._run(
                copy.deepcopy(screen.candidate), quick=False,
                phase="MOE_MTP_SPEED_CONFIRM", reference=reference,
            )
            if not self._is_recommendable_full(full):
                if self._is_good(full):
                    self._emit(
                        f"  MTP speed confirmation is runtime-successful but not recommendation-eligible: "
                        f"VRAM class={self._vram_class(full).value}, free={full.metrics.vram_free_min_mb} MiB. "
                        "Keep non-MTP."
                    )
                else:
                    self._emit(f"  MTP speed confirmation failed ({full.reason}); keep non-MTP.")
                return None
            self._mark_tight(full)
            full_dr = decode_relation(full, reference, self.noise_policy)
            full_lr = latency_relation(full, reference, self.workload_profile, self.noise_policy)
            if full_dr <= 0 and full_lr <= 0:
                self._emit(
                    "  MTP FULL no longer has a material decode or workload-latency advantage; "
                    "discard the speculative branch."
                )
                return None
            self._mtp_speed_only = full_dr > 0 and full_lr <= 0
            if self._mtp_speed_only:
                self._emit(
                    "  MTP retained as a confirmed speed-only candidate for FASTEST. OPTIMAL stays "
                    "eligible to prefer non-MTP, and no p-min/ubatch expansion will run."
                )
            else:
                self._emit(
                    "  Direct MTP FULL also improves representative workload latency; retain it as "
                    "a normal final candidate."
                )
            return full

        probes = [screen]
        # Grow MTP ubatch through a midpoint. Jumping 512 -> 2048 and then trying ncmoe+1/+2
        # wasted several cold launches in the 256K Qwen log: speculative workspace, not expert
        # residency, dominated the failed allocation. A failed 1024 point closes the larger-ubatch
        # family; each size gets at most one expert-placement recovery.
        largest_ub = min(2048, max(512, base.ubatch))
        refine_ubs = [ub for ub in (1024, largest_ub)
                      if screen.candidate.ubatch < ub <= largest_ub]
        refine_ubs = list(dict.fromkeys(refine_ubs))
        for target_ub in refine_ubs:
            if not self.budget_ok():
                break
            cc = copy.deepcopy(screen.candidate)
            cc.ubatch = target_ub; cc.batch = max(2048, target_ub)
            previous_free: int | None = None
            stable: CandidateResult | None = None
            for attempt in range(2):
                rr = self._run(cc, quick=True, phase="MOE_MTP_REFINE_SCREEN", reference=None)
                free = int(rr.metrics.vram_free_min_mb or 0)
                if self._is_good(rr) and free >= floor:
                    stable = rr
                    break
                recoverable = self._recoverable_boundary_reason(rr.reason) or (self._is_good(rr) and free < floor)
                if not recoverable or cc.ncmoe is None or cc.ncmoe >= blocks or attempt >= 1:
                    break
                if previous_free is not None and free - previous_free < 64 and free < floor:
                    self._emit(
                        "  MTP ubatch recovery produced <64 MiB from another CPU expert layer; "
                        "workspace dominates, so stop this ubatch family."
                    )
                    break
                previous_free = free
                cc = copy.deepcopy(cc); cc.ncmoe += 1
                self._emit(
                    f"  MTP ubatch={target_ub} gets one placement recovery -> ncmoe={cc.ncmoe} "
                    f"(previous free={free} MiB)."
                )
            if stable is None:
                self._emit(
                    f"  MTP ubatch={target_ub} did not reach the {floor} MiB SCREEN floor; "
                    "do not try a larger ubatch."
                )
                break
            probes.append(stable)

        best = choose_preferred(probes, self.workload_profile, self.noise_policy) or probes[0]
        if decode_relation(best, reference, self.noise_policy) <= 0 \
                and latency_relation(best, reference, self.workload_profile, self.noise_policy) <= 0:
            self._emit("  MTP refinement lost its material advantage; keep non-MTP.")
            return None

        self._emit(
            f"  MTP earned one FULL confirmation: n-max={best.candidate.mtp_n_max}, "
            f"ub={best.candidate.ubatch}, ncmoe={best.candidate.ncmoe}."
        )
        full = self._run(copy.deepcopy(best.candidate), quick=False, phase="MOE_MTP_CONFIRM", reference=reference)
        if not self._is_recommendable_full(full):
            if self._is_good(full):
                self._emit(
                    f"  MTP confirmation is runtime-successful but not recommendation-eligible: "
                    f"VRAM class={self._vram_class(full).value}, free={full.metrics.vram_free_min_mb} MiB. "
                    "Keep non-MTP."
                )
            else:
                self._emit(f"  MTP confirmation failed ({full.reason}); keep non-MTP.")
            return None
        self._mark_tight(full)
        full_dr = decode_relation(full, reference, self.noise_policy)
        full_lr = latency_relation(full, reference, self.workload_profile, self.noise_policy)
        if full_dr <= 0 and full_lr <= 0:
            self._emit("  MTP FULL no longer clears the material-gain threshold; keep non-MTP.")
            return None
        self._mtp_speed_only = full_dr > 0 and full_lr <= 0 and not force_expand
        if self._mtp_speed_only:
            self._emit(
                "  MTP refinement's FULL result retained only a decode advantage. Keep it for "
                "FASTEST, but close p-min and any further automatic expansion."
            )
        return full

    def _dense_mtp_full_gpu_search(self, base: Candidate,
                                    reference: CandidateResult | None = None,
                                    force_probe: bool = False) -> CandidateResult | None:
        """Noise-aware Dense MTP search with strict run budgeting.

        MTP is only interesting when it produces a *material* generation/latency win. A 44 -> 45 t/s
        probe is treated as a tie and is not worth hundreds of MiB of speculative state or another
        long FULL run. Target-model placement always remains ngl=all.
        """
        self._last_dense_mtp_outcome = "SEARCHING"
        self._emit(
            "\n[Phase 5 / DENSE_MTP] Noise-aware full-GPU MTP search: require a material decode/latency "
            "win before spending a FULL confirmation."
        )

        full_gpu_plain = [
            r for r in self.results
            if (not r.candidate.mtp and r.candidate.ngl == "all"
                and self._is_recommendable_full(r) and r.candidate.ctx == base.ctx)
        ]
        if base.ngl != "all":
            if not full_gpu_plain:
                self._emit("  Dense MTP skipped: non-MTP ngl=all was not proven stable.")
                self._last_dense_mtp_outcome = "NO_FULL_GPU_REFERENCE"
                return None
            reference = choose_preferred(full_gpu_plain, self.workload_profile, self.noise_policy)
        elif reference is None and full_gpu_plain:
            reference = choose_preferred(full_gpu_plain, self.workload_profile, self.noise_policy)
        if reference is None:
            self._emit("  Dense MTP skipped: no comparable non-MTP FULL reference.")
            self._last_dense_mtp_outcome = "NO_REFERENCE"
            return None
        if self._is_tight(reference) and not force_probe:
            self._emit(
                "  Dense MTP auto-search skipped: the non-MTP reference is TIGHT, so speculative "
                "state cannot borrow from its protected reserve. Use explicit --mtp on only for a diagnostic probe."
            )
            self._last_dense_mtp_outcome = "MEMORY_TIGHT_REFERENCE"
            return None

        # Static GGUF geometry + the already measured full-GPU correction can often prove that
        # MTP has no safe memory window before launching another server. Keep a conservative
        # uncertainty band; explicit --mtp on still forces a real probe.
        free512: int | None = None
        free256: int | None = None
        if not force_probe and self.search_mode != "deep":
            p512 = copy.deepcopy(base); p512.ngl = "all"; p512.mtp = True; p512.mtp_n_max = 4; p512.mtp_p_min = 0.8; p512.batch = 2048; p512.ubatch = 512
            p256 = copy.deepcopy(base); p256.ngl = "all"; p256.mtp = True; p256.mtp_n_max = 4; p256.mtp_p_min = 0.8; p256.batch = 1024; p256.ubatch = 256
            free512 = self._predicted_free_for(p512)
            free256 = self._predicted_free_for(p256)
            uncertainty = 160
            if free512 is not None and free256 is not None:
                self._emit(
                    f"  MTP static feasibility: predicted free≈{free512} MiB at ub512 / {free256} MiB at ub256 "
                    f"(uncertainty band ±{uncertainty} MiB)."
                )
                if free256 < self.absolute_vram_floor_mb - uncertainty:
                    self._emit(
                        "  MTP skipped before launch: even n-max=4/ub256 is predicted well below the hard VRAM floor."
                    )
                    self._last_dense_mtp_outcome = "MEMORY_STATIC"
                    return None

        nmax_values = [4] if self.search_mode == "quick" else ([4, 8] if self.search_mode == "normal" else [4, 8, 12, 16])
        probes: list[CandidateResult] = []
        preferred_ub = 512
        if (not force_probe and self.search_mode != "deep" and free512 is not None
                and free512 < self.absolute_vram_floor_mb + 96 and free256 is not None):
            preferred_ub = 256
            self._emit("  MTP starts directly at ubatch=256; static model says ubatch=512 is too close to the floor.")

        def probe(nmax: int, ub: int) -> CandidateResult:
            c = copy.deepcopy(base)
            c.ngl = "all"; c.ncmoe = None; c.mtp = True
            c.mtp_n_max = nmax; c.mtp_p_min = 0.8
            c.ubatch = ub; c.batch = 1024 if ub <= 256 else max(2048, ub)
            return self._run(c, quick=True, phase=f"DENSE_MTP_PROBE_N{nmax}_UB{ub}", reference=None)

        for nmax in nmax_values:
            if not self.budget_ok():
                break
            self._emit(f"  probing MTP n-max={nmax}, ubatch={preferred_ub}.")
            r = probe(nmax, preferred_ub)
            if not self._is_good(r) and self._recoverable_boundary_reason(r.reason) and preferred_ub > 256:
                self._emit(
                    f"  n-max={nmax}/ubatch={preferred_ub} is memory-bound; one retry at ubatch=256."
                )
                r = probe(nmax, 256)
                if self._is_good(r):
                    preferred_ub = 256
            if not self._is_good(r):
                if self._recoverable_boundary_reason(r.reason):
                    self._emit(
                        f"  n-max={nmax} cannot fit full-GPU with the minimum useful ubatch; stop larger draft depths."
                    )
                    self._last_dense_mtp_outcome = "MEMORY_PROBE"
                else:
                    self._emit(f"  MTP probe stopped by {r.reason}.")
                    self._last_dense_mtp_outcome = (
                        "ENVIRONMENT" if self._environmental_final_failure(r) else "PROBE_FAILED"
                    )
                break

            probes.append(r)
            dr = decode_relation(r, reference, self.noise_policy)
            lr = latency_relation(r, reference, self.workload_profile, self.noise_policy)
            self._emit(
                f"  MTP n-max={nmax}/ub={r.candidate.ubatch}: TG={self._robust_tg(r):.1f} vs "
                f"{self._robust_tg(reference):.1f} t/s; decode relation="
                f"{'MATERIAL_GAIN' if dr>0 else ('MATERIAL_LOSS' if dr<0 else 'NOISE_TIE')}."
            )

            if (self.search_mode != "deep" and nmax == 4
                    and (r.metrics.vram_free_min_mb or 0) <= self.absolute_vram_floor_mb + 96):
                self._emit(
                    f"  MTP memory stop: n-max=4 already leaves only {r.metrics.vram_free_min_mb} MiB; "
                    "larger draft depths cannot earn another launch in NORMAL mode."
                )
                break

            # If even n=4 is not materially better, a more memory-hungry draft depth does not earn
            # another probe in NORMAL mode. DEEP is explicitly allowed to explore beyond this.
            if self.search_mode != "deep" and dr <= 0 and lr <= 0:
                self._emit(
                    "  MTP early-stop: speculative decoding is not materially faster than non-MTP; "
                    "do not chase 1-2 t/s noise with larger draft depths."
                )
                break

        if not probes:
            self._emit("  Dense MTP has no feasible full-GPU probe; keep non-MTP.")
            if self._last_dense_mtp_outcome == "SEARCHING":
                self._last_dense_mtp_outcome = "MEMORY_PROBE"
            return None

        # Only probes with a material decode OR representative-latency win deserve an expensive FULL.
        competitive = [
            r for r in probes
            if decode_relation(r, reference, self.noise_policy) > 0
            or latency_relation(r, reference, self.workload_profile, self.noise_policy) > 0
        ]
        if not competitive and self.search_mode != "deep":
            self._emit("  Dense MTP rejected: every feasible probe is inside the noise zone or slower.")
            self._last_dense_mtp_outcome = "NO_MATERIAL_GAIN"
            return None
        pool = competitive or probes
        best_probe = choose_preferred(pool, self.workload_profile, self.noise_policy) or pool[0]

        c = copy.deepcopy(best_probe.candidate)
        self._emit(
            f"  FULL-confirming one materially promising MTP candidate: n-max={c.mtp_n_max}, ubatch={c.ubatch}."
        )
        full = self._run(c, quick=False, phase="DENSE_MTP_CONFIRM", reference=reference)
        if not self._is_recommendable_full(full):
            if self._is_good(full):
                self._emit(
                    f"  Dense MTP confirmation is runtime-successful but not recommendation-eligible: "
                    f"VRAM class={self._vram_class(full).value}, free={full.metrics.vram_free_min_mb} MiB. "
                    "Keep non-MTP."
                )
            else:
                self._emit(f"  Dense MTP confirmation failed ({full.reason}); keep non-MTP.")
            self._last_dense_mtp_outcome = (
                "MEMORY_CONFIRM" if self._is_good(full) or self._recoverable_boundary_reason(full.reason)
                else "CONFIRM_FAILED"
            )
            return None
        self._mark_tight(full)

        dr = decode_relation(full, reference, self.noise_policy)
        lr = latency_relation(full, reference, self.workload_profile, self.noise_policy)
        if self.search_mode != "deep" and dr <= 0 and lr <= 0:
            self._emit(
                "  Dense MTP rejected after FULL: any apparent gain is inside the configured noise zone. "
                "Skip p-min and final speculative validation."
            )
            self._last_dense_mtp_outcome = "NO_MATERIAL_GAIN"
            return None

        self._emit(
            f"  Dense MTP candidate materially improves the workload: {full.candidate.short()} → "
            f"{self._summary(full)} | {self.workload_profile} cycle≈"
            f"{workload_latency_seconds(full, self.workload_profile):.2f}s"
        )
        self._last_dense_mtp_outcome = "SUCCESS"
        return full

    def _dense_mtp_kv_rescue(
        self,
        base: Candidate,
        primary_reference: CandidateResult,
        incumbent_mtp: CandidateResult | None,
        *,
        force_probe: bool = False,
    ) -> CandidateResult | None:
        """Re-open one lower-KV point when MTP memory changes the Dense optimum.

        This is deliberately a bounded rescue, not a second context×KV×MTP Cartesian
        search.  NORMAL chooses the highest lower-precision KV family whose static n=4
        estimate has a plausible operating window, FULL-confirms its plain control, and
        then lets the ordinary MTP funnel spend its usual probes.  A lower-KV result is
        promoted only when it materially beats the original higher-KV control/incumbent.
        """
        if base.ngl != "all" or self.search_mode == "quick":
            return None
        ladder = kv_degradation_ladder(base.kv_k, base.kv_v)
        if not ladder:
            return None

        uncertainty = 160
        selected: tuple[str, str, str] | None = None
        predicted_selected: int | None = None
        for kk, vv, note in ladder:
            # Mixed Q8/Q4 kernels are not a free midpoint.  The supplied long-context
            # run showed a severe prefill cliff even though the point had ample VRAM.
            # NORMAL therefore compares primary tiers only; DEEP may still diagnose
            # mixed caches, and an explicitly selected mixed base is preserved.
            if (self.search_mode != "deep" and kv_precision(kk, vv).tier == "MIXED"
                    and kv_precision(base.kv_k, base.kv_v).tier != "MIXED"):
                continue
            probe = copy.deepcopy(base)
            probe.kv_k = kk; probe.kv_v = vv
            probe.mtp = True; probe.mtp_n_max = 4; probe.mtp_p_min = 0.8
            probe.batch = 1024; probe.ubatch = 256
            predicted = self._predicted_free_for(probe)
            # Pick the first/highest-precision family with a plausible operational
            # window.  If none reaches it, the most memory-saving family is the only
            # useful diagnostic in DEEP/forced mode.
            selected = (kk, vv, note)
            predicted_selected = predicted
            floor = self._vram_thresholds(probe).operational_floor_mb
            if predicted is None or predicted >= floor - uncertainty:
                break
        if selected is None:
            return None
        if (predicted_selected is not None
                and predicted_selected < self.absolute_vram_floor_mb - uncertainty
                and not force_probe and self.search_mode != "deep"):
            self._emit(
                "  Dense MTP KV rescue skipped: even the lowest KV family remains well below "
                "the hard floor after the measured correction."
            )
            return None

        kk, vv, note = selected
        lower = copy.deepcopy(base)
        lower.kv_k = kk; lower.kv_v = vv
        lower.mtp = False; lower.mtp_n_max = 4; lower.mtp_p_min = 0.8
        self._emit(
            "\n[Phase 5b / DENSE_MTP_KV_RESCUE] MTP changed the memory frontier; reopen one "
            f"bounded KV point: {base.kv_k}/{base.kv_v}→{kk}/{vv}."
        )
        self._emit(f"  trade-off disclosure: {note}. Model weights are unchanged.")

        plain = self._best_exact_result(lower, full_only=True)
        if plain is None:
            plain = self._run(
                lower, quick=False, phase="DENSE_MTP_KV_RESCUE_PLAIN_CONFIRM",
                reference=primary_reference,
            )
        if not self._is_recommendable_full(plain):
            self._emit("  lower-KV plain control did not earn recommendation status; close KV rescue.")
            return None

        rescued = self._dense_mtp_full_gpu_search(
            copy.deepcopy(plain.candidate), reference=plain, force_probe=force_probe,
        )
        if not self._is_recommendable_full(rescued):
            return None
        comparator = incumbent_mtp or primary_reference
        if (
            decode_relation(rescued, comparator, self.noise_policy) <= 0
            and latency_relation(rescued, comparator, self.workload_profile, self.noise_policy) <= 0
        ):
            self._emit(
                "  lower-KV MTP fits, but does not materially beat the higher-KV incumbent; "
                "keep the higher-precision command."
            )
            return None
        self._emit(
            f"  lower-KV MTP rescue earned promotion: {rescued.candidate.short()} → {self._summary(rescued)}."
        )
        return rescued

    def _moe_screen_vram_floor_mb(self, *, vision: bool = False) -> int:
        """Working VRAM floor for MoE SCREEN candidates in NORMAL/QUICK.

        MoE can trade one routed-expert layer for hundreds of MiB with a much smaller penalty
        than Dense whole-layer offload, so its useful operating floor can be closer to the hard
        floor.  Still, treating 300-380 MiB as a normal winner causes later staged prompts to
        cross the hard floor.  Keep a modest architecture-specific uncertainty guard.
        """
        probe = Candidate(ctx=1, ncmoe=0, vision=vision)
        return self._vram_thresholds(probe).tight_floor_mb

    def _operational_vram_floor_mb(self, *, vision: bool = False) -> int:
        """Return the NORMAL/DEEP operating floor used for *recommended* candidates.

        The absolute floor answers only "can this launch at all?". A candidate sitting 10-50 MiB
        above that floor is a technical ceiling, not a stable recommendation: small allocator/workload
        differences can push it below the floor during FULL validation. Keep a separate operating
        guard so OPTIMAL/MAX_KV_PRECISION tuning never converts a fragile full-GPU scout into Dense CPU
        offload merely to preserve an over-aggressive context boundary.
        """
        probe = Candidate(ctx=1, ngl="all", vision=vision)
        return self._vram_thresholds(probe).operational_floor_mb

    def _dense_boundary_context(self, *, kv_k: str, kv_v: str, target_ctx: int, cores: int,
                                vision: bool, mmproj: str | None,
                                floor_mb: int | None = None) -> tuple[int | None, int | None]:
        """Estimate the highest *full-GPU* context worth probing for one KV family.

        Fixed 16/32/64/128K ladders can miss the real knee (for example 48K Q8 or 88K Q4).
        After one real full-GPU scout calibrates the backend bias, binary-search the static KV
        slope on a 4K grid and target a small guard above the hard floor. The returned point is
        still only a scout seed; real llama-server VRAM remains authoritative.
        """
        step = 4096
        upper = max(step, (int(target_ctx) // step) * step)
        guard = int(floor_mb if floor_mb is not None else self._operational_vram_floor_mb(vision=vision))
        lo_i, hi_i = 1, max(1, upper // step)
        best_ctx: int | None = None
        best_free: int | None = None
        while lo_i <= hi_i:
            mid_i = (lo_i + hi_i) // 2
            ctx = mid_i * step
            c = Candidate(
                ctx=ctx, ngl="all", batch=512, ubatch=256,
                threads=max(1, cores), threads_batch=max(1, cores),
                kv_k=kv_k, kv_v=kv_v, vision=vision, mmproj=mmproj,
                extra_args=list(self.base_extra_args),
            )
            free = self._predicted_free_for(c)
            if free is None:
                return None, None
            if free >= guard:
                best_ctx, best_free = ctx, free
                lo_i = mid_i + 1
            else:
                hi_i = mid_i - 1
        return best_ctx, best_free

    def _dense_boundary_option(self, *, context: int, kv_k: str, kv_v: str,
                               target_ctx: int, preferred_k: str, preferred_v: str,
                               predicted_free_mb: int | None, vision: bool,
                               mmproj: str | None, rank: int) -> SolutionOption:
        degradation: list[DegradationKind] = []
        notes: list[str] = []
        if context < target_ctx:
            degradation.append(DegradationKind.CAPABILITY)
            notes.append(
                f"Adaptive full-GPU boundary reduces context {target_ctx}→{context}; "
                "the boundary is measured rather than chosen from a fixed context ladder."
            )
        if (kv_k, kv_v) != (preferred_k, preferred_v):
            degradation.append(DegradationKind.QUALITY_RISK)
            notes.append(
                f"KV cache uses {kv_k}/{kv_v} instead of preferred {preferred_k}/{preferred_v}; "
                "this is an explicit KV-cache precision trade-off; model weights are unchanged."
            )
        comfortable = max(self.vram_margin_mb * 2, int(self.hardware.vram_total_mb * 0.25))
        if predicted_free_mb is None:
            rc = ResourceClass.UNKNOWN
        elif predicted_free_mb < self.absolute_vram_floor_mb:
            rc = ResourceClass.INFEASIBLE
        elif predicted_free_mb >= comfortable:
            rc = ResourceClass.COMFORTABLE
        else:
            rc = ResourceClass.CONSTRAINED
        label = f"{kv_k}_{kv_v}".upper().replace("_0", "").replace("/", "_")
        return SolutionOption(
            name=f"ADAPTIVE_MAX_FULL_GPU_{label}_{context}",
            context=context, kv_k=kv_k, kv_v=kv_v,
            strategy="full-gpu-adaptive-boundary",
            predicted_free_mb=predicted_free_mb, predicted_placement="all",
            resource_class=rc, degradation=degradation, degradation_notes=notes,
            recommended_rank=rank, exact_target=(context == target_ctx),
            vision_required=vision, mmproj=mmproj,
        )

    def _dense_oversized_key(self, candidate: Candidate) -> tuple[int, str, str, int] | None:
        if candidate.ngl == "all":
            return None
        return (int(candidate.ctx), candidate.kv_k, candidate.kv_v, int(candidate.ngl))

    def _dense_numeric_layer_mb(self, ngl: int) -> int:
        """Approximate one target-layer VRAM step near a numeric -ngl boundary."""
        main = model_main_block_count(self.model)
        ids = sorted(self.model.block_tensor_bytes)[:main]
        if not ids:
            return 192
        n = min(main, max(1, int(ngl)))
        start = max(0, len(ids) - n)
        sample_ids = ids[start:min(len(ids), start + 3)]
        vals = [int(self.model.block_tensor_bytes.get(i, 0)) for i in sample_ids if self.model.block_tensor_bytes.get(i, 0)]
        if not vals:
            vals = [int(self.model.block_tensor_bytes.get(i, 0)) for i in ids[-min(3, len(ids)):]]
        if not vals:
            return 192
        return max(64, int(sum(vals) / len(vals) / (1024 * 1024)))

    def _dense_is_oversized(self, *, target_ctx: int, cores: int, vision: bool, mmproj: str | None) -> bool:
        """True when even a tiny Q4/Q4 slot cannot keep the Dense target fully GPU-resident.

        This is intentionally stricter than "the requested context does not fit".  The class is for
        models where reducing context can reclaim KV memory and therefore *more GPU layers*, but can
        never reach ngl=all because model weights + minimal runtime already exceed the device budget.
        """
        tiny = Candidate(
            ctx=min(max(4096, int(target_ctx // 32) or 4096), 8192), ngl="all",
            batch=512, ubatch=256, threads=max(1, cores), threads_batch=max(1, cores),
            kv_k="q4_0", kv_v="q4_0", vision=vision, mmproj=mmproj,
            extra_args=list(self.base_extra_args),
        )
        free = estimate_candidate_free_mb(self.model, self.hardware, self.baseline_vram_mb, tiny)
        return free is not None and free < self.absolute_vram_floor_mb

    def _dense_oversized_option(self, *, context: int, kv_k: str, kv_v: str,
                                target_ctx: int, preferred_k: str, preferred_v: str,
                                cores: int, vision: bool, mmproj: str | None, rank: int) -> SolutionOption:
        seed = Candidate(
            ctx=context, ngl="all", batch=512, ubatch=256,
            threads=max(1, cores), threads_batch=max(1, cores),
            kv_k=kv_k, kv_v=kv_v, vision=vision, mmproj=mmproj,
            extra_args=list(self.base_extra_args),
        )
        est = estimate_static_memory(self.model, self.hardware, self.baseline_vram_mb, self.vram_margin_mb, seed)
        placement = est.predicted_dense_ngl if isinstance(est.predicted_dense_ngl, int) else max(0, model_main_block_count(self.model) - 1)
        degradation: list[DegradationKind] = [DegradationKind.PERFORMANCE]
        notes = [
            "Model weights cannot become fully GPU-resident on this device; numeric Dense layer placement is unavoidable."
        ]
        if context < target_ctx:
            degradation.append(DegradationKind.CAPABILITY)
            notes.append(
                f"Context reduced {target_ctx}→{context} so KV memory can be traded for more GPU-resident target layers."
            )
        if (kv_k, kv_v) != (preferred_k, preferred_v):
            degradation.append(DegradationKind.QUALITY_RISK)
            notes.append(
                f"KV cache uses {kv_k}/{kv_v} instead of preferred {preferred_k}/{preferred_v}; "
                "the KV-cache precision trade-off is explicit and model weights are unchanged."
            )
        return SolutionOption(
            name=f"OVERSIZED_DENSE_CTX_{context}_{kv_k}_{kv_v}".upper().replace("_0", ""),
            context=context, kv_k=kv_k, kv_v=kv_v,
            strategy="dense-cpu-offload-oversized-knee",
            predicted_free_mb=est.predicted_free_mb, predicted_placement=placement,
            resource_class=ResourceClass.CONSTRAINED,
            degradation=degradation, degradation_notes=notes, recommended_rank=rank,
            exact_target=(context == target_ctx), vision_required=vision, mmproj=mmproj,
        )

    def _screen_dense_oversized_option(self, option: SolutionOption, *, cores: int,
                                       stable_floor: int, placement_bias: dict[tuple[str, str], int]) -> CandidateResult | None:
        """Find a stable numeric-ngl point with tiny recon samples, never a long FULL walk."""
        seed = option.to_candidate(cores=cores, extra_args=self.base_extra_args)
        if seed.ngl == "all":
            return None
        pair = (seed.kv_k, seed.kv_v)
        predicted_n = int(seed.ngl)
        n = max(0, min(model_main_block_count(self.model), predicted_n + placement_bias.get(pair, 0)))
        technical_floor = self.absolute_vram_floor_mb + 64
        best_stable: CandidateResult | None = None
        seen: set[int] = set()
        attempts = 0

        while attempts < (4 if self.search_mode == "deep" else 3) and self.budget_ok():
            n = max(0, min(model_main_block_count(self.model), n))
            if n in seen:
                break
            seen.add(n)
            c = copy.deepcopy(seed); c.ngl = n; c.batch = 512; c.ubatch = 256
            self._emit(
                f"  oversized SCREEN: ctx={c.ctx} KV={c.kv_k}/{c.kv_v} ngl={n} "
                f"(predicted seed={predicted_n}, stable floor={stable_floor} MiB)"
            )
            r = self._run(c, quick=True, phase="DENSE_OVERSIZED_SCREEN", recon=True)
            free = int(r.metrics.vram_free_min_mb or 0)
            if self._is_good(r) and r.metrics.pp_tps and r.metrics.tg_tps:
                if free >= stable_floor:
                    best_stable = r
                    placement_bias[pair] = int(n - predicted_n)
                    # If the stable point has enough room for another target layer, spend one tiny
                    # scout to see whether decode improves. Never cross below the technical floor.
                    layer_mb = self._dense_numeric_layer_mb(n + 1)
                    if (n < model_main_block_count(self.model) and attempts == 0
                            and free >= stable_floor + layer_mb + 64):
                        n += 1
                        attempts += 1
                        continue
                    return best_stable
                if free >= technical_floor:
                    self._emit(
                        f"  oversized technical point: ngl={n} is runnable with {free} MiB, but below "
                        f"the {stable_floor} MiB recommendation floor; back off before ranking OPTIMAL."
                    )
                # Runnable but too close to the floor: back off enough layers in one calibrated jump.
                layer_mb = self._dense_numeric_layer_mb(n)
                need = max(1, stable_floor - free + 64)
                n -= max(1, int(ceil(need / max(1, layer_mb))))
                attempts += 1
                continue

            if not self._recoverable_boundary_reason(r.reason):
                self._emit(f"  oversized SCREEN stopped by non-memory failure: {r.reason}")
                return None
            layer_mb = self._dense_numeric_layer_mb(n)
            if free > 0:
                need = max(layer_mb, stable_floor - free + 64)
                step = max(1, int(ceil(need / max(1, layer_mb))))
            else:
                step = 2
            n -= step
            attempts += 1

        return best_stable

    def _choose_dense_oversized_pair(
        self, pairs: list[tuple[SolutionOption, CandidateResult]], *, target_ctx: int,
        preferred_k: str, preferred_v: str,
    ) -> tuple[
        tuple[SolutionOption, CandidateResult],
        tuple[SolutionOption, CandidateResult] | None,
        tuple[SolutionOption, CandidateResult],
        tuple[SolutionOption, CandidateResult],
    ]:
        """Select one semantic frontier point; safe to call repeatedly for fallback ordering."""
        max_tg = max(float(r.metrics.tg_tps or 0.0) for _, r in pairs)
        max_pp = max(float(r.metrics.pp_tps or 0.0) for _, r in pairs)
        competitive = [
            (o, r) for o, r in pairs
            if float(r.metrics.tg_tps or 0.0) >= max_tg * .72
            and float(r.metrics.pp_tps or 0.0) >= max_pp * .50
        ] or pairs
        preferred = [(o, r) for o, r in competitive if (o.kv_k, o.kv_v) == (preferred_k, preferred_v)]
        quality_pair = max(
            preferred, key=lambda x: (x[0].context, x[1].metrics.tg_tps or 0.0), default=None
        )
        max_context_pair = max(
            competitive, key=lambda x: (x[0].context, x[1].metrics.tg_tps or 0.0)
        )
        fastest_pair = max(
            competitive,
            key=lambda x: (float(x[1].metrics.tg_tps or 0.0), float(x[1].metrics.pp_tps or 0.0)),
        )

        if self.selection_priority == "quality" and quality_pair is not None:
            winner = quality_pair
        elif self.selection_priority == "context":
            context_ok = [
                (o, r) for o, r in competitive
                if float(r.metrics.tg_tps or 0.0) >= max_tg * .55
            ]
            winner = max(context_ok or competitive, key=lambda x: x[0].context)
        elif self.selection_priority == "speed":
            winner = fastest_pair
        else:
            balanced_pool = [
                (o, r) for o, r in competitive
                if o.context >= int(target_ctx * .50)
                and float(r.metrics.tg_tps or 0.0) >= max_tg * .80
            ]
            pool = balanced_pool or competitive
            winner_opt, winner_res = pool[0]
            for o, r in pool[1:]:
                if self._performance_equivalent(r, winner_res):
                    # Tied speed: preserve the higher KV-cache precision tier unless the lower
                    # tier buys at least 1.5x context. This distinguishes FP16, Q8 and Q4.
                    wp = kv_precision_key(winner_opt.kv_k, winner_opt.kv_v)
                    rp = kv_precision_key(o.kv_k, o.kv_v)
                    if wp != rp:
                        higher = (winner_opt, winner_res) if wp > rp else (o, r)
                        lower = (winner_opt, winner_res) if wp < rp else (o, r)
                        winner_opt, winner_res = (
                            lower if lower[0].context >= int(higher[0].context * 1.50) else higher
                        )
                    elif o.context > winner_opt.context:
                        winner_opt, winner_res = o, r
                elif choose_preferred(
                    [winner_res, r], self.workload_profile, self.noise_policy
                ) is r:
                    winner_opt, winner_res = o, r
            winner = (winner_opt, winner_res)
        return winner, quality_pair, max_context_pair, fastest_pair

    def _recon_dense_oversized_options(self, options: list[SolutionOption], cores: int) -> list[SolutionOption]:
        """Map context × KV × numeric-ngl before tuning batches for a Dense model larger than VRAM."""
        eligible = [o for o in options if o.resource_class != ResourceClass.INFEASIBLE]
        if not eligible:
            return options
        declared_exact = next((o for o in options if o.name == "EXACT_TARGET"), None)
        exact = next((o for o in eligible if o.name == "EXACT_TARGET"), None)
        target = declared_exact or exact
        target_ctx = target.context if target is not None else max(o.context for o in eligible)
        preferred_k = target.kv_k if target is not None else "f16"
        preferred_v = target.kv_v if target is not None else "f16"
        vision = bool(target.vision_required) if target is not None else any(o.vision_required for o in eligible)
        mmproj = target.mmproj if target is not None else next((o.mmproj for o in eligible if o.mmproj), None)
        stable_floor = self._operational_vram_floor_mb(vision=vision)
        self._dense_oversized_active = True
        self.phase = "DENSE_OVERSIZED_FRONTIER"
        self._emit("\n[Phase 0] Oversized Dense frontier: context × KV × GPU placement")
        self._emit(
            "  Full GPU is structurally unavailable even at tiny context. NORMAL therefore spends its "
            "budget first on context/KV/NGL scouts: reducing context is valuable because KV memory can "
            "buy back target layers and improve decode. Batch/ubatch comes only after a placement winner exists."
        )

        # Keep the previously useful 75% and 25% oversized-Dense knees. QUICK maps two context
        # regimes; NORMAL/DEEP map target/75%/50%/25% plus the planner's minimum useful context
        # when that is lower. These are tiny placement scouts and remain bounded, rather than
        # opening a full Cartesian sweep.
        available_contexts = sorted({o.context for o in eligible if o.context <= target_ctx}, reverse=True)
        if not available_contexts:
            available_contexts = [target_ctx, max(4096, target_ctx // 2)]
        wanted: list[int] = []
        context_fractions = {
            "quick": (1.0, .50),
            "normal": (1.0, .75, .50, .25),
            "deep": (1.0, .75, .50, .25),
        }.get(self.search_mode, (1.0, .75, .50))
        for fraction in context_fractions:
            goal = int(target_ctx * fraction)
            chosen = min(available_contexts, key=lambda c: abs(c - goal))
            if chosen not in wanted:
                wanted.append(chosen)
        if self.search_mode != "quick" and available_contexts:
            floor_ctx = min(available_contexts)
            if floor_ctx not in wanted:
                wanted.append(floor_ctx)
        primary_pairs: list[tuple[str, str]] = []
        for pair in [(preferred_k, preferred_v), ("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")]:
            if pair not in primary_pairs and kv_precision_key(*pair) <= kv_precision_key(preferred_k, preferred_v):
                primary_pairs.append(pair)
        specs: list[SolutionOption] = []
        rank = 1
        for kk, vv in primary_pairs:
            for ctx in wanted:
                specs.append(self._dense_oversized_option(
                    context=ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                    preferred_k=preferred_k, preferred_v=preferred_v, cores=cores,
                    vision=vision, mmproj=mmproj, rank=rank,
                )); rank += 1
        if self.search_mode == "deep":
            mid = min(wanted or [target_ctx], key=lambda c: abs(c - target_ctx * .5))
            specs.append(self._dense_oversized_option(
                context=mid, kv_k="q8_0", kv_v="q4_0", target_ctx=target_ctx,
                preferred_k=preferred_k, preferred_v=preferred_v, cores=cores,
                vision=vision, mmproj=mmproj, rank=rank,
            ))

        # Deduplicate exact (ctx, KV), and keep explicit per-mode scout budgets.
        dedup: list[SolutionOption] = []
        seen_specs: set[tuple[int, str, str]] = set()
        for o in specs:
            k = (o.context, o.kv_k, o.kv_v)
            if k not in seen_specs:
                seen_specs.add(k); dedup.append(o)
        mode_cap = {"quick": 6, "normal": 15, "deep": 16}.get(self.search_mode, 15)
        specs = dedup[:mode_cap]

        placement_bias: dict[tuple[str, str], int] = {}
        success: list[tuple[SolutionOption, CandidateResult]] = []
        for o in specs:
            if not self.budget_ok():
                break
            r = self._screen_dense_oversized_option(o, cores=cores, stable_floor=stable_floor,
                                                    placement_bias=placement_bias)
            if r is None or not self._is_good(r) or not r.metrics.pp_tps or not r.metrics.tg_tps:
                self._emit(f"  oversized result: ctx={o.context} KV={o.kv_k}/{o.kv_v} -> no stable point")
                continue
            o.predicted_placement = int(r.candidate.ngl)
            o.predicted_free_mb = r.metrics.vram_free_min_mb
            o.strategy = "dense-cpu-offload-oversized-knee"
            self._dense_oversized_evidence[self._dense_oversized_key(r.candidate)] = r
            success.append((o, r))
            self._emit(
                f"  oversized result: ctx={o.context} KV={o.kv_k}/{o.kv_v} ngl={r.candidate.ngl} -> "
                f"PP={r.metrics.pp_tps:.1f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB"
            )

        if not success:
            self._emit("  oversized Dense frontier found no stable numeric placement; fall back to planner order.")
            return options

        winner, quality_pair, max_context_pair, fastest_pair = self._choose_dense_oversized_pair(
            success, target_ctx=target_ctx, preferred_k=preferred_k, preferred_v=preferred_v
        )
        winner_opt, winner_res = winner

        self._emit("  measured oversized Dense frontier (SCOUT; one winner will be FULL-confirmed):")
        if quality_pair:
            o, r = quality_pair
            self._emit(f"    MAX_KV_PRECISION: ctx={o.context} KV={o.kv_k}/{o.kv_v} ngl={r.candidate.ngl} | TG={r.metrics.tg_tps:.1f} | PP={r.metrics.pp_tps:.0f}")
        o, r = max_context_pair
        self._emit(f"    MAX_CONTEXT: ctx={o.context} KV={o.kv_k}/{o.kv_v} ngl={r.candidate.ngl} | TG={r.metrics.tg_tps:.1f} | PP={r.metrics.pp_tps:.0f}")
        o, r = fastest_pair
        self._emit(f"    FASTEST: ctx={o.context} KV={o.kv_k}/{o.kv_v} ngl={r.candidate.ngl} | TG={r.metrics.tg_tps:.1f} | PP={r.metrics.pp_tps:.0f}")
        self._emit(
            f"    OPTIMAL: ctx={winner_opt.context} KV={winner_opt.kv_k}/{winner_opt.kv_v} "
            f"ngl={winner_res.candidate.ngl} | TG={winner_res.metrics.tg_tps:.1f} | PP={winner_res.metrics.pp_tps:.0f}"
        )

        # Re-rank the remaining measured frontier after every removal. Falling back in planner/list
        # order made v0.6.3 try a 5.6 t/s exact-context branch after an 8.4 t/s winner missed a soft
        # reserve by only 36 MiB. Every fallback now uses the same semantic Pareto policy as #1.
        remaining = list(success)
        ordered_pairs: list[tuple[SolutionOption, CandidateResult]] = []
        while remaining:
            nxt, _, _, _ = self._choose_dense_oversized_pair(
                remaining, target_ctx=target_ctx, preferred_k=preferred_k, preferred_v=preferred_v
            )
            ordered_pairs.append(nxt)
            remaining.remove(nxt)
        ordered: list[SolutionOption] = []
        for o, r in ordered_pairs:
            o.predicted_placement = int(r.candidate.ngl)
            o.predicted_free_mb = r.metrics.vram_free_min_mb
            if all(x.name != o.name for x in ordered):
                ordered.append(o)
        return ordered

    def _recon_dense_solution_options(self, options: list[SolutionOption], cores: int) -> list[SolutionOption]:
        """Build a measured Dense Pareto frontier using *stable* full-GPU knees.

        v0.5.7 deliberately searched the technical context ceiling, but a scout with only
        ~300 MiB free could become OPTIMAL and then fail a stronger FULL load. The generic
        placement recovery subsequently moved target layers to CPU and destroyed decode speed.

        v0.5.8 separates two concepts:
          * technical ceiling: useful for MAX_CONTEXT diagnostics, may sit near the hard floor;
          * stable knee: eligible for MAX_KV_PRECISION/OPTIMAL/FASTEST and deep tuning, must keep an
            operational uncertainty reserve above the hard floor.

        Only stable knees are context-scouted and allowed to win OPTIMAL. Technical ceilings
        remain visible in the measured frontier but are never silently converted to numeric ngl.
        """
        declared_exact = next((o for o in options if o.name == "EXACT_TARGET"), None)
        exact = next((o for o in options if o.name == "EXACT_TARGET" and o.resource_class != ResourceClass.INFEASIBLE), None)
        if exact is not None and not exact.degradation and exact.strategy == "full-gpu":
            return options
        eligible = [o for o in options if o.resource_class != ResourceClass.INFEASIBLE]
        if not eligible:
            return options

        target = declared_exact or exact
        target_ctx = target.context if target is not None else max(o.context for o in eligible)
        preferred_k = target.kv_k if target is not None else "f16"
        preferred_v = target.kv_v if target is not None else "f16"
        vision = bool(target.vision_required) if target is not None else any(o.vision_required for o in eligible)
        mmproj = target.mmproj if target is not None else next((o.mmproj for o in eligible if o.mmproj), None)
        operational_floor = self._operational_vram_floor_mb(vision=vision)
        technical_floor = self.absolute_vram_floor_mb + (32 if self.search_mode == "deep" else 64)

        full_gpu = [o for o in eligible if not o.strategy.startswith("dense-cpu-offload")]
        preferred_existing = [o for o in full_gpu if (o.kv_k, o.kv_v) == (preferred_k, preferred_v)]
        if self._dense_is_oversized(
            target_ctx=target_ctx, cores=cores, vision=vision, mmproj=mmproj
        ):
            return self._recon_dense_oversized_options(options, cores)
        if not preferred_existing:
            return options

        self.phase = "SOLUTION_RECON"
        self._emit("\n[Phase 0] Dense stable Pareto frontier")
        self._emit(
            f"  SCREEN → REFINE → CONFIRM: technical ceiling uses ~{technical_floor} MiB floor, "
            f"but OPTIMAL/MAX_KV_PRECISION candidates need >= {operational_floor} MiB operational headroom. "
            "A fragile ceiling is reported, never deep-tuned via Dense CPU layer fallback."
        )

        stable_success: list[tuple[SolutionOption, CandidateResult]] = []
        technical_success: list[tuple[SolutionOption, CandidateResult]] = []
        failed_names: set[str] = set()
        dynamic: list[SolutionOption] = []

        def runtime_memory_miss(result: CandidateResult | None) -> bool:
            if result is None or result.metrics.vram_free_min_mb is None:
                return False
            if int(result.metrics.vram_free_min_mb) >= operational_floor:
                return False
            return self._is_good(result) or result.reason in {
                "EARLY_REJECT_ABSOLUTE_VRAM_FLOOR", "FAIL_OOM", "FAIL_CUDA_OOM",
            }

        def lookup(option: SolutionOption) -> CandidateResult | None:
            for o, r in stable_success + technical_success:
                if (o.context == option.context and (o.kv_k, o.kv_v) == (option.kv_k, option.kv_v)
                        and r.candidate.ngl == "all"):
                    return r
            return None

        def probe(option: SolutionOption, label: str, *, technical: bool = False) -> CandidateResult | None:
            existing = lookup(option)
            if existing is not None:
                return existing
            if not self.budget_ok():
                return None
            seed = option.to_candidate(cores=cores, extra_args=self.base_extra_args)
            seed.ngl = "all"
            seed.ncmoe = None
            self._static_estimate = estimate_static_memory(
                self.model, self.hardware, self.baseline_vram_mb, self.vram_margin_mb, seed
            )
            self._emit(
                f"  {label}: {option.name} | ctx={seed.ctx} KV={seed.kv_k}/{seed.kv_v} "
                f"placement=all | predicted free≈{self._predicted_free_for(seed)} MiB"
            )
            r = self._run(seed, quick=True, phase="SOLUTION_RECON_PROBE", recon=True)
            if self._is_good(r) and r.metrics.pp_tps and r.metrics.tg_tps:
                free = r.metrics.vram_free_min_mb or 0
                bucket = technical_success if technical or free < operational_floor else stable_success
                bucket.append((option, r))
                state = "FRAGILE/DIAGNOSTIC" if bucket is technical_success else "STABLE"
                self._emit(
                    f"  recon result: {option.name} -> PP={r.metrics.pp_tps:.1f} t/s | "
                    f"TG={r.metrics.tg_tps:.1f} t/s | free={free} MiB | {state}"
                )
                return r
            failed_names.add(option.name)
            self._emit(f"  recon result: {option.name} -> rejected ({r.reason if r else 'NO_RESULT'})")
            return r

        # 1) One known preferred-KV full-GPU point calibrates dense:all.
        quality_seed = max(preferred_existing, key=lambda o: o.context)
        quality_r = probe(quality_seed, "preferred-KV calibration")
        # ``probe`` classifies runtime misses immediately; calibration evidence below the operating
        # floor is diagnostic and cannot leak into the stable winner pool.

        # Explicitly map the three primary KV-cache precision levels (FP16, Q8, Q4).
        # A mixed Q8/Q4 point remains a DEEP diagnostic unless it is the user's preferred pair.
        present_pairs = {(o.kv_k, o.kv_v) for o in full_gpu}
        primary_pairs = [(preferred_k, preferred_v), ("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")]
        if self.search_mode == "deep" or (preferred_k, preferred_v) == ("q8_0", "q4_0"):
            primary_pairs.append(("q8_0", "q4_0"))
        family_specs: list[tuple[str, str, int]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in primary_pairs:
            if pair in present_pairs and pair not in seen_pairs:
                seen_pairs.add(pair)
                family_specs.append((pair[0], pair[1], 2 + len(family_specs)))

        # 2) Per KV family: report a technical ceiling, but discover a separate stable knee.
        for kk, vv, rank in family_specs:
            tech_ctx, tech_free = self._dense_boundary_context(
                kv_k=kk, kv_v=vv, target_ctx=target_ctx, cores=cores,
                vision=vision, mmproj=mmproj, floor_mb=technical_floor,
            )
            stable_ctx, stable_free = self._dense_boundary_context(
                kv_k=kk, kv_v=vv, target_ctx=target_ctx, cores=cores,
                vision=vision, mmproj=mmproj, floor_mb=operational_floor,
            )
            if stable_ctx is None:
                continue

            existing_stable = next(
                (o for o in full_gpu if (o.kv_k, o.kv_v) == (kk, vv) and o.context == stable_ctx), None
            )
            stable_opt = existing_stable or self._dense_boundary_option(
                context=stable_ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                preferred_k=preferred_k, preferred_v=preferred_v, predicted_free_mb=stable_free,
                vision=vision, mmproj=mmproj, rank=rank,
            )
            if existing_stable is None:
                stable_opt.name = stable_opt.name.replace("ADAPTIVE_MAX_FULL_GPU", "STABLE_FULL_GPU_KNEE")
                stable_opt.strategy = "full-gpu-stable-knee"
                stable_opt.degradation_notes.insert(0, f"Stable operating knee targets >= {operational_floor} MiB VRAM headroom.")
                dynamic.append(stable_opt)
            stable_r = probe(stable_opt, "stable-knee scout")

            # If real VRAM missed the operating floor, back off using the measured KV slope rather
            # than handing the candidate to numeric ngl recovery.
            if runtime_memory_miss(stable_r) and stable_ctx > 16384:
                step = 4096
                measured_free = stable_r.metrics.vram_free_min_mb or 0
                # Static KV is near-linear in context. Interpolate at 1K granularity and target a
                # small explicit guard instead of paying a coarse 4K step plus another entire
                # safety step. One real scout remains authoritative.
                c_lo = Candidate(ctx=max(4096, stable_ctx-step), ngl="all", batch=512, ubatch=256,
                                 threads=cores, threads_batch=cores, kv_k=kk, kv_v=vv,
                                 vision=vision, mmproj=mmproj, extra_args=list(self.base_extra_args))
                pred_here = self._predicted_free_for(stable_r.candidate) or measured_free
                pred_lo = self._predicted_free_for(c_lo) or pred_here
                repair_ctx = self._interpolated_context_repair(
                    context=stable_ctx,
                    measured_free_mb=measured_free,
                    target_free_mb=operational_floor,
                    predicted_free_here_mb=pred_here,
                    predicted_free_lower_mb=pred_lo,
                    lower_context=c_lo.ctx,
                )
                repair_c = Candidate(ctx=repair_ctx, ngl="all", batch=512, ubatch=256,
                                     threads=cores, threads_batch=cores, kv_k=kk, kv_v=vv,
                                     vision=vision, mmproj=mmproj, extra_args=list(self.base_extra_args))
                repair_free = self._predicted_free_for(repair_c)
                repair = self._dense_boundary_option(
                    context=repair_ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                    preferred_k=preferred_k, preferred_v=preferred_v, predicted_free_mb=repair_free,
                    vision=vision, mmproj=mmproj, rank=rank,
                )
                repair.name = repair.name.replace("ADAPTIVE_MAX_FULL_GPU", "STABLE_FULL_GPU_REPAIR")
                repair.strategy = "full-gpu-stable-knee"
                repair.degradation_notes.insert(0, "Runtime VRAM interpolation selected the largest likely stable full-GPU context; CPU layer offload was not used.")
                dynamic.append(repair)
                rr = probe(repair, "stable-knee runtime repair")
                fragile_ctx = stable_ctx
                fragile_free = measured_free
                repaired_stable_ctx: int | None = None
                repaired_stable_free: int | None = None
                if (rr is not None and self._is_good(rr)
                        and (rr.metrics.vram_free_min_mb or 0) >= operational_floor):
                    repaired_stable_ctx = repair_ctx
                    repaired_stable_free = int(rr.metrics.vram_free_min_mb or 0)
                # If the interpolated scout still misses the floor, use the two *runtime* points
                # to make one final correction. This is bounded binary/interpolation repair, not
                # a fixed context ladder.
                if runtime_memory_miss(rr) and repair_ctx > 16384:
                    fragile_ctx = repair_ctx
                    fragile_free = int(rr.metrics.vram_free_min_mb or 0)
                    second_ctx = self._interpolated_context_repair(
                        context=stable_ctx,
                        measured_free_mb=measured_free,
                        target_free_mb=operational_floor,
                        predicted_free_here_mb=measured_free,
                        predicted_free_lower_mb=rr.metrics.vram_free_min_mb,
                        lower_context=repair_ctx,
                        guard_mb=48,
                    )
                    if 16384 <= second_ctx < repair_ctx:
                        repair2 = self._dense_boundary_option(
                            context=second_ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                            preferred_k=preferred_k, preferred_v=preferred_v,
                            predicted_free_mb=None, vision=vision, mmproj=mmproj, rank=rank,
                        )
                        repair2.name = repair2.name.replace("ADAPTIVE_MAX_FULL_GPU", "STABLE_FULL_GPU_REPAIR_2")
                        repair2.strategy = "full-gpu-stable-knee"
                        repair2.degradation_notes.insert(0, "Second runtime-calibrated context repair after the first interpolated scout missed the operating floor.")
                        dynamic.append(repair2)
                        rr2 = probe(repair2, "stable-knee measured-slope repair")
                        if rr2 is not None:
                            rr = rr2
                        if (rr2 is not None and self._is_good(rr2)
                                and (rr2.metrics.vram_free_min_mb or 0) >= operational_floor):
                            repaired_stable_ctx = second_ctx
                            repaired_stable_free = int(rr2.metrics.vram_free_min_mb or 0)

                # The repair above is conservative by construction.  If it cleared the floor,
                # use the measured fragile↔stable bracket once to recover context that the static
                # KV slope left on the table (for example 73,728→61,440 can refine to ~65,536).
                if (repaired_stable_ctx is not None and repaired_stable_free is not None
                        and fragile_ctx - repaired_stable_ctx >= 2048
                        and self.search_mode != "quick" and self.budget_ok()):
                    refined_ctx = self._runtime_bracket_context(
                        high_context=fragile_ctx,
                        high_free_mb=fragile_free,
                        low_context=repaired_stable_ctx,
                        low_free_mb=repaired_stable_free,
                        target_free_mb=operational_floor,
                        guard_mb=16,
                    )
                    if refined_ctx is not None:
                        refine = self._dense_boundary_option(
                            context=refined_ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                            preferred_k=preferred_k, preferred_v=preferred_v,
                            predicted_free_mb=None, vision=vision, mmproj=mmproj, rank=rank,
                        )
                        refine.name = refine.name.replace(
                            "ADAPTIVE_MAX_FULL_GPU", "STABLE_FULL_GPU_BRACKET_REFINE"
                        )
                        refine.strategy = "full-gpu-stable-knee"
                        refine.degradation_notes.insert(
                            0,
                            "Measured fragile/stable VRAM bracket recovered the largest likely "
                            "1K-aligned full-GPU context.",
                        )
                        dynamic.append(refine)
                        probe(refine, "stable-knee measured-bracket refinement")
                # Each repair was independently bucketed by its measured headroom in ``probe``.

            # A separate technical-ceiling launch is a DEEP diagnostic. NORMAL already records
            # any stable-knee attempt that proved fragile at runtime, which is enough to expose
            # the useful ceiling without spending another server launch per KV family.
            if (tech_ctx is not None and tech_ctx > stable_ctx and (kk, vv) != ("q8_0", "q4_0")
                    and self.search_mode == "deep"):
                tech_opt = self._dense_boundary_option(
                    context=tech_ctx, kv_k=kk, kv_v=vv, target_ctx=target_ctx,
                    preferred_k=preferred_k, preferred_v=preferred_v, predicted_free_mb=tech_free,
                    vision=vision, mmproj=mmproj, rank=rank+50,
                )
                tech_opt.name = tech_opt.name.replace("ADAPTIVE_MAX_FULL_GPU", "TECHNICAL_CEILING_FRAGILE")
                tech_opt.strategy = "full-gpu-technical-ceiling"
                tech_opt.degradation_notes.insert(0, "Technical memory ceiling only; not eligible for OPTIMAL deep tuning unless runtime headroom also clears the stable operating floor.")
                dynamic.append(tech_opt)
                tr = probe(tech_opt, "technical-ceiling scout", technical=True)
                if tr is not None and self._is_good(tr) and (tr.metrics.vram_free_min_mb or 0) >= operational_floor:
                    # It turned out stable in reality: promote it to stable pool.
                    technical_success[:] = [(o, r) for o, r in technical_success if r is not tr]
                    tech_opt.strategy = "full-gpu-stable-knee"
                    stable_success.append((tech_opt, tr))

        # 3) A separate low-context anchor rarely changes Dense decode once a stable preferred-KV
        # knee already exists; keep it as a DEEP diagnostic instead of spending a NORMAL launch.
        if self.search_mode == "deep":
            low_quality = min(preferred_existing, key=lambda o: o.context)
            if lookup(low_quality) is None:
                lr = probe(low_quality, "low-context preferred-KV/speed anchor")
                # ``probe`` already keeps a low-headroom DEEP anchor out of stable eligibility.

        # 4) Quantify exact requested context cost with the tiny CPU-offload scout.
        exact_result: CandidateResult | None = None
        if exact is not None and self.budget_ok():
            seed = exact.to_candidate(cores=cores, extra_args=self.base_extra_args)
            self._static_estimate = estimate_static_memory(
                self.model, self.hardware, self.baseline_vram_mb, self.vram_margin_mb, seed
            )
            self._emit(f"  exact-target cost scout: {exact.name} | ctx={seed.ctx} KV={seed.kv_k}/{seed.kv_v} placement={seed.ngl}")
            exact_result = self._run(seed, quick=True, phase="SOLUTION_RECON_PROBE", recon=True)
            if self._is_good(exact_result) and exact_result.metrics.pp_tps and exact_result.metrics.tg_tps:
                self._emit(
                    f"  recon result: EXACT_TARGET -> PP={exact_result.metrics.pp_tps:.1f} t/s | "
                    f"TG={exact_result.metrics.tg_tps:.1f} t/s | free={exact_result.metrics.vram_free_min_mb} MiB"
                )
                refined_seed = (
                    self._calibrated_exact_dense_scout(seed, exact_result)
                    if self.search_mode != "quick" else None
                )
                if refined_seed is not None and self.budget_ok():
                    self._emit(
                        f"  exact-target calibrated placement scout: ngl={seed.ngl}→{refined_seed.ngl}; "
                        "one bounded measurement improves MAX_CONTEXT evidence without starting a layer walk."
                    )
                    refined_exact = self._run(
                        refined_seed, quick=True, phase="SOLUTION_RECON_EXACT_REFINE", recon=True
                    )
                    if (self._is_good(refined_exact) and refined_exact.metrics.pp_tps
                            and refined_exact.metrics.tg_tps):
                        self._emit(
                            f"  exact-target refined result: ngl={refined_exact.candidate.ngl} | "
                            f"PP={refined_exact.metrics.pp_tps:.1f} t/s | "
                            f"TG={refined_exact.metrics.tg_tps:.1f} t/s | "
                            f"free={refined_exact.metrics.vram_free_min_mb} MiB"
                        )
                        if ((refined_exact.metrics.vram_free_min_mb or 0) >= operational_floor
                                and self._perf_score(refined_exact) > self._perf_score(exact_result)):
                            exact_result = refined_exact
                            exact.predicted_placement = int(refined_exact.candidate.ngl)
                            exact.predicted_free_mb = refined_exact.metrics.vram_free_min_mb
            else:
                failed_names.add(exact.name)

        if not stable_success:
            self._emit("  no stable full-GPU Dense knee was measured; fall back to explicit planner alternatives.")
            all_options = list(options) + [o for o in dynamic if all(x.name != o.name for x in options)]
            return all_options

        max_short_tg = max(float(r.metrics.tg_tps or 0.0) for _, r in stable_success)
        max_short_pp = max(float(r.metrics.pp_tps or 0.0) for _, r in stable_success)
        competitive = [
            (o, r) for o, r in stable_success
            if float(r.metrics.tg_tps or 0.0) >= max_short_tg * 0.80
            and float(r.metrics.pp_tps or 0.0) >= max_short_pp * 0.65
            and (r.metrics.vram_free_min_mb or 0) >= operational_floor
        ] or stable_success

        # Build one representative per primary KV-cache precision family.  This preserves the
        # FP16 → Q8 → Q4 sweet-spot search instead of collapsing every lower tier into one boolean
        # "quality risk" bucket. Mixed families remain diagnostic unless DEEP selected them above.
        finalists: list[tuple[SolutionOption, CandidateResult]] = []
        families: dict[tuple[str, str], list[tuple[SolutionOption, CandidateResult]]] = {}
        for o, r in competitive:
            families.setdefault((o.kv_k, o.kv_v), []).append((o, r))
        ordered_families = sorted(
            families,
            key=lambda pair: kv_precision_key(pair[0], pair[1]),
            reverse=True,
        )
        for pair in ordered_families:
            representative = max(families[pair], key=lambda x: (x[0].context, x[1].metrics.tg_tps or 0.0))
            finalists.append(representative)
            if self.search_mode != "deep" and len(finalists) >= 3:
                break
        # The preferred precision family remains visible even if it missed the performance cutoff.
        preferred_any = [(o, r) for o, r in stable_success if (o.kv_k, o.kv_v) == (preferred_k, preferred_v)]
        if preferred_any and not any((o.kv_k, o.kv_v) == (preferred_k, preferred_v) for o, _ in finalists):
            finalists.insert(0, max(preferred_any, key=lambda x: x[0].context))
            if self.search_mode != "deep":
                finalists = finalists[:3]

        stronger: dict[str, CandidateResult] = {}
        kv_runtime_failed: set[str] = set()
        context_disqualified: set[str] = set()
        context_diagnostics: list[tuple[SolutionOption, CandidateResult]] = []
        if self.search_mode != "quick" and len(finalists) >= 2:
            limit = len(finalists) if self.search_mode == "deep" else min(3, len(finalists))
            self._emit(f"  REFINE: context-scouting {limit} stable KV-precision Pareto representative(s).")
            for o, short_r in finalists[:limit]:
                if not self.budget_ok():
                    break
                long_target = self._kv_long_scout_target(short_r.candidate)
                if long_target is not None:
                    self._emit(
                        f"  Q4 occupied-cache qualification: fill ~{long_target} tokens before "
                        "promotion; short-context TG is not authoritative for this KV tier."
                    )
                cr = self._run(copy.deepcopy(short_r.candidate), quick=True, phase="SOLUTION_CONTEXT_SCOUT",
                               recon_context=True, recon_context_target=long_target,
                               reference=short_r)
                if self._is_good(cr) and cr.metrics.pp_tps and cr.metrics.tg_tps \
                        and (cr.metrics.vram_free_min_mb or 0) >= self._vram_thresholds(cr.candidate).tight_floor_mb:
                    stronger[short_r.candidate.key()] = cr
                    self._emit(
                        f"  context scout result: ctx={o.context} KV={o.kv_k}/{o.kv_v} -> "
                        f"PP={cr.metrics.pp_tps:.1f} t/s | TG={cr.metrics.tg_tps:.1f} t/s | "
                        f"filled={cr.metrics.long_context_tokens or cr.metrics.prompt_total_tokens or cr.metrics.prompt_tokens} tok | "
                        f"free={cr.metrics.vram_free_min_mb} MiB"
                    )
                else:
                    local_memory_or_scaling_failure = (
                        (self._is_good(cr) and cr.metrics.vram_free_min_mb is not None)
                        or cr.reason in {
                            "EARLY_REJECT_ABSOLUTE_VRAM_FLOOR", "FAIL_OOM", "FAIL_CUDA_OOM",
                            "EARLY_REJECT_SEVERE_PERFORMANCE_CLIFF",
                            "EARLY_REJECT_LONG_CONTEXT_CLIFF",
                        }
                    )
                    if local_memory_or_scaling_failure and not self._environmental_final_failure(cr):
                        key = short_r.candidate.key()
                        context_disqualified.add(key)
                        context_diagnostics.append((o, cr))
                        failed_names.add(o.name)
                        if long_target is not None:
                            kv_runtime_failed.add(key)
                            self._emit(
                                "  Q4 occupied-cache qualification failed; retain the point as diagnostic/MAX_CONTEXT "
                                "evidence but exclude it from the automatic OPTIMAL branch."
                            )
                        else:
                            self._emit(
                                f"  context scout invalidated the short stable label for ctx={o.context} "
                                f"KV={o.kv_k}/{o.kv_v} ({cr.reason}, "
                                f"free={cr.metrics.vram_free_min_mb} MiB); the short SCOUT will not be reused "
                                "as an automatic winner."
                            )

        def rr(o: SolutionOption, r: CandidateResult) -> CandidateResult:
            return stronger.get(r.candidate.key(), r)
        selection_pool = [
            (o, rr(o, r)) for o, r in finalists
            if r.candidate.key() not in kv_runtime_failed
            and r.candidate.key() not in context_disqualified
        ] or [
            (o, rr(o, r)) for o, r in finalists
            if not kv_requires_long_context_probe(o.kv_k, o.kv_v)
            and r.candidate.key() not in context_disqualified
        ] or [
            (o, r) for o, r in competitive if r.candidate.key() not in context_disqualified
        ] or [
            (o, r) for o, r in stable_success if r.candidate.key() not in context_disqualified
        ]

        if not selection_pool:
            self._emit(
                "  every context-scouted Dense representative lost its short-run stable status; "
                "fall back to explicit planner alternatives instead of reviving contradicted SCOUT evidence."
            )
            all_options = list(options) + [
                o for o in dynamic if all(x.name != o.name for x in options)
            ]
            remaining = [o for o in all_options if o.name not in failed_names]
            return remaining or all_options

        def tie_prefer(a_opt, a_res, b_opt, b_res):
            if self.selection_priority == "context":
                return (a_opt, a_res) if a_opt.context >= b_opt.context else (b_opt, b_res)
            ap = kv_precision_key(a_opt.kv_k, a_opt.kv_v)
            bp = kv_precision_key(b_opt.kv_k, b_opt.kv_v)
            if self.selection_priority == "quality" and ap != bp:
                return (a_opt, a_res) if ap > bp else (b_opt, b_res)
            if self.selection_priority == "balanced":
                if ap == bp:
                    return (a_opt, a_res) if a_opt.context >= b_opt.context else (b_opt, b_res)
                higher = (a_opt, a_res) if ap > bp else (b_opt, b_res)
                lower = (a_opt, a_res) if ap < bp else (b_opt, b_res)
                if self._balanced_lower_kv_allowed(*higher, *lower):
                    return lower
                return higher
            pref = choose_preferred([a_res, b_res], self.workload_profile, self.noise_policy)
            return (b_opt, b_res) if pref is b_res else (a_opt, a_res)

        winner_opt, winner_res = selection_pool[0]
        for option, result in selection_pool[1:]:
            if self.selection_priority == "balanced":
                ap = kv_precision_key(winner_opt.kv_k, winner_opt.kv_v)
                bp = kv_precision_key(option.kv_k, option.kv_v)
                if ap != bp:
                    higher = (winner_opt, winner_res) if ap > bp else (option, result)
                    lower = (winner_opt, winner_res) if ap < bp else (option, result)
                    if kv_requires_long_context_probe(lower[0].kv_k, lower[0].kv_v) \
                            and not self._balanced_lower_kv_allowed(*higher, *lower):
                        winner_opt, winner_res = higher
                        continue
            if self._performance_equivalent(result, winner_res):
                old = winner_opt.name
                winner_opt, winner_res = tie_prefer(winner_opt, winner_res, option, result)
                if old != winner_opt.name:
                    self._emit(f"  REFINE tie: performance is inside noise; {self.selection_priority} policy prefers {winner_opt.name}.")
            else:
                pref = choose_preferred([winner_res, result], self.workload_profile, self.noise_policy)
                if pref is result:
                    winner_opt, winner_res = option, result

        # Evidence table: distinguish stable recommendations from fragile technical ceilings.
        recommendation_stable = [
            (o, r) for o, r in stable_success if r.candidate.key() not in context_disqualified
        ]
        q_pool = [(o, rr(o, r)) for o, r in recommendation_stable if (o.kv_k, o.kv_v) == (preferred_k, preferred_v)]
        quality_pair = max(q_pool, key=lambda x: x[0].context, default=None)
        stable_max = max([(o, rr(o, r)) for o, r in recommendation_stable], key=lambda x: x[0].context, default=None)
        technical_max = max(
            technical_success + [
                (o, r) for o, r in context_diagnostics
                if r.metrics.pp_tps is not None and r.metrics.tg_tps is not None
            ],
            key=lambda x: x[0].context, default=None,
        )
        fastest = max(
            recommendation_stable,
            key=lambda x: (float(x[1].metrics.tg_tps or 0.0), x[1].metrics.vram_free_min_mb or 0),
        )

        self._emit(
            f"  reconnaissance winner: {winner_opt.name} | ctx={winner_opt.context} KV={winner_opt.kv_k}/{winner_opt.kv_v} | "
            f"PP={winner_res.metrics.pp_tps:.1f} t/s | TG={winner_res.metrics.tg_tps:.1f} t/s. "
            "Only this stable branch proceeds to expensive tuning."
        )
        self.provisional_recommendation_key = winner_res.candidate.key()
        self._emit("  measured Dense solution frontier:")
        if quality_pair:
            o, r = quality_pair
            self._emit(f"    MAX_KV_PRECISION_STABLE_FULL_GPU: ctx={o.context} KV={o.kv_k}/{o.kv_v} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB")
        if stable_max:
            o, r = stable_max
            self._emit(f"    MAX_STABLE_FULL_GPU_CONTEXT: ctx={o.context} KV={o.kv_k}/{o.kv_v} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB")
        if technical_max:
            o, r = technical_max
            self._emit(f"    MAX_TECHNICAL_CONTEXT_FRAGILE: ctx={o.context} KV={o.kv_k}/{o.kv_v} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB")
        if exact_result is not None and self._is_good(exact_result):
            self._emit(f"    EXACT_TARGET_COST: ctx={exact_result.candidate.ctx} placement={exact_result.candidate.ngl} | PP={exact_result.metrics.pp_tps or 0:.0f} | TG={exact_result.metrics.tg_tps or 0:.1f}")
        o, r = fastest
        self._emit(f"    FASTEST_STABLE_SHORT: ctx={o.context} KV={o.kv_k}/{o.kv_v} | TG={r.metrics.tg_tps or 0:.1f} | free={r.metrics.vram_free_min_mb} MiB")
        self._emit(f"    OPTIMAL_STABLE: ctx={winner_opt.context} KV={winner_opt.kv_k}/{winner_opt.kv_v}")

        all_options = list(options)
        known = {o.name for o in all_options}
        for o in dynamic:
            if o.name not in known:
                all_options.append(o); known.add(o.name)

        # Try measured stable branches first. Fragile technical ceilings are diagnostics/MAX_CONTEXT,
        # not fallback candidates for NORMAL deep tuning.
        ordered = [winner_opt]
        remaining_stable = [
            (o, rr(o, r)) for o, r in recommendation_stable if o.name != winner_opt.name
        ]
        while remaining_stable:
            next_opt, next_res = remaining_stable[0]
            for option, result in remaining_stable[1:]:
                if self._performance_equivalent(result, next_res):
                    next_opt, next_res = tie_prefer(next_opt, next_res, option, result)
                elif choose_preferred(
                    [next_res, result], self.workload_profile, self.noise_policy
                ) is result:
                    next_opt, next_res = option, result
            ordered.append(next_opt)
            remaining_stable = [(o, r) for o, r in remaining_stable if o.name != next_opt.name]
        for o in all_options:
            if o.strategy == "full-gpu-technical-ceiling":
                continue
            if o.name not in failed_names and all(x.name != o.name for x in ordered):
                ordered.append(o)
        return ordered

    def _recon_solution_options(self, options: list[SolutionOption], cores: int) -> list[SolutionOption]:
        """Cheaply measure competing *solution-level* full-GPU trade-offs before deep tuning.

        Static memory planning can rank feasibility, but it cannot know that (for example) a mixed
        Q8/Q4 KV path is much slower than Q4/Q4 on a particular llama.cpp/GPU combination. When the
        requested preferred-KV-precision target is not already a clean full-GPU option, probe a small
        Pareto-like shortlist and let real PP/TG/VRAM measurements choose the branch. This avoids
        spending the whole 5-minute search on the first merely-runnable compromise.
        """
        if len(options) <= 1 or not self.budget_ok():
            return options
        if self.model.kind == ModelKind.DENSE and any(o.name == "EXACT_TARGET" for o in options):
            return self._recon_dense_solution_options(options, cores)
        first = options[0]
        if first.exact_target and not first.degradation and first.strategy == "full-gpu":
            return options

        # Dense reconnaissance is an intentionally small *frontier*, not "the first four static ranks".
        # NORMAL guarantees one representative for each primary FP16/Q8/Q4 family.  A mixed Q8/Q4
        # kernel is DEEP-only unless the user explicitly selected that pair; it must never displace
        # the Q8 sweet spot merely because the shortlist hit its launch cap.
        declared_exact = next((o for o in options if o.name == "EXACT_TARGET"), None)
        if declared_exact is None:
            declared_exact = min(
                (o for o in options if o.exact_target),
                key=lambda o: o.recommended_rank,
                default=None,
            )
        exact = next((o for o in options if o.name == "EXACT_TARGET" and o.resource_class.value != "INFEASIBLE"), None)
        eligible = [o for o in options if o.resource_class.value != "INFEASIBLE"]
        shortlist: list[SolutionOption] = []

        def add(opt: SolutionOption | None) -> None:
            if opt is not None and all(x.name != opt.name for x in shortlist):
                shortlist.append(opt)

        if self.model.kind == ModelKind.DENSE:
            full_gpu = [o for o in eligible if not o.strategy.startswith("dense-cpu-offload")]
            target = declared_exact or exact
            preferred_k = target.kv_k if target is not None else "f16"
            preferred_v = target.kv_v if target is not None else "f16"
            preferred = [o for o in full_gpu if (o.kv_k, o.kv_v) == (preferred_k, preferred_v)]
            q8q8 = [o for o in full_gpu if (o.kv_k, o.kv_v) == ("q8_0", "q8_0")]
            q8q4 = [o for o in full_gpu if (o.kv_k, o.kv_v) == ("q8_0", "q4_0")]
            q4q4 = [o for o in full_gpu if (o.kv_k, o.kv_v) == ("q4_0", "q4_0")]

            # Start with useful GPU-resident anchors so the first real calibration is also full-GPU.
            add(max(preferred, key=lambda o: o.context, default=None))             # preferred precision
            add(max(q8q8, key=lambda o: o.context, default=None))                   # explicit Q8 sweet spot
            add(max(q4q4, key=lambda o: o.context, default=None))                   # context-efficient Q4
            if self.search_mode == "deep" or (preferred_k, preferred_v) == ("q8_0", "q4_0"):
                add(max(q8q4, key=lambda o: o.context, default=None))               # mixed diagnostic
            add(exact)                                                               # exact target cost, if runnable
            if self.search_mode == "deep":
                add(min(preferred, key=lambda o: o.context, default=None))          # speed anchor
            # Fill a missing primary slot with the highest-headroom full-GPU option.
            cap = {"quick": 3, "normal": 4, "deep": 6}.get(self.search_mode, 4)
            if len(shortlist) < cap and full_gpu:
                add(max(full_gpu, key=lambda o: (o.predicted_free_mb or -10**9, -o.context)))
            shortlist = shortlist[:cap]
        else:
            # MoE NORMAL should compare *meaningfully different residency trade-offs*, not the
            # first four static ranks.  Exact preferred-KV is always measured.  Then measure one
            # same-context KV-saving option (if available) and one preferred-KV reduced-context
            # option that is predicted to buy additional GPU-resident experts.
            if exact is not None:
                add(exact)
            requested_ctx = exact.context if exact is not None else max(o.context for o in eligible)
            exact_place = int(exact.predicted_placement) if exact is not None and isinstance(exact.predicted_placement, int) else None
            # Measure the Q8 sweet spot explicitly before Q4.  Treating all lower KV types as
            # one boolean degradation can skip the most useful MoE residency regime.
            for pair in [("q8_0", "q8_0"), ("q4_0", "q4_0")]:
                if kv_precision_key(*pair) >= kv_precision_key(
                    exact.kv_k if exact is not None else "f16",
                    exact.kv_v if exact is not None else "f16",
                ):
                    continue
                family = [
                    o for o in eligible if o is not exact and o.context == requested_ctx
                    and (o.kv_k, o.kv_v) == pair
                ]
                if family:
                    add(min(family, key=lambda o: (
                        int(o.predicted_placement) if isinstance(o.predicted_placement, int) else 999,
                        -(o.predicted_free_mb if o.predicted_free_mb is not None else -10**9),
                        o.recommended_rank,
                    )))
            regime_cap = {"quick": 3, "normal": 4, "deep": 5}.get(self.search_mode, 4)
            quality_ctx = [
                o for o in eligible if o is not exact and DegradationKind.QUALITY_RISK not in o.degradation
                and o.context < requested_ctx
            ]
            if quality_ctx and len(shortlist) < regime_cap:
                if exact_place is not None:
                    buys_experts = [o for o in quality_ctx if isinstance(o.predicted_placement, int) and int(o.predicted_placement) < exact_place]
                else:
                    buys_experts = []
                # Context is not a useful independent MoE axis. In NORMAL, only spend a cold
                # launch when lower KV memory is predicted to cross an expert-residency boundary.
                # If placement is unknown, keep one calibration point; if it is known and unchanged,
                # the reduced-context branch is pruned before runtime.
                if buys_experts:
                    add(max(buys_experts, key=lambda o: o.context))
                elif exact_place is None:
                    add(max(quality_ctx, key=lambda o: o.context))
            # NORMAL admits one context point only when it buys expert residency; QUICK remains
            # at the three same-context KV tiers, and DEEP may add one more diagnostic regime.
            if self.search_mode == "deep":
                for o in eligible:
                    if o is exact:
                        continue
                    add(o)
                    if len(shortlist) >= regime_cap:
                        break
            shortlist = shortlist[:regime_cap]

        if len(shortlist) <= 1:
            return options

        self.phase = "SOLUTION_RECON"
        self._emit("\n[Phase 0] Solution reconnaissance: quick-measuring the most promising target trade-offs")
        self._emit(
            "  Static planning chooses what is worth testing; real PP/TG/VRAM compares the exact target "
            "against a small set of architecture-aware alternatives before expensive tuning begins."
        )
        successful: list[tuple[SolutionOption, CandidateResult]] = []
        failed_names: set[str] = set()
        for option in shortlist:
            if not self.budget_ok():
                break
            seed = option.to_candidate(cores=cores, extra_args=self.base_extra_args)
            if self.model.kind == ModelKind.MOE and "moe" in self._static_free_corrections_mb:
                adjusted = self._moe_seed_for_operational_floor(seed)
                if adjusted.ncmoe != seed.ncmoe:
                    self._emit(f"  calibrated MoE seed: ncmoe={seed.ncmoe} -> {adjusted.ncmoe} before launch.")
                seed = adjusted
                option.predicted_placement = seed.ncmoe
            self._static_estimate = estimate_static_memory(
                self.model, self.hardware, self.baseline_vram_mb, self.vram_margin_mb, seed
            )
            self._emit(
                f"  recon #{option.recommended_rank}: {option.name} | ctx={seed.ctx} "
                f"KV={seed.kv_k}/{seed.kv_v} | predicted free≈{option.predicted_free_mb} MiB"
            )
            r = self._run(seed, quick=True, phase="SOLUTION_RECON_PROBE", recon=True)
            # Static MoE placement is approximate. If the seed misses the hard floor by one or
            # two expert layers, recover cheaply instead of deleting an otherwise useful solution
            # branch from reconnaissance.
            if (not self._is_good(r) and self.model.kind == ModelKind.MOE
                    and self._recoverable_boundary_reason(r.reason) and seed.ncmoe is not None):
                safer = self._moe_seed_for_operational_floor(seed)
                if safer.ncmoe == seed.ncmoe:
                    safer = copy.deepcopy(seed)
                    safer.ncmoe = min(model_main_block_count(self.model), int(seed.ncmoe) + 1)
                attempts = [safer]
                if safer.ncmoe is not None and safer.ncmoe < model_main_block_count(self.model):
                    one_more = copy.deepcopy(safer); one_more.ncmoe += 1; attempts.append(one_more)
                for safer in attempts:
                    self._emit(f"  recon recovery: calibrated jump for {option.name} -> ncmoe={safer.ncmoe}.")
                    rr = self._run(safer, quick=True, phase="SOLUTION_RECON_RECOVERY", recon=True)
                    if self._is_good(rr):
                        r = rr
                        option.predicted_placement = safer.ncmoe
                        option.predicted_free_mb = rr.metrics.vram_free_min_mb
                        break
            # A MoE scout can technically pass a 1.2K prompt with only ~300-400 MiB
            # remaining.  Do not compare that fragile placement against an operational
            # Q4/F16 point: first spend one expert-offload step to put Q8 and Q4 on the
            # same recommendation footing.
            if (self.model.kind == ModelKind.MOE and self._is_good(r)
                    and self._vram_class(r) == VramOperatingClass.FRAGILE
                    and r.candidate.ncmoe is not None and self.budget_ok()):
                safer = self._moe_seed_for_operational_floor(r.candidate)
                if safer.ncmoe is None or safer.ncmoe <= r.candidate.ncmoe:
                    safer = copy.deepcopy(r.candidate)
                    safer.ncmoe = min(
                        model_main_block_count(self.model), int(r.candidate.ncmoe) + 1,
                    )
                if safer.ncmoe != r.candidate.ncmoe:
                    self._emit(
                        f"  recon operational recovery: {option.name} passed only as FRAGILE; "
                        f"retry once at ncmoe={safer.ncmoe}."
                    )
                    rr = self._run(
                        safer, quick=True, phase="SOLUTION_RECON_OPERATIONAL_RECOVERY", recon=True,
                    )
                    if self._is_good(rr) and (
                        (rr.metrics.vram_free_min_mb or 0) > (r.metrics.vram_free_min_mb or 0)
                    ):
                        r = rr
                        option.predicted_placement = safer.ncmoe
                        option.predicted_free_mb = rr.metrics.vram_free_min_mb
            recommendation_eligible_scout = not (
                self.model.kind == ModelKind.MOE
                and self._vram_class(r) == VramOperatingClass.FRAGILE
            )
            if self._is_good(r) and r.metrics.pp_tps and r.metrics.tg_tps \
                    and recommendation_eligible_scout:
                successful.append((option, r))
                self._emit(
                    f"  recon result: {option.name} -> {r.candidate.short()} | PP={r.metrics.pp_tps:.1f} t/s | "
                    f"TG={r.metrics.tg_tps:.1f} t/s | free={r.metrics.vram_free_min_mb} MiB"
                )
            else:
                failed_names.add(option.name)
                suffix = "FRAGILE after bounded recovery" if self._is_good(r) else r.reason
                self._emit(f"  recon result: {option.name} -> rejected ({suffix})")

        if not successful:
            self._emit("  reconnaissance found no usable full-GPU compromise; continue with architecture-aware fallbacks.")
            return [o for o in options if o.name not in failed_names] or options

        # Q4 can gain GPU expert residency and look excellent at an almost-empty KV
        # cache, yet lose that advantage once tens of thousands of entries are occupied.
        # For long-context MoE, compare the best Q4 point with the nearest higher tier at
        # one identical fill level before either can become OPTIMAL/FASTEST.
        if self.model.kind == ModelKind.MOE and self.search_mode != "quick" \
                and self.workload_profile == "long-context":
            sensitive = [
                pair for pair in successful
                if kv_requires_long_context_probe(pair[0].kv_k, pair[0].kv_v)
            ]
            if sensitive:
                lower_opt, lower_short = max(
                    sensitive,
                    key=lambda pair: (pair[0].context, pair[1].metrics.tg_tps or 0.0),
                )
                lower_precision = kv_precision_key(lower_opt.kv_k, lower_opt.kv_v)
                higher_pool = [
                    pair for pair in successful
                    if kv_precision_key(pair[0].kv_k, pair[0].kv_v) > lower_precision
                ]
                if higher_pool:
                    higher_opt, higher_short = min(
                        higher_pool,
                        key=lambda pair: (
                            kv_precision_key(pair[0].kv_k, pair[0].kv_v),
                            -pair[0].context,
                            pair[0].recommended_rank,
                        ),
                    )
                    target = self._kv_long_scout_target(lower_short.candidate)
                    if target is not None:
                        target = min(
                            target,
                            max(2_048, lower_short.candidate.ctx - 1_024),
                            max(2_048, higher_short.candidate.ctx - 1_024),
                        )
                        self._emit(
                            f"  KV long-context A/B: {higher_opt.kv_k}/{higher_opt.kv_v} vs "
                            f"{lower_opt.kv_k}/{lower_opt.kv_v} at one ~{target}-token fill. "
                            "This measures runtime scaling only; it cannot prove task-quality equivalence."
                        )
                        estimated_pair_seconds = sum(
                            target / max(1.0, float(short.metrics.pp_tps or 1.0))
                            for short in (lower_short, higher_short)
                        )
                        self._emit(
                            f"  estimated A/B prefill time from short scouts: "
                            f"~{estimated_pair_seconds / 60.0:.1f} min total. "
                            "Q4 runs first; a severe Q4-only scaling collapse skips the control launch."
                        )
                        replacements: dict[str, CandidateResult] = {}
                        lower_failed = False
                        for opt, short in [(lower_opt, lower_short), (higher_opt, higher_short)]:
                            if lower_failed and short is higher_short:
                                self._emit(
                                    "  higher-KV long control skipped: Q4 already failed its own occupied-cache "
                                    "scaling gate, so a second multi-minute launch cannot promote it."
                                )
                                break
                            long_result = self._run(
                                copy.deepcopy(short.candidate), quick=True,
                                phase="SOLUTION_KV_LONG_SCOUT", recon_context=True,
                                recon_context_target=target, reference=short,
                            )
                            if (self._is_good(long_result) and long_result.metrics.pp_tps
                                    and long_result.metrics.tg_tps
                                    and self._vram_class(long_result) != VramOperatingClass.FRAGILE):
                                replacements[short.candidate.key()] = long_result
                                self._emit(
                                    f"  KV long A/B result: {opt.kv_k}/{opt.kv_v} | "
                                    f"PP={long_result.metrics.pp_tps:.1f} | TG={long_result.metrics.tg_tps:.1f} | "
                                    f"filled={long_result.metrics.long_context_tokens or long_result.metrics.prompt_total_tokens or long_result.metrics.prompt_tokens} tok | "
                                    f"free={long_result.metrics.vram_free_min_mb} MiB"
                                )
                                if short is lower_short:
                                    short_tg = float(lower_short.metrics.tg_tps or 0.0)
                                    long_tg = float(long_result.metrics.tg_tps or 0.0)
                                    if short_tg > 0 and long_tg < short_tg * 0.75:
                                        lower_failed = True
                                        replacements.pop(short.candidate.key(), None)
                                        self._emit(
                                            f"  Q4 occupied-cache scaling collapsed to {long_tg / short_tg:.0%} "
                                            "of its own short-scout decode; automatic promotion is rejected."
                                        )
                            elif short is lower_short and not self._environmental_final_failure(long_result):
                                lower_failed = True
                        successful = [
                            (opt, replacements.get(short.candidate.key(), short))
                            for opt, short in successful
                            if not (lower_failed and short.candidate.key() == lower_short.candidate.key())
                        ]
                        if lower_failed:
                            failed_names.add(lower_opt.name)
                            self._emit(
                                "  Q4 long-context runtime qualification failed; keep its scout as a "
                                "diagnostic, not an automatic recommendation."
                            )

        # Start from the planner's semantic preference, but do not let a fixed rank silently
        # decide a large target-fidelity trade-off after real performance has tied. Example:
        # for a requested 256K context, 64K Q8 and 128K Q4 can have identical PP/TG; choosing
        # 64K merely because a higher KV-precision tier has a lower static rank throws away half the usable
        # context without a measured performance benefit.
        successful.sort(key=lambda item: item[0].recommended_rank)
        target_ctx = max((o.context for o in options if o.exact_target), default=max(o.context for o, _ in successful))

        def semantic_tie_prefer(
            a_opt: SolutionOption, a_res: CandidateResult, b_opt: SolutionOption, b_res: CandidateResult
        ) -> tuple[SolutionOption, CandidateResult]:
            priority = self.selection_priority
            if priority == "context":
                if a_opt.context != b_opt.context:
                    return (a_opt, a_res) if a_opt.context > b_opt.context else (b_opt, b_res)
            elif priority == "quality":
                ap = kv_precision_key(a_opt.kv_k, a_opt.kv_v)
                bp = kv_precision_key(b_opt.kv_k, b_opt.kv_v)
                if ap != bp:
                    return (a_opt, a_res) if ap > bp else (b_opt, b_res)
            elif priority == "balanced":
                ap = kv_precision_key(a_opt.kv_k, a_opt.kv_v)
                bp = kv_precision_key(b_opt.kv_k, b_opt.kv_v)
                if ap == bp and a_opt.context != b_opt.context:
                    return (a_opt, a_res) if a_opt.context > b_opt.context else (b_opt, b_res)
                if ap != bp:
                    higher = (a_opt, a_res) if ap > bp else (b_opt, b_res)
                    lower = (a_opt, a_res) if ap < bp else (b_opt, b_res)
                    if self._balanced_lower_kv_allowed(*higher, *lower):
                        return lower
                    return higher

            # Speed priority, or a semantic tie after the rules above: use the normal
            # noise-aware PP/TG/headroom selector, then deterministic planner rank.
            preferred = choose_preferred([a_res, b_res], self.workload_profile, self.noise_policy)
            if preferred is b_res:
                return b_opt, b_res
            if preferred is a_res:
                return a_opt, a_res
            return (a_opt, a_res) if a_opt.recommended_rank <= b_opt.recommended_rank else (b_opt, b_res)

        winner_opt, winner_res = successful[0]
        for option, result in successful[1:]:
            if self.selection_priority == "balanced":
                ap = kv_precision_key(winner_opt.kv_k, winner_opt.kv_v)
                bp = kv_precision_key(option.kv_k, option.kv_v)
                if ap != bp:
                    higher = (winner_opt, winner_res) if ap > bp else (option, result)
                    lower = (winner_opt, winner_res) if ap < bp else (option, result)
                    if kv_requires_long_context_probe(lower[0].kv_k, lower[0].kv_v) \
                            and not self._balanced_lower_kv_allowed(*higher, *lower):
                        winner_opt, winner_res = higher
                        continue
            if self._performance_equivalent(result, winner_res):
                old_name = winner_opt.name
                winner_opt, winner_res = semantic_tie_prefer(winner_opt, winner_res, option, result)
                if winner_opt.name != old_name:
                    self._emit(
                        f"  recon tie-break: measured performance is inside the noise zone; "
                        f"{self.selection_priority} target-fidelity policy prefers {winner_opt.name}."
                    )
                continue
            preferred = choose_preferred([winner_res, result], self.workload_profile, self.noise_policy)
            if preferred is result:
                winner_opt, winner_res = option, result

        self._emit(
            f"  reconnaissance winner: {winner_opt.name} | ctx={winner_opt.context} "
            f"KV={winner_opt.kv_k}/{winner_opt.kv_v} | PP={winner_res.metrics.pp_tps:.1f} t/s | "
            f"TG={winner_res.metrics.tg_tps:.1f} t/s. Deep tuning starts from this measured branch."
        )
        self.provisional_recommendation_key = winner_res.candidate.key()
        if self.model.kind == ModelKind.DENSE:
            # Show the user the measured envelope immediately. These are scouts, not final validation.
            preferred_kv = [(o, r) for o, r in successful if DegradationKind.QUALITY_RISK not in o.degradation]
            preferred_kv_full_gpu = [(o, r) for o, r in preferred_kv if r.candidate.ngl == "all"]
            quality_pair = max(preferred_kv_full_gpu, key=lambda x: (x[0].context, x[1].metrics.tg_tps or 0.0), default=None)
            context_pair = max(successful, key=lambda x: (x[0].context, x[1].metrics.tg_tps or 0.0))
            fastest_tg = max(float(r.metrics.tg_tps or 0.0) for _, r in successful)
            fast_pool = [(o, r) for o, r in successful
                         if fastest_tg - float(r.metrics.tg_tps or 0.0)
                         <= decode_noise_threshold(float(r.metrics.tg_tps or 0.0), fastest_tg,
                                                   self.noise_policy, conservative=True)]
            speed_pair = max(fast_pool, key=lambda x: (x[1].metrics.pp_tps or 0.0, x[1].metrics.vram_free_min_mb or 0))
            self._emit("  measured Dense solution frontier (SCOUT; selected branch will be fully validated):")
            if quality_pair is not None:
                o, r = quality_pair
                self._emit(f"    MAX_KV_PRECISION_FULL_GPU: ctx={o.context} KV={o.kv_k}/{o.kv_v} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB")
            o, r = context_pair
            self._emit(f"    MAX_CONTEXT (SCOUT): ctx={o.context} KV={o.kv_k}/{o.kv_v} placement={r.candidate.ngl} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f}")
            o, r = speed_pair
            self._emit(f"    FASTEST (SCOUT): ctx={o.context} KV={o.kv_k}/{o.kv_v} | PP={r.metrics.pp_tps:.0f} | TG={r.metrics.tg_tps:.1f} | free={r.metrics.vram_free_min_mb} MiB")
            self._emit(f"    OPTIMAL (SCOUT): ctx={winner_opt.context} KV={winner_opt.kv_k}/{winner_opt.kv_v}")
        else:
            exact_runtime = next((r for o, r in successful if o.exact_target), None)
            exact_n = exact_runtime.candidate.ncmoe if exact_runtime is not None else None
            self._emit("  measured MoE regime frontier (SCOUT; context only matters when residency changes):")
            for o, r in sorted(successful, key=lambda item: (-item[0].context, item[0].recommended_rank)):
                n = r.candidate.ncmoe
                gained = (int(exact_n) - int(n)) if exact_n is not None and n is not None else None
                residency = f"expert layers returned to GPU={gained:+d}" if gained is not None else "residency delta=unknown"
                self._emit(
                    f"    ctx={o.context} KV={o.kv_k}/{o.kv_v} ncmoe={n} | {residency} | "
                    f"PP={r.metrics.pp_tps or 0:.0f} | TG={r.metrics.tg_tps or 0:.1f} | "
                    f"free={r.metrics.vram_free_min_mb} MiB"
                )
            self._emit(
                "  MoE funnel continues only from the selected context/KV/ncmoe regime; "
                "Phase 2 jointly screens ubatch endpoints and one adjacent expert placement."
            )
        ordered: list[SolutionOption] = [winner_opt]
        # Keep other successful probes as fallbacks if the winner later fails staged/long validation.
        for option, _ in successful:
            if option.name != winner_opt.name:
                ordered.append(option)
        # Preserve unprobed architecture-aware fallbacks, but do not retry options already rejected
        # by a real quick probe in the same session.
        probed_names = {o.name for o, _ in successful} | failed_names
        ordered.extend(o for o in options if o.name not in probed_names)
        return ordered

    def tune(self, target_ctx: int, mtp_mode: str = "auto",
             solution_options: list[SolutionOption] | None = None) -> list[CandidateResult]:
        cores = max(1, self.hardware.physical_cores)
        self._declared_target_ctx = int(target_ctx)
        self._emit(
            f"\nAutotune budget: max {self.max_runs} candidate runs / {self.max_minutes} min. "
            f"Preferred VRAM reserve: {self.vram_margin_mb} MiB. "
            f"Absolute VRAM floor: {self.absolute_vram_floor_mb} MiB."
            + (" Preferred reserve is STRICT for recommendations." if self.require_preferred_vram_reserve else "")
        )

        # v0.4+ target-aware semantics: contexts/KV/capabilities are supplied as explicit
        # solution options. A lower context or reduced KV cache is never silently treated
        # as the same target. v0.5.4 adds a cheap solution-level reconnaissance when the
        # preferred exact target already needs a trade-off, so static ranking is not mistaken
        # for measured performance.
        attempts: list[tuple[SolutionOption | None, Candidate]] = []
        if solution_options is not None:
            declared = next((o for o in solution_options if o.exact_target), None)
            if declared is not None:
                self._declared_target_ctx = int(declared.context)
                self._declared_kv = (declared.kv_k, declared.kv_v)
            solution_options = self._recon_solution_options(list(solution_options), cores)
            self._solution_options_ordered = list(solution_options)
            if self._startup_blocker:
                self.target_status = "UNRESOLVED"
                self.stop_reason = "MODEL_STARTUP_FAILED"
                self._emit(
                    "\nAutotune stopped before configuration ranking: repeated split-GGUF startup "
                    "failure. Inspect the candidate server log and verify all shards/llama.cpp support."
                )
                return self.results
            for option in solution_options:
                # An infeasible exact option may be retained as the declared target so recon can
                # derive the correct context/KV/Vision frontier.  It is metadata, never a launch.
                if option.resource_class == ResourceClass.INFEASIBLE:
                    continue
                attempts.append((option, option.to_candidate(cores=cores, extra_args=self.base_extra_args)))
        else:
            attempts.append((None, Candidate(
                ctx=target_ctx, ngl="all", batch=512, ubatch=256,
                threads=cores, threads_batch=cores, kv_k="f16", kv_v="f16",
                extra_args=list(self.base_extra_args),
            )))

        current: Candidate | None = None
        for option, seed in attempts:
            if not self.budget_ok():
                return self.results
            if option is None:
                self._emit(f"\nTrying exact target context: {seed.ctx}")
            else:
                deg = ",".join(d.value for d in option.degradation) if option.degradation else "none"
                self._emit(
                    f"\nTrying solution option #{option.recommended_rank}: {option.name} | "
                    f"ctx={seed.ctx} KV={seed.kv_k}/{seed.kv_v} strategy={option.strategy} "
                    f"degradation={deg}"
                )
                for note in option.degradation_notes:
                    self._emit(f"  trade-off disclosure: {note}")
            attempt_start = len(self.results)
            preserve_full_gpu = bool(
                self.model.kind == ModelKind.DENSE and option is not None
                and option.strategy.startswith("full-gpu")
            )
            current = self._search_placement(seed, preserve_full_gpu=preserve_full_gpu)
            if current is not None:
                self.provisional_recommendation_key = current.key()
                anchor = self._best_exact_result(current, full_only=True)
                # anchor is None both when a candidate genuinely failed and when its FULL
                # confirmation was deliberately deferred to a later phase (e.g. MoE placement's
                # "defer expensive FULL to the joint ubatch/placement winner"). Only reject on an
                # actual measured shortfall; a merely-not-yet-confirmed candidate must be allowed
                # to reach that later phase instead of being abandoned for "no full benchmark result".
                if (self.min_tg_tps is not None or self.min_pp_tps is not None) and anchor is not None \
                        and not self._meets_minimum_performance(anchor):
                    self._emit(
                        "  option is technically runnable but does not satisfy the user's minimum performance target: "
                        + self._minimum_performance_text(anchor)
                    )
                    # Keep the measurements for diagnostics, but prevent a technically-runnable/too-slow
                    # option from reappearing later as FASTEST/OPTIMAL merely because it passed llama.cpp.
                    for measured in self.results[attempt_start:]:
                        if measured.status == RunStatus.PASS:
                            measured.status = RunStatus.PASS_DEGRADED
                            measured.reason = "TARGET_PERFORMANCE_NOT_MET"
                    current = None
                    continue
                if option is None:
                    self.selected_option = None
                    self.target_status = "SATISFIED"
                else:
                    self._set_selected_candidate_status(current)
                break
            if self._startup_blocker:
                self.target_status = "UNRESOLVED"
                self.stop_reason = "MODEL_STARTUP_FAILED"
                self._emit(
                    "  stopping solution fallback: repeated split-GGUF startup failure is not "
                    "evidence that another context or KV precision is feasible or infeasible."
                )
                break
            if option is not None:
                self._emit(f"  option {option.name} could not be validated; moving to the next explicitly disclosed alternative.")

        if current is None:
            if self._startup_blocker:
                self.target_status = "UNRESOLVED"
                self.stop_reason = "MODEL_STARTUP_FAILED"
            else:
                self.target_status = "NOT_FEASIBLE"
                if self.stop_reason == "RUNNING":
                    self.stop_reason = "NO_FEASIBLE_CONFIGURATION"
            return self.results

        # Joint ubatch + placement optimization. The old v0.1.x pipeline optimized these
        # independently and could get stuck at ncmoe=15/ub=512 even though ncmoe=18..19/ub=1536
        # was much faster. From this point onward, every memory-sensitive knob may recover placement.
        self.phase = "JOINT_UBATCH_PLACEMENT"
        base_ref = self._best_exact_result(current, full_only=True)
        moe_screen_refined = False
        if self.model.kind == ModelKind.DENSE and current.ngl == "all" and self.search_mode != "deep":
            joint_non_mtp = self._dense_full_gpu_ubatch_search(current, reference=base_ref)
        elif (self.model.kind == ModelKind.DENSE and current.ngl != "all"
              and self._dense_oversized_active and self.search_mode != "deep"):
            joint_non_mtp = self._dense_partial_ubatch_search(current, reference=base_ref)
        elif self.model.kind == ModelKind.MOE and self.search_mode != "deep":
            joint_non_mtp = self._moe_screen_ubatch_placement_search(
                current, "Phase 2 / NON_MTP", reference=base_ref
            )
            moe_screen_refined = self._is_recommendable_full(joint_non_mtp)
        else:
            joint_non_mtp = self._joint_ubatch_placement_search(
                current, "Phase 2 / NON_MTP", reference=base_ref
            )
        if self._is_recommendable_full(joint_non_mtp):
            current = copy.deepcopy(joint_non_mtp.candidate)
            self.provisional_recommendation_key = current.key()

        # CPU batch threads matter little when a Dense target model is fully GPU-resident. In
        # NORMAL/QUICK, skip three extra server launches; DEEP still measures them explicitly.
        if self.model.kind == ModelKind.DENSE and current.ngl == "all" and self.search_mode != "deep":
            self._emit("\n[Phase 3] CPU thread search skipped: Dense ngl=all is GPU-bound; keep existing t/tb.")
        elif (self.model.kind == ModelKind.DENSE and current.ngl != "all"
              and self._dense_oversized_active and self.search_mode != "deep"):
            self._emit("\n[Phase 3] CPU thread sweep skipped in NORMAL oversized-Dense: placement/context dominates; DEEP can measure t/tb explicitly.")
        elif self.model.kind == ModelKind.MOE and self.search_mode != "deep":
            self._emit("\n[Phase 3] CPU thread search skipped in NORMAL: MoE joint SCREEN already optimized the dominant ubatch/expert-residency trade-off; DEEP can sweep t/tb explicitly.")
        else:
            self.phase = "THREAD_SEARCH"
            self._emit("\n[Phase 3] CPU thread search")
            thread_probes: list[CandidateResult] = []
            thread_values = ([cores, self.hardware.logical_cores] if self.search_mode == "quick"
                             else [max(1, cores//2), cores, self.hardware.logical_cores])
            current_ref = self._best_exact_result(current, full_only=True)
            for tb in sorted(set(thread_values)):
                if not self.budget_ok():
                    return self.results
                c = copy.deepcopy(current)
                c.threads_batch = tb
                r = self._run(c, quick=True, phase="THREAD_SEARCH", reference=current_ref)
                if self._is_good(r):
                    thread_probes.append(r)
            if thread_probes:
                best_thread_probe = choose_preferred(thread_probes, self.workload_profile, self.noise_policy)
                baseline_probe = next((r for r in thread_probes if r.candidate.threads_batch == current.threads_batch), None)
                material = bool(
                    best_thread_probe is not None and baseline_probe is not None
                    and best_thread_probe.candidate.threads_batch != current.threads_batch
                    and (
                        decode_relation(best_thread_probe, baseline_probe, self.noise_policy) > 0
                        or prefill_relation(best_thread_probe, baseline_probe, self.workload_profile, self.noise_policy) > 0
                        or latency_relation(best_thread_probe, baseline_probe, self.workload_profile, self.noise_policy) > 0
                    )
                )
                if material and best_thread_probe is not None:
                    self._emit(
                        f"  tb={best_thread_probe.candidate.threads_batch} has a material noise-adjusted gain; full-confirming."
                    )
                    full_thread = self._guarded_full(
                        copy.deepcopy(best_thread_probe.candidate), "THREAD_CONFIRM", current_ref
                    )
                    if self._is_recommendable_full(full_thread):
                        current = copy.deepcopy(full_thread.candidate)
                else:
                    self._emit(f"  keeping tb={current.threads_batch}; alternatives are inside the noise zone.")

        # Small placement neighborhood at the winning non-MTP ubatch. This protects against noise
        # in the recovery walk while avoiding a second broad placement search.
        self.phase = "PLACEMENT_LOCAL_REFINE"
        self._emit("\n[Phase 4] Local placement refinement at the winning ubatch")
        local_ref = self._best_exact_result(current, full_only=True)
        local_results: list[CandidateResult] = [local_ref] if local_ref else []
        if self.model.kind == ModelKind.MOE and current.ncmoe is not None:
            if self.search_mode != "deep" and moe_screen_refined:
                self._emit(
                    "  MoE local placement FULL sweep skipped in NORMAL: adjacent ncmoe was already "
                    "screened before the single Phase-2 confirmation."
                )
            else:
                blocks = model_main_block_count(self.model)
                for n in sorted({current.ncmoe - 1, current.ncmoe + 1}):
                    if not self.budget_ok() or not (0 <= n <= blocks):
                        continue
                    c = copy.deepcopy(current); c.ncmoe = n
                    r = self._guarded_full(c, "PLACEMENT_LOCAL_REFINE", local_ref)
                    if self._is_recommendable_full(r):
                        local_results.append(r)
        elif self.model.kind == ModelKind.DENSE:
            total = model_main_block_count(self.model)
            if current.ngl == "all":
                self._emit(
                    "  Dense ngl=all is already the maximum target-model placement; "
                    "skip numeric local refinement because numeric ngl is not equivalent to `all`."
                )
            else:
                numeric = int(current.ngl)
                if self._dense_oversized_active and self.search_mode != "deep":
                    self._emit(
                        "  oversized Dense local layer sweep skipped in NORMAL: Phase 0 already mapped numeric "
                        "placement across contexts and Phase 1 FULL-confirmed the selected ngl."
                    )
                else:
                    for n in sorted({max(0, numeric - 1), min(total, numeric + 1)}):
                        if not self.budget_ok():
                            continue
                        c = copy.deepcopy(current); c.ngl = n
                        r = self._guarded_full(c, "PLACEMENT_LOCAL_REFINE", local_ref)
                        if self._is_recommendable_full(r):
                            local_results.append(r)
        if local_results:
            best_local = max(local_results, key=self._perf_score)
            current = copy.deepcopy(best_local.candidate)
            self._emit(f"  non-MTP local winner: {current.short()}")

        # The performance optimum is intentionally close to the VRAM boundary. Before speculative
        # tuning, measure whether useful headroom can reach the user's preferred reserve.
        safe_reference = self._best_exact_result(current, full_only=True)
        self._find_safe_reserve_candidate(current, reference=safe_reference)

        # MTP is a coupled memory/performance dimension, but Dense and MoE need different
        # recovery policies. MoE may safely trade routed-expert residency for VRAM; Dense should
        # preserve target-model full offload and spend memory first on draft depth/ubatch.
        mtp_supported = self.model.has_mtp and self.caps.supports("--spec-type")
        best_mtp: CandidateResult | None = None
        if mtp_mode == "on" and not mtp_supported:
            self._emit("\nMTP was explicitly requested, but the model/build does not expose draft-mtp support.")
        elif mtp_mode != "off" and mtp_supported and self.budget_ok():
            non_mtp_ref = self._best_exact_result(current, full_only=True)

            if self.model.kind == ModelKind.DENSE:
                self.phase = "DENSE_MTP_SEARCH"
                best_mtp = self._dense_mtp_full_gpu_search(
                    current, reference=non_mtp_ref, force_probe=(mtp_mode == "on")
                )
                memory_outcomes = {
                    "MEMORY_TIGHT_REFERENCE", "MEMORY_STATIC", "MEMORY_PROBE", "MEMORY_CONFIRM",
                }
                mtp_below_preferred = bool(
                    self._is_recommendable_full(best_mtp)
                    and int(best_mtp.metrics.vram_free_min_mb or 0) < self.vram_margin_mb
                )
                if (
                    non_mtp_ref is not None and self.budget_ok()
                    and (
                        (best_mtp is None and self._last_dense_mtp_outcome in memory_outcomes)
                        or mtp_below_preferred
                    )
                ):
                    rescued_mtp = self._dense_mtp_kv_rescue(
                        current, non_mtp_ref, best_mtp, force_probe=(mtp_mode == "on"),
                    )
                    if self._is_recommendable_full(rescued_mtp):
                        best_mtp = rescued_mtp
            else:
                self.phase = "MTP_JOINT_SEARCH"
                if self.search_mode != "deep":
                    best_mtp = self._moe_mtp_screen_search(
                        current, non_mtp_ref, force_expand=(mtp_mode == "on")
                    )
                else:
                    self._emit(
                        "\n[Phase 5] DEEP MoE MTP joint search: enable n-max=8/p-min=0.8, then re-tune ubatch + placement."
                    )
                    mtp_seed = copy.deepcopy(current)
                    mtp_seed.mtp = True
                    mtp_seed.mtp_n_max = 8
                    mtp_seed.mtp_p_min = 0.8
                    best_mtp = self._joint_ubatch_placement_search(
                        mtp_seed, "Phase 5 / MTP8", reference=non_mtp_ref
                    )

                    # Broad n-max refinement is a DEEP-only cost.
                    if self._is_recommendable_full(best_mtp) and self.budget_ok():
                        self._emit(
                            f"\n[Phase 5b] DEEP MTP n-max refinement around joint winner: {best_mtp.candidate.short()}"
                        )
                        nmax_results = [best_mtp]
                        for nmax in [4, 16]:
                            if not self.budget_ok():
                                break
                            c = copy.deepcopy(best_mtp.candidate)
                            c.mtp_n_max = nmax
                            r = self._recover_full(c, f"MTP_NMAX_{nmax}", reference=best_mtp)
                            if self._is_recommendable_full(r):
                                nmax_results.append(r)
                        best_mtp = choose_preferred(nmax_results, self.workload_profile, self.noise_policy) or nmax_results[0]
                        self._emit(
                            f"  best DEEP MTP n-max after FULL checks: {best_mtp.candidate.mtp_n_max}, "
                            f"ncmoe={best_mtp.candidate.ncmoe}, ub={best_mtp.candidate.ubatch}, "
                            f"TG={best_mtp.metrics.tg_tps:.1f} t/s"
                        )

            # p-min affects predictability more than target-model placement. Run the same local sweep
            # only when an MTP candidate is still competitive after the architecture-specific search.
            dense_pmin_ok = True
            if self.model.kind == ModelKind.DENSE and best_mtp is not None and non_mtp_ref is not None:
                dense_pmin_ok = (
                    decode_relation(best_mtp, non_mtp_ref, self.noise_policy) > 0
                    or latency_relation(best_mtp, non_mtp_ref, self.workload_profile, self.noise_policy) > 0
                    or self.search_mode == "deep"
                )
                if not dense_pmin_ok:
                    self._emit(
                        "  skip Dense p-min sweep: confirmed MTP is not materially better than non-MTP "
                        "outside the noise zone."
                    )
            moe_pmin_ok = not (
                self.model.kind == ModelKind.MOE
                and self._mtp_speed_only
                and self.search_mode != "deep"
            )
            if not moe_pmin_ok:
                self._emit(
                    "  skip MoE p-min sweep: MTP was retained only as a confirmed FASTEST branch, "
                    "not as an end-to-end workload winner."
                )
            if (self._is_recommendable_full(best_mtp) and dense_pmin_ok and moe_pmin_ok
                    and self.search_mode != "quick" and self.budget_ok()):
                self._emit("\n[Phase 5c] MTP p-min sparse sweep: baseline 0.8 is already FULL; probe only 0.7/0.9")
                p_probes: list[CandidateResult] = []
                pmin_values = [0.7] if (self.model.kind == ModelKind.MOE and self.search_mode == "normal") else [0.7, 0.9]
                for pmin in pmin_values:
                    if not self.budget_ok():
                        break
                    c = copy.deepcopy(best_mtp.candidate)
                    c.mtp_p_min = pmin
                    r = self._run(c, quick=True, phase="MTP_PMIN_PROBE", reference=best_mtp)
                    if self._is_good(r):
                        if self._vram_class(r) in {VramOperatingClass.REJECT, VramOperatingClass.FRAGILE}:
                            self._emit(
                                f"  p-min={pmin:g} is {self._vram_class(r).value} at "
                                f"{r.metrics.vram_free_min_mb} MiB; it cannot earn FULL."
                            )
                            continue
                        p_probes.append(r)
                        if (pmin == 0.7 and self.search_mode != "deep" and (
                                decode_relation(r, best_mtp, self.noise_policy) > 0
                                or latency_relation(r, best_mtp, self.workload_profile, self.noise_policy) > 0)):
                            self._emit(
                                "  p-min=0.7 already shows a material gain; skip 0.9 in NORMAL and "
                                "spend the next run on FULL confirmation instead."
                            )
                            break
                if p_probes:
                    p_winner = choose_preferred(p_probes, self.workload_profile, self.noise_policy) or p_probes[0]
                    material = (
                        decode_relation(p_winner, best_mtp, self.noise_policy) > 0
                        or latency_relation(p_winner, best_mtp, self.workload_profile, self.noise_policy) > 0
                    )
                    if material:
                        c = copy.deepcopy(p_winner.candidate)
                        self._emit(
                            f"  p-min={c.mtp_p_min:g} shows a material noise-adjusted gain; FULL-confirming once."
                        )
                        # Dense p-min confirmation must keep exact full-GPU placement; MoE may
                        # still use adaptive expert recovery.
                        if self.model.kind == ModelKind.DENSE:
                            confirmed = self._run(c, quick=False, phase="MTP_PMIN_CONFIRM", reference=best_mtp)
                        else:
                            confirmed = self._recover_full(c, "MTP_PMIN_CONFIRM", reference=best_mtp)
                        if confirmed is not None and self._is_recommendable_full(confirmed):
                            material_full = (
                                decode_relation(confirmed, best_mtp, self.noise_policy) > 0
                                or latency_relation(confirmed, best_mtp, self.workload_profile, self.noise_policy) > 0
                            )
                            if material_full:
                                best_mtp = confirmed
                                self._emit(
                                    f"  confirmed p-min={confirmed.candidate.mtp_p_min:g} remains materially better; "
                                    "promote it to the final MTP candidate."
                                )
                            else:
                                self._emit(
                                    f"  confirmed p-min={confirmed.candidate.mtp_p_min:g} no longer clears the noise boundary; "
                                    f"keep p-min={best_mtp.candidate.mtp_p_min:g}."
                                )
                    else:
                        self._emit("  p-min alternatives are inside the noise zone; keep 0.8 without another FULL run.")

            if self._is_recommendable_full(best_mtp):
                self._preferred_mtp_key = best_mtp.candidate.key()

        # Final long-context validation. Only these candidates may earn HIGH confidence / MAX_CONTEXT.
        self.phase = "FINAL_VALIDATION"
        self._emit("\n[Phase 6] Final validation of distinct top candidates at a materially longer prompt")
        validation = self._select_validation_candidates(current.ctx)
        if self.search_mode == "quick":
            validation = validation[:1]
        if not validation:
            self.stop_reason = "NO_FULL_CANDIDATE_FOR_VALIDATION"
            return self.results
        final_attempts: list[CandidateResult] = []
        excluded_final_keys: set[str] = set()
        for c in validation:
            if not self.budget_ok():
                return self.results
            excluded_final_keys.add(c.key())
            final_ref = self._best_exact_result(c, full_only=True)
            final_attempts.append(
                self._run(c, quick=False, phase="FINAL_VALIDATION", long_validate=True, reference=final_ref)
            )

        validated = [
            r for r in self.results
            if self._is_recommendable_full(r)
            and r.metrics.benchmark_kind == "validation" and r.metrics.long_context_passed
        ]
        if not validated and final_attempts and not any(
            self._environmental_final_failure(r) for r in final_attempts
        ) and self.budget_ok():
            fallback_validated, fallback_candidate = self._run_final_fallbacks(
                final_attempts[0], excluded_final_keys,
            )
            validated.extend(fallback_validated)
            if fallback_candidate is not None:
                current = fallback_candidate
                self._set_selected_candidate_status(current)
        if validated:
            final_frontier = pareto_frontier(validated, self.workload_profile, self.noise_policy) or validated
            final_best = choose_preferred(final_frontier, self.workload_profile, self.noise_policy) or final_frontier[0]
            self._emit(
                "  post-FINAL re-ranking: stronger robustness/context measurements replace short-run priors; "
                f"preferred {self.workload_profile} candidate={final_best.candidate.short()} | "
                f"effective prefill={profile_prefill_tps(final_best, self.workload_profile):.0f} t/s | "
                f"effective decode={profile_decode_tps(final_best, self.workload_profile):.1f} t/s | "
                f"cycle≈{workload_latency_seconds(final_best, self.workload_profile):.2f}s."
            )
        else:
            self.completed = False
            self.stop_reason = "NO_RECOMMENDABLE_FINAL_CANDIDATE"
            self._emit(
                "  FINAL produced no long-context candidate above the recommendation floor. "
                "The measured runs remain in the report, but the session is not marked completed."
            )
            return self.results

        self.completed = True
        if self.target_status == "SATISFIED":
            self.stop_reason = "COMPLETED_EXACT_TARGET"
        elif self.target_status == "SATISFIED_WITH_KV_PRECISION_TRADEOFF":
            self.stop_reason = "COMPLETED_WITH_KV_PRECISION_TRADEOFF"
        elif self.target_status == "SATISFIED_WITH_PERFORMANCE_TRADEOFF":
            self.stop_reason = "COMPLETED_WITH_PERFORMANCE_TRADEOFF"
        elif self.target_status == "ALTERNATIVE_CAPABILITY_REDUCED":
            self.stop_reason = "COMPLETED_ALTERNATIVE_TARGET"
        else:
            self.stop_reason = "COMPLETED"
        self.phase = "DONE"
        self._emit(
            "\nAutotune search completed and final candidates were long-context validated. "
            f"Target status: {self.target_status}."
        )

        self._discover_max_context_upsize(current)
        return self.results

    def _discover_max_context_upsize(self, current: Candidate) -> None:
        """Quick-probe discovery-only "upsize" candidates so MAX_CONTEXT can show a real ceiling.

        When the requested target was comfortably full-GPU, the main search never tries anything
        *larger* than what the user asked for (every other envelope family only shrinks context or
        degrades KV), so MAX_CONTEXT would otherwise just duplicate OPTIMAL even with plainly unused
        VRAM headroom. target.py's build_feasibility_plan generates a bounded two-tier doubling
        ladder (Q8 first, Q4 continuing only where Q8 hit a VRAM ceiling) up to the model's native
        context for exactly this case; probe it here with any remaining budget.

        A rung that is runnable but lands below the preferred reserve (FRAGILE, not the genuinely
        SAFE ceiling MAX_CONTEXT should report) is not simply accepted or discarded: one bounded
        bisection refine (reusing the same _runtime_bracket_context interpolation already proven
        for live FINAL context repair) is tried between it and the last SAFE point in the *same* KV
        tier, so MAX_CONTEXT reports the real knee for that tier instead of stopping short at a
        coarse doubling step. Purely additive measurement -- never touches current/selected_option/
        target_status, so it cannot change what OPTIMAL/EXACT_TARGET is.
        """
        if self.model.kind != ModelKind.DENSE or current.ngl != "all" or not self.budget_ok():
            return
        upsize_options = sorted(
            (o for o in self._solution_options_ordered if o.strategy == "full-gpu-context-upsize"),
            key=lambda o: o.context,
        )
        if not upsize_options:
            return
        cores = max(1, self.hardware.physical_cores)
        self._emit(
            "\n[Phase 7] MAX_CONTEXT discovery: probing safe Q8/Q4 context growth beyond the "
            "requested target while comfortable full-GPU headroom allows it."
        )
        last_safe_by_kv: dict[tuple[str, str], tuple[Candidate, int]] = {}
        for opt in upsize_options:
            if not self.budget_ok():
                break
            seed = opt.to_candidate(cores=cores, extra_args=self.base_extra_args)
            kv_key = (seed.kv_k, seed.kv_v)
            probe = self._run(seed, quick=True, phase="MAX_CONTEXT_DISCOVERY", recon=True)
            free = int(probe.metrics.vram_free_min_mb or 0)
            if not self._is_good(probe) or free < self._vram_thresholds(seed).hard_floor_mb:
                self._emit(f"  MAX_CONTEXT discovery stopped at ctx={seed.ctx}: {probe.reason}.")
                break
            if free >= self.vram_margin_mb:
                self._emit(f"  MAX_CONTEXT discovery: ctx={seed.ctx} confirmed runnable, {self._summary(probe)}.")
                last_safe_by_kv[kv_key] = (copy.deepcopy(seed), free)
                continue

            self._emit(
                f"  MAX_CONTEXT discovery: ctx={seed.ctx} runnable but only {free} MiB free "
                f"(below the {self.vram_margin_mb} MiB preferred reserve); refine within this KV tier "
                "instead of accepting a fragile ceiling or jumping to the next tier."
            )
            prior = last_safe_by_kv.get(kv_key)
            if prior is not None:
                low_ctx, low_free = prior[0].ctx, prior[1]
                high_ctx, high_free = seed.ctx, free
                # Iterative, bounded bisection -- not just one shot. A single linear interpolation
                # between two endpoints that are far apart (e.g. 131072/262144) can land on a point
                # that is *still* not SAFE if the real VRAM-vs-context curve isn't perfectly linear
                # across that whole span -- observed live: two misses of only 36-54 MiB before a
                # third attempt finally cleared the reserve. guard_mb=200 (not the small 16 MiB used
                # for context *repair* elsewhere, which is recovering a specific known-fragile point)
                # deliberately aims a bit past the reserve so a single interpolation usually lands
                # SAFE on the first try, trading a modest amount of context for reliably converging
                # in one round instead of three. Each miss still narrows the bracket (the new
                # FRAGILE point replaces the high/fragile end) and re-interpolates as a bounded
                # fallback, converging toward the true knee instead of giving up after one attempt.
                for _ in range(3):
                    if not self.budget_ok():
                        break
                    refined_ctx = self._runtime_bracket_context(
                        high_context=high_ctx, high_free_mb=high_free,
                        low_context=low_ctx, low_free_mb=low_free,
                        target_free_mb=self.vram_margin_mb, guard_mb=200,
                    )
                    if refined_ctx is None:
                        break
                    refined = copy.deepcopy(seed)
                    refined.ctx = refined_ctx
                    refined_probe = self._run(refined, quick=True, phase="MAX_CONTEXT_DISCOVERY", recon=True)
                    if not self._is_good(refined_probe):
                        break
                    refined_free = int(refined_probe.metrics.vram_free_min_mb or 0)
                    if refined_free >= self.vram_margin_mb:
                        self._emit(
                            f"  MAX_CONTEXT discovery: refined ctx={refined_ctx} confirmed SAFE, "
                            f"{self._summary(refined_probe)}."
                        )
                        last_safe_by_kv[kv_key] = (copy.deepcopy(refined), refined_free)
                        break
                    self._emit(
                        f"  MAX_CONTEXT discovery: refined ctx={refined_ctx} still not SAFE "
                        f"({refined_free} MiB); narrowing further."
                    )
                    high_ctx, high_free = refined_ctx, refined_free
            # This KV tier does not grow further after a FRAGILE miss; continue to the next
            # envelope option (e.g. the next KV tier target.py already decided was worth trying).

        if not last_safe_by_kv or not self.budget_ok():
            return

        # Every point recorded above is only a short recon probe (~1-2K tokens): the same class
        # of short-scout evidence that Phase 6 FINAL_VALIDATION exists specifically to not trust
        # for the *requested* target, because a filled context can still cross the VRAM floor a
        # short probe never reached (observed live: a recon-level "OPERATIONAL" ctx later measured
        # FRAGILE once FINAL actually filled the context). MAX_CONTEXT must not be held to a lower
        # evidentiary bar than every other reported profile, so FINAL-validate the single winning
        # candidate (the largest safe context across KV tiers) the same way Phase 6 does.
        best_kv, (best_candidate, best_free) = max(
            last_safe_by_kv.items(), key=lambda item: item[1][0].ctx,
        )
        self._emit(
            f"\n[Phase 7] MAX_CONTEXT FINAL validation: confirming ctx={best_candidate.ctx} "
            f"{best_kv[0]}/{best_kv[1]} at a materially longer prompt before it can outrank the "
            "short discovery probe."
        )
        final = self._run(
            copy.deepcopy(best_candidate), quick=False, phase="MAX_CONTEXT_FINAL_VALIDATION",
            long_validate=True,
        )
        if self._is_recommendable_full(final) and final.metrics.long_context_passed:
            self._emit(f"  MAX_CONTEXT FINAL validation confirmed: {self._summary(final)}.")
            return

        self._emit(
            f"  MAX_CONTEXT FINAL validation did not hold at ctx={best_candidate.ctx}: "
            f"{final.reason}. Repairing context with the same KV/Vision/full-GPU semantics "
            "instead of leaving the unvalidated recon probe as the reported ceiling."
        )
        repaired = self._repair_fragile_dense_full_gpu_context(best_candidate, final)
        if repaired is not None:
            self._emit(f"  MAX_CONTEXT repaired to ctx={repaired.ctx}.")
        else:
            self._emit(
                "  MAX_CONTEXT could not be repaired to a FULL-confirmed context above the "
                "discovery ceiling; the report will fall back to the last FINAL-validated profile."
            )
