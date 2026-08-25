from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from pathlib import Path
from typing import Iterable

from llama_autotuner.models import Candidate, HardwareInfo, ModelInfo, ModelKind
from llama_autotuner.tuning.kv import kv_degradation_ladder, kv_precision
from llama_autotuner.tuning.static_memory import estimate_candidate_free_mb, estimate_static_memory, model_main_block_count


class VisionRequirement(str, Enum):
    OFF = "off"
    REQUIRED = "required"
    AUTO = "auto"


class DegradationPolicy(str, Enum):
    STRICT = "strict"
    REPORT = "report"
    AUTO = "auto"


class ResourceClass(str, Enum):
    COMFORTABLE = "COMFORTABLE"
    CONSTRAINED = "CONSTRAINED"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"


class CapabilityState(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_BY_MODEL = "UNSUPPORTED_BY_MODEL"
    MISSING_REQUIRED_COMPONENT = "MISSING_REQUIRED_COMPONENT"
    NOT_REQUESTED = "NOT_REQUESTED"
    UNKNOWN = "UNKNOWN"




# Some llama.cpp multimodal families store Vision in a separate mmproj while the main
# text GGUF keeps a generic architecture name and may not expose explicit Vision keys.
# Qwen3.5/Qwen3.8 GGUFs are a current example: the text model is qwen35/qwen35moe and
# the companion projector carries the image encoder. The projector still must be
# supplied and llama.cpp runtime-validates embedding compatibility.
_SPLIT_MULTIMODAL_ARCHITECTURES = {"qwen35", "qwen35moe"}


def model_supports_split_vision(model: ModelInfo) -> bool:
    arch = str(model.architecture or "").strip().lower()
    return bool(model.has_vision_hint or arch in _SPLIT_MULTIMODAL_ARCHITECTURES)

class DegradationKind(str, Enum):
    NONE = "none"
    PERFORMANCE = "performance"
    # Kept as QUALITY_RISK internally for source compatibility.  The serialized
    # value names the real axis: only KV-cache/attention precision changes here;
    # model-weight quantization is fixed by the GGUF selected by the user.
    QUALITY_RISK = "kv-cache-precision-risk"
    CAPABILITY = "capability"


@dataclass(slots=True)
class TargetSpec:
    context: int
    workload: str = "agent"
    priority: str = "balanced"
    vision: VisionRequirement = VisionRequirement.OFF
    mmproj: str | None = None
    degradation_policy: DegradationPolicy = DegradationPolicy.AUTO
    preferred_kv_k: str = "f16"
    preferred_kv_v: str = "f16"
    allow_kv_degradation: bool = True
    allow_context_reduction: bool = True
    preferred_vram_reserve_mb: int = 1024
    absolute_vram_floor_mb: int = 300
    min_tg_tps: float | None = None
    min_pp_tps: float | None = None
    # Expert-mode locks. None means optimizer may tune the dimension.
    lock_ngl: str | int | None = None
    lock_kv: bool = False
    tune_placement: bool = True
    tune_batch: bool = True
    tune_mtp: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vision"] = self.vision.value
        data["degradation_policy"] = self.degradation_policy.value
        return data


@dataclass(slots=True)
class CapabilityAnalysis:
    vision: CapabilityState
    mtp: CapabilityState
    native_context: int | None
    architecture: str | None
    model_kind: str
    notes: list[str] = field(default_factory=list)

    @property
    def target_blocked(self) -> bool:
        return self.vision in {
            CapabilityState.UNSUPPORTED_BY_MODEL,
            CapabilityState.MISSING_REQUIRED_COMPONENT,
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vision"] = self.vision.value
        data["mtp"] = self.mtp.value
        return data


@dataclass(slots=True)
class SolutionOption:
    name: str
    context: int
    kv_k: str
    kv_v: str
    strategy: str
    predicted_free_mb: int | None
    predicted_placement: str | int | None
    resource_class: ResourceClass
    degradation: list[DegradationKind] = field(default_factory=list)
    degradation_notes: list[str] = field(default_factory=list)
    recommended_rank: int = 999
    exact_target: bool = False
    vision_required: bool = False
    mmproj: str | None = None

    @property
    def quality_risk(self) -> bool:
        return DegradationKind.QUALITY_RISK in self.degradation

    @property
    def kv_precision_risk(self) -> bool:
        return DegradationKind.QUALITY_RISK in self.degradation

    @property
    def capability_loss(self) -> bool:
        return DegradationKind.CAPABILITY in self.degradation

    def to_candidate(self, *, cores: int, extra_args: list[str] | None = None) -> Candidate:
        ngl: str | int = "all"
        ncmoe: int | None = None
        if self.strategy.startswith("dense-cpu-offload") and self.predicted_placement is not None:
            ngl = self.predicted_placement
        if self.strategy.startswith("moe-expert-offload") and isinstance(self.predicted_placement, int):
            ncmoe = self.predicted_placement
        return Candidate(
            ctx=self.context,
            ngl=ngl,
            ncmoe=ncmoe,
            batch=512,
            ubatch=256,
            threads=max(1, cores),
            threads_batch=max(1, cores),
            kv_k=self.kv_k,
            kv_v=self.kv_v,
            vision=self.vision_required,
            mmproj=self.mmproj,
            extra_args=list(extra_args or []),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["resource_class"] = self.resource_class.value
        data["degradation"] = [x.value for x in self.degradation]
        return data


@dataclass(slots=True)
class FeasibilityPlan:
    target: TargetSpec
    capabilities: CapabilityAnalysis
    resource_class: ResourceClass
    options: list[SolutionOption]
    exact_likely_feasible: bool
    exact_without_quality_degradation: bool
    summary: str

    @property
    def exact_without_kv_precision_degradation(self) -> bool:
        """Honest name for the compatibility field retained from older JSON schemas."""
        return self.exact_without_quality_degradation

    def to_dict(self) -> dict:
        return {
            "target": self.target.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "resource_class": self.resource_class.value,
            "exact_likely_feasible": self.exact_likely_feasible,
            "exact_without_quality_degradation": self.exact_without_quality_degradation,
            "exact_without_kv_precision_degradation": self.exact_without_kv_precision_degradation,
            "summary": self.summary,
            "options": [x.to_dict() for x in self.options],
        }


def _looks_like_vision_projector(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.is_file() and p.suffix.lower() == ".gguf"


def analyze_capabilities(model: ModelInfo, target: TargetSpec, caps=None) -> CapabilityAnalysis:
    notes: list[str] = []
    if target.vision == VisionRequirement.OFF:
        vision = CapabilityState.NOT_REQUESTED
    elif model_supports_split_vision(model):
        if target.mmproj and not _looks_like_vision_projector(target.mmproj):
            vision = CapabilityState.MISSING_REQUIRED_COMPONENT
            notes.append("Vision is requested but the supplied mmproj path is not a readable GGUF file.")
        elif target.mmproj:
            vision = CapabilityState.SUPPORTED
            if not model.has_vision_hint:
                notes.append(
                    f"Vision capability is inferred from split-multimodal architecture {model.architecture} "
                    "plus the supplied companion mmproj; model/projector compatibility will be runtime-validated."
                )
        else:
            vision = CapabilityState.MISSING_REQUIRED_COMPONENT
            if model.has_vision_hint:
                notes.append("Vision metadata is present, but no --mmproj was supplied; text-only fallback is not allowed for a required capability.")
            else:
                notes.append(
                    f"Architecture {model.architecture} supports split Vision, but no companion mmproj was supplied/found."
                )
    else:
        # A projector file by itself must never manufacture Vision capability for arbitrary model
        # architectures. Only explicit model metadata/template evidence or a known split-multimodal
        # family may pair with an mmproj.
        vision = CapabilityState.UNSUPPORTED_BY_MODEL
        if target.mmproj and _looks_like_vision_projector(target.mmproj):
            notes.append(
                "A projector GGUF was supplied/found, but this model architecture has no known Vision capability; "
                "the projector is ignored for capability purposes."
            )
        else:
            notes.append("The selected GGUF exposes no known Vision capability.")

    mtp = CapabilityState.SUPPORTED if model.has_mtp else CapabilityState.UNSUPPORTED_BY_MODEL
    if model.context_length and target.context > model.context_length:
        notes.append(
            f"Requested context {target.context} exceeds GGUF native context metadata {model.context_length}; "
            "this is a capability-risk request rather than a normal memory-only optimization."
        )
    return CapabilityAnalysis(
        vision=vision,
        mtp=mtp,
        native_context=model.context_length,
        architecture=model.architecture,
        model_kind=model.kind.value,
        notes=notes,
    )


def _resource_class(free_mb: int | None, hardware: HardwareInfo, reserve_mb: int, floor_mb: int) -> ResourceClass:
    if free_mb is None:
        return ResourceClass.UNKNOWN
    if free_mb < floor_mb:
        return ResourceClass.INFEASIBLE
    # Reserve alone is not enough to call the system comfortable. A genuinely loose model should
    # have either ~25% total VRAM free or substantially more than the requested safety reserve.
    comfortable_threshold = max(reserve_mb * 2, int(hardware.vram_total_mb * 0.25))
    if free_mb >= comfortable_threshold:
        return ResourceClass.COMFORTABLE
    return ResourceClass.CONSTRAINED


def _context_alternatives(target_ctx: int) -> list[int]:
    """Return a compact context-capability→speed ladder.

    The old 75/50/25% ladder stopped at 64K for a 256K request. That meant a
    16 GiB GPU could never surface obvious preferred-KV 32K/16K options,
    even though those are often the most useful full-GPU fallbacks. Keep the
    ladder small, aligned to 4K, but continue to 16K for large requested
    contexts. 16K is the automatic floor; users can still request smaller
    contexts explicitly.
    """
    fractions = (0.75, 0.50, 0.25, 0.125, 0.0625)
    auto_floor = 16_384 if target_ctx >= 32_768 else 4_096
    out: list[int] = []
    for fraction in fractions:
        value = int(target_ctx * fraction)
        value = max(auto_floor, (value // 4096) * 4096)
        if value < target_ctx and value not in out:
            out.append(value)
        if value <= auto_floor:
            break
    return out


def _moe_floor_placement(
    model: ModelInfo, hardware: HardwareInfo, baseline_vram_mb: int, candidate: Candidate, floor_mb: int
) -> tuple[int | None, int | None]:
    """Return the smallest ncmoe that clears the hard floor, not the preferred reserve.

    For MoE, lower ncmoe keeps more routed experts on GPU and is normally the
    performance-first direction. The preferred reserve is handled later as a VRAM
    headroom property; it must not silently force extra expert layers onto CPU.
    """
    blocks = model_main_block_count(model)
    last_free: int | None = None
    for n in range(0, blocks + 1):
        c = replace(candidate, ncmoe=n)
        free = estimate_candidate_free_mb(model, hardware, baseline_vram_mb, c)
        last_free = free
        if free is not None and free >= floor_mb:
            return n, free
    return None, last_free


def _kv_ladder(k: str, v: str) -> list[tuple[str, str, str]]:
    # FP16 → Q8 → Q4 are the explicit user-facing levels.  A mixed Q8/Q4
    # point is retained as an intermediate measured VRAM option.
    return kv_degradation_ladder(k, v)


def build_feasibility_plan(
    model: ModelInfo,
    hardware: HardwareInfo,
    baseline_vram_mb: int,
    target: TargetSpec,
    *,
    caps=None,
) -> FeasibilityPlan:
    capabilities = analyze_capabilities(model, target, caps=caps)
    if capabilities.target_blocked:
        return FeasibilityPlan(
            target=target,
            capabilities=capabilities,
            resource_class=ResourceClass.INFEASIBLE,
            options=[],
            exact_likely_feasible=False,
            exact_without_quality_degradation=False,
            summary="Requested capability is unsupported or a required component is missing; benchmarking is intentionally blocked.",
        )

    cores = max(1, hardware.physical_cores)
    exact_seed = Candidate(
        ctx=target.context,
        ngl="all",
        batch=512,
        ubatch=256,
        threads=cores,
        threads_batch=cores,
        kv_k=target.preferred_kv_k,
        kv_v=target.preferred_kv_v,
        vision=target.vision == VisionRequirement.REQUIRED,
        mmproj=target.mmproj,
    )
    exact_est = estimate_static_memory(
        model, hardware, baseline_vram_mb, target.preferred_vram_reserve_mb, exact_seed
    )
    full_free = exact_est.predicted_all_free_mb if model.kind == ModelKind.DENSE else exact_est.predicted_moe_all_free_mb
    exact_class = _resource_class(full_free, hardware, target.preferred_vram_reserve_mb, target.absolute_vram_floor_mb)

    options: list[SolutionOption] = []
    exact_strategy = "full-gpu"
    exact_place: str | int | None = "all"
    exact_degradation: list[DegradationKind] = []
    exact_notes: list[str] = []

    # Preferred reserve is a preference, not a hard capability requirement. If the exact target is
    # predicted to fit full-GPU above the absolute safety floor, try that first even when it misses
    # the preferred reserve. For Dense this avoids sacrificing 20-60% decode speed merely to buy
    # extra headroom before we have measured whether that headroom is actually necessary.
    full_gpu_clears_floor = full_free is not None and full_free >= target.absolute_vram_floor_mb

    if not full_gpu_clears_floor and model.kind == ModelKind.DENSE and exact_est.predicted_dense_ngl != "all":
        exact_strategy = "dense-cpu-offload"
        exact_place = exact_est.predicted_dense_ngl
        exact_degradation = [DegradationKind.PERFORMANCE]
        exact_notes.append(
            "Full-GPU is predicted below the absolute VRAM floor; target layers may need CPU placement "
            "to preserve the requested context/KV. Expected generation penalty is HIGH for Dense models."
        )
    elif not full_gpu_clears_floor and model.kind == ModelKind.MOE:
        floor_place, floor_free = _moe_floor_placement(
            model, hardware, baseline_vram_mb, exact_seed, target.absolute_vram_floor_mb
        )
        if floor_place is not None:
            exact_strategy = "moe-expert-offload"
            exact_place = floor_place
            exact_degradation = [DegradationKind.PERFORMANCE]
            exact_notes.append(
                f"Full-expert GPU placement (ncmoe=0) is predicted below the absolute VRAM floor; "
                f"the smallest performance-first recovery seed is ncmoe={floor_place}. "
                "More expert offload is a headroom alternative, not part of the exact full-GPU claim."
            )
            full_free = floor_free
    elif full_gpu_clears_floor and exact_class == ResourceClass.CONSTRAINED:
        exact_notes.append(
            "Exact target is predicted to fit full-GPU, but with less than the preferred VRAM reserve. "
            "It will be measured before any KV-cache precision/capability degradation is considered."
        )

    exact_option_free = full_free if exact_strategy == "full-gpu" else (
        full_free if model.kind == ModelKind.MOE else exact_est.predicted_free_mb
    )
    exact_option_class = _resource_class(
        exact_option_free, hardware, target.preferred_vram_reserve_mb, target.absolute_vram_floor_mb
    )
    exact_likely = exact_option_class != ResourceClass.INFEASIBLE and (
        exact_est.predicted_dense_ngl is not None or exact_est.predicted_moe_ncmoe is not None or exact_option_free is not None
    )
    options.append(SolutionOption(
        name="EXACT_TARGET",
        context=target.context,
        kv_k=target.preferred_kv_k,
        kv_v=target.preferred_kv_v,
        strategy=exact_strategy,
        predicted_free_mb=exact_option_free,
        predicted_placement=exact_place,
        resource_class=exact_option_class,
        degradation=exact_degradation,
        degradation_notes=exact_notes,
        recommended_rank=0 if exact_strategy == "full-gpu" else 30,
        exact_target=True,
        vision_required=target.vision == VisionRequirement.REQUIRED,
        mmproj=target.mmproj,
    ))

    # Preserve requested context by spending KV precision. These are deliberately separate options:
    # changing KV representation affects attention-cache precision and is not a free memory optimization.
    if target.allow_kv_degradation and not target.lock_kv:
        for idx, (kk, vv, note) in enumerate(_kv_ladder(target.preferred_kv_k, target.preferred_kv_v), start=1):
            c = Candidate(
                ctx=target.context, ngl="all", batch=512, ubatch=256,
                threads=cores, threads_batch=cores, kv_k=kk, kv_v=vv,
                vision=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
            )
            est = estimate_static_memory(model, hardware, baseline_vram_mb, target.preferred_vram_reserve_mb, c)
            free = est.predicted_all_free_mb if model.kind == ModelKind.DENSE else est.predicted_moe_all_free_mb
            placement: str | int | None = "all"
            strategy = "full-gpu-kv-degraded"
            # v0.5.4: preferred reserve is not a hard boundary for *any* Dense solution option.
            # The old planner correctly applied this rule to EXACT_TARGET but accidentally forced
            # degraded-KV/context alternatives to numeric ngl whenever the reserve-oriented static
            # seed was numeric. On a 16 GiB Dense model that could turn a perfectly runnable
            # full-GPU Q8/Q4 option into a 2x-slower CPU-offload option. Only cross the PCIe/CPU
            # boundary when full GPU is predicted below the absolute safety floor.
            full_gpu_clears_floor = free is not None and free >= target.absolute_vram_floor_mb
            if model.kind == ModelKind.DENSE and not full_gpu_clears_floor and est.predicted_dense_ngl != "all":
                placement = est.predicted_dense_ngl
                strategy = "dense-cpu-offload-kv-degraded"
                free = est.predicted_free_mb
            elif model.kind == ModelKind.MOE and not full_gpu_clears_floor:
                placement, floor_free = _moe_floor_placement(
                    model, hardware, baseline_vram_mb, c, target.absolute_vram_floor_mb
                )
                if placement is not None and placement > 0:
                    strategy = "moe-expert-offload-kv-degraded"
                    free = floor_free
            rc = _resource_class(free, hardware, target.preferred_vram_reserve_mb, target.absolute_vram_floor_mb)
            kv_info = kv_precision(kk, vv)
            if kv_info.tier == "Q8":
                risk_note = (
                    "Q8 is treated as the low-risk automatic runtime tier, but this is not a "
                    "universal semantic-quality proof for every model/task."
                )
            else:
                risk_note = (
                    "Q4/mixed-KV behavior is task-, architecture- and context-dependent; automatic "
                    "promotion requires an occupied-context throughput probe, while semantic quality "
                    "still needs a task-specific evaluation."
                )
            options.append(SolutionOption(
                name=f"PRESERVE_CONTEXT_KV_{idx}", context=target.context, kv_k=kk, kv_v=vv,
                strategy=strategy, predicted_free_mb=free, predicted_placement=placement,
                resource_class=rc, degradation=[DegradationKind.QUALITY_RISK],
                degradation_notes=[
                    note + f". {risk_note} Model weights are unchanged."
                ], recommended_rank=10 + idx, exact_target=True,
                vision_required=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
            ))

    # Preserve preferred KV/model semantics but reduce capability (context). These are alternatives,
    # never silently re-labelled as satisfying the exact target.
    if target.allow_context_reduction:
        for idx, ctx in enumerate(_context_alternatives(target.context), start=1):
            c = Candidate(
                ctx=ctx, ngl="all", batch=512, ubatch=256,
                threads=cores, threads_batch=cores,
                kv_k=target.preferred_kv_k, kv_v=target.preferred_kv_v,
                vision=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
            )
            est = estimate_static_memory(model, hardware, baseline_vram_mb, target.preferred_vram_reserve_mb, c)
            free = est.predicted_all_free_mb if model.kind == ModelKind.DENSE else est.predicted_moe_all_free_mb
            placement: str | int | None = "all"
            strategy = "full-gpu-context-reduced"
            full_gpu_clears_floor = free is not None and free >= target.absolute_vram_floor_mb
            if model.kind == ModelKind.DENSE and not full_gpu_clears_floor and est.predicted_dense_ngl != "all":
                placement = est.predicted_dense_ngl
                strategy = "dense-cpu-offload-context-reduced"
                free = est.predicted_free_mb
            elif model.kind == ModelKind.MOE and not full_gpu_clears_floor:
                placement, floor_free = _moe_floor_placement(
                    model, hardware, baseline_vram_mb, c, target.absolute_vram_floor_mb
                )
                if placement is not None and placement > 0:
                    strategy = "moe-expert-offload-context-reduced"
                    free = floor_free
            rc = _resource_class(free, hardware, target.preferred_vram_reserve_mb, target.absolute_vram_floor_mb)
            options.append(SolutionOption(
                name=f"PRESERVE_KV_PRECISION_CTX_{ctx}", context=ctx,
                kv_k=target.preferred_kv_k, kv_v=target.preferred_kv_v,
                strategy=strategy,
                predicted_free_mb=free, predicted_placement=placement,
                resource_class=rc, degradation=[DegradationKind.CAPABILITY],
                degradation_notes=[f"Context capability reduced {target.context}→{ctx}; selected GGUF weights and preferred KV-cache precision are preserved."],
                recommended_rank=20 + idx, exact_target=False,
                vision_required=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
            ))

    # Cross trade-offs matter near the memory boundary. A 75% context with only V-cache reduced
    # can be a better Pareto point than either the full requested context with Q4/Q4 or a 50%
    # context with Q8/Q8. Older planners generated only the two extremes and could miss this knee.
    # Keep the grid deliberately small and, for Dense, only retain full-GPU mixed options; stacking
    # both KV/context degradation *and* Dense CPU offload is almost never an attractive automatic choice.
    if target.allow_context_reduction and target.allow_kv_degradation and not target.lock_kv:
        kv_steps = _kv_ladder(target.preferred_kv_k, target.preferred_kv_v)
        for cidx, ctx in enumerate(_context_alternatives(target.context), start=1):
            for kidx, (kk, vv, note) in enumerate(kv_steps, start=1):
                c = Candidate(
                    ctx=ctx, ngl="all", batch=512, ubatch=256,
                    threads=cores, threads_batch=cores, kv_k=kk, kv_v=vv,
                    vision=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
                )
                est = estimate_static_memory(model, hardware, baseline_vram_mb, target.preferred_vram_reserve_mb, c)
                free = est.predicted_all_free_mb if model.kind == ModelKind.DENSE else est.predicted_moe_all_free_mb
                placement: str | int | None = "all"
                strategy = "full-gpu-mixed-tradeoff"
                if model.kind == ModelKind.DENSE:
                    if free is None or free < target.absolute_vram_floor_mb:
                        continue
                elif model.kind == ModelKind.MOE:
                    full_gpu_clears_floor = free is not None and free >= target.absolute_vram_floor_mb
                    if not full_gpu_clears_floor:
                        placement, floor_free = _moe_floor_placement(
                            model, hardware, baseline_vram_mb, c, target.absolute_vram_floor_mb
                        )
                        if placement is not None and placement > 0:
                            strategy = "moe-expert-offload-mixed-tradeoff"
                            free = floor_free
                rc = _resource_class(free, hardware, target.preferred_vram_reserve_mb, target.absolute_vram_floor_mb)
                kv_info = kv_precision(kk, vv)
                kv_policy_note = (
                    "Q8 is the low-risk automatic runtime tier"
                    if kv_info.tier == "Q8" else
                    "Q4/mixed KV requires occupied-context throughput qualification and task-specific quality validation"
                )
                options.append(SolutionOption(
                    name=f"TRADEOFF_CTX_{ctx}_KV_{kidx}", context=ctx, kv_k=kk, kv_v=vv,
                    strategy=strategy, predicted_free_mb=free, predicted_placement=placement,
                    resource_class=rc,
                    degradation=[DegradationKind.QUALITY_RISK, DegradationKind.CAPABILITY],
                    degradation_notes=[
                        f"Context capability reduced {target.context}→{ctx}; {note}. "
                        f"{kv_policy_note}. This option is retained because it may be a better full-GPU knee than either extreme."
                    ],
                    recommended_rank=26 + cidx * 3 + kidx,
                    exact_target=False,
                    vision_required=target.vision == VisionRequirement.REQUIRED, mmproj=target.mmproj,
                ))

    # Architecture-aware ordering. For Dense, heavy target-layer CPU offload is intentionally pushed
    # behind a full-GPU KV/context trade-off; for MoE, expert offload remains a first-class option.
    def sort_key(opt: SolutionOption) -> tuple[int, int, int, int]:
        rank = opt.recommended_rank
        dense_cpu = model.kind == ModelKind.DENSE and opt.strategy.startswith("dense-cpu-offload")
        moe_cpu = model.kind == ModelKind.MOE and opt.strategy.startswith("moe-expert-offload")
        kv_precision_risk = DegradationKind.QUALITY_RISK in opt.degradation
        capability_loss = DegradationKind.CAPABILITY in opt.degradation

        priority = (target.priority or "balanced").lower()
        if priority == "context":
            # Requested context is king: keep exact-context options ahead of lower-context alternatives.
            rank += 50 if capability_loss else 0
            rank += 10 if dense_cpu else 0
        elif priority == "quality":
            # Preserve KV-cache numerical precision; capability/performance trade-offs are preferable to KV degradation.
            rank += 60 if kv_precision_risk else 0
            rank += 15 if dense_cpu else 0
        elif priority == "speed":
            # Static speed prior: avoid CPU execution strongly; smaller full-GPU contexts are acceptable.
            rank += 70 if dense_cpu else 0
            rank += 15 if moe_cpu else 0
            rank -= min(20, max(0, (target.context - opt.context) // max(4096, target.context // 16))) if capability_loss else 0
        else:  # balanced
            if dense_cpu:
                rank += 25
            if moe_cpu:
                rank -= 5

        infeasible = 1 if opt.resource_class == ResourceClass.INFEASIBLE else 0
        unknown = 1 if opt.resource_class == ResourceClass.UNKNOWN else 0
        # Prefer more context as a final stable tie-break unless speed priority explicitly pushed it down.
        ctx_penalty = max(0, target.context - opt.context) // 4096
        return (infeasible, unknown, rank, ctx_penalty)

    options.sort(key=sort_key)
    for i, option in enumerate(options, start=1):
        option.recommended_rank = i

    best = next((o for o in options if o.resource_class in {ResourceClass.COMFORTABLE, ResourceClass.CONSTRAINED}), None)
    exact_no_kv_precision_loss = any(
        o.exact_target and DegradationKind.QUALITY_RISK not in o.degradation
        and o.resource_class != ResourceClass.INFEASIBLE for o in options
    )
    if exact_strategy == "full-gpu" and exact_option_class == ResourceClass.COMFORTABLE:
        summary = "Exact target is statically comfortable; optimize performance without exploring degradation paths."
    elif exact_strategy == "full-gpu" and exact_option_class == ResourceClass.CONSTRAINED:
        summary = "Exact target likely fits but is memory-constrained; preserve KV-cache precision first and optimize headroom/performance."
    elif best:
        if exact_likely and options and next((o for o in options if o.name == "EXACT_TARGET"), None) and next(o for o in options if o.name == "EXACT_TARGET").strategy == "full-gpu":
            summary = (
                "Exact preferred-KV-precision target appears runnable but constrained; it will be tried first. "
                "Alternatives are retained only as explicitly disclosed fallbacks."
            )
        else:
            summary = (
                "Exact preferred-KV-precision target is constrained or infeasible; alternatives were generated explicitly. "
                "No KV/context degradation is allowed to happen silently."
            )
    else:
        summary = "No statically credible option was found; runtime probes may still be attempted only if policy permits."

    overall = exact_option_class
    if exact_strategy != "full-gpu" and model.kind == ModelKind.DENSE:
        overall = ResourceClass.CONSTRAINED if exact_likely else ResourceClass.INFEASIBLE

    return FeasibilityPlan(
        target=target,
        capabilities=capabilities,
        resource_class=overall,
        options=options,
        exact_likely_feasible=exact_likely,
        exact_without_quality_degradation=exact_no_kv_precision_loss,
        summary=summary,
    )


def format_capabilities(analysis: CapabilityAnalysis, target: TargetSpec) -> list[str]:
    lines = [
        "[Capability analysis]",
        f"  architecture={analysis.architecture or 'unknown'} | kind={analysis.model_kind}",
        f"  native context={analysis.native_context or 'unknown'} | requested={target.context}",
        f"  vision={analysis.vision.value} | requested={target.vision.value}",
        f"  MTP={analysis.mtp.value}",
    ]
    for note in analysis.notes:
        lines.append(f"  note: {note}")
    return lines


def format_feasibility(plan: FeasibilityPlan) -> list[str]:
    lines = [
        "[Target feasibility]",
        f"  target: ctx={plan.target.context} | vision={plan.target.vision.value} | workload={plan.target.workload} | priority={plan.target.priority}",
        f"  degradation policy={plan.target.degradation_policy.value}",
        f"  resource class={plan.resource_class.value}",
        f"  exact requested semantics statically runnable={str(plan.exact_likely_feasible).lower()}",
        "  note: this is a static context/KV/capability placement estimate, not a full-GPU "
        "claim; measured runtime probes remain authoritative.",
        f"  {plan.summary}",
    ]
    if plan.options:
        lines.append("  solution envelope:")
        for opt in plan.options:
            deg = ",".join(x.value for x in opt.degradation) if opt.degradation else "none"
            free = f"{opt.predicted_free_mb} MiB" if opt.predicted_free_mb is not None else "unknown"
            placement = opt.predicted_placement if opt.predicted_placement is not None else "unknown"
            lines.append(
                f"    {opt.recommended_rank}. {opt.name}: ctx={opt.context} KV={opt.kv_k}/{opt.kv_v} "
                f"strategy={opt.strategy} placement={placement} free≈{free} "
                f"class={opt.resource_class.value} degradation={deg}"
            )
            for note in opt.degradation_notes:
                lines.append(f"       trade-off: {note}")
    return lines


def runnable_options(
    plan: FeasibilityPlan, *, include_exact_declaration: bool = False,
) -> list[SolutionOption]:
    options = [
        o for o in plan.options
        if o.resource_class != ResourceClass.INFEASIBLE
    ]
    if include_exact_declaration:
        # Reconnaissance needs the user's *declared* target even when that exact runtime
        # candidate is statically impossible.  Without it, a 262K FP16 request can arrive at
        # the optimizer as an anonymous collection of 16K/32K/64K compromises; Dense then takes
        # the generic shortlist and can spend the MAX_KV_PRECISION slot on a redundant short Q4
        # speed anchor.  Keep the descriptor for semantics only. ``AutotuneEngine.tune`` never
        # launches an INFEASIBLE option.
        exact = next((o for o in plan.options if o.name == "EXACT_TARGET"), None)
        if exact is not None and all(o.name != exact.name for o in options):
            options.append(exact)
    return options
