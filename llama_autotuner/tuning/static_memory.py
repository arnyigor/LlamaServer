from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from llama_autotuner.models import Candidate, HardwareInfo, ModelInfo, ModelKind

_MIB = 1024 * 1024

# Effective storage bytes per scalar in llama.cpp KV cache types.
# Quantized types include their scale/header overhead, not just nominal bits/value.
_KV_BYTES_PER_VALUE: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
}


@dataclass(slots=True)
class StaticMemoryEstimate:
    ctx: int
    kv_cache_mb: int | None
    runtime_buffer_mb: int
    baseline_mb: int
    preferred_reserve_mb: int
    usable_for_weights_mb: int | None
    model_tensor_mb: int | None
    predicted_dense_ngl: int | str | None = None
    predicted_moe_ncmoe: int | None = None
    # Exact free VRAM estimate with no expert offload (-ncmoe 0). For MoE this must
    # be kept separate from predicted_free_mb, which is the reserve-oriented seed.
    predicted_moe_all_free_mb: int | None = None
    # Free VRAM for the selected preferred-reserve seed. Kept for backward compatibility.
    predicted_free_mb: int | None = None
    # Dense full-offload is a special llama.cpp mode and must be estimated/displayed separately
    # from numeric ngl=n_layer. This prevents a numeric safe seed from masquerading as the
    # predicted memory footprint of ngl=all.
    predicted_all_free_mb: int | None = None
    confidence: str = "LOW"
    layout_complete: bool = False
    notes: list[str] = field(default_factory=list)

    def seed_label(self) -> str:
        if self.predicted_moe_ncmoe is not None:
            return f"ncmoe={self.predicted_moe_ncmoe}"
        if self.predicted_dense_ngl is not None:
            return f"ngl={self.predicted_dense_ngl}"
        return "unknown"


def _first_meta_suffix(model: ModelInfo, *suffixes: str):
    for suffix in suffixes:
        for key, value in model.metadata.items():
            if key.endswith(suffix):
                return value
    return None


def _first_int_suffix(model: ModelInfo, *suffixes: str) -> int | None:
    value = _first_meta_suffix(model, *suffixes)
    return int(value) if isinstance(value, int) else None




def model_mtp_block_count(model: ModelInfo) -> int:
    if model.mtp_block_count:
        return max(0, int(model.mtp_block_count))
    value = _first_int_suffix(model, "nextn_predict_layers") or 0
    return max(0, int(value))


def model_main_block_count(model: ModelInfo) -> int:
    if model.main_block_count:
        return max(1, int(model.main_block_count))
    stored = model.block_count or (max(model.block_tensor_bytes) + 1 if model.block_tensor_bytes else 1)
    return max(1, int(stored) - model_mtp_block_count(model))


def _kv_bytes(model: ModelInfo, ctx: int, kv_k: str, kv_v: str, mtp: bool = False) -> int | None:
    blocks = model.block_count
    if not blocks or blocks <= 0:
        return None

    emb = _first_int_suffix(model, "embedding_length")
    heads = _first_int_suffix(model, "attention.head_count", "head_count")
    kv_meta = _first_meta_suffix(model, "attention.head_count_kv", "head_count_kv")
    key_len = _first_int_suffix(model, "attention.key_length", "key_length")
    value_len = _first_int_suffix(model, "attention.value_length", "value_length")

    if key_len is None and emb and heads:
        key_len = max(1, emb // heads)
    if value_len is None:
        value_len = key_len
    if not key_len or not value_len:
        return None

    k_bpv = _KV_BYTES_PER_VALUE.get(kv_k.lower())
    v_bpv = _KV_BYTES_PER_VALUE.get(kv_v.lower())
    if k_bpv is None or v_bpv is None:
        return None

    nextn = model_mtp_block_count(model)
    main_blocks = model_main_block_count(model)

    # Qwen3.5/3.6/3.8 GGUF can encode head_count_kv as an array: recurrent
    # Gated-DeltaNet layers have 0 KV heads while full-attention layers have non-zero
    # entries. This is much more accurate than multiplying KV by every block.
    if isinstance(kv_meta, list) and kv_meta and all(isinstance(x, int) for x in kv_meta):
        entries = [int(x) for x in kv_meta[:blocks]]
        if not mtp and nextn and len(entries) >= nextn:
            entries = entries[: max(0, len(entries) - nextn)]
        kv_head_layer_sum = sum(max(0, x) for x in entries)
    else:
        kv_heads = int(kv_meta) if isinstance(kv_meta, int) else (heads or 0)
        if kv_heads <= 0:
            return None
        interval = _first_int_suffix(model, "full_attention_interval")
        arch = (model.architecture or "").lower()
        if interval and interval > 1 and arch in {"qwen35", "qwen35moe"}:
            # qwen35 is hybrid recurrent + full attention. Full attention occurs at
            # a fixed interval; NextN/MTP layers are dense attention-only draft layers.
            attention_layers = max(1, (main_blocks + interval - 1) // interval)
            if mtp:
                attention_layers += nextn
        else:
            attention_layers = main_blocks + (nextn if mtp else 0)
        kv_head_layer_sum = kv_heads * attention_layers

    if kv_head_layer_sum <= 0:
        return None

    k_values = ctx * kv_head_layer_sum * key_len
    v_values = ctx * kv_head_layer_sum * value_len
    raw = int(k_values * k_bpv + v_values * v_bpv)
    return int(raw * 1.03)



def _vision_projector_bytes(candidate: Candidate) -> int:
    if not candidate.vision or not candidate.mmproj:
        return 0
    try:
        return max(0, Path(candidate.mmproj).stat().st_size)
    except OSError:
        return 0

def _runtime_buffer_bytes(model: ModelInfo, candidate: Candidate) -> int:
    emb = _first_int_suffix(model, "embedding_length") or 4096
    ub = max(1, candidate.ubatch)

    # This is intentionally an empirical *seed* estimate, not a pass/fail oracle.
    # CUDA/context/graph overhead has a fixed component plus an activation/workspace
    # component that grows roughly with ubatch * embedding. Flash-attention and backend
    # graph details make exact static calculation impractical across llama.cpp builds.
    fixed = 384 * _MIB
    activation = int(6.0 * ub * emb * 4)
    mtp_extra = 320 * _MIB if candidate.mtp else 0
    return fixed + activation + mtp_extra


def _dense_gpu_weight_bytes(model: ModelInfo, ngl: int | str, include_mtp: bool = False) -> int | None:
    if not model.block_tensor_bytes:
        return None

    main_blocks = model_main_block_count(model)
    ids = sorted(model.block_tensor_bytes)
    main_ids = ids[:main_blocks]
    mtp_ids = ids[main_blocks:]

    if ngl == "all":
        # `-ngl all` is a special full-offload mode in llama.cpp. It can also keep
        # output/non-block tensors on GPU, unlike a numeric ngl equal to n_layer.
        # When MTP is disabled, do not charge the auxiliary NextN blocks as target weights.
        block_bytes = sum(model.block_tensor_bytes.get(i, 0) for i in main_ids)
        if include_mtp:
            block_bytes += sum(model.block_tensor_bytes.get(i, 0) for i in mtp_ids)
        return block_bytes + model.non_block_tensor_bytes

    n = min(main_blocks, max(0, int(ngl)))
    if n == 0:
        return 0
    # Numeric -ngl offloads repeated target-model layers; auxiliary NextN blocks are
    # controlled by speculative/MTP machinery and must not inflate the numeric range.
    selected = main_ids[-n:]
    return sum(model.block_tensor_bytes.get(i, 0) for i in selected)


def _moe_gpu_weight_bytes(model: ModelInfo, ncmoe: int, *, include_mtp: bool = False) -> int | None:
    if not model.tensor_data_bytes or not model.block_expert_bytes:
        return None
    main_blocks = model_main_block_count(model)
    n = min(main_blocks, max(0, int(ncmoe)))

    # Keep the target-model and auxiliary NextN/MTP accounting consistent with Dense.
    # When speculative decoding is disabled, the auxiliary stored block(s) are not part
    # of the target runtime placement and must not make MoE look larger than it is.
    base = int(model.tensor_data_bytes)
    if not include_mtp:
        stored = model.block_count or (max(model.block_tensor_bytes) + 1 if model.block_tensor_bytes else main_blocks)
        for i in range(main_blocks, int(stored)):
            base -= int(model.block_tensor_bytes.get(i, 0))

    # -ncmoe keeps routed expert weights for the first N target-model MoE layers on CPU.
    # N=0 therefore means all routed experts stay on GPU; larger N is progressively safer
    # for VRAM but may reduce PP/TG.
    cpu_expert = sum(int(model.block_expert_bytes.get(i, 0)) for i in range(n))
    return max(0, base - cpu_expert)



def estimate_candidate_free_mb(
    model: ModelInfo, hardware: HardwareInfo, baseline_vram_mb: int, candidate: Candidate
) -> int | None:
    """Best-effort free-VRAM estimate for one exact candidate, before reserve policy.

    Unlike ``estimate_static_memory`` this does not change placement to satisfy the preferred
    reserve; it answers "if I run *this* ngl/ncmoe/MTP/ubatch, roughly how much VRAM remains?".
    Session calibration in the optimizer may then correct the backend/workspace bias.
    """
    kv = _kv_bytes(model, candidate.ctx, candidate.kv_k, candidate.kv_v, candidate.mtp)
    if kv is None:
        return None
    runtime = _runtime_buffer_bytes(model, candidate)
    if model.kind == ModelKind.DENSE:
        weight = _dense_gpu_weight_bytes(model, candidate.ngl, include_mtp=candidate.mtp)
    elif model.kind == ModelKind.MOE:
        weight = _moe_gpu_weight_bytes(model, candidate.ncmoe or 0, include_mtp=candidate.mtp)
    else:
        weight = model.tensor_data_bytes or None
    if weight is None:
        return None
    projector = _vision_projector_bytes(candidate)
    used = baseline_vram_mb * _MIB + kv + runtime + weight + projector
    return int((hardware.vram_total_mb * _MIB - used) // _MIB)

def estimate_static_memory(
    model: ModelInfo,
    hardware: HardwareInfo,
    baseline_vram_mb: int,
    preferred_reserve_mb: int,
    candidate: Candidate,
) -> StaticMemoryEstimate:
    kv = _kv_bytes(model, candidate.ctx, candidate.kv_k, candidate.kv_v, candidate.mtp)
    runtime = _runtime_buffer_bytes(model, candidate)
    model_tensor = model.tensor_data_bytes or None

    est = StaticMemoryEstimate(
        ctx=candidate.ctx,
        kv_cache_mb=ceil(kv / _MIB) if kv is not None else None,
        runtime_buffer_mb=ceil(runtime / _MIB),
        baseline_mb=baseline_vram_mb,
        preferred_reserve_mb=preferred_reserve_mb,
        usable_for_weights_mb=None,
        model_tensor_mb=ceil(model_tensor / _MIB) if model_tensor else None,
        layout_complete=model.tensor_layout_complete,
    )

    if kv is None:
        est.notes.append("KV geometry could not be derived from GGUF metadata; placement seed is less precise.")
        kv = 0

    projector = _vision_projector_bytes(candidate)
    if projector:
        est.notes.append(f"Vision projector adds ≈{ceil(projector / _MIB)} MiB of static file-backed memory pressure before image-workload buffers.")
    weight_budget = (hardware.vram_total_mb * _MIB - baseline_vram_mb * _MIB - preferred_reserve_mb * _MIB
                     - kv - runtime - projector)
    est.usable_for_weights_mb = max(0, int(weight_budget // _MIB))

    if model.split_count > 1 and model.tensor_layout_complete:
        est.notes.append(
            f"Tensor placement was aggregated across all {model.split_count} GGUF shards; "
            "startup and weight sizing use the complete logical model."
        )
    if not model.tensor_layout_complete:
        est.notes.append("Tensor layout is incomplete (for example a split GGUF); using static analysis only as a weak hint.")

    if model.kind == ModelKind.DENSE and model.block_tensor_bytes:
        blocks = model_main_block_count(model)
        all_bytes = _dense_gpu_weight_bytes(model, "all", include_mtp=candidate.mtp)
        if all_bytes is not None:
            all_used = baseline_vram_mb * _MIB + kv + runtime + all_bytes + projector
            est.predicted_all_free_mb = int((hardware.vram_total_mb * _MIB - all_used) // _MIB)
        if all_bytes is not None and all_bytes <= weight_budget:
            est.predicted_dense_ngl = "all"
            predicted_weight = all_bytes
        else:
            best = 0
            predicted_weight = 0
            for n in range(1, blocks + 1):
                b = _dense_gpu_weight_bytes(model, n, include_mtp=candidate.mtp)
                if b is not None and b <= weight_budget:
                    best = n
                    predicted_weight = b
                else:
                    break
            est.predicted_dense_ngl = best
        used = baseline_vram_mb * _MIB + kv + runtime + predicted_weight + projector
        est.predicted_free_mb = int((hardware.vram_total_mb * _MIB - used) // _MIB)
        est.confidence = "MEDIUM" if model.tensor_layout_complete and est.kv_cache_mb is not None else "LOW"
        est.notes.append("Dense seed distinguishes target-model blocks from auxiliary MTP/NextN blocks; `ngl=all` remains a special full-offload candidate.")
        return est

    if model.kind == ModelKind.MOE and model.block_expert_bytes and model.tensor_data_bytes:
        blocks = model_main_block_count(model)
        all_weight = _moe_gpu_weight_bytes(model, 0, include_mtp=candidate.mtp)
        if all_weight is not None:
            all_used = baseline_vram_mb * _MIB + kv + runtime + all_weight + projector
            est.predicted_moe_all_free_mb = int((hardware.vram_total_mb * _MIB - all_used) // _MIB)

        # Backward-compatible preferred-reserve seed. This is NOT the same thing as
        # full-GPU feasibility; callers that need capability/performance decisions must
        # inspect predicted_moe_all_free_mb or an exact candidate estimate.
        chosen = blocks
        predicted_weight = _moe_gpu_weight_bytes(model, chosen, include_mtp=candidate.mtp) or 0
        for n in range(0, blocks + 1):
            b = _moe_gpu_weight_bytes(model, n, include_mtp=candidate.mtp)
            if b is not None and b <= weight_budget:
                chosen = n
                predicted_weight = b
                break
        est.predicted_moe_ncmoe = chosen
        used = baseline_vram_mb * _MIB + kv + runtime + predicted_weight + projector
        est.predicted_free_mb = int((hardware.vram_total_mb * _MIB - used) // _MIB)
        est.confidence = "MEDIUM" if model.tensor_layout_complete and est.kv_cache_mb is not None else "LOW"
        est.notes.append(
            "MoE reports ncmoe=0 full-expert-GPU feasibility separately from the preferred-reserve expert-offload seed; real probes remain authoritative."
        )
        return est

    # Fallback: file/tensor size is still useful for explaining whether full offload is plausible,
    # but do not pretend it gives a layer-accurate seed.
    est.confidence = "LOW"
    est.notes.append("Per-layer tensor layout is unavailable; falling back to the existing guarded search order.")
    return est


def dense_seed_order(model: ModelInfo, estimate: StaticMemoryEstimate | None) -> list[str | int]:
    total = model_main_block_count(model)
    if not estimate or estimate.predicted_dense_ngl is None or estimate.confidence == "LOW":
        return []

    pred = estimate.predicted_dense_ngl
    if pred == "all":
        values: list[str | int] = ["all", total, max(0, total - 2), max(0, total - 4), max(0, total - 8), 0]
    else:
        p = int(pred)
        # The Dense search expects aggressive -> safe ordering. Start only a few layers
        # above the static safe prediction, then descend through the estimated boundary.
        start_n = min(total, p + 4)
        values = [start_n, min(total, p + 2), p, max(0, p - 2), max(0, p - 4), max(0, p - 8), 0]
        # Only test `all` when the estimate is already very close to full offload.
        if p >= total - 1:
            values.insert(0, "all")

    out: list[str | int] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def moe_seed_order(model: ModelInfo, estimate: StaticMemoryEstimate | None) -> list[int]:
    blocks = max(1, model.block_count or 40)
    if not estimate or estimate.predicted_moe_ncmoe is None or estimate.confidence == "LOW":
        return []
    p = int(estimate.predicted_moe_ncmoe)
    # MoE placement search expects safe -> aggressive (large ncmoe -> small ncmoe).
    # Begin a conservative local window above the prediction, then cross it.
    guard = max(3, int(round(blocks * 0.12)))
    values = [min(blocks, p + guard), min(blocks, p + max(2, guard // 2)), p, max(0, p - 2), max(0, p - 4), 0]
    out: list[int] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def format_static_estimate(model: ModelInfo, estimate: StaticMemoryEstimate) -> list[str]:
    lines = [
        "[Static analysis]",
        f"  GGUF tensors: {model.tensor_count or 'unknown'} | tensor data: "
        + (f"{estimate.model_tensor_mb} MiB" if estimate.model_tensor_mb is not None else "unknown"),
        f"  blocks: stored={model.block_count or 'unknown'} | main={model_main_block_count(model)} | MTP/NextN={model_mtp_block_count(model)}",
        f"  target ctx={estimate.ctx} | estimated KV="
        + (f"{estimate.kv_cache_mb} MiB" if estimate.kv_cache_mb is not None else "unknown")
        + f" | runtime/workspace allowance≈{estimate.runtime_buffer_mb} MiB",
        f"  baseline={estimate.baseline_mb} MiB | preferred reserve={estimate.preferred_reserve_mb} MiB "
        f"| estimated weight budget≈{estimate.usable_for_weights_mb} MiB",
    ]
    if model.kind == ModelKind.DENSE and estimate.predicted_all_free_mb is not None:
        lines.append(f"  predicted full-GPU: ngl=all | free≈{estimate.predicted_all_free_mb} MiB")
    if estimate.predicted_dense_ngl is not None:
        label = "preferred-reserve seed" if estimate.predicted_dense_ngl != "all" else "preferred-reserve seed (full GPU)"
        lines.append(
            f"  {label}: ngl={estimate.predicted_dense_ngl}"
            + (f" | predicted free≈{estimate.predicted_free_mb} MiB" if estimate.predicted_free_mb is not None else "")
        )
    if estimate.predicted_moe_all_free_mb is not None:
        lines.append(f"  predicted full-expert GPU: ncmoe=0 | free≈{estimate.predicted_moe_all_free_mb} MiB")
    if estimate.predicted_moe_ncmoe is not None:
        lines.append(
            f"  preferred-reserve MoE seed: ncmoe={estimate.predicted_moe_ncmoe}"
            + (f" | predicted free≈{estimate.predicted_free_mb} MiB" if estimate.predicted_free_mb is not None else "")
        )
    lines.append(f"  static-estimate confidence: {estimate.confidence}; real llama-server probes remain authoritative.")
    for note in estimate.notes:
        lines.append(f"  note: {note}")
    return lines
