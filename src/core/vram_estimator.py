"""Оценка потребления VRAM и рекомендации по MoE CPU offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

_KV_TYPE_BYTES: Dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0,
    "q5_1": 0.6875,
    "q5_0": 0.625,
    "q4_1": 0.5625,
    "q4_0": 0.5,
    "iq4_nl": 0.5,
}

_FA_KV_REDUCTION = 0.70


@dataclass(frozen=True)
class VRAMEstimate:
    """Результат оценки потребления VRAM."""

    model_vram_gib: float
    kv_cache_gib: float
    overhead_gib: float
    total_gib: float
    kv_per_1k_ctx_mib: float
    expert_layer_mib: float
    is_moe: bool


def estimate_kv_cache(
    block_count: int,
    head_count: int,
    embedding_length: int,
    ctx_size: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    kv_head_count: int = 0,
) -> float:
    """Оценка размера KV-cache в GiB."""
    if not (block_count and head_count and embedding_length):
        return 0.0

    head_dim = embedding_length // head_count
    effective_kv_heads = kv_head_count or head_count
    k_bytes = _KV_TYPE_BYTES.get(cache_type_k.lower(), 2.0)
    v_bytes = _KV_TYPE_BYTES.get(cache_type_v.lower(), 2.0)

    kv_per_token = block_count * effective_kv_heads * head_dim * (k_bytes + v_bytes)
    total_bytes = kv_per_token * ctx_size * parallel_slots

    if flash_attn:
        total_bytes *= _FA_KV_REDUCTION

    return total_bytes / (1024**3)


def estimate_model_vram(
    size_gib: float,
    gpu_layers: int,
    block_count: int,
    expert_count: int = 0,
    expert_used: int = 0,
) -> float:
    """Оценка VRAM под веса модели."""
    if not block_count or not size_gib:
        layer_fraction = min(1.0, gpu_layers / max(block_count, 1))
        return size_gib * layer_fraction * 1.05

    layer_fraction = min(1.0, gpu_layers / block_count)

    if expert_count > 1 and expert_used > 0:
        shared_fraction = 0.30
        expert_fraction = 0.70

        shared_vram = size_gib * shared_fraction * layer_fraction
        expert_vram = size_gib * expert_fraction * layer_fraction

        return (shared_vram + expert_vram) * 1.05
    else:
        return size_gib * layer_fraction * 1.05


def estimate_expert_layer_size_mib(
    size_gib: float,
    block_count: int,
    expert_count: int,
    expert_used: int,
) -> float:
    """Размер одного слоя неактивных экспертов в MiB."""
    if not (block_count and expert_count and size_gib):
        return 0.0

    expert_total_gib = size_gib * 0.70
    per_block_gib = expert_total_gib / block_count
    inactive = max(0, expert_count - expert_used)
    if expert_count == 0:
        return 0.0
    inactive_fraction = inactive / expert_count
    return per_block_gib * inactive_fraction * 1024


def full_vram_estimate(
    info: Dict[str, Any],
    ctx_size: int,
    gpu_layers: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    ncmoe: int = 0,
) -> VRAMEstimate:
    """Полная оценка потребления VRAM."""
    block_count = info.get("block_count", 0)
    head_count = info.get("head_count", 0)
    kv_head_count = info.get("head_count_kv", 0)
    embedding_len = info.get("embedding_length", 0)
    expert_count = info.get("expert_count", 0)
    expert_used = info.get("expert_used", 0)
    size_gib = info.get("size_gib", 0.0)
    is_moe = expert_count > 1

    kv_gib = estimate_kv_cache(
        block_count,
        head_count,
        embedding_len,
        ctx_size,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
        kv_head_count,
    )

    model_gib = estimate_model_vram(
        size_gib,
        gpu_layers,
        block_count,
        expert_count,
        expert_used,
    )

    expert_layer_mib = estimate_expert_layer_size_mib(
        size_gib, block_count, expert_count, expert_used
    )

    if is_moe and ncmoe > 0 and block_count > 0:
        saved_fraction = min(1.0, ncmoe / block_count)
        saved_gib = (expert_layer_mib / 1024) * saved_fraction * block_count
        model_gib = max(0.0, model_gib - saved_gib)

    overhead_gib = 0.3 + ctx_size * 0.000_001
    total = model_gib + kv_gib + overhead_gib

    kv_per_1k = (kv_gib * 1024 / ctx_size * 1000) if ctx_size else 0.0

    return VRAMEstimate(
        model_vram_gib=round(model_gib, 2),
        kv_cache_gib=round(kv_gib, 2),
        overhead_gib=round(overhead_gib, 2),
        total_gib=round(total, 2),
        kv_per_1k_ctx_mib=round(kv_per_1k, 1),
        expert_layer_mib=round(expert_layer_mib, 1),
        is_moe=is_moe,
    )
