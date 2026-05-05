"""Советник по настройке CPU MoE layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.vram_estimator import full_vram_estimate, VRAMEstimate


@dataclass
class MoEAdvice:
    """Результат анализа MoE конфигурации."""

    is_moe: bool
    expert_count: int
    expert_used: int
    recommended_ncmoe: int
    vram_with_ncmoe: VRAMEstimate
    vram_without_ncmoe: VRAMEstimate
    vram_saved_gib: float
    reasoning: List[str]
    warning: Optional[str]


def compute_moe_advice(
    info: Dict[str, Any],
    ctx_size: int,
    gpu_layers: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    batch_size: int = 2048,
) -> MoEAdvice:
    """Полный анализ и рекомендация по CPU MoE layers."""
    expert_count = info.get("expert_count", 0)
    expert_used = info.get("expert_used", 0)
    block_count = info.get("block_count", 0)
    size_gib = info.get("size_gib", 0.0)
    reasoning: List[str] = []
    warning: Optional[str] = None

    if not expert_count or expert_count <= 1:
        baseline = full_vram_estimate(
            info,
            ctx_size,
            gpu_layers,
            cache_type_k,
            cache_type_v,
            flash_attn,
            parallel_slots,
            ncmoe=0,
        )
        return MoEAdvice(
            is_moe=False,
            expert_count=0,
            expert_used=0,
            recommended_ncmoe=0,
            vram_with_ncmoe=baseline,
            vram_without_ncmoe=baseline,
            vram_saved_gib=0.0,
            reasoning=["Модель не является MoE"],
            warning=None,
        )

    reasoning.append(f"MoE: {expert_count} экспертов, {expert_used} активных")

    baseline = full_vram_estimate(
        info,
        ctx_size,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
        ncmoe=0,
    )

    pressure_score = 0.0

    if size_gib >= 30:
        pressure_score += 0.4
        reasoning.append(f"Крупная модель ({size_gib:.1f} GiB)")
    elif size_gib >= 20:
        pressure_score += 0.25
    elif size_gib >= 10:
        pressure_score += 0.1

    if ctx_size >= 65536:
        pressure_score += 0.4
        reasoning.append(f"Контекст {ctx_size:,} (KV {baseline.kv_cache_gib:.2f} GiB)")
    elif ctx_size >= 32768:
        pressure_score += 0.25
        reasoning.append(f"Контекст {ctx_size:,}")
    elif ctx_size >= 16384:
        pressure_score += 0.15
    elif ctx_size >= 8192:
        pressure_score += 0.05

    kv_heavy = cache_type_k in ("f16", "f32", "bf16")
    if kv_heavy:
        pressure_score += 0.1

    if not flash_attn:
        pressure_score += 0.1
    else:
        reasoning.append("Flash Attention вкл.")

    if parallel_slots > 1:
        pressure_score += (parallel_slots - 1) * 0.05

    inactive_ratio = (expert_count - expert_used) / expert_count if expert_used else 0.5

    target_ncmoe = int(block_count * pressure_score * inactive_ratio)
    max_ncmoe = max(0, int(block_count * inactive_ratio))
    target_ncmoe = min(target_ncmoe, max_ncmoe)

    if gpu_layers < block_count:
        gpu_fraction = gpu_layers / block_count
        if gpu_fraction < 0.5:
            warning = f"gpu_layers={gpu_layers} < block_count={block_count}"
            target_ncmoe = min(target_ncmoe, int(max_ncmoe * gpu_fraction))

    target_ncmoe = max(0, target_ncmoe)

    with_ncmoe = full_vram_estimate(
        info,
        ctx_size,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
        ncmoe=target_ncmoe,
    )

    saved = round(max(0.0, baseline.total_gib - with_ncmoe.total_gib), 2)

    if target_ncmoe == 0:
        reasoning.append("Рекомендация: ncmoe=0")
    else:
        reasoning.append(f"Рекомендация: ncmoe={target_ncmoe} (-{saved:.2f} GiB)")

    return MoEAdvice(
        is_moe=True,
        expert_count=expert_count,
        expert_used=expert_used,
        recommended_ncmoe=target_ncmoe,
        vram_with_ncmoe=with_ncmoe,
        vram_without_ncmoe=baseline,
        vram_saved_gib=saved,
        reasoning=reasoning,
        warning=warning,
    )
