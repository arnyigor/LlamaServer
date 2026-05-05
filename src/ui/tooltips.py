"""Генераторы tooltip-текстов для виджетов."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.moe_advisor import MoEAdvice, compute_moe_advice
from src.core.vram_estimator import full_vram_estimate, VRAMEstimate


def _vram_bar(used: float, total_ref: float = 24.0, width: int = 20) -> str:
    """ASCII прогресс-бар для VRAM."""
    fraction = min(1.0, used / total_ref) if total_ref > 0 else 0.0
    filled = int(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(fraction * 100)
    return f"[{bar}] {pct}%"


def build_ncmoe_tooltip(
    info: Dict[str, Any],
    ctx_size: int,
    gpu_layers: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    current_ncmoe: int = 0,
    vram_budget_gib: Optional[float] = None,
) -> str:
    """Tooltip для виджета cpu_moe_layers."""
    expert_count = info.get("expert_count", 0)
    expert_used = info.get("expert_used", 0)
    block_count = info.get("block_count", 0)

    if not expert_count or expert_count <= 1:
        return (
            "CPU MoE Layers (-ncmoe)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Модель не MoE.\n"
            "Параметр не эффектен."
        )

    advice = compute_moe_advice(
        info,
        ctx_size,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
    )

    current_est = full_vram_estimate(
        info,
        ctx_size,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
        ncmoe=current_ncmoe,
    )

    lines = []
    lines.append("CPU MoE Layers  (-ncmoe)")
    lines.append("━" * 40)

    lines.append("📐 Архитектура:")
    gpu_note = f" (GPU: {gpu_layers})" if gpu_layers < block_count else " (все на GPU)"
    lines.append(f"   Блоки: {block_count}{gpu_note}")
    lines.append(f"   Эксперты: {expert_count} все, {expert_used} акт/токен")
    if expert_used:
        inactive = expert_count - expert_used
        inactive_pct = int(inactive / expert_count * 100)
        lines.append(f"   Спящих: {inactive} ({inactive_pct}%)")
    if advice.vram_without_ncmoe.expert_layer_mib > 0:
        lines.append(
            f"   ~{advice.vram_without_ncmoe.expert_layer_mib:.0f} MiB/слой неакт."
        )

    lines.append("")
    lines.append(f"📊 VRAM (ncmoe={current_ncmoe}):")

    ref_vram = vram_budget_gib or max(current_est.total_gib * 1.2, 8.0)
    lines.append(f"   Веса:    {current_est.model_vram_gib:6.2f} GiB")
    lines.append(
        f"   KV:     {current_est.kv_cache_gib:6.2f} GiB"
        f" ({current_est.kv_per_1k_ctx_mib:.0f} MiB/1K)"
    )
    lines.append(f"   Overhead:{current_est.overhead_gib:6.2f} GiB")
    lines.append(f"   {'─' * 30}")
    lines.append(
        f"   Итого:  {current_est.total_gib:6.2f} GiB  "
        + _vram_bar(current_est.total_gib, ref_vram)
    )

    if vram_budget_gib:
        margin = vram_budget_gib - current_est.total_gib
        if margin >= 0:
            lines.append(f"   Запас: {margin:.2f} GiB")
        else:
            lines.append(f"   Превыш: {abs(margin):.2f} GiB")

    lines.append("")
    lines.append("📉 ncmoe → VRAM:")
    lines.append(f"   {'ncmoe':>6} {'Веса':>8} {'Итого':>8} {'Экон.':>8}")
    lines.append(f"   {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")

    max_ncmoe = max(0, int(block_count * (expert_count - expert_used) / expert_count))
    steps = _table_steps(advice.recommended_ncmoe, max_ncmoe, current_ncmoe)

    for step in steps:
        est = full_vram_estimate(
            info,
            ctx_size,
            gpu_layers,
            cache_type_k,
            cache_type_v,
            flash_attn,
            parallel_slots,
            ncmoe=step,
        )
        saved = max(0.0, advice.vram_without_ncmoe.total_gib - est.total_gib)
        marker = ""
        if step == current_ncmoe:
            marker = " ◄ тек."
        elif step == advice.recommended_ncmoe:
            marker = " ◄ рек."

        lines.append(
            f"   {step:>6}  {est.model_vram_gib:>7.2f}G"
            f" {est.total_gib:>7.2f}G"
            f" -{saved:>6.2f}G"
            f"{marker}"
        )

    lines.append("")
    lines.append("💡 Анализ:")
    for reason in advice.reasoning:
        lines.append(f"   {reason}")

    if advice.warning:
        lines.append(f"⚠️  {advice.warning}")

    return "\n".join(lines)


def build_ctx_tooltip(
    info: Dict[str, Any],
    current_ctx: int,
    gpu_layers: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    vram_budget_gib: float = 0.0,
) -> str:
    """Расширенный tooltip для ctx_size."""
    from src.core.context_advisor import compute_large_ctx_advice
    from src.core.vram_estimator import estimate_kv_cache

    native_ctx = info.get("context_length", 4096) or 4096
    block_count = info.get("block_count", 0)
    head_count = info.get("head_count", 0)
    emb_len = info.get("embedding_length", 0)
    arch = info.get("architecture", "?")
    quant = info.get("quant", "?")
    rec_ctx = info.get("recommended_ctx", 0)

    advice = compute_large_ctx_advice(
        info,
        current_ctx,
        gpu_layers,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
        vram_budget_gib,
    )

    lines = ["Context Size  (-c)", "━" * 42]

    lines.append(f"Model: {arch} | {quant}")
    lines.append(f"Native ctx: {native_ctx:,}")
    if rec_ctx:
        lines.append(f"Recommended for {quant}: {rec_ctx:,}")

    lines.append("")
    if current_ctx <= native_ctx:
        lines.append(f"Current ctx: {current_ctx:,} - within native")
    elif current_ctx <= native_ctx * 2:
        lines.append(f"Current ctx: {current_ctx:,} - moderate exceed")
    elif current_ctx <= native_ctx * 4:
        lines.append(f"Current ctx: {current_ctx:,} - significant exceed")
    else:
        lines.append(f"Current ctx: {current_ctx:,} - EXTREME exceed")

    lines.append("")
    lines.append("KV-cache:")

    kv_gib = estimate_kv_cache(
        block_count,
        head_count,
        emb_len,
        current_ctx,
        cache_type_k,
        cache_type_v,
        flash_attn,
        parallel_slots,
    )
    kv_per_1k = (kv_gib / current_ctx * 1000 * 1024) if current_ctx else 0

    lines.append(f"  At ctx={current_ctx:,}: ~{kv_gib:.2f} GiB")
    lines.append(f"  Per 1K tokens: ~{kv_per_1k:.1f} MiB")
    lines.append(f"  Type: K={cache_type_k}, V={cache_type_v}")
    if flash_attn:
        lines.append("  Flash Attn: -30%")

    lines.append("")
    lines.append("KV by context:")
    lines.append(f"  {'Ctx':>10} {'KV(GiB)':>9} {'Status':}")
    lines.append(f"  {'-' * 10} {'-' * 9} {'-' * 20}")

    ctx_steps = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    ctx_steps = sorted(set(ctx_steps) | {current_ctx, native_ctx})
    ctx_steps = [c for c in ctx_steps if c <= max(current_ctx * 2, 262144)]

    for ctx_step in ctx_steps:
        kv = estimate_kv_cache(
            block_count,
            head_count,
            emb_len,
            ctx_step,
            cache_type_k,
            cache_type_v,
            flash_attn,
            parallel_slots,
        )
        marker = ""
        if ctx_step == current_ctx:
            marker = " << current"
        elif ctx_step == native_ctx:
            marker = " << native"

        if ctx_step <= native_ctx:
            status = f"OK{marker}"
        elif ctx_step <= native_ctx * 2:
            status = f"mod{marker}"
        else:
            status = f"WARN{marker}"

        lines.append(f"  {ctx_step:>10,} {kv:>8.2f}G  {status}")

    if advice.needs_rope_scaling:
        lines.append("")
        lines.append("RoPE params needed:")
        rope_args = advice.rope_config.to_args()
        if rope_args:
            it = iter(rope_args)
            for flag, val in zip(it, it):
                lines.append(f"  {flag} {val}")
        lines.append("Add to 'Extra params'")

    if advice.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in advice.warnings:
            lines.append(f"  {w}")

    if advice.info_messages:
        lines.append("")
        lines.append("Recommendations:")
        for msg in advice.info_messages:
            lines.append(f"  {msg}")

    if advice.needs_rope_scaling and advice.rope_config.to_args():
        lines.append("")
        lines.append("━" * 42)
        lines.append("Extra params string:")
        lines.append(f"  {' '.join(advice.rope_config.to_args())}")

    lines.append("")
    lines.append("━" * 42)
    lines.append("Context ranges:")
    lines.append("  <= native  -> max quality")
    lines.append("  x1-2      -> linear RoPE")
    lines.append("  x2-8      -> YaRN RoPE")
    lines.append("  >x8       -> quality loss")
    lines.append("")
    lines.append("  ctx >= 32K: enable Flash Attn")
    lines.append("  ctx >= 32K: quantize KV (q8_0/q4_0)")
    lines.append("  ctx >= 64K: ubatch 512-1024")
    lines.append("  ctx >= 128K: use --ctx-checkpoints")

    return "\n".join(lines)


def _table_steps(
    recommended: int,
    max_ncmoe: int,
    current: int,
) -> list[int]:
    """Формирование шагов для таблицы ncmoe."""
    steps = {0, recommended, current}
    if max_ncmoe > 0:
        for fraction in (0.25, 0.5, 0.75):
            steps.add(int(max_ncmoe * fraction))
    steps.add(max_ncmoe)
    return sorted(s for s in steps if 0 <= s <= max_ncmoe)
