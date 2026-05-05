"""Советник по настройке большого контекста."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.vram_estimator import (
    VRAMEstimate,
    estimate_kv_cache,
    full_vram_estimate,
)

_NATIVE_LONG_CTX_ARCHITECTURES = frozenset(
    {
        "llama",
        "qwen2",
        "mistral",
        "phi3",
        "gemma2",
        "deepseek2",
        "command-r",
        "cohere",
        "internlm2",
    }
)

_NEEDS_ROPE_SCALING = frozenset(
    {
        "llama2",
        "falcon",
        "mpt",
        "rwkv",
        "gpt2",
        "mistral_v01",
    }
)

_ROPE_BASE_BY_CTX: List[Tuple[int, float]] = [
    (8_192, 10_000.0),
    (16_384, 20_000.0),
    (32_768, 50_000.0),
    (65_536, 100_000.0),
    (131_072, 500_000.0),
    (262_144, 2_000_000.0),
    (524_288, 10_000_000.0),
]


@dataclass
class RoPEConfig:
    """Конфигурация RoPE масштабирования."""

    scaling_type: str = "none"
    freq_base: float = 0.0
    freq_scale: float = 1.0
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    orig_ctx: int = 0

    def to_args(self) -> List[str]:
        args = []
        if self.scaling_type not in ("none", ""):
            args += ["--rope-scaling", self.scaling_type]
        if self.freq_base > 0:
            args += ["--rope-freq-base", f"{self.freq_base:.1f}"]
        if self.freq_scale != 1.0:
            args += ["--rope-freq-scale", f"{self.freq_scale:.6f}"]
        if self.scaling_type == "yarn":
            if self.yarn_ext_factor >= 0:
                args += ["--yarn-ext-factor", f"{self.yarn_ext_factor:.2f}"]
            args += ["--yarn-attn-factor", f"{self.yarn_attn_factor:.2f}"]
            args += ["--yarn-beta-fast", f"{self.yarn_beta_fast:.1f}"]
            args += ["--yarn-beta-slow", f"{self.yarn_beta_slow:.1f}"]
            if self.orig_ctx > 0:
                args += ["--yarn-orig-ctx", str(self.orig_ctx)]
        return args


@dataclass
class KVCacheConfig:
    """Конфигурация KV-cache для большого контекста."""

    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    ctx_checkpoints: int = -1
    cache_ram_mib: int = -2
    defrag_thold: float = 0.1


@dataclass
class LargeContextAdvice:
    """Полные рекомендации для большого контекста."""

    ctx_size: int
    native_max_ctx: int
    needs_rope_scaling: bool
    rope_config: RoPEConfig
    kv_config: KVCacheConfig
    flash_attn_required: bool
    ubatch_recommendation: int
    vram_estimate: Optional[VRAMEstimate]
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    extra_args_preview: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.extra_args_preview = self.rope_config.to_args()


def _recommend_rope_base(native_ctx: int, target_ctx: int) -> float:
    """Рекомендация rope_freq_base для заданного масштаба."""
    if target_ctx <= native_ctx:
        return 0.0

    for ctx_limit, base in _ROPE_BASE_BY_CTX:
        if target_ctx <= ctx_limit:
            return base
    return _ROPE_BASE_BY_CTX[-1][1] * (target_ctx / _ROPE_BASE_BY_CTX[-1][0])


def _recommend_kv_types(
    ctx_size: int,
    block_count: int,
    head_count: int,
    embedding_length: int,
    flash_attn: bool,
    parallel_slots: int,
    vram_budget_gib: float = 24.0,
    model_vram_gib: float = 0.0,
) -> Tuple[str, str, str]:
    """Рекомендация типов KV-cache."""
    available_for_kv = max(0.0, vram_budget_gib - model_vram_gib - 0.5)

    candidates = [
        ("f16", "f16", "max quality"),
        ("q8_0", "q8_0", "high quality, -50%"),
        ("q4_0", "f16", "K quantized, V precise"),
        ("q4_0", "q8_0", "both quantized"),
        ("q4_0", "q4_0", "max economy"),
        ("iq4_nl", "iq4_nl", "IQ better than q4_0"),
    ]

    for k_type, v_type, reason in candidates:
        kv_gib = estimate_kv_cache(
            block_count,
            head_count,
            embedding_length,
            ctx_size,
            k_type,
            v_type,
            flash_attn,
            parallel_slots,
        )
        if kv_gib <= available_for_kv:
            return k_type, v_type, reason

    return "q4_0", "q4_0", "forced: not enough VRAM"


def compute_large_ctx_advice(
    info: Dict[str, Any],
    target_ctx: int,
    gpu_layers: int,
    current_cache_k: str = "f16",
    current_cache_v: str = "f16",
    flash_attn: bool = True,
    parallel_slots: int = 1,
    vram_budget_gib: float = 0.0,
) -> LargeContextAdvice:
    """Полный анализ параметров для большого контекста."""
    arch = (info.get("architecture") or "").lower()
    native_ctx = info.get("context_length", 4096) or 4096
    block_count = info.get("block_count", 0)
    head_count = info.get("head_count", 0)
    embedding_len = info.get("embedding_length", 0)
    size_gib = info.get("size_gib", 0.0)

    warnings: List[str] = []
    infos: List[str] = []

    is_native_long = arch in _NATIVE_LONG_CTX_ARCHITECTURES
    needs_rope = target_ctx > native_ctx

    if target_ctx > native_ctx * 4:
        warnings.append(
            f"Context {target_ctx:,} exceeds native by "
            f"{target_ctx / native_ctx:.1f}x - quality loss possible"
        )

    rope_cfg = RoPEConfig()

    if not needs_rope:
        infos.append(f"Context {target_ctx:,} within native ({native_ctx:,})")
        infos.append("No RoPE scaling needed")

    elif is_native_long and target_ctx <= native_ctx:
        infos.append(f"Model {arch} supports native {native_ctx:,}")
        infos.append("No additional RoPE params needed")

    elif is_native_long and target_ctx > native_ctx:
        scale = target_ctx / native_ctx
        rope_cfg = RoPEConfig(
            scaling_type="yarn",
            freq_base=_recommend_rope_base(native_ctx, target_ctx),
            yarn_ext_factor=round(0.1 * (scale - 1), 2),
            yarn_attn_factor=max(0.5, 1.0 - 0.1 * (scale - 1)),
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            orig_ctx=native_ctx,
        )
        infos.append(
            f"Model {arch} native up to {native_ctx:,}, requested {target_ctx:,} - YaRN needed"
        )

    else:
        scale = target_ctx / native_ctx
        rec_base = _recommend_rope_base(native_ctx, target_ctx)

        if scale <= 2.0:
            rope_cfg = RoPEConfig(
                scaling_type="linear",
                freq_scale=round(1.0 / scale, 6),
            )
            infos.append(f"Linear RoPE: scale={1 / scale:.3f}")
        else:
            rope_cfg = RoPEConfig(
                scaling_type="yarn",
                freq_base=rec_base,
                yarn_ext_factor=round(0.1 * (scale - 1), 2),
                yarn_attn_factor=max(0.5, 1.0 - 0.05 * (scale - 1)),
                yarn_beta_fast=32.0,
                yarn_beta_slow=1.0,
                orig_ctx=native_ctx,
            )
            infos.append(f"YaRN RoPE: base={rec_base:.0f}")

    fa_required = target_ctx >= 32_768
    if fa_required and not flash_attn:
        warnings.append("Flash Attention STRONGLY recommended for ctx>=32K!")
    elif fa_required:
        infos.append("Flash Attention enabled - critical for large ctx")

    if target_ctx >= 32_768:
        rec_k, rec_v, kv_reason = _recommend_kv_types(
            target_ctx,
            block_count,
            head_count,
            embedding_len,
            flash_attn,
            parallel_slots,
            vram_budget_gib or (size_gib * 1.5),
            size_gib * 0.9,
        )
        kv_cfg = KVCacheConfig(
            cache_type_k=rec_k,
            cache_type_v=rec_v,
        )
        if rec_k != current_cache_k or rec_v != current_cache_v:
            infos.append(f"KV-cache: recommended {rec_k}/{rec_v} ({kv_reason})")
    else:
        kv_cfg = KVCacheConfig(
            cache_type_k=current_cache_k,
            cache_type_v=current_cache_v,
        )

    ctx_checkpoints = -1
    if target_ctx >= 131_072:
        ctx_checkpoints = 4
        infos.append("--ctx-checkpoints 4: saves VRAM at very large context")
    elif target_ctx >= 65_536:
        ctx_checkpoints = 2
        infos.append("--ctx-checkpoints 2: recommended for ctx>=64K")
    kv_cfg.ctx_checkpoints = ctx_checkpoints

    cache_ram_mib = -2
    if target_ctx >= 131_072 and vram_budget_gib > 0:
        kv_gib = estimate_kv_cache(
            block_count,
            head_count,
            embedding_len,
            target_ctx,
            kv_cfg.cache_type_k,
            kv_cfg.cache_type_v,
            flash_attn,
            parallel_slots,
        )
        model_vram = size_gib * min(1.0, gpu_layers / max(block_count, 1)) * 1.05
        if model_vram + kv_gib > vram_budget_gib:
            overflow_gib = (model_vram + kv_gib) - vram_budget_gib
            cache_ram_mib = int(overflow_gib * 1024 * 1.2)
            warnings.append(
                f"KV doesn't fit in VRAM: recommend --cache-ram {cache_ram_mib} "
                f"(overflow ~{overflow_gib:.2f} GiB)"
            )
    kv_cfg.cache_ram_mib = cache_ram_mib

    if target_ctx >= 32_768:
        kv_cfg.defrag_thold = 0.1
        infos.append("--defrag-thold 0.1: recommended for large ctx")

    if target_ctx >= 65_536:
        ubatch_rec = 512
    elif target_ctx >= 32_768:
        ubatch_rec = 1024
    else:
        ubatch_rec = 2048

    vram_est = None
    try:
        vram_est = full_vram_estimate(
            info,
            target_ctx,
            gpu_layers,
            kv_cfg.cache_type_k,
            kv_cfg.cache_type_v,
            flash_attn,
            parallel_slots,
        )
    except Exception:
        pass

    return LargeContextAdvice(
        ctx_size=target_ctx,
        native_max_ctx=native_ctx,
        needs_rope_scaling=needs_rope,
        rope_config=rope_cfg,
        kv_config=kv_cfg,
        flash_attn_required=fa_required,
        ubatch_recommendation=ubatch_rec,
        vram_estimate=vram_est,
        warnings=warnings,
        info_messages=infos,
    )
