"""Подсчёт score для результатов AutoTune."""

from __future__ import annotations

from typing import Any, Dict

from src.core.benchmark_models import BenchmarkResult


_KV_QUALITY_PENALTY = {
    ("f16", "f16"): 0.0,
    ("q8_0", "q8_0"): 1.5,
    ("q4_0", "q8_0"): 5.0,
    ("q4_0", "q4_0"): 9.0,
    ("iq4_nl", "iq4_nl"): 8.0,
}


def _prompt_score(prompt_tok_s: float) -> float:
    # Нормализация без знания глобального максимума: 2000 tok/s ~= 100 баллов.
    return min(max(prompt_tok_s, 0.0) / 20.0, 100.0)


def _memory_margin_score(result: BenchmarkResult) -> float:
    if result.metrics.vram_free_mib > 0:
        return min(result.metrics.vram_free_mib / 256.0, 100.0)
    if result.vram_used_mib > 0:
        # Если свободная VRAM неизвестна, мягко поощряем меньший расход.
        return max(0.0, 100.0 - result.vram_used_mib / 256.0)
    return 50.0


def _stability_score(result: BenchmarkResult) -> float:
    if result.status == "success":
        return 100.0
    if result.status == "failed_timeout":
        return 10.0
    return 0.0


def _failure_penalty(result: BenchmarkResult) -> float:
    if result.status == "success":
        return 0.0
    if result.status == "failed_oom":
        return 1000.0
    if result.status == "failed_timeout":
        return 500.0
    return 750.0


def _context_mismatch_penalty(params: Dict[str, Any], target_key: str) -> float:
    """
    Штраф за риск, который не виден llama-bench.

    Актуальный llama-bench не принимает -c, поэтому для 128K+ он меряет только
    micro-speed на коротком prompt/gen и может выбрать f16 KV, хотя реальный
    llama-server на большом контексте будет сильно медленнее из-за KV/prompt cache.
    """
    ctx_size = int(params.get("ctx_size") or 0)
    if ctx_size < 131072:
        return 0.0

    penalty = 0.0
    kv = (str(params.get("cache_type_k", "")), str(params.get("cache_type_v", "")))
    model_type = str(params.get("model_type", "")).lower()
    if kv == ("f16", "f16") and target_key != "quality_kv":
        penalty += 30.0
    elif model_type == "moe" and target_key not in {"low_vram", "quality_kv"}:
        # Для MoE на 128K q8 обычно намного безопаснее по качеству long-context,
        # а по логам q4/q4 почти не даёт выигрыша скорости. q4 оставляем для Low VRAM.
        if kv == ("q4_0", "q4_0"):
            penalty += 5.0
        elif kv == ("q4_0", "q8_0"):
            penalty += 8.0

    if int(params.get("cache_ram", -2)) != 0:
        penalty += 10.0
    if int(params.get("ctx_checkpoints", -1)) != 0:
        penalty += 6.0

    return penalty


def score_result(result: BenchmarkResult, params: Dict[str, Any], target: str) -> float:
    """Возвращает score. Неуспешные прогоны получают 0."""
    if result.status != "success":
        result.score = 0.0
        return 0.0

    gen = max(result.generation_tok_s, 0.0)
    prompt = _prompt_score(result.prompt_tok_s)
    memory = _memory_margin_score(result)
    stability = _stability_score(result)
    load_penalty = min(max(result.load_time_sec, 0.0) / 10.0, 15.0)
    failure_penalty = _failure_penalty(result)

    target_key = (target or "balanced").strip().lower().replace(" ", "_")

    if target_key == "max_speed":
        score = gen * 0.65 + prompt * 0.25 - load_penalty - failure_penalty
    elif target_key == "low_vram":
        score = memory * 0.45 + gen * 0.30 + prompt * 0.15 + stability * 0.10
    elif target_key == "quality_kv":
        kv_key = (str(params.get("cache_type_k", "")), str(params.get("cache_type_v", "")))
        penalty = _KV_QUALITY_PENALTY.get(kv_key, 6.0)
        score = gen * 0.40 + prompt * 0.20 + memory * 0.15 + stability * 0.10 - penalty
    elif target_key == "moe_optimized":
        score = gen * 0.50 + prompt * 0.20 + memory * 0.20 + stability * 0.10
    else:
        score = gen * 0.45 + prompt * 0.25 + memory * 0.20 + stability * 0.10

    score -= _context_mismatch_penalty(params, target_key)

    result.score = round(max(score, 0.0), 3)
    return result.score
