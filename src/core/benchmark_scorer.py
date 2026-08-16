"""Подсчёт score для результатов AutoTune."""

from __future__ import annotations

from typing import Any, Dict

from src.core.benchmark_models import BenchmarkResult

# --- Шкалы нормализации -------------------------------------------------------
# Нормализация без знания глобального максимума: 2000 tok/s ~= 100 баллов
# (верх современных GPU попадает в шкалу, не обрезаясь сразу).
PROMPT_SCORE_FULL_SCALE_TOK_S = 2000.0
# 256 MiB свободной VRAM ~= 100 баллов memory-margin: достаточно для рабочего
# окна контекста, дальше запас почти не добавляет ценности.
MEMORY_MARGIN_FULL_SCALE_MIB = 256.0
# Нет данных по VRAM вовсе — нейтральная середина шкалы.
MEMORY_MARGIN_UNKNOWN = 50.0
# Время загрузки модели: 1 балл штрафа за каждые 10 с, потолок 15 баллов
# (медленная загрузка важна, но не должна вытеснить скорость генерации).
LOAD_PENALTY_SECONDS_PER_POINT = 10.0
LOAD_PENALTY_MAX = 15.0
# Timeout всё же выдал числа — небольшой кредит стабильности.
STABILITY_SCORE_TIMEOUT = 10.0

# --- Штрафы за неуспех --------------------------------------------------------
# OOM — худший исход (кандидат непригоден в принципе), прочие сбои — между
# OOM и timeout.
FAILURE_PENALTY_OOM = 1000.0
FAILURE_PENALTY_TIMEOUT = 500.0
FAILURE_PENALTY_OTHER = 750.0

# --- Контекст 128K+: риски, которых не видит llama-bench ---------------------
LONG_CTX_THRESHOLD = 131072
# f16 KV на большом контексте разрастается и режет скорость реального сервера.
LONG_CTX_F16_KV_PENALTY = 30.0
# Для MoE на 128K q8 обычно безопаснее по качеству long-context, а q4/q4 почти
# не даёт выигрыша скорости — q4 оставляем только для Low VRAM.
LONG_CTX_MOE_Q4Q4_PENALTY = 5.0
LONG_CTX_MOE_Q4Q8_PENALTY = 8.0
# cache_ram/checkpoints на 128K+ конфликтуют с KV-бюджетом.
LONG_CTX_CACHE_RAM_PENALTY = 10.0
LONG_CTX_CHECKPOINTS_PENALTY = 6.0

# --- Ресурсные риски ----------------------------------------------------------
RISK_BLOCKED_PENALTY = 80.0
RISK_HIGH_PENALTY = 22.0
RISK_HIGH_PENALTY_LOW_VRAM = 12.0
RISK_MEDIUM_PENALTY = 7.0
RISK_MEDIUM_PENALTY_MAX_SPEED = 3.0
# >=99% VRAM — сервер скорее всего упадёт на рабочей нагрузке.
VRAM_USAGE_CRITICAL_PCT = 99.0
VRAM_USAGE_CRITICAL_PENALTY = 12.0
# Для balanced 95% — слишком близко к пределу; max_speed готов рисковать.
VRAM_USAGE_HIGH_PCT = 95.0
VRAM_USAGE_HIGH_BALANCED_PENALTY = 5.0
# KV-пресет с неизвестной парой квантований — умеренный штраф по умолчанию.
KV_QUALITY_DEFAULT_PENALTY = 6.0

_KV_QUALITY_PENALTY = {
    ("f16", "f16"): 0.0,
    ("q8_0", "q8_0"): 1.5,
    ("q4_0", "q8_0"): 5.0,
    ("q4_0", "q4_0"): 9.0,
    ("iq4_nl", "iq4_nl"): 8.0,
}


def _prompt_score(prompt_tok_s: float) -> float:
    scale = PROMPT_SCORE_FULL_SCALE_TOK_S / 100.0
    return min(max(prompt_tok_s, 0.0) / scale, 100.0)


def _memory_margin_score(result: BenchmarkResult) -> float:
    if result.metrics.vram_free_mib > 0:
        return min(result.metrics.vram_free_mib / MEMORY_MARGIN_FULL_SCALE_MIB, 100.0)
    if result.vram_used_mib > 0:
        # Если свободная VRAM неизвестна, мягко поощряем меньший расход.
        return max(
            0.0, 100.0 - result.vram_used_mib / MEMORY_MARGIN_FULL_SCALE_MIB
        )
    return MEMORY_MARGIN_UNKNOWN


def _stability_score(result: BenchmarkResult) -> float:
    if result.status == "success":
        return 100.0
    if result.status == "failed_timeout":
        return STABILITY_SCORE_TIMEOUT
    return 0.0


def _failure_penalty(result: BenchmarkResult) -> float:
    if result.status == "success":
        return 0.0
    if result.status == "failed_oom":
        return FAILURE_PENALTY_OOM
    if result.status == "failed_timeout":
        return FAILURE_PENALTY_TIMEOUT
    return FAILURE_PENALTY_OTHER


def _context_mismatch_penalty(params: Dict[str, Any], target_key: str) -> float:
    """
    Штраф за риск, который не виден llama-bench.

    Актуальный llama-bench не принимает -c, поэтому для 128K+ он меряет только
    micro-speed на коротком prompt/gen и может выбрать f16 KV, хотя реальный
    llama-server на большом контексте будет сильно медленнее из-за KV/prompt cache.
    """
    ctx_size = int(params.get("ctx_size") or 0)
    if ctx_size < LONG_CTX_THRESHOLD:
        return 0.0

    penalty = 0.0
    kv = (str(params.get("cache_type_k", "")), str(params.get("cache_type_v", "")))
    model_type = str(params.get("model_type", "")).lower()
    if kv == ("f16", "f16") and target_key != "quality_kv":
        penalty += LONG_CTX_F16_KV_PENALTY
    elif model_type == "moe" and target_key not in {"low_vram", "quality_kv"}:
        if kv == ("q4_0", "q4_0"):
            penalty += LONG_CTX_MOE_Q4Q4_PENALTY
        elif kv == ("q4_0", "q8_0"):
            penalty += LONG_CTX_MOE_Q4Q8_PENALTY

    if int(params.get("cache_ram", -2)) != 0:
        penalty += LONG_CTX_CACHE_RAM_PENALTY
    if int(params.get("ctx_checkpoints", -1)) != 0:
        penalty += LONG_CTX_CHECKPOINTS_PENALTY

    return penalty


def _resource_risk_penalty(params: Dict[str, Any], target_key: str) -> float:
    risk = str(params.get("_risk") or "").lower()
    vram_pct = float(params.get("_vram_pct") or 0.0)
    penalty = 0.0
    if risk == "blocked":
        penalty += RISK_BLOCKED_PENALTY
    elif risk == "high":
        penalty += RISK_HIGH_PENALTY if target_key != "low_vram" else RISK_HIGH_PENALTY_LOW_VRAM
    elif risk == "medium":
        penalty += (
            RISK_MEDIUM_PENALTY if target_key != "max_speed" else RISK_MEDIUM_PENALTY_MAX_SPEED
        )
    if vram_pct >= VRAM_USAGE_CRITICAL_PCT:
        penalty += VRAM_USAGE_CRITICAL_PENALTY
    elif vram_pct >= VRAM_USAGE_HIGH_PCT and target_key == "balanced":
        penalty += VRAM_USAGE_HIGH_BALANCED_PENALTY
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
    load_penalty = min(
        max(result.load_time_sec, 0.0) / LOAD_PENALTY_SECONDS_PER_POINT,
        LOAD_PENALTY_MAX,
    )
    failure_penalty = _failure_penalty(result)

    target_key = (target or "balanced").strip().lower().replace(" ", "_")

    if target_key == "max_speed":
        score = gen * 0.65 + prompt * 0.25 - load_penalty - failure_penalty
    elif target_key == "low_vram":
        score = memory * 0.45 + gen * 0.30 + prompt * 0.15 + stability * 0.10
    elif target_key == "quality_kv":
        kv_key = (str(params.get("cache_type_k", "")), str(params.get("cache_type_v", "")))
        penalty = _KV_QUALITY_PENALTY.get(kv_key, KV_QUALITY_DEFAULT_PENALTY)
        score = gen * 0.40 + prompt * 0.20 + memory * 0.15 + stability * 0.10 - penalty
    elif target_key == "moe_optimized":
        score = gen * 0.50 + prompt * 0.20 + memory * 0.20 + stability * 0.10
    else:
        score = gen * 0.45 + prompt * 0.25 + memory * 0.20 + stability * 0.10

    score -= _context_mismatch_penalty(params, target_key)
    score -= _resource_risk_penalty(params, target_key)

    result.score = round(max(score, 0.0), 3)
    return result.score
