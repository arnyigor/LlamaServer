"""Построение планов AutoTune benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate
from src.core.moe_advisor import compute_moe_advice
from src.core.vram_estimator import full_vram_estimate
from src.utils.subprocess_utils import no_console_kwargs

# --- Доли VRAM-бюджета для поиска ncmoe ---------------------------------------
# Цель — 94% бюджета: запас под KV-фрагментацию и потребности монитора/десктопа;
# 98% — жёсткий предел, выше которого кандидат штрафуется.
VRAM_TARGET_FRACTION = 0.94
VRAM_HARD_LIMIT_FRACTION = 0.98
# Окно поиска ncmoe вокруг текущего/рекомендуемого значения: эстиматор для
# MoE GGUF приблизительный, поэтому не прыгаем сразу к "все эксперты на CPU".
NCMOE_SEARCH_ANCHOR_PAD = 6
NCMOE_SEARCH_MAX_FRACTION = 0.35
# Штраф за превышение hard limit: доминирует над отклонением от цели,
# чтобы кандидат над пределом не выигрывал у "чуть недобравшего".
NCMOE_OVER_LIMIT_PENALTY = 3.0
# Соседи лучшего значения: локальная проверка ±2 слоя.
NCMOE_NEIGHBOR_SPAN = 2
# Dense-модели на 128K+ упираются в KV memory — стартуем с меньшего ubatch.
DENSE_LONG_CTX_UBATCH_CAP = 256
LONG_CTX_THRESHOLD = 131072
# Дефолтный порог early stop: падение скорости на 3% от пика.
EARLY_STOP_DROP_PCT_DEFAULT = 3.0

_TIME_BUDGETS = {
    "smart": 10 * 60,
    "quick": 15 * 60,
    "normal": 45 * 60,
    "deep": 120 * 60,
}

_MAX_RUNS = {
    "smart": 10,
    "quick": 12,
    "normal": 60,
    "deep": 120,
}


def _mode_key(mode: str) -> str:
    return (mode or "quick").strip().lower()


def _target_key(target: str) -> str:
    return (target or "balanced").strip().lower().replace(" ", "_")


def _is_auto_target(target: str) -> bool:
    return _target_key(target) in {"auto", "smart"}


def _ctx_from_settings(settings: Any, model_info: Dict[str, Any]) -> int:
    ctx = int(getattr(settings, "ctx_size", 0) or 0)
    if ctx > 0:
        return ctx
    rec = int(model_info.get("recommended_ctx") or 0)
    if rec > 0:
        return rec
    native = int(model_info.get("context_length") or 0)
    if native > 0:
        return min(native, 32768)
    return 8192


def _threads_candidates(settings: Any) -> List[int]:
    logical = max(os.cpu_count() or 4, 1)
    current = int(getattr(settings, "threads", logical) or logical)
    vals = [current, max(1, logical // 2), max(1, logical - 1)]
    result: List[int] = []
    for v in vals:
        if v not in result:
            result.append(v)
    return result


def _is_moe(model_info: Dict[str, Any]) -> bool:
    return int(model_info.get("expert_count") or 0) > 1


def _is_mtp_model(model_info: Dict[str, Any]) -> bool:
    text = " ".join(
        str(model_info.get(k) or "") for k in ("path", "name", "display", "_model_path")
    ).lower()
    return "mtp" in text


def _kv_candidates(
    target: str,
    mode: str,
    ctx_size: int = 0,
    model_info: Dict[str, Any] | None = None,
) -> List[Tuple[str, str]]:
    key = _target_key(target)
    huge_context = int(ctx_size or 0) >= 131072
    dense_model = not _is_moe(model_info or {})
    quick = _mode_key(mode) == "quick"
    if dense_model and key != "quality_kv":
        # Для dense-моделей f16/f16 имеет смысл только если ТОЧНО влезает в VRAM
        # с запасом на контекст. Практически всегда безопаснее q8_0/q8_0,
        # а для 128K+ — q4_0/q4_0 как рабочая точка.
        if huge_context:
            vals = [("q4_0", "q4_0"), ("q8_0", "q8_0")]
            if not quick:
                vals.append(("q4_0", "q8_0"))
        else:
            vals = [("q8_0", "q8_0"), ("q4_0", "q4_0")]
            if not quick:
                vals.insert(1, ("f16", "f16"))
    elif huge_context and key != "quality_kv":
        # llama-bench не принимает -c и не валидирует реальный большой KV-cache.
        # Для 128K+ нельзя давать f16/f16 ранний приоритет: он часто быстрый в
        # микробенче, но проваливается/замедляется в реальном llama-server.
        vals = [("q8_0", "q8_0"), ("q4_0", "q4_0"), ("f16", "f16")]
        if not quick:
            vals.insert(2, ("q4_0", "q8_0"))
    elif _is_mtp_model(model_info or {}) and key != "low_vram":
        # MTP план стартует с Q8 KV: он обычно ближе к рабочей VRAM-цели
        # на 32K+, а f16 оставляем как quality-соседа для проверки.
        vals = [("q8_0", "q8_0"), ("f16", "f16"), ("q4_0", "q4_0")]
    elif key == "quality_kv":
        vals = [("f16", "f16"), ("q8_0", "q8_0")]
        if not quick:
            vals.append(("q4_0", "q8_0"))
    elif key == "low_vram":
        vals = [("q4_0", "q4_0"), ("q8_0", "q8_0")]
        if not quick:
            vals[1:1] = [("iq4_nl", "iq4_nl"), ("q4_0", "q8_0")]
    else:
        vals = [("q8_0", "q8_0"), ("f16", "f16"), ("q4_0", "q4_0")]
        if not quick:
            vals.insert(2, ("q4_0", "q8_0"))
    if _mode_key(mode) != "quick":
        vals.append(("iq4_nl", "iq4_nl"))
    return vals


def _block_count(model_info: Dict[str, Any]) -> int:
    return max(0, int(model_info.get("block_count") or 0))


def _full_ngl_for_model(model_info: Dict[str, Any]) -> int:
    # Если GGUF metadata содержит число transformer-блоков, AutoTune должен
    # показывать и сохранять реальное NGL (например 40), а не llama.cpp sentinel
    # 99. Sentinel оставляем только как fallback для моделей без block_count.
    blocks = _block_count(model_info)
    return blocks if blocks > 0 else 99


def _clamp_layer_value(value: int, model_info: Dict[str, Any]) -> int:
    value = max(0, int(value))
    blocks = _block_count(model_info)
    return min(value, blocks) if blocks > 0 else value


def _gpu_layers_for_estimate(settings: Any, model_info: Dict[str, Any]) -> int:
    if getattr(settings, "gpu_auto", True):
        return _full_ngl_for_model(model_info)
    return _clamp_layer_value(int(getattr(settings, "gpu_layers", 0) or 0), model_info)


def _ngl_candidates(
    settings: Any, model_info: Dict[str, Any], mode: str, target: str
) -> List[int]:
    """Small NGL search space: full offload plus nearby/current fallbacks."""
    full = _full_ngl_for_model(model_info)
    current = (
        full
        if getattr(settings, "gpu_auto", True)
        else _clamp_layer_value(
            int(getattr(settings, "gpu_layers", 0) or 0), model_info
        )
    )
    vals = [current, full]

    # Для dense-моделей промежуточный ngl почти всегда хуже полного или 0
    # из-за CPU↔GPU bottleneck. Тестируем только full и low_vram fallback.
    is_dense = not _is_moe(model_info)
    if is_dense:
        if _target_key(target) == "low_vram" and full > 0:
            vals.append(0)
    else:
        # MoE может выиграть от частичного offload + ncmoe тюнинга
        if full > 8:
            vals.append(max(0, full - 4))
        if _mode_key(mode) != "quick" and full > 16:
            vals.extend([max(0, full - 8), max(0, full // 2)])
        if _target_key(target) == "low_vram" and full > 0:
            vals.append(0)

    result: List[int] = []
    for v in vals:
        v = _clamp_layer_value(int(v), model_info)
        if v not in result:
            result.append(v)
    return result


def _clamp_ncmoe_value(
    value: int, model_info: Dict[str, Any], ngl: int | None = None
) -> int:
    if value < 0:
        return -1
    blocks = _block_count(model_info)
    max_value = blocks if blocks > 0 else int(value)
    if ngl is not None and ngl >= 0:
        # -ncmoe отвечает за MoE-слои, которые иначе попали бы в GPU-offload.
        # При частичном NGL нет смысла тестировать ncmoe больше числа GPU layers.
        max_value = min(max_value, int(ngl))
    return max(0, min(int(value), max_value))


def _recommended_ncmoe(
    settings: Any, model_info: Dict[str, Any], ctx_size: int, ngl: int | None = None
) -> int:
    if not _is_moe(model_info):
        # Для dense-моделей -ncmoe неприменим. Не переносим сюда stale-значение
        # из UI, оставшееся после MoE-модели.
        return -1
    effective_ngl = (
        _gpu_layers_for_estimate(settings, model_info)
        if ngl is None
        else _clamp_layer_value(ngl, model_info)
    )
    mtp_model = _is_mtp_model(model_info)
    advice = compute_moe_advice(
        model_info,
        ctx_size,
        effective_ngl,
        "q8_0",
        "q8_0",
        bool(getattr(settings, "flash_attn", True)),
        1 if mtp_model else max(1, int(getattr(settings, "parallel_slots", 1) or 1)),
    )
    return _clamp_ncmoe_value(int(advice.recommended_ncmoe), model_info, effective_ngl)


def _detect_total_vram_gib() -> float:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            encoding="utf-8",
            errors="ignore",
            **no_console_kwargs(),
        )
        values = [
            float(x.strip()) / 1024.0
            for x in (proc.stdout or "").splitlines()
            if x.strip()
        ]
        return max(values) if values else 0.0
    except Exception:
        return 0.0


def _detect_system_ram_gib() -> float:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullTotalPhys) / (1024**3)
        except Exception:
            return 0.0
    # os.sysconf есть только на POSIX — mypy на Windows ругается на прямой вызов.
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return 0.0
    try:
        pages = sysconf("SC_PHYS_PAGES")
        page_size = sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return 0.0


def _vram_budget_gib(settings: Any, model_info: Dict[str, Any]) -> float:
    for source in (settings, model_info):
        for key in ("vram_budget_gib", "vram_gib", "gpu_vram_gib"):
            value = (
                getattr(source, key, None) if source is settings else source.get(key)
            )
            try:
                if value and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                pass
    return _detect_total_vram_gib()


def _auto_target_for_constraints(
    settings: Any, model_info: Dict[str, Any], ctx_size: int
) -> str:
    budget = _vram_budget_gib(settings, model_info)
    base = _base_params(settings, model_info, ctx_size)
    estimated = _estimate_vram_gib(base, model_info) if model_info else 0.0
    pct = estimated / budget if budget > 0 and estimated > 0 else 0.0
    if pct >= 0.92 or ctx_size >= 131072:
        return "low_vram"
    if _is_moe(model_info) or _is_mtp_model(model_info):
        return "balanced"
    if pct > 0 and pct <= 0.72:
        return "max_speed"
    return "balanced"


def _candidate_vram_metadata(
    params: Dict[str, Any], model_info: Dict[str, Any], budget_gib: float
) -> Dict[str, Any]:
    estimated = _estimate_vram_gib(params, model_info) if model_info else 0.0
    pct = (estimated / budget_gib * 100.0) if budget_gib > 0 and estimated > 0 else 0.0
    if pct >= 103.0:
        risk = "blocked"
        note = "estimated above VRAM capacity"
    elif pct >= 97.0:
        risk = "high"
        note = "very low VRAM margin"
    elif pct >= 90.0:
        risk = "medium"
        note = "tight VRAM margin"
    elif estimated > 0:
        risk = "low"
        note = "fits estimated VRAM budget"
    else:
        risk = "unknown"
        note = "VRAM estimate unavailable"
    return {
        "_estimated_vram_gib": round(estimated, 2) if estimated > 0 else 0.0,
        "_vram_pct": round(pct, 1) if pct > 0 else 0.0,
        "_risk": risk,
        "_risk_note": note,
    }


def _annotate_candidates(
    candidates: List[BenchmarkCandidate],
    model_info: Dict[str, Any],
    budget_gib: float,
) -> None:
    for candidate in candidates:
        candidate.params.update(
            _candidate_vram_metadata(candidate.params, model_info, budget_gib)
        )


def _filter_smart_candidates(
    candidates: List[BenchmarkCandidate], max_runs: int
) -> List[BenchmarkCandidate]:
    if not candidates:
        return []
    kept: List[BenchmarkCandidate] = []
    blocked: List[BenchmarkCandidate] = []
    for candidate in candidates:
        if candidate.params.get("_risk") == "blocked":
            blocked.append(candidate)
        else:
            kept.append(candidate)
    if not kept:
        kept = candidates[:1]
    # Smart tune should cover major dimensions first, not spend all runs in one stage.
    preferred_order = {
        "baseline": 0,
        "kv": 1,
        "batch": 2,
        "ubatch": 3,
        "threads": 4,
        "moe_vram": 5,
        "ngl": 6,
        "moe": 7,
        "memory": 8,
    }
    kept.sort(
        key=lambda c: (
            preferred_order.get(c.stage, 20),
            {"low": 0, "medium": 1, "unknown": 2, "high": 3}.get(
                str(c.params.get("_risk")), 4
            ),
            c.id,
        )
    )
    selected = kept[: max(1, max_runs)]
    for index, candidate in enumerate(selected, start=1):
        candidate.id = f"run_{index:03d}"
    return selected


def _smart_rescue_candidates(
    settings: Any, model_info: Dict[str, Any], ctx_size: int
) -> List[BenchmarkCandidate]:
    base = _base_params(settings, model_info, ctx_size)
    seen: set = set()
    candidates: List[BenchmarkCandidate] = []
    ctx_values = []
    for value in (ctx_size // 2, 65536, 32768, 16384, 8192):
        if value > 0 and value < ctx_size and value not in ctx_values:
            ctx_values.append(value)
    ngl_values = _ngl_candidates(settings, model_info, "quick", "low_vram")
    if 0 not in ngl_values:
        ngl_values.append(0)
    for ctx in ctx_values:
        for ngl in ngl_values[:3]:
            p = dict(
                base,
                ctx_size=ctx,
                ngl=ngl,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                batch_size=512,
                ubatch_size=256,
                parallel_slots=1,
                kv_unified=False,
                ncmoe=-1 if not _is_moe(model_info) else base.get("ncmoe", -1),
            )
            _append_unique(
                candidates,
                seen,
                p,
                f"Smart fallback ctx={ctx:,} ngl={ngl}",
                "fallback",
            )
    return candidates


def _plan_constraints(
    settings: Any,
    model_info: Dict[str, Any],
    ctx_size: int,
    target: str,
    candidates: List[BenchmarkCandidate],
    requested_target: str,
) -> Dict[str, Any]:
    budget = _vram_budget_gib(settings, model_info)
    ram_gib = _detect_system_ram_gib()
    logical = max(os.cpu_count() or 0, 0)
    baseline = candidates[0].params if candidates else {}
    estimated = float(baseline.get("_estimated_vram_gib") or 0.0)
    vram_pct = float(baseline.get("_vram_pct") or 0.0)
    blocked = sum(1 for c in candidates if c.params.get("_risk") == "blocked")
    high = sum(1 for c in candidates if c.params.get("_risk") == "high")
    model_kind = "MoE" if _is_moe(model_info) else "Dense"
    if _is_mtp_model(model_info):
        model_kind += " + MTP"
    notes = [
        f"{model_kind} model, ctx={ctx_size:,}",
        f"target={target}" + (" (auto-selected)" if _is_auto_target(requested_target) else ""),
    ]
    if budget > 0 and estimated > 0:
        margin = budget - estimated
        notes.append(
            f"baseline VRAM estimate {estimated:.2f}/{budget:.2f} GiB ({vram_pct:.1f}%, margin {margin:.2f} GiB)"
        )
    elif estimated > 0:
        notes.append(f"baseline VRAM estimate {estimated:.2f} GiB; GPU capacity unknown")
    else:
        notes.append("VRAM estimate unavailable; plan keeps conservative defaults")
    if blocked:
        notes.append(f"{blocked} candidate(s) were above estimated VRAM capacity")
    if high:
        notes.append(f"{high} candidate(s) have very low VRAM margin")
    if _is_moe(model_info):
        notes.append("MoE search includes ncmoe values clamped to active GPU layers")
    if _is_mtp_model(model_info):
        notes.append("MTP search keeps one slot and Q8-first KV baseline")
    return {
        "gpu_vram_gib": round(budget, 2) if budget > 0 else 0.0,
        "system_ram_gib": round(ram_gib, 2) if ram_gib > 0 else 0.0,
        "cpu_threads": logical,
        "model_kind": model_kind,
        "native_ctx": int(model_info.get("context_length") or 0),
        "ctx_size": int(ctx_size),
        "requested_target": _target_key(requested_target),
        "selected_target": target,
        "baseline_estimated_vram_gib": round(estimated, 2) if estimated > 0 else 0.0,
        "baseline_vram_pct": round(vram_pct, 1) if vram_pct > 0 else 0.0,
        "risk_counts": {
            "blocked": blocked,
            "high": high,
            "medium": sum(1 for c in candidates if c.params.get("_risk") == "medium"),
            "low": sum(1 for c in candidates if c.params.get("_risk") == "low"),
            "unknown": sum(1 for c in candidates if c.params.get("_risk") == "unknown"),
        },
        "notes": notes,
    }


def _estimate_vram_gib(params: Dict[str, Any], model_info: Dict[str, Any]) -> float:
    slots = int(params.get("parallel_slots") or 1)
    # unified KV-cache is closer to one shared context than N independent slots
    # for planning purposes; otherwise MTP 4-slot plans look impossible.
    effective_slots = 1 if bool(params.get("kv_unified", False)) else max(1, slots)
    ncmoe = int(params.get("ncmoe", 0) if int(params.get("ncmoe", -1)) >= 0 else 0)
    estimate = full_vram_estimate(
        model_info,
        int(params.get("ctx_size") or 0),
        int(params.get("ngl") or 0),
        str(params.get("cache_type_k") or "f16"),
        str(params.get("cache_type_v") or "f16"),
        bool(params.get("flash_attn", True)),
        effective_slots,
        ncmoe=ncmoe,
    )
    return float(estimate.total_gib)


def _vram_targeted_ncmoe_values(
    settings: Any, model_info: Dict[str, Any], base: Dict[str, Any]
) -> List[int]:
    if not _is_moe(model_info):
        return []
    ngl = int(base.get("ngl") or 0)
    blocks = _block_count(model_info)
    max_ncmoe = min(ngl, blocks) if blocks > 0 else ngl
    if max_ncmoe <= 0:
        return []

    values: List[int] = []
    current = int(getattr(settings, "cpu_moe_layers", -1) or -1)
    if current >= 0:
        values.append(_clamp_ncmoe_value(current, model_info, ngl))

    recommended = _recommended_ncmoe(
        settings, model_info, int(base.get("ctx_size") or 0), ngl
    )
    if recommended >= 0:
        values.append(recommended)

    budget = _vram_budget_gib(settings, model_info)
    if budget > 0:
        target = budget * VRAM_TARGET_FRACTION
        hard_limit = budget * VRAM_HARD_LIMIT_FRACTION
        best = None
        best_key = None
        # The estimator is intentionally approximate for MoE GGUFs. Avoid
        # jumping straight to "all experts on CPU" just because a formula says
        # it fits; search around current/recommended values first.
        anchor = max([v for v in values if v >= 0] or [0])
        max_search = min(
            max_ncmoe,
            max(anchor + NCMOE_SEARCH_ANCHOR_PAD, int(max_ncmoe * NCMOE_SEARCH_MAX_FRACTION)),
        )
        for ncmoe in range(0, max_search + 1):
            params = dict(base, ncmoe=ncmoe)
            estimated = _estimate_vram_gib(params, model_info)
            # prefer candidates under the hard limit and closest to 94% VRAM;
            # if estimator is pessimistic, still pick the least-bad nearby value.
            over_penalty = max(0.0, estimated - hard_limit) * NCMOE_OVER_LIMIT_PENALTY
            key = abs(estimated - target) + over_penalty
            if best_key is None or key < best_key:
                best_key = key
                best = ncmoe
        if best is not None:
            values.extend(
                [
                    best,
                    max(0, best - NCMOE_NEIGHBOR_SPAN),
                    min(max_ncmoe, best + NCMOE_NEIGHBOR_SPAN),
                ]
            )

    values.extend([0, -1])
    result: List[int] = []
    for value in values:
        value = _clamp_ncmoe_value(value, model_info, ngl)
        if value not in result:
            result.append(value)
    return result


def _base_params(
    settings: Any, model_info: Dict[str, Any], ctx_size: int
) -> Dict[str, Any]:
    logical = max(os.cpu_count() or 4, 1)
    threads = int(getattr(settings, "threads", logical) or logical)
    batch = int(getattr(settings, "batch_size", 512) or 512)
    if batch <= 0:
        batch = 512
    ubatch = int(getattr(settings, "ubatch_size", min(batch, 512)) or min(batch, 512))
    if ubatch <= 0:
        ubatch = min(batch, 512)
    if ctx_size >= LONG_CTX_THRESHOLD and not _is_moe(model_info):
        # Dense 128K+ сильнее упирается в KV memory; безопаснее стартовать с
        # меньшего micro-batch и уже потом проверять соседние варианты.
        ubatch = min(ubatch, DENSE_LONG_CTX_UBATCH_CAP)

    ctx_checkpoints = int(getattr(settings, "ctx_checkpoints", -1))
    cache_ram = int(getattr(settings, "cache_ram", -2))
    # llama-bench не измеряет prompt cache/checkpoints. Чтобы применённый после
    # AutoTune server работал как benchmark, нейтрализуем эти server-only механизмы
    # во всех AutoTune-пресетах, если пользователь не задал явное значение.
    if ctx_checkpoints < 0:
        ctx_checkpoints = 0
    if cache_ram < 0:
        cache_ram = 0

    mtp_model = _is_mtp_model(model_info)
    parallel_slots = 1
    current_ncmoe = int(getattr(settings, "cpu_moe_layers", -1) or -1)
    base_ncmoe = (
        _clamp_ncmoe_value(
            current_ncmoe, model_info, _gpu_layers_for_estimate(settings, model_info)
        )
        if mtp_model and current_ncmoe >= 0
        else -1
    )

    return {
        "ngl": _gpu_layers_for_estimate(settings, model_info),
        "ctx_size": int(ctx_size),
        "batch_size": batch,
        "ubatch_size": min(ubatch, batch),
        "cache_type_k": str(getattr(settings, "cache_type_k", "q8_0") or "q8_0"),
        "cache_type_v": str(getattr(settings, "cache_type_v", "q8_0") or "q8_0"),
        "threads": threads,
        # Current llama-bench builds do not expose -tb/--threads-batch.
        # Keep this in preset metadata only; cli_builder must not pass it to bench.
        "threads_batch": int(getattr(settings, "threads_batch", 0) or 0),
        "verbose": bool(getattr(settings, "verbose", False)),
        # llama-bench не тестирует server multi-slot (-np). Если утащить stale
        # -np=2 из UI, сервер делит контекст по слотам и может стать в 1.5-2 раза
        # медленнее, хотя bench показывал высокий TG. Quick AutoTune подбирает
        # latency preset для одного слота; server throughput-тест будет отдельным engine.
        "parallel_slots": parallel_slots,
        "kv_unified": False,
        "speculative_mtp": mtp_model,
        "spec_draft_n_max": 8 if mtp_model else 3,
        "spec_draft_p_min": 0.8 if mtp_model else 0.0,
        "gpu_layers_all": mtp_model,
        "spec_draft_gpu_layers": "all",
        "flash_attn": bool(getattr(settings, "flash_attn", True)),
        # Не даём llama-server делать auto-fit после AutoTune. Иначе он может
        # перекинуть часть тензоров на CPU, что llama-bench не измерял.
        "fit_off": True,
        # Prompt cache тоже не измеряется llama-bench и меняет server-поведение.
        "cache_prompt": False,
        # Базовый прогон всегда без принудительного CPU MoE offload.
        # Иначе stale/current ncmoe из UI портит все KV/batch/threads кандидаты
        # (на gemma4 32K/65K это снижало TG примерно со 120+ до 60-80 tok/s).
        "ncmoe": base_ncmoe,
        "ctx_checkpoints": ctx_checkpoints,
        "cache_ram": cache_ram,
        # llama-bench не загружает mmproj. Для server/benchmark parity после
        # AutoTune отключаем mmproj в применяемом пресете; иначе server может
        # тратить VRAM/время на projector, который не участвовал в benchmark.
        "use_mmproj": False,
        "model_type": "moe" if _is_moe(model_info) else "dense",
    }


def _append_unique(
    candidates: List[BenchmarkCandidate],
    seen: set,
    params: Dict[str, Any],
    reason: str,
    stage: str,
) -> None:
    norm = tuple(sorted((k, str(v)) for k, v in params.items()))
    if norm in seen:
        return
    seen.add(norm)
    cid = f"run_{len(candidates) + 1:03d}"
    candidates.append(BenchmarkCandidate(cid, dict(params), reason, stage))


def _quick_candidates(
    settings: Any, model_info: Dict[str, Any], ctx_size: int, target: str
) -> List[BenchmarkCandidate]:
    base = _base_params(settings, model_info, ctx_size)
    candidates: List[BenchmarkCandidate] = []
    seen: set = set()

    safe_ubatch = 256 if ctx_size >= 131072 and not _is_moe(model_info) else 512
    if _is_mtp_model(model_info):
        safe_kv = ("q8_0", "q8_0")
    else:
        safe_kv = (
            ("q4_0", "q4_0")
            if ctx_size >= 131072 and not _is_moe(model_info)
            else ("q8_0", "q8_0")
        )
    base.update(
        {
            "cache_type_k": safe_kv[0],
            "cache_type_v": safe_kv[1],
            "batch_size": 512,
            "ubatch_size": safe_ubatch,
        }
    )
    _append_unique(candidates, seen, base, "safe VRAM baseline", "baseline")

    # In Quick/Balanced the first few runs must cover speed-sensitive knobs.
    # MoE CPU-offload is often a VRAM-saving tradeoff and can slow TG, so do not
    # spend the whole small run budget on ncmoe variants before testing KV/batch.
    early_moe_vram = _target_key(target) in {"low_vram", "moe_optimized"} or ctx_size >= 131072
    if _is_moe(model_info) and early_moe_vram:
        for ncmoe in _vram_targeted_ncmoe_values(settings, model_info, base):
            p = dict(base, ncmoe=ncmoe)
            estimated = _estimate_vram_gib(p, model_info)
            _append_unique(
                candidates,
                seen,
                p,
                f"VRAM-target MoE ncmoe {ncmoe} (~{estimated:.1f} GiB)",
                "moe_vram",
            )

    # Speed-first order for small Max runs: test KV and batch before NGL/MoE.
    # Full NGL for MoE can be slower due VRAM pressure; batch/ubatch usually has
    # a better chance to reproduce manual server speed improvements.
    kv_candidates = _kv_candidates(target, "quick", ctx_size, model_info)
    if _target_key(target) == "balanced":
        kv_candidates = [kv for kv in kv_candidates if kv != safe_kv]
    for ctk, ctv in kv_candidates:
        p = dict(base, cache_type_k=ctk, cache_type_v=ctv)
        _append_unique(candidates, seen, p, f"KV {ctk}/{ctv}", "kv")
        if _target_key(target) == "balanced" and len([c for c in candidates if c.stage == "kv"]) >= 1:
            break

    for batch in (1024, 2048, 4096):
        p = dict(base, batch_size=batch, ubatch_size=min(512, batch))
        _append_unique(candidates, seen, p, f"batch {batch}", "batch")

    for ubatch in (256, 1024):
        p = dict(base, batch_size=max(1024, ubatch), ubatch_size=ubatch)
        _append_unique(candidates, seen, p, f"ubatch {ubatch}", "ubatch")

    current_threads = int(base.get("threads") or 0)
    for threads in _threads_candidates(settings):
        if threads == current_threads:
            continue
        p = dict(base, threads=threads, threads_batch=threads)
        _append_unique(candidates, seen, p, f"threads {threads}", "threads")
        if len([c for c in candidates if c.stage == "threads"]) >= 2:
            break

    for ngl in _ngl_candidates(settings, model_info, "quick", target):
        p = dict(base, ngl=ngl)
        _append_unique(candidates, seen, p, f"GPU layers ngl={ngl}", "ngl")

    if _is_moe(model_info) and not early_moe_vram:
        for ncmoe in _vram_targeted_ncmoe_values(settings, model_info, base):
            p = dict(base, ncmoe=ncmoe)
            estimated = _estimate_vram_gib(p, model_info)
            _append_unique(
                candidates,
                seen,
                p,
                f"VRAM-target MoE ncmoe {ncmoe} (~{estimated:.1f} GiB)",
                "moe_vram",
            )

    # flash-attn off часто просто падает на современных сборках; оставляем для Normal/Deep.

    expert_count = int(model_info.get("expert_count") or 0)
    block_count = int(model_info.get("block_count") or 0)
    should_test_moe = (
        _target_key(target) == "moe_optimized"
        or ctx_size >= 131072
        or _is_mtp_model(model_info)
    )
    if expert_count > 1 and should_test_moe:
        base_ngl = int(base.get("ngl") or 0)
        recommended = _recommended_ncmoe(settings, model_info, ctx_size, base_ngl)
        moe_values = [recommended, 0]
        if block_count > 0 and _target_key(target) == "moe_optimized":
            moe_values += [max(1, block_count // 4), max(1, block_count // 2)]
        for ncmoe in moe_values:
            ncmoe = _clamp_ncmoe_value(ncmoe, model_info, base_ngl)
            p = dict(base, ncmoe=ncmoe)
            _append_unique(candidates, seen, p, f"MoE ncmoe {ncmoe}", "moe")

    return candidates


def _normal_or_deep_candidates(
    settings: Any, model_info: Dict[str, Any], ctx_size: int, target: str, mode: str
) -> List[BenchmarkCandidate]:
    candidates = _quick_candidates(settings, model_info, ctx_size, target)
    seen = {tuple(sorted((k, str(v)) for k, v in c.params.items())) for c in candidates}
    base = _base_params(settings, model_info, ctx_size)
    safe_ubatch = 256 if ctx_size >= 131072 and not _is_moe(model_info) else 512
    if _is_mtp_model(model_info):
        safe_kv = ("q8_0", "q8_0")
    else:
        safe_kv = (
            ("q4_0", "q4_0")
            if ctx_size >= 131072 and not _is_moe(model_info)
            else ("q8_0", "q8_0")
        )
    base.update(
        {
            "cache_type_k": safe_kv[0],
            "cache_type_v": safe_kv[1],
            "batch_size": 1024,
            "ubatch_size": safe_ubatch,
        }
    )

    batch_values: Iterable[int] = (512, 1024, 2048, 4096)
    ubatch_values: Iterable[int] = (128, 256, 512, 1024)
    if _mode_key(mode) == "deep":
        batch_values = (512, 1024, 2048, 4096, 8192)
        ubatch_values = (128, 256, 512, 1024, 2048)

    for ngl in _ngl_candidates(settings, model_info, mode, target):
        p = dict(base, ngl=ngl)
        _append_unique(candidates, seen, p, f"staged GPU layers ngl={ngl}", "ngl")

    # Staged search instead of full KV x batch Cartesian product.  It reaches the
    # influential dimensions first and keeps Normal/Deep responsive; if a user
    # needs exhaustive checks they can still raise Max runs.
    for ctk, ctv in _kv_candidates(target, mode, ctx_size, model_info):
        p = dict(base, cache_type_k=ctk, cache_type_v=ctv)
        _append_unique(candidates, seen, p, f"staged KV {ctk}/{ctv}", "kv")

    for batch in batch_values:
        p = dict(base, batch_size=batch, ubatch_size=min(safe_ubatch, batch))
        _append_unique(candidates, seen, p, f"staged batch {batch}", "batch")

    if _mode_key(mode) == "deep":
        for ctk, ctv in _kv_candidates(target, mode, ctx_size, model_info)[:3]:
            for batch in (2048, 4096):
                p = dict(
                    base,
                    cache_type_k=ctk,
                    cache_type_v=ctv,
                    batch_size=batch,
                    ubatch_size=min(safe_ubatch, batch),
                )
                _append_unique(
                    candidates,
                    seen,
                    p,
                    f"deep KV/batch {ctk}/{ctv} b={batch}",
                    "kv_batch",
                )

    best_batch = 2048
    for ubatch in ubatch_values:
        if ubatch <= best_batch:
            p = dict(base, batch_size=best_batch, ubatch_size=ubatch)
            _append_unique(candidates, seen, p, f"staged ubatch {ubatch}", "ubatch")

    logical = max(os.cpu_count() or 4, 1)
    for threads in (8, 12, 16, 24, max(1, logical - 1)):
        if threads <= logical:
            for tb in (0, threads, max(1, logical - 1)):
                p = dict(base, threads=threads, threads_batch=tb)
                _append_unique(
                    candidates, seen, p, f"threads {threads}/tb {tb}", "threads"
                )

    if ctx_size >= 65536:
        for checkpoints in (0, 2, 4):
            p = dict(base, ctx_checkpoints=checkpoints)
            _append_unique(
                candidates, seen, p, f"ctx-checkpoints {checkpoints}", "memory"
            )
        for cache_ram in (0, 2048, 4096, 8192):
            p = dict(base, cache_ram=cache_ram)
            _append_unique(candidates, seen, p, f"cache-ram {cache_ram}", "memory")

    expert_count = int(model_info.get("expert_count") or 0)
    block_count = int(model_info.get("block_count") or 0)
    if expert_count > 1:
        base_ngl = int(base.get("ngl") or 0)
        recommended = _recommended_ncmoe(settings, model_info, ctx_size, base_ngl)
        moe_values = [recommended, 0]
        if block_count > 0:
            moe_values += [
                max(1, block_count // 4),
                max(1, block_count // 2),
                max(1, block_count * 3 // 4),
            ]
        moe_values.append(-1)
        for ncmoe in moe_values:
            ncmoe = _clamp_ncmoe_value(ncmoe, model_info, base_ngl)
            p = dict(base, ncmoe=ncmoe)
            _append_unique(candidates, seen, p, f"MoE ncmoe {ncmoe}", "moe")

    return candidates


def build_autotune_plan(
    settings: Any,
    model_path: str,
    model_info: Dict[str, Any] | None,
    mode: str = "quick",
    target: str = "balanced",
    engine: str = "llama-bench",
    time_budget_sec: int | None = None,
    max_runs: int | None = None,
    repeat_top: int = 1,
    early_stop_on_peak: bool = False,
    early_stop_min_successes: int = 3,
    early_stop_drop_pct: float = EARLY_STOP_DROP_PCT_DEFAULT,
) -> AutoTunePlan:
    """Создаёт ограниченный staged-план AutoTune."""
    info = dict(model_info or {})
    info.setdefault("_model_path", model_path)
    mode_key = _mode_key(mode)
    ctx_size = _ctx_from_settings(settings, info)
    requested_target = target
    target_key = (
        _auto_target_for_constraints(settings, info, ctx_size)
        if _is_auto_target(target)
        else _target_key(target)
    )
    budget = int(time_budget_sec or _TIME_BUDGETS.get(mode_key, _TIME_BUDGETS["quick"]))
    run_limit = int(max_runs or _MAX_RUNS.get(mode_key, _MAX_RUNS["quick"]))

    build_mode = "quick" if mode_key == "smart" else mode_key
    if build_mode == "quick":
        candidates = _quick_candidates(settings, info, ctx_size, target_key)
    else:
        candidates = _normal_or_deep_candidates(
            settings, info, ctx_size, target_key, build_mode
        )

    vram_budget = _vram_budget_gib(settings, info)
    _annotate_candidates(candidates, info, vram_budget)
    if mode_key == "smart":
        if candidates and all(c.params.get("_risk") == "blocked" for c in candidates):
            rescue = _smart_rescue_candidates(settings, info, ctx_size)
            _annotate_candidates(rescue, info, vram_budget)
            candidates.extend(rescue)
        candidates = _filter_smart_candidates(candidates, run_limit)

    candidates = candidates[: max(1, run_limit)]
    constraints = _plan_constraints(
        settings, info, ctx_size, target_key, candidates, requested_target
    )
    return AutoTunePlan(
        model_path=model_path,
        ctx_size=ctx_size,
        mode=mode_key,
        target=target_key,
        engine=engine,
        time_budget_sec=budget,
        max_runs=run_limit,
        repeat_top=max(1, int(repeat_top or 1)),
        candidates=candidates,
        early_stop_on_peak=bool(early_stop_on_peak),
        early_stop_min_successes=max(3, int(early_stop_min_successes or 3)),
        early_stop_drop_pct=max(0.0, float(early_stop_drop_pct or 0.0)),
        constraints=constraints,
    )
