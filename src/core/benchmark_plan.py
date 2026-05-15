"""Построение планов AutoTune benchmark."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate
from src.core.moe_advisor import compute_moe_advice


_TIME_BUDGETS = {
    "quick": 15 * 60,
    "normal": 45 * 60,
    "deep": 120 * 60,
}

_MAX_RUNS = {
    "quick": 12,
    "normal": 60,
    "deep": 120,
}


def _mode_key(mode: str) -> str:
    return (mode or "quick").strip().lower()


def _target_key(target: str) -> str:
    return (target or "balanced").strip().lower().replace(" ", "_")


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
    if huge_context and dense_model and key != "quality_kv":
        # Dense-модели не имеют MoE-offload запаса. На 128K+ выбираем KV,
        # который с большей вероятностью реально поместится в llama-server.
        vals = [("q4_0", "q4_0"), ("q8_0", "q8_0")]
        if not quick:
            vals.append(("q4_0", "q8_0"))
    elif huge_context and key != "quality_kv":
        # llama-bench не принимает -c и не валидирует реальный большой KV-cache.
        # Для 128K+ нельзя давать f16/f16 ранний приоритет: он часто быстрый в
        # микробенче, но проваливается/замедляется в реальном llama-server.
        vals = [("q8_0", "q8_0"), ("q4_0", "q4_0"), ("f16", "f16")]
        if not quick:
            vals.insert(2, ("q4_0", "q8_0"))
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


def _gpu_layers_for_estimate(settings: Any, model_info: Dict[str, Any]) -> int:
    if getattr(settings, "gpu_auto", True):
        return int(model_info.get("block_count") or 999)
    return int(getattr(settings, "gpu_layers", 0) or 0)


def _recommended_ncmoe(settings: Any, model_info: Dict[str, Any], ctx_size: int) -> int:
    if not _is_moe(model_info):
        # Для dense-моделей -ncmoe неприменим. Не переносим сюда stale-значение
        # из UI, оставшееся после MoE-модели.
        return -1
    advice = compute_moe_advice(
        model_info,
        ctx_size,
        _gpu_layers_for_estimate(settings, model_info),
        "q8_0",
        "q8_0",
        bool(getattr(settings, "flash_attn", True)),
        max(1, int(getattr(settings, "parallel_slots", 1) or 1)),
    )
    return max(0, int(advice.recommended_ncmoe))


def _base_params(settings: Any, model_info: Dict[str, Any], ctx_size: int) -> Dict[str, Any]:
    logical = max(os.cpu_count() or 4, 1)
    threads = int(getattr(settings, "threads", logical) or logical)
    batch = int(getattr(settings, "batch_size", 512) or 512)
    if batch <= 0:
        batch = 512
    ubatch = int(getattr(settings, "ubatch_size", min(batch, 512)) or min(batch, 512))
    if ubatch <= 0:
        ubatch = min(batch, 512)
    if ctx_size >= 131072 and not _is_moe(model_info):
        # Dense 128K+ сильнее упирается в KV memory; безопаснее стартовать с
        # меньшего micro-batch и уже потом проверять соседние варианты.
        ubatch = min(ubatch, 256)

    ctx_checkpoints = int(getattr(settings, "ctx_checkpoints", -1))
    cache_ram = int(getattr(settings, "cache_ram", -2))
    if ctx_size >= 131072:
        # Для 128K+ дефолтный prompt cache/checkpoints может сильно замедлять
        # реальный llama-server, хотя llama-bench этого не видит.
        if ctx_checkpoints < 0:
            ctx_checkpoints = 0
        if cache_ram < 0:
            cache_ram = 0

    return {
        "ngl": "auto" if getattr(settings, "gpu_auto", True) else int(getattr(settings, "gpu_layers", 0)),
        "ctx_size": int(ctx_size),
        "batch_size": batch,
        "ubatch_size": min(ubatch, batch),
        "cache_type_k": str(getattr(settings, "cache_type_k", "q8_0") or "q8_0"),
        "cache_type_v": str(getattr(settings, "cache_type_v", "q8_0") or "q8_0"),
        "threads": threads,
        "threads_batch": int(getattr(settings, "threads_batch", 0) or 0),
        # llama-bench не тестирует server multi-slot (-np). Если утащить stale
        # -np=2 из UI, сервер делит контекст по слотам и может стать в 1.5-2 раза
        # медленнее, хотя bench показывал высокий TG. Quick AutoTune подбирает
        # latency preset для одного слота; server throughput-тест будет отдельным engine.
        "parallel_slots": 1,
        "flash_attn": bool(getattr(settings, "flash_attn", True)),
        # Не даём llama-server делать auto-fit после AutoTune. Иначе он может
        # перекинуть часть тензоров на CPU, что llama-bench не измерял.
        "fit_off": True,
        # Prompt cache тоже не измеряется llama-bench и меняет server-поведение.
        "cache_prompt": False,
        # Базовый прогон всегда без принудительного CPU MoE offload.
        # Иначе stale/current ncmoe из UI портит все KV/batch/threads кандидаты
        # (на gemma4 32K/65K это снижало TG примерно со 120+ до 60-80 tok/s).
        "ncmoe": -1,
        "ctx_checkpoints": ctx_checkpoints,
        "cache_ram": cache_ram,
        "use_mmproj": bool(getattr(settings, "use_mmproj", True)),
        "model_type": "moe" if _is_moe(model_info) else "dense",
    }


def _append_unique(
    candidates: List[BenchmarkCandidate], seen: set, params: Dict[str, Any], reason: str, stage: str
) -> None:
    norm = tuple(sorted((k, str(v)) for k, v in params.items()))
    if norm in seen:
        return
    seen.add(norm)
    cid = f"run_{len(candidates) + 1:03d}"
    candidates.append(BenchmarkCandidate(cid, dict(params), reason, stage))


def _quick_candidates(settings: Any, model_info: Dict[str, Any], ctx_size: int, target: str) -> List[BenchmarkCandidate]:
    base = _base_params(settings, model_info, ctx_size)
    candidates: List[BenchmarkCandidate] = []
    seen: set = set()

    safe_ubatch = 256 if ctx_size >= 131072 and not _is_moe(model_info) else 512
    safe_kv = ("q4_0", "q4_0") if ctx_size >= 131072 and not _is_moe(model_info) else ("q8_0", "q8_0")
    base.update(
        {
            "cache_type_k": safe_kv[0],
            "cache_type_v": safe_kv[1],
            "batch_size": 512,
            "ubatch_size": safe_ubatch,
        }
    )
    _append_unique(candidates, seen, base, "safe baseline", "baseline")

    for ctk, ctv in _kv_candidates(target, "quick", ctx_size, model_info):
        p = dict(base, cache_type_k=ctk, cache_type_v=ctv)
        _append_unique(candidates, seen, p, f"KV {ctk}/{ctv}", "kv")

    for batch in (1024, 2048):
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

    # flash-attn off часто просто падает на современных сборках; оставляем для Normal/Deep.

    expert_count = int(model_info.get("expert_count") or 0)
    block_count = int(model_info.get("block_count") or 0)
    should_test_moe = _target_key(target) == "moe_optimized" or ctx_size >= 131072
    if expert_count > 1 and should_test_moe:
        recommended = _recommended_ncmoe(settings, model_info, ctx_size)
        moe_values = [recommended, 0]
        if block_count > 0 and _target_key(target) == "moe_optimized":
            moe_values += [max(1, block_count // 4), max(1, block_count // 2)]
        for ncmoe in moe_values:
            p = dict(base, ncmoe=ncmoe)
            _append_unique(candidates, seen, p, f"MoE ncmoe {ncmoe}", "moe")

    return candidates


def _normal_or_deep_candidates(settings: Any, model_info: Dict[str, Any], ctx_size: int, target: str, mode: str) -> List[BenchmarkCandidate]:
    candidates = _quick_candidates(settings, model_info, ctx_size, target)
    seen = {tuple(sorted((k, str(v)) for k, v in c.params.items())) for c in candidates}
    base = _base_params(settings, model_info, ctx_size)
    safe_ubatch = 256 if ctx_size >= 131072 and not _is_moe(model_info) else 512
    safe_kv = ("q4_0", "q4_0") if ctx_size >= 131072 and not _is_moe(model_info) else ("q8_0", "q8_0")
    base.update({"cache_type_k": safe_kv[0], "cache_type_v": safe_kv[1], "batch_size": 1024, "ubatch_size": safe_ubatch})

    batch_values: Iterable[int] = (512, 1024, 2048, 4096)
    ubatch_values: Iterable[int] = (128, 256, 512, 1024)
    if _mode_key(mode) == "deep":
        batch_values = (512, 1024, 2048, 4096, 8192)
        ubatch_values = (128, 256, 512, 1024, 2048)

    for ctk, ctv in _kv_candidates(target, mode, ctx_size, model_info):
        for batch in batch_values:
            p = dict(base, cache_type_k=ctk, cache_type_v=ctv, batch_size=batch, ubatch_size=min(512, batch))
            _append_unique(candidates, seen, p, f"staged KV/batch {ctk}/{ctv} b={batch}", "kv_batch")

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
                _append_unique(candidates, seen, p, f"threads {threads}/tb {tb}", "threads")

    if ctx_size >= 65536:
        for checkpoints in (0, 2, 4):
            p = dict(base, ctx_checkpoints=checkpoints)
            _append_unique(candidates, seen, p, f"ctx-checkpoints {checkpoints}", "memory")
        for cache_ram in (0, 2048, 4096, 8192):
            p = dict(base, cache_ram=cache_ram)
            _append_unique(candidates, seen, p, f"cache-ram {cache_ram}", "memory")

    expert_count = int(model_info.get("expert_count") or 0)
    block_count = int(model_info.get("block_count") or 0)
    if expert_count > 1:
        recommended = _recommended_ncmoe(settings, model_info, ctx_size)
        moe_values = [recommended, 0]
        if block_count > 0:
            moe_values += [max(1, block_count // 4), max(1, block_count // 2), max(1, block_count * 3 // 4)]
        moe_values.append(-1)
        for ncmoe in moe_values:
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
) -> AutoTunePlan:
    """Создаёт ограниченный staged-план AutoTune."""
    info = model_info or {}
    mode_key = _mode_key(mode)
    ctx_size = _ctx_from_settings(settings, info)
    budget = int(time_budget_sec or _TIME_BUDGETS.get(mode_key, _TIME_BUDGETS["quick"]))
    run_limit = int(max_runs or _MAX_RUNS.get(mode_key, _MAX_RUNS["quick"]))

    if mode_key == "quick":
        candidates = _quick_candidates(settings, info, ctx_size, target)
    else:
        candidates = _normal_or_deep_candidates(settings, info, ctx_size, target, mode_key)

    candidates = candidates[: max(1, run_limit)]
    return AutoTunePlan(
        model_path=model_path,
        ctx_size=ctx_size,
        mode=mode_key,
        target=_target_key(target),
        engine=engine,
        time_budget_sec=budget,
        max_runs=run_limit,
        repeat_top=max(1, int(repeat_top or 1)),
        candidates=candidates,
    )
