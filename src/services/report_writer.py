"""Запись JSON/Markdown отчётов AutoTune."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate, BenchmarkResult


def _candidate_by_id(plan: AutoTunePlan) -> Dict[str, BenchmarkCandidate]:
    return {c.id: c for c in plan.candidates}


def collect_hardware_info() -> Dict[str, Any]:
    return {
        "gpu": "unknown",
        "vram_mib": 0,
        "ram_mib": 0,
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count() or 0,
        "platform": platform.platform(),
    }


def results_payload(
    plan: AutoTunePlan,
    model_info: Dict[str, Any],
    results: Iterable[BenchmarkResult],
    best: Optional[BenchmarkResult],
    llama_cpp_build: str = "unknown",
) -> Dict[str, Any]:
    runs = list(results)
    by_id = _candidate_by_id(plan)
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "path": plan.model_path,
            "name": Path(plan.model_path).name,
            "architecture": model_info.get("architecture", ""),
            "quant": model_info.get("quant", ""),
            "size_gib": model_info.get("size_gib", 0),
            "ctx_native": model_info.get("context_length", 0),
            "expert_count": model_info.get("expert_count", 0),
            "expert_used": model_info.get("expert_used", 0),
        },
        "hardware": collect_hardware_info(),
        "benchmark": {
            "context_size": plan.ctx_size,
            "mode": plan.mode,
            "target": plan.target,
            "engine": plan.engine,
            "time_budget_sec": plan.time_budget_sec,
            "max_runs": plan.max_runs,
        },
        "llama_cpp_build": llama_cpp_build,
        "best_run_id": best.candidate_id if best else None,
        "runs": [
            {
                **r.to_dict(),
                "params": by_id.get(r.candidate_id).params if by_id.get(r.candidate_id) else {},
            }
            for r in runs
        ],
    }


def write_json_report(
    output_dir: str,
    plan: AutoTunePlan,
    model_info: Dict[str, Any],
    results: List[BenchmarkResult],
    best: Optional[BenchmarkResult],
    llama_cpp_build: str = "unknown",
) -> str:
    path = Path(output_dir) / "results.json"
    payload = results_payload(plan, model_info, results, best, llama_cpp_build)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_plan(output_dir: str, plan: AutoTunePlan) -> str:
    path = Path(output_dir) / "plan.json"
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_best(output_dir: str, best: Optional[BenchmarkResult], params: Dict[str, Any]) -> str:
    path = Path(output_dir) / "best.json"
    payload = {"best": best.to_dict() if best else None, "params": params if best else {}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_markdown_report(
    output_dir: str,
    plan: AutoTunePlan,
    model_info: Dict[str, Any],
    results: List[BenchmarkResult],
    best: Optional[BenchmarkResult],
) -> str:
    by_id = _candidate_by_id(plan)
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]
    top = sorted(successful, key=lambda r: r.score, reverse=True)[:5]
    best_params = by_id.get(best.candidate_id).params if best and by_id.get(best.candidate_id) else {}

    lines = [
        "# AutoTune Report",
        "",
        "## Summary",
        f"Best preset: {best.candidate_id if best else 'not found'}",
        f"Score: {best.score:.3f}" if best else "Score: 0",
        f"Mode: {plan.mode}",
        f"Target: {plan.target}",
        f"Context: {plan.ctx_size}",
        "",
        "## Model",
        f"- Path: `{plan.model_path}`",
        f"- Architecture: {model_info.get('architecture', '?')}",
        f"- Quant: {model_info.get('quant', '?')}",
        f"- Size: {model_info.get('size_gib', 0):.2f} GiB",
        "",
        "## Best Parameters",
    ]
    if best_params:
        for key, value in best_params.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("No successful candidate.")

    lines += [
        "",
        "## Top Results",
        "| Run | Score | PP tok/s | TG tok/s | VRAM MiB | RAM MiB | Params |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in top:
        params = by_id.get(r.candidate_id).params if by_id.get(r.candidate_id) else {}
        param_text = ", ".join(
            f"{k}={params.get(k)}" for k in ("ngl", "cache_type_k", "cache_type_v", "batch_size", "ubatch_size", "threads")
        )
        lines.append(
            f"| {r.candidate_id} | {r.score:.3f} | {r.prompt_tok_s:.1f} | {r.generation_tok_s:.1f} | "
            f"{r.vram_used_mib:.0f} | {r.ram_used_mib:.0f} | {param_text} |"
        )

    lines += [
        "",
        "## Failed Runs",
        "| Run | Status | Reason |",
        "|---|---|---|",
    ]
    for r in failed:
        lines.append(f"| {r.candidate_id} | {r.status} | {r.error or '-'} |")

    lines += ["", "## Observations"]
    if best:
        lines.append("- Best result is selected by target-specific score, not by a single raw speed metric.")
        if best_params.get("cache_type_k") == "q8_0" and best_params.get("cache_type_v") == "q8_0":
            lines.append("- q8_0/q8_0 KV was selected as a balanced speed/quality option.")
        if best.status == "success":
            lines.append("- Selected candidate completed successfully without detected OOM.")
    if any(r.status == "failed_oom" for r in results):
        lines.append("- Some candidates failed by OOM and were excluded from ranking.")

    if best:
        lines += ["", "## Full command", "```", " ".join(best.command), "```"]

    path = Path(output_dir) / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
