"""Non-interactive, callback-driven entry point for driving AutotuneEngine.

Adapted from ``llama_autotuner.cli:main`` (the upstream CLI script). Unlike
the CLI this module exposes no argparse, no interactive prompts, no SQLite
persistence, and expects a single explicit model path (no library scan) —
the caller (a Qt worker thread) already knows the model/server paths from
the host application's own settings. Cancellation is cooperative via a
``threading.Event`` instead of OS signals, since ``signal.signal`` handlers
only fire on the main thread and this is meant to run inside a QThread.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from llama_autotuner.hardware.nvidia import NvidiaSmiBackend, NvidiaSmiError
from llama_autotuner.hardware.preflight import run_preflight
from llama_autotuner.hardware.system import detect_hardware
from llama_autotuner.llama.capabilities import discover
from llama_autotuner.llama.fit_oracle import discover_fit_params, query_fit_params
from llama_autotuner.llama.gguf import inspect_gguf
from llama_autotuner.llama.server import cleanup_stale_servers
from llama_autotuner.models import CandidateResult, HardwareInfo, LaunchProfile, ModelInfo
from llama_autotuner.report.generate import build_profiles, write_reports
from llama_autotuner.tuning.optimizer import AutotuneEngine
from llama_autotuner.tuning.scoring import NoisePolicy
from llama_autotuner.tuning.target import (
    DegradationPolicy,
    ResourceClass,
    TargetSpec,
    VisionRequirement,
    build_feasibility_plan,
    format_capabilities,
    format_feasibility,
    model_supports_split_vision,
    runnable_options,
)

_MODE_DEFAULTS = {"quick": (24, 15), "normal": (50, 45), "deep": (100, 120)}


class AutotuneCancelled(Exception):
    """Raised from the progress callback to cooperatively unwind AutotuneEngine.tune()."""


class AutotuneSessionError(RuntimeError):
    """Raised for conditions that stop the session before any candidate is tried."""


@dataclass
class SessionConfig:
    server_exe: str
    model_path: str
    ctx: int = 65536
    workload: str = "auto"  # auto | chat | agent | long-context
    priority: str = "balanced"  # balanced | context | quality | speed
    vision: str = "off"  # off | required | auto
    mmproj: Optional[str] = None
    degradation_policy: str = "auto"  # strict | report | auto
    kv_k: str = "f16"
    kv_v: str = "f16"
    allow_kv_degradation: bool = True
    allow_context_reduction: bool = True
    min_tg_tps: Optional[float] = None
    min_pp_tps: Optional[float] = None
    decode_noise_pct: float = 5.0
    prefill_noise_pct: float = 10.0
    mode: str = "normal"  # quick | normal | deep
    vram_margin_mb: int = 1024
    require_vram_margin: bool = False
    absolute_vram_floor_mb: int = 300
    mtp_mode: str = "auto"  # auto | on | off
    max_minutes: Optional[int] = None
    max_runs: Optional[int] = None
    port: Optional[int] = None
    output_dir: Optional[Path] = None
    heartbeat_seconds: float = 5.0
    runtime_args: List[str] = field(default_factory=list)
    write_report_files: bool = False


@dataclass
class SessionResult:
    status: str  # COMPLETED | PARTIAL | FAILED | INTERRUPTED
    stop_reason: str
    profiles: List[LaunchProfile]
    results: List[CandidateResult]
    target: TargetSpec
    model: ModelInfo
    hardware: HardwareInfo
    elapsed_seconds: float
    output_dir: Path


def resolve_workload_profile(profile: str, ctx: int) -> str:
    """Same automatic workload mapping as llama_autotuner.cli.resolve_workload_profile."""
    if profile != "auto":
        return profile
    if ctx <= 16_384:
        return "chat"
    if ctx <= 65_536:
        return "agent"
    return "long-context"


def _session_status(engine: AutotuneEngine) -> str:
    if engine.completed:
        return "COMPLETED"
    if engine.stop_reason == "USER_CANCELLED":
        return "INTERRUPTED"
    if engine.stop_reason in {
        "NO_FEASIBLE_CONFIGURATION",
        "UNKNOWN_MODEL_ARCHITECTURE",
        "MODEL_STARTUP_FAILED",
    }:
        return "FAILED"
    return "PARTIAL"


def _find_free_local_port(preferred: int = 8080) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise AutotuneSessionError("could not find a free local port")


def run_session(
    config: SessionConfig,
    progress: Callable[[str], None],
    on_result: Callable[[CandidateResult], None],
    cancel_event: Optional[threading.Event] = None,
) -> SessionResult:
    """Run one autotune session in-process. Intended to be called from a worker thread."""

    def _progress(message: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AutotuneCancelled()
        progress(message)

    out = Path(config.output_dir) if config.output_dir else (
        Path("autotune_runs") / time.strftime("%Y%m%d_%H%M%S")
    )
    session_logs = out / "logs"
    session_logs.mkdir(parents=True, exist_ok=True)

    server = str(Path(config.server_exe).resolve())
    process_registry = out.parent / ".llama-autotuner-processes"
    cleanup = cleanup_stale_servers(process_registry, server, include_legacy_orphans=False)
    if cleanup.stopped:
        _progress(
            "Stopped stale autotuner llama-server process(es): "
            + ", ".join(str(pid) for pid in cleanup.stopped_pids)
        )
        time.sleep(0.75)
    for error in cleanup.errors:
        _progress(f"WARNING: startup server cleanup: {error}")

    gpu = NvidiaSmiBackend()
    try:
        hw = detect_hardware(gpu)
    except (NvidiaSmiError, RuntimeError) as exc:
        raise AutotuneSessionError(f"Unsupported hardware: {exc}") from exc
    if hw.gpu_count != 1:
        raise AutotuneSessionError(
            f"Autotuner supports exactly one NVIDIA GPU, detected {hw.gpu_count}"
        )

    _progress("GPU preflight...")
    pf = run_preflight(gpu, hw.vram_total_mb)
    if pf.state == "BUSY" and cleanup.stopped:
        for attempt in range(2):
            _progress(f"GPU still busy after stale-server cleanup; recheck {attempt + 1}/2...")
            time.sleep(1.5)
            pf = run_preflight(gpu, hw.vram_total_mb, samples=3, interval=0.5)
            if pf.state != "BUSY":
                break
    _progress(
        f"GPU: {hw.gpu_name} | VRAM {pf.baseline_used_mb}/{hw.vram_total_mb} MiB used | "
        f"state={pf.state}"
    )

    _progress("Inspecting llama.cpp capabilities...")
    caps = discover(server)
    _progress("Inspecting GGUF metadata...")
    model = inspect_gguf(str(Path(config.model_path).expanduser().resolve()))
    if model.split_parts_found < model.split_count:
        raise AutotuneSessionError(
            f"Incomplete split GGUF: found {model.split_parts_found}/{model.split_count} shards."
        )
    _progress(
        f"Model: arch={model.architecture} kind={model.kind.value} blocks={model.block_count} "
        f"MTP={model.has_mtp} vision-hint={model.has_vision_hint}"
    )

    resolved_profile = resolve_workload_profile(config.workload, config.ctx)

    mmproj = str(Path(config.mmproj).expanduser().resolve()) if config.mmproj else None
    vision = VisionRequirement(config.vision)
    if vision == VisionRequirement.AUTO:
        vision = (
            VisionRequirement.REQUIRED
            if (model_supports_split_vision(model) and mmproj)
            else VisionRequirement.OFF
        )

    target = TargetSpec(
        context=max(4096, int(config.ctx)),
        workload=resolved_profile,
        priority=config.priority,
        vision=vision,
        mmproj=mmproj,
        degradation_policy=DegradationPolicy(config.degradation_policy),
        preferred_kv_k=config.kv_k,
        preferred_kv_v=config.kv_v,
        allow_kv_degradation=config.allow_kv_degradation,
        allow_context_reduction=config.allow_context_reduction,
        preferred_vram_reserve_mb=config.vram_margin_mb,
        absolute_vram_floor_mb=config.absolute_vram_floor_mb,
        min_tg_tps=config.min_tg_tps,
        min_pp_tps=config.min_pp_tps,
        lock_kv=not config.allow_kv_degradation,
    )
    plan = build_feasibility_plan(model, hw, pf.baseline_used_mb, target, caps=caps)
    for line in format_capabilities(plan.capabilities, target):
        _progress(line)
    for line in format_feasibility(plan):
        _progress(line)

    if plan.capabilities.target_blocked:
        raise AutotuneSessionError(
            "Requested capability cannot be satisfied: " + plan.capabilities.vision.value
        )

    default_runs, default_minutes = _MODE_DEFAULTS[config.mode]
    max_runs = config.max_runs or default_runs
    max_minutes = config.max_minutes or default_minutes

    fit_exe = discover_fit_params(config.server_exe)
    if fit_exe:
        try:
            query_fit_params(
                server_exe=config.server_exe,
                model_path=model.path,
                context=target.context,
                kv_k=target.preferred_kv_k,
                kv_v=target.preferred_kv_v,
                margin_mb=config.vram_margin_mb,
            )
        except Exception:
            pass  # advisory-only; never blocks a session

    noise_policy = NoisePolicy(
        decode_rel=max(0.0, config.decode_noise_pct / 100.0),
        decode_probe_rel=max(0.10, config.decode_noise_pct / 100.0),
        prefill_rel=max(0.0, config.prefill_noise_pct / 100.0),
    )

    port = config.port or _find_free_local_port(8080)

    engine = AutotuneEngine(
        server,
        model,
        hw,
        caps,
        session_logs,
        pf.baseline_used_mb,
        vram_margin_mb=config.vram_margin_mb,
        port=port,
        max_runs=max_runs,
        max_minutes=max_minutes,
        progress=_progress,
        on_result=on_result,
        heartbeat_seconds=config.heartbeat_seconds,
        search_mode=config.mode,
        absolute_vram_floor_mb=config.absolute_vram_floor_mb,
        base_extra_args=list(config.runtime_args),
        workload_profile=resolved_profile,
        noise_policy=noise_policy,
        min_tg_tps=target.min_tg_tps,
        min_pp_tps=target.min_pp_tps,
        selection_priority=config.priority,
        require_preferred_vram_reserve=config.require_vram_margin,
        server_lease_dir=process_registry,
    )

    options = runnable_options(plan, include_exact_declaration=True)
    if target.degradation_policy == DegradationPolicy.STRICT:
        options = [
            o
            for o in options
            if o.name == "EXACT_TARGET" and o.resource_class != ResourceClass.INFEASIBLE
        ]

    interrupted = False
    try:
        results = engine.tune(target.context, config.mtp_mode, solution_options=options)
    except (KeyboardInterrupt, AutotuneCancelled):
        interrupted = True
        engine.mark_interrupted()
        results = engine.results

    status = "INTERRUPTED" if interrupted else _session_status(engine)
    profiles = build_profiles(
        results,
        server,
        model.path,
        caps,
        config.vram_margin_mb,
        session_complete=engine.completed,
        workload_profile=resolved_profile,
        noise_policy=engine.noise_policy,
        absolute_vram_floor_mb=config.absolute_vram_floor_mb,
        search_mode=config.mode,
        require_preferred_margin=config.require_vram_margin,
        preferred_candidate_key=engine.provisional_recommendation_key,
    )

    if config.write_report_files:
        context = {
            "session_status": status,
            "stop_reason": engine.stop_reason,
            "environment_state": pf.state,
            "workload_profile": resolved_profile,
            "search_mode": config.mode,
            "elapsed_seconds": engine.elapsed_seconds(),
            "target_spec": target.to_dict(),
            "feasibility_plan": plan.to_dict(),
            "preferred_vram_reserve_mb": config.vram_margin_mb,
            "absolute_vram_floor_mb": config.absolute_vram_floor_mb,
            "require_vram_margin": config.require_vram_margin,
            "decode_noise_pct": engine.noise_policy.decode_rel * 100.0,
            "prefill_noise_pct": engine.noise_policy.prefill_rel * 100.0,
            "model": {
                "path": model.path,
                "filename": Path(model.path).name,
                "architecture": model.architecture,
                "kind": model.kind.value,
            },
        }
        write_reports(out, profiles, results, context)

    return SessionResult(
        status=status,
        stop_reason=engine.stop_reason,
        profiles=profiles,
        results=results,
        target=target,
        model=model,
        hardware=hw,
        elapsed_seconds=engine.elapsed_seconds(),
        output_dir=out,
    )
