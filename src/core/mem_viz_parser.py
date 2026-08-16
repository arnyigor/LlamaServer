"""Парсер логов llama.cpp для визуализации памяти (RAM/VRAM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Memory Components
# ─────────────────────────────────────────────────────────────

COMPONENT_ORDER = [
    "weights",
    "kv_cache",
    "prompt_cache",
    "recurrent_state",
    "compute_pp",
    "compute",
    "process_working_set",
    "process_gpu_memory",
]

COMPONENT_META = {
    "weights": {"label": "Weights", "color": 68, "char": "█"},
    "kv_cache": {"label": "KV Cache", "color": 208, "char": "▓"},
    "prompt_cache": {"label": "Prompt Cache", "color": 141, "char": "█"},
    "recurrent_state": {"label": "RS State", "color": 99, "char": "▓"},
    "compute_pp": {"label": "Compute PP", "color": 160, "char": "▒"},
    "compute": {"label": "Compute", "color": 71, "char": "░"},
    "process_working_set": {"label": "Process Working Set", "color": 45, "char": "█"},
    "process_gpu_memory": {"label": "Process GPU Memory", "color": 190, "char": "█"},
}

KIND_TO_COMPONENT = {
    "model": "weights",
    "kv": "kv_cache",
    "rs": "recurrent_state",
    "output": "compute",
    "compute": "compute",
    "compute pp": "compute_pp",
}

# ─────────────────────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────────────────────

BUFFER_RE = re.compile(
    r"""
    (?P<device>[A-Za-z][A-Za-z0-9_]*)
    \s+
    (?P<kind>
        compute\s+pp |
        model |
        KV |
        RS |
        output |
        compute
    )
    \s+buffer\s+size
    \s*=\s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>KiB|MiB|GiB|KB|MB|GB)
    """,
    re.IGNORECASE | re.VERBOSE,
)

PROMPT_CACHE_RE = re.compile(
    r"prompt\s+cache\s+is\s+enabled,\s+size\s+limit:\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)",
    re.IGNORECASE,
)

GGUF_KV_RE = re.compile(
    r"llama_model_loader:\s+- kv\s+\d+:\s+"
    r"(?P<key>[A-Za-z0-9_.]+)\s+"
    r"(?P<type>\S+)\s*=\s*"
    r"(?P<value>.+)$"
)

PRINT_INFO_RE = re.compile(
    r"print_info:\s+(?P<key>[A-Za-z0-9_. ]+?)\s*=\s*(?P<value>.+)$"
)

CONTEXT_INFO_RE = re.compile(
    r"llama_context:\s+(?P<key>[A-Za-z0-9_]+)\s*=\s*(?P<value>.+)$"
)

OFFLOADED_RE = re.compile(
    r"load_tensors:\s+offloaded\s+(\d+)/(\d+)\s+layers",
    re.IGNORECASE,
)

OOM_RE = re.compile(
    r"(out of memory|cudaMalloc failed|failed to allocate|failed to initialize the context)",
    re.IGNORECASE,
)

MLOCK_WARNING_RE = re.compile(
    r"failed to mlock .*cannot allocate memory",
    re.IGNORECASE,
)

ALLOC_MIB_RE = re.compile(
    r"allocating\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)\s+on device",
    re.IGNORECASE,
)

ALLOC_BYTES_RE = re.compile(
    r"failed to allocate .* buffer of size\s+(?P<bytes>\d+)",
    re.IGNORECASE,
)

CUDA_TOTAL_RE = re.compile(
    r"Total VRAM:\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)",
    re.IGNORECASE,
)

CUDA_FREE_RE = re.compile(
    r"llama_prepare_model_devices:.*-\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)\s+free",
    re.IGNORECASE,
)

STOP_MARKERS = [
    "main: server is listening",
    "llama server listening",
    "llama_server: model loaded",
    "llama_server: listening on",
    "llama_print_timings",
    "llama_perf_context_print",
]


# ─────────────────────────────────────────────────────────────
# Data Containers
# ─────────────────────────────────────────────────────────────


@dataclass
class MemoryData:
    raw_devices: dict[str, dict[str, float]] = field(default_factory=dict)
    system_memory: dict[str, float] = field(default_factory=dict)
    model_info: dict[str, str] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)
    failure_lines: list[str] = field(default_factory=list)

    fatal_error: str | None = None
    fatal_score: int = 0
    oom: bool = False

    failed_component: str | None = None
    failed_category: str | None = None
    failed_alloc_mib: float | None = None

    server_ready: bool = False
    process_exit_code: int | None = None
    cli_args: str = ""

    def add_raw(self, device: str, component: str, mib: float) -> None:
        self.raw_devices.setdefault(device, {})
        self.raw_devices[device][component] = (
            self.raw_devices[device].get(component, 0.0) + mib
        )

    def clear_loaded_model(self) -> None:
        """Сбрасывает данные, которые относятся к выгруженной модели."""
        self.raw_devices.clear()
        self.model_info.clear()
        self.server_ready = False
        self.process_exit_code = None

    def note_warning(self, line: str) -> None:
        if line not in self.warnings:
            self.warnings.append(line)
        self.warnings = self.warnings[-10:]

    def note_failure(
        self,
        line: str,
        component: str | None = None,
        category: str | None = None,
        alloc_mib: float | None = None,
    ) -> None:
        self.oom = True

        if line not in self.failure_lines:
            self.failure_lines.append(line)
        self.failure_lines = self.failure_lines[-10:]

        score = failure_score(line)
        if score >= self.fatal_score:
            self.fatal_error = line
            self.fatal_score = score

        if component:
            if self.failed_component is None or component_specificity(
                component
            ) >= component_specificity(self.failed_component):
                self.failed_component = component

        if category:
            self.failed_category = category

        if alloc_mib is not None:
            if self.failed_alloc_mib is None or alloc_mib > self.failed_alloc_mib:
                self.failed_alloc_mib = alloc_mib

    def get_aggregated(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {"VRAM": {}, "RAM": {}}

        for dev, comps in self.raw_devices.items():
            if dev.startswith("CUDA") and dev != "CUDA_HOST":
                target = "VRAM"
            else:
                target = "RAM"

            for comp, mib in comps.items():
                result[target][comp] = result[target].get(comp, 0.0) + mib

        return result

    def total(self, cat: str) -> float:
        return sum(self.get_aggregated().get(cat, {}).values())

    def grand_total(self) -> float:
        return self.total("VRAM") + self.total("RAM")

    def components_used(self) -> list[str]:
        agg = self.get_aggregated()
        all_comps: set[str] = set()
        for comps in agg.values():
            all_comps.update(comps.keys())
        return [c for c in COMPONENT_ORDER if c in all_comps]

    def utilization(self, cat: str) -> float | None:
        cap = self.system_memory.get(cat)
        if not cap:
            return None
        return self.total(cat) / cap * 100.0

    def to_dict(self) -> dict:
        agg = self.get_aggregated()
        failure_cat, remaining_mib, deficit_mib = estimate_shortfall_mib(self)
        utilization = {"VRAM": self.utilization("VRAM"), "RAM": self.utilization("RAM")}
        return {
            "cli_args": self.cli_args,
            "model": self.model_info,
            "system_memory_mib": {
                k: round(v, 2) for k, v in sorted(self.system_memory.items())
            },
            "allocated": {
                cat: {
                    "components_mib": {
                        k: round(v, 2) for k, v in sorted(agg.get(cat, {}).items())
                    },
                    "total_mib": round(self.total(cat), 2),
                    "capacity_mib": round(self.system_memory[cat], 2)
                    if cat in self.system_memory
                    else None,
                    "utilization_pct": round(util_pct, 2) if util_pct is not None else None,
                }
                for cat, util_pct in (
                    ("VRAM", utilization["VRAM"]),
                    ("RAM", utilization["RAM"]),
                )
            },
            "grand_total_mib": round(self.grand_total(), 2),
            "status": {
                "oom": self.oom,
                "fatal_error": self.fatal_error,
                "failed_component": self.failed_component,
                "failed_category": self.failed_category or failure_cat,
                "failed_alloc_mib": round(self.failed_alloc_mib, 2)
                if self.failed_alloc_mib is not None
                else None,
                "estimated_remaining_mib": round(remaining_mib, 2)
                if remaining_mib is not None
                else None,
                "estimated_deficit_mib": round(deficit_mib, 2)
                if deficit_mib is not None
                else None,
                "server_ready": self.server_ready,
                "exit_code": self.process_exit_code,
                "warnings": self.warnings[-10:],
                "failure_lines": self.failure_lines[-10:],
            },
        }


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────


def to_mib(value: float, unit: str) -> float:
    u = unit.upper()
    if u == "KIB":
        return value / 1024.0
    if u == "MIB":
        return value
    if u == "GIB":
        return value * 1024.0
    if u == "KB":
        return value * 1000 / 1024 / 1024
    if u == "MB":
        return value * 1000 * 1000 / 1024 / 1024
    if u == "GB":
        return value * 1000 * 1000 * 1000 / 1024 / 1024
    return value


def fmt_mem(mib: float, short: bool = False) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.2f} GiB" if not short else f"{mib / 1024:.2f}G"
    return f"{mib:.2f} MiB" if not short else f"{mib:.0f}M"


def normalize_device(device: str) -> str:
    return device.upper()


def normalize_kind(kind: str) -> str:
    return re.sub(r"\s+", " ", kind.lower().strip())


def normalize_info_key(key: str) -> str:
    return re.sub(r"\s+", "_", key.strip().lower())


def failure_score(line: str) -> int:
    low = line.lower()
    score = 0
    if "failed to initialize the context" in low:
        score += 100
    if "failed to allocate" in low:
        score += 80
    if "cudaMalloc failed" in line:
        score += 60
    if "out of memory" in low:
        score += 50
    if "graph_reserve" in low or "sched_reserve" in low:
        score += 20
    return score


def component_specificity(component: str) -> int:
    order = {
        "compute": 1,
        "weights": 1,
        "prompt_cache": 1,
        "kv_cache": 1,
        "recurrent_state": 1,
        "compute_pp": 2,
    }
    return order.get(component, 0)


def infer_failed_component(line: str) -> str | None:
    low = line.lower()
    if "compute pp" in low or "compute_pp" in low:
        return "compute_pp"
    if "compute buffers" in low:
        return "compute"
    if "kv cache" in low or "kv buffer" in low:
        return "kv_cache"
    if "output layer" in low or "output buffer" in low:
        return "compute"
    if "recurrent" in low or "rs buffer" in low:
        return "recurrent_state"
    if "model buffer" in low:
        return "weights"
    return None


def infer_failed_category(line: str, component: str | None = None) -> str | None:
    low = line.lower()

    if "cuda" in low or "gpu" in low or "vram" in low:
        return "VRAM"
    if "cpu" in low or "host" in low or "ram" in low:
        return "RAM"

    if component in {"compute_pp", "compute", "kv_cache"}:
        return "VRAM"

    return None


def should_stop(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in STOP_MARKERS)


def estimate_shortfall_mib(
    data: MemoryData,
) -> tuple[str | None, float | None, float | None]:
    if not data.fatal_error:
        return None, None, None

    cat = data.failed_category
    low = data.fatal_error.lower()

    if cat is None:
        if "cuda" in low or "gpu" in low or "vram" in low:
            cat = "VRAM"
        elif "cpu" in low or "host" in low or "ram" in low:
            cat = "RAM"
        elif data.failed_component in COMPONENT_ORDER:
            cat = (
                "RAM"
                if data.failed_component in {"weights", "prompt_cache"}
                else "VRAM"
            )

    if cat is None:
        return None, None, None

    cap = data.system_memory.get(f"{cat}_FREE")
    if cap is None:
        cap = data.system_memory.get(cat)

    if cap is None:
        return cat, None, None

    remaining = max(0.0, cap - data.total(cat))
    deficit = None
    if data.failed_alloc_mib is not None:
        deficit = max(0.0, data.failed_alloc_mib - remaining)

    return cat, remaining, deficit


# ─────────────────────────────────────────────────────────────
# Parser Engine
# ─────────────────────────────────────────────────────────────


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_line(line: str, data: MemoryData, debug: bool = False) -> None:
    line = strip_ansi(line.rstrip("\n"))

    if MLOCK_WARNING_RE.search(line):
        data.note_warning(line)

    if m := CUDA_TOTAL_RE.search(line):
        data.system_memory["VRAM"] = to_mib(float(m.group("value")), m.group("unit"))

    if m := CUDA_FREE_RE.search(line):
        data.system_memory["VRAM_FREE"] = to_mib(
            float(m.group("value")), m.group("unit")
        )

    for m in BUFFER_RE.finditer(line):
        device = normalize_device(m.group("device"))
        kind = normalize_kind(m.group("kind"))
        value = float(m.group("value"))
        unit = m.group("unit")

        comp = KIND_TO_COMPONENT.get(kind)
        if not comp:
            continue

        mib = to_mib(value, unit)
        data.add_raw(device, comp, mib)

    if m := PROMPT_CACHE_RE.search(line):
        value = float(m.group("value"))
        unit = m.group("unit")
        mib = to_mib(value, unit)
        data.add_raw("CPU", "prompt_cache", mib)

    if m := GGUF_KV_RE.search(line):
        key = m.group("key").strip()
        kv_value = m.group("value").strip()
        if key == "general.name":
            data.model_info.setdefault("name", kv_value)
        elif key == "general.architecture":
            data.model_info.setdefault("arch", kv_value)

    if m := PRINT_INFO_RE.search(line):
        key = normalize_info_key(m.group("key"))
        info_value = m.group("value").strip()
        mapping = {
            "file_format": "file_format",
            "file_type": "file_type",
            "file_size": "file_size",
            "model_type": "model_type",
            "model_params": "params",
            "general.name": "name",
        }
        if key in mapping:
            data.model_info.setdefault(mapping[key], info_value)

    if m := CONTEXT_INFO_RE.search(line):
        key = m.group("key").strip()
        ctx_value = m.group("value").strip()
        if key == "n_ctx":
            data.model_info.setdefault("ctx", ctx_value)
        elif key == "n_seq_max":
            data.model_info.setdefault("n_seq_max", ctx_value)

    if m := OFFLOADED_RE.search(line):
        data.model_info["layers_offloaded"] = m.group(1)
        data.model_info["layers_total"] = m.group(2)

    if OOM_RE.search(line):
        component = infer_failed_component(line)
        category = infer_failed_category(line, component=component)

        alloc_mib = None
        if m := ALLOC_MIB_RE.search(line):
            alloc_mib = to_mib(float(m.group("value")), m.group("unit"))
        elif m := ALLOC_BYTES_RE.search(line):
            alloc_mib = float(m.group("bytes")) / (1024.0 * 1024.0)

        data.note_failure(
            line=line,
            component=component,
            category=category,
            alloc_mib=alloc_mib,
        )

    # Parse model unload / memory release
    # Format: load_tensors: offloading output layer to GPU
    # Format: load_tensors: offloading 35 repeating layers to GPU
    if m := re.search(
        r"load_tensors:\s+offloading\s+(\d+)\s+repeating\s+layers\s+to\s+(\w+)",
        line,
    ):
        # Track layer offloading changes
        pass

    # Parse memory release during model unload
    # Format: llama_model_unload: model unloaded
    if re.search(
        r"llama_model_unload|model\s+unloaded|freeing\s+model", line, re.IGNORECASE
    ):
        # При выгрузке модель освобождает RAM/VRAM, поэтому старые числа
        # больше нельзя показывать как актуальные.
        data.clear_loaded_model()
        return

    if should_stop(line):
        data.server_ready = True


def collect_from_text(text: str, debug: bool = False) -> MemoryData:
    data = MemoryData()
    for line in text.splitlines():
        parse_line(line, data, debug=debug)
    return data


def collect_from_file(path: Path, debug: bool = False) -> MemoryData:
    data = MemoryData()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parse_line(line, data, debug=debug)
    return data
