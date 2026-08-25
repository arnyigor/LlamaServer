from __future__ import annotations

import os
import platform
import psutil

from llama_autotuner.hardware.nvidia import NvidiaSmiBackend
from llama_autotuner.models import HardwareInfo


def detect_hardware(gpu: NvidiaSmiBackend | None = None) -> HardwareInfo:
    gpu = gpu or NvidiaSmiBackend()
    devices = gpu.devices()
    if not devices:
        raise RuntimeError("No NVIDIA GPU detected")
    d = devices[0]
    vm = psutil.virtual_memory()
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU")
    return HardwareInfo(
        gpu_name=d.name, gpu_count=len(devices), vram_total_mb=d.total_mb, vram_used_mb=d.used_mb,
        vram_free_mb=d.free_mb, gpu_util_percent=d.util_percent, gpu_temp_c=d.temp_c,
        driver_version=d.driver_version, cpu_name=cpu, physical_cores=physical, logical_cores=logical,
        ram_total_mb=int(vm.total / 1024 / 1024), ram_available_mb=int(vm.available / 1024 / 1024),
        os_name=f"{platform.system()} {platform.release()}",
    )
