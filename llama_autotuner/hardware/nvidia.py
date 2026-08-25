from __future__ import annotations

import csv
import io
import subprocess
import time
from dataclasses import dataclass

from llama_autotuner.models import GpuSnapshot


class NvidiaSmiError(RuntimeError):
    pass


@dataclass(slots=True)
class NvidiaDevice:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    util_percent: float
    temp_c: float | None
    driver_version: str | None


class NvidiaSmiBackend:
    def __init__(self, exe: str = "nvidia-smi") -> None:
        self.exe = exe

    def _query(self, fields: list[str]) -> list[list[str]]:
        cmd = [self.exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise NvidiaSmiError(f"nvidia-smi query failed: {exc}") from exc
        return [row for row in csv.reader(io.StringIO(cp.stdout), skipinitialspace=True) if row]

    def devices(self) -> list[NvidiaDevice]:
        fields = [
            "index", "name", "memory.total", "memory.used", "memory.free",
            "utilization.gpu", "temperature.gpu", "driver_version",
        ]
        rows = self._query(fields)
        result: list[NvidiaDevice] = []
        for r in rows:
            result.append(NvidiaDevice(
                index=int(r[0]), name=r[1], total_mb=int(float(r[2])), used_mb=int(float(r[3])),
                free_mb=int(float(r[4])), util_percent=float(r[5]),
                temp_c=None if r[6] in {"N/A", "[N/A]"} else float(r[6]), driver_version=r[7],
            ))
        return result

    def snapshot(self, index: int = 0) -> GpuSnapshot:
        fields = [
            "memory.used", "memory.free", "utilization.gpu", "temperature.gpu",
            "power.draw", "clocks.current.graphics", "clocks.current.memory",
        ]
        row = self._query(fields)[index]
        def num(v: str) -> float | None:
            try:
                return float(v)
            except ValueError:
                return None
        return GpuSnapshot(
            timestamp=time.time(), used_mb=int(float(row[0])), free_mb=int(float(row[1])),
            util_percent=float(row[2]), temperature_c=num(row[3]), power_w=num(row[4]),
            graphics_clock_mhz=num(row[5]), memory_clock_mhz=num(row[6]),
        )

    def compute_processes(self) -> list[dict[str, str]]:
        cmd = [self.exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        except OSError:
            return []
        out = []
        for row in csv.reader(io.StringIO(cp.stdout), skipinitialspace=True):
            if len(row) >= 3:
                out.append({"pid": row[0], "name": row[1], "used_memory_mb": row[2]})
        return out
