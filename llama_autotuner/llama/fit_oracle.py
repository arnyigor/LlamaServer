from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FitSuggestion:
    executable: str
    context: int | None
    ngl: int | None
    tensor_overrides: str | None
    raw_args: str
    elapsed_seconds: float | None = None


def discover_fit_params(server_exe: str) -> str | None:
    """Find llama-fit-params next to the selected llama-server build.

    Keeping discovery sibling-local ensures the oracle comes from the same llama.cpp build family
    instead of accidentally mixing a different executable from PATH.
    """
    server = Path(server_exe)
    names = ["llama-fit-params.exe", "llama-fit-params"] if server.suffix.lower() == ".exe" else ["llama-fit-params", "llama-fit-params.exe"]
    for name in names:
        p = server.with_name(name)
        if p.is_file():
            return str(p)
    return None


def parse_fit_output(text: str, executable: str = "llama-fit-params") -> FitSuggestion | None:
    # The tool prints diagnostics first and the fitted CLI arguments as the final '-c ... -ngl ...' line.
    matches = list(re.finditer(r"(?m)^\s*(-c\s+\d+\s+-ngl\s+-?\d+[^\r\n]*)\s*$", text or ""))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    cm = re.search(r"(?:^|\s)-c\s+(\d+)", raw)
    nm = re.search(r"(?:^|\s)-ngl\s+(-?\d+)", raw)
    om = re.search(r'''(?:^|\s)-ot\s+["']([^"']+)["']''', raw)
    elapsed = None
    em = re.search(r"fitting params to free memory took\s+([0-9.]+)\s+seconds", text or "", re.I)
    if em:
        elapsed = float(em.group(1))
    return FitSuggestion(
        executable=executable,
        context=int(cm.group(1)) if cm else None,
        ngl=int(nm.group(1)) if nm else None,
        tensor_overrides=om.group(1) if om else None,
        raw_args=raw,
        elapsed_seconds=elapsed,
    )


def query_fit_params(*, server_exe: str, model_path: str, context: int,
                     kv_k: str, kv_v: str, margin_mb: int,
                     timeout: float = 30.0) -> FitSuggestion | None:
    exe = discover_fit_params(server_exe)
    if not exe:
        return None
    cmd = [
        exe, "--model", model_path,
        "-c", str(int(context)),
        "-ctk", kv_k, "-ctv", kv_v,
        "-fitt", str(int(margin_mb)),
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    if cp.returncode != 0:
        return None
    return parse_fit_output(text, executable=exe)
