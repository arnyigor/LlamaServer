"""Helpers for launching subprocesses without flashing console windows."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict


def no_console_kwargs() -> Dict[str, Any]:
    """Return subprocess kwargs that suppress transient console windows on Windows.

    Sets both CREATE_NO_WINDOW and STARTUPINFO/SW_HIDE — belt-and-suspenders,
    since a spawned console-subsystem build can otherwise flash briefly even
    with just the creation flag.
    """
    if os.name != "nt":
        return {}
    kwargs: Dict[str, Any] = {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if flags:
        kwargs["creationflags"] = flags
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    kwargs["startupinfo"] = startupinfo
    return kwargs
