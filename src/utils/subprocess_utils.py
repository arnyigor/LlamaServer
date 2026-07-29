"""Helpers for launching subprocesses without flashing console windows."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict


def no_console_kwargs() -> Dict[str, Any]:
    """Return subprocess kwargs that suppress transient console windows on Windows."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flags} if flags else {}
