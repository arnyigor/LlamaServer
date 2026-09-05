"""Feature detection: which llama-server CLI flags the configured binary
actually supports, parsed from its own ``--help`` output.

Without this, the UI can happily build flags (e.g. MTP's --spec-type) that an
older llama-server build has never heard of: the server then either ignores
them silently or fails to start, and nothing in the app explains why. Probing
--help once per selected binary and cross-checking against PARAM_REGISTRY
(the single source of truth for which flags each parameter emits) lets the UI
flag that mismatch instead of leaving the parameter looking broken.

Detection is advisory only and fails open: a probe that can't run (missing
binary, crash, timeout) or output that doesn't parse returns ``None``, and
callers must treat that as "don't filter" rather than disabling everything.
"""

from __future__ import annotations

import re
import subprocess
from typing import Iterable, Optional, Set

from src.core.param_registry import ParamSpec
from src.utils.subprocess_utils import no_console_kwargs

_FLAG_RE = re.compile(r"(?<![\w-])(--?[a-zA-Z][a-zA-Z0-9-]*)")


def parse_supported_flags(help_text: str) -> Set[str]:
    """Extract every ``-x``/``--long-flag`` token mentioned in --help text."""
    return set(_FLAG_RE.findall(help_text or ""))


def probe_supported_flags(exe_path: str, timeout: float = 5.0) -> Optional[Set[str]]:
    """Run ``<exe_path> --help`` and return the flags it mentions.

    Returns ``None`` when the probe itself is inconclusive (missing binary,
    crash, timeout, or empty output) so callers skip filtering instead of
    marking every parameter unsupported.
    """
    if not exe_path:
        return None
    try:
        result = subprocess.run(
            [exe_path, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            **no_console_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    flags = parse_supported_flags(text)
    return flags or None


def is_spec_supported(spec: ParamSpec, supported_flags: Optional[Iterable[str]]) -> bool:
    """True when ``spec`` should be treated as available.

    Always true when detection is unavailable (``supported_flags`` is
    ``None``) or the parameter has no CLI flag of its own.
    """
    if supported_flags is None:
        return True
    flags = spec.cli_flags + spec.cli_neg_flags
    if not flags:
        return True
    supported = set(supported_flags)
    return any(flag in supported for flag in flags)
