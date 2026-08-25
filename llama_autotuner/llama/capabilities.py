from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class LlamaCapabilities:
    version_text: str
    help_text: str
    flags: set[str]

    def supports(self, flag: str) -> bool:
        return flag in self.flags or flag.lstrip("-") in self.help_text


def discover(exe: str) -> LlamaCapabilities:
    def run(arg: str) -> str:
        cp = subprocess.run([exe, arg], capture_output=True, text=True, timeout=15, check=False)
        return (cp.stdout or "") + "\n" + (cp.stderr or "")
    version = run("--version")
    help_text = run("--help")
    flags = set(re.findall(r"(?<!\w)(--?[a-zA-Z][\w-]*)", help_text))
    return LlamaCapabilities(version.strip(), help_text, flags)
