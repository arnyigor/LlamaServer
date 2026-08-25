"""Headless smoke test for the PyInstaller-built exe.

Launches dist_next/LlamaServerGUI.exe under the offscreen Qt platform and
checks it survives startup (no early crash). A Python-level exception during
UI construction / signal wiring would make the process exit immediately with
a non-zero code; a successful startup keeps the event loop alive.
"""

import os
import sys
import time
import subprocess

os.environ["QT_QPA_PLATFORM"] = "offscreen"

EXE = "dist_next/LlamaServerGUI.exe"
if not os.path.exists(EXE):
    print(f"MISSING_EXE: {EXE}")
    sys.exit(3)

proc = subprocess.Popen(
    [EXE],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
MAX_WAIT = 12.0
step = 0.5
elapsed = 0.0
while elapsed < MAX_WAIT:
    rc = proc.poll()
    if rc is not None:
        # Process exited before the timeout -> startup crash.
        print(
            f"STARTUP_CRASH: process exited early with code {rc} after {elapsed:.1f}s"
        )
        sys.exit(2)
    time.sleep(step)
    elapsed += step

print(
    f"STARTUP_OK: process alive for {elapsed:.1f}s (event loop running, no early crash)"
)
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
print("CLEANED_UP")
sys.exit(0)
