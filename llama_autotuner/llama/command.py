from __future__ import annotations

from llama_autotuner.models import Candidate
from llama_autotuner.llama.capabilities import LlamaCapabilities


def build_server_command(exe: str, model: str, c: Candidate, port: int = 8080,
                         caps: LlamaCapabilities | None = None, media_path: str | None = None) -> list[str]:
    args = [
        exe, "-m", model, "--host", "127.0.0.1", "--port", str(port),
        "-ngl", str(c.ngl), "-t", str(c.threads), "-tb", str(c.threads_batch),
        "-c", str(c.ctx), "-b", str(c.batch), "-ub", str(c.ubatch), "-np", "1",
        "-ctk", c.kv_k, "-ctv", c.kv_v,
    ]
    if c.ncmoe is not None:
        args += ["-ncmoe", str(c.ncmoe)]
    args += [
        "--fit", "off", "--flash-attn", "on", "--jinja", "--metrics",
        "--device", "CUDA0", "--split-mode", "none", "--main-gpu", "0",
    ]
    if c.vision:
        if not c.mmproj:
            raise ValueError("Vision candidate requires a companion mmproj GGUF; text-only fallback is forbidden")
        args += ["--mmproj", c.mmproj]
        if media_path:
            args += ["--media-path", media_path]
    else:
        args += ["--no-mmproj"]
    if c.load_mode:
        args += ["--load-mode", c.load_mode]
    args += ["--cache-prompt"]
    if c.mtp:
        args += [
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(c.mtp_n_max),
            "--spec-draft-p-min", str(c.mtp_p_min), "--spec-draft-ngl", "all",
            "--spec-draft-device", "CUDA0",
        ]
        if c.draft_kv_k:
            args += ["-ctkd", c.draft_kv_k]
        if c.draft_kv_v:
            args += ["-ctvd", c.draft_kv_v]
    args.extend(c.extra_args)
    if caps:
        unknown = [x for x in args[1:] if x.startswith("--") and not caps.supports(x)]
        allow = {"--host", "--port"}
        unknown = [x for x in unknown if x not in allow]
        if unknown:
            raise ValueError(f"Unsupported llama-server arguments: {sorted(set(unknown))}")
    return args
