"""Best-effort model load progress detection from llama.cpp logs."""

from __future__ import annotations


def progress_from_load_line(line: str) -> tuple[int, str] | None:
    """Return a monotonic phase estimate for loading a model into memory."""
    low = line.lower()
    if (
        "main: server is listening" in low
        or "llama server listening" in low
        or "llama_server: model loaded" in low
        or "llama_server: listening on" in low
        or "llama_print_timings" in low
        or "llama_perf_context_print" in low
    ):
        return 100, "ready"
    if "llama_context:" in low or "llama_init_from_model" in low:
        return 90, "creating context"
    if "buffer size" in low:
        if "kv" in low:
            return 82, "allocating KV cache"
        if "model" in low:
            return 75, "allocating weights"
        return 78, "allocating buffers"
    if "load_tensors:" in low:
        if "offloaded" in low or "offloading" in low:
            return 65, "offloading tensors"
        if "loading" in low or "tensor" in low:
            return 55, "loading tensors"
        return 50, "load tensors"
    if "print_info:" in low:
        return 35, "model info"
    if "llama_model_load:" in low:
        return 20, "model loader"
    if "llama_model_loader:" in low and "- kv" in low:
        return 12, "metadata"
    if "llama_model_loader:" in low:
        return 8, "opening GGUF"
    return None
