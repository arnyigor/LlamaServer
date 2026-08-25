from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llama_autotuner.llama.api import chat_completion

ProgressFn = Callable[[str], None] | None


@dataclass(slots=True)
class VisionBenchmarkResult:
    passed: bool
    latency_seconds: float
    answer: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    error: str | None = None
    transport: str = "file"


def bundled_vision_asset() -> Path:
    path = Path(__file__).resolve().parent.parent / "assets" / "vision_test_731.png"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled Vision test image is missing: {path}")
    return path


def _content_text(response: dict) -> str:
    try:
        message = response["choices"][0]["message"]
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
    except (KeyError, IndexError, TypeError):
        return ""
    # Reasoning models can spend a small max_tokens budget entirely in reasoning and return an
    # empty visible content field.  The diagnostic may still prove image recognition from the
    # server's reasoning_content, while the retry below asks for a normal visible answer.
    if not content and isinstance(message.get("reasoning_content"), str):
        content = message["reasoning_content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()
    return str(content).strip()


def answer_recognizes_code(answer: str, expected: str = "731") -> bool:
    # Word boundaries are not reliable around CJK/punctuation, so guard only against adjacent digits.
    return bool(re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", answer or ""))


def _image_url(image_path: Path, use_media_path: bool) -> tuple[str, str]:
    if use_media_path:
        return f"file://{image_path.name}", "file"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}", "base64"


def benchmark_vision_recognition(
    base_url: str,
    image_path: Path,
    *,
    use_media_path: bool,
    progress: ProgressFn = None,
    expected_code: str = "731",
    max_tokens: int = 64,
    force_no_think_prompt: bool = False,
) -> VisionBenchmarkResult:
    """Run a deterministic image->text smoke benchmark.

    This is intentionally a capability/VRAM/latency test, not a broad VLM quality benchmark.
    The bundled image contains a large three-digit code plus colored shapes. Requiring the code
    prevents a projector from being considered validated merely because it loads successfully.
    """
    url, transport = _image_url(image_path, use_media_path)
    if progress:
        progress(f"vision recognition test: {image_path.name} (expected code {expected_code}, transport={transport})")
    payload = {
        "model": "autotuner",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": (
                    "Look at the image. What three-digit code is printed in very large black digits? "
                    "Reply with only the three digits."
                    + (" /no_think" if force_no_think_prompt else "")
                )},
            ],
        }],
        "max_tokens": max(16, int(max_tokens)),
        "temperature": 0,
        "seed": 731,
        # llama.cpp forwards this to templates that expose an enable_thinking switch (notably
        # Qwen). Templates that do not use it simply ignore the value.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.monotonic()
    try:
        response = chat_completion(base_url, payload, timeout=120.0)
    except Exception as exc:
        return VisionBenchmarkResult(
            passed=False,
            latency_seconds=time.monotonic() - start,
            answer="",
            error=str(exc),
            transport=transport,
        )
    elapsed = time.monotonic() - start
    answer = _content_text(response)
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    generated_tokens = int(usage.get("completion_tokens") or 0)
    passed = answer_recognizes_code(answer, expected_code)
    return VisionBenchmarkResult(
        passed=passed,
        latency_seconds=elapsed,
        answer=answer,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        error=None if passed else f"expected code {expected_code} was not found in the response",
        transport=transport,
    )
