from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


@dataclass(slots=True)
class ModelEntry:
    path: Path
    relative_path: str
    size_bytes: int
    split_count: int = 1
    split_parts_found: int = 1

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024 ** 3)


def _is_auxiliary_gguf(path: Path) -> bool:
    name = path.name.lower()
    # Projectors and standalone draft files are not target models for the main --model selector.
    return name.startswith("mmproj") or "-mmproj" in name or name.startswith("draft-")


def _entry_size(path: Path) -> tuple[int, int, int]:
    m = _SPLIT_RE.match(path.name)
    if not m:
        return path.stat().st_size, 1, 1
    count = int(m.group("count"))
    if int(m.group("idx")) != 1:
        return 0, count, 0
    base = m.group("base")
    total = 0
    found = 0
    for idx in range(1, count + 1):
        shard = path.with_name(f"{base}-{idx:05d}-of-{count:05d}.gguf")
        if shard.exists():
            total += shard.stat().st_size
            found += 1
    return total or path.stat().st_size, count, found


def model_entry(path: str | Path, relative_path: str | None = None) -> ModelEntry:
    """Describe a launchable GGUF path using the complete local split set."""
    resolved = Path(path).expanduser().resolve()
    match = _SPLIT_RE.match(resolved.name)
    if match and int(match.group("idx")) != 1:
        first = resolved.with_name(
            f"{match.group('base')}-00001-of-{int(match.group('count')):05d}.gguf"
        )
        if first.is_file():
            resolved = first
            relative_path = None
    size, split_count, parts_found = _entry_size(resolved)
    return ModelEntry(
        path=resolved,
        relative_path=relative_path or resolved.name,
        size_bytes=size,
        split_count=split_count,
        split_parts_found=parts_found,
    )


def scan_models(root: str | Path) -> list[ModelEntry]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Models root does not exist or is not a directory: {root_path}")

    entries: list[ModelEntry] = []
    for path in root_path.rglob("*.gguf"):
        if _is_auxiliary_gguf(path):
            continue
        m = _SPLIT_RE.match(path.name)
        if m and int(m.group("idx")) != 1:
            continue
        try:
            size, split_count, parts_found = _entry_size(path)
        except OSError:
            continue
        entries.append(ModelEntry(
            path=path.resolve(),
            relative_path=str(path.relative_to(root_path)),
            size_bytes=size,
            split_count=split_count,
            split_parts_found=parts_found,
        ))
    entries.sort(key=lambda e: e.relative_path.lower())
    return entries


def filter_models(entries: list[ModelEntry], query: str | None) -> list[ModelEntry]:
    if not query:
        return entries
    q = query.lower()
    return [e for e in entries if q in e.relative_path.lower() or q in e.path.name.lower()]


def preferred_mmproj(paths: list[str | Path]) -> Path | None:
    """Choose a unique, practical projector when siblings differ only by precision.

    BF16 is the preferred automatic choice: it is substantially smaller than F32 without using a
    quantized projector.  If several files share the best rank, compatibility is still ambiguous
    and the caller must ask the user instead of guessing between model-specific projectors.
    """
    candidates = [Path(p).resolve() for p in paths]
    if not candidates:
        return None

    def rank(path: Path) -> int:
        name = path.name.lower()
        if "bf16" in name:
            return 0
        if "f16" in name or "fp16" in name:
            return 1
        if "q8" in name:
            return 2
        if "f32" in name or "fp32" in name:
            return 4
        return 3

    best_rank = min(rank(p) for p in candidates)
    best = [p for p in candidates if rank(p) == best_rank]
    return best[0] if len(best) == 1 else None
