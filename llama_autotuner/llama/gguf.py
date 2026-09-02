from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import BinaryIO, Any

from llama_autotuner.models import ModelInfo, ModelKind

_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
_BLOCK_RE = re.compile(r"^(?:blk|block)\.(\d+)\.", re.IGNORECASE)
_SPLIT_FILE_RE = re.compile(
    r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE
)


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _string(f: BinaryIO) -> str:
    n = _u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def _value(f: BinaryIO, typ: int) -> Any:
    if typ == 8:
        return _string(f)
    if typ == 9:
        elem = _u32(f)
        n = _u64(f)
        if n <= 4096:
            return [_value(f, elem) for _ in range(n)]
        for _ in range(n):
            _skip_value(f, elem)
        return f"<array:{n}>"
    if typ in _TYPES:
        fmt, size = _TYPES[typ]
        return struct.unpack("<" + fmt, f.read(size))[0]
    raise ValueError(f"Unsupported GGUF metadata type {typ}")


def _skip_value(f: BinaryIO, typ: int) -> None:
    if typ == 8:
        n = _u64(f); f.seek(n, 1); return
    if typ == 9:
        elem = _u32(f); n = _u64(f)
        for _ in range(n):
            _skip_value(f, elem)
        return
    if typ in _TYPES:
        f.seek(_TYPES[typ][1], 1); return
    raise ValueError(f"Unsupported GGUF metadata type {typ}")


def _align(value: int, alignment: int) -> int:
    alignment = max(1, alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _is_expert_tensor(name: str) -> bool:
    n = name.lower()
    markers = (
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", ".experts.",
        ".expert.", "expert_gate", "expert_up", "expert_down", "ffn_exps",
    )
    return any(marker in n for marker in markers)


def _is_cpu_resident_tensor(name: str) -> bool:
    """Gemma3n-style per-layer embedding tensors are huge per-token lookup tables that
    llama.cpp keeps on CPU regardless of -ngl/-ncmoe; charging them against the GPU
    weight budget produces a wildly pessimistic (and wrong) feasibility estimate."""
    return "per_layer" in name.lower()


def _read_gguf_shard(path: Path) -> dict[str, Any]:
    """Read one GGUF header without touching its tensor payload."""
    size = os.path.getsize(path)
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = _u32(f)
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF version {version}")
        tensor_count = _u64(f)
        kv_count = _u64(f)
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(f)
            typ = _u32(f)
            meta[key] = _value(f, typ)

        tensor_infos: list[dict[str, Any]] = []
        tensor_names: list[str] = []
        type_hist: dict[int, int] = {}
        for _ in range(tensor_count):
            name = _string(f)
            tensor_names.append(name)
            n_dims = _u32(f)
            dims = [_u64(f) for _ in range(n_dims)]
            ggml_type = _u32(f)
            offset = _u64(f)
            type_hist[ggml_type] = type_hist.get(ggml_type, 0) + 1
            tensor_infos.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})

        alignment = int(meta.get("general.alignment", 32) or 32)
        data_start = _align(f.tell(), alignment)

    return {
        "path": path,
        "size": size,
        "tensor_count": int(tensor_count),
        "meta": meta,
        "tensor_infos": tensor_infos,
        "tensor_names": tensor_names,
        "type_hist": type_hist,
        "alignment": alignment,
        "data_start": data_start,
    }


def _metadata_split_count(meta: dict[str, Any]) -> int:
    for key, value in meta.items():
        if (key.endswith("split.count") or key == "split.count") and isinstance(value, int):
            return max(1, int(value))
    return 1


def _split_parts(path: Path, metadata_count: int) -> tuple[list[Path], int]:
    """Return existing shards in canonical order and the declared part count."""
    match = _SPLIT_FILE_RE.match(path.name)
    filename_count = int(match.group("count")) if match else 1
    split_count = max(1, metadata_count, filename_count)
    if not match or split_count == 1:
        return [path], split_count

    base = match.group("base")
    expected = [
        path.with_name(f"{base}-{idx:05d}-of-{split_count:05d}.gguf")
        for idx in range(1, split_count + 1)
    ]
    existing = [part for part in expected if part.is_file()]
    # The supplied path can use a non-canonical count in malformed collections. Keep it
    # inspectable, while still marking the aggregate layout incomplete below.
    if path not in existing and path.is_file():
        existing.append(path)
    return existing or [path], split_count


def inspect_gguf(path: str) -> ModelInfo:
    input_path = Path(path).expanduser().resolve()
    first_read = _read_gguf_shard(input_path)
    shard_paths, split_count = _split_parts(input_path, _metadata_split_count(first_read["meta"]))

    shards: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        if shard_path == input_path:
            shards.append(first_read)
        else:
            shards.append(_read_gguf_shard(shard_path))

    # The model command should always point at shard 00001 so llama.cpp discovers the
    # remaining parts.  For ordinary GGUFs this remains the original path.
    canonical_path = shard_paths[0] if shard_paths else input_path
    size = sum(int(shard["size"]) for shard in shards)
    meta: dict[str, Any] = dict(first_read["meta"])
    tensor_infos: list[dict[str, Any]] = []
    tensor_names: list[str] = []
    type_hist: dict[int, int] = {}
    block_bytes: dict[int, int] = {}
    block_expert_bytes: dict[int, int] = {}
    non_block_bytes = 0
    cpu_resident_bytes = 0
    tensor_data_bytes = 0

    for shard in shards:
        for key, value in shard["meta"].items():
            meta.setdefault(key, value)
        shard_infos = list(shard["tensor_infos"])
        tensor_infos.extend(shard_infos)
        tensor_names.extend(shard["tensor_names"])
        for ggml_type, count in shard["type_hist"].items():
            type_hist[int(ggml_type)] = type_hist.get(int(ggml_type), 0) + int(count)

        # Tensor offsets restart at zero inside every shard. Derive storage per file,
        # then aggregate block/expert totals across the logical model.
        shard_data_bytes = max(0, int(shard["size"]) - int(shard["data_start"]))
        tensor_data_bytes += shard_data_bytes
        sorted_infos = sorted(shard_infos, key=lambda x: int(x["offset"]))
        for i, info in enumerate(sorted_infos):
            start = int(info["offset"])
            end = int(sorted_infos[i + 1]["offset"]) if i + 1 < len(sorted_infos) else shard_data_bytes
            storage = max(0, end - start)
            match = _BLOCK_RE.match(str(info["name"]))
            if match:
                block = int(match.group(1))
                block_bytes[block] = block_bytes.get(block, 0) + storage
                if _is_expert_tensor(str(info["name"])):
                    block_expert_bytes[block] = block_expert_bytes.get(block, 0) + storage
            else:
                non_block_bytes += storage
                if _is_cpu_resident_tensor(str(info["name"])):
                    cpu_resident_bytes += storage

    arch = meta.get("general.architecture")

    def first_suffix(suffix: str):
        for k, v in meta.items():
            if k.endswith(suffix) and isinstance(v, int):
                return int(v)
        return None

    expert_count = first_suffix("expert_count")
    expert_used = first_suffix("expert_used_count")
    block_count = first_suffix("block_count")
    nextn_predict_layers = first_suffix("nextn_predict_layers") or 0
    main_block_count = None
    if block_count is not None:
        # Qwen35-family GGUFs can count integrated NextN/MTP blocks in block_count.
        # Those blocks are auxiliary draft state, not ordinary target-model layers for -ngl.
        main_block_count = max(1, int(block_count) - int(nextn_predict_layers))
    ctx = first_suffix("context_length")
    kind = ModelKind.MOE if (expert_count or expert_used or (arch and "moe" in str(arch).lower())) else ModelKind.DENSE
    has_mtp = (
        any("nextn" in n.lower() or "mtp" in n.lower() for n in tensor_names)
        or any("mtp" in k.lower() or "nextn" in k.lower() for k in meta)
        or nextn_predict_layers > 0
    )
    arch_l = str(arch or "").lower()
    vision_arch_markers = ("qwen2vl", "qwen2.5vl", "qwen3vl", "llava", "minicpmv", "vision", "clip")
    vision_value_markers = (
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
        "<image>", "image_token", "vision_token",
    )
    metadata_text_values = (str(v).lower() for v in meta.values() if isinstance(v, str))
    has_vision_hint = (
        any(marker in arch_l for marker in vision_arch_markers)
        or any(("vision" in k.lower() or "clip" in k.lower() or "mmproj" in k.lower()) for k in meta)
        or any(("vision" in n.lower() or "clip" in n.lower() or "mmproj" in n.lower()) for n in tensor_names)
        or any(any(marker in value for marker in vision_value_markers) for value in metadata_text_values)
    )

    parts_found = len({Path(shard["path"]).resolve() for shard in shards})
    layout_complete = bool(
        parts_found == split_count
        and tensor_infos
        and all(int(shard["data_start"]) <= int(shard["size"]) for shard in shards)
    )
    alignment = max(int(shard["alignment"]) for shard in shards)

    return ModelInfo(
        path=str(canonical_path),
        size_bytes=size,
        architecture=str(arch) if arch else None,
        kind=kind,
        block_count=block_count,
        context_length=ctx,
        expert_count=expert_count,
        expert_used_count=expert_used,
        has_mtp=has_mtp,
        has_vision_hint=has_vision_hint,
        metadata=meta,
        tensor_count=sum(int(shard["tensor_count"]) for shard in shards),
        tensor_data_bytes=tensor_data_bytes,
        tensor_layout_complete=layout_complete,
        tensor_alignment=alignment,
        block_tensor_bytes=block_bytes,
        block_expert_bytes=block_expert_bytes,
        non_block_tensor_bytes=non_block_bytes,
        tensor_type_histogram=type_hist,
        main_block_count=main_block_count,
        mtp_block_count=int(nextn_predict_layers),
        split_count=split_count,
        split_parts_found=parts_found,
        cpu_resident_tensor_bytes=cpu_resident_bytes,
    )
