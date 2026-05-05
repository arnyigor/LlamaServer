"""GGUF парсер и утилиты для работы с моделями."""

import re
import struct
from pathlib import Path
from typing import Any, Dict, Union

from src.core.constants import (
    CTX_RECOMMENDATIONS,
    DEFAULT_CTX_SIZE,
    LARGE_MODEL_THRESHOLD,
    MAX_GGUF_METADATA_COUNT,
    MEDIUM_MODEL_THRESHOLD,
    SMALL_MODEL_THRESHOLD,
)


class GGUFParseError(Exception):
    """Исключение при парсинге GGUF файла."""

    pass


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    33: "Q4_0_4_4",
    34: "Q4_0_4_8",
    35: "Q4_0_8_8",
    36: "TQ1_0",
    37: "TQ2_0",
}


def read_gguf_string(f):
    """Чтение строки из GGUF файла."""
    size_data = f.read(8)
    if len(size_data) != 8:
        raise ValueError("Unexpected GGUF EOF while reading string size")
    size = struct.unpack("<Q", size_data)[0]
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Unexpected GGUF EOF while reading string")
    return data.decode("utf-8", errors="replace")


def skip_gguf_value(f, value_type):
    """Пропуск значения в GGUF файле."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type == 8:
        size = struct.unpack("<Q", f.read(8))[0]
        f.seek(size, 1)
    elif value_type == 9:
        child_type = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        for _ in range(length):
            skip_gguf_value(f, child_type)
    elif value_type in sizes:
        f.seek(sizes[value_type], 1)
    else:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")


def read_gguf_value(f, value_type):
    """Чтение значения из GGUF файла."""
    if value_type == 0:
        return struct.unpack("<B", f.read(1))[0]
    if value_type == 1:
        return struct.unpack("<b", f.read(1))[0]
    if value_type == 2:
        return struct.unpack("<H", f.read(2))[0]
    if value_type == 3:
        return struct.unpack("<h", f.read(2))[0]
    if value_type == 4:
        return struct.unpack("<I", f.read(4))[0]
    if value_type == 5:
        return struct.unpack("<i", f.read(4))[0]
    if value_type == 6:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == 7:
        return struct.unpack("<?", f.read(1))[0]
    if value_type == 8:
        return read_gguf_string(f)
    if value_type == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if value_type == 11:
        return struct.unpack("<q", f.read(8))[0]
    if value_type == 12:
        return struct.unpack("<d", f.read(8))[0]
    skip_gguf_value(f, value_type)
    return None


def read_gguf_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Чтение метаданных из GGUF файла.

    Args:
        path: Путь к GGUF файлу.

    Returns:
        Словарь с метаданными.

    Raises:
        GGUFParseError: При повреждении или подозрительных данных.
    """
    metadata: Dict[str, Any] = {}
    file_path = Path(path)

    if not file_path.exists():
        return metadata

    if file_path.stat().st_size < 8:
        raise GGUFParseError(f"GGUF file too small: {file_path}")

    try:
        with open(file_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return metadata
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                tensor_count = struct.unpack("<I", f.read(4))[0]
                metadata_count = struct.unpack("<I", f.read(4))[0]
            else:
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                metadata_count = struct.unpack("<Q", f.read(8))[0]

            if metadata_count > MAX_GGUF_METADATA_COUNT:
                raise GGUFParseError(
                    f"Suspicious GGUF metadata count: {metadata_count} "
                    f"(max: {MAX_GGUF_METADATA_COUNT})"
                )

            metadata["gguf.version"] = version
            metadata["gguf.tensor_count"] = tensor_count
            for _ in range(metadata_count):
                key = read_gguf_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                metadata[f"{key}.type"] = GGUF_VALUE_TYPES.get(
                    value_type, str(value_type)
                )
                metadata[key] = read_gguf_value(f, value_type)
                arch = metadata.get("general.architecture", "")
                has_file_type = "general.file_type" in metadata
                has_context = any(
                    k.endswith(".context_length") and isinstance(v, int)
                    for k, v in metadata.items()
                )
                if arch and has_file_type and has_context:
                    break
    except struct.error as e:
        raise GGUFParseError(f"Corrupted GGUF structure: {e}")
    except OSError as e:
        raise GGUFParseError(f"Cannot read GGUF file: {e}")

    return metadata


def quant_from_filename(path: Union[str, Path]) -> str:
    """Определение квантования из имени файла."""
    name = Path(path).name.upper()
    # Сначала ищем полные квантования с суффиксами (Q4_K_M, Q8_0 и т.д.)
    match = re.search(
        r"(IQ[1-4]_[A-Z0-9_]+|TQ[12]_0|Q[2-8]_[A-Z0-9_]+|Q4_0_4_[48]|Q4_0_8_8|F16|F32|BF16)",
        name,
    )
    if match:
        return match.group(1)
    # Затем базовые квантования (Q4_0, Q8_0 и т.д.)
    match = re.search(r"(Q[2-8]_[0-9]+|Q[2-8])", name)
    return match.group(1) if match else ""


def detect_mmproj_for_model(path: Union[str, Path]) -> str:
    """Поиск mmproj файла рядом с моделью."""
    model_path = Path(path)
    candidates = []
    patterns = ("*mmproj*", "*projector*")
    for directory in [model_path.parent, model_path.parent.parent]:
        if not directory.exists():
            continue
        for pattern in patterns:
            for item in directory.glob(pattern):
                if item == model_path or not item.is_file():
                    continue
                if item.suffix.lower() not in {".gguf", ".bin"}:
                    continue
                name = item.name.lower()
                if "mmproj" in name or "projector" in name:
                    candidates.append(item)
    if not candidates:
        return ""
    candidates.sort(
        key=lambda i: (i.parent != model_path.parent, len(i.name), i.name.lower())
    )
    return str(candidates[0])


def is_projector_file(path: Union[str, Path]) -> bool:
    """Проверка, является ли файл projector-файлом."""
    name = Path(path).name.lower()
    return "mmproj" in name or "projector" in name


def recommend_context(info: Dict[str, Any]) -> int:
    """Рекомендация размера контекста на основе параметров модели.

    Args:
        info: Словарь с информацией о модели.

    Returns:
        Рекомендуемый размер контекста (кратный 512).
    """
    quant = (info.get("quant") or "").upper()
    size_gib = info.get("size_gib") or 0
    model_ctx = info.get("context_length") or 0

    recommended = DEFAULT_CTX_SIZE
    for prefix, ctx in CTX_RECOMMENDATIONS.items():
        if quant.startswith(prefix):
            recommended = ctx
            break

    if size_gib >= LARGE_MODEL_THRESHOLD:
        recommended = min(recommended, 8192)
    elif size_gib >= MEDIUM_MODEL_THRESHOLD:
        recommended = min(recommended, 12288)
    elif size_gib <= SMALL_MODEL_THRESHOLD and quant:
        # Для маленьких моделей увеличиваем контекст, но не выше модельного
        recommended = max(recommended, 8192)
        if model_ctx:
            recommended = min(recommended, model_ctx)

    if model_ctx:
        recommended = min(recommended, model_ctx)

    return max(512, int(recommended // 512 * 512))


def extract_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Извлечение информации о модели из GGUF файла.

    Args:
        path: Путь к GGUF файлу.

    Returns:
        Словарь с информацией о модели.
    """
    file_path = Path(path)
    info: Dict[str, Any] = {
        "path": str(file_path),
        "name": file_path.name,
        "size_gib": round(file_path.stat().st_size / (1024**3), 2)
        if file_path.exists()
        else 0,
        "architecture": "",
        "context_length": 0,
        "quant": quant_from_filename(file_path),
        "mmproj_path": detect_mmproj_for_model(file_path),
        "metadata_error": "",
    }
    try:
        metadata = read_gguf_metadata(file_path)
        arch = metadata.get("general.architecture", "")
        info["architecture"] = arch
        file_type = metadata.get("general.file_type")
        if isinstance(file_type, int) and file_type in GGUF_FILE_TYPES:
            info["quant"] = GGUF_FILE_TYPES[file_type]
        ctx_key = f"{arch}.context_length" if arch else ""
        context_length = metadata.get(ctx_key)
        if not isinstance(context_length, int):
            for key, value in metadata.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    context_length = value
                    break
        if isinstance(context_length, int):
            info["context_length"] = context_length
    except GGUFParseError as exc:
        info["metadata_error"] = f"GGUF parse error: {exc}"
    except Exception as exc:
        info["metadata_error"] = f"Unexpected error: {exc}"
    info["recommended_ctx"] = recommend_context(info)
    return info
