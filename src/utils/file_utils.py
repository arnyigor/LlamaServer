"""Утилиты для работы с файлами, JSON и валидацией."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union


def validate_path(
    path: Union[str, Path], must_exist: bool = False, base_dir: Optional[Path] = None
) -> Path:
    """Валидация пути для защиты от Path Traversal.

    Args:
        path: Путь для валидации.
        must_exist: Требовать существование файла/директории.
        base_dir: Базовая директория. Путь должен находиться внутри неё.

    Returns:
        Валидированный Path объект.

    Raises:
        ValueError: При недопустимом пути или выходе за пределы base_dir.
    """
    target = Path(path).resolve()

    if must_exist and not target.exists():
        raise ValueError(f"Path does not exist: {target}")

    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            target.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path must be inside {base_resolved}, got: {target}")

    return target


def read_json_file(path: Union[str, Path]) -> Any:
    """Безопасное чтение JSON файла.

    Args:
        path: Путь к JSON файлу.

    Returns:
        Распарсенные JSON данные.
    """
    target = validate_path(path, must_exist=True)
    with open(target, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _detect_bom(path: Path) -> str:
    """Определение кодировки по наличию BOM."""
    try:
        with open(path, "rb") as f:
            return "utf-8-sig" if f.read(3) == b"\xef\xbb\xbf" else "utf-8"
    except OSError:
        return "utf-8"


def write_json_file_safely(path: Union[str, Path], data: Any) -> None:
    """Атомарная запись JSON через временный файл.

    Гарантирует целостность: либо файл записан полностью,
    либо остался нетронутым. Сохраняет оригинальную кодировку (BOM).

    Args:
        path: Путь к файлу.
        data: Данные для записи (должны быть JSON-сериализуемы).

    Raises:
        OSError: При ошибке записи или переименования.
        TypeError: При несериализуемых данных.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    encoding = _detect_bom(target) if target.exists() else "utf-8"

    try:
        content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as e:
        raise TypeError(f"Data is not JSON-serializable: {e}") from e

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.tmp",
        suffix=".json",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_or_create_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Загрузка или создание JSON файла.

    Args:
        path: Путь к JSON файлу.

    Returns:
        Словарь с данными.

    Raises:
        ValueError: Если JSON не является объектом.
    """
    if not path:
        raise ValueError("Путь к JSON не указан")
    target = Path(path)
    if target.exists():
        data = read_json_file(target)
        if not isinstance(data, dict):
            raise ValueError("Корень JSON должен быть объектом")
        return data
    return {}
