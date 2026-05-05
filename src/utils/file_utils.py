"""Утилиты для работы с файлами, JSON и валидацией."""

import json
import os
import shutil
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


def json_write_encoding(path: Union[str, Path]) -> str:
    """Определение кодировки для записи JSON.

    Args:
        path: Путь к файлу.

    Returns:
        Кодировка (utf-8 или utf-8-sig).
    """
    target = Path(path)
    if target.exists():
        try:
            with open(target, "rb") as f:
                if f.read(3) == b"\xef\xbb\xbf":
                    return "utf-8-sig"
        except OSError:
            pass
    return "utf-8"


def write_json_file_safely(path: Union[str, Path], data: Any) -> None:
    """Запись JSON файла.

    Простая прямая запись без временных файлов (для совместимости с PyInstaller).

    Args:
        path: Путь к файлу.
        data: Данные для записи.
    """
    target = Path(path)
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        # Fallback: если не удалось записать, пробуем через временный файл
        import tempfile
        import os as os_module

        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
                temp_name = f.name
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            if target.exists():
                os_module.remove(target)
            os_module.rename(temp_name, target)
        except Exception:
            if temp_name and os_module.exists(temp_name):
                try:
                    os_module.unlink(temp_name)
                except OSError:
                    pass
            raise


def load_or_create_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Загрузка или создание JSON файла.

    Args:
        path: Путь к JSON файлу.

    Returns:
        Словарь с данными.
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
