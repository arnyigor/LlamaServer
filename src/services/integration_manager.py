# src/services/integration_manager.py
"""Менеджер интеграции с OpenCode и PI — полная реализация."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.constants import DEFAULT_LOCAL_BASE_URL, LLAMACPP_PROVIDER_ID
from src.services.integration import (
    ensure_opencode_llamacpp_provider,
    ensure_pi_llamacpp_provider,
    get_model_ids,
)
from src.utils.file_utils import load_or_create_json, write_json_file_safely


@dataclass
class IntegrationResult:
    """Результат операции интеграции."""
    success: bool
    message: str
    model_ids: List[str] = None

    def __post_init__(self):
        if self.model_ids is None:
            self.model_ids = []


class IntegrationManager:
    """
    Управляет добавлением/удалением моделей в конфиги OpenCode и PI.

    Разделяет бизнес-логику от UI, позволяет тестировать независимо.
    """

    def __init__(self, base_url: str = DEFAULT_LOCAL_BASE_URL):
        self.base_url = base_url

    def check_models(
            self, config_path: str, target: str
    ) -> IntegrationResult:
        """
        Проверка текущего состояния конфига.

        Args:
            config_path: Путь к JSON-конфигу.
            target: 'opencode' или 'pi'.

        Returns:
            IntegrationResult с текущим списком моделей.
        """
        if not config_path or not config_path.strip():
            return IntegrationResult(
                success=False,
                message="Путь к конфигу не указан",
            )

        path = Path(config_path.strip())
        if not path.exists():
            return IntegrationResult(
                success=False,
                message=f"Файл не найден: {path}",
            )

        try:
            data = load_or_create_json(path)
            model_ids = get_model_ids(data, target)
            provider_id = LLAMACPP_PROVIDER_ID
            count = len(model_ids)
            return IntegrationResult(
                success=True,
                message=(
                    f"Провайдер '{provider_id}': {count} "
                    f"{'модель' if count == 1 else 'моделей'}"
                ),
                model_ids=model_ids,
            )
        except (ValueError, OSError, KeyError) as e:
            return IntegrationResult(
                success=False,
                message=f"Ошибка чтения конфига: {e}",
            )

    def add_model(
            self,
            config_path: str,
            target: str,
            model_id: str,
            base_url: Optional[str] = None,
    ) -> IntegrationResult:
        """
        Добавление модели в конфиг.

        Args:
            config_path: Путь к JSON-конфигу.
            target: 'opencode' или 'pi'.
            model_id: Идентификатор модели (например, stem имени файла).
            base_url: URL сервера (по умолчанию используется self.base_url).

        Returns:
            IntegrationResult с результатом операции.
        """
        if not model_id or not model_id.strip():
            return IntegrationResult(
                success=False, message="Не выбрана модель"
            )

        validation = self._validate_config_path(config_path)
        if not validation.success:
            return validation

        url = base_url or self.base_url
        model_id = model_id.strip()

        try:
            path = Path(config_path.strip())
            data = load_or_create_json(path)

            if target == "opencode":
                _, models = ensure_opencode_llamacpp_provider(data, url)
                if model_id in models:
                    return IntegrationResult(
                        success=True,
                        message=f"Модель '{model_id}' уже добавлена",
                        model_ids=get_model_ids(data, target),
                    )
                models[model_id] = {}
            elif target == "pi":
                _, models_list = ensure_pi_llamacpp_provider(data, url)
                existing_ids = [
                    m.get("id") or m.get("name")
                    for m in models_list
                    if isinstance(m, dict)
                ]
                if model_id in existing_ids:
                    return IntegrationResult(
                        success=True,
                        message=f"Модель '{model_id}' уже добавлена",
                        model_ids=get_model_ids(data, target),
                    )
                models_list.append({"id": model_id, "name": model_id})
            else:
                return IntegrationResult(
                    success=False,
                    message=f"Неизвестный target: {target}",
                )

            write_json_file_safely(path, data)
            return IntegrationResult(
                success=True,
                message=f"✅ Модель '{model_id}' добавлена",
                model_ids=get_model_ids(data, target),
            )
        except (ValueError, OSError) as e:
            return IntegrationResult(
                success=False,
                message=f"Ошибка записи конфига: {e}",
            )

    def remove_model(
            self,
            config_path: str,
            target: str,
            model_id: str,
    ) -> IntegrationResult:
        """
        Удаление модели из конфига.

        Args:
            config_path: Путь к JSON-конфигу.
            target: 'opencode' или 'pi'.
            model_id: Идентификатор модели для удаления.

        Returns:
            IntegrationResult с результатом операции.
        """
        if not model_id:
            return IntegrationResult(
                success=False, message="Не выбрана модель для удаления"
            )

        validation = self._validate_config_path(config_path)
        if not validation.success:
            return validation

        try:
            path = Path(config_path.strip())
            data = load_or_create_json(path)
            removed = False

            if target == "opencode":
                providers = (
                        data.get("provider")
                        or data.get("providers")
                        or data
                )
                provider = (
                    providers.get(LLAMACPP_PROVIDER_ID, {})
                    if isinstance(providers, dict)
                    else {}
                )
                models = provider.get("models", {})
                if isinstance(models, dict) and model_id in models:
                    del models[model_id]
                    removed = True

            elif target == "pi":
                providers = (
                        data.get("providers")
                        or data.get("provider")
                        or data
                )
                provider = (
                    providers.get(LLAMACPP_PROVIDER_ID, {})
                    if isinstance(providers, dict)
                    else {}
                )
                models_list = provider.get("models", [])
                if isinstance(models_list, list):
                    before = len(models_list)
                    provider["models"] = [
                        m
                        for m in models_list
                        if isinstance(m, dict)
                           and (m.get("id") or m.get("name")) != model_id
                    ]
                    removed = len(provider["models"]) < before

            if not removed:
                return IntegrationResult(
                    success=False,
                    message=f"Модель '{model_id}' не найдена в конфиге",
                    model_ids=get_model_ids(data, target),
                )

            write_json_file_safely(path, data)
            return IntegrationResult(
                success=True,
                message=f"🗑️ Модель '{model_id}' удалена",
                model_ids=get_model_ids(data, target),
            )
        except (ValueError, OSError) as e:
            return IntegrationResult(
                success=False,
                message=f"Ошибка записи конфига: {e}",
            )

    def _validate_config_path(self, config_path: str) -> IntegrationResult:
        """Валидация пути к конфигу."""
        if not config_path or not config_path.strip():
            return IntegrationResult(
                success=False, message="Путь к конфигу не указан"
            )
        path = Path(config_path.strip())
        if not path.exists():
            return IntegrationResult(
                success=False, message=f"Файл не найден: {path}"
            )
        if path.suffix.lower() != ".json":
            return IntegrationResult(
                success=False,
                message=f"Ожидается JSON файл, получен: {path.suffix}",
            )
        return IntegrationResult(success=True, message="OK")