"""Интеграция с OpenCode и PI (llama.cpp локальный сервер)."""

from typing import Any, Dict, List, Tuple

from src.core.constants import DEFAULT_LOCAL_BASE_URL, LLAMACPP_PROVIDER_ID
def provider_container(data: Dict[str, Any], preferred_key: str) -> Dict[str, Any]:
    """Получение контейнера провайдеров из конфига."""
    if isinstance(data.get(LLAMACPP_PROVIDER_ID), dict):
        return data
    for key in (preferred_key, "providers", "provider"):
        if isinstance(data.get(key), dict):
            return data[key]
    data[preferred_key] = {}
    return data[preferred_key]


def ensure_opencode_llamacpp_provider(
    data: Dict[str, Any], base_url: str, max_context: int = 0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Обеспечение наличия OpenCode провайдера.

    Args:
        data: Конфиг (изменяется на месте).
        base_url: URL сервера llama.cpp.
        max_context: Размер окна контекста (токены). Если > 0, проставляется
            models[model_id]["limit"]["context"], чтобы агент корректно
            сжимал контекст.
    """
    providers = provider_container(data, "provider")
    provider = providers.setdefault(LLAMACPP_PROVIDER_ID, {})
    provider.setdefault("name", "llama.cpp (local)")
    provider.setdefault("npm", "@ai-sdk/openai-compatible")
    options = provider.setdefault("options", {})
    options.setdefault("baseURL", base_url or DEFAULT_LOCAL_BASE_URL)
    models = provider.setdefault("models", {})
    if not isinstance(models, dict):
        provider["models"] = {}
        models = provider["models"]
    if max_context and max_context > 0:
        for model in models.values():
            if isinstance(model, dict):
                model.setdefault("limit", {})["context"] = max_context
    return provider, models


def ensure_pi_llamacpp_provider(
    data: Dict[str, Any], base_url: str, max_context: int = 0
) -> Tuple[Dict[str, Any], List[Any]]:
    """Обеспечение наличия PI провайдера.

    Args:
        data: Конфиг (изменяется на месте).
        base_url: URL сервера llama.cpp.
        max_context: Размер окна контекста (токены). Если > 0, проставляется
            limit.context у каждой модели, чтобы агент корректно сжимал
            контекст.
    """
    providers = provider_container(data, "providers")
    provider = providers.setdefault(LLAMACPP_PROVIDER_ID, {})
    provider.setdefault("api", "openai-completions")
    provider.setdefault("apiKey", "llamacpp")
    provider.setdefault("baseUrl", base_url or DEFAULT_LOCAL_BASE_URL)
    models = provider.setdefault("models", [])
    if not isinstance(models, list):
        provider["models"] = []
        models = provider["models"]
    if max_context and max_context > 0:
        for model in models:
            if isinstance(model, dict):
                model.setdefault("limit", {})["context"] = max_context
    return provider, models


def get_model_ids(data: Dict[str, Any], target: str = "opencode") -> List[str]:
    """Универсальная функция получения списка моделей.

    Args:
        data: Данные конфигурации.
        target: Тип конфига ('opencode' или 'pi').

    Returns:
        Отсортированный список ID моделей.
    """
    providers = data.get("provider") or data.get("providers") or data
    provider = (
        providers.get(LLAMACPP_PROVIDER_ID, {}) if isinstance(providers, dict) else {}
    )
    models = provider.get("models", {} if target == "opencode" else [])

    if isinstance(models, dict):
        return sorted(str(m) for m in models.keys())
    if isinstance(models, list):
        return sorted(
            str(m.get("id") or m.get("name"))
            for m in models
            if isinstance(m, dict) and (m.get("id") or m.get("name"))
        )
    return []


def get_opencode_model_ids(data: Dict[str, Any]) -> List[str]:
    """Получение списка моделей из OpenCode конфига."""
    return get_model_ids(data, "opencode")


def get_pi_model_ids(data: Dict[str, Any]) -> List[str]:
    """Получение списка моделей из PI конфига."""
    return get_model_ids(data, "pi")
