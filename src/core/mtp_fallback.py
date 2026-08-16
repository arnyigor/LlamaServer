"""MTP: правила авто-подбора draft и состояние fallback-перезапуска.

MtpModelRules — чистые решения по (info модели, settings): встроенный
MTP-режим, авто-draft, ручные пути и списки отключённых моделей.

MtpFallbackController — состояние повторного запуска без MTP: детекция
ошибки draft по логам, вырезание MTP-флагов и решение о retry. UI-часть
(логи, force stop, чекбоксы) остаётся в LlamaGUI.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

_MTP_VALUE_FLAGS = {
    "-md",
    "--model-draft",
    "--spec-draft-device",
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-n-min",
    "--spec-draft-p-min",
    "--spec-draft-ngl",
    "--spec-draft-type-k",
    "--spec-draft-type-v",
}


def strip_mtp_args(args: List[str]) -> List[str]:
    """Убрать из argv все MTP-специфичные флаги (со значениями)."""
    stripped = []
    i = 0
    while i < len(args):
        arg = args[i]
        base = arg.split("=", 1)[0] if str(arg).startswith("-") else arg
        if base in _MTP_VALUE_FLAGS:
            if "=" not in str(arg) and i + 1 < len(args):
                i += 2
            else:
                i += 1
            continue
        stripped.append(arg)
        i += 1
    return stripped


def has_mtp_flags(args: List[str]) -> bool:
    return any(flag in args for flag in ("--spec-type", "--model-draft", "-md"))


class MtpModelRules:
    """Решения о MTP по метаданным модели и настройкам."""

    @staticmethod
    def model_key(model_path) -> str:
        text = str(model_path or "").strip()
        if not text:
            return ""
        return os.path.normcase(os.path.abspath(text))

    @classmethod
    def info_model_key(cls, info: Optional[Dict[str, Any]], current_path: str = "") -> str:
        info = info or {}
        model_path = (
            info.get("path") or info.get("_model_path") or current_path
        )
        return cls.model_key(model_path)

    @staticmethod
    def uses_embedded_mtp_mode(info: Dict[str, Any]) -> bool:
        """True, когда llama.cpp должен использовать --spec-type draft-mtp
        без --model-draft (встроенные MTP-слои основной GGUF)."""
        arch = str(info.get("architecture") or "").lower()
        name_text = " ".join(
            str(info.get(k) or "") for k in ("path", "name", "display", "_model_path")
        ).lower()
        return (
            arch.startswith(("gemma4", "qwen"))
            and bool(info.get("mtp_capable"))
            and not info.get("is_qat")
            and "qat" not in name_text
        )

    @classmethod
    def is_draft_auto_disabled(
        cls, settings: Any, info: Optional[Dict[str, Any]] = None, current_path: str = ""
    ) -> bool:
        model_key = cls.info_model_key(info, current_path)
        disabled = getattr(settings, "spec_draft_auto_disabled_models", [])
        if not model_key or not isinstance(disabled, list):
            return False
        return model_key in {cls.model_key(path) for path in disabled}

    @classmethod
    def set_draft_auto_disabled(
        cls,
        settings: Any,
        disabled: bool,
        info: Optional[Dict[str, Any]] = None,
        current_path: str = "",
    ) -> None:
        model_key = cls.info_model_key(info, current_path)
        if not model_key:
            return
        saved = getattr(settings, "spec_draft_auto_disabled_models", [])
        saved = saved if isinstance(saved, list) else []
        normalized = {
            cls.model_key(path) for path in saved if str(path or "").strip()
        }
        if disabled:
            normalized.add(model_key)
        else:
            normalized.discard(model_key)
        settings.spec_draft_auto_disabled_models = sorted(normalized)

    @classmethod
    def manual_draft_path(
        cls, settings: Any, info: Optional[Dict[str, Any]] = None, current_path: str = ""
    ) -> str:
        model_key = cls.info_model_key(info, current_path)
        saved = getattr(settings, "spec_draft_manual_paths", {})
        if not model_key or not isinstance(saved, dict):
            return ""
        for saved_model, draft_path in saved.items():
            if cls.model_key(saved_model) == model_key:
                return str(draft_path or "").strip()
        return ""

    @classmethod
    def set_manual_draft_path(
        cls,
        settings: Any,
        draft_path: str,
        info: Optional[Dict[str, Any]] = None,
        current_path: str = "",
    ) -> None:
        model_key = cls.info_model_key(info, current_path)
        if not model_key:
            return
        saved = getattr(settings, "spec_draft_manual_paths", {})
        saved = dict(saved) if isinstance(saved, dict) else {}
        normalized = {
            cls.model_key(saved_model): str(path or "").strip()
            for saved_model, path in saved.items()
            if str(saved_model or "").strip() and str(path or "").strip()
        }
        text = str(draft_path or "").strip()
        if text:
            normalized[model_key] = text
        else:
            normalized.pop(model_key, None)
        settings.spec_draft_manual_paths = normalized
        # Пустое поле = явный отказ от авто-draft для этой модели.
        cls.set_draft_auto_disabled(settings, not bool(text), info, current_path)

    @classmethod
    def auto_mtp_supported(
        cls, settings: Any, info: Dict[str, Any], current_path: str = ""
    ) -> bool:
        """Авто-включение: встроенный MTP или доступный не-отключённый draft."""
        if cls.uses_embedded_mtp_mode(info):
            return True
        draft_path = str(info.get("mtp_draft_path") or "").strip()
        return bool(
            draft_path
            and os.path.isfile(draft_path)
            and not cls.is_draft_auto_disabled(settings, info, current_path)
        )

    @classmethod
    def auto_mtp_draft_path(
        cls, settings: Any, info: Dict[str, Any], current_path: str = ""
    ) -> str:
        if not cls.auto_mtp_supported(settings, info, current_path):
            return ""
        if cls.uses_embedded_mtp_mode(info):
            return ""
        manual_draft = cls.manual_draft_path(settings, info, current_path)
        if manual_draft and os.path.isfile(manual_draft):
            return manual_draft
        return str(info.get("mtp_draft_path") or "").strip()


class MtpFallbackController:
    """Состояние MTP-fallback: одна попытка перезапуска без MTP.

    Жизненный цикл: reset при каждом запуске с MTP-флагами → mark_failed
    по паттернам ошибок в логах → retry_plan(exit_code) при падении
    процесса решает, перезапускаться ли с вырезанными MTP-флагами.
    """

    def __init__(self):
        self._draft_error_seen = False
        self._failure_reason = ""
        self._fallback_attempted = False
        self._auto_abort_requested = False
        self._last_launch: Optional[Tuple[str, List[str], Optional[Dict[str, str]]]] = None

    @property
    def draft_error_seen(self) -> bool:
        return self._draft_error_seen

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    @property
    def fallback_attempted(self) -> bool:
        return self._fallback_attempted

    @property
    def last_launch(self):
        return self._last_launch

    def remember_launch(self, exe: str, args: List[str], env, is_retry: bool) -> None:
        self._last_launch = (exe, list(args), dict(env or {}))
        if has_mtp_flags(args):
            self._draft_error_seen = False
            self._failure_reason = ""
            self._auto_abort_requested = False
            if not is_retry:
                self._fallback_attempted = False

    def mark_failed(self, reason: str, fatal: bool = False) -> bool:
        """Зафиксировать ошибку MTP. True — фатальная, нужен немедленный abort."""
        self._draft_error_seen = True
        self._failure_reason = reason
        if fatal and not self._auto_abort_requested:
            self._auto_abort_requested = True
            return True
        return False

    def retry_plan(self, exit_code: int):
        """Решение о перезапуске без MTP.

        Возвращает (True, (exe, args, env), reason) для повторного запуска
        или (False, None, ""). Повтор выполняется ровно один раз.
        """
        if exit_code == 0 or self._fallback_attempted:
            return False, None, ""
        if not self._draft_error_seen or not self._last_launch:
            return False, None, ""

        exe, args, env = self._last_launch
        if not has_mtp_flags(args):
            return False, None, ""

        fallback_args = strip_mtp_args(args)
        if fallback_args == args:
            return False, None, ""

        self._fallback_attempted = True
        reason = self._failure_reason or "MTP initialization failed"
        self._draft_error_seen = False
        self._failure_reason = ""
        self._auto_abort_requested = False
        return True, (exe, fallback_args, env), reason
