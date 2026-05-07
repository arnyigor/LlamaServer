"""HTTP клиент для опроса метрик llama-server в реальном времени."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class SlotMetrics:
    """Метрики слота из endpoint /slots."""

    id: int = 0
    is_processing: bool = False
    n_decoded: int = 0
    n_ctx: int = 0
    prompt_per_second: float = 0.0
    predicted_per_second: float = 0.0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    prompt_n: int = 0
    predicted_n: int = 0
    cache_n: int = 0


@dataclass
class ServerMetrics:
    """Метрики сервера из endpoint /metrics."""

    prompt_tokens_total: int = 0
    tokens_predicted_total: int = 0
    prompt_tokens_seconds: float = 0.0
    predicted_tokens_seconds: float = 0.0
    kv_cache_usage_ratio: float = 0.0
    kv_cache_tokens: int = 0
    requests_processing: int = 0
    requests_deferred: int = 0


class MetricsPoller(QObject):
    """Опрос метрик llama-server через HTTP."""

    slot_metrics_updated = Signal(list)  # list[SlotMetrics]
    server_metrics_updated = Signal(ServerMetrics)
    error_occurred = Signal(str)

    def __init__(
        self, base_url: str = "http://127.0.0.1:8080", poll_interval_ms: int = 1000
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._poll)
        self._is_running = False

    def start(self):
        """Запуск опроса."""
        if not self._is_running:
            self._is_running = True
            self._timer.start()
            self._poll()  # Первый опрос сразу

    def stop(self):
        """Остановка опроса."""
        self._is_running = False
        self._timer.stop()

    def set_url(self, base_url: str):
        """Изменение URL сервера."""
        self.base_url = base_url.rstrip("/")

    def _poll(self):
        """Опрос endpoint'ов."""
        if not self._is_running:
            return

        # Опрос /slots
        try:
            slots_data = self._fetch_json("/slots")
            if slots_data:
                slots = self._parse_slots(slots_data)
                self.slot_metrics_updated.emit(slots)
        except Exception as e:
            pass  # Тихо игнорируем ошибки опроса

        # Опрос /metrics
        try:
            metrics_data = self._fetch_text("/metrics")
            if metrics_data:
                metrics = self._parse_metrics(metrics_data)
                self.server_metrics_updated.emit(metrics)
        except Exception as e:
            pass

    def _fetch_json(self, endpoint: str) -> Optional[list | dict]:
        """GET запрос к endpoint, возвращает JSON."""
        try:
            url = f"{self.base_url}{endpoint}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=2) as response:
                data = response.read().decode("utf-8")
                return json.loads(data)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None
        except Exception:
            return None

    def _fetch_text(self, endpoint: str) -> Optional[str]:
        """GET запрос к endpoint, возвращает текст."""
        try:
            url = f"{self.base_url}{endpoint}"
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=2) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError):
            return None
        except Exception:
            return None

    def _parse_slots(self, data: list) -> list[SlotMetrics]:
        """Парсинг ответа /slots."""
        slots = []
        for slot_data in data:
            if not isinstance(slot_data, dict):
                continue

            slot = SlotMetrics()
            slot.id = slot_data.get("id", 0)
            slot.is_processing = slot_data.get("is_processing", False)
            slot.n_ctx = slot_data.get("n_ctx", 0)

            # Парсинг next_token
            next_token = slot_data.get("next_token", {})
            if isinstance(next_token, dict):
                slot.n_decoded = next_token.get("n_decoded", 0)

            # Парсинг timings если есть
            timings = slot_data.get("timings", {})
            if isinstance(timings, dict):
                slot.prompt_per_second = timings.get("prompt_per_second", 0.0)
                slot.predicted_per_second = timings.get("predicted_per_second", 0.0)
                slot.prompt_ms = timings.get("prompt_ms", 0.0)
                slot.predicted_ms = timings.get("predicted_ms", 0.0)
                slot.prompt_n = timings.get("prompt_n", 0)
                slot.predicted_n = timings.get("predicted_n", 0)
                slot.cache_n = timings.get("cache_n", 0)

            slots.append(slot)

        return slots

    def _parse_metrics(self, text: str) -> ServerMetrics:
        """Парсинг ответа /metrics (Prometheus format)."""
        metrics = ServerMetrics()

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Парсинг формата: name value
            parts = line.split()
            if len(parts) < 2:
                continue

            name = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue

            if name == "llamacpp:prompt_tokens_total":
                metrics.prompt_tokens_total = int(value)
            elif name == "llamacpp:tokens_predicted_total":
                metrics.tokens_predicted_total = int(value)
            elif name == "llamacpp:prompt_tokens_seconds":
                metrics.prompt_tokens_seconds = value
            elif name == "llamacpp:predicted_tokens_seconds":
                metrics.predicted_tokens_seconds = value
            elif name == "llamacpp:kv_cache_usage_ratio":
                metrics.kv_cache_usage_ratio = value
            elif name == "llamacpp:kv_cache_tokens":
                metrics.kv_cache_tokens = int(value)
            elif name == "llamacpp:requests_processing":
                metrics.requests_processing = int(value)
            elif name == "llamacpp:requests_deferred":
                metrics.requests_deferred = int(value)

        return metrics
