"""HTTP клиент для опроса метрик llama-server в реальном времени.

HTTP-запросы выполняются в фоновом потоке (_MetricsFetchWorker), чтобы не
блокировать UI-поток: llama-server может отвечать на /slots медленно во
время генерации, а синхронный urlopen(timeout=2) на главном потоке каждые
250 мс приводил к зависанию интерфейса.

Парсинг и расчёт скоростей остаются в MetricsPoller (главный поток) — это
дешёвые операции. Воркер передаёт вместе с сырыми данными метку времени
опроса, чтобы скорости считались по реальному времени опроса, а не по
времени обработки (важно при накоплении сигналов в очереди событий).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal


@dataclass
class SlotMetrics:
    """Метрики слота из endpoint /slots.

    Поля соответствуют актуальному JSON llama-server (server_slot::to_json()).
    Мгновенные скорости prompt_per_second/predicted_per_second вычисляются
    в poller'е по дельтам между опросами, так как /slots больше не содержит
    устаревший объект timings.
    """

    id: int = 0
    is_processing: bool = False
    n_ctx: int = 0
    n_prompt_tokens: int = 0
    n_prompt_tokens_processed: int = 0
    n_prompt_tokens_cache: int = 0
    n_decoded: int = 0
    n_remain: int = -1
    has_next_token: bool = False
    prompt_per_second: float = 0.0
    predicted_per_second: float = 0.0


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


class _MetricsFetchWorker(QThread):
    """Фоновый поток: только HTTP-опросы /slots и /metrics.

    Отдаёт сырые данные вместе с меткой времени опроса (time.monotonic),
    чтобы парсинг на главном потоке мог корректно вычислить скорости даже
    при задержке обработки сигналов в очереди событий.
    """

    slots_fetched = Signal(object, float)  # (list[dict], timestamp)
    metrics_fetched = Signal(str, float)  # (text, timestamp)
    error_occurred = Signal(str)

    def __init__(
        self,
        base_url: str,
        slots_interval_ms: int,
        metrics_interval_ms: int,
    ):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._slots_interval_s = max(slots_interval_ms, 50) / 1000.0
        self._metrics_interval_s = max(metrics_interval_ms, 500) / 1000.0
        self._stop_flag = threading.Event()
        self._last_metrics_at = 0.0

    def set_base_url(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def stop(self):
        self._stop_flag.set()

    def run(self):
        while not self._stop_flag.is_set():
            now = time.monotonic()
            try:
                slots_data = self._fetch_json("/slots")
                if slots_data is not None:
                    self.slots_fetched.emit(slots_data, now)
            except Exception as e:
                self.error_occurred.emit(f"/slots poll failed: {e}")

            if now - self._last_metrics_at >= self._metrics_interval_s:
                self._last_metrics_at = now
                try:
                    metrics_text = self._fetch_text("/metrics")
                    if metrics_text is not None:
                        self.metrics_fetched.emit(metrics_text, now)
                except Exception as e:
                    self.error_occurred.emit(f"/metrics poll failed: {e}")

            self._stop_flag.wait(self._slots_interval_s)

    def _fetch_json(self, endpoint: str) -> Optional[list | dict]:
        """GET запрос к endpoint, возвращает JSON."""
        try:
            url = f"{self._base_url}{endpoint}"
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
            url = f"{self._base_url}{endpoint}"
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=2) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError):
            return None
        except Exception:
            return None


class MetricsPoller(QObject):
    """Опрос метрик llama-server через HTTP в фоновом потоке."""

    slot_metrics_updated = Signal(list)  # list[SlotMetrics]
    server_metrics_updated = Signal(ServerMetrics)
    error_occurred = Signal(str)

    def __init__(
        self, base_url: str = "http://127.0.0.1:8080", poll_interval_ms: int = 1000
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._poll_interval_ms = max(poll_interval_ms, 50)
        # /metrics меняется редко (кумулятивные счётчики), опрашиваем реже.
        self._metrics_interval_ms = max(self._poll_interval_ms * 8, 2000)
        self._worker: Optional[_MetricsFetchWorker] = None
        self._is_running = False
        # Состояние предыдущего опроса для расчёта скорости по дельтам.
        self._slot_rate_state: dict[int, dict] = {}

    def start(self):
        """Запуск опроса."""
        self._slot_rate_state = {}
        if not self._is_running:
            self._is_running = True
            self._start_worker()

    def stop(self):
        """Остановка опроса."""
        self._is_running = False
        self._stop_worker()

    def set_url(self, base_url: str):
        """Изменение URL сервера."""
        self.base_url = base_url.rstrip("/")
        self._slot_rate_state = {}
        if self._worker is not None:
            self._worker.set_base_url(self.base_url)

    def _start_worker(self):
        self._stop_worker()
        worker = _MetricsFetchWorker(
            self.base_url, self._poll_interval_ms, self._metrics_interval_ms
        )
        self._worker = worker
        worker.slots_fetched.connect(self._on_slots_fetched)
        worker.metrics_fetched.connect(self._on_metrics_fetched)
        worker.error_occurred.connect(self.error_occurred)
        worker.start()

    def _stop_worker(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.stop()
            worker.wait(2000)
            worker.deleteLater()

    def _on_slots_fetched(self, slots_data, timestamp: float):
        if not self._is_running or self._worker is None:
            return
        slots = self._parse_slots(slots_data, now=timestamp)
        self.slot_metrics_updated.emit(slots)

    def _on_metrics_fetched(self, metrics_text: str, timestamp: float):
        if not self._is_running or self._worker is None:
            return
        metrics = self._parse_metrics(metrics_text)
        self.server_metrics_updated.emit(metrics)

    def _parse_slots(
        self, data: list, now: Optional[float] = None
    ) -> list[SlotMetrics]:
        """Парсинг ответа /slots.

        Мгновенные скорости считаются по дельте счётчиков между опросами:
        predicted_per_second = дельта n_decoded / время между опросами,
        prompt_per_second   = дельта n_prompt_tokens_processed / время между опросами.
        Сброс счётчиков на новом запросе (уменьшение значений) не даёт отрицательную
        скорость — дельта в этом случае просто игнорируется.

        Параметр now — метка времени опроса из фонового воркера; если не задан,
        используется текущее время (важно для тестов).
        """
        now = time.monotonic() if now is None else now
        slots = []
        for slot_data in data:
            if not isinstance(slot_data, dict):
                continue

            slot = SlotMetrics()
            slot.id = slot_data.get("id", 0)
            slot.is_processing = bool(slot_data.get("is_processing", False))
            slot.n_ctx = slot_data.get("n_ctx", 0)
            slot.n_prompt_tokens = slot_data.get("n_prompt_tokens", 0)
            slot.n_prompt_tokens_processed = slot_data.get(
                "n_prompt_tokens_processed", 0
            )
            slot.n_prompt_tokens_cache = slot_data.get("n_prompt_tokens_cache", 0)

            # Реальный llama-server отдаёт next_token как список из одного
            # объекта (server_slot::to_json), в некоторых сборках — как объект.
            next_token = slot_data.get("next_token", {})
            if isinstance(next_token, dict):
                next_token_obj = next_token
            elif (
                isinstance(next_token, list)
                and next_token
                and isinstance(next_token[0], dict)
            ):
                next_token_obj = next_token[0]
            else:
                next_token_obj = {}
            slot.n_decoded = next_token_obj.get("n_decoded", 0)
            slot.n_remain = next_token_obj.get("n_remain", -1)
            slot.has_next_token = bool(next_token_obj.get("has_next_token", False))

            prev = self._slot_rate_state.get(slot.id)
            if prev is not None:
                dt = now - prev["time"]
                if dt > 0 and slot.is_processing and prev["processing"]:
                    decoded_delta = slot.n_decoded - prev["n_decoded"]
                    if 0 <= decoded_delta <= 1_000_000:
                        slot.predicted_per_second = decoded_delta / dt
                    prompt_delta = (
                        slot.n_prompt_tokens_processed - prev["n_prompt_processed"]
                    )
                    if 0 <= prompt_delta <= 10_000_000:
                        slot.prompt_per_second = prompt_delta / dt

            self._slot_rate_state[slot.id] = {
                "time": now,
                "processing": slot.is_processing,
                "n_decoded": slot.n_decoded,
                "n_prompt_processed": slot.n_prompt_tokens_processed,
            }
            slots.append(slot)

        current_ids = {s.id for s in slots}
        if len(self._slot_rate_state) != len(current_ids):
            self._slot_rate_state = {
                slot_id: state
                for slot_id, state in self._slot_rate_state.items()
                if slot_id in current_ids
            }

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
