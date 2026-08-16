"""Runtime statistics: контроллер накопления и экспорт в Markdown."""

from __future__ import annotations

import time
from collections.abc import Mapping

from PySide6.QtCore import QObject, Signal

from src.core.constants import MAX_ACTIVE_TIME_DT, format_duration


class RuntimeStatsController(QObject):
    """Накопление runtime-статистики (токены, активное время, saved).

    Логика подсчёта не знает про виджеты: результаты публикуются сигналами,
    LlamaGUI подключает их к QLabel. Источники: дельты /slots (основной),
    /metrics (только догоняющая синхронизация вверх), логи
    llama_print_timings (точное время текущего запроса).
    """

    # {total, task, prompt, generated} — числа для лейбла Tokens.
    tokens_changed = Signal(dict)
    # (pp, tg) секунды для лейбла Active (total за сессию).
    active_time_changed = Signal(float, float)
    # (pp, tg) секунды для лейбла Current (текущий/последний запрос).
    current_time_changed = Signal(float, float)
    # (last, total) для лейбла Saved.
    saved_changed = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._latest_token_total = 0
        self._latest_prompt_total = 0
        self._latest_predicted_total = 0
        self._token_baseline_total = 0
        self._saved_token_total = 0
        self._saved_last_total = 0
        # Baseline-смещения "сессии": total/task токены и активное время
        # отображаются относительно точки Reset session, поэтому следующий
        # опрос /metrics не вернёт старые значения на экран.
        self._session_base_prompt = 0
        self._session_base_predicted = 0
        self._session_base_total = 0
        self._session_base_active_pp = 0.0
        self._session_base_active_tg = 0.0
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._last_slots = []
        self._request_token_base = {}
        # Кумулятивные значения из /metrics (с момента старта сервера).
        # Используются только для "догоняющей" синхронизации вверх, т.к.
        # llama.cpp обновляет их лишь по завершении запросов.
        self._metrics_prompt_total = 0
        self._metrics_predicted_total = 0
        # Активное время работы модели (секунды PP/TG).
        #   _active_*    — total: сумма интервалов /slots за запуск сервера;
        #   _cur_*       — current: время текущего/последнего запроса
        #                  (точное значение приходит из llama_print_timings).
        self._active_prompt_s = 0.0
        self._active_predicted_s = 0.0
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._was_processing = False
        self._last_poll_time = None

    # -- Публикация для лейблов ------------------------------------------

    def token_display(self) -> dict:
        total = max(self._latest_token_total - self._session_base_total, 0)
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        prompt = max(self._latest_prompt_total - self._session_base_prompt, 0)
        generated = max(self._latest_predicted_total - self._session_base_predicted, 0)
        return {
            "total": total,
            "task": task_total,
            "prompt": prompt,
            "generated": generated,
        }

    def active_time_display(self) -> tuple[float, float]:
        """(pp, tg) активного времени total за текущую сессию."""
        pp = max(self._active_prompt_s - self._session_base_active_pp, 0.0)
        tg = max(self._active_predicted_s - self._session_base_active_tg, 0.0)
        return pp, tg

    def current_time_display(self) -> tuple[float, float]:
        return self._cur_prompt_s, self._cur_predicted_s

    def saved_display(self) -> tuple[int, int]:
        return int(self._saved_last_total or 0), int(self._saved_token_total or 0)

    def refresh_all(self):
        """Переизлучить все сигналы (например, после ручной правки состояния)."""
        self._emit_tokens()
        self._emit_active_time()
        self._emit_current_time()
        self._emit_saved()

    def _emit_tokens(self):
        self.tokens_changed.emit(self.token_display())

    def _emit_active_time(self):
        self.active_time_changed.emit(*self.active_time_display())

    def _emit_current_time(self):
        self.current_time_changed.emit(*self.current_time_display())

    def _emit_saved(self, last_total=None):
        last = max(int(last_total if last_total is not None else self._saved_last_total), 0)
        self.saved_changed.emit(last, self._saved_token_total)

    # -- Источники данных --------------------------------------------------

    def update_slot_metrics(self, slots):
        """Обработать очередной опрос /slots.

        Возвращает словарь для отрисовки speed/request лейблов или None,
        если опрос пришёл без слотов. Накопление токенов выполняется в конце,
        чтобы request-счётчики считались по состоянию до дельт этого опроса.
        """
        self._last_slots = list(slots)
        # Активное время: интервалы опросов, пока хоть один слот обрабатывает.
        # Большой зазор между опросами (> MAX_ACTIVE_TIME_DT) — пауза/простой,
        # его не считаем: точный current приходит из логов llama_print_timings.
        now = time.monotonic()
        active = [slot for slot in slots if getattr(slot, "is_processing", False)]
        if active and not self._was_processing:
            # Переход idle → processing: начался новый запрос, время текущего
            # запроса обнуляется.
            self._cur_prompt_s = 0.0
            self._cur_predicted_s = 0.0
            self._emit_current_time()
        self._was_processing = bool(active)
        if self._last_poll_time is not None and active:
            dt = max(now - self._last_poll_time, 0.0)
            if dt <= MAX_ACTIVE_TIME_DT:
                self._accumulate_active_time(dt, active)
        self._last_poll_time = now

        if not slots:
            return None

        visible = active or [
            slot
            for slot in slots
            if getattr(slot, "n_prompt_tokens", 0) or getattr(slot, "n_decoded", 0)
        ]
        if not visible:
            return {"visible": False, "prompt_speed": 0.0, "predicted_speed": 0.0,
                    "prompt_tokens": 0, "predicted_tokens": 0}

        prompt_speed = sum(
            max(float(getattr(slot, "prompt_per_second", 0.0) or 0.0), 0.0)
            for slot in visible
        )
        predicted_speed = sum(
            max(float(getattr(slot, "predicted_per_second", 0.0) or 0.0), 0.0)
            for slot in visible
        )
        prompt_tokens = sum(
            self._request_counter_value(slot, "n_prompt_tokens", 0)
            for slot in visible
        )
        predicted_tokens = sum(
            self._request_counter_value(slot, "n_decoded", 1) for slot in visible
        )

        self._accumulate_slot_tokens(slots)
        return {
            "visible": True,
            "prompt_speed": prompt_speed,
            "predicted_speed": predicted_speed,
            "prompt_tokens": int(prompt_tokens),
            "predicted_tokens": int(predicted_tokens),
        }

    def update_server_metrics(self, metrics):
        """Обработать опрос /metrics (кумулятивные счётчики сервера)."""
        prompt_total = int(getattr(metrics, "prompt_tokens_total", 0) or 0)
        predicted_total = int(getattr(metrics, "tokens_predicted_total", 0) or 0)
        total = prompt_total + predicted_total
        if total <= 0:
            return
        self._metrics_prompt_total = prompt_total
        self._metrics_predicted_total = predicted_total
        if total < self._token_baseline_total:
            self._token_baseline_total = 0
        self._apply_metrics_catch_up()
        # НЕ используем llamacpp:prompt_tokens_seconds / predicted_tokens_seconds
        # из /metrics как длительность: это throughput (токены/сек), а не время.
        # Точное время PP/TG берём из логов llama_print_timings.
        self._sync_latest_token_totals()
        self._emit_tokens()

    def set_log_timing(self, pp_seconds: float, tg_seconds: float):
        """Точное время завершённого запроса из llama_print_timings — им
        заменяем current (живой подсчёт теряет первый интервал опроса).
        Total остаётся живой суммой интервалов /slots."""
        self._cur_prompt_s = pp_seconds
        self._cur_predicted_s = tg_seconds
        self._emit_current_time()

    # -- Сбросы --------------------------------------------------------------

    def reset_task(self) -> int:
        """Сохранить текущую задачу в Saved и начать отсчёт новой с нуля.

        Обнуляет task-счётчик, Current time и Request. Total-токены и Active
        время (server-scope) не трогает — для них есть reset_session.
        Возвращает число сохранённых токенов (для лога).
        """
        self._sync_latest_token_totals()
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        self._saved_token_total += task_total
        self._token_baseline_total = self._latest_token_total
        self._reset_request_token_baseline()
        self._saved_last_total = max(int(task_total), 0)
        self._emit_saved(last_total=task_total)
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._last_poll_time = time.monotonic()
        self._emit_current_time()
        self._emit_tokens()
        return task_total

    def reset_session(self):
        """Обнулить все живые счётчики сессии (total/task, время, Request).

        Saved-история сохраняется. Реализовано через baseline-смещения,
        поэтому следующий опрос /metrics не вернёт старые значения на экран.
        """
        self._sync_latest_token_totals()
        self._session_base_prompt = self._latest_prompt_total
        self._session_base_predicted = self._latest_predicted_total
        self._session_base_total = self._latest_token_total
        self._session_base_active_pp = self._active_prompt_s
        self._session_base_active_tg = self._active_predicted_s
        self._token_baseline_total = self._latest_token_total
        self._reset_request_token_baseline()
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._last_poll_time = time.monotonic()
        self._emit_tokens()
        self._emit_active_time()
        self._emit_current_time()

    def reset_saved(self):
        """Обнулить накопленную Saved-историю (last и total)."""
        self._saved_token_total = 0
        self._emit_saved(last_total=0)

    def reset_server_scope(self):
        """Сброс всего server-scope состояния на старте нового запуска сервера."""
        self._slot_prompt_total = 0
        self._slot_predicted_total = 0
        self._slot_token_seen = {}
        self._last_slots = []
        self._request_token_base = {}
        self._metrics_prompt_total = 0
        self._metrics_predicted_total = 0
        self._session_base_prompt = 0
        self._session_base_predicted = 0
        self._session_base_total = 0
        self._session_base_active_pp = 0.0
        self._session_base_active_tg = 0.0
        self._token_baseline_total = 0
        # _latest_* сознательно не сбрасываются (как и в старом коде):
        # до первых данных нового запуска продолжаем показывать прошлые totals.
        self._active_prompt_s = 0.0
        self._active_predicted_s = 0.0
        self._cur_prompt_s = 0.0
        self._cur_predicted_s = 0.0
        self._was_processing = False
        self._last_poll_time = None
        self._emit_active_time()
        self._emit_current_time()

    # -- Внутренняя логика накопления ------------------------------------

    def _sync_latest_token_totals(self):
        slot_total = int(getattr(self, "_slot_prompt_total", 0) or 0) + int(
            getattr(self, "_slot_predicted_total", 0) or 0
        )
        metrics_total = int(getattr(self, "_metrics_prompt_total", 0) or 0) + int(
            getattr(self, "_metrics_predicted_total", 0) or 0
        )
        if slot_total <= 0 and metrics_total <= 0 and self._latest_token_total > 0:
            return
        self._apply_metrics_catch_up()
        self._latest_prompt_total = self._slot_prompt_total
        self._latest_predicted_total = self._slot_predicted_total
        self._latest_token_total = (
            self._latest_prompt_total + self._latest_predicted_total
        )

    def _apply_metrics_catch_up(self):
        """Синхронизация счётчиков с /metrics — только вверх.

        /metrics считает кумулятивно с момента старта сервера, но llama.cpp
        обновляет его счётчики только по завершении запроса (см. вызовы
        metrics.on_prediction / metrics.on_prompt_eval в server-context.cpp).
        Поэтому /metrics не может быть единственным источником (во время
        генерации числа замирают), а используется лишь как "догоняющий":
        если сервер знает больше — подтягиваем суммы вверх.
        """
        if self._metrics_prompt_total > self._slot_prompt_total:
            self._slot_prompt_total = self._metrics_prompt_total
        if self._metrics_predicted_total > self._slot_predicted_total:
            self._slot_predicted_total = self._metrics_predicted_total

    def _accumulate_slot_tokens(self, slots):
        """Накопление токенов из дельт /slots.

        Слоты накапливаются всегда: счётчики n_prompt_tokens_processed /
        n_decoded сохраняются у слота даже после завершения запроса
        (сбрасываются только при старте следующего), поэтому дельты
        покрывают и "быстрые" запросы, целиком уложившиеся между опросами.
        """
        for slot in slots:
            slot_id = int(getattr(slot, "id", 0) or 0)
            prompt_tokens = int(getattr(slot, "n_prompt_tokens_processed", 0) or 0)
            predicted_tokens = int(getattr(slot, "n_decoded", 0) or 0)
            previous_prompt, previous_predicted = self._slot_token_seen.get(
                slot_id, (0, 0)
            )
            prompt_delta = (
                prompt_tokens - previous_prompt
                if prompt_tokens >= previous_prompt
                else prompt_tokens
            )
            predicted_delta = (
                predicted_tokens - previous_predicted
                if predicted_tokens >= previous_predicted
                else predicted_tokens
            )
            self._slot_prompt_total += max(prompt_delta, 0)
            self._slot_predicted_total += max(predicted_delta, 0)
            self._slot_token_seen[slot_id] = (prompt_tokens, predicted_tokens)
        self._sync_latest_token_totals()
        self._emit_tokens()

    def _accumulate_active_time(self, dt: float, active):
        """Накопление активного времени по интервалу между опросами.

        dt — время между двумя последовательными опросами /slots, когда
        активен хотя бы один слот. Распределение между PP и TG — по долям
        мгновенных скоростей; точную разбивку завершённого запроса потом
        дают логи llama_print_timings (для current).
        """
        prompt_speed = sum(
            max(float(getattr(slot, "prompt_per_second", 0.0) or 0.0), 0.0)
            for slot in active
        )
        predicted_speed = sum(
            max(float(getattr(slot, "predicted_per_second", 0.0) or 0.0), 0.0)
            for slot in active
        )
        if prompt_speed > 0 and predicted_speed > 0:
            total_speed = prompt_speed + predicted_speed
            dt_pp = dt * prompt_speed / total_speed
            dt_tg = dt * predicted_speed / total_speed
        elif predicted_speed > 0:
            dt_pp, dt_tg = 0.0, dt
        elif prompt_speed > 0:
            dt_pp, dt_tg = dt, 0.0
        else:
            # Скорость ещё не измерилась (первый опрос запроса): интервал
            # пропускаем — current получит точное значение из логов.
            return
        self._active_prompt_s += dt_pp
        self._active_predicted_s += dt_tg
        self._cur_prompt_s += dt_pp
        self._cur_predicted_s += dt_tg
        self._emit_active_time()
        self._emit_current_time()

    def _reset_request_token_baseline(self):
        self._request_token_base = {}
        for slot in getattr(self, "_last_slots", []):
            slot_id = int(getattr(slot, "id", 0) or 0)
            self._request_token_base[slot_id] = (
                int(getattr(slot, "n_prompt_tokens", 0) or 0),
                int(getattr(slot, "n_decoded", 0) or 0),
            )

    def _request_counter_value(self, slot, attr_name: str, base_index: int) -> int:
        slot_id = int(getattr(slot, "id", 0) or 0)
        value = int(getattr(slot, attr_name, 0) or 0)
        base = getattr(self, "_request_token_base", {}).get(slot_id)
        if base is None:
            return value
        base_value = int(base[base_index] or 0)
        if value < base_value:
            base_values = list(base)
            base_values[base_index] = 0
            self._request_token_base[slot_id] = tuple(base_values)
            return value
        return max(value - base_value, 0)

    def _current_request_token_counts(self) -> tuple[int, int]:
        slots = list(getattr(self, "_last_slots", []) or [])
        active = [slot for slot in slots if getattr(slot, "is_processing", False)]
        visible = active or [
            slot
            for slot in slots
            if getattr(slot, "n_prompt_tokens", 0) or getattr(slot, "n_decoded", 0)
        ]
        if not visible:
            return 0, 0

        prompt = sum(
            self._request_counter_value(slot, "n_prompt_tokens", 0)
            for slot in visible
        )
        generated = sum(
            self._request_counter_value(slot, "n_decoded", 1) for slot in visible
        )
        return max(int(prompt), 0), max(int(generated), 0)

    # -- Снапшот для экспорта ----------------------------------------------

    def stats_snapshot(self) -> dict:
        """Числовая часть runtime_stats_snapshot (токены и время)."""
        self._sync_latest_token_totals()
        request_prompt, request_generated = self._current_request_token_counts()
        total = max(self._latest_token_total - self._session_base_total, 0)
        task_total = max(self._latest_token_total - self._token_baseline_total, 0)
        prompt = max(self._latest_prompt_total - self._session_base_prompt, 0)
        generated = max(
            self._latest_predicted_total - self._session_base_predicted, 0
        )
        active_prompt_s = max(self._active_prompt_s - self._session_base_active_pp, 0.0)
        active_generated_s = max(
            self._active_predicted_s - self._session_base_active_tg, 0.0
        )
        return {
            "tokens": {
                "total": int(total),
                "task": int(task_total),
                "prompt": int(prompt),
                "generated": int(generated),
                "request_prompt": int(request_prompt),
                "request_generated": int(request_generated),
                "saved_last": int(getattr(self, "_saved_last_total", 0) or 0),
                "saved_total": int(getattr(self, "_saved_token_total", 0) or 0),
            },
            "time_seconds": {
                "active_total": active_prompt_s + active_generated_s,
                "active_prompt": active_prompt_s,
                "active_generated": active_generated_s,
                "current_total": self._cur_prompt_s + self._cur_predicted_s,
                "current_prompt": self._cur_prompt_s,
                "current_generated": self._cur_predicted_s,
            },
        }


def _number(value, default=0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value) -> int:
    return int(max(_number(value), 0))


def _seconds(value) -> float:
    return max(_number(value), 0.0)


def _table_value(mapping: Mapping, key: str) -> int:
    return _int(mapping.get(key, 0))


def format_runtime_stats_markdown(snapshot: Mapping) -> str:
    """Return a Markdown report for a runtime stats snapshot."""

    tokens = snapshot.get("tokens") or {}
    times = snapshot.get("time_seconds") or {}
    model = snapshot.get("model") or {}
    server = snapshot.get("server") or {}
    lines = [
        "# LlamaServer Runtime Stats",
        "",
        f"- Exported: {snapshot.get('exported_at') or '-'}",
        f"- Model: {model.get('id') or '-'}",
        f"- Model path: {model.get('path') or '-'}",
        f"- Server: {server.get('base_url') or '-'}",
        f"- Running: {'yes' if server.get('running') else 'no'}",
        "",
        "## Tokens",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total | {_table_value(tokens, 'total')} |",
        f"| Task | {_table_value(tokens, 'task')} |",
        f"| Prompt | {_table_value(tokens, 'prompt')} |",
        f"| Generated | {_table_value(tokens, 'generated')} |",
        f"| Request prompt | {_table_value(tokens, 'request_prompt')} |",
        f"| Request generated | {_table_value(tokens, 'request_generated')} |",
        f"| Saved last | {_table_value(tokens, 'saved_last')} |",
        f"| Saved total | {_table_value(tokens, 'saved_total')} |",
        "",
        "## Time",
        "",
        "| Metric | Seconds | Formatted |",
        "|---|---:|---:|",
    ]

    for caption, key in (
        ("Active total", "active_total"),
        ("Active prompt", "active_prompt"),
        ("Active generated", "active_generated"),
        ("Current total", "current_total"),
        ("Current prompt", "current_prompt"),
        ("Current generated", "current_generated"),
    ):
        seconds = _seconds(times.get(key))
        lines.append(f"| {caption} | {seconds:.3f} | {format_duration(seconds)} |")

    return "\n".join(lines) + "\n"
