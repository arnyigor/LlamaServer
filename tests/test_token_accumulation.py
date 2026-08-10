"""Тесты накопления токенов и активного времени в main.py.

Регрессионные тесты на фикс: раньше `_accumulate_slot_tokens` останавливался
навсегда после первого ненулевого /metrics (`_metrics_total_seen`), из-за чего
generation-токены замирали на нуле (llama.cpp обновляет /metrics только по
завершении запроса). Теперь дельты слотов накапливаются всегда, а /metrics
используется только для догоняющей синхронизации вверх.

Активное время модели: total — живая сумма интервалов /slots за запуск
сервера; current — время текущего/последнего запроса, точное значение
которого приходит из логов llama_print_timings. ВАЖНО: /metrics
(*tokens_seconds) — это throughput (токены/сек), а не секунды, поэтому для
времени он не используется.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LlamaGUI
from src.core.metrics_poller import SlotMetrics


def _make_gui() -> LlamaGUI:
    """Создаёт LlamaGUI без тяжёлого __init__ (без UI/QApplication)."""
    gui = LlamaGUI.__new__(LlamaGUI)
    gui._slot_prompt_total = 0
    gui._slot_predicted_total = 0
    gui._slot_token_seen = {}
    gui._last_slots = []
    gui._request_token_base = {}
    gui._metrics_prompt_total = 0
    gui._metrics_predicted_total = 0
    gui._latest_prompt_total = 0
    gui._latest_predicted_total = 0
    gui._latest_token_total = 0
    gui._token_baseline_total = 0
    gui._saved_token_total = 0
    gui._session_base_prompt = 0
    gui._session_base_predicted = 0
    gui._session_base_total = 0
    gui._session_base_active_pp = 0.0
    gui._session_base_active_tg = 0.0
    gui._active_prompt_s = 0.0
    gui._active_predicted_s = 0.0
    gui._cur_prompt_s = 0.0
    gui._cur_predicted_s = 0.0
    gui._was_processing = False
    gui._last_poll_time = None
    gui.ui = MagicMock()
    gui.log_mgr = MagicMock()
    gui.log_mgr.has_speed = False
    return gui


def _slot(slot_id=0, prompt_processed=0, decoded=0, processing=False) -> SlotMetrics:
    slot = SlotMetrics()
    slot.id = slot_id
    slot.n_prompt_tokens_processed = prompt_processed
    slot.n_decoded = decoded
    slot.is_processing = processing
    return slot


def _metrics(prompt_total: int, predicted_total: int) -> MagicMock:
    m = MagicMock()
    m.prompt_tokens_total = prompt_total
    m.tokens_predicted_total = predicted_total
    m.prompt_tokens_seconds = 0.0
    m.predicted_tokens_seconds = 0.0
    return m


class TestTokenAccumulation(unittest.TestCase):
    def test_accumulates_deltas_across_polls(self):
        """TG-токены накапливаются по дельтам n_decoded между опросами."""
        gui = _make_gui()
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=0)])
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=50)])
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=120)])
        self.assertEqual(gui._latest_prompt_total, 100)
        self.assertEqual(gui._latest_predicted_total, 120)
        self.assertEqual(gui._latest_token_total, 220)

    def test_metrics_does_not_freeze_slot_accumulation(self):
        """После получения /metrics накопление по слотам продолжается."""
        gui = _make_gui()
        # Первый запрос накоплен слотами
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=50)])
        # /metrics знает только про завершённые запросы
        gui._on_server_metrics_updated(_metrics(100, 50))
        # Новый запрос стартует: счётчики слота обнулились
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=0, decoded=0)])
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=200, decoded=80)])
        # Сумма: 100+200 prompt, 50+80 predicted — накопление не замерло
        self.assertEqual(gui._latest_prompt_total, 300)
        self.assertEqual(gui._latest_predicted_total, 130)
        self.assertEqual(gui._latest_token_total, 430)

    def test_metrics_catch_up_increases_totals(self):
        """Если /metrics знает больше (опрос подключился позже) — синхронизация вверх."""
        gui = _make_gui()
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=50)])
        gui._on_server_metrics_updated(_metrics(500, 200))
        self.assertEqual(gui._latest_prompt_total, 500)
        self.assertEqual(gui._latest_predicted_total, 200)
        self.assertEqual(gui._latest_token_total, 700)

    def test_metrics_catch_up_never_decreases(self):
        """/metrics не может уменьшить суммы, уже накопленные слотами."""
        gui = _make_gui()
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=50)])
        gui._on_server_metrics_updated(_metrics(30, 10))
        self.assertEqual(gui._latest_prompt_total, 100)
        self.assertEqual(gui._latest_predicted_total, 50)

    def test_slot_reset_counts_only_new_request(self):
        """Сброс счётчиков слота на новом запросе не задваивает токены."""
        gui = _make_gui()
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=100, decoded=50)])
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=0, decoded=0)])  # reset
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=30, decoded=5)])
        self.assertEqual(gui._latest_prompt_total, 130)
        self.assertEqual(gui._latest_predicted_total, 55)

    def test_fast_request_between_polls_is_counted(self):
        """Запрос, целиком уложившийся между опросами, не теряется:
        idle-слот сохраняет финальные n_decoded/n_prompt_tokens_processed."""
        gui = _make_gui()
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=0, decoded=0)])
        # Между опросами прошёл целый запрос, слот снова idle
        gui._accumulate_slot_tokens([_slot(0, prompt_processed=150, decoded=40)])
        self.assertEqual(gui._latest_prompt_total, 150)
        self.assertEqual(gui._latest_predicted_total, 40)

    def test_request_tokens_label_from_visible_slots(self):
        """Request label использует n_prompt_tokens и n_decoded слотов."""
        gui = _make_gui()
        slot = _slot(0, prompt_processed=150, decoded=40, processing=True)
        slot.n_prompt_tokens = 150
        gui._on_slot_metrics_updated([slot])
        text = gui.ui.request_tokens_label.setText.call_args[0][0]
        # Rich-text HTML: подписи и значения в отдельных span
        self.assertIn("prompt", text)
        self.assertIn("150", text)
        self.assertIn("generated", text)
        self.assertIn("40", text)

    def test_fmt_counter_plain_no_separator(self):
        """_fmt_counter не ставит запятые-разделители тысяч (сбивают с толку)."""
        gui = _make_gui()
        self.assertEqual(gui._fmt_counter(1234567), "1234567")
        self.assertEqual(gui._fmt_counter(0), "0")

    def test_format_speed_thousands_vs_fractions(self):
        """format_speed: большие значения — 1 знак после запятой, малые — 2."""
        from src.core.constants import format_speed

        self.assertEqual(format_speed(1234.56), "1234.6")
        self.assertEqual(format_speed(25.38), "25.38")
        self.assertEqual(format_speed(0), "0.00")

    def test_speed_label_from_slots_without_log_timing(self):
        """Пока нет замера из логов, /slots показывает живую скорость."""
        gui = _make_gui()
        slot = _slot(0, prompt_processed=0, decoded=0, processing=True)
        slot.prompt_per_second = 1234.56
        gui._on_slot_metrics_updated([slot])
        text = gui.ui.speed_label.setText.call_args[0][0]
        self.assertIn("1234.6", text)
        self.assertIn("tok/s", text)

    def test_log_speed_wins_over_slots(self):
        """Когда лог-менеджер извлёк скорость из логов, /slots не перетирает её."""
        gui = _make_gui()
        gui.log_mgr.has_speed = True
        slot = _slot(0, prompt_processed=0, decoded=0, processing=True)
        slot.prompt_per_second = 1234.56
        gui.ui.speed_label.setText.reset_mock()
        gui._on_slot_metrics_updated([slot])
        gui.ui.speed_label.setText.assert_not_called()

    def test_log_speed_updated_always_sets_label(self):
        """_on_log_speed_updated обновляет лейбл даже при работающем поллере."""
        gui = _make_gui()
        gui.metrics = MagicMock()
        gui.metrics._is_running = True
        gui._on_log_speed_updated("Speed: PP 1234.6 tok/s | TG 25.38 tok/s")
        text = gui.ui.speed_label.setText.call_args[0][0]
        self.assertEqual(text, "Speed: PP 1234.6 tok/s | TG 25.38 tok/s")

    def test_reset_task_tokens_saves_task(self):
        """reset_task_tokens сохраняет накопленное и сбрасывает baseline."""
        gui = _make_gui()
        gui._latest_token_total = 1000
        gui.reset_task_tokens()
        self.assertEqual(gui._token_baseline_total, 1000)
        self.assertEqual(gui._saved_token_total, 1000)
        gui._latest_token_total = 1500
        gui.reset_task_tokens()
        self.assertEqual(gui._saved_token_total, 1500)

    def test_reset_task_resets_current_time_and_request(self):
        """Reset task обнуляет Current time и Request, но не трогает Active."""
        gui = _make_gui()
        gui._latest_token_total = 1000
        gui._active_prompt_s = 10.0
        gui._active_predicted_s = 20.0
        gui._cur_prompt_s = 5.0
        gui._cur_predicted_s = 3.0
        gui.reset_task_tokens()
        self.assertAlmostEqual(gui._cur_prompt_s, 0.0)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.0)
        self.assertAlmostEqual(gui._active_prompt_s, 10.0)  # не трогаем
        self.assertAlmostEqual(gui._active_predicted_s, 20.0)
        text = gui.ui.request_tokens_label.setText.call_args[0][0]
        self.assertEqual(text, "Request: -")
        gui.log_mgr.reset_runtime_extractors.assert_called_with(
            reset_speed=False, reset_timing=True
        )

    def test_reset_task_keeps_stale_idle_request_hidden(self):
        gui = _make_gui()
        slot = _slot(0, prompt_processed=120, decoded=40, processing=False)
        slot.n_prompt_tokens = 120
        gui._on_slot_metrics_updated([slot])
        gui.reset_task_tokens()
        gui.ui.request_tokens_label.setText.reset_mock()
        gui._on_slot_metrics_updated([slot])
        text = gui.ui.request_tokens_label.setText.call_args[0][0]
        self.assertEqual(text, "Request: -")

    def test_reset_session_zeroes_display_and_sticks(self):
        """Reset session обнуляет отображение total/task/времени; новые токены
        не тянут старые значения (baseline-смещения защищают от /metrics)."""
        gui = _make_gui()
        gui._latest_prompt_total = 500
        gui._latest_predicted_total = 200
        gui._latest_token_total = 700
        gui._active_prompt_s = 10.0
        gui._active_predicted_s = 20.0
        gui.reset_session()
        gui.log_mgr.reset_runtime_extractors.assert_called_with(
            reset_speed=True, reset_timing=True
        )
        text = gui.ui.tokens_label.setText.call_args[0][0]
        self.assertNotIn("500", text)
        self.assertNotIn("700", text)
        self.assertNotIn("200", text)
        # Active = 0 после сброса
        text = gui.ui.active_time_label.setText.call_args[0][0]
        self.assertIn("0:00", text)
        # Новые токены: отображается только дельта (740-700=40, 520-500=20)
        gui._latest_prompt_total = 520
        gui._latest_predicted_total = 220
        gui._latest_token_total = 740
        gui._refresh_token_label()
        text = gui.ui.tokens_label.setText.call_args[0][0]
        self.assertIn("40", text)
        self.assertNotIn("700", text)
        # Активное время после сброса: отображается дельта
        gui._active_prompt_s = 135.0
        gui._active_predicted_s = 145.0
        gui._refresh_active_time_label()
        text = gui.ui.active_time_label.setText.call_args[0][0]
        self.assertIn("2:05", text)  # 125 с

    def test_reset_session_keeps_saved_history(self):
        """Reset session не трогает накопленную Saved-историю."""
        gui = _make_gui()
        gui._latest_token_total = 1000
        gui._saved_token_total = 500
        gui.reset_task_tokens()
        self.assertEqual(gui._saved_token_total, 1500)
        gui.reset_session()
        self.assertEqual(gui._saved_token_total, 1500)

    def test_reset_saved_total_zeroes_history(self):
        """reset_saved_total обнуляет Saved-историю (last и total)."""
        gui = _make_gui()
        gui._saved_token_total = 1234
        gui.reset_saved_total()
        self.assertEqual(gui._saved_token_total, 0)
        text = gui.ui.tokens_saved_label.setText.call_args[0][0]
        self.assertNotIn("1234", text)

    # ------------------------------------------------------------------
    # Активное время работы модели (PP/TG): total + current
    # ------------------------------------------------------------------

    def test_format_duration(self):
        """format_duration: H:MM:SS / M:SS."""
        from src.core.constants import format_duration

        self.assertEqual(format_duration(0), "0:00")
        self.assertEqual(format_duration(59), "0:59")
        self.assertEqual(format_duration(125), "2:05")
        self.assertEqual(format_duration(7235), "2:00:35")
        self.assertEqual(format_duration(3600), "1:00:00")
        self.assertEqual(format_duration(-5), "0:00")

    def test_active_time_ticks_while_processing(self):
        """Интервалы опросов, пока слот обрабатывает, идут и в total, и в current."""
        gui = _make_gui()
        slot = _slot(0, decoded=0, processing=True)
        slot.predicted_per_second = 10.0
        with patch("main.time.monotonic", side_effect=[100.0, 100.5, 101.0, 101.5]):
            gui._on_slot_metrics_updated([slot])  # старт запроса: current сброшен
            gui._on_slot_metrics_updated([slot])  # dt=0.5 → TG
            gui._on_slot_metrics_updated([slot])  # dt=0.5 → TG
        self.assertAlmostEqual(gui._active_predicted_s, 1.0)  # total
        self.assertAlmostEqual(gui._cur_predicted_s, 1.0)  # current
        self.assertAlmostEqual(gui._active_prompt_s, 0.0)
        self.assertAlmostEqual(gui._cur_prompt_s, 0.0)

    def test_active_time_idle_not_counted(self):
        """Простой (idle-слоты) не увеличивает активное время."""
        gui = _make_gui()
        slot = _slot(0, decoded=0, processing=False)
        with patch("main.time.monotonic", side_effect=[100.0, 100.5, 101.0]):
            gui._on_slot_metrics_updated([slot])
            gui._on_slot_metrics_updated([slot])
        self.assertAlmostEqual(gui._active_predicted_s, 0.0)
        self.assertAlmostEqual(gui._active_prompt_s, 0.0)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.0)
        self.assertAlmostEqual(gui._cur_prompt_s, 0.0)

    def test_active_time_split_prompt_and_generation(self):
        """Одновременные PP и TG распределяются пропорционально скоростям."""
        gui = _make_gui()
        slot = _slot(0, decoded=0, processing=True)
        slot.prompt_per_second = 100.0
        slot.predicted_per_second = 300.0
        with patch("main.time.monotonic", side_effect=[100.0, 100.4, 100.8]):
            gui._on_slot_metrics_updated([slot])
            gui._on_slot_metrics_updated([slot])  # dt=0.4 → PP 0.1, TG 0.3
        self.assertAlmostEqual(gui._active_prompt_s, 0.1)
        self.assertAlmostEqual(gui._active_predicted_s, 0.3)
        self.assertAlmostEqual(gui._cur_prompt_s, 0.1)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.3)

    def test_active_time_large_gap_not_counted(self):
        """Зазор опроса больше MAX_ACTIVE_TIME_DT — пауза, время не считаем."""
        gui = _make_gui()
        slot = _slot(0, decoded=0, processing=True)
        slot.predicted_per_second = 10.0
        with patch("main.time.monotonic", side_effect=[100.0, 100.25, 110.0]):
            gui._on_slot_metrics_updated([slot])  # 100.0, старт запроса
            gui._on_slot_metrics_updated([slot])  # dt=0.25 → TG
            gui._on_slot_metrics_updated([slot])  # dt=9.75 > 5 → не считаем
        self.assertAlmostEqual(gui._active_predicted_s, 0.25)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.25)

    def test_current_time_reset_on_new_request(self):
        """Переход idle → processing сбрасывает current (новый запрос)."""
        gui = _make_gui()
        gui._cur_prompt_s = 5.0
        gui._cur_predicted_s = 3.0
        gui._was_processing = False
        slot = _slot(0, decoded=0, processing=True)
        slot.predicted_per_second = 10.0
        with patch("main.time.monotonic", side_effect=[100.0, 100.5]):
            gui._on_slot_metrics_updated([slot])
        self.assertAlmostEqual(gui._cur_prompt_s, 0.0)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.0)

    def test_current_time_keeps_accumulating_on_same_request(self):
        """Пока слот processing, current не сбрасывается повторно."""
        gui = _make_gui()
        slot = _slot(0, decoded=0, processing=True)
        slot.predicted_per_second = 10.0
        with patch("main.time.monotonic", side_effect=[100.0, 100.5, 101.0]):
            gui._on_slot_metrics_updated([slot])
            gui._on_slot_metrics_updated([slot])  # dt=0.5
        # current копился, не сброшен (processing всё время True)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.5)

    def test_active_time_log_timing_sets_current(self):
        """llama_print_timings из логов даёт точное значение current."""
        gui = _make_gui()
        gui._on_log_timing_updated(6.77, 1.30)
        self.assertAlmostEqual(gui._cur_prompt_s, 6.77)
        self.assertAlmostEqual(gui._cur_predicted_s, 1.30)
        # total не затронут логами (остаётся живой суммой)
        self.assertAlmostEqual(gui._active_prompt_s, 0.0)
        self.assertAlmostEqual(gui._active_predicted_s, 0.0)

    def test_total_accumulates_across_requests(self):
        """total копится между запросами, current показывает последний."""
        gui = _make_gui()
        s1 = _slot(0, decoded=0, processing=True)
        s1.predicted_per_second = 10.0
        idle = _slot(0, decoded=0, processing=False)
        s2 = _slot(0, decoded=0, processing=True)
        s2.predicted_per_second = 10.0
        with patch(
            "main.time.monotonic",
            side_effect=[100.0, 100.5, 101.0, 101.5, 102.0],
        ):
            gui._on_slot_metrics_updated([s1])  # 100.0: старт запроса 1
            gui._on_slot_metrics_updated([s1])  # 100.5: total 0.5, cur 0.5
            gui._on_slot_metrics_updated([idle])  # 101.0: простой
            gui._on_slot_metrics_updated([s2])  # 101.5: старт запроса 2 → cur сброс
            gui._on_slot_metrics_updated([s2])  # 102.0: total 1.5, cur 1.0
        # total = 0.5 (запрос 1) + 0.5 + 0.5 (запрос 2, включая интервал
        # после idle-опроса, т.к. точный момент старта неизвестен)
        self.assertAlmostEqual(gui._active_predicted_s, 1.5)
        # current = время запроса 2 (после сброса) = 0.5 + 0.5
        self.assertAlmostEqual(gui._cur_predicted_s, 1.0)

    def test_active_time_metrics_seconds_not_used(self):
        """/metrics prompt_tokens_seconds — throughput (токены/сек), а не
        секунды: он не должен затирать живое активное время."""
        gui = _make_gui()
        gui._active_prompt_s = 6.0
        m = _metrics(100, 50)
        m.prompt_tokens_seconds = 1154.77  # фактически токены/сек PP
        m.predicted_tokens_seconds = 41.5  # фактически токены/сек TG
        gui._on_server_metrics_updated(m)
        self.assertAlmostEqual(gui._active_prompt_s, 6.0)
        self.assertAlmostEqual(gui._active_predicted_s, 0.0)

    def test_log_timing_patterns(self):
        """Паттерны извлекают время PP/TG из llama_print_timings (мс)."""
        from src.ui.log_manager import _PP_TIME_PATTERN, _TG_TIME_PATTERN

        line_pp = (
            "0.34.820.765 I slot print_timing: id  0 | task 97 | "
            "prompt eval time =    6770.15 ms /  7818 tokens"
        )
        line_tg = (
            "0.34.820.770 I slot print_timing: id  0 | task 97 | "
            "eval time =    1301.15 ms /    54 tokens"
        )
        m = _PP_TIME_PATTERN.search(line_pp)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "6770.15")
        m = _TG_TIME_PATTERN.search(line_tg)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "1301.15")
        # "prompt eval time" не должен матчиться как TG
        self.assertIsNone(_TG_TIME_PATTERN.search(line_pp))
        self.assertIsNone(_PP_TIME_PATTERN.search(line_tg))

    def test_active_time_label_format(self):
        """Лейбл Active (total): Active: total (PP pp | TG tg)."""
        gui = _make_gui()
        gui._active_prompt_s = 125
        gui._active_predicted_s = 7235
        gui._refresh_active_time_label()
        text = gui.ui.active_time_label.setText.call_args[0][0]
        self.assertIn("Active", text)
        self.assertIn("2:00:35", text)  # TG
        self.assertIn("2:05", text)  # PP

    def test_current_time_label_format(self):
        """Лейбл Current: Current: total (PP pp | TG tg)."""
        gui = _make_gui()
        gui._cur_prompt_s = 6.77
        gui._cur_predicted_s = 1.30
        gui._refresh_current_time_label()
        text = gui.ui.current_time_label.setText.call_args[0][0]
        self.assertIn("Current", text)
        self.assertIn("0:08", text)  # total = 6.77+1.30 = 8.07 → 0:08
        self.assertIn("0:07", text)  # PP 6.77 → 0:07
        self.assertIn("0:01", text)  # TG 1.30 → 0:01

    def test_start_metrics_polling_resets_active_time(self):
        """Каждый старт сервера сбрасывает total и current время."""
        gui = _make_gui()
        gui.metrics = MagicMock()
        gui._active_prompt_s = 10.0
        gui._active_predicted_s = 20.0
        gui._cur_prompt_s = 5.0
        gui._cur_predicted_s = 3.0
        gui._was_processing = True
        gui._last_poll_time = 5.0
        gui._start_metrics_polling()
        self.assertAlmostEqual(gui._active_prompt_s, 0.0)
        self.assertAlmostEqual(gui._active_predicted_s, 0.0)
        self.assertAlmostEqual(gui._cur_prompt_s, 0.0)
        self.assertAlmostEqual(gui._cur_predicted_s, 0.0)
        self.assertFalse(gui._was_processing)
        self.assertIsNone(gui._last_poll_time)


if __name__ == "__main__":
    unittest.main()
