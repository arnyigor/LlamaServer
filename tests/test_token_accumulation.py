"""Тесты накопления токенов в main.py (счётчик prompt/generation).

Регрессионные тесты на фикс: раньше `_accumulate_slot_tokens` останавливался
навсегда после первого ненулевого /metrics (`_metrics_total_seen`), из-за чего
generation-токены замирали на нуле (llama.cpp обновляет /metrics только по
завершении запроса). Теперь дельты слотов накапливаются всегда, а /metrics
используется только для догоняющей синхронизации вверх.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LlamaGUI
from src.core.metrics_poller import SlotMetrics


def _make_gui() -> LlamaGUI:
    """Создаёт LlamaGUI без тяжёлого __init__ (без UI/QApplication)."""
    gui = LlamaGUI.__new__(LlamaGUI)
    gui._slot_prompt_total = 0
    gui._slot_predicted_total = 0
    gui._slot_token_seen = {}
    gui._metrics_prompt_total = 0
    gui._metrics_predicted_total = 0
    gui._latest_prompt_total = 0
    gui._latest_predicted_total = 0
    gui._latest_token_total = 0
    gui._token_baseline_total = 0
    gui._saved_token_total = 0
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


if __name__ == "__main__":
    unittest.main()
