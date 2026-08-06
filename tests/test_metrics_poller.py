"""Тесты для src/core/metrics_poller.py"""

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.metrics_poller import MetricsPoller


def _make_poller() -> MetricsPoller:
    """Создаёт poller без реального QObject/QTimer и фонового воркера."""
    poller = MetricsPoller.__new__(MetricsPoller)
    poller._slot_rate_state = {}
    poller._is_running = False
    poller._worker = None
    poller.base_url = "http://127.0.0.1:8080"
    poller._poll_interval_ms = 250
    poller._metrics_interval_ms = 2000
    return poller


_REAL_SLOT = {
    "id": 0,
    "id_task": 135,
    "n_ctx": 65536,
    "speculative": False,
    "is_processing": True,
    "n_prompt_tokens": 123,
    "n_prompt_tokens_processed": 123,
    "n_prompt_tokens_cache": 0,
    "params": {},
    "prompt": "Hello",
    "generated": " World",
    # Реальный llama-server (server_slot::to_json) отдаёт next_token списком.
    "next_token": [
        {
            "has_next_token": True,
            "has_new_line": False,
            "n_remain": -1,
            "n_decoded": 42,
        }
    ],
}

# Сборки, где next_token приходит объектом (старые/иные версии llama.cpp).
_NEXT_TOKEN_DICT = {
    "has_next_token": True,
    "has_new_line": False,
    "n_remain": -1,
    "n_decoded": 42,
}


class TestParseSlots(unittest.TestCase):
    def test_real_slots_format(self):
        poller = _make_poller()
        slots = poller._parse_slots([dict(_REAL_SLOT)])
        self.assertEqual(len(slots), 1)
        slot = slots[0]
        self.assertEqual(slot.id, 0)
        self.assertTrue(slot.is_processing)
        self.assertEqual(slot.n_ctx, 65536)
        self.assertEqual(slot.n_prompt_tokens, 123)
        self.assertEqual(slot.n_prompt_tokens_processed, 123)
        self.assertEqual(slot.n_prompt_tokens_cache, 0)
        self.assertEqual(slot.n_decoded, 42)
        self.assertEqual(slot.n_remain, -1)
        self.assertTrue(slot.has_next_token)

    def test_next_token_as_dict_compat(self):
        """Сборки со next_token-объектом (не списком) тоже парсятся."""
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        data["next_token"] = dict(_NEXT_TOKEN_DICT)
        slots = poller._parse_slots([data])
        self.assertEqual(slots[0].n_decoded, 42)
        self.assertEqual(slots[0].n_remain, -1)
        self.assertTrue(slots[0].has_next_token)

    def test_next_token_missing_or_empty(self):
        """Пустой/отсутствующий next_token даёт нули без падения."""
        poller = _make_poller()
        for bad in (None, [], {}, ""):
            data = dict(_REAL_SLOT)
            data["next_token"] = bad
            slots = poller._parse_slots([data])
            self.assertEqual(slots[0].n_decoded, 0, f"bad={bad!r}")
            self.assertFalse(slots[0].has_next_token, f"bad={bad!r}")

    def test_missing_fields_default_to_zero(self):
        """Слот без новых полей (например, старый сервер) не падает."""
        poller = _make_poller()
        slots = poller._parse_slots([{"id": 0, "is_processing": False}])
        self.assertEqual(slots[0].n_decoded, 0)
        self.assertEqual(slots[0].n_prompt_tokens, 0)
        self.assertEqual(slots[0].n_prompt_tokens_processed, 0)
        self.assertEqual(slots[0].prompt_per_second, 0.0)
        self.assertEqual(slots[0].predicted_per_second, 0.0)

    def test_non_dict_slots_skipped(self):
        poller = _make_poller()
        slots = poller._parse_slots([{"id": 0}, "not-a-dict", 42, None])
        self.assertEqual(len(slots), 1)


class TestDeltaSpeed(unittest.TestCase):
    def test_tg_delta_speed(self):
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        with patch("src.core.metrics_poller.time.monotonic", return_value=1000.0):
            poller._parse_slots([data])

        data["next_token"] = [dict(_REAL_SLOT["next_token"][0])]
        data["next_token"][0]["n_decoded"] = 52
        with patch("src.core.metrics_poller.time.monotonic", return_value=1001.0):
            slots = poller._parse_slots([data])

        self.assertAlmostEqual(slots[0].predicted_per_second, 10.0)
        self.assertEqual(slots[0].prompt_per_second, 0.0)

    def test_worker_timestamp_used_for_speed(self):
        """Метка времени опроса из воркера даёт корректную скорость,
        даже если парсинг произошёл позже реального опроса."""
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        # Парсинг первого опроса произошёл на 5с позже реального времени
        # (очередь сигналов), но метка времени воркера — точная.
        poller._parse_slots([data], now=1000.0)

        data["next_token"] = [dict(_REAL_SLOT["next_token"][0])]
        data["next_token"][0]["n_decoded"] = 52
        slots = poller._parse_slots([data], now=1001.0)

        self.assertAlmostEqual(slots[0].predicted_per_second, 10.0)

    def test_pp_delta_speed(self):
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        data["n_prompt_tokens_processed"] = 100
        data["next_token"] = [dict(_REAL_SLOT["next_token"][0])]
        data["next_token"][0]["n_decoded"] = 0
        with patch("src.core.metrics_poller.time.monotonic", return_value=1000.0):
            poller._parse_slots([data])

        data["n_prompt_tokens_processed"] = 150
        with patch("src.core.metrics_poller.time.monotonic", return_value=1002.0):
            slots = poller._parse_slots([data])

        self.assertAlmostEqual(slots[0].prompt_per_second, 25.0)
        self.assertEqual(slots[0].predicted_per_second, 0.0)

    def test_counter_reset_produces_no_negative_speed(self):
        """Сброс счётчиков на новом запросе не даёт отрицательной скорости."""
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        data["next_token"] = [dict(_REAL_SLOT["next_token"][0])]
        with patch("src.core.metrics_poller.time.monotonic", return_value=1000.0):
            poller._parse_slots([data])

        # новый запрос: счётчики обнулены
        data["is_processing"] = True
        data["n_prompt_tokens_processed"] = 5
        data["next_token"] = {"has_next_token": True, "n_decoded": 3, "n_remain": 10}
        with patch("src.core.metrics_poller.time.monotonic", return_value=1001.0):
            slots = poller._parse_slots([data])

        self.assertEqual(slots[0].prompt_per_second, 0.0)
        self.assertEqual(slots[0].predicted_per_second, 0.0)

    def test_idle_slot_resets_rate(self):
        """Переход is_processing=False сбрасывает скорость для слота."""
        poller = _make_poller()
        data = dict(_REAL_SLOT)
        data["next_token"] = [dict(_REAL_SLOT["next_token"][0])]
        with patch("src.core.metrics_poller.time.monotonic", return_value=1000.0):
            poller._parse_slots([data])

        data["is_processing"] = False
        with patch("src.core.metrics_poller.time.monotonic", return_value=1001.0):
            slots = poller._parse_slots([data])
        self.assertEqual(slots[0].predicted_per_second, 0.0)

    def test_slot_removed_prunes_state(self):
        poller = _make_poller()
        with patch("src.core.metrics_poller.time.monotonic", return_value=1000.0):
            poller._parse_slots([dict(_REAL_SLOT), dict(_REAL_SLOT, id=1)])
        with patch("src.core.metrics_poller.time.monotonic", return_value=1001.0):
            poller._parse_slots([dict(_REAL_SLOT)])
        self.assertEqual(len(poller._slot_rate_state), 1)


class TestParseMetrics(unittest.TestCase):
    def test_real_metrics_text(self):
        poller = _make_poller()
        text = (
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 1234\n"
            "llamacpp:prompt_seconds_total 456\n"
            "llamacpp:tokens_predicted_total 5678\n"
            "llamacpp:tokens_predicted_seconds_total 789\n"
            "llamacpp:n_decode_total 12\n"
            "llamacpp:prompt_tokens_seconds 800.5\n"
            "llamacpp:predicted_tokens_seconds 120.25\n"
            "llamacpp:requests_processing 2\n"
            "llamacpp:requests_deferred 1\n"
        )
        m = poller._parse_metrics(text)
        self.assertEqual(m.prompt_tokens_total, 1234)
        self.assertEqual(m.tokens_predicted_total, 5678)
        self.assertAlmostEqual(m.prompt_tokens_seconds, 800.5)
        self.assertAlmostEqual(m.predicted_tokens_seconds, 120.25)
        self.assertEqual(m.requests_processing, 2)
        self.assertEqual(m.requests_deferred, 1)

    def test_empty_metrics(self):
        poller = _make_poller()
        m = poller._parse_metrics("")
        self.assertEqual(m.prompt_tokens_total, 0)
        self.assertEqual(m.tokens_predicted_total, 0)


class TestStateReset(unittest.TestCase):
    def test_start_resets_rate_state(self):
        poller = _make_poller()
        poller._slot_rate_state = {0: {"time": 1.0, "processing": True}}
        poller._is_running = False
        with patch.object(MetricsPoller, "_start_worker") as mock_start:
            poller.start()

        self.assertEqual(poller._slot_rate_state, {})
        self.assertTrue(poller._is_running)
        mock_start.assert_called_once()

    def test_start_ignores_second_call_while_running(self):
        poller = _make_poller()
        poller._is_running = True
        with patch.object(MetricsPoller, "_start_worker") as mock_start:
            poller.start()
        mock_start.assert_not_called()

    def test_stop_sets_running_false_and_stops_worker(self):
        poller = _make_poller()
        poller._is_running = True
        fake_worker = MagicMock()
        poller._worker = fake_worker
        poller.stop()
        self.assertFalse(poller._is_running)
        fake_worker.stop.assert_called_once()
        fake_worker.wait.assert_called_once()

    def test_set_url_resets_rate_state(self):
        poller = _make_poller()
        poller._slot_rate_state = {0: {"time": 1.0, "processing": True}}
        poller.set_url("http://127.0.0.1:8080")
        self.assertEqual(poller._slot_rate_state, {})

    def test_set_url_propagates_to_worker(self):
        poller = _make_poller()
        fake_worker = MagicMock()
        poller._worker = fake_worker
        poller.set_url("http://127.0.0.1:9090/")
        self.assertEqual(poller.base_url, "http://127.0.0.1:9090")
        fake_worker.set_base_url.assert_called_once_with("http://127.0.0.1:9090")


class TestSignalHandlers(unittest.TestCase):
    def test_on_slots_fetched_parses_and_emits(self):
        poller = _make_poller()
        poller._is_running = True
        poller._worker = MagicMock()
        poller.slot_metrics_updated = MagicMock()
        poller.server_metrics_updated = MagicMock()

        poller._on_slots_fetched([dict(_REAL_SLOT)], 1000.0)

        slots = poller.slot_metrics_updated.emit.call_args[0][0]
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0].n_decoded, 42)
        poller.server_metrics_updated.emit.assert_not_called()

    def test_on_metrics_fetched_parses_and_emits(self):
        poller = _make_poller()
        poller._is_running = True
        poller._worker = MagicMock()
        poller.slot_metrics_updated = MagicMock()
        poller.server_metrics_updated = MagicMock()

        text = (
            "llamacpp:prompt_tokens_total 1234\nllamacpp:tokens_predicted_total 5678\n"
        )
        poller._on_metrics_fetched(text, 1000.0)

        metrics = poller.server_metrics_updated.emit.call_args[0][0]
        self.assertEqual(metrics.prompt_tokens_total, 1234)
        self.assertEqual(metrics.tokens_predicted_total, 5678)
        poller.slot_metrics_updated.emit.assert_not_called()

    def test_handlers_ignore_data_after_stop(self):
        poller = _make_poller()
        poller._is_running = False
        poller._worker = None
        poller.slot_metrics_updated = MagicMock()
        poller.server_metrics_updated = MagicMock()

        poller._on_slots_fetched([dict(_REAL_SLOT)], 1000.0)
        poller._on_metrics_fetched("llamacpp:prompt_tokens_total 1\n", 1000.0)

        poller.slot_metrics_updated.emit.assert_not_called()
        poller.server_metrics_updated.emit.assert_not_called()


class TestMetricsFetchWorker(unittest.TestCase):
    def test_worker_interval_defaults(self):
        from src.core.metrics_poller import _MetricsFetchWorker

        worker = _MetricsFetchWorker("http://127.0.0.1:8080/", 250, 2000)
        self.assertEqual(worker._base_url, "http://127.0.0.1:8080")
        self.assertAlmostEqual(worker._slots_interval_s, 0.25)
        self.assertAlmostEqual(worker._metrics_interval_s, 2.0)
        # Отрицательные/нулевые интервалы не дают busy-loop
        worker2 = _MetricsFetchWorker("http://x", 0, 0)
        self.assertAlmostEqual(worker2._slots_interval_s, 0.05)
        self.assertAlmostEqual(worker2._metrics_interval_s, 0.5)

    def test_worker_fetch_returns_none_on_error(self):
        from unittest.mock import patch as _patch

        from src.core.metrics_poller import _MetricsFetchWorker

        worker = _MetricsFetchWorker("http://127.0.0.1:1", 250, 2000)
        with _patch("src.core.metrics_poller.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("connection refused")
            self.assertIsNone(worker._fetch_json("/slots"))
            self.assertIsNone(worker._fetch_text("/metrics"))


if __name__ == "__main__":
    unittest.main()
