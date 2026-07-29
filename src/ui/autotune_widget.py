"""Вкладка AutoTune benchmark."""

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.benchmark_models import AutoTunePlan, BenchmarkCandidate, BenchmarkResult


_COLUMNS = [
    "#",
    "Status",
    "Score",
    "%Best",
    "ΔTG",
    "Prompt tok/s",
    "Gen tok/s",
    "Load sec",
    "VRAM",
    "RAM",
    "ngl",
    "ncmoe",
    "ctk",
    "ctv",
    "batch",
    "ubatch",
    "threads",
    "threads_batch",
    "np",
    "flash_attn",
    "mmproj",
    "ctx_checkpoints",
    "cache_ram",
    "error",
]

_PARAM_COLUMNS = {
    10: "ngl",
    11: "ncmoe",
    12: "cache_type_k",
    13: "cache_type_v",
    14: "batch_size",
    15: "ubatch_size",
    16: "threads",
    17: "threads_batch",
    18: "parallel_slots",
    19: "flash_attn",
    20: "use_mmproj",
    21: "ctx_checkpoints",
    22: "cache_ram",
}

_INT_PARAM_KEYS = {
    "ngl",
    "ncmoe",
    "batch_size",
    "ubatch_size",
    "threads",
    "threads_batch",
    "parallel_slots",
    "ctx_checkpoints",
    "cache_ram",
}
_QUANT_CHOICES = ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]
_BOOL_CHOICES = ["true", "false"]


class AutoTuneWidget(QWidget):
    build_plan_requested = Signal()
    start_requested = Signal()
    cancel_requested = Signal()
    apply_best_requested = Signal()
    save_best_requested = Signal()
    export_report_requested = Signal()
    open_results_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_by_id: Dict[str, int] = {}
        self._scores: Dict[
            str, float
        ] = {}  # candidate_id -> score для вычисления %Best
        self._gen_tg: Dict[str, float] = {}  # candidate_id -> gen_tok_s для ΔTG
        self._done_runs = 0
        self._total_runs = 0
        self._per_run_timeout_sec = 0
        self._run_started_at = 0.0
        self._current_run_started_at = 0.0
        self._current_run_id = ""
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_time_labels)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        settings_group = QGroupBox("AutoTune settings")
        settings = QVBoxLayout(settings_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Quick", "Normal", "Deep"])
        self.mode_combo.setToolTip(
            "Quick: small staged plan for fast checks.\n"
            "Normal: more KV/batch/ubatch/thread candidates.\n"
            "Deep: wider search, slower but more thorough."
        )
        row1.addWidget(self.mode_combo)

        row1.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(
            ["Balanced", "Max Speed", "Low VRAM", "Quality KV", "MoE Optimized"]
        )
        self.target_combo.setToolTip(
            "Balanced: speed + stability + memory margin.\n"
            "Max Speed: prioritizes tok/s.\n"
            "Low VRAM: favors memory-saving settings.\n"
            "Quality KV: favors higher-quality KV cache.\n"
            "MoE Optimized: explores CPU MoE offload more aggressively."
        )
        row1.addWidget(self.target_combo)

        row1.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["llama-bench", "hybrid", "llama-server"])
        self.engine_combo.setToolTip(
            "llama-bench: current implemented engine, fastest micro-benchmark.\n"
            "hybrid/server: planned modes; currently blocked with explanation."
        )
        row1.addWidget(self.engine_combo)
        settings.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Time budget (min):"))
        self.time_budget = QSpinBox()
        self.time_budget.setRange(1, 240)
        self.time_budget.setValue(15)
        row2.addWidget(self.time_budget)

        row2.addWidget(QLabel("Max runs:"))
        self.max_runs = QSpinBox()
        self.max_runs.setRange(1, 500)
        self.max_runs.setValue(12)
        row2.addWidget(self.max_runs)

        row2.addWidget(QLabel("Repeat top:"))
        self.repeat_top = QSpinBox()
        self.repeat_top.setRange(1, 5)
        self.repeat_top.setValue(1)
        row2.addWidget(self.repeat_top)

        row2.addWidget(QLabel("Per-run timeout (sec):"))
        self.per_run_timeout = QSpinBox()
        self.per_run_timeout.setRange(120, 7200)
        self.per_run_timeout.setValue(300)
        row2.addWidget(self.per_run_timeout)
        settings.addLayout(row2)

        row2b = QHBoxLayout()
        self.early_stop_peak = QCheckBox("Early stop after peak drop")
        self.early_stop_peak.setToolTip(
            "Stop after at least 3 successful runs when a new successful run is slower than the current peak."
        )
        self.verify_server_after_apply = QCheckBox(
            "Start server and verify after Apply Best"
        )
        self.verify_server_after_apply.setToolTip(
            "After Apply Best, start/restart llama-server with the same saved parameters and send one short test request."
        )
        row2b.addWidget(self.early_stop_peak)
        row2b.addWidget(self.verify_server_after_apply)
        row2b.addStretch(1)
        settings.addLayout(row2b)

        row3 = QHBoxLayout()
        self.build_plan_btn = QPushButton("Build Plan")
        self.start_btn = QPushButton("Start AutoTune")
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.apply_best_btn = QPushButton("Apply Best")
        self.apply_best_btn.setEnabled(False)
        self.save_best_btn = QPushButton("Save Best Preset")
        self.save_best_btn.setEnabled(False)
        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.setEnabled(False)
        self.open_results_btn = QPushButton("Open Results Folder")
        self.open_results_btn.setEnabled(False)

        for btn in [
            self.build_plan_btn,
            self.start_btn,
            self.pause_btn,
            self.cancel_btn,
            self.apply_best_btn,
            self.save_best_btn,
            self.export_report_btn,
            self.open_results_btn,
        ]:
            row3.addWidget(btn)
        settings.addLayout(row3)
        layout.addWidget(settings_group)

        self.hint_label = QLabel(
            "Uses current selected model, Context Size and Prompt/Gen benchmark sizes. "
            "After Build Plan you can edit candidate rows before Start: numeric cells are editable, "
            "KV/FA/mmproj use drop-downs. Edits are applied when AutoTune starts."
        )
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self.status_label = QLabel("Build a plan or start Quick AutoTune.")
        layout.addWidget(self.status_label)

        self.progress_summary = QLabel("Progress: idle")
        self.progress_summary.setWordWrap(True)
        self.progress_summary.setStyleSheet(
            "background-color: #263238; color: #E0F7FA; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        layout.addWidget(self.progress_summary)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(24)
        self.current_run_label = QLabel("Idle")
        self.current_run_label.setWordWrap(True)
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.current_run_label, 2)
        layout.addLayout(progress_row)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        # Internal buffer only. Do not add a second console under Benchmark:
        # the main Logs panel already receives the same AutoTune events.
        self.activity_log = QTextEdit(readOnly=True)
        self.activity_log.setVisible(False)

        self.best_text = QTextEdit(readOnly=True)
        self.best_text.setMaximumHeight(150)
        self.best_text.setPlaceholderText("Best result will appear here.")
        layout.addWidget(self.best_text)

        self.build_plan_btn.clicked.connect(self.build_plan_requested.emit)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.apply_best_btn.clicked.connect(self.apply_best_requested.emit)
        self.save_best_btn.clicked.connect(self.save_best_requested.emit)
        self.export_report_btn.clicked.connect(self.export_report_requested.emit)
        self.open_results_btn.clicked.connect(self.open_results_requested.emit)

    def options(self) -> Dict[str, object]:
        return {
            "mode": self.mode_combo.currentText().lower(),
            "target": self.target_combo.currentText().lower().replace(" ", "_"),
            "engine": self.engine_combo.currentText(),
            "time_budget_sec": self.time_budget.value() * 60,
            "max_runs": self.max_runs.value(),
            "repeat_top": self.repeat_top.value(),
            "per_run_timeout_sec": self.per_run_timeout.value(),
            "early_stop_on_peak": self.early_stop_peak.isChecked(),
            "verify_server_after_apply": self.verify_server_after_apply.isChecked(),
        }

    def set_running(self, running: bool) -> None:
        self.build_plan_btn.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.start_btn.setText("AutoTune running..." if running else "Start AutoTune")
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setVisible(running or self.progress_bar.value() > 0)
        if running:
            now = time.monotonic()
            self._run_started_at = now
            self._current_run_started_at = now
            self.progress_bar.setRange(0, max(self._total_runs, 1))
            self.progress_bar.setValue(self._done_runs)
            self.current_run_label.setText("Starting llama-bench process...")
            self.status_label.setText("AutoTune starting...")
            self.append_activity("AutoTune started")
            self._timer.start()
            self._update_time_labels()
        else:
            self._timer.stop()
            self._update_time_labels(finished=True)
        self.apply_best_btn.setEnabled(
            False if running else self.apply_best_btn.isEnabled()
        )
        self.save_best_btn.setEnabled(
            False if running else self.save_best_btn.isEnabled()
        )

    def clear_results(self) -> None:
        self._row_by_id.clear()
        self._scores.clear()
        self._gen_tg.clear()
        self.table.setRowCount(0)
        self.best_text.clear()
        self.activity_log.clear()
        self._done_runs = 0
        self._total_runs = 0
        self._run_started_at = 0.0
        self._current_run_started_at = 0.0
        self._current_run_id = ""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.progress_summary.setText("Progress: idle")
        self.current_run_label.setText("Idle")
        self.apply_best_btn.setEnabled(False)
        self.save_best_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
        self.open_results_btn.setEnabled(False)

    def set_plan(self, plan: AutoTunePlan) -> None:
        self.clear_results()
        self._total_runs = len(plan.candidates)
        self.status_label.setText(
            f"Plan: {len(plan.candidates)} candidates | ctx={plan.ctx_size:,} | mode={plan.mode} | target={plan.target}"
        )
        self.progress_summary.setText(
            f"Ready: {len(plan.candidates)} runs planned. Time budget and per-run timeout are shown above."
        )
        self.table.setRowCount(len(plan.candidates))
        for row, candidate in enumerate(plan.candidates):
            self._row_by_id[candidate.id] = row
            self._fill_candidate_row(row, candidate)

    def _set_item(self, row: int, col: int, value: object, editable: bool = False) -> None:
        item = QTableWidgetItem(str(value))
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter
            if col != len(_COLUMNS) - 1
            else Qt.AlignmentFlag.AlignLeft
        )
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, col, item)

    def _set_combo_cell(self, row: int, col: int, value: object, choices: list[str]) -> None:
        combo = QComboBox()
        combo.addItems(choices)
        text = str(value).strip().lower()
        if isinstance(value, bool):
            text = "true" if value else "false"
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.setToolTip("Editable AutoTune candidate value")
        self.table.setCellWidget(row, col, combo)

    def _cell_text(self, row: int, col: int) -> str:
        widget = self.table.cellWidget(row, col)
        if isinstance(widget, QComboBox):
            return widget.currentText()
        item = self.table.item(row, col)
        return item.text().strip() if item else ""

    def _coerce_param_value(self, key: str, value: str, old_value: object) -> object:
        text = str(value).strip()
        if key in {"flash_attn", "use_mmproj"}:
            return text.lower() in {"1", "true", "yes", "on"}
        if key in _INT_PARAM_KEYS:
            if text.lower() in {"auto", "all"} and key == "ngl":
                return text.lower()
            try:
                return int(text)
            except ValueError:
                return old_value
        return text

    def apply_table_edits_to_plan(self, plan: AutoTunePlan) -> AutoTunePlan:
        """Copies edited table cells back into AutoTunePlan candidates."""
        by_id = {candidate.id: candidate for candidate in plan.candidates}
        changed = 0
        for row in range(self.table.rowCount()):
            cid = self._cell_text(row, 0)
            candidate = by_id.get(cid)
            if not candidate:
                continue
            for col, key in _PARAM_COLUMNS.items():
                old_value = candidate.params.get(key, "")
                new_value = self._coerce_param_value(key, self._cell_text(row, col), old_value)
                if new_value != old_value:
                    candidate.params[key] = new_value
                    changed += 1
        if changed:
            self.append_activity(f"Applied {changed} edited plan value(s)")
        return plan

    def _fill_candidate_row(self, row: int, candidate: BenchmarkCandidate) -> None:
        p = candidate.params
        values = [
            candidate.id,
            "pending",
            "",  # score
            "",  # %Best
            "",  # ΔTG
            "",  # prompt tok/s
            "",  # gen tok/s
            "",  # load sec
            "",  # VRAM
            "",  # RAM
            p.get("ngl", ""),
            p.get("ncmoe", ""),
            p.get("cache_type_k", ""),
            p.get("cache_type_v", ""),
            p.get("batch_size", ""),
            p.get("ubatch_size", ""),
            p.get("threads", ""),
            p.get("threads_batch", ""),
            p.get("parallel_slots", ""),
            p.get("flash_attn", ""),
            p.get("use_mmproj", ""),
            p.get("ctx_checkpoints", ""),
            p.get("cache_ram", ""),
            candidate.reason,
        ]
        for col, value in enumerate(values):
            if col in (12, 13):
                self._set_combo_cell(row, col, value, _QUANT_CHOICES)
            elif col in (19, 20):
                self._set_combo_cell(row, col, value, _BOOL_CHOICES)
            else:
                self._set_item(row, col, value, editable=col in _PARAM_COLUMNS)

    def mark_running(self, candidate: BenchmarkCandidate) -> None:
        row = self._row_by_id.get(candidate.id)
        p = candidate.params
        self._current_run_id = candidate.id
        self._current_run_started_at = time.monotonic()
        self.current_run_label.setText(
            "Current: "
            f"{candidate.id} | KV {p.get('cache_type_k')}/{p.get('cache_type_v')} | "
            f"ngl={p.get('ngl')} | b={p.get('batch_size')} ub={p.get('ubatch_size')} | "
            f"t={p.get('threads')} ncmoe={p.get('ncmoe')}"
        )
        self.append_activity(
            f"START {candidate.id}: ngl={p.get('ngl')}, KV {p.get('cache_type_k')}/{p.get('cache_type_v')}, "
            f"batch={p.get('batch_size')}, ubatch={p.get('ubatch_size')}, threads={p.get('threads')}"
        )
        self._update_time_labels()
        if row is not None:
            self._set_item(row, 1, "running")
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))

    def update_result(self, result: BenchmarkResult) -> None:
        row = self._row_by_id.get(result.candidate_id)
        if row is None:
            return
        # Сохраняем для вычисления %Best и ΔTG
        if result.status == "success":
            self._scores[result.candidate_id] = result.score
            self._gen_tg[result.candidate_id] = result.generation_tok_s
        else:
            self._scores.pop(result.candidate_id, None)
            self._gen_tg.pop(result.candidate_id, None)
        values = {
            1: result.status,
            2: f"{result.score:.3f}",
            5: f"{result.prompt_tok_s:.1f}",
            6: f"{result.generation_tok_s:.1f}",
            7: f"{result.load_time_sec:.1f}",
            8: f"{result.vram_used_mib:.0f}",
            9: f"{result.ram_used_mib:.0f}",
            23: result.error,
        }
        for col, value in values.items():
            self._set_item(row, col, value)
        self._refresh_deltas()
        self.append_activity(
            f"DONE {result.candidate_id}: {result.status}, "
            f"PP={result.prompt_tok_s:.1f}, TG={result.generation_tok_s:.1f}, score={result.score:.3f}"
            + (f", error={result.error}" if result.error else "")
        )

    def _refresh_deltas(self) -> None:
        """Пересчитывает %Best (колонка 3) и ΔTG (колонка 4) для всех строк."""
        if not self._scores:
            return
        max_score = (
            max(v for v in self._scores.values() if v > 0)
            if any(v > 0 for v in self._scores.values())
            else None
        )
        max_tg = (
            max(v for v in self._gen_tg.values() if v > 0)
            if any(v > 0 for v in self._gen_tg.values())
            else None
        )
        for cid, row in self._row_by_id.items():
            if cid in self._scores and max_score and max_score > 0:
                pct = (self._scores[cid] / max_score) * 100.0
                self._set_item(row, 3, f"{pct:.0f}%")
            else:
                self._set_item(row, 3, "")
            if cid in self._gen_tg and max_tg and max_tg > 0:
                delta = self._gen_tg[cid] - max_tg
                self._set_item(row, 4, f"{delta:+.1f}")
            else:
                self._set_item(row, 4, "")

    def show_best(
        self,
        best: Optional[BenchmarkResult],
        params: Dict[str, object],
        output_dir: str,
    ) -> None:
        self._timer.stop()
        self.progress_bar.setRange(0, max(self._total_runs, 1))
        self.progress_bar.setValue(self._done_runs)
        self.progress_bar.setFormat(
            f"{self.progress_bar.value()}/{max(self._total_runs, 1)} runs"
        )
        self.current_run_label.setText(
            f"Finished. Results folder: {output_dir}" if output_dir else "Finished"
        )
        self._update_time_labels(finished=True)
        self.export_report_btn.setEnabled(bool(output_dir))
        self.open_results_btn.setEnabled(bool(output_dir))
        if not best:
            self.best_text.setPlainText(
                f"No successful result. Results folder: {output_dir}"
            )
            self.apply_best_btn.setEnabled(False)
            self.save_best_btn.setEnabled(False)
            return
        self.apply_best_btn.setEnabled(True)
        self.save_best_btn.setEnabled(True)

        # Вычисляем сравнение с baseline (первый успешный кандидат)
        baseline_tg = None
        for cid, row in sorted(self._row_by_id.items(), key=lambda x: x[1]):
            if cid in self._gen_tg and self._gen_tg[cid] > 0:
                baseline_tg = self._gen_tg.get(cid)
                break

        lines = [
            "Best result",
            "━━━━━━━━━━━━━━━━",
            f"Run: {best.candidate_id}",
            f"Score: {best.score:.3f}",
            f"Prompt: {best.prompt_tok_s:.1f} tok/s",
            f"Generation: {best.generation_tok_s:.1f} tok/s",
            f"Load: {best.load_time_sec:.1f} sec",
            f"VRAM: {best.vram_used_mib:.0f} MiB",
            f"RAM: {best.ram_used_mib:.0f} MiB",
            "Stable: yes",
            "",
            "Why selected:",
            "- highest target-specific score among successful candidates",
            "- no detected OOM/crash",
            "- stable llama-bench completion",
        ]
        if baseline_tg and baseline_tg > 0:
            imp = ((best.generation_tok_s / baseline_tg) - 1.0) * 100.0
            lines.append(f"- TG improvement vs baseline: {imp:+.1f}%")
        lines += [
            "",
            "Parameters:",
        ]
        for key, value in params.items():
            lines.append(f"- {key}: {value}")
        lines.append(f"\nResults folder: {output_dir}")
        self.best_text.setPlainText("\n".join(lines))

    def prepare_run(self, total: int, per_run_timeout_sec: int) -> None:
        self._done_runs = 0
        self._total_runs = max(int(total), 1)
        self._per_run_timeout_sec = max(int(per_run_timeout_sec), 0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self._total_runs)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0/{self._total_runs} runs")
        self.progress_summary.setText(
            f"Prepared: 0/{self._total_runs} runs. Per-run timeout: {self._format_duration(self._per_run_timeout_sec)}."
        )

    def append_activity(self, text: str) -> None:
        if not text:
            return
        stamp = time.strftime("%H:%M:%S")
        self.activity_log.append(f"[{stamp}] {text}")

    def _format_duration(self, seconds: float) -> str:
        seconds = max(int(seconds), 0)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {sec:02d}s"
        if minutes:
            return f"{minutes}m {sec:02d}s"
        return f"{sec}s"

    def _update_time_labels(self, finished: bool = False) -> None:
        now = time.monotonic()
        total_elapsed = now - self._run_started_at if self._run_started_at else 0
        current_elapsed = (
            now - self._current_run_started_at if self._current_run_started_at else 0
        )
        remaining = max(self._total_runs - self._done_runs, 0)
        if self._done_runs > 0 and total_elapsed > 0:
            avg = total_elapsed / self._done_runs
            eta = avg * remaining
            eta_text = self._format_duration(eta)
        elif self._per_run_timeout_sec and remaining:
            eta_text = (
                f"up to {self._format_duration(self._per_run_timeout_sec * remaining)}"
            )
        else:
            eta_text = "estimating"

        prefix = "Finished" if finished else "Running"
        self.progress_summary.setText(
            f"{prefix}: {self._done_runs}/{max(self._total_runs, 1)} runs | "
            f"elapsed {self._format_duration(total_elapsed)} | "
            f"current {self._current_run_id or '-'} {self._format_duration(current_elapsed)} | "
            f"ETA {eta_text}"
        )

    def set_progress(self, done: int, total: int) -> None:
        total = max(int(total), 1)
        done = max(0, min(int(done), total))
        self._done_runs = done
        self._total_runs = total
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        percent = int((done / total) * 100) if total else 0
        self.progress_bar.setFormat(f"{done}/{total} runs ({percent}%)")
        self.status_label.setText(f"AutoTune running: {done}/{total}")
        self._update_time_labels()
