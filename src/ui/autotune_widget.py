"""Вкладка AutoTune поверх движка llama_autotuner."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from llama_autotuner.models import CandidateResult, LaunchProfile, RunStatus
from src.ui.widgets import CollapsiblePanel

_COLUMNS = [
    "#",
    "Status",
    "Phase",
    "Score",
    "ctx",
    "Placement",
    "KV",
    "batch/ubatch",
    "threads/tb",
    "MTP",
    "Vision",
    "PP tok/s",
    "TG tok/s",
    "VRAM peak",
    "VRAM class",
    "Reason",
]

_GOAL_PRESETS = {
    "Recommended fast": {"mode": "quick", "priority": "balanced"},
    "Max context": {"mode": "normal", "priority": "context"},
    "Best quality": {"mode": "normal", "priority": "quality"},
    "Full validation": {"mode": "deep", "priority": "balanced"},
}

_KV_CHOICES = ["f16/f16", "q8_0/q8_0", "q4_0/q4_0"]

_PROFILE_ORDER = ["OPTIMAL", "MAX_KV_PRECISION", "FASTEST", "MAX_CONTEXT"]

_PROFILE_PURPOSES = {
    "OPTIMAL": "Best overall balance of speed, context and VRAM headroom.",
    "MAX_KV_PRECISION": "Highest KV-cache/attention precision that still runs.",
    "FASTEST": "Highest measured decode speed (tok/s).",
    "MAX_CONTEXT": "Largest successfully measured, non-fragile context.",
}

_STATUS_COLORS = {
    RunStatus.PASS: QColor(200, 255, 200),
    RunStatus.PASS_DEGRADED: QColor(230, 255, 200),
    RunStatus.EARLY_REJECT: QColor(255, 230, 190),
    RunStatus.FAILED: QColor(255, 210, 210),
    RunStatus.INVALID_ENVIRONMENT: QColor(255, 200, 180),
    RunStatus.FATAL: QColor(255, 180, 180),
}


class AutoTuneWidget(QWidget):
    start_requested = Signal()
    cancel_requested = Signal()
    apply_requested = Signal(str)  # имя профиля (OPTIMAL/MAX_KV_PRECISION/...)
    open_results_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row_by_key: Dict[str, int] = {}
        self._done_runs = 0
        self._max_runs = 0
        self._profiles: List[LaunchProfile] = []
        self._last_output_dir: str = ""
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        settings_group = QGroupBox("AutoTune settings")
        settings = QVBoxLayout(settings_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Goal:"))
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(list(_GOAL_PRESETS.keys()))
        self.goal_combo.setToolTip(
            "Recommended fast — короткий прогон, сбалансированный приоритет.\n"
            "Max context — приоритет максимально большого контекста.\n"
            "Best quality — приоритет точности KV-кэша.\n"
            "Full validation — самый широкий и долгий поиск."
        )
        row1.addWidget(self.goal_combo)

        row1.addWidget(QLabel("Context:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(4096, 1_048_576)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(65536)
        self.ctx_spin.setToolTip("Целевой контекст (токенов), который должен работать.")
        row1.addWidget(self.ctx_spin)
        settings.addLayout(row1)

        row2 = QHBoxLayout()
        self.vision_chk = QCheckBox("Vision required")
        self.vision_chk.setToolTip(
            "Требовать рабочую поддержку изображений (mmproj) — иначе поиск "
            "прекратится, если она недоступна."
        )
        row2.addWidget(self.vision_chk)
        row2.addWidget(QLabel("mmproj:"))
        self.mmproj_edit = QLineEdit()
        self.mmproj_edit.setPlaceholderText("auto-detect if empty")
        row2.addWidget(self.mmproj_edit, 1)
        self.mmproj_browse_btn = QPushButton("...")
        self.mmproj_browse_btn.setFixedWidth(28)
        self.mmproj_browse_btn.clicked.connect(self._browse_mmproj)
        row2.addWidget(self.mmproj_browse_btn)
        settings.addLayout(row2)

        advanced = CollapsiblePanel("Advanced", settings_key="panel_autotune_advanced")
        adv = advanced.content_layout

        adv_row1 = QHBoxLayout()
        adv_row1.addWidget(QLabel("Search depth:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["quick", "normal", "deep"])
        adv_row1.addWidget(self.mode_combo)
        adv_row1.addWidget(QLabel("Priority:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["balanced", "context", "quality", "speed"])
        adv_row1.addWidget(self.priority_combo)
        adv_row1.addWidget(QLabel("Preferred KV:"))
        self.kv_combo = QComboBox()
        self.kv_combo.addItems(_KV_CHOICES)
        adv_row1.addWidget(self.kv_combo)
        adv.addLayout(adv_row1)

        adv_row2 = QHBoxLayout()
        adv_row2.addWidget(QLabel("Degradation policy:"))
        self.degradation_combo = QComboBox()
        self.degradation_combo.addItems(["auto", "report", "strict"])
        adv_row2.addWidget(self.degradation_combo)
        self.allow_kv_degradation_chk = QCheckBox("Allow KV degradation")
        self.allow_kv_degradation_chk.setChecked(True)
        adv_row2.addWidget(self.allow_kv_degradation_chk)
        self.allow_context_reduction_chk = QCheckBox("Allow context reduction")
        self.allow_context_reduction_chk.setChecked(True)
        adv_row2.addWidget(self.allow_context_reduction_chk)
        adv.addLayout(adv_row2)

        adv_row3 = QHBoxLayout()
        adv_row3.addWidget(QLabel("Min TG t/s:"))
        self.min_tg_spin = QDoubleSpinBox()
        self.min_tg_spin.setRange(0.0, 1000.0)
        self.min_tg_spin.setSpecialValueText("none")
        adv_row3.addWidget(self.min_tg_spin)
        adv_row3.addWidget(QLabel("Min PP t/s:"))
        self.min_pp_spin = QDoubleSpinBox()
        self.min_pp_spin.setRange(0.0, 5000.0)
        self.min_pp_spin.setSpecialValueText("none")
        adv_row3.addWidget(self.min_pp_spin)
        adv_row3.addWidget(QLabel("MTP:"))
        self.mtp_combo = QComboBox()
        self.mtp_combo.addItems(["auto", "on", "off"])
        adv_row3.addWidget(self.mtp_combo)
        adv.addLayout(adv_row3)

        adv_row4 = QHBoxLayout()
        adv_row4.addWidget(QLabel("Preferred VRAM margin (MiB):"))
        self.vram_margin_spin = QSpinBox()
        self.vram_margin_spin.setRange(0, 65536)
        self.vram_margin_spin.setValue(1024)
        adv_row4.addWidget(self.vram_margin_spin)
        self.require_vram_margin_chk = QCheckBox("Require margin (production-safe)")
        adv_row4.addWidget(self.require_vram_margin_chk)
        adv_row4.addWidget(QLabel("Absolute VRAM floor (MiB):"))
        self.vram_floor_spin = QSpinBox()
        self.vram_floor_spin.setRange(0, 8192)
        self.vram_floor_spin.setValue(300)
        adv_row4.addWidget(self.vram_floor_spin)
        adv.addLayout(adv_row4)

        adv_row5 = QHBoxLayout()
        adv_row5.addWidget(QLabel("Max time (min, 0=auto):"))
        self.max_time_spin = QSpinBox()
        self.max_time_spin.setRange(0, 600)
        adv_row5.addWidget(self.max_time_spin)
        adv_row5.addWidget(QLabel("Max runs (0=auto):"))
        self.max_runs_spin = QSpinBox()
        self.max_runs_spin.setRange(0, 500)
        adv_row5.addWidget(self.max_runs_spin)
        adv.addLayout(adv_row5)

        adv_row6 = QHBoxLayout()
        adv_row6.addWidget(QLabel("Extra runtime args:"))
        self.runtime_args_edit = QLineEdit()
        self.runtime_args_edit.setPlaceholderText("preserved verbatim in every tested/final command")
        adv_row6.addWidget(self.runtime_args_edit, 1)
        adv.addLayout(adv_row6)

        settings.addWidget(advanced)

        row3 = QHBoxLayout()
        self.start_btn = QPushButton("Run AutoTune")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.open_results_btn = QPushButton("Open Results Folder")
        self.open_results_btn.setEnabled(False)
        row3.addWidget(self.start_btn)
        row3.addWidget(self.cancel_btn)
        row3.addWidget(self.open_results_btn)
        settings.addLayout(row3)
        layout.addWidget(settings_group)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        self.profiles_group = QGroupBox("Recommended profiles")
        self.profiles_layout = QVBoxLayout(self.profiles_group)
        self.profiles_placeholder = QLabel("Profiles will appear here after a session completes.")
        self.profiles_placeholder.setWordWrap(True)
        self.profiles_layout.addWidget(self.profiles_placeholder)
        layout.addWidget(self.profiles_group)

        self.start_btn.clicked.connect(self.start_requested.emit)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.open_results_btn.clicked.connect(self.open_results_requested.emit)
        self.goal_combo.currentTextChanged.connect(self._apply_goal_preset)
        self._apply_goal_preset(self.goal_combo.currentText())

    def _browse_mmproj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mmproj GGUF", "", "GGUF files (*.gguf);;All files (*.*)"
        )
        if path:
            self.mmproj_edit.setText(path)

    def _apply_goal_preset(self, label: str) -> None:
        preset = _GOAL_PRESETS.get(label)
        if not preset:
            return
        self.mode_combo.setCurrentText(preset["mode"])
        self.priority_combo.setCurrentText(preset["priority"])

    # -------------------------------------------------------------- options

    def options(self) -> Dict[str, object]:
        kv_k, kv_v = self.kv_combo.currentText().split("/", 1)
        return {
            "ctx": self.ctx_spin.value(),
            "vision": "required" if self.vision_chk.isChecked() else "auto",
            "mmproj": self.mmproj_edit.text().strip() or None,
            "mode": self.mode_combo.currentText(),
            "priority": self.priority_combo.currentText(),
            "kv_k": kv_k,
            "kv_v": kv_v,
            "degradation_policy": self.degradation_combo.currentText(),
            "allow_kv_degradation": self.allow_kv_degradation_chk.isChecked(),
            "allow_context_reduction": self.allow_context_reduction_chk.isChecked(),
            "min_tg_tps": self.min_tg_spin.value() or None,
            "min_pp_tps": self.min_pp_spin.value() or None,
            "mtp_mode": self.mtp_combo.currentText(),
            "vram_margin_mb": self.vram_margin_spin.value(),
            "require_vram_margin": self.require_vram_margin_chk.isChecked(),
            "absolute_vram_floor_mb": self.vram_floor_spin.value(),
            "max_minutes": self.max_time_spin.value() or None,
            "max_runs": self.max_runs_spin.value() or None,
            "runtime_args": self.runtime_args_edit.text().split(),
        }

    # ------------------------------------------------------------ run state

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.start_btn.setText("AutoTune running..." if running else "Run AutoTune")
        self.cancel_btn.setEnabled(running)
        self.progress_bar.setVisible(running or self.progress_bar.value() > 0)
        if running:
            self.status_label.setText("Starting autotune session...")

    def clear_results(self) -> None:
        self._row_by_key.clear()
        self.table.setRowCount(0)
        self._done_runs = 0
        self._max_runs = 0
        self._profiles = []
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.status_label.setText("Idle.")
        self._clear_profiles_panel()
        self.open_results_btn.setEnabled(False)

    def mark_started(self, max_runs: int) -> None:
        self.clear_results()
        self._max_runs = max(int(max_runs), 0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(self._max_runs, 1))
        self.progress_bar.setValue(0)
        self.status_label.setText("AutoTune running...")

    def set_progress(self, done: int, max_runs: int) -> None:
        self._done_runs = max(0, int(done))
        if max_runs:
            self._max_runs = max(int(max_runs), self._done_runs)
        total = max(self._max_runs, self._done_runs, 1)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(self._done_runs)
        self.progress_bar.setFormat(f"{self._done_runs}/{total} runs")
        self.status_label.setText(f"AutoTune running: {self._done_runs}/{total} candidates tried")

    def show_error(self, message: str) -> None:
        self.status_label.setText(f"AutoTune failed: {message}")

    # --------------------------------------------------------------- table

    def _set_item(self, row: int, col: int, value: object) -> None:
        item = QTableWidgetItem(str(value) if value is not None else "")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, col, item)

    def _apply_row_color(self, row: int, status: RunStatus) -> None:
        color = _STATUS_COLORS.get(status)
        if color is None:
            return
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(color)

    def add_result(self, result: CandidateResult) -> None:
        c = result.candidate
        m = result.metrics
        row = self.table.rowCount()
        self.table.insertRow(row)
        key = c.key()
        self._row_by_key[key] = row
        placement = f"ncmoe={c.ncmoe}" if c.ncmoe is not None else f"ngl={c.ngl}"
        values = [
            row + 1,
            result.status.value,
            result.phase,
            f"{result.score:.3f}" if result.score is not None else "",
            c.ctx,
            placement,
            f"{c.kv_k}/{c.kv_v}",
            f"{c.batch}/{c.ubatch}",
            f"{c.threads}/{c.threads_batch}",
            f"{c.mtp_n_max}/{c.mtp_p_min:g}" if c.mtp else "off",
            "on" if c.vision else "off",
            f"{m.pp_tps:.1f}" if m.pp_tps else "",
            f"{m.tg_tps:.1f}" if m.tg_tps else "",
            f"{m.vram_peak_mb:.0f}" if m.vram_peak_mb else "",
            m.vram_operating_class or "",
            result.reason,
        ]
        for col, value in enumerate(values):
            self._set_item(row, col, value)
        self._apply_row_color(row, result.status)
        self.table.scrollToBottom()

    # ------------------------------------------------------------ profiles

    def _clear_profiles_panel(self) -> None:
        while self.profiles_layout.count():
            child = self.profiles_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        self.profiles_layout.addWidget(self.profiles_placeholder)
        self.profiles_placeholder.setVisible(True)

    def show_session_result(self, status: str, stop_reason: str, profiles: List[LaunchProfile],
                             elapsed_seconds: float, output_dir: str) -> None:
        self._profiles = list(profiles)
        self._last_output_dir = output_dir
        self.open_results_btn.setEnabled(bool(output_dir))
        self.status_label.setText(
            f"Session {status} ({stop_reason}) | {self._done_runs} candidates | "
            f"{elapsed_seconds:.0f}s | results: {output_dir}"
        )
        self._clear_profiles_panel()
        if not profiles:
            self.profiles_placeholder.setText("No stable profile found. Check the log for details.")
            return
        self.profiles_placeholder.setVisible(False)
        by_name = {p.name: p for p in profiles}
        ordered = [by_name[n] for n in _PROFILE_ORDER if n in by_name]
        ordered += [p for p in profiles if p.name not in _PROFILE_ORDER]
        for profile in ordered:
            self.profiles_layout.addWidget(self._build_profile_card(profile))

    def _build_profile_card(self, profile: LaunchProfile) -> QGroupBox:
        m = profile.result.metrics
        title = profile.name + (" (provisional)" if profile.provisional else "")
        card = QGroupBox(title)
        card_layout = QVBoxLayout(card)

        purpose = _PROFILE_PURPOSES.get(profile.name, profile.rationale)
        purpose_label = QLabel(purpose)
        purpose_label.setWordWrap(True)
        card_layout.addWidget(purpose_label)

        summary = QLabel(
            f"Confidence: {profile.confidence} | ctx={profile.candidate.ctx} | "
            f"PP={m.pp_tps:.1f} t/s | TG={m.tg_tps:.1f} t/s | VRAM class: {m.vram_operating_class or 'n/a'}"
            if m.pp_tps and m.tg_tps
            else f"Confidence: {profile.confidence}"
        )
        summary.setWordWrap(True)
        card_layout.addWidget(summary)

        command_label = QLabel(" ".join(profile.command))
        command_label.setWordWrap(True)
        command_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        command_label.setStyleSheet(
            "font-family: Consolas, monospace; background-color: #1e1e1e; color: #d4d4d4; "
            "padding: 6px; border-radius: 4px;"
        )
        card_layout.addWidget(command_label)

        apply_btn = QPushButton(f"Apply {profile.name}")
        apply_btn.clicked.connect(lambda: self.apply_requested.emit(profile.name))
        card_layout.addWidget(apply_btn)
        return card

    def profile_by_name(self, name: str) -> Optional[LaunchProfile]:
        for profile in self._profiles:
            if profile.name == name:
                return profile
        return None

    def last_output_dir(self) -> str:
        return self._last_output_dir
