"""Вкладка AutoTune поверх движка llama_autotuner."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
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
    "RAM peak",
    "Reason",
]

_KV_CHOICES = ["f16/f16", "q8_0/q8_0", "q4_0/q4_0"]

# Fixed search strategy (matches upstream llama_autotuner's GUI RECOMMENDED_PROFILE):
# exposing mode/priority/budget as user choices let people pick the wrong branch
# under VRAM pressure before search even started, so upstream hard-coded these
# and removed the picker. Do not reintroduce a Quick/Balanced/Thorough selector.
_SEARCH_MODE = "quick"
_SEARCH_PRIORITY = "balanced"
_SEARCH_MAX_MINUTES = 8
_SEARCH_MAX_RUNS = 12
_ABSOLUTE_VRAM_FLOOR_MB = 300

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

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self._main_window = main_window
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

        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.setMinimumContentsLength(60)
        self.model_combo.setToolTip(
            "Target GGUF model for AutoTune (kept in sync with the model "
            "selected on the Launch page)."
        )
        model_row.addWidget(self.model_combo, 1)
        settings.addLayout(model_row)

        main_grid = QGridLayout()
        main_grid.setHorizontalSpacing(10)
        main_grid.setVerticalSpacing(6)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(4096, 1_048_576)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(65536)
        self.ctx_spin.setToolTip("Целевой контекст (токенов), который должен работать.")
        main_grid.addWidget(QLabel("Context:"), 0, 0)
        main_grid.addWidget(self.ctx_spin, 0, 1, 1, 3)

        # Быстрые кнопки контекста — тот же набор степеней 2, что на Launch.
        ctx_quick_row = QHBoxLayout()
        ctx_quick_row.setSpacing(4)
        ctx_quick_row.addWidget(QLabel("Quick:"))
        self.ctx_quick_buttons: List[QPushButton] = []
        for label, value in [
            ("8K", 8192),
            ("16K", 16384),
            ("32K", 32768),
            ("64K", 65536),
            ("128K", 131072),
            ("256K", 262144),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(42 if len(label) <= 3 else 50)
            btn.setFixedHeight(24)
            btn.setToolTip(f"Set Context to {value}")
            btn.clicked.connect(lambda _checked=False, v=value: self.ctx_spin.setValue(v))
            self.ctx_quick_buttons.append(btn)
            ctx_quick_row.addWidget(btn)
        ctx_quick_row.addStretch(1)
        main_grid.addLayout(ctx_quick_row, 1, 0, 1, 4)

        self.vision_chk = QCheckBox("Vision required")
        self.vision_chk.setToolTip(
            "Требовать рабочую поддержку изображений (mmproj) — иначе поиск "
            "прекратится, если она недоступна. Путь к mmproj — на вкладке Advanced."
        )
        main_grid.addWidget(self.vision_chk, 2, 0, 1, 2)

        main_grid.setColumnStretch(1, 1)
        main_grid.setColumnStretch(3, 1)
        settings.addLayout(main_grid)

        advanced = CollapsiblePanel("Advanced", settings_key="panel_autotune_advanced")
        adv = advanced.content_layout
        adv_grid = QGridLayout()
        adv_grid.setHorizontalSpacing(10)
        adv_grid.setVerticalSpacing(6)

        self.kv_combo = QComboBox()
        self.kv_combo.addItems(_KV_CHOICES)
        # Q8 KV is the practical default: near-F16 quality at roughly half the
        # KV-cache VRAM, so most searches don't need the F16 precision ceiling.
        self.kv_combo.setCurrentText("q8_0/q8_0")
        adv_grid.addWidget(QLabel("Preferred KV-cache precision:"), 0, 0)
        adv_grid.addWidget(self.kv_combo, 0, 1, 1, 3)
        kv_note = QLabel(
            "This tunes the attention/KV cache only. The weight quantization "
            "of the selected GGUF is fixed."
        )
        kv_note.setWordWrap(True)
        adv_grid.addWidget(kv_note, 1, 0, 1, 4)

        self.strict_chk = QCheckBox("Exact target only (do not use context/KV alternatives)")
        adv_grid.addWidget(self.strict_chk, 2, 0, 1, 4)

        self.allow_kv_degradation_chk = QCheckBox("Allow lower-precision KV cache when necessary")
        self.allow_kv_degradation_chk.setChecked(True)
        adv_grid.addWidget(self.allow_kv_degradation_chk, 3, 0, 1, 4)

        self.allow_context_reduction_chk = QCheckBox("Allow lower-context alternatives")
        self.allow_context_reduction_chk.setChecked(True)
        adv_grid.addWidget(self.allow_context_reduction_chk, 4, 0, 1, 4)

        self.min_tg_spin = QDoubleSpinBox()
        self.min_tg_spin.setRange(0.0, 1000.0)
        self.min_tg_spin.setSpecialValueText("none")
        self.vram_margin_spin = QSpinBox()
        self.vram_margin_spin.setRange(0, 65536)
        self.vram_margin_spin.setValue(1024)
        adv_grid.addWidget(QLabel("Minimum generation speed (t/s, optional):"), 5, 0)
        adv_grid.addWidget(self.min_tg_spin, 5, 1)
        adv_grid.addWidget(QLabel("Preferred free VRAM (MiB):"), 5, 2)
        adv_grid.addWidget(self.vram_margin_spin, 5, 3)

        self.require_vram_margin_chk = QCheckBox(
            "Production-safe: require the full preferred VRAM reserve in FULL/FINAL"
        )
        adv_grid.addWidget(self.require_vram_margin_chk, 6, 0, 1, 4)

        self.mtp_combo = QComboBox()
        self.mtp_combo.addItems(["auto", "on", "off"])
        adv_grid.addWidget(QLabel("MTP/speculative decoding:"), 7, 0)
        adv_grid.addWidget(self.mtp_combo, 7, 1)

        self.runtime_args_edit = QLineEdit()
        self.runtime_args_edit.setPlaceholderText("preserved verbatim in every tested/final command")
        adv_grid.addWidget(QLabel("Extra runtime args:"), 8, 0)
        adv_grid.addWidget(self.runtime_args_edit, 8, 1, 1, 3)

        self.mmproj_edit = QLineEdit()
        self.mmproj_edit.setPlaceholderText("auto-detect if empty")
        self.mmproj_browse_btn = QPushButton("...")
        self.mmproj_browse_btn.setFixedWidth(28)
        self.mmproj_browse_btn.clicked.connect(self._browse_mmproj)
        mmproj_row = QHBoxLayout()
        mmproj_row.setSpacing(6)
        mmproj_row.addWidget(self.mmproj_edit, 1)
        mmproj_row.addWidget(self.mmproj_browse_btn)
        adv_grid.addWidget(QLabel("Vision projector override:"), 9, 0)
        adv_grid.addLayout(mmproj_row, 9, 1, 1, 3)

        adv_grid.setColumnStretch(1, 1)
        adv_grid.setColumnStretch(3, 1)
        adv.addLayout(adv_grid)

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

    def _sync_model_items(self) -> None:
        """Repopulate ``model_combo`` from the Launch page's model combo.

        Full repopulation (rather than reacting to selection changes) avoids
        the earlier bug where ``currentIndexChanged`` fired while the source
        combo still had a single item, leaving this picker with just one
        entry after a scan.
        """
        source = getattr(self._main_window, "model_combo", None)
        if source is None:
            return
        current_path = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for i in range(source.count()):
                self.model_combo.addItem(source.itemText(i), source.itemData(i))
        finally:
            self.model_combo.blockSignals(False)
        idx = self.model_combo.findData(current_path)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else (0 if source.count() else -1))

    def _browse_mmproj(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mmproj GGUF", "", "GGUF files (*.gguf);;All files (*.*)"
        )
        if path:
            self.mmproj_edit.setText(path)

    # -------------------------------------------------------------- options

    def options(self) -> Dict[str, object]:
        kv_k, kv_v = self.kv_combo.currentText().split("/", 1)
        return {
            "ctx": self.ctx_spin.value(),
            # "off", never "auto": session.py silently promotes "auto" to
            # REQUIRED whenever an mmproj happens to be auto-detected next to
            # the model, regardless of this checkbox. Upstream's own GUI
            # never sends "auto" for the same reason — it's a CLI-only power
            # option, not something a checkbox should implicitly trigger.
            "vision": "required" if self.vision_chk.isChecked() else "off",
            "mmproj": self.mmproj_edit.text().strip() or None,
            # Fixed search strategy — see upstream llama_autotuner's GUI hard
            # rule (SPECIFICATION.md "GUI one-click workflow"): never expose
            # mode/priority/budget as a pre-search choice.
            "mode": _SEARCH_MODE,
            "priority": _SEARCH_PRIORITY,
            "kv_k": kv_k,
            "kv_v": kv_v,
            "degradation_policy": "strict" if self.strict_chk.isChecked() else "auto",
            "allow_kv_degradation": self.allow_kv_degradation_chk.isChecked(),
            "allow_context_reduction": self.allow_context_reduction_chk.isChecked(),
            "min_tg_tps": self.min_tg_spin.value() or None,
            "min_pp_tps": None,
            "mtp_mode": self.mtp_combo.currentText(),
            "vram_margin_mb": self.vram_margin_spin.value(),
            "require_vram_margin": self.require_vram_margin_chk.isChecked(),
            "absolute_vram_floor_mb": _ABSOLUTE_VRAM_FLOOR_MB,
            "max_minutes": _SEARCH_MAX_MINUTES,
            "max_runs": _SEARCH_MAX_RUNS,
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
            f"{m.ram_peak_mb:.0f}" if m.ram_peak_mb else "",
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
            if widget is not None and widget is not self.profiles_placeholder:
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

        buttons_row = QHBoxLayout()
        apply_btn = QPushButton(f"Apply {profile.name}")
        apply_btn.clicked.connect(lambda: self.apply_requested.emit(profile.name))
        buttons_row.addWidget(apply_btn)
        copy_btn = QPushButton("Copy command")
        command_text = " ".join(profile.command)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(command_text)
        )
        buttons_row.addWidget(copy_btn)
        buttons_row.addStretch(1)
        card_layout.addLayout(buttons_row)
        return card

    def profile_by_name(self, name: str) -> Optional[LaunchProfile]:
        for profile in self._profiles:
            if profile.name == name:
                return profile
        return None

    def last_output_dir(self) -> str:
        return self._last_output_dir
