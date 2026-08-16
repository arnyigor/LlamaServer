"""Панель Paths/llama.cpp как отдельный виджет (Этап 3.1: секция → класс).

Паттерн для дальнейшего разбиения left panel: панель не знает про
LlamaGUI, действия наружу — через сигналы; MainWindowUI реэкспортирует
виджеты атрибутами, поэтому config._FIELD_WIDGET_MAP и main.py работают
без изменений.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.ui.widgets import CollapsiblePanel


class PathsPanel(CollapsiblePanel):
    browse_exe_requested = Signal()
    browse_models_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Advanced: Paths and llama.cpp", parent=parent)
        box = QGroupBox("Paths")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 18, 12, 12)
        lay.setSpacing(8)

        self.exe_path = QLineEdit(placeholderText="Base folder with llama.cpp builds")
        self.exe_path.setToolTip(
            "Select the folder that contains version folders, e.g.\n"
            "G:/AIModels/llamacpp/\n\n"
            "Expected subfolders are auto-detected by CUDA version:\n"
            "llama-win-cuda-12.4-x64 / llama-win-cuda-13.3-x64"
        )
        self.bench_path = QLineEdit(placeholderText="Auto-detected llama-bench.exe")
        self.bench_path.setVisible(False)
        self.model_dir = QLineEdit(placeholderText="Base folder with models")

        for line, label, signal in (
            (self.exe_path, "Llama.cpp:", self.browse_exe_requested),
            (self.model_dir, "Models:", self.browse_models_requested),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(line, 1)
            btn = QPushButton("...")
            btn.setFixedWidth(32)
            btn.clicked.connect(lambda _checked=False, sig=signal: sig.emit())
            row.addWidget(btn)
            lay.addLayout(row)

        upd_row = QHBoxLayout()
        self.cuda_version_combo = QComboBox()
        self.cuda_version_combo.addItem("CUDA 12", "12")
        self.cuda_version_combo.addItem("CUDA 13", "13")
        self.cuda_version_combo.setMaximumWidth(110)
        self.cuda_version_combo.setToolTip(
            "CUDA major version for llama.cpp builds.\n"
            "CUDA 13 also downloads additional cudart DLLs.\n"
            "Minor version (12.4 / 13.3) is auto-detected from release."
        )
        self.update_llama_btn = QPushButton("Update llama.cpp")
        self.update_status = QLabel("idle", wordWrap=True)
        upd_row.addWidget(self.update_llama_btn)
        upd_row.addWidget(self.update_status, 1)
        lay.addLayout(upd_row)

        self.update_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        lay.addWidget(self.update_progress)
        lay.addLayout(self._build_language_row())

        self.add_widget(box)

    def _build_language_row(self) -> QHBoxLayout:
        """Язык интерфейса (Этап 4): применяется после перезапуска приложения."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Русский", "ru")
        ui_settings = QSettings("LlamaServerGUI", "UIState")
        saved_lang = str(ui_settings.value("language", "en") or "en").strip().lower()
        self.language_combo.setCurrentIndex(
            max(self.language_combo.findData(saved_lang), 0)
        )
        self.language_combo.setToolTip(
            "Interface language. Applied after the application restarts "
            "(or launch with --lang=ru / --lang=en)."
        )
        self.language_combo.currentIndexChanged.connect(
            lambda index: ui_settings.setValue(
                "language", str(self.language_combo.itemData(index) or "en")
            )
        )
        row.addWidget(self.language_combo)
        row.addStretch(1)
        return row
