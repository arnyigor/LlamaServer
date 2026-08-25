"""Панель Paths/llama.cpp как отдельный виджет (Этап 3.1: секция → класс).

Паттерн для дальнейшего разбиения left panel: панель не знает про
LlamaGUI, действия наружу — через сигналы; MainWindowUI реэкспортирует
виджеты атрибутами, поэтому config._FIELD_WIDGET_MAP и main.py работают
без изменений.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
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

        self.add_widget(box)
