"""Models / Library nav page: local model manager and Hugging Face download.

Built as a self-contained ``QScrollArea`` page; widget references are created on
``mw`` (MainWindowUI) so ``main.py`` keeps working unchanged.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QTableWidget,
    QProgressBar,
    QAbstractItemView,
    QHeaderView,
)
from PySide6.QtCore import Qt

from src.ui.widgets import CollapsiblePanel


class ModelLibraryPage(QScrollArea):
    """Models / Library page: local models + Hugging Face download."""

    def __init__(self, mw):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        self.setWidget(inner)
        self._build_hf_models_section(mw)
        lay.addWidget(mw.models_panel)
        lay.addStretch()

    def _build_hf_models_section(self, mw):
        # === 2a. Локальные модели + загрузка с Hugging Face ===
        mw.models_panel = CollapsiblePanel(
            mw.tr("Local model manager and download"),
            settings_key="panel_models",
            collapsible=False,
        )
        local = mw.models_panel.content_layout

        local_row = QHBoxLayout()
        mw.local_models_refresh_btn = QPushButton(mw.tr("Refresh local models"))
        mw.local_models_delete_btn = QPushButton(mw.tr("Delete selected"))
        mw.local_models_delete_btn.setEnabled(False)
        mw.local_models_delete_btn.setToolTip(
            "Safely delete the selected model folder/file from the Models base folder."
        )
        local_row.addWidget(QLabel(mw.tr("All models under Models:")))
        local_row.addStretch(1)
        local_row.addWidget(mw.local_models_refresh_btn)
        local_row.addWidget(mw.local_models_delete_btn)
        local.addLayout(local_row)

        mw.local_models_list = QTableWidget(0, 5)
        mw.local_models_list.setHorizontalHeaderLabels(
            ["Name", "Type", "GGUF", "Size", "Examples"]
        )
        mw.local_models_list.setMaximumHeight(130)
        mw.local_models_list.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        mw.local_models_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        mw.local_models_list.horizontalHeader().setStretchLastSection(True)
        mw.local_models_list.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        mw.local_models_list.setToolTip(
            "Shows every local model folder/file found under the Models path, not only HF downloads. "
            "Projectors/MTP drafts are included in folder deletion but not shown as standalone models."
        )
        local.addWidget(mw.local_models_list)

        mw.local_models_status = QLabel(mw.tr("Refresh to list local models"))
        mw.local_models_status.setWordWrap(True)
        local.addWidget(mw.local_models_status)

        # --- Подблок загрузки с Hugging Face (можно скрыть) ---
        mw.show_hf_download = QCheckBox(mw.tr("Show Hugging Face download"))
        mw.show_hf_download.setChecked(True)
        mw.show_hf_download.setToolTip(
            "Uncheck to hide the Hugging Face download section and keep the panel compact."
        )
        local.addWidget(mw.show_hf_download)

        mw.hf_section = QWidget()
        hf = QVBoxLayout(mw.hf_section)
        hf.setContentsMargins(0, 0, 0, 0)
        hf.setSpacing(6)

        mw.hf_repo = QLineEdit(
            placeholderText="repo or URL, e.g. unsloth/Qwen3.6-27B-MTP-GGUF"
        )
        mw.hf_repo.setToolTip(
            "Paste Hugging Face repo id or model URL. Files are saved as:\n"
            "<Models>/<author>/<model>/<file>.gguf, compatible with LM Studio."
        )
        hf.addWidget(mw.hf_repo)

        hf_filter_row = QHBoxLayout()
        mw.hf_quant_filter = QLineEdit(placeholderText="filter: Q4_K_M or Q3-BF16")
        mw.hf_quant_filter.setToolTip(
            "Optional filter. Examples: Q4_K_M, IQ4, Q3-BF16.\n"
            "Q3-BF16 means show quants from Q3 up to BF16."
        )
        mw.hf_scan_btn = QPushButton(mw.tr("Scan HF"))
        hf_filter_row.addWidget(mw.hf_quant_filter, 1)
        hf_filter_row.addWidget(mw.hf_scan_btn)
        hf.addLayout(hf_filter_row)

        hf_opts_row = QHBoxLayout()
        mw.hf_include_mmproj = QCheckBox(mw.tr("also vision/mmproj"))
        mw.hf_include_mmproj.setChecked(True)
        mw.hf_download_btn = QPushButton(mw.tr("Download selected models"))
        mw.hf_download_btn.setEnabled(False)
        mw.hf_pause_btn = QPushButton(mw.tr("Pause selected"))
        mw.hf_pause_btn.setEnabled(False)
        mw.hf_cancel_btn = QPushButton(mw.tr("Cancel selected"))
        mw.hf_cancel_btn.setEnabled(False)
        hf_opts_row.addWidget(mw.hf_include_mmproj)
        hf_opts_row.addWidget(mw.hf_download_btn)
        hf_opts_row.addWidget(mw.hf_pause_btn)
        hf_opts_row.addWidget(mw.hf_cancel_btn)
        hf_opts_row.addStretch(1)
        hf.addLayout(hf_opts_row)

        mw.hf_files = QTableWidget(0, 4)
        mw.hf_files.setHorizontalHeaderLabels(["Name", "Quant", "Size", "Progress"])
        mw.hf_files.setMaximumHeight(120)
        mw.hf_files.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        mw.hf_files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        mw.hf_files.horizontalHeader().setStretchLastSection(True)
        mw.hf_files.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        mw.hf_files.setToolTip(
            "Select one or several GGUF files (Ctrl/Shift). Each file is downloaded "
            "as an independent concurrent task. Vision projector is added once."
        )
        hf.addWidget(mw.hf_files)

        mw.hf_status = QLabel(mw.tr("Paste repo and scan"))
        mw.hf_status.setWordWrap(True)
        mw.hf_progress = QProgressBar(visible=False, minimum=0, maximum=100)
        hf.addWidget(mw.hf_status)
        hf.addWidget(mw.hf_progress)

        hf.addWidget(QLabel(mw.tr("Downloads (select tasks to pause/cancel):")))
        mw.hf_downloads = QTableWidget(0, 6)
        mw.hf_downloads.setHorizontalHeaderLabels(
            ["Name", "Status", "Progress", "Size", "Speed", "ETA"]
        )
        mw.hf_downloads.setMaximumHeight(220)
        mw.hf_downloads.setWordWrap(False)
        mw.hf_downloads.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        mw.hf_downloads.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        mw.hf_downloads.horizontalHeader().setStretchLastSection(True)
        mw.hf_downloads.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        mw.hf_downloads.setToolTip(
            "Independent parallel downloads with downloaded/total size, remaining size, "
            "speed and ETA. Select one or several tasks before Pause/Cancel."
        )
        hf.addWidget(mw.hf_downloads)

        hf_local_row = QHBoxLayout()
        mw.hf_refresh_local_btn = QPushButton(mw.tr("Refresh local"))
        mw.hf_delete_local_folder_btn = QPushButton(mw.tr("Delete local folder"))
        mw.hf_delete_local_folder_btn.setEnabled(False)
        hf_local_row.addWidget(QLabel(mw.tr("Local files:")))
        hf_local_row.addStretch(1)
        hf_local_row.addWidget(mw.hf_refresh_local_btn)
        hf_local_row.addWidget(mw.hf_delete_local_folder_btn)
        hf.addLayout(hf_local_row)

        mw.hf_local_files = QTableWidget(0, 3)
        mw.hf_local_files.setHorizontalHeaderLabels(["Name", "Status", "Size"])
        mw.hf_local_files.setMaximumHeight(90)
        mw.hf_local_files.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        mw.hf_local_files.horizontalHeader().setStretchLastSection(True)
        mw.hf_local_files.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        mw.hf_local_files.setToolTip(
            "Files already present in <Models>/<author>/<model>. Delete local folder removes the whole repo folder including mmproj/vision files."
        )
        hf.addWidget(mw.hf_local_files)
        local.addWidget(mw.hf_section)
        mw.show_hf_download.toggled.connect(mw.hf_section.setVisible)
