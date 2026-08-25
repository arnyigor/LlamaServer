"""Integration nav page (OpenCode / PI).

Built as a self-contained ``QScrollArea`` page; widget references are created on
``mw`` (MainWindowUI) so ``main.py`` keeps working unchanged. The
``_on_integration_target_changed`` handler stays on ``MainWindowUI`` and is
connected here.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QListWidget,
)

from src.ui.widgets import CollapsiblePanel


class IntegrationPage(QScrollArea):
    """Integration (OpenCode / PI) page."""

    def __init__(self, mw):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        self.setWidget(inner)
        self._build_integration_section(mw)
        lay.addWidget(mw.int_panel)
        lay.addStretch()

    def _build_integration_section(self, mw):
        # === 5. Интеграция ===
        mw.int_panel = CollapsiblePanel(
            mw.tr("Integration (OpenCode / PI)"),
            settings_key="panel_integration",
            collapsible=False,
        )

        # Каждый путь завёрнут в контейнер, чтобы показывать только тот,
        # что соответствует выбранному Target (см. _on_integration_target_changed).
        mw.opencode_row = QWidget()
        oc_layout = QHBoxLayout(mw.opencode_row)
        oc_layout.setContentsMargins(0, 0, 0, 0)
        oc_layout.addWidget(QLabel(mw.tr("OpenCode JSON:")))
        mw.opencode_config_path = QLineEdit(placeholderText="Path to opencode.json")
        oc_btn = QPushButton(mw.tr("..."))
        oc_btn.clicked.connect(mw._browse_opencode_clicked)
        oc_layout.addWidget(mw.opencode_config_path)
        oc_layout.addWidget(oc_btn)
        mw.int_panel.add_widget(mw.opencode_row)

        mw.pi_row = QWidget()
        pi_layout = QHBoxLayout(mw.pi_row)
        pi_layout.setContentsMargins(0, 0, 0, 0)
        pi_layout.addWidget(QLabel(mw.tr("PI JSON:")))
        mw.pi_config_path = QLineEdit(placeholderText="Path to PI config.json")
        pi_btn = QPushButton(mw.tr("..."))
        pi_btn.clicked.connect(mw._browse_pi_clicked)
        pi_layout.addWidget(mw.pi_config_path)
        pi_layout.addWidget(pi_btn)
        mw.int_panel.add_widget(mw.pi_row)

        # Максимальный контекст: авто (0) или ручное значение (токены).
        # При 0 значение подтягивается с сервера (GET /slots -> n_ctx) в main.py.
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel(mw.tr("Max context (tokens, 0=auto):")))
        mw.integration_max_context = QSpinBox()
        mw.integration_max_context.setRange(0, 2_000_000)
        mw.integration_max_context.setValue(0)
        mw.integration_max_context.setSingleStep(1024)
        mw.integration_max_context.setToolTip(
            mw.tr(
                "Размер окна контекста сервера. 0 — авто (считывается с "
                "запущенного сервера). Агент будет корректно сжимать контекст."
            )
        )
        ctx_layout.addWidget(mw.integration_max_context)
        mw.int_panel.add_layout(ctx_layout)

        tgt_layout = QHBoxLayout()
        tgt_layout.addWidget(QLabel(mw.tr("Target:")))
        mw.integration_target = QComboBox()
        mw.integration_target.addItem("OpenCode", "opencode")
        mw.integration_target.addItem("PI", "pi")
        tgt_layout.addWidget(mw.integration_target)
        mw.integration_check_btn = QPushButton(mw.tr("Check"))
        tgt_layout.addWidget(mw.integration_check_btn)
        mw.int_panel.add_layout(tgt_layout)

        mw.integration_target.currentIndexChanged.connect(
            mw._on_integration_target_changed
        )
        mw._on_integration_target_changed()

        mw.integration_model_label = QLabel("Model to add: not selected", wordWrap=True)
        mw.int_panel.add_widget(mw.integration_model_label)

        mw.integration_models_list = QListWidget()
        mw.integration_models_list.setMinimumHeight(80)
        mw.int_panel.add_widget(mw.integration_models_list)

        act_layout = QHBoxLayout()
        mw.integration_add_btn = QPushButton(mw.tr("Add"))
        mw.integration_remove_btn = QPushButton(mw.tr("Remove"))
        act_layout.addWidget(mw.integration_add_btn)
        act_layout.addWidget(mw.integration_remove_btn)
        mw.int_panel.add_layout(act_layout)

        mw.integration_status = QLabel(
            "Specify config path and click Check", wordWrap=True
        )
        mw.int_panel.add_widget(mw.integration_status)
