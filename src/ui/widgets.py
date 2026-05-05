"""UI виджеты для LlamaServer GUI."""

from PySide6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CollapsiblePanel(QWidget):
    """Виджет-спойлер с возможностью сворачивания."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.base_title = title
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(2)

        self.toggle_btn = QPushButton(f"▶ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet(
            "text-align: left; font-weight: bold; border: 1px solid #444; "
            "padding: 5px; background: #2a2a2a; color: #ccc; border-radius: 4px;"
        )
        self.toggle_btn.clicked.connect(self.toggle_visibility)
        self.main_layout.addWidget(self.toggle_btn)

        self.content_widget = QWidget()
        self.content_widget.setVisible(False)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 6, 8, 6)
        self.content_layout.setSpacing(6)
        self.main_layout.addWidget(self.content_widget)

    def toggle_visibility(self):
        is_open = self.toggle_btn.isChecked()
        self.content_widget.setVisible(is_open)
        self.toggle_btn.setText(f"{'▼' if is_open else '▶'} {self.base_title}")

    def add_widget(self, w):
        self.content_layout.addWidget(w)

    def add_layout(self, l):
        self.content_layout.addLayout(l)
