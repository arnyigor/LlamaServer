"""UI виджеты для LlamaServer GUI."""

from PySide6.QtCore import QEvent, QObject, QPointF, QSettings
from PySide6.QtGui import QWheelEvent

from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def add_clear_button(row_layout, widget, on_clear=None, tooltip=None):
    """Small "x" button next to an optional input.

    Resets the widget to its empty/auto/minimum state — the same state the
    CLI builder already treats as "flag not set" — so the parameter is
    excluded from the launch command instead of surviving as a stale value.
    ``on_clear`` runs after the reset for widgets needing extra bookkeeping
    (e.g. persisting that auto-fill should stay off for this field).

    ``tooltip`` overrides the default "excluded from the command" wording —
    use it for a field whose flag is always emitted regardless (e.g. a
    numeric tuning knob required whenever its parent feature is on), where
    the button can only reset the value, not remove the flag.
    """
    btn = QToolButton()
    btn.setText("✕")
    btn.setToolTip(tooltip or "Clear (exclude from launch command)")
    btn.setFixedSize(20, 20)
    btn.setStyleSheet(
        "QToolButton { color: #888; border: none; }"
        "QToolButton:hover { color: #e66; }"
    )

    def _clear():
        if isinstance(widget, QLineEdit):
            widget.setText("")
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(widget.minimum())
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(0)
        if on_clear:
            on_clear()

    btn.clicked.connect(_clear)
    row_layout.addWidget(btn)
    return btn


class NoWheelValueChangeFilter(QObject):
    """Запрещает колесу менять числовые поля и списки.

    Колесо перенаправляется ближайшей области прокрутки, поэтому пользователь
    продолжает прокручивать форму, даже когда курсор находится над полем.
    """

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.Wheel or not isinstance(
            watched, (QAbstractSpinBox, QComboBox)
        ):
            return super().eventFilter(watched, event)

        parent = watched.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()

        if parent is not None:
            viewport = parent.viewport()
            local_pos = QPointF(
                viewport.mapFromGlobal(event.globalPosition().toPoint())
            )
            forwarded = QWheelEvent(
                local_pos,
                event.globalPosition(),
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                event.phase(),
                event.inverted(),
                event.source(),
                event.pointingDevice(),
            )
            QApplication.sendEvent(viewport, forwarded)
        return True


class CollapsiblePanel(QWidget):
    """Виджет-контейнер с заголовком.

    При ``collapsible=True`` (по умолчанию) ведёт себя как спойлер со
    сворачиванием. При ``collapsible=False`` заголовок статичен, контент всегда
    виден — используется, когда секция уже вынесена на отдельную страницу
    навигации и сворачивание избыточно.
    """

    def __init__(self, title, parent=None, settings_key=None, collapsible=True):
        super().__init__(parent)
        self.base_title = title
        self._settings_key = settings_key or f"panel_{title}"
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(2)

        if collapsible:
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

            # Restore saved open/closed state
            settings = QSettings("LlamaServerGUI", "UIState")
            is_open = settings.value(self._settings_key, False, type=bool)
            self.toggle_btn.setChecked(is_open)
            self.content_widget.setVisible(is_open)
            if is_open:
                self.toggle_btn.setText(f"▼ {title}")
        else:
            # Статичный контейнер: заголовок-метка + всегда видимый контент.
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-weight: bold; color: #ccc; padding: 2px 0;")
            self.main_layout.addWidget(title_lbl)
            self.content_widget = QWidget()
            self.content_widget.setVisible(True)
            self.content_layout = QVBoxLayout(self.content_widget)
            self.content_layout.setContentsMargins(8, 6, 8, 6)
            self.content_layout.setSpacing(6)
            self.main_layout.addWidget(self.content_widget)

    def toggle_visibility(self):
        is_open = self.toggle_btn.isChecked()
        self.content_widget.setVisible(is_open)
        self.toggle_btn.setText(f"{'▼' if is_open else '▶'} {self.base_title}")
        settings = QSettings("LlamaServerGUI", "UIState")
        settings.setValue(self._settings_key, is_open)

    def add_widget(self, w):
        self.content_layout.addWidget(w)

    def add_layout(self, l):
        self.content_layout.addLayout(l)
