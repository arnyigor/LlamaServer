"""Шапка приложения (Этап 1): профиль + Save flyout + язык.

Компактная горизонтальная панель вверху окна. Не знает про LlamaGUI:
действия наружу — через сигналы (profile_selected, save_requested,
language_changed). MainWindowUI реэкспортирует language_combo, чтобы
config/main.py продолжали работать как раньше.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMenu,
    QWidget,
)


class HeaderBar(QWidget):
    profile_selected = Signal(str)
    save_requested = Signal(str)
    language_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)

        # === Профиль ===
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(150)
        self.profile_combo.setToolTip("Select configuration profile")
        self.profile_combo.currentTextChanged.connect(
            lambda name: self.profile_selected.emit(name)
        )
        lay.addWidget(QLabel(self.tr("Profile:")))
        lay.addWidget(self.profile_combo)

        # === Save flyout ===
        self.save_btn = QPushButton(self.tr("Save"))
        self.save_menu = QMenu(self)
        for action in (
            "Save",
            "Save As",
            "Rename",
            "Clone",
            "Export",
            "Import",
            "Delete",
        ):
            act = QAction(action, self)
            act.triggered.connect(
                lambda _checked=False, a=action: self.save_requested.emit(a)
            )
            self.save_menu.addAction(act)
        self.save_btn.setMenu(self.save_menu)
        self.save_btn.setToolTip(
            "Save profile, or open the menu for Save As / Rename / Clone / "
            "Export / Import / Delete"
        )
        lay.addWidget(self.save_btn)

        lay.addStretch(1)

        # === Язык (переехал из PathsPanel) ===
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
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        lay.addWidget(QLabel(self.tr("Language:")))
        lay.addWidget(self.language_combo)

    def _on_language_changed(self, index: int):
        lang = str(self.language_combo.itemData(index) or "en")
        QSettings("LlamaServerGUI", "UIState").setValue("language", lang)
        self.language_changed.emit(lang)

    def set_profiles(self, names, current: str = ""):
        """Заполняет комбо профиля (блокируя сигналы во избежание ложного выбора)."""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(names or ["Default"])
        if current:
            idx = self.profile_combo.findText(current)
            if idx >= 0:
                self.profile_combo.setCurrentText(current)
        self.profile_combo.blockSignals(False)
