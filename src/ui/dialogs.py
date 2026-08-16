"""Единые диалоги подтверждения деструктивных действий (Этап 3.4 плана).

Раньше уровень строгости различался: удаление моделей спрашивало с No по
умолчанию, Force Stop — с No, часть диалогов — просто Yes/No без дефолта.
Теперь все опасные действия идут через confirm_destructive_action.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


def confirm_destructive_action(
    parent: QWidget,
    title: str,
    text: str,
    default_no: bool = True,
) -> bool:
    """Да/Нет для необратимых действий. По умолчанию фокус на No."""
    buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    default = (
        QMessageBox.StandardButton.No if default_no else QMessageBox.StandardButton.Yes
    )
    reply = QMessageBox.question(parent, title, text, buttons, default)
    return reply == QMessageBox.StandardButton.Yes
