"""Виджеты для визуализации памяти (RAM/VRAM) в UI."""

from __future__ import annotations

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QGroupBox,
    QGridLayout,
    QSizePolicy,
)

from src.core.mem_viz_parser import MemoryData, COMPONENT_META, COMPONENT_ORDER, fmt_mem

_TR = QCoreApplication.translate


class MemoryBar(QWidget):
    """Горизонтальная полоса с сегментами памяти."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(20)
        self.setMaximumHeight(28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.segments: list[tuple[float, QColor, str]] = []
        self.total: float = 0.0

    def set_data(self, components: dict[str, float], component_order: list[str]):
        self.segments = []
        self.total = sum(components.values())
        if self.total <= 0:
            self.update()
            return

        for comp in component_order:
            val = components.get(comp, 0.0)
            if val <= 0:
                continue
            meta = COMPONENT_META.get(comp)
            if meta:
                color = QColor.fromHsv(
                    (meta["color"] * 137) % 360,
                    180,
                    220,
                )
                self.segments.append((val, color, meta["label"]))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        if self.total <= 0 or not self.segments:
            painter.setPen(QColor(80, 80, 80))
            painter.setBrush(QColor(40, 40, 40))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(rect, Qt.AlignCenter, _TR("MemoryBar", "No data"))
            return

        x = rect.left()
        total_width = rect.width()

        for val, color, label in self.segments:
            width = max(1, int(total_width * (val / self.total)))
            seg_rect = rect.__class__(x, rect.top(), width, rect.height())
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(seg_rect, 3, 3)
            x += width

        if x < rect.right():
            remaining = rect.__class__(x, rect.top(), rect.right() - x, rect.height())
            painter.setBrush(QColor(40, 40, 40))
            painter.drawRoundedRect(remaining, 3, 3)


class MemoryCategoryWidget(QGroupBox):
    """Виджет для отображения категории памяти (VRAM/RAM)."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.total_label = QLabel("—")
        self.total_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(self.total_label)

        self.util_label = QLabel("")
        self.util_label.setStyleSheet("color: #888;")
        header.addWidget(self.util_label)
        header.addStretch()
        layout.addLayout(header)

        self.util_bar = QProgressBar()
        self.util_bar.setRange(0, 100)
        self.util_bar.setValue(0)
        self.util_bar.setTextVisible(False)
        self.util_bar.setMaximumHeight(8)
        self.util_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #2a2a2a;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.util_bar)

        self.mem_bar = MemoryBar()
        layout.addWidget(self.mem_bar)

        self.components_grid = QGridLayout()
        self.components_grid.setColumnStretch(1, 1)
        self.components_grid.setColumnStretch(2, 0)
        layout.addLayout(self.components_grid)

        self.comp_widgets: dict[str, tuple[QLabel, QLabel, MemoryBar]] = {}

    def update_data(self, data: MemoryData, category: str):
        agg = data.get_aggregated()
        comps = agg.get(category, {})
        total = sum(comps.values())

        self.total_label.setText(f"{fmt_mem(total)}")

        util = data.utilization(category)
        if util is not None:
            self.util_label.setText(
                f"{util:.1f}% {self.tr('of')} {fmt_mem(data.system_memory.get(category, 0))}"
            )
            self.util_bar.setValue(int(min(100, util)))
            if util >= 90:
                self.util_bar.setStyleSheet("""
                    QProgressBar { border: none; background: #2a2a2a; border-radius: 4px; }
                    QProgressBar::chunk { background: #f44336; border-radius: 4px; }
                """)
            elif util >= 75:
                self.util_bar.setStyleSheet("""
                    QProgressBar { border: none; background: #2a2a2a; border-radius: 4px; }
                    QProgressBar::chunk { background: #FF9800; border-radius: 4px; }
                """)
            else:
                self.util_bar.setStyleSheet("""
                    QProgressBar { border: none; background: #2a2a2a; border-radius: 4px; }
                    QProgressBar::chunk { background: #4CAF50; border-radius: 4px; }
                """)
        else:
            self.util_label.setText("—")
            self.util_bar.setValue(0)

        components = data.components_used()
        self.mem_bar.set_data(comps, components)

        for i, comp in enumerate(components):
            mib = comps.get(comp, 0.0)
            if mib <= 0:
                continue

            if comp not in self.comp_widgets:
                meta = COMPONENT_META.get(comp, {})
                color = QColor.fromHsv((meta.get("color", 0) * 137) % 360, 180, 220)

                label = QLabel(meta.get("label", comp))
                label.setStyleSheet(f"color: {color.name()};")

                bar = MemoryBar()
                bar.setMaximumHeight(16)

                value_label = QLabel("—")
                value_label.setAlignment(Qt.AlignRight)

                row = len(self.comp_widgets)
                self.components_grid.addWidget(label, row, 0)
                self.components_grid.addWidget(bar, row, 1)
                self.components_grid.addWidget(value_label, row, 2)
                self.comp_widgets[comp] = (label, value_label, bar)
            else:
                label, value_label, bar = self.comp_widgets[comp]
                label.setVisible(True)
                value_label.setVisible(True)
                bar.setVisible(True)

            pct = (mib / total * 100) if total > 0 else 0
            value_label.setText(f"{fmt_mem(mib)} ({pct:.1f}%)")
            bar.set_data({comp: mib}, [comp])

        for comp, (label, value_label, bar) in self.comp_widgets.items():
            if comp not in components or comps.get(comp, 0) <= 0:
                label.setVisible(False)
                value_label.setVisible(False)
                bar.setVisible(False)


class MemoryVisualizationWidget(QWidget):
    """Главный виджет визуализации памяти."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        self.model_info = QLabel(self.tr("No model selected"))
        self.model_info.setStyleSheet("font-weight: bold; color: #b5cea8;")
        layout.addWidget(self.model_info)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.vram_widget = MemoryCategoryWidget("VRAM (GPU)")
        layout.addWidget(self.vram_widget)

        self.ram_widget = MemoryCategoryWidget("RAM (CPU)")
        layout.addWidget(self.ram_widget)

        self.total_label = QLabel(self.tr("Total: —"))
        self.total_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #d4d4d4;"
        )
        layout.addWidget(self.total_label)

        layout.addStretch()

    def update_from_data(self, data: MemoryData):
        info = data.model_info
        if info:
            parts = []
            if "name" in info:
                parts.append(info["name"])
            for key, label in [
                ("arch", "arch"),
                ("model_type", "type"),
                ("file_type", "quant"),
                ("params", "params"),
                ("ctx", "ctx"),
            ]:
                if key in info:
                    parts.append(f"{label}: {info[key]}")
            if "layers_offloaded" in info and "layers_total" in info:
                parts.append(
                    f"offload: {info['layers_offloaded']}/{info['layers_total']}"
                )
            self.model_info.setText(" | ".join(parts))
        else:
            self.model_info.setText(self.tr("No model selected"))

        status_parts = []
        if data.fatal_error:
            status_parts.append(f"{self.tr('Error')}: {data.fatal_error}")
            if data.failed_component:
                meta = COMPONENT_META.get(data.failed_component, {})
                status_parts.append(
                    f"{self.tr('Component')}: {meta.get('label', data.failed_component)}"
                )
            if data.failed_alloc_mib:
                status_parts.append(f"{self.tr('Requested')}: {fmt_mem(data.failed_alloc_mib)}")
        elif data.warnings:
            status_parts.append(f"{self.tr('Warnings')}: {len(data.warnings)}")
        elif data.server_ready:
            status_parts.append(self.tr("Server ready"))
        else:
            status_parts.append(self.tr("Waiting for data..."))

        self.status_label.setText("\n".join(status_parts))
        self.status_label.setStyleSheet(
            "color: #f44336;"
            if data.fatal_error
            else "color: #FF9800;"
            if data.warnings
            else "color: #4CAF50;"
        )

        self.vram_widget.update_data(data, "VRAM")
        self.ram_widget.update_data(data, "RAM")

        total = data.grand_total()
        self.total_label.setText(f"{self.tr('Total used')}: {fmt_mem(total)}")

    def clear(self):
        self.model_info.setText(self.tr("No model selected"))
        self.status_label.setText(self.tr("Server stopped"))
        self.status_label.setStyleSheet("color: #888;")
        self.vram_widget.update_data(MemoryData(), "VRAM")
        self.ram_widget.update_data(MemoryData(), "RAM")
        self.total_label.setText(self.tr("Total: —"))
