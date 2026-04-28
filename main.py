import sys
import json
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit,
    QTextEdit, QFileDialog, QGroupBox, QMessageBox, QListWidget,
    QDoubleSpinBox, QCheckBox, QSplitter
)
from PySide6.QtCore import QProcess, Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

class ModelScanner(QThread):
    """Сканирование папок с GGUF в отдельном потоке"""
    models_found = Signal(list)

    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def run(self):
        models = []
        base = Path(self.base_path)
        if base.exists():
            for gguf_file in base.rglob("*.gguf"):
                rel_path = gguf_file.relative_to(base)
                models.append((str(rel_path), str(gguf_file)))
        self.models_found.emit(models)

class LlamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLama.cpp GUI Manager")
        self.setGeometry(100, 100, 1000, 750)

        # Процесс для сервера
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.stateChanged.connect(self.handle_state)

        # Процесс для бенчмарка (отдельный)
        self.bench_process = QProcess()
        self.bench_process.readyReadStandardOutput.connect(self.handle_bench_stdout)
        self.bench_process.readyReadStandardError.connect(self.handle_bench_stderr)
        self.bench_process.finished.connect(self.handle_bench_finished)

        self.profiles_file = "profiles.json"
        self.settings_file = "settings.json"
        self.profiles = {}
        self.settings = {}

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Левая панель (управление)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Группа путей
        path_group = QGroupBox("Пути")
        path_layout = QVBoxLayout(path_group)

        # Путь к llama-server.exe
        exe_layout = QHBoxLayout()
        self.exe_path = QLineEdit()
        self.exe_path.setPlaceholderText("Путь к llama-server.exe")
        self.exe_path.textChanged.connect(self.auto_detect_bench)  # Автоопределение bench
        exe_btn = QPushButton("Обзор")
        exe_btn.clicked.connect(self.browse_exe)
        exe_layout.addWidget(self.exe_path)
        exe_layout.addWidget(exe_btn)
        path_layout.addLayout(exe_layout)

        # Путь к llama-bench.exe
        bench_layout = QHBoxLayout()
        self.bench_path = QLineEdit()
        self.bench_path.setPlaceholderText("Путь к llama-bench.exe (авто)")
        bench_btn = QPushButton("Обзор")
        bench_btn.clicked.connect(self.browse_bench)
        bench_layout.addWidget(QLabel("Benchmark:"))
        bench_layout.addWidget(self.bench_path)
        bench_layout.addWidget(bench_btn)
        path_layout.addLayout(bench_layout)

        # Базовая папка моделей
        model_dir_layout = QHBoxLayout()
        self.model_dir = QLineEdit()
        self.model_dir.setPlaceholderText("Базовая папка с моделями")
        model_dir_btn = QPushButton("Обзор")
        model_dir_btn.clicked.connect(self.browse_model_dir)
        scan_btn = QPushButton("🔍 Сканировать")
        scan_btn.clicked.connect(self.scan_models)
        model_dir_layout.addWidget(self.model_dir)
        model_dir_layout.addWidget(model_dir_btn)
        model_dir_layout.addWidget(scan_btn)
        path_layout.addLayout(model_dir_layout)

        left_layout.addWidget(path_group)

        # Выбор модели
        model_group = QGroupBox("Модель")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        model_layout.addWidget(QLabel("Найденные GGUF:"))
        model_layout.addWidget(self.model_combo)

        left_layout.addWidget(model_group)

        # Параметры генерации
        params_group = QGroupBox("Параметры генерации")
        params_layout = QVBoxLayout(params_group)

        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        self.temperature.setDecimals(2)
        temp_layout.addWidget(self.temperature)
        temp_layout.addStretch()
        params_layout.addLayout(temp_layout)

        # Repeat Penalty
        penalty_layout = QHBoxLayout()
        penalty_layout.addWidget(QLabel("Repeat Penalty:"))
        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(0.0, 2.0)
        self.repeat_penalty.setSingleStep(0.01)
        self.repeat_penalty.setValue(1.1)
        self.repeat_penalty.setDecimals(2)
        penalty_layout.addWidget(self.repeat_penalty)
        penalty_layout.addStretch()
        params_layout.addLayout(penalty_layout)

        # GPU Layers
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("GPU Layers (-ngl):"))
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 200)
        self.gpu_layers.setValue(33)
        gpu_layout.addWidget(self.gpu_layers)
        params_layout.addLayout(gpu_layout)

        # Context Size
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Size (-c):"))
        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(512, 128000)
        self.ctx_size.setSingleStep(512)
        self.ctx_size.setValue(4096)
        ctx_layout.addWidget(self.ctx_size)
        params_layout.addLayout(ctx_layout)

        # Threads
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads (-t):"))
        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(os.cpu_count() or 4)
        threads_layout.addWidget(self.threads)
        params_layout.addLayout(threads_layout)

        # Port
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8080)
        port_layout.addWidget(self.port)
        params_layout.addLayout(port_layout)

        # Flash Attention
        self.flash_attn = QCheckBox("Flash Attention (-fa)")
        self.flash_attn.setChecked(True)
        params_layout.addWidget(self.flash_attn)

        # Доп. параметры
        params_layout.addWidget(QLabel("Доп. параметры:"))
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("--top-p 0.9 --min-p 0.05 ...")
        params_layout.addWidget(self.extra_args)

        left_layout.addWidget(params_group)

        # Параметры бенчмарка
        bench_params_group = QGroupBox("Параметры тестирования")
        bench_layout = QHBoxLayout(bench_params_group)

        bench_layout.addWidget(QLabel("Prompt (-p):"))
        self.bench_prompt = QSpinBox()
        self.bench_prompt.setRange(16, 4096)
        self.bench_prompt.setValue(128)
        self.bench_prompt.setSingleStep(64)
        bench_layout.addWidget(self.bench_prompt)

        bench_layout.addWidget(QLabel("Gen (-n):"))
        self.bench_gen = QSpinBox()
        self.bench_gen.setRange(16, 4096)
        self.bench_gen.setValue(256)
        self.bench_gen.setSingleStep(64)
        bench_layout.addWidget(self.bench_gen)

        left_layout.addWidget(bench_params_group)

        # Профили
        profile_group = QGroupBox("Профили")
        profile_layout = QVBoxLayout(profile_group)

        profile_input_layout = QHBoxLayout()
        self.profile_name = QLineEdit()
        self.profile_name.setPlaceholderText("Имя профиля")
        save_prof_btn = QPushButton("💾 Сохранить")
        save_prof_btn.clicked.connect(self.save_profile)
        del_prof_btn = QPushButton("🗑 Удалить")
        del_prof_btn.clicked.connect(self.delete_profile)
        profile_input_layout.addWidget(self.profile_name)
        profile_input_layout.addWidget(save_prof_btn)
        profile_input_layout.addWidget(del_prof_btn)
        profile_layout.addLayout(profile_input_layout)

        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.load_profile)
        profile_layout.addWidget(self.profile_list)

        left_layout.addWidget(profile_group)

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.test_btn = QPushButton("🧪 Тестировать")
        self.test_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.test_btn.clicked.connect(self.run_benchmark)

        self.start_btn = QPushButton("▶ Старт Server")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_server)

        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_server)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.test_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        main_layout.addWidget(left_panel, 1)

        # Правая панель (логи)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("Логи:"))
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        right_layout.addWidget(self.logs)

        clear_btn = QPushButton("🧹 Очистить логи")
        clear_btn.clicked.connect(self.logs.clear)
        right_layout.addWidget(clear_btn)

        main_layout.addWidget(right_panel, 2)

    def auto_detect_bench(self):
        """Автоопределение пути к llama-bench рядом с llama-server"""
        server_path = self.exe_path.text()
        if server_path and os.path.exists(server_path):
            dir_path = os.path.dirname(server_path)
            base_name = os.path.basename(server_path).replace("server", "bench")
            bench_path = os.path.join(dir_path, base_name)
            if os.path.exists(bench_path):
                self.bench_path.setText(bench_path)

    def browse_exe(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите llama-server", "", "Executable (*.exe)")
        if file:
            self.exe_path.setText(file)
            self.save_settings()

    def browse_bench(self):
        file, _ = QFileDialog.getOpenFileName(self, "Выберите llama-bench", "", "Executable (*.exe)")
        if file:
            self.bench_path.setText(file)
            self.save_settings()

    def browse_model_dir(self):
        dir = QFileDialog.getExistingDirectory(self, "Выберите папку с моделями")
        if dir:
            self.model_dir.setText(dir)
            self.save_settings()
            self.scan_models()

    def scan_models(self):
        base_path = self.model_dir.text()
        if not base_path or not os.path.exists(base_path):
            QMessageBox.warning(self, "Ошибка", "Укажите существующую базовую папку")
            return

        self.log("🔍 Сканирование моделей...")
        self.scanner = ModelScanner(base_path)
        self.scanner.models_found.connect(self.on_models_found)
        self.scanner.start()

    def on_models_found(self, models):
        self.model_combo.clear()
        for name, path in models:
            self.model_combo.addItem(name, path)
        self.log(f"✅ Найдено моделей: {len(models)}")

    def build_args(self, for_benchmark=False):
        """Сборка аргументов командной строки"""
        args = []

        # Модель
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "Ошибка", "Выберите модель")
            return None

        if for_benchmark:
            # Для llama-bench: -m model -p 128 -n 256 -ngl 46
            args.extend(["-m", model_path])
            args.extend(["-p", str(self.bench_prompt.value())])
            args.extend(["-n", str(self.bench_gen.value())])
            args.extend(["-ngl", str(self.gpu_layers.value())])
            if self.flash_attn.isChecked():
                args.extend(["-fa", "1"])
            # Для бенчмарка можно добавить тип кэша
            args.extend(["-ctk", "q8_0"])
            args.extend(["-ctv", "q8_0"])
        else:
            # Для сервера
            args.extend(["-m", model_path])
            args.extend(["--port", str(self.port.value())])
            args.extend(["-ngl", str(self.gpu_layers.value())])
            args.extend(["-c", str(self.ctx_size.value())])
            args.extend(["-t", str(self.threads.value())])

            # Параметры сэмплирования для сервера
            args.extend(["--temp", str(self.temperature.value())])
            args.extend(["--repeat-penalty", str(self.repeat_penalty.value())])

            if self.flash_attn.isChecked():
                args.extend(["--flash-attn", "on"])

            # Доп. аргументы
            if self.extra_args.text():
                args.extend(self.extra_args.text().split())

        return args

    def run_benchmark(self):
        """Запуск llama-bench с текущими параметрами"""
        bench_exe = self.bench_path.text()
        if not bench_exe or not os.path.exists(bench_exe):
            # Попробуем автоопределить если не задан
            self.auto_detect_bench()
            bench_exe = self.bench_path.text()
            if not bench_exe or not os.path.exists(bench_exe):
                QMessageBox.critical(self, "Ошибка", "Укажите путь к llama-bench.exe")
                return

        args = self.build_args(for_benchmark=True)
        if not args:
            return

        self.log(f"🧪 Запуск бенчмарка: {os.path.basename(bench_exe)}")
        self.log(f"   Модель: {self.model_combo.currentText()}")
        self.log(f"   Параметры: {' '.join(args)}")

        if self.bench_process.state() == QProcess.ProcessState.Running:
            self.log("⚠️ Бенчмарк уже запущен, останавливаем...")
            self.bench_process.kill()

        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Тестирование...")
        self.bench_process.start(bench_exe, args)

    def handle_bench_stdout(self):
        data = self.bench_process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        self.log(data, "bench")

    def handle_bench_stderr(self):
        data = self.bench_process.readAllStandardError().data().decode('utf-8', errors='ignore')
        self.log(data, "error")

    def handle_bench_finished(self, exit_code):
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🧪 Тестировать")
        if exit_code == 0:
            self.log("✅ Тестирование завершено успешно")
        else:
            self.log(f"❌ Ошибка тестирования (код: {exit_code})", "error")

    def start_server(self):
        exe = self.exe_path.text()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(self, "Ошибка", "Укажите путь к llama-server.exe")
            return

        args = self.build_args(for_benchmark=False)
        if not args:
            return

        self.log(f"▶ Запуск сервера: {exe}")
        self.log(f"   Аргументы: {' '.join(args)}")

        self.process.start(exe, args)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_server(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⏹ Остановка сервера...")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='ignore')
        self.log(data, "info")

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8', errors='ignore')
        self.log(data, "error")

    def handle_state(self, state):
        states = {
            QProcess.ProcessState.NotRunning: "Остановлен",
            QProcess.ProcessState.Starting: "Запуск...",
            QProcess.ProcessState.Running: "Работает"
        }
        status = states[state]
        if state == QProcess.ProcessState.NotRunning:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log(f"⏹ Сервер остановлен (код: {self.process.exitCode()})")

    def log(self, text, level="info"):
        """Добавление текста в лог с цветом"""
        cursor = self.logs.textCursor()
        fmt = QTextCharFormat()

        if level == "error":
            fmt.setForeground(QColor("#f48771"))
        elif level == "bench":
            fmt.setForeground(QColor("#4ec9b0"))  # Бирюзовый для бенчмарка
        elif "error" in text.lower() or "failed" in text.lower():
            fmt.setForeground(QColor("#f48771"))
        elif "loading model" in text.lower():
            fmt.setForeground(QColor("#4ec9b0"))
        elif "server started" in text.lower():
            fmt.setForeground(QColor("#b5cea8"))

        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, fmt)
        self.logs.setTextCursor(cursor)
        self.logs.ensureCursorVisible()

    # === Управление данными ===

    def load_data(self):
        """Загрузка настроек и профилей"""
        # Загрузка базовых настроек
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
                    self.exe_path.setText(self.settings.get("exe", ""))
                    self.bench_path.setText(self.settings.get("bench", ""))
                    self.model_dir.setText(self.settings.get("model_dir", ""))

                    # Загрузка параметров бенчмарка
                    self.bench_prompt.setValue(self.settings.get("bench_prompt", 128))
                    self.bench_gen.setValue(self.settings.get("bench_gen", 256))

                    if self.settings.get("model_dir") and os.path.exists(self.settings.get("model_dir")):
                        self.scan_models()
            except Exception as e:
                self.log(f"Ошибка загрузки настроек: {e}", "error")

        # Загрузка профилей
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    self.profiles = json.load(f)
                    self.refresh_profile_list()
            except Exception as e:
                self.log(f"Ошибка загрузки профилей: {e}", "error")

    def save_settings(self):
        """Сохранение базовых настроек"""
        self.settings = {
            "exe": self.exe_path.text(),
            "bench": self.bench_path.text(),
            "model_dir": self.model_dir.text(),
            "bench_prompt": self.bench_prompt.value(),
            "bench_gen": self.bench_gen.value()
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Ошибка сохранения настроек: {e}", "error")

    def save_profiles(self):
        try:
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Ошибка сохранения профилей: {e}", "error")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить профиль: {e}")

    def refresh_profile_list(self):
        self.profile_list.clear()
        for name in self.profiles.keys():
            self.profile_list.addItem(name)

    def save_profile(self):
        name = self.profile_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите имя профиля")
            return

        self.profiles[name] = {
            "model": self.model_combo.currentText(),
            "model_path": self.model_combo.currentData(),
            "temperature": self.temperature.value(),
            "repeat_penalty": self.repeat_penalty.value(),
            "gpu_layers": self.gpu_layers.value(),
            "ctx_size": self.ctx_size.value(),
            "threads": self.threads.value(),
            "port": self.port.value(),
            "flash_attn": self.flash_attn.isChecked(),
            "extra_args": self.extra_args.text(),
            "bench_prompt": self.bench_prompt.value(),
            "bench_gen": self.bench_gen.value()
        }
        self.save_profiles()
        self.refresh_profile_list()
        self.log(f"💾 Профиль сохранен: {name}")

    def load_profile(self, item):
        name = item.text()
        if name not in self.profiles:
            return

        p = self.profiles[name]
        self.profile_name.setText(name)

        # Установка значений
        idx = self.model_combo.findText(p.get("model", ""))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.temperature.setValue(p.get("temperature", 0.7))
        self.repeat_penalty.setValue(p.get("repeat_penalty", 1.1))
        self.gpu_layers.setValue(p.get("gpu_layers", 33))
        self.ctx_size.setValue(p.get("ctx_size", 4096))
        self.threads.setValue(p.get("threads", 4))
        self.port.setValue(p.get("port", 8080))
        self.flash_attn.setChecked(p.get("flash_attn", True))
        self.extra_args.setText(p.get("extra_args", ""))
        self.bench_prompt.setValue(p.get("bench_prompt", 128))
        self.bench_gen.setValue(p.get("bench_gen", 256))

        self.log(f"📂 Загружен профиль: {name}")

    def delete_profile(self):
        name = self.profile_name.text().strip()
        if name in self.profiles:
            del self.profiles[name]
            self.save_profiles()
            self.refresh_profile_list()
            self.profile_name.clear()
            self.log(f"🗑 Профиль удален: {name}")

    def closeEvent(self, event):
        # Сохранение настроек при выходе
        self.save_settings()

        # Остановка процессов
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            self.process.waitForFinished(2000)

        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.bench_process.terminate()
            self.bench_process.waitForFinished(2000)

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LlamaGUI()
    window.show()
    sys.exit(app.exec())