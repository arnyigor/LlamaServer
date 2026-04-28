import sys
import json
import os
import re
import shlex
import struct
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit,
    QTextEdit, QFileDialog, QGroupBox, QMessageBox, QListWidget,
    QDoubleSpinBox, QCheckBox, QProgressBar
)
from PySide6.QtCore import QProcess, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    33: "Q4_0_4_4",
    34: "Q4_0_4_8",
    35: "Q4_0_8_8",
    36: "TQ1_0",
    37: "TQ2_0",
}


def read_gguf_string(f):
    size_data = f.read(8)
    if len(size_data) != 8:
        raise ValueError("Unexpected GGUF EOF while reading string size")
    size = struct.unpack("<Q", size_data)[0]
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Unexpected GGUF EOF while reading string")
    return data.decode("utf-8", errors="replace")


def skip_gguf_value(f, value_type):
    sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if value_type == 8:
        size = struct.unpack("<Q", f.read(8))[0]
        f.seek(size, os.SEEK_CUR)
    elif value_type == 9:
        child_type = struct.unpack("<I", f.read(4))[0]
        length = struct.unpack("<Q", f.read(8))[0]
        for _ in range(length):
            skip_gguf_value(f, child_type)
    elif value_type in sizes:
        f.seek(sizes[value_type], os.SEEK_CUR)
    else:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")


def read_gguf_value(f, value_type):
    if value_type == 0:
        return struct.unpack("<B", f.read(1))[0]
    if value_type == 1:
        return struct.unpack("<b", f.read(1))[0]
    if value_type == 2:
        return struct.unpack("<H", f.read(2))[0]
    if value_type == 3:
        return struct.unpack("<h", f.read(2))[0]
    if value_type == 4:
        return struct.unpack("<I", f.read(4))[0]
    if value_type == 5:
        return struct.unpack("<i", f.read(4))[0]
    if value_type == 6:
        return struct.unpack("<f", f.read(4))[0]
    if value_type == 7:
        return struct.unpack("<?", f.read(1))[0]
    if value_type == 8:
        return read_gguf_string(f)
    if value_type == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if value_type == 11:
        return struct.unpack("<q", f.read(8))[0]
    if value_type == 12:
        return struct.unpack("<d", f.read(8))[0]
    skip_gguf_value(f, value_type)
    return None


def read_gguf_metadata(path):
    metadata = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return metadata

        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            tensor_count = struct.unpack("<I", f.read(4))[0]
            metadata_count = struct.unpack("<I", f.read(4))[0]
        else:
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_count = struct.unpack("<Q", f.read(8))[0]

        if metadata_count > 20000:
            raise ValueError(f"Suspicious GGUF metadata count: {metadata_count}")

        metadata["gguf.version"] = version
        metadata["gguf.tensor_count"] = tensor_count
        for _ in range(metadata_count):
            key = read_gguf_string(f)
            value_type = struct.unpack("<I", f.read(4))[0]
            metadata[f"{key}.type"] = GGUF_VALUE_TYPES.get(value_type, str(value_type))
            metadata[key] = read_gguf_value(f, value_type)

            arch = metadata.get("general.architecture", "")
            has_file_type = "general.file_type" in metadata
            has_context = any(
                key.endswith(".context_length") and isinstance(value, int)
                for key, value in metadata.items()
            )
            if arch and has_file_type and has_context:
                break
    return metadata


def quant_from_filename(path):
    name = Path(path).name.upper()
    match = re.search(
        r"(IQ[1-4]_[A-Z0-9_]+|TQ[12]_0|Q[2-8](?:_K)?(?:_[SML])?|Q4_0_4_[48]|Q4_0_8_8|F16|F32|BF16)",
        name,
    )
    return match.group(1) if match else ""


def detect_mmproj_for_model(path):
    model_path = Path(path)
    candidates = []
    patterns = ("*mmproj*", "*projector*")

    for directory in [model_path.parent, model_path.parent.parent]:
        if not directory.exists():
            continue
        for pattern in patterns:
            for item in directory.glob(pattern):
                if item == model_path or not item.is_file():
                    continue
                if item.suffix.lower() not in {".gguf", ".bin"}:
                    continue
                name = item.name.lower()
                if "mmproj" in name or "projector" in name:
                    candidates.append(item)

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item.parent != model_path.parent, len(item.name), item.name.lower()))
    return str(candidates[0])


def extract_model_info(path):
    file_path = Path(path)
    info = {
        "path": str(file_path),
        "name": file_path.name,
        "size_gib": round(file_path.stat().st_size / (1024 ** 3), 2) if file_path.exists() else 0,
        "architecture": "",
        "context_length": 0,
        "quant": quant_from_filename(file_path),
        "mmproj_path": detect_mmproj_for_model(file_path),
        "metadata_error": "",
    }

    try:
        metadata = read_gguf_metadata(file_path)
        arch = metadata.get("general.architecture", "")
        info["architecture"] = arch
        file_type = metadata.get("general.file_type")
        if isinstance(file_type, int) and file_type in GGUF_FILE_TYPES:
            info["quant"] = GGUF_FILE_TYPES[file_type]

        ctx_key = f"{arch}.context_length" if arch else ""
        context_length = metadata.get(ctx_key)
        if not isinstance(context_length, int):
            for key, value in metadata.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    context_length = value
                    break
        if isinstance(context_length, int):
            info["context_length"] = context_length
    except Exception as exc:
        info["metadata_error"] = str(exc)

    info["recommended_ctx"] = recommend_context(info)
    return info


def recommend_context(info):
    quant = (info.get("quant") or "").upper()
    size_gib = info.get("size_gib") or 0
    model_ctx = info.get("context_length") or 0

    if quant.startswith(("IQ1", "IQ2", "Q2")):
        recommended = 4096
    elif quant.startswith(("IQ3", "Q3")):
        recommended = 6144
    elif quant.startswith(("IQ4", "Q4")):
        recommended = 8192
    elif quant.startswith(("Q5", "IQ5")):
        recommended = 12288
    elif quant.startswith("Q6"):
        recommended = 16384
    elif quant.startswith("Q8"):
        recommended = 24576
    elif quant in {"F16", "BF16", "F32"}:
        recommended = 32768
    else:
        recommended = 8192

    if size_gib >= 24:
        recommended = min(recommended, 8192)
    elif size_gib >= 14:
        recommended = min(recommended, 12288)
    elif size_gib <= 5 and quant:
        recommended = max(recommended, 8192)

    if model_ctx:
        recommended = min(recommended, model_ctx)

    return max(512, int(recommended // 512 * 512))

class ModelScanner(QThread):
    """Сканирование папок с GGUF в отдельном потоке"""
    models_found = Signal(list)
    progress = Signal(str)

    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def run(self):
        models = []
        base = Path(self.base_path)
        if base.exists():
            for gguf_file in base.rglob("*.gguf"):
                if self.isInterruptionRequested():
                    break
                rel_path = gguf_file.relative_to(base)
                info = extract_model_info(gguf_file)
                info["display"] = str(rel_path)
                models.append(info)
                if len(models) % 25 == 0:
                    self.progress.emit(f"Найдено моделей: {len(models)}")
        models.sort(key=lambda item: item["display"].lower())
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
        self.models = []
        self.models_by_path = {}
        self.scanner = None
        self.loading_profile = False
        self.server_stop_requested = False
        self.bench_stop_requested = False
        self.scan_cancel_requested = False

        self.setup_ui()
        self.load_data()
        QTimer.singleShot(250, self.auto_scan_models)

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
        self.scan_btn = QPushButton("🔍 Сканировать")
        self.scan_btn.clicked.connect(self.scan_models)
        model_dir_layout.addWidget(self.model_dir)
        model_dir_layout.addWidget(model_dir_btn)
        model_dir_layout.addWidget(self.scan_btn)
        path_layout.addLayout(model_dir_layout)

        self.scan_status = QLabel("Модели не сканировались")
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setVisible(False)
        path_layout.addWidget(self.scan_status)
        path_layout.addWidget(self.scan_progress)

        left_layout.addWidget(path_group)

        # Выбор модели
        model_group = QGroupBox("Модель")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(QLabel("Найденные GGUF:"))
        model_layout.addWidget(self.model_combo)

        self.auto_params = QCheckBox("Автонастройка ctx/GPU/cache по GGUF")
        self.auto_params.setChecked(True)
        model_layout.addWidget(self.auto_params)

        mmproj_layout = QHBoxLayout()
        self.use_mmproj = QCheckBox("Использовать mmproj, если найден")
        self.use_mmproj.setChecked(True)
        self.mmproj_offload = QCheckBox("mmproj offload")
        self.mmproj_offload.setChecked(True)
        mmproj_layout.addWidget(self.use_mmproj)
        mmproj_layout.addWidget(self.mmproj_offload)
        model_layout.addLayout(mmproj_layout)

        self.model_info = QLabel("Выберите модель")
        self.model_info.setWordWrap(True)
        model_layout.addWidget(self.model_info)

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
        self.gpu_auto = QCheckBox("auto")
        self.gpu_auto.setChecked(True)
        self.gpu_auto.toggled.connect(self.gpu_layers.setDisabled)
        self.gpu_layers.setDisabled(True)
        gpu_layout.addWidget(self.gpu_layers)
        gpu_layout.addWidget(self.gpu_auto)
        params_layout.addLayout(gpu_layout)

        # Context Size
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Size (-c):"))
        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(512, 1048576)
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

        # Параметры загрузки и памяти
        runtime_group = QGroupBox("Память и сервер")
        runtime_layout = QVBoxLayout(runtime_group)

        memory_flags_layout = QHBoxLayout()
        self.use_mmap = QCheckBox("mmap")
        self.use_mmap.setChecked(True)
        self.use_mlock = QCheckBox("mlock")
        self.verbose = QCheckBox("verbose")
        self.log_timestamps = QCheckBox("log timestamps")
        memory_flags_layout.addWidget(self.use_mmap)
        memory_flags_layout.addWidget(self.use_mlock)
        memory_flags_layout.addWidget(self.verbose)
        memory_flags_layout.addWidget(self.log_timestamps)
        runtime_layout.addLayout(memory_flags_layout)

        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("KV cache K/V:"))
        self.cache_type_k = QComboBox()
        self.cache_type_v = QComboBox()
        for cache_type in ["f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "f32"]:
            self.cache_type_k.addItem(cache_type)
            self.cache_type_v.addItem(cache_type)
        self.cache_type_k.setCurrentText("f16")
        self.cache_type_v.setCurrentText("f16")
        cache_layout.addWidget(self.cache_type_k)
        cache_layout.addWidget(self.cache_type_v)
        runtime_layout.addLayout(cache_layout)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch/ubatch:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(128, 32768)
        self.batch_size.setSingleStep(128)
        self.batch_size.setValue(2048)
        self.ubatch_size = QSpinBox()
        self.ubatch_size.setRange(64, 8192)
        self.ubatch_size.setSingleStep(64)
        self.ubatch_size.setValue(512)
        batch_layout.addWidget(self.batch_size)
        batch_layout.addWidget(self.ubatch_size)
        runtime_layout.addLayout(batch_layout)

        server_flags_layout = QHBoxLayout()
        server_flags_layout.addWidget(QLabel("Slots:"))
        self.parallel_slots = QSpinBox()
        self.parallel_slots.setRange(1, 16)
        self.parallel_slots.setValue(1)
        self.cont_batching = QCheckBox("cont batching")
        self.cont_batching.setChecked(True)
        self.cache_prompt = QCheckBox("cache prompt")
        self.cache_prompt.setChecked(True)
        server_flags_layout.addWidget(self.parallel_slots)
        server_flags_layout.addWidget(self.cont_batching)
        server_flags_layout.addWidget(self.cache_prompt)
        runtime_layout.addLayout(server_flags_layout)

        extra_flags_layout = QHBoxLayout()
        self.context_shift = QCheckBox("context shift")
        self.no_webui = QCheckBox("no webui")
        extra_flags_layout.addWidget(self.context_shift)
        extra_flags_layout.addWidget(self.no_webui)
        runtime_layout.addLayout(extra_flags_layout)

        params_layout.addWidget(runtime_group)

        # Доп. параметры
        params_layout.addWidget(QLabel("Доп. параметры:"))
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("--top-p 0.9 --min-p 0.05 --rope-scaling yarn ...")
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
        self.stop_btn.clicked.connect(self.stop_work)
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

    def auto_scan_models(self):
        if self.models:
            self.scan_status.setText(f"Кэш моделей: {len(self.models)}. Фоновая проверка...")
        base_path = self.model_dir.text()
        if base_path and os.path.exists(base_path):
            self.scan_models(silent=True)

    def scan_models(self, silent=False):
        base_path = self.model_dir.text()
        if not base_path or not os.path.exists(base_path):
            if not silent:
                QMessageBox.warning(self, "Ошибка", "Укажите существующую базовую папку")
            return

        if self.scanner and self.scanner.isRunning():
            if not silent:
                self.cancel_scan()
            return

        self.scan_cancel_requested = False
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("⏹ Отменить")
        self.scan_progress.setVisible(True)
        self.scan_status.setText("Сканирование GGUF...")
        self.update_action_buttons()
        if not silent:
            self.log("🔍 Сканирование моделей...")
        self.scanner = ModelScanner(base_path)
        self.scanner.progress.connect(self.on_scan_progress)
        self.scanner.models_found.connect(self.on_models_found)
        self.scanner.finished.connect(self.on_scan_finished)
        self.scanner.start()

    def cancel_scan(self):
        if self.scanner and self.scanner.isRunning():
            self.scan_cancel_requested = True
            self.scan_status.setText("Отмена сканирования...")
            self.log("⏹ Отмена сканирования моделей...")
            self.scanner.requestInterruption()
            self.scan_btn.setEnabled(False)

    def on_scan_progress(self, text):
        self.scan_status.setText(text)

    def on_models_found(self, models):
        if self.scan_cancel_requested:
            return

        current_path = self.model_combo.currentData() or self.settings.get("last_model_path", "")
        self.models = models
        self.models_by_path = {item["path"]: item for item in models}
        self.model_combo.clear()
        for item in models:
            self.model_combo.addItem(item["display"], item["path"])

        if current_path:
            idx = self.model_combo.findData(current_path)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        elif models:
            self.model_combo.setCurrentIndex(0)

        self.save_settings()
        self.log(f"✅ Найдено моделей: {len(models)}")
        self.scan_status.setText(f"Найдено моделей: {len(models)}")

    def on_scan_finished(self):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("🔍 Сканировать")
        self.scan_progress.setVisible(False)
        if self.scan_cancel_requested:
            self.scan_status.setText("Сканирование отменено")
            self.scan_cancel_requested = False
        self.update_action_buttons()

    def on_model_selected(self, *_):
        model_path = self.model_combo.currentData()
        if not model_path:
            self.model_info.setText("Выберите модель")
            return

        info = self.models_by_path.get(model_path)
        if not info:
            info = extract_model_info(model_path)
            info["display"] = self.model_combo.currentText()
            self.models_by_path[model_path] = info

        max_ctx = info.get("context_length") or "не указан"
        quant = info.get("quant") or "не определено"
        arch = info.get("architecture") or "не определено"
        size_gib = info.get("size_gib", 0)
        recommended_ctx = info.get("recommended_ctx", 4096)
        mmproj = info.get("mmproj_path") or "не найден"
        error = f"\nMetadata: {info['metadata_error']}" if info.get("metadata_error") else ""
        self.model_info.setText(
            f"Архитектура: {arch}; квант: {quant}; размер: {size_gib} GiB; "
            f"ctx модели: {max_ctx}; рекомендовано: {recommended_ctx}; "
            f"mmproj: {mmproj}{error}"
        )

        self.settings["last_model_path"] = model_path
        if self.auto_params.isChecked() and not self.loading_profile:
            self.apply_recommended_params(info)

    def apply_recommended_params(self, info):
        recommended_ctx = info.get("recommended_ctx")
        if recommended_ctx:
            self.ctx_size.setValue(recommended_ctx)

        quant = (info.get("quant") or "").upper()
        if quant.startswith(("Q2", "Q3", "IQ1", "IQ2", "IQ3")):
            self.cache_type_k.setCurrentText("q8_0")
            self.cache_type_v.setCurrentText("q8_0")
        elif info.get("recommended_ctx", 0) >= 16384:
            self.cache_type_k.setCurrentText("q8_0")
            self.cache_type_v.setCurrentText("q8_0")
        else:
            self.cache_type_k.setCurrentText("f16")
            self.cache_type_v.setCurrentText("f16")

        if self.gpu_auto.isChecked():
            self.gpu_layers.setDisabled(True)

        if info.get("recommended_ctx", 0) >= 16384:
            self.batch_size.setValue(1024)
            self.ubatch_size.setValue(256)
        else:
            self.batch_size.setValue(2048)
            self.ubatch_size.setValue(512)

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
            args.extend(["-ngl", self.gpu_layers_arg(for_benchmark=True)])
            if self.flash_attn.isChecked():
                args.extend(["-fa", "1"])
            args.extend(["-ctk", self.cache_type_k.currentText()])
            args.extend(["-ctv", self.cache_type_v.currentText()])
            args.extend(["-b", str(self.batch_size.value())])
            args.extend(["-ub", str(min(self.ubatch_size.value(), self.batch_size.value()))])
        else:
            # Для сервера
            args.extend(["-m", model_path])
            args.extend(["--port", str(self.port.value())])
            args.extend(["-ngl", self.gpu_layers_arg()])
            args.extend(["-c", str(self.ctx_size.value())])
            args.extend(["-t", str(self.threads.value())])
            args.extend(["-b", str(self.batch_size.value())])
            args.extend(["-ub", str(min(self.ubatch_size.value(), self.batch_size.value()))])
            args.extend(["-ctk", self.cache_type_k.currentText()])
            args.extend(["-ctv", self.cache_type_v.currentText()])
            args.extend(["-np", str(self.parallel_slots.value())])

            # Параметры сэмплирования для сервера
            args.extend(["--temp", str(self.temperature.value())])
            args.extend(["--repeat-penalty", str(self.repeat_penalty.value())])

            if self.flash_attn.isChecked():
                args.extend(["--flash-attn", "on"])

            model_info = self.models_by_path.get(model_path) or {}
            mmproj_path = model_info.get("mmproj_path", "")
            if self.use_mmproj.isChecked():
                if mmproj_path:
                    args.extend(["-mm", mmproj_path])
                if not self.mmproj_offload.isChecked():
                    args.append("--no-mmproj-offload")
            else:
                args.append("--no-mmproj")

            if self.use_mmap.isChecked():
                args.append("--mmap")
            else:
                args.append("--no-mmap")
            if self.use_mlock.isChecked():
                args.append("--mlock")
            if self.verbose.isChecked():
                args.append("--verbose")
            if self.log_timestamps.isChecked():
                args.append("--log-timestamps")
            if not self.cont_batching.isChecked():
                args.append("--no-cont-batching")
            if not self.cache_prompt.isChecked():
                args.append("--no-cache-prompt")
            if self.context_shift.isChecked():
                args.append("--context-shift")
            if self.no_webui.isChecked():
                args.append("--no-webui")

            # Доп. аргументы
            if self.extra_args.text():
                try:
                    args.extend(shlex.split(self.extra_args.text()))
                except ValueError as exc:
                    QMessageBox.warning(self, "Ошибка", f"Не удалось разобрать доп. параметры: {exc}")
                    return None

        return args

    def gpu_layers_arg(self, for_benchmark=False):
        if not self.gpu_auto.isChecked():
            return str(self.gpu_layers.value())

        # llama-server supports "auto"; llama-bench in current builds expects a numeric range/value.
        return "99" if for_benchmark else "auto"

    def run_benchmark(self):
        """Запуск llama-bench с текущими параметрами"""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Сервер запущен", "Остановите сервер перед запуском benchmark")
            return

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
            self.stop_benchmark()
            return

        self.bench_stop_requested = False
        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Тестирование...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
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
        self.start_btn.setEnabled(self.process.state() == QProcess.ProcessState.NotRunning)
        self.update_action_buttons()
        if self.bench_stop_requested:
            self.log("⏹ Тестирование остановлено")
        elif exit_code == 0:
            self.log("✅ Тестирование завершено успешно")
        else:
            self.log(f"❌ Ошибка тестирования (код: {exit_code})", "error")
        self.bench_stop_requested = False

    def stop_benchmark(self):
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.bench_stop_requested = True
            self.log("⏹ Остановка benchmark...")
            self.bench_process.terminate()
            QTimer.singleShot(2500, self.kill_benchmark_if_running)

    def kill_benchmark_if_running(self):
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Benchmark не завершился штатно, принудительная остановка")
            self.bench_process.kill()

    def start_server(self):
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Benchmark запущен", "Остановите benchmark перед запуском сервера")
            return

        exe = self.exe_path.text()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(self, "Ошибка", "Укажите путь к llama-server.exe")
            return

        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Сервер уже запущен")
            return

        args = self.build_args(for_benchmark=False)
        if not args:
            return

        self.log(f"▶ Запуск сервера: {exe}")
        self.log(f"   Аргументы: {' '.join(args)}")

        self.server_stop_requested = False
        self.process.start(exe, args)
        self.start_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_work(self):
        stopped_any = False
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.stop_server()
            stopped_any = True
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.stop_benchmark()
            stopped_any = True
        if self.scanner and self.scanner.isRunning():
            self.cancel_scan()
            stopped_any = True
        if not stopped_any:
            self.update_action_buttons()

    def stop_server(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.server_stop_requested = True
            self.log("⏹ Остановка сервера...")
            self.process.terminate()
            QTimer.singleShot(3000, self.kill_server_if_running)

    def kill_server_if_running(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Сервер не завершился штатно, принудительная остановка")
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
            self.test_btn.setEnabled(self.bench_process.state() == QProcess.ProcessState.NotRunning)
            self.update_action_buttons()
            if self.server_stop_requested:
                self.log("⏹ Сервер остановлен")
            else:
                self.log(f"⏹ Сервер остановлен (код: {self.process.exitCode()})")
            self.server_stop_requested = False

    def update_action_buttons(self):
        server_running = self.process.state() != QProcess.ProcessState.NotRunning
        bench_running = self.bench_process.state() != QProcess.ProcessState.NotRunning
        scan_running = self.scanner is not None and self.scanner.isRunning()
        busy = server_running or bench_running or scan_running
        self.stop_btn.setEnabled(busy)
        if not server_running and not bench_running:
            self.start_btn.setEnabled(True)
            self.test_btn.setEnabled(True)

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

                    # Загрузка параметров UI без запуска тяжелого сканирования.
                    self.bench_prompt.setValue(self.settings.get("bench_prompt", 128))
                    self.bench_gen.setValue(self.settings.get("bench_gen", 256))
                    self.auto_params.setChecked(self.settings.get("auto_params", True))
                    self.use_mmproj.setChecked(self.settings.get("use_mmproj", True))
                    self.mmproj_offload.setChecked(self.settings.get("mmproj_offload", True))
                    self.gpu_auto.setChecked(self.settings.get("gpu_auto", True))
                    self.gpu_layers.setValue(self.settings.get("gpu_layers", 33))
                    self.ctx_size.setValue(self.settings.get("ctx_size", 4096))
                    self.threads.setValue(self.settings.get("threads", os.cpu_count() or 4))
                    self.port.setValue(self.settings.get("port", 8080))
                    self.temperature.setValue(self.settings.get("temperature", 0.7))
                    self.repeat_penalty.setValue(self.settings.get("repeat_penalty", 1.1))
                    self.flash_attn.setChecked(self.settings.get("flash_attn", True))
                    self.use_mmap.setChecked(self.settings.get("use_mmap", True))
                    self.use_mlock.setChecked(self.settings.get("use_mlock", False))
                    self.verbose.setChecked(self.settings.get("verbose", False))
                    self.log_timestamps.setChecked(self.settings.get("log_timestamps", False))
                    self.cache_type_k.setCurrentText(self.settings.get("cache_type_k", "f16"))
                    self.cache_type_v.setCurrentText(self.settings.get("cache_type_v", "f16"))
                    self.batch_size.setValue(self.settings.get("batch_size", 2048))
                    self.ubatch_size.setValue(self.settings.get("ubatch_size", 512))
                    self.parallel_slots.setValue(self.settings.get("parallel_slots", 1))
                    self.cont_batching.setChecked(self.settings.get("cont_batching", True))
                    self.cache_prompt.setChecked(self.settings.get("cache_prompt", True))
                    self.context_shift.setChecked(self.settings.get("context_shift", False))
                    self.no_webui.setChecked(self.settings.get("no_webui", False))
                    self.extra_args.setText(self.settings.get("extra_args", ""))

                    cached_models = self.settings.get("model_cache", [])
                    if cached_models:
                        self.on_models_found(cached_models)
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
            "bench_gen": self.bench_gen.value(),
            "auto_params": self.auto_params.isChecked(),
            "use_mmproj": self.use_mmproj.isChecked(),
            "mmproj_offload": self.mmproj_offload.isChecked(),
            "last_model_path": self.model_combo.currentData() or self.settings.get("last_model_path", ""),
            "model_cache": self.models,
            "temperature": self.temperature.value(),
            "repeat_penalty": self.repeat_penalty.value(),
            "use_mmproj": self.use_mmproj.isChecked(),
            "mmproj_offload": self.mmproj_offload.isChecked(),
            "gpu_auto": self.gpu_auto.isChecked(),
            "gpu_layers": self.gpu_layers.value(),
            "ctx_size": self.ctx_size.value(),
            "threads": self.threads.value(),
            "port": self.port.value(),
            "flash_attn": self.flash_attn.isChecked(),
            "use_mmap": self.use_mmap.isChecked(),
            "use_mlock": self.use_mlock.isChecked(),
            "verbose": self.verbose.isChecked(),
            "log_timestamps": self.log_timestamps.isChecked(),
            "cache_type_k": self.cache_type_k.currentText(),
            "cache_type_v": self.cache_type_v.currentText(),
            "batch_size": self.batch_size.value(),
            "ubatch_size": self.ubatch_size.value(),
            "parallel_slots": self.parallel_slots.value(),
            "cont_batching": self.cont_batching.isChecked(),
            "cache_prompt": self.cache_prompt.isChecked(),
            "context_shift": self.context_shift.isChecked(),
            "no_webui": self.no_webui.isChecked(),
            "extra_args": self.extra_args.text()
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
            "gpu_auto": self.gpu_auto.isChecked(),
            "gpu_layers": self.gpu_layers.value(),
            "ctx_size": self.ctx_size.value(),
            "threads": self.threads.value(),
            "port": self.port.value(),
            "flash_attn": self.flash_attn.isChecked(),
            "use_mmap": self.use_mmap.isChecked(),
            "use_mlock": self.use_mlock.isChecked(),
            "verbose": self.verbose.isChecked(),
            "log_timestamps": self.log_timestamps.isChecked(),
            "cache_type_k": self.cache_type_k.currentText(),
            "cache_type_v": self.cache_type_v.currentText(),
            "batch_size": self.batch_size.value(),
            "ubatch_size": self.ubatch_size.value(),
            "parallel_slots": self.parallel_slots.value(),
            "cont_batching": self.cont_batching.isChecked(),
            "cache_prompt": self.cache_prompt.isChecked(),
            "context_shift": self.context_shift.isChecked(),
            "no_webui": self.no_webui.isChecked(),
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
        self.loading_profile = True

        # Установка значений
        idx = self.model_combo.findData(p.get("model_path", ""))
        if idx < 0:
            idx = self.model_combo.findText(p.get("model", ""))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.temperature.setValue(p.get("temperature", 0.7))
        self.repeat_penalty.setValue(p.get("repeat_penalty", 1.1))
        self.use_mmproj.setChecked(p.get("use_mmproj", True))
        self.mmproj_offload.setChecked(p.get("mmproj_offload", True))
        self.gpu_auto.setChecked(p.get("gpu_auto", True))
        self.gpu_layers.setValue(p.get("gpu_layers", 33))
        self.ctx_size.setValue(p.get("ctx_size", 4096))
        self.threads.setValue(p.get("threads", 4))
        self.port.setValue(p.get("port", 8080))
        self.flash_attn.setChecked(p.get("flash_attn", True))
        self.use_mmap.setChecked(p.get("use_mmap", True))
        self.use_mlock.setChecked(p.get("use_mlock", False))
        self.verbose.setChecked(p.get("verbose", False))
        self.log_timestamps.setChecked(p.get("log_timestamps", False))
        self.cache_type_k.setCurrentText(p.get("cache_type_k", "f16"))
        self.cache_type_v.setCurrentText(p.get("cache_type_v", "f16"))
        self.batch_size.setValue(p.get("batch_size", 2048))
        self.ubatch_size.setValue(p.get("ubatch_size", 512))
        self.parallel_slots.setValue(p.get("parallel_slots", 1))
        self.cont_batching.setChecked(p.get("cont_batching", True))
        self.cache_prompt.setChecked(p.get("cache_prompt", True))
        self.context_shift.setChecked(p.get("context_shift", False))
        self.no_webui.setChecked(p.get("no_webui", False))
        self.extra_args.setText(p.get("extra_args", ""))
        self.bench_prompt.setValue(p.get("bench_prompt", 128))
        self.bench_gen.setValue(p.get("bench_gen", 256))
        self.loading_profile = False
        self.on_model_selected()

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

        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            self.bench_process.terminate()

        if self.scanner and self.scanner.isRunning():
            self.scanner.requestInterruption()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LlamaGUI()
    window.show()
    sys.exit(app.exec())
