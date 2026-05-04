import sys
import json
import os
import re
import shutil
import shlex
import struct
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QMessageBox,
    QListWidget,
    QDoubleSpinBox,
    QCheckBox,
    QProgressBar,
    QScrollArea,
)
from PySide6.QtCore import QProcess, Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest


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

    candidates.sort(
        key=lambda item: (
            item.parent != model_path.parent,
            len(item.name),
            item.name.lower(),
        )
    )
    return str(candidates[0])


def extract_model_info(path):
    file_path = Path(path)
    info = {
        "path": str(file_path),
        "name": file_path.name,
        "size_gib": round(file_path.stat().st_size / (1024**3), 2)
        if file_path.exists()
        else 0,
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


class LlamaCppUpdater(QThread):
    progress = Signal(str)
    percent = Signal(int)
    completed = Signal(bool, str)

    API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

    def __init__(self, server_path):
        super().__init__()
        self.server_path = Path(server_path)

    def run(self):
        try:
            if not self.server_path.exists():
                raise FileNotFoundError(f"llama-server.exe not found: {self.server_path}")

            target_dir = self.server_path.parent
            current_build = self.get_current_build()
            release = self.fetch_latest_release()
            latest_build = self.parse_build_number(release.get("tag_name", ""))
            if latest_build is None:
                raise RuntimeError(f"Cannot parse release tag: {release.get('tag_name')}")

            current_text = current_build if current_build is not None else "unknown"
            self.progress.emit(
                f"llama.cpp local build: {current_text}, latest: {latest_build}"
            )
            if current_build is not None and current_build >= latest_build:
                self.percent.emit(100)
                self.completed.emit(False, f"Already up to date: build {current_build}")
                return

            assets = self.select_assets(release)
            if not assets:
                raise RuntimeError("No Windows x64 CUDA 12.4 release asset found")

            with tempfile.TemporaryDirectory(prefix="llamacpp-update-") as temp_dir:
                temp_path = Path(temp_dir)
                extract_dir = temp_path / "extract"
                extract_dir.mkdir()

                for index, asset in enumerate(assets, start=1):
                    name = asset["name"]
                    archive_path = temp_path / name
                    self.progress.emit(f"Downloading {name} ({index}/{len(assets)})")
                    self.download(asset["browser_download_url"], archive_path)
                    self.progress.emit(f"Extracting {name}")
                    self.safe_extract_zip(archive_path, extract_dir)

                install_root = self.find_install_root(extract_dir)
                if not (install_root / "llama-server.exe").exists():
                    raise RuntimeError("Downloaded archive does not contain llama-server.exe")

                self.progress.emit(f"Installing into {target_dir}")
                self.copy_tree_contents(install_root, target_dir)

            self.percent.emit(100)
            self.completed.emit(True, f"Updated llama.cpp to build {latest_build}")
        except Exception as exc:
            self.completed.emit(False, f"Update failed: {exc}")

    def get_current_build(self):
        try:
            result = subprocess.run(
                [str(self.server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(self.server_path.parent),
                check=False,
            )
        except Exception as exc:
            self.progress.emit(f"Cannot read local version: {exc}")
            return None

        text = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"version:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def fetch_latest_release(self):
        request = urllib.request.Request(
            self.API_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "LlamaServerGUI",
            },
        )
        self.progress.emit("Checking latest llama.cpp release")
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def parse_build_number(self, value):
        match = re.search(r"b?(\d+)", value or "")
        return int(match.group(1)) if match else None

    def select_assets(self, release):
        assets = release.get("assets", [])
        pattern = re.compile(r"^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$")
        for asset in assets:
            if pattern.match(asset.get("name", "")):
                return [asset]
        return []

    def download(self, url, destination):
        request = urllib.request.Request(url, headers={"User-Agent": "LlamaServerGUI"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(destination, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        self.percent.emit(min(99, int(done * 100 / total)))

    def safe_extract_zip(self, archive_path, destination):
        destination = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe zip entry: {member.filename}")
            archive.extractall(destination)

    def find_install_root(self, extract_dir):
        candidates = sorted(extract_dir.rglob("llama-server.exe"))
        return candidates[0].parent if candidates else extract_dir

    def copy_tree_contents(self, source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


class LlamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLama.cpp GUI Manager")
        self.setGeometry(100, 100, 1150, 720)
        self.setMinimumSize(900, 560)

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
        self.updater = None
        self.loading_profile = False
        self.server_stop_requested = False
        self.bench_stop_requested = False
        self.scan_cancel_requested = False
        self.prompt_speed = None
        self.generation_speed = None
        self.tokens_cached = None
        self.kv_cache_tokens = None
        self.kv_cache_usage_ratio = None
        self.effective_ctx_size = None
        self.decode_last_by_task = {}
        self.decode_speed_ema = None
        self.stats_failures = 0
        self.server_ready = False
        self.stats_manager = QNetworkAccessManager(self)
        self.stats_manager.finished.connect(self.handle_stats_reply)
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(2000)
        self.stats_timer.timeout.connect(self.poll_server_stats)

        self.setup_ui()
        self.load_data()
        QTimer.singleShot(250, self.auto_scan_models)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Левая панель (управление)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # Группа путей
        path_group = QGroupBox("Пути")
        path_layout = QVBoxLayout(path_group)

        # Путь к llama-server.exe
        exe_layout = QHBoxLayout()
        self.exe_path = QLineEdit()
        self.exe_path.setPlaceholderText("Путь к llama-server.exe")
        self.exe_path.textChanged.connect(
            self.auto_detect_bench
        )  # Автоопределение bench
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

        update_layout = QHBoxLayout()
        self.update_llama_btn = QPushButton("Update llama.cpp")
        self.update_llama_btn.clicked.connect(self.update_llamacpp)
        self.update_status = QLabel("llama.cpp updater idle")
        self.update_status.setWordWrap(True)
        self.update_progress = QProgressBar()
        self.update_progress.setRange(0, 100)
        self.update_progress.setVisible(False)
        update_layout.addWidget(self.update_llama_btn)
        update_layout.addWidget(self.update_status, 1)
        path_layout.addLayout(update_layout)
        path_layout.addWidget(self.update_progress)

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

        # Кнопки управления
        btn_layout = QHBoxLayout()

        self.test_btn = QPushButton("🧪 Тестировать")
        self.test_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold;"
        )
        self.test_btn.clicked.connect(self.run_benchmark)

        self.start_btn = QPushButton("▶ Старт Server")
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self.start_btn.clicked.connect(self.start_server)

        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_work)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.test_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

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
        for cache_type in [
            "f16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
            "f32",
        ]:
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
        self.server_metrics = QCheckBox("metrics")
        self.server_metrics.setChecked(True)
        extra_flags_layout.addWidget(self.context_shift)
        extra_flags_layout.addWidget(self.no_webui)
        extra_flags_layout.addWidget(self.server_metrics)
        runtime_layout.addLayout(extra_flags_layout)

        params_layout.addWidget(runtime_group)

        # Доп. параметры
        params_layout.addWidget(QLabel("Доп. параметры:"))
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText(
            "--top-p 0.9 --min-p 0.05 --rope-scaling yarn ..."
        )
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

        # Скрытые виджеты оставлены для совместимости существующей логики профилей.
        self.profile_name = QLineEdit()
        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.load_profile)

        left_layout.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(390)
        left_scroll.setMaximumWidth(520)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(left_scroll, 0)

        # Правая панель (логи)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Логи:"))
        self.speed_label = QLabel("Скорость: нет данных")
        self.speed_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        log_header.addWidget(self.speed_label, 1)
        self.context_label = QLabel("Контекст: нет данных")
        self.context_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        log_header.addWidget(self.context_label)
        self.autoscroll_logs = QCheckBox("Автоскролл")
        self.autoscroll_logs.setChecked(True)
        log_header.addWidget(self.autoscroll_logs)
        right_layout.addLayout(log_header)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 9))
        self.logs.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        right_layout.addWidget(self.logs)

        clear_btn = QPushButton("🧹 Очистить логи")
        clear_btn.clicked.connect(self.logs.clear)
        right_layout.addWidget(clear_btn)

        main_layout.addWidget(right_panel, 2)
        self.setup_tooltips()

    def setup_tooltips(self):
        tips = {
            self.exe_path: "Путь к llama-server.exe. Это основной сервер llama.cpp, который будет слушать API-порт.",
            self.bench_path: "Путь к llama-bench.exe. Используется только для теста скорости prompt processing и generation.",
            self.model_dir: "Корневая папка, где GUI рекурсивно ищет .gguf модели и mmproj/projector файлы.",
            self.scan_btn: "Сканирует папку моделей в фоне. Во время сканирования эта кнопка отменяет обход.",
            self.scan_status: "Статус фонового сканирования моделей.",
            self.scan_progress: "Индикатор активного фонового сканирования.",
            self.model_combo: "Выбранная GGUF-модель. Можно выбрать найденную модель или ввести путь вручную.",
            self.auto_params: "Автоматически выставляет примерный ctx, KV cache и batch по GGUF metadata, кванту и размеру модели.",
            self.use_mmproj: "Для multimodal/vision моделей добавляет -mm с найденным projector-файлом. Отключите, если запускаете только текст.",
            self.mmproj_offload: "Разрешает offload mmproj/projector части. Если выключить, будет добавлен --no-mmproj-offload.",
            self.model_info: "Краткая информация из GGUF metadata: архитектура, квант, размер, контекст и найденный mmproj.",
            self.temperature: "Sampling temperature. Ниже - более предсказуемо, выше - разнообразнее.",
            self.repeat_penalty: "Штраф повторов. Обычно 1.05-1.20; слишком высокое значение может портить стиль ответа.",
            self.gpu_layers: "Сколько слоев весов модели выгружать на GPU. Влияет на VRAM и скорость, но не задает размер контекста.",
            self.gpu_auto: "Для server передает -ngl auto. Для bench используется 99, потому что llama-bench не принимает auto.",
            self.ctx_size: "Размер контекста. Большие значения резко увеличивают KV cache и могут вызвать OOM уже при старте.",
            self.threads: "CPU-потоки для llama.cpp. Полезно для CPU-части и если часть модели/кэша не на GPU.",
            self.port: "HTTP-порт llama-server. OpenAI-compatible API будет доступен на 127.0.0.1:<port>.",
            self.flash_attn: "Flash Attention. Обычно ускоряет и уменьшает память на поддерживаемых GPU/backend.",
            self.use_mmap: "Memory mapping модели. Обычно включен по умолчанию и ускоряет/упрощает загрузку больших файлов.",
            self.use_mlock: "Пытается удержать модель в RAM. Используйте только если достаточно памяти.",
            self.verbose: "Подробный лог llama.cpp. Полезно для диагностики, но увеличивает поток логов.",
            self.log_timestamps: "Добавляет timestamps в лог llama.cpp, если сборка поддерживает этот флаг.",
            self.cache_type_k: "Тип KV cache для key. q8_0/q4_0 экономят память, f16 обычно точнее.",
            self.cache_type_v: "Тип KV cache для value. q8_0/q4_0 экономят память, f16 обычно точнее.",
            self.batch_size: "Batch size для обработки prompt. Больше может ускорить prompt, но повышает пиковое потребление памяти.",
            self.ubatch_size: "Micro-batch size. Уменьшайте при OOM или нестабильности на GPU.",
            self.parallel_slots: "Количество server slots. Каждый слот увеличивает потребление KV cache.",
            self.cont_batching: "Continuous batching. Обычно полезно для сервера с несколькими запросами.",
            self.cache_prompt: "Prompt cache. Может ускорять повторное использование общего префикса.",
            self.context_shift: "Context shift позволяет серверу продолжать работу при заполнении контекста, сдвигая старые токены.",
            self.no_webui: "Отключает встроенный Web UI llama-server, оставляя API.",
            self.server_metrics: "Включает --metrics. GUI начнет опрашивать /metrics только после полной загрузки сервера.",
            self.extra_args: "Дополнительные параметры llama.cpp. Разбираются с учетом кавычек через shlex.",
            self.bench_prompt: "Количество prompt-токенов для llama-bench (-p). Используется только в тесте.",
            self.bench_gen: "Количество генерируемых токенов для llama-bench (-n). Используется только в тесте.",
            self.profile_name: "Имя профиля для сохранения текущих настроек.",
            self.profile_list: "Список сохраненных профилей. Клик по профилю загружает его параметры.",
            self.test_btn: "Запускает llama-bench с текущей моделью и параметрами. Можно остановить кнопкой Стоп.",
            self.start_btn: "Запускает llama-server с текущими параметрами.",
            self.stop_btn: "Останавливает активный server, benchmark или сканирование без блокировки UI.",
            self.speed_label: "Последняя найденная скорость prompt/gen из строк llama.cpp с tok/s.",
            self.context_label: "Заполненность контекста по tokens_cached из JSON-ответов llama-server, если такие строки попадают в лог.",
            self.autoscroll_logs: "Если включено, лог всегда прокручивается к новым строкам. Отключите для чтения старого вывода.",
            self.logs: "stdout/stderr активного процесса. Цветом выделяются ошибки и benchmark.",
        }
        tips[self.update_llama_btn] = (
            "Checks the latest llama.cpp GitHub release and installs Windows x64 "
            "CUDA 12.4 build into the current llama-server.exe directory."
        )
        tips[self.update_status] = "llama.cpp updater status."
        tips[self.update_progress] = (
            "Download progress for the current llama.cpp archive."
        )
        for widget, text in tips.items():
            widget.setToolTip(text)

    def add_tooltip(self, widget, text):
        widget.setToolTip(text)

    def update_llamacpp(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "llama.cpp updater",
                "Stop llama-server before updating llama.cpp.",
            )
            return
        if self.bench_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "llama.cpp updater",
                "Stop benchmark before updating llama.cpp.",
            )
            return
        if self.updater and self.updater.isRunning():
            return

        exe = self.exe_path.text().strip()
        if not exe or not os.path.exists(exe):
            QMessageBox.critical(
                self, "llama.cpp updater", "Select an existing llama-server.exe first."
            )
            return

        self.update_progress.setValue(0)
        self.update_progress.setVisible(True)
        self.update_status.setText("Checking release...")
        self.log("llama.cpp update: checking latest release\n")
        self.updater = LlamaCppUpdater(exe)
        self.updater.progress.connect(self.on_update_progress)
        self.updater.percent.connect(self.update_progress.setValue)
        self.updater.completed.connect(self.on_update_completed)
        self.updater.finished.connect(self.on_update_thread_finished)
        self.updater.start()
        self.update_action_buttons()

    def on_update_progress(self, text):
        self.update_status.setText(text)
        self.log(f"llama.cpp update: {text}\n")

    def on_update_completed(self, changed, message):
        self.update_status.setText(message)
        level = "error" if "failed" in message.lower() else "info"
        self.log(f"llama.cpp update: {message}\n", level)
        if changed:
            self.auto_detect_bench()
            self.save_settings()

    def on_update_thread_finished(self):
        self.update_progress.setVisible(False)
        self.update_action_buttons()

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
        file, _ = QFileDialog.getOpenFileName(
            self, "Выберите llama-server", "", "Executable (*.exe)"
        )
        if file:
            self.exe_path.setText(file)
            self.save_settings()

    def browse_bench(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Выберите llama-bench", "", "Executable (*.exe)"
        )
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
            self.scan_status.setText(
                f"Кэш моделей: {len(self.models)}. Фоновая проверка..."
            )
        base_path = self.model_dir.text()
        if base_path and os.path.exists(base_path):
            self.scan_models(silent=True)

    def scan_models(self, silent=False):
        base_path = self.model_dir.text()
        if not base_path or not os.path.exists(base_path):
            if not silent:
                QMessageBox.warning(
                    self, "Ошибка", "Укажите существующую базовую папку"
                )
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

        current_path = self.model_combo.currentData() or self.settings.get(
            "last_model_path", ""
        )
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
        error = (
            f"\nMetadata: {info['metadata_error']}"
            if info.get("metadata_error")
            else ""
        )
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
            args.extend(
                ["-ub", str(min(self.ubatch_size.value(), self.batch_size.value()))]
            )
        else:
            # Для сервера
            args.extend(["-m", model_path])
            args.extend(["--port", str(self.port.value())])
            args.extend(["-ngl", self.gpu_layers_arg()])
            args.extend(["-c", str(self.ctx_size.value())])
            args.extend(["-t", str(self.threads.value())])
            args.extend(["-b", str(self.batch_size.value())])
            args.extend(
                ["-ub", str(min(self.ubatch_size.value(), self.batch_size.value()))]
            )
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
            if self.server_metrics.isChecked():
                args.append("--metrics")
                args.append("--slots")

            # Доп. аргументы
            if self.extra_args.text():
                try:
                    args.extend(shlex.split(self.extra_args.text()))
                except ValueError as exc:
                    QMessageBox.warning(
                        self, "Ошибка", f"Не удалось разобрать доп. параметры: {exc}"
                    )
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
            QMessageBox.warning(
                self, "Сервер запущен", "Остановите сервер перед запуском benchmark"
            )
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

        self.reset_speed_metrics()
        self.bench_stop_requested = False
        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Тестирование...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.bench_process.start(bench_exe, args)

    def handle_bench_stdout(self):
        data = (
            self.bench_process.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self.update_speed_metrics(data)
        self.log(data, "bench")

    def handle_bench_stderr(self):
        data = (
            self.bench_process.readAllStandardError()
            .data()
            .decode("utf-8", errors="ignore")
        )
        self.update_speed_metrics(data)
        self.log(data, "error")

    def handle_bench_finished(self, exit_code):
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🧪 Тестировать")
        self.start_btn.setEnabled(
            self.process.state() == QProcess.ProcessState.NotRunning
        )
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
            QMessageBox.warning(
                self, "Benchmark запущен", "Остановите benchmark перед запуском сервера"
            )
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
        self.server_ready = False
        self.stats_timer.stop()
        self.stats_failures = 0
        self.reset_speed_metrics()
        self.process.start(exe, args)
        QTimer.singleShot(1500, self.force_start_server_monitoring)
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
            if self.server_stop_requested:
                return
            self.server_stop_requested = True
            self.stats_timer.stop()
            self.log("⏹ Остановка сервера...")
            self.process.terminate()
            QTimer.singleShot(3000, self.kill_server_if_running)

    def kill_server_if_running(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.log("⚠️ Сервер не завершился штатно, принудительная остановка")
            self.process.kill()

    def handle_stdout(self):
        data = (
            self.process.readAllStandardOutput().data().decode("utf-8", errors="ignore")
        )
        self.maybe_start_server_monitoring(data)
        self.update_speed_metrics(data)
        self.log(data, "info")

    def handle_stderr(self):
        data = (
            self.process.readAllStandardError().data().decode("utf-8", errors="ignore")
        )
        self.maybe_start_server_monitoring(data)
        self.update_speed_metrics(data)
        self.log(data, "error")

    def handle_state(self, state):
        states = {
            QProcess.ProcessState.NotRunning: "Остановлен",
            QProcess.ProcessState.Starting: "Запуск...",
            QProcess.ProcessState.Running: "Работает",
        }
        status = states[state]
        if state == QProcess.ProcessState.NotRunning:
            self.stats_timer.stop()
            self.server_ready = False
            self.start_btn.setEnabled(True)
            self.test_btn.setEnabled(
                self.bench_process.state() == QProcess.ProcessState.NotRunning
            )
            self.update_action_buttons()
            if self.server_stop_requested:
                self.log("⏹ Сервер остановлен")
            else:
                self.log(f"⏹ Сервер остановлен (код: {self.process.exitCode()})")
            self.server_stop_requested = False

    def maybe_start_server_monitoring(self, text):
        if (
            self.server_ready
            or self.server_stop_requested
            or not self.server_metrics.isChecked()
        ):
            return

        lower = text.lower()
        ready_markers = (
            "server is listening",
            "starting the main loop",
            "main: model loaded",
        )
        if any(marker in lower for marker in ready_markers):
            self.server_ready = True
            self.stats_failures = 0
            self.stats_timer.start()
            self.poll_server_stats()

    def force_start_server_monitoring(self):
        if (
            self.process.state() == QProcess.ProcessState.Running
            and not self.server_stop_requested
            and self.server_metrics.isChecked()
        ):
            self.server_ready = True
            self.stats_failures = 0
            if not self.stats_timer.isActive():
                self.stats_timer.start()
            self.poll_server_stats()

    def update_action_buttons(self):
        server_running = self.process.state() != QProcess.ProcessState.NotRunning
        bench_running = self.bench_process.state() != QProcess.ProcessState.NotRunning
        scan_running = self.scanner is not None and self.scanner.isRunning()
        update_running = self.updater is not None and self.updater.isRunning()
        busy = server_running or bench_running or scan_running
        self.stop_btn.setEnabled(busy)
        self.update_llama_btn.setEnabled(
            not server_running and not bench_running and not update_running
        )
        if update_running:
            self.start_btn.setEnabled(False)
            self.test_btn.setEnabled(False)
        if not server_running and not bench_running and not update_running:
            self.start_btn.setEnabled(True)
            self.test_btn.setEnabled(True)

    def poll_server_stats(self):
        if (
            self.process.state() != QProcess.ProcessState.Running
            or not self.server_metrics.isChecked()
            or not self.server_ready
            or self.server_stop_requested
        ):
            self.stats_timer.stop()
            return

        base_url = f"http://127.0.0.1:{self.port.value()}"

        for endpoint in ("metrics", "slots"):
            request = QNetworkRequest(QUrl(f"{base_url}/{endpoint}"))
            request.setTransferTimeout(1500)
            request.setRawHeader(b"User-Agent", b"LlamaServerGUI")
            reply = self.stats_manager.get(request)
            reply.setProperty("endpoint", endpoint)

    def handle_stats_reply(self, reply):
        endpoint = reply.property("endpoint")
        if (
            self.process.state() != QProcess.ProcessState.Running
            or self.server_stop_requested
        ):
            reply.deleteLater()
            return

        if reply.error():
            self.stats_failures += 1
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            err_text = reply.errorString()

            if self.stats_failures <= 3:
                self.log(
                    f"⚠️ Мониторинг /{endpoint} недоступен: HTTP={status}, error={err_text}\n",
                    "error",
                )

            reply.deleteLater()

            if self.stats_failures >= 8:
                self.context_label.setText("Контекст: мониторинг недоступен")
            return

        self.stats_failures = 0
        data = bytes(reply.readAll()).decode("utf-8", errors="ignore").strip()
        reply.deleteLater()

        if (
            endpoint == "metrics"
            and len(data) >= 2
            and data[0] == '"'
            and data[-1] == '"'
        ):
            try:
                data = bytes(data[1:-1], "utf-8").decode("unicode_escape")
            except Exception:
                pass

        if endpoint == "metrics":
            self.update_metrics_from_prometheus(data)
        elif endpoint == "slots":
            self.update_metrics_from_slots(data)

    def update_metrics_from_prometheus(self, text):
        metrics = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or " " not in line:
                continue
            name, value = line.rsplit(None, 1)
            name = name.split("{", 1)[0]
            if name.startswith("llamacpp:"):
                name = name[len("llamacpp:") :]
            try:
                metrics[name] = float(value)
            except ValueError:
                continue

        prompt_speed = self.first_metric(
            metrics,
            (
                "prompt_tokens_seconds",
                "prompt_per_second",
                "prompt_tokens_per_second",
            ),
        )
        generation_speed = self.first_metric(
            metrics,
            (
                "predicted_tokens_seconds",
                "predicted_per_second",
                "generation_tokens_per_second",
                "tokens_predicted_seconds",
            ),
        )
        kv_tokens = self.first_metric(
            metrics,
            (
                "kv_cache_tokens",
                "kv_cache_used_cells",
                "kv_cache_used",
            ),
        )
        kv_ratio = self.first_metric(
            metrics,
            (
                "kv_cache_usage_ratio",
                "kv_cache_usage",
                "kv_cache_used_ratio",
            ),
        )
        ctx_total = self.first_metric(
            metrics,
            (
                "kv_cache_tokens_total",
                "kv_cache_size",
                "n_ctx",
            ),
        )

        if prompt_speed is not None and prompt_speed > 0:
            self.prompt_speed = prompt_speed
        if generation_speed is not None and generation_speed > 0:
            self.generation_speed = generation_speed
        if kv_tokens is not None and kv_tokens >= 0:
            self.kv_cache_tokens = int(kv_tokens)
        if kv_ratio is not None and kv_ratio >= 0:
            self.kv_cache_usage_ratio = kv_ratio / 100 if kv_ratio > 1 else kv_ratio
        if ctx_total is not None and ctx_total > 0:
            self.effective_ctx_size = int(ctx_total)

        self.refresh_speed_label()
        self.refresh_context_label()

    def first_metric(self, metrics, names):
        for name in names:
            if name in metrics:
                return metrics[name]
        return None

    def update_metrics_from_slots(self, text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return

        slots = (
            data
            if isinstance(data, list)
            else data.get("slots", [])
            if isinstance(data, dict)
            else []
        )
        if not isinstance(slots, list):
            return

        ctx_values = []
        used_values = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            ctx = self.first_int_value(slot, ("n_ctx", "n_ctx_slot", "ctx_size"))
            used = self.first_int_value(
                slot,
                ("n_past", "n_tokens", "tokens_cached", "n_cache_tokens", "n_decoded"),
            )
            if ctx:
                ctx_values.append(ctx)
            if used:
                used_values.append(used)

        if ctx_values:
            self.effective_ctx_size = max(ctx_values)
        if used_values and self.kv_cache_tokens is None:
            self.kv_cache_tokens = max(used_values)
        elif used_values:
            self.kv_cache_tokens = max(used_values)

        self.refresh_context_label()

    def first_int_value(self, data, names):
        for name in names:
            value = data.get(name)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return None

    def reset_speed_metrics(self):
        self.prompt_speed = None
        self.generation_speed = None
        self.tokens_cached = None
        self.kv_cache_tokens = None
        self.kv_cache_usage_ratio = None
        self.effective_ctx_size = None
        self.decode_last_by_task = {}
        self.decode_speed_ema = None
        self.speed_label.setText("Скорость: нет данных")
        self.context_label.setText("Контекст: нет данных")

    def update_speed_metrics(self, text):
        if not text:
            return

        self.update_metrics_from_server_log_lines(text)
        self.update_metrics_from_json(text)

        for line in text.splitlines():
            lower = line.lower()

            prompt_match = re.search(
                r"prompt eval.*?([0-9]+(?:[.,][0-9]+)?)\s*(?:tokens?/s|tok/s|t/s)",
                lower,
            )
            if prompt_match:
                self.prompt_speed = self.parse_speed_value(prompt_match.group(1))

            eval_match = re.search(
                r"(?<!prompt )\beval(?: time)?\b.*?([0-9]+(?:[.,][0-9]+)?)\s*(?:tokens?/s|tok/s|t/s)",
                lower,
            )
            if eval_match:
                self.generation_speed = self.parse_speed_value(eval_match.group(1))

            pp_match = re.search(
                r"\bpp\s+[0-9]+.*?([0-9]+(?:[.,][0-9]+)?)\s*(?:tokens?/s|tok/s|t/s)",
                lower,
            )
            if pp_match:
                self.prompt_speed = self.parse_speed_value(pp_match.group(1))

            tg_match = re.search(
                r"\btg\s+[0-9]+.*?([0-9]+(?:[.,][0-9]+)?)\s*(?:tokens?/s|tok/s|t/s)",
                lower,
            )
            if tg_match:
                self.generation_speed = self.parse_speed_value(tg_match.group(1))

        self.refresh_speed_label()

    def update_metrics_from_json(self, text):
        for match in re.finditer(r'"timings"\s*:\s*\{', text):
            start = text.rfind("{", 0, match.start())
            if start < 0:
                continue
            payload = self.extract_json_object(text, start)
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            timings = data.get("timings")
            if not isinstance(timings, dict):
                continue

            prompt_speed = timings.get("prompt_per_second")
            generation_speed = timings.get("predicted_per_second")
            tokens_cached = data.get("tokens_cached")

            if isinstance(prompt_speed, (int, float)):
                self.prompt_speed = float(prompt_speed)
            if isinstance(generation_speed, (int, float)):
                self.generation_speed = float(generation_speed)
            if isinstance(tokens_cached, int):
                self.tokens_cached = tokens_cached

            self.refresh_context_label()

    def update_metrics_from_server_log_lines(self, text):
        now = time.monotonic()
        changed = False

        for line in text.splitlines():
            lower = line.lower()

            ctx_match = re.search(
                r"n_ctx\s*=\s*(\d+)\s*,\s*n_tokens\s*=\s*(\d+)", lower
            )
            if ctx_match:
                ctx = int(ctx_match.group(1))
                used = int(ctx_match.group(2))
                if ctx > 0:
                    self.effective_ctx_size = ctx
                if used >= 0:
                    self.kv_cache_tokens = used
                changed = True

            ctx_match2 = re.search(
                r"slot\s+update_batch.*?n_ctx\s*=\s*(\d+).*?n_tokens\s*=\s*(\d+)", lower
            )
            if ctx_match2:
                ctx = int(ctx_match2.group(1))
                used = int(ctx_match2.group(2))
                if ctx > 0:
                    self.effective_ctx_size = ctx
                if used >= 0:
                    self.kv_cache_tokens = used
                changed = True

            decoded_match = re.search(
                r"\|\s*task\s+(\d+)\s*\|.*?n_decoded\s*=\s*(\d+)", lower
            )
            if decoded_match:
                task_id = decoded_match.group(1)
                decoded = int(decoded_match.group(2))

                prev = self.decode_last_by_task.get(task_id)
                self.decode_last_by_task[task_id] = (decoded, now)

                if prev:
                    prev_decoded, prev_time = prev
                    dt = now - prev_time
                    dd = decoded - prev_decoded

                    if dt > 0.05 and dd > 0:
                        instant_speed = dd / dt

                        if 0.1 <= instant_speed <= 1000:
                            if self.decode_speed_ema is None:
                                self.decode_speed_ema = instant_speed
                            else:
                                self.decode_speed_ema = (
                                    self.decode_speed_ema * 0.75 + instant_speed * 0.25
                                )

                            self.generation_speed = self.decode_speed_ema
                            changed = True

            dec_match = re.search(r"n_decoded\s*=\s*(\d+)", lower)
            if dec_match:
                decoded = int(dec_match.group(1))
                task_id = "0"

                prev = self.decode_last_by_task.get(task_id)
                self.decode_last_by_task[task_id] = (decoded, now)

                if prev:
                    prev_decoded, prev_time = prev
                    dt = now - prev_time
                    dd = decoded - prev_decoded

                    if dt > 0.05 and dd > 0:
                        instant_speed = dd / dt

                        if 0.1 <= instant_speed <= 1000:
                            if self.decode_speed_ema is None:
                                self.decode_speed_ema = instant_speed
                            else:
                                self.decode_speed_ema = (
                                    self.decode_speed_ema * 0.75 + instant_speed * 0.25
                                )

                            self.generation_speed = self.decode_speed_ema
                            changed = True

            tok_match = re.search(r"n_tokens\s*=\s*(\d+)", lower)
            if tok_match:
                tokens = int(tok_match.group(1))
                if tokens > 0:
                    self.kv_cache_tokens = tokens
                    if self.effective_ctx_size is None:
                        self.effective_ctx_size = 8192
                    changed = True

        if changed:
            self.refresh_speed_label()
            self.refresh_context_label()

    def extract_json_object(self, text, start):
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

    def parse_speed_value(self, value):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None

    def refresh_speed_label(self):
        parts = []
        if self.prompt_speed is not None:
            parts.append(f"prompt {self.prompt_speed:.2f} tok/s")
        if self.generation_speed is not None:
            parts.append(f"gen {self.generation_speed:.2f} tok/s")
        if parts:
            self.speed_label.setText("Скорость: " + " | ".join(parts))

    def refresh_context_label(self):
        ctx = self.effective_ctx_size or self.ctx_size.value()

        if self.kv_cache_usage_ratio is not None:
            percent = min(999.9, self.kv_cache_usage_ratio * 100)
            if self.kv_cache_tokens is not None and ctx > 0:
                self.context_label.setText(
                    f"Контекст: KV {self.kv_cache_tokens}/{ctx} ({percent:.1f}%)"
                )
            else:
                self.context_label.setText(f"Контекст: KV {percent:.1f}%")
            return

        if self.kv_cache_tokens is not None and ctx > 0:
            percent = min(999.9, self.kv_cache_tokens / ctx * 100)
            self.context_label.setText(
                f"Контекст: KV {self.kv_cache_tokens}/{ctx} ({percent:.1f}%)"
            )
            return

        if self.tokens_cached is not None and ctx > 0:
            percent = min(999.9, self.tokens_cached / ctx * 100)
            self.context_label.setText(
                f"Контекст: cache {self.tokens_cached}/{ctx} ({percent:.1f}%)"
            )
            return

        if self.effective_ctx_size is not None and self.effective_ctx_size > 0:
            self.context_label.setText(f"Контекст: {self.effective_ctx_size} (idle)")

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
        if self.autoscroll_logs.isChecked():
            self.logs.setTextCursor(cursor)
            self.logs.ensureCursorVisible()
            scrollbar = self.logs.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    # === Управление данными ===

    def load_data(self):
        """Загрузка настроек и профилей"""
        # Загрузка базовых настроек
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                    self.exe_path.setText(self.settings.get("exe", ""))
                    self.bench_path.setText(self.settings.get("bench", ""))
                    self.model_dir.setText(self.settings.get("model_dir", ""))

                    # Загрузка параметров UI без запуска тяжелого сканирования.
                    self.bench_prompt.setValue(self.settings.get("bench_prompt", 128))
                    self.bench_gen.setValue(self.settings.get("bench_gen", 256))
                    self.auto_params.setChecked(self.settings.get("auto_params", True))
                    self.use_mmproj.setChecked(self.settings.get("use_mmproj", True))
                    self.mmproj_offload.setChecked(
                        self.settings.get("mmproj_offload", True)
                    )
                    self.gpu_auto.setChecked(self.settings.get("gpu_auto", True))
                    self.gpu_layers.setValue(self.settings.get("gpu_layers", 33))
                    self.ctx_size.setValue(self.settings.get("ctx_size", 4096))
                    self.threads.setValue(
                        self.settings.get("threads", os.cpu_count() or 4)
                    )
                    self.port.setValue(self.settings.get("port", 8080))
                    self.temperature.setValue(self.settings.get("temperature", 0.7))
                    self.repeat_penalty.setValue(
                        self.settings.get("repeat_penalty", 1.1)
                    )
                    self.flash_attn.setChecked(self.settings.get("flash_attn", True))
                    self.use_mmap.setChecked(self.settings.get("use_mmap", True))
                    self.use_mlock.setChecked(self.settings.get("use_mlock", False))
                    self.verbose.setChecked(self.settings.get("verbose", False))
                    self.log_timestamps.setChecked(
                        self.settings.get("log_timestamps", False)
                    )
                    self.cache_type_k.setCurrentText(
                        self.settings.get("cache_type_k", "f16")
                    )
                    self.cache_type_v.setCurrentText(
                        self.settings.get("cache_type_v", "f16")
                    )
                    self.batch_size.setValue(self.settings.get("batch_size", 2048))
                    self.ubatch_size.setValue(self.settings.get("ubatch_size", 512))
                    self.parallel_slots.setValue(self.settings.get("parallel_slots", 1))
                    self.cont_batching.setChecked(
                        self.settings.get("cont_batching", True)
                    )
                    self.cache_prompt.setChecked(
                        self.settings.get("cache_prompt", True)
                    )
                    self.context_shift.setChecked(
                        self.settings.get("context_shift", False)
                    )
                    self.no_webui.setChecked(self.settings.get("no_webui", False))
                    self.server_metrics.setChecked(
                        self.settings.get("server_metrics", True)
                    )
                    self.extra_args.setText(self.settings.get("extra_args", ""))

                    cached_models = self.settings.get("model_cache", [])
                    if cached_models:
                        self.on_models_found(cached_models)
            except Exception as e:
                self.log(f"Ошибка загрузки настроек: {e}", "error")

        # Загрузка профилей
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, "r", encoding="utf-8") as f:
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
            "last_model_path": self.model_combo.currentData()
            or self.settings.get("last_model_path", ""),
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
            "server_metrics": self.server_metrics.isChecked(),
            "extra_args": self.extra_args.text(),
        }
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"Ошибка сохранения настроек: {e}", "error")

    def save_profiles(self):
        try:
            with open(self.profiles_file, "w", encoding="utf-8") as f:
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
            "server_metrics": self.server_metrics.isChecked(),
            "extra_args": self.extra_args.text(),
            "bench_prompt": self.bench_prompt.value(),
            "bench_gen": self.bench_gen.value(),
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
        self.server_metrics.setChecked(p.get("server_metrics", True))
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
