"""Константы и перечисления для LlamaServer GUI."""

import os
from enum import Enum, auto

# ============================================================================
# GGUF константы
# ============================================================================

# Максимальное количество метаданных GGUF для защиты от DoS
MAX_GGUF_METADATA_COUNT = 20000

# Таймауты (секунды)
TIMEOUT_VERSION_CHECK = 20
TIMEOUT_GITHUB_API = 30
TIMEOUT_DOWNLOAD = 60

# Таймауты принудительной остановки (мс)
KILL_TIMEOUT_SERVER = 3000
KILL_TIMEOUT_BENCHMARK = 2500

# Максимальный размер логов
MAX_LOG_LINES = 10000

# Размер чанка для скачивания (байты)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# Размер контекста по умолчанию
DEFAULT_CTX_SIZE = 4096
DEFAULT_BATCH_SIZE = 2048
DEFAULT_UBATCH_SIZE = 2048
DEFAULT_PORT = 8080
DEFAULT_TEMPERATURE = 0.7
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_THREADS = os.cpu_count() or 4
DEFAULT_GPU_LAYERS = 33

# Рекомендуемые контексты по квантованию
CTX_RECOMMENDATIONS = {
    "Q2": 4096,
    "IQ1": 4096,
    "IQ2": 4096,
    "Q3": 6144,
    "IQ3": 6144,
    "Q4": 8192,
    "IQ4": 8192,
    "Q5": 12288,
    "IQ5": 12288,
    "Q6": 16384,
    "Q8": 24576,
    "F16": 32768,
    "BF16": 32768,
    "F32": 32768,
}

# Размер модели для ограничения контекста (GiB)
LARGE_MODEL_THRESHOLD = 24
MEDIUM_MODEL_THRESHOLD = 14
SMALL_MODEL_THRESHOLD = 5


LLAMACPP_PROVIDER_ID = "llamacpp"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"


# ============================================================================
# Форматирование Runtime stats (токены/скорость)
# ============================================================================

# Цвета на тёмном фоне (~#1e1e1e): подписи серые, значения яркие и разные
# по смыслу, чтобы цифры было легко сканировать.
STAT_COLOR_CAPTION = "#9a9a9a"
STAT_COLOR_SEP = "#555555"
STAT_COLOR_TOTAL = "#ffffff"
STAT_COLOR_TASK = "#ffcc66"
STAT_COLOR_PROMPT = "#4fc1ff"
STAT_COLOR_GENERATED = "#6bff8f"
STAT_COLOR_SAVED = "#b0b0b0"


def format_speed(value) -> str:
    """Форматирование скорости: разделитель тысяч для больших значений.

    1234.56 -> "1,234.6" (запятая = тысячи)
    25.38   -> "25.38"   (без запятой = доли)
    """
    v = max(float(value or 0), 0.0)
    if v >= 100:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def stat_kv(caption: str, value: str, color: str = STAT_COLOR_TOTAL) -> str:
    """HTML-пара 'подпись значение' для Runtime stats."""
    return (
        f'<span style="color:{STAT_COLOR_CAPTION}">{caption}</span> '
        f'<span style="color:{color}; font-weight:bold;">{value}</span>'
    )


def stat_sep() -> str:
    """Разделитель между метриками в Runtime stats."""
    return f'<span style="color:{STAT_COLOR_SEP};"> | </span>'


class AppState(Enum):
    """Состояния приложения для безопасной работы с процессами."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


# Допустимые флаги командной строки llama.cpp
LLAMA_ALLOWED_FLAGS = frozenset(
    {
        "--port",
        "--host",
        "--ctx-size",
        "--threads",
        "--threads-batch",
        "--batch-size",
        "--ubatch-size",
        "--n-gpu-layers",
        "--split-mode",
        "--main-gpu",
        "--tensor-split",
        "--cache-type-k",
        "--cache-type-v",
        "--flash-attn",
        "--no-mmap",
        "--mmap",
        "--mlock",
        "--no-mlock",
        "--verbose",
        "--log-timestamps",
        "--cont-batching",
        "--no-cont-batching",
        "--cache-prompt",
        "--no-cache-prompt",
        "--ctx-checkpoints",
        "--cache-ram",
        "--chat-template-kwargs",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-p-min",
        "--spec-draft-ngl",
        "--spec-draft-device",
        "--spec-draft-type-k",
        "--spec-draft-type-v",
        "--model-draft",
        "--temp",
        "--repeat-penalty",
        "--top-p",
        "--min-p",
        "--seed",
        "--jinja",
        "--no-webui",
        "--mmproj",
        "--no-mmproj",
        "--no-mmproj-offload",
        "--fit",
        "--rope-scaling",
        "--rope-freq-base",
        "--rope-freq-scale",
        "--yarn-ext-factor",
        "--yarn-attn-factor",
        "--yarn-beta-fast",
        "--yarn-beta-slow",
        "--yarn-orig-ctx",
    }
)
