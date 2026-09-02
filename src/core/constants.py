"""Константы и перечисления для LlamaServer GUI."""

import os
from enum import Enum, auto

# Версия приложения (единственный источник правды; обновляется при релизном теге).
# Совпадает с последним git-тегом вида vX.Y.Z.
APP_VERSION = "v1.5.5"

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
# Sentinel-значения числовых полей настроек
# ============================================================================

# Числа входят в сериализованный формат settings.json и в CLI-пороги
# cli_builder, поэтому менять можно только имена, не значения.
AUTO_SENTINEL = -1  # "auto": GUI опускает флаг, выбор за llama-server
SERVER_DEFAULT_SENTINEL = -2  # "default": использовать встроенное значение сервера
SAMPLING_AUTO_FLOAT = -1.0  # sampling float-поля: авто (флаг не передаётся)
SAMPLING_AUTO_INT = -1  # top_k: авто
SAMPLING_LAST_N_AUTO = -2  # repeat_last_n: авто
SAMPLING_SEED_AUTO = -2  # seed: авто
SAMPLING_PENALTY_AUTO = -3.0  # presence/frequency penalty: авто

# ============================================================================
# Палитра UI (Этап 3.3): единые цвета статусов вместо hex-хардкода
# ============================================================================

STATUS_COLOR_RUNNING = "#4CAF50"  # зелёный: успех / кнопка Start
STATUS_COLOR_READY = "#2e7d32"  # тёмно-зелёный: сервер READY
STATUS_COLOR_ERROR = "#f44336"  # красный: ошибка / кнопка Stop
STATUS_COLOR_WARNING = "#FF9800"  # оранжевый: предупреждение / кнопка Restart
STATUS_COLOR_PENDING = "#b26a00"  # тёмно-оранжевый: загрузка / рестарт
STATUS_COLOR_BENCH = "#1565c0"  # синий: идёт benchmark
STATUS_COLOR_MUTED = "#888"  # серый: неактивный статус
STATUS_COLOR_MUTED_DARK = "#666"  # серый: вторичный текст
STATUS_COLOR_TEXT = "#1a1a1a"  # основной текст на светлом фоне

# ============================================================================
# Форматирование Runtime stats (токены/скорость)
# ============================================================================

# Палитра для СВЕТЛОГО фона (дефолтный серый Windows ~#f0f0f0):
# подписи и значения тёмные и насыщенные, чтобы читались на сером.
STAT_COLOR_CAPTION = "#1a1a1a"
STAT_COLOR_SEP = "#909090"
STAT_COLOR_TOTAL = "#000000"
STAT_COLOR_TASK = "#b35c00"
STAT_COLOR_PROMPT = "#0b6ac2"
STAT_COLOR_GENERATED = "#1e7e34"
STAT_COLOR_SAVED = "#3a3a3a"
STAT_COLOR_TIME = "#5e35b1"

# Максимальный интервал между опросами /slots, который ещё считается
# "непрерывной работой" для подсчёта активного времени. Бóльший зазор
# означает паузу опроса/простой — такие дельты отбрасываются (их догонит
# точный /metrics по завершении запроса).
MAX_ACTIVE_TIME_DT = 5.0


def format_speed(value) -> str:
    """Форматирование скорости: большие значения с одним знаком после запятой.

    1234.56 -> "1234.6"
    25.38   -> "25.38"
    Запятые-разделители тысяч намеренно не используются: пользователь счёл
    их сбивающими с толку. Доли целого видны по десятичной точке.
    """
    v = max(float(value or 0), 0.0)
    if v >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def stat_kv(caption: str, value: str, color: str = STAT_COLOR_TOTAL) -> str:
    """HTML-пара 'подпись значение' для Runtime stats.

    Подписи жирные и тёмные — обязательное условие читаемости на светлом
    фоне (пользовательские жалобы на невидимые подписи при сером фоне).
    """
    return (
        f'<span style="color:{STAT_COLOR_CAPTION}; font-weight:bold;">'
        f"{caption}</span> "
        f'<span style="color:{color}; font-weight:bold;">{value}</span>'
    )


def stat_sep() -> str:
    """Разделитель между метриками в Runtime stats."""
    return f'<span style="color:{STAT_COLOR_SEP};"> | </span>'


def format_duration(total_seconds: float) -> str:
    """Форматирование длительности: H:MM:SS / M:SS.

    7235 -> "2:00:35"
    125  -> "2:05"
    0    -> "0:00"
    """
    total = max(int(round(float(total_seconds or 0))), 0)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


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
