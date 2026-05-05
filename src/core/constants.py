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

# Допустимые флаги llama.cpp для extra_args (whitelist)
LLAMA_ALLOWED_FLAGS = {
    "--top-p",
    "--min-p",
    "--top-k",
    "--typical",
    "--repeat-last-n",
    "--frequency-penalty",
    "--presence-penalty",
    "--tfs",
    "--mirostat",
    "--mirostat-lr",
    "--mirostat-ent",
    "--rope-freq-base",
    "--rope-freq-scale",
    "--rope-scaling",
    "--yarn-ext-factor",
    "--yarn-attn-factor",
    "--yarn-beta-fast",
    "--yarn-beta-slow",
    "--defrag-thold",
    "--pooling",
    "--attention",
    "--input-prefix",
    "--input-suffix",
    "--reverse-prompt",
    "--grammar",
    "--grammar-file",
    "--json-schema",
    "--chat-template",
    "--chat-template-file",
    "--samplers",
    "--seed",
    "--override-tensor",
    "--lora",
    "--lora-scaled",
    "--mmproj",
    "--host",
    "--path",
    "--api-key",
    "--api-key-file",
    "--ctx-checkpoints",
    "--cache-ram",
}

LLAMACPP_PROVIDER_ID = "llamacpp"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"


class AppState(Enum):
    """Состояния приложения для безопасной работы с процессами."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()
