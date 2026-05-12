# LlamaServer GUI

Windows GUI-менеджер для локального запуска `llama-server.exe` из `llama.cpp`. Приложение помогает выбрать GGUF-модель, подобрать параметры контекста/памяти, запускать сервер и benchmark, обновлять локальную сборку `llama.cpp`, смотреть логи и добавлять локальные модели в конфиги OpenCode/PI.

Проект написан на Python + PySide6 и разделён на модули: UI, core-логика, сервисы фоновых задач и утилиты.

## Что умеет приложение

### Модели и GGUF

- Рекурсивно сканирует папку с `.gguf` моделями.
- Исключает projector-файлы (`mmproj`, `projector`) из основного списка моделей.
- Кэширует найденные модели в `settings.json`, чтобы следующий запуск был быстрее.
- Читает ключевую GGUF metadata без полной загрузки модели:
  - `general.architecture`;
  - `general.file_type` / квантование;
  - native context length;
  - число блоков/слоёв;
  - число attention heads;
  - embedding length;
  - MoE metadata: experts / experts used.
- Определяет квантование по metadata или имени файла.
- Автоматически ищет `mmproj` / `projector` рядом с моделью.

### Запуск llama-server

- Запускает и останавливает `llama-server.exe` через `QProcess`.
- Показывает CLI preview перед запуском.
- Поддерживает основные параметры `llama-server`:
  - `--port`;
  - `-c / --ctx-size`;
  - `-ngl / --n-gpu-layers`;
  - `-t`, `-tb`;
  - `-b`, `-ub`;
  - `-ctk`, `-ctv`;
  - `-np / --parallel`;
  - `-ncmoe`;
  - `--flash-attn`;
  - `--fit off`;
  - `-rea`;
  - `--chat-template-kwargs {"enable_thinking":...}`;
  - `--mmap` / `--no-mmap`;
  - `--mlock`;
  - `--verbose`;
  - `--log-timestamps`;
  - `--no-cont-batching`;
  - `--no-cache-prompt`;
  - `--context-shift`;
  - `--no-webui`;
  - `--jinja`;
  - `-mm`, `--no-mmproj`, `--no-mmproj-offload`.
- Позволяет добавить дополнительные аргументы вручную.
- Для дополнительных аргументов выполняется базовая защита:
  - проверка путей у path-like флагов (`--grammar-file`, `--lora`, `--mmproj`, `--chat-template-file` и др.);
  - запрет `--host 0.0.0.0`, `--host ::`, потому что это открывает сервер на все интерфейсы.

### Benchmark

- Автоматически ищет `llama-bench.exe` рядом с `llama-server.exe`.
- Запускает `llama-bench` с выбранной моделью.
- Настраивает prompt/generation длину для benchmark.
- Парсит `tok/s` / `tokens/s` из логов и показывает скорость.

### Автонастройка контекста и памяти

- Рекомендует context size по квантованию, размеру модели и native context из GGUF.
- Для маленьких моделей может предлагать больший контекст.
- Для крупных моделей ограничивает контекст осторожнее.
- Для больших контекстов tooltip показывает:
  - native context;
  - текущий context;
  - примерный расход KV-cache;
  - расход KV-cache на 1K tokens;
  - таблицу KV-cache для популярных context sizes;
  - предупреждения о выходе за native context;
  - рекомендации по RoPE/YaRN;
  - рекомендации по Flash Attention, KV quantization, ubatch и ctx checkpoints.
- Есть быстрые кнопки context size: `8K`, `16K`, `24K`, `32K`, `41K`, `65K`, `128K`, `256K`.

### MoE и VRAM

- Для MoE-моделей читает число экспертов и активных экспертов.
- Параметр `CPU MoE (-ncmoe)` можно оставить в UI-режиме `auto`/default (флаг не передаётся) или задать вручную.
- Tooltip для `-ncmoe` показывает:
  - структуру MoE;
  - примерный расход VRAM на веса, KV и overhead;
  - таблицу экономии VRAM при разных `ncmoe`;
  - рекомендуемое значение.
- Встроенный оценщик VRAM учитывает:
  - размер модели;
  - число GPU layers;
  - KV cache type K/V;
  - context size;
  - parallel slots;
  - Flash Attention;
  - MoE/offload эвристику.

### Визуализация памяти по логам

Во вкладке `Memory` отображается разбор логов `llama.cpp`:

- VRAM/RAM по категориям;
- веса модели;
- KV cache;
- prompt cache;
- recurrent state;
- compute buffers;
- предупреждения и OOM/allocate errors;
- offloaded layers;
- примерное состояние готовности сервера.

Данные берутся из stdout/stderr `llama.cpp`, поэтому полнота визуализации зависит от версии и подробности логов `llama-server`.

### Логи

- Цветной лог stdout/stderr внутри приложения.
- Буферизация логов, чтобы UI меньше тормозил на частом выводе.
- Ограничение количества строк (`MAX_LOG_LINES = 10000`).
- Автоопределение уровней: info/warn/error/bench.
- Автоскролл можно выключить при ручном просмотре старых строк.

### Профили и performance presets

Есть два уровня сохранения:

1. `settings.json` — текущие глобальные настройки:
   - пути к exe/bench/model dir;
   - последняя модель;
   - кэш моделей;
   - текущие UI-параметры.
2. `profiles.json` — профили и performance presets.

Performance preset сохраняется для пары:

```text
конкретный GGUF path + конкретный ctx-size
```

В preset входят параметры производительности и памяти: GPU layers, MoE, threads, KV cache, batch/ubatch, Flash Attention, mmap/mlock, server flags, extra args, thinking и т.д.

При выборе сохранённого context size preset загружается автоматически.

### Обновление llama.cpp

Кнопка `Update llama.cpp`:

- читает текущий build через `llama-server.exe --version`;
- запрашивает latest release из GitHub `ggml-org/llama.cpp`;
- выбирает Windows asset по приоритету:
  1. CUDA 12.4;
  2. любой CUDA;
  3. Vulkan;
  4. AVX2;
  5. generic Windows;
- создаёт backup текущих `.exe`/`.dll`;
- безопасно распаковывает zip с защитой от path traversal;
- копирует новую сборку в папку с `llama-server.exe`.

Хранится до 5 последних backup-каталогов.

### Интеграция с OpenCode и PI

Приложение может добавить текущую модель как локальную OpenAI-compatible модель в JSON-конфиги OpenCode/PI.

Для OpenCode добавляется provider:

```json
{
  "llamacpp": {
    "name": "llama.cpp (local)",
    "npm": "@ai-sdk/openai-compatible",
    "options": {
      "baseURL": "http://127.0.0.1:8080/v1"
    },
    "models": {
      "model-id": {}
    }
  }
}
```

Для PI добавляется provider:

```json
{
  "llamacpp": {
    "api": "openai-completions",
    "apiKey": "llamacpp",
    "baseUrl": "http://127.0.0.1:8080/v1",
    "models": [
      {
        "id": "model-id",
        "name": "model-id"
      }
    ]
  }
}
```

Фактический контейнер provider определяется гибко: `provider`, `providers` или корневой объект.

## Требования

- Windows.
- Python 3.10+.
- `llama.cpp` Windows build с `llama-server.exe`.
- Опционально: `llama-bench.exe`.
- GGUF-модели.

Python-зависимости указаны в `requirements.txt`:

```text
PySide6>=6.5.0
```

## Установка

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск из исходников

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

При первом запуске укажите:

1. путь к `llama-server.exe`;
2. путь к `llama-bench.exe` или оставьте автоопределение;
3. папку с GGUF-моделями;
4. нажмите `Scan`.

После выбора модели можно нажать:

- `Test Speed` — запустить `llama-bench`;
- `Start Server` — запустить OpenAI-compatible сервер;
- `Stop` — остановить server/benchmark/сканирование.

По умолчанию сервер слушает:

```text
http://127.0.0.1:8080
```

OpenAI-compatible endpoint:

```text
http://127.0.0.1:8080/v1
```

Проверка после запуска:

```powershell
curl http://127.0.0.1:8080/v1/models
```

## Сборка exe

Сборка `dist\LlamaServerGUI.exe` через PyInstaller:

```powershell
.\.venv\Scripts\Activate.ps1
pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name LlamaServerGUI main.py
```

В репозитории также есть spec-файлы:

- `LlamaServer.spec`;
- `LlamaServerGUI.spec`.

## Основной сценарий работы

1. Указать `llama-server.exe`.
2. Указать папку с моделями.
3. Нажать `Scan`.
4. Выбрать GGUF-модель.
5. Оставить `Auto setup ctx/GPU/cache by GGUF`, если нужны безопасные стартовые параметры.
6. При необходимости изменить:
   - context size;
   - GPU layers / auto;
   - CPU MoE layers;
   - KV cache K/V;
   - batch / ubatch;
   - threads;
   - parallel slots;
   - port;
   - mmap/mlock/debug/server flags.
7. Проверить `CLI Preview`.
8. Нажать `Test Speed` или `Start Server`.

## Multimodal / vision модели

Если рядом с моделью найден projector-файл, приложение сохранит путь в metadata модели.

Ищутся файлы с именами:

- `*mmproj*.gguf`;
- `*mmproj*.bin`;
- `*projector*.gguf`;
- `*projector*.bin`.

Если `Use mmproj` включён, в CLI будет добавлено:

```text
-mm path\to\mmproj.gguf
```

Если `Use mmproj` выключен:

```text
--no-mmproj
```

Если `mmproj offload` выключен:

```text
--no-mmproj-offload
```

## Структура проекта и анализ модулей

```text
main.py                         точка входа и связка UI/core/services
requirements.txt                Python-зависимости
settings.json                   локальные настройки и кэш моделей
profiles.json                   профили и performance presets

src/core/cli_builder.py         сборка аргументов llama-server/llama-bench
src/core/config.py              dataclass настроек, JSON load/save, presets
src/core/constants.py           константы, defaults, allowed flags
src/core/context_advisor.py     рекомендации для большого контекста и RoPE
src/core/gguf_parser.py         быстрый парсер GGUF metadata
src/core/mem_viz_parser.py      парсер логов памяти llama.cpp
src/core/metrics_poller.py      HTTP poller /slots и /metrics
src/core/moe_advisor.py         рекомендации по CPU MoE offload
src/core/server_manager.py      QProcess-управление server/benchmark
src/core/vram_estimator.py      оценка VRAM/KV/model memory

src/services/threads.py         ModelScanner и LlamaCppUpdater
src/services/integration.py     функции формирования OpenCode/PI provider
src/services/integration_manager.py  бизнес-логика интеграции

src/ui/main_window.py           построение главного окна
src/ui/log_manager.py           логирование, цвета, скорость
src/ui/mem_viz_widget.py        виджеты визуализации памяти
src/ui/tooltips.py              расширенные подсказки ctx/ncmoe
src/ui/widgets.py               CollapsiblePanel

src/utils/file_utils.py         JSON I/O, атомарная запись, path validation

tests/                          unit/UI tests
```

### `main.py`

`LlamaGUI` — главный coordinator:

- создаёт `MainWindowUI`, `ConfigManager`, `ServerManager`;
- загружает настройки;
- соединяет сигналы UI с обработчиками;
- запускает автосканирование моделей;
- обновляет CLI preview;
- применяет auto params;
- обрабатывает выбор модели;
- запускает server/benchmark;
- сбрасывает и обновляет memory visualization;
- управляет tray icon;
- сохраняет настройки при выходе.

### `src/core/cli_builder.py`

Отвечает только за построение списка CLI-аргументов.

Плюсы реализации:

- server и benchmark собираются одной функцией `build_args`;
- benchmark не получает server-only параметры вроде `--port`;
- `enable_thinking` нормализуется для legacy bool/string значений;
- дополнительные параметры разбираются через `shlex.split`;
- path-like аргументы проверяются на выход за пределы model dir.

Важно: whitelist `LLAMA_ALLOWED_FLAGS` пока не используется как строгий фильтр для extra args. Сейчас неизвестные флаги разрешены, кроме специальных проверок путей и host.

### `src/core/config.py`

Содержит:

- `AppSettings` — единая dataclass-модель настроек;
- `_FIELD_WIDGET_MAP` — явный маппинг поле настроек → UI widget;
- универсальные `_widget_get` / `_widget_set`;
- загрузку/сохранение `settings.json`;
- загрузку/сохранение `profiles.json`;
- performance presets по хэшу абсолютного пути модели + ctx.

Сильная сторона: добавление нового UI-параметра обычно требует добавить поле в `AppSettings`, widget mapping и, при необходимости, список `_PERF_PRESET_FIELDS`.

### `src/core/gguf_parser.py`

Парсер читает только metadata GGUF и останавливается, когда собраны нужные ключи. Это быстрее, чем полный проход по файлу.

Есть защита:

- минимальный размер файла;
- проверка magic `GGUF`;
- лимит metadata count;
- обработка повреждённой структуры через `GGUFParseError`.

### `src/core/context_advisor.py`

Рассчитывает `LargeContextAdvice`:

- нужен ли RoPE scaling;
- какие параметры RoPE/YaRN предложить;
- какие KV cache types рекомендовать;
- нужен ли Flash Attention;
- какой ubatch выбрать;
- стоит ли использовать ctx checkpoints/cache RAM;
- примерный VRAM estimate.

### `src/core/vram_estimator.py`

Даёт эвристическую оценку памяти:

```text
total = model weights on GPU + KV cache + overhead
```

KV-cache зависит от:

- number of blocks;
- heads;
- embedding length;
- context size;
- K/V cache type;
- Flash Attention;
- parallel slots.

Оценка полезна как ориентир, но не заменяет реальный запуск, потому что `llama.cpp`, backend, драйвер и конкретная архитектура модели могут расходовать память иначе.

### `src/core/server_manager.py`

Изолирует работу с процессами:

- отдельный `QProcess` для server;
- отдельный `QProcess` для benchmark;
- чтение stdout/stderr;
- штатный `terminate`;
- fallback `kill` по таймеру;
- сигналы состояния для UI.

### `src/services/threads.py`

`ModelScanner` работает в `QThread`, чтобы рекурсивное сканирование `.gguf` не блокировало UI.

`LlamaCppUpdater` выполняет сетевой запрос, скачивание, распаковку и копирование также в фоне.

### `src/services/integration_manager.py`

Слой бизнес-логики для OpenCode/PI без привязки к UI. Благодаря этому интеграция покрыта unit tests.

### `src/ui/log_manager.py`

Логи буферизуются через `deque` и flush-таймер. Это снижает нагрузку на QTextEdit при большом stdout/stderr.

### `src/core/metrics_poller.py`

Содержит HTTP-клиент для опроса:

- `/slots`;
- `/metrics`.

На текущий момент основной UI получает скорость и память преимущественно из логов. `MetricsPoller` подготовлен как отдельный модуль для более точных live-метрик.

## Проверка и тесты

Синтаксис:

```powershell
.\.venv\Scripts\Activate.ps1
python -m py_compile main.py
```

Unit tests:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Текущий результат проверки в локальном `.venv`:

```text
50 passed, 12 subtests passed
```

Если запускать `pytest` системным Python без зависимостей, возможна ошибка `ModuleNotFoundError: No module named 'PySide6'`. Используйте виртуальное окружение, где установлен `requirements.txt`.

## Важные замечания

- Все рекомендации по context/VRAM являются эвристикой.
- Большой context резко увеличивает KV-cache.
- Для context `>= 32K` почти всегда полезны Flash Attention и/или quantized KV cache.
- Для context `>= 64K` часто нужно уменьшать `ubatch`.
- Для context `>= 128K` могут помочь `--ctx-checkpoints` и осторожная настройка `--cache-ram`.
- `mmap` в актуальных сборках `llama.cpp` обычно включён по умолчанию, но переключатель оставлен для явного управления.
- `mlock` имеет смысл только при достаточном объёме RAM и разрешениях ОС.
- `Update llama.cpp` заменяет файлы в папке выбранного `llama-server.exe`; backup создаётся автоматически, но лучше не обновлять рабочую сборку во время активных задач.
- Открывать сервер наружу (`--host 0.0.0.0`) через extra args намеренно запрещено базовой валидацией.
