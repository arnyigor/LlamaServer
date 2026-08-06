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

### Менеджер локальных моделей и загрузка с Hugging Face

Панель `Local model manager and download` (`models_panel`) объединяет два подблока:

- **Локальные модели** (`Local model manager`):
  - показывает все модели и папки под `Models` (не только HF-загрузки);
  - `Refresh local models` — перечитать список;
  - `Delete selected` — безопасное удаление выбранной папки/файла модели (projectors и MTP draft включаются в удаление папки, но не показываются отдельными моделями).
- **Загрузка GGUF с Hugging Face** (подблок можно скрыть чекбоксом `Show Hugging Face download`):
  - вставка repo id или URL (например, `unsloth/Qwen3.6-27B-MTP-GGUF`);
  - файлы сохраняются как `<Models>/<author>/<model>/<file>.gguf` — совместимо с LM Studio;
  - `Scan HF` — сканирование репозитория через Hugging Face API, список `.gguf` файлов с фильтром по квантованию (например, `Q4_K_M`, `IQ4`, `Q3-BF16` — от Q3 до BF16);
  - опция `also vision/mmproj` — включить в список projector-файлы;
  - `Download selected` с прогрессом, `Pause` / `Cancel` (отмена удаляет частично скачанные файлы);
  - блок `Local files:` показывает уже скачанные файлы для репозитория; `Delete local folder` удаляет всю папку репо включая vision-файлы;
  - фоновая загрузка в `QThread` (`HfRepoScanner` / `HfModelDownloader`), UI не блокируется.

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
  - `-rea / --reasoning`;
  - `--ctx-checkpoints`, `--cache-ram`, `-kvu`;
  - `--chat-template-kwargs {"enable_thinking":...}`;
  - `--chat-template-file` (свой шаблон, напр. `templates/qwen3_claude_relaxed.jinja`);
  - `--jinja`;
  - `--mmap` / `--no-mmap`;
  - `--mlock`;
  - `--verbose`;
  - `--log-timestamps`;
  - `--no-cont-batching`;
  - `--no-cache-prompt`;
  - `--context-shift`;
  - `--no-webui`;
  - `-mm`, `--no-mmproj`, `--no-mmproj-offload`.
- **MTP speculative decoding** (чекбокс `MTP speculative` в Launch settings):
  - `--spec-type`, `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`, `--spec-draft-ngl`, `--spec-draft-device`, `--spec-draft-type-k` / `-ctkd`, `--spec-draft-type-v` / `-ctvd`;
  - отдельный draft GGUF через `--model-draft` (поле `MTP draft GGUF`);
  - минималистичный CLI намеренно: лишние draft KV/device/ngl флаги могут конфликтовать с Gemma4Assistant draft GGUF в текущих сборках `llama.cpp`.
- Позволяет добавить дополнительные аргументы вручную.
- Для дополнительных аргументов выполняется базовая защита:
  - проверка путей у path-like флагов (`--grammar-file`, `--lora`, `--mmproj`, `--chat-template-file` и др.);
  - запрет `--host 0.0.0.0`, `--host ::`, потому что это открывает сервер на все интерфейсы.

### Benchmark

- Автоматически ищет `llama-bench.exe` рядом с `llama-server.exe`.
- Запускает `llama-bench` с выбранной моделью.
- Настраивает prompt/generation длину для benchmark.
- Парсит `tok/s` / `tokens/s` из логов и показывает скорость.

### AutoTune

Кнопка `AutoTune...` в панели Benchmark открывает встроенный AutoTune widget:

- строит план кандидатов из текущих настроек модели/контекста (GPU layers, MoE, ctx, KV cache, batch/ubatch, threads, Flash Attention, etc.);
- прогоняет `llama-bench` для каждого кандидата в фоновом `QThread` (`AutoTuneManager`), UI не блокируется;
- показывает результаты с дельтами относительно baseline, early stop после пика;
- кандидатов можно редактировать прямо в таблице;
- сохраняет отчёты (текстовый/JSON) и выбирает лучший набор параметров;
- поддерживает отмену и прогресс по шагам.

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

Парсер `src/core/mem_viz_parser.py` разбирает логи `llama.cpp` и собирает:

- VRAM/RAM по категориям;
- веса модели;
- KV cache;
- prompt cache;
- recurrent state;
- compute buffers;
- предупреждения и OOM/allocate errors;
- offloaded layers;
- примерное состояние готовности сервера.

В текущей версии UI вкладка `Memory` скрыта: данные разбора выводятся в общий лог после загрузки модели (блок `📊 Memory after load:`), а `MemoryVisualizationWidget` остаётся подключённым к данным на будущее.

Данные берутся из stdout/stderr `llama.cpp`, поэтому полнота визуализации зависит от версии и подробности логов `llama-server`.

### Логи

- Цветной лог stdout/stderr внутри приложения.
- Буферизация логов, чтобы UI меньше тормозил на частом выводе.
- Ограничение количества строк (`MAX_LOG_LINES = 10000`).
- Автоопределение уровней: info/warn/error/bench.
- Автоскролл можно выключить при ручном просмотре старых строк.

### Runtime stats

Блок `Runtime stats` показывает живые метрики сервера:

- **Speed** — текущая скорость генерации, tok/s;
- **Tokens: total | task** — накопленные токены (total — за сессию, task — текущей задачи);
- **Request** — prompt/generated токены последнего запроса;
- **Saved** — история «закрытых» задач;
- **Active** — суммарное время работы модели (PP + TG) с момента старта сервера или последнего `Reset session`; простой и ожидание в очереди не считаются. Берётся из логов `/slots` как сумма интервалов;
- **Current** — точное время последнего запроса (PP/TG), извлекается из `llama_print_timings` в логах (`src/ui/log_manager.py`, сигнал `timing_updated`). `/metrics` не используется, потому что `*_tokens_seconds` — это throughput, а не время.

Три кнопки сброса:

| Кнопка | Действие |
|---|---|
| `Reset task` | сохраняет текущую задачу в Saved и начинает следующую с нуля: сбрасывает task-счётчик, Request и Current time |
| `Reset session` | обнуляет все живые счётчики (total/task токены, prompt/generated, Active и Current time, Request) через baseline-смещения; Saved-история сохраняется |
| `Reset saved` | обнуляет накопленную историю Saved (last и total) |

Сброс реализован в `main.py` через сессионные смещения `_session_base_*`; при каждом старте сервера статистика автоматически сбрасывается.

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
- в UI можно выбрать мажорную версию CUDA (12 или 13) через `cuda_version_combo`; для CUDA 13 дополнительно скачиваются cudart DLL, minor-версия (12.4/13.3) определяется из release автоматически;
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

## Структура окна

Левая панель собрана в `src/ui/main_window.py` в следующем порядке:

1. **Кнопки управления** — `Start Server`, `Restart`, `Stop`, `Force Stop` (вверху, чтобы не скролить);
2. **Paths** — пути к llama.cpp и моделям + `Update llama.cpp`;
3. **Model** — селектор модели (read-only) и `Scan`;
4. **Launch settings** — видимый блок сразу за Model (context size с быстрыми кнопками, GPU offload, batch/ubatch, threads, Save Preset, MTP speculative);
5. **Runtime stats** — живые метрики и кнопки Reset;
6. **Advanced: Paths and llama.cpp** — спойлер с путями и обновлением;
7. **Advanced: Memory, Sampling, Server** — спойлер со всеми параметрами производительности и памяти;
8. **Local model manager and download** — менеджер моделей + HF;
9. **Integration (OpenCode / PI)**;
10. **Benchmark** — `Test Speed`, `AutoTune...` + AutoTune widget;
11. **CLI Preview** — финальная строка запуска.

Правая панель — вкладки **Logs** и **AutoTune** (`MemoryVisualizationWidget` скрыт).

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

В репозитории есть spec-файл `LlamaServerGUI.spec` (onefile, windowed, иконка `assets/llama_server_icon.ico`):

```powershell
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean LlamaServerGUI.spec
```

`dist/` и `build/` находятся в `.gitignore` и не коммитятся.

## Основной сценарий работы

1. Указать `llama-server.exe`.
2. Указать папку с моделями.
3. Нажать `Scan`.
4. Выбрать GGUF-модель.
5. Оставить `Auto setup ctx/GPU/cache by GGUF`, если нужны безопасные стартовые параметры.
6. При необходимости изменить:
   - context size (быстрые кнопки `8K`–`256K`);
   - GPU layers / auto / all;
   - CPU MoE layers;
   - KV cache K/V;
   - batch / ubatch;
   - threads;
   - parallel slots;
   - port;
   - MTP speculative decoding и draft GGUF;
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
src/core/benchmark_models.py    генерация кандидатов AutoTune
src/core/benchmark_plan.py      план AutoTune и ранжирование кандидатов
src/core/benchmark_scorer.py    скоринг результатов benchmark

src/services/autotune_manager.py  фоновый QThread-прогон AutoTune
src/services/benchmark_runner.py  запуск llama-bench для кандидата
src/services/hf_downloader.py     сканирование HF-репо и загрузка GGUF
src/services/report_writer.py     запись отчётов AutoTune
src/services/threads.py         ModelScanner, LlamaCppUpdater, HfRepoScanner
src/services/integration.py     функции формирования OpenCode/PI provider
src/services/integration_manager.py  бизнес-логика интеграции

src/ui/main_window.py           построение главного окна
src/ui/log_manager.py           логирование, цвета, скорость, llama_print_timings
src/ui/mem_viz_widget.py        виджеты визуализации памяти (скрыты)
src/ui/autotune_widget.py       UI-панель AutoTune
src/ui/tooltips.py              расширенные подсказки ctx/ncmoe
src/ui/widgets.py               CollapsiblePanel

src/utils/file_utils.py         JSON I/O, атомарная запись, path validation
src/utils/subprocess_utils.py   helpers для subprocess/QProcess

templates/qwen3_claude_relaxed.jinja  chat template для Qwen3.6 tool calls

tests/                          unit/UI tests
```

### `main.py`

`LlamaGUI` — главный coordinator:

- создаёт `MainWindowUI`, `ConfigManager`, `ServerManager`, `MetricsPoller`;
- загружает настройки;
- соединяет сигналы UI с обработчиками;
- запускает автосканирование моделей;
- обновляет CLI preview;
- применяет auto params;
- обрабатывает выбор модели;
- запускает server/benchmark/AutoTune;
- агрегирует runtime stats (токены, скорость, Active/Current время) и обрабатывает Reset task/session/saved через сессионные смещения `_session_base_*`;
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

- `/slots` — состояния слотов и расчёт скорости по дельтам токенов;
- `/metrics` — кумулятивные серверные счётчики (опрашиваются реже).

`MetricsPoller` работает в фоновом `QThread` (`_MetricsFetchWorker`), сигналы `slot_metrics_updated` / `server_metrics_updated` приходят в UI. Точное время запроса (PP/TG) берётся из `llama_print_timings` в логах через `log_manager.py`, а `/metrics` используется как дополнительный источник.

### `src/ui/autotune_widget.py`

Панель AutoTune: таблица кандидатов с редактируемыми параметрами, запуск/остановка, прогресс, дельты к baseline, выбор лучшего результата. Работает через `AutoTuneManager` (QThread) и `benchmark_runner.py`.

### `src/services/hf_downloader.py`

`HfRepoScanner` — сканирование Hugging Face репозитория (repo id или URL) и фильтрация `.gguf` по квантованию. `HfModelDownloader` — фоновая загрузка выбранных файлов с прогрессом, pause/cancel (отмена удаляет частичные файлы), сохранение в `<Models>/<author>/<model>/`.

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
142 passed, 12 subtests passed
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
