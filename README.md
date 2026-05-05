# LlamaServer GUI

GUI-менеджер для запуска `llama-server.exe` из `llama.cpp` на Windows. Приложение помогает выбрать GGUF-модель, подобрать базовые параметры запуска, сохранить профили и быстро проверить конфигурацию через `llama-bench.exe`.

## Возможности

- Автоматическое сканирование папки с `.gguf` моделями.
- Кэш найденных моделей в `settings.json`, чтобы окно открывалось быстрее.
- Чтение базовой GGUF metadata: архитектура, квантование, максимальный контекст модели.
- Автообнаружение `mmproj`/projector-файла рядом с моделью для vision/multimodal моделей.
- Примерная автонастройка `ctx`, KV cache и batch/ubatch по модели.
- Автоопределение `llama-bench.exe` рядом с `llama-server.exe`.
- Запуск и остановка `llama-server` из интерфейса.
- Запуск `llama-bench` для быстрой проверки скорости.
- Отмена активной работы: server, benchmark и фоновое сканирование останавливаются без блокировки UI.
- Автоскролл логов с возможностью отключить его при ручном просмотре старых строк.
- Отображение скорости prompt processing и generation, если `llama.cpp` печатает `tok/s` в логах.
- Отображение заполненности контекста по `tokens_cached`, если в лог попадает JSON-ответ `llama-server` с `timings`.
- Прокручиваемая панель настроек и tooltip-подсказки над основными параметрами.
- Логи stdout/stderr внутри приложения.

## Требования

- Windows.
- Python 3.10+.
- `llama.cpp` Windows build с `llama-server.exe`.
- Опционально: `llama-bench.exe` для тестирования.
- GGUF-модели.

Python-зависимости:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

## Сборка

Собрать `dist\LlamaServerGUI.exe` через локальное окружение `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name LlamaServerGUI main.py
```

При первом запуске укажите:

- путь к `llama-server.exe`;
- путь к `llama-bench.exe`, если он не определился автоматически;
- базовую папку с GGUF-моделями.

После выбора папки нажмите `Сканировать`. При следующих запусках список моделей будет загружаться из кэша, а обновление пойдет в фоне.

## Основной сценарий

1. Укажите путь к `llama-server.exe`.
2. Укажите папку с моделями.
3. Нажмите `Сканировать`.
4. Выберите модель из списка.
5. Оставьте `Автонастройка ctx/GPU/cache по GGUF`, если хотите примерные безопасные параметры.
6. При необходимости измените порт, контекст, потоки, GPU layers и параметры памяти.
7. Нажмите `Тестировать` для `llama-bench` или `Старт Server` для запуска API-сервера.

По умолчанию сервер слушает порт `8080`.

Кнопка `Стоп` останавливает текущую активную работу: сервер, benchmark или фоновое сканирование. Сначала отправляется штатный `terminate`, затем, если процесс не завершился сам, выполняется принудительный `kill` по таймеру. UI при этом не должен зависать.

## Логи и скорость

В правой панели есть переключатель `Автоскролл`. Если он включен, новые строки всегда прокручивают лог вниз. Если нужно изучить старый вывод, отключите его.

Поле `Скорость` обновляется из вывода `llama-bench` и `llama-server`, когда в логах появляются строки с `tok/s`:

- prompt / prompt eval speed;
- generation / eval speed.

Если `llama-server` не печатает timing-строки для конкретного запроса, скорость появится только после benchmark или после включения более подробного вывода.

Если в лог попадает JSON-ответ `llama-server` с блоком `timings`, GUI берет оттуда:

- `prompt_per_second`;
- `predicted_per_second`;
- `tokens_cached`.

`tokens_cached` отображается как заполненность контекста относительно текущего `ctx-size`.

## Автонастройка моделей

Приложение читает заголовок GGUF-файла и пытается определить:

- `general.architecture`;
- `general.file_type`;
- `{architecture}.context_length`;
- размер файла;
- квантование из metadata или имени файла;
- `mmproj`/projector-файл рядом с моделью, если он есть.

Рекомендуемый `ctx` считается приблизительно. Максимальный контекст берется из GGUF metadata, но реальная стабильная величина зависит от RAM/VRAM, размера модели, KV cache, batch и числа параллельных слотов.

Примерная логика:

- `Q2/IQ2` - небольшой контекст;
- `Q4` - обычно хороший баланс;
- `Q5/Q6/Q8` - можно пробовать больший контекст;
- большие модели ограничиваются осторожнее из-за расхода памяти.

Если автоматика ошиблась, отключите `Автонастройка ctx/GPU/cache по GGUF` и задайте параметры вручную.

## Multimodal / vision модели

Некоторые vision-модели требуют отдельный projector-файл (`mmproj`). Приложение ищет рядом с выбранной моделью файлы с именами вида:

- `*mmproj*.gguf`;
- `*mmproj*.bin`;
- `*projector*.gguf`;
- `*projector*.bin`.

Если файл найден и включена галочка `Использовать mmproj, если найден`, при запуске будет добавлен аргумент:

```text
-mm path\to\mmproj.gguf
```

Если visual/projector часть не нужна, снимите галочку. Тогда будет добавлен:

```text
--no-mmproj
```

Галочка `mmproj offload` управляет offload projector-части. Если ее выключить, будет добавлен `--no-mmproj-offload`.

## Важные параметры запуска

В UI вынесены наиболее полезные параметры `llama-server`:

- `--ctx-size` / `-c` - размер контекста.
- `--threads` / `-t` - CPU-потоки.
- `--n-gpu-layers` / `-ngl` - число GPU-слоев или `auto`.
- `--flash-attn` - Flash Attention.
- `-mm`, `--no-mmproj`, `--no-mmproj-offload` - multimodal projector.
- `--mmap` / `--no-mmap` - memory mapping модели.
- `--mlock` - попытка удерживать модель в RAM.
- `--verbose` - подробные логи.
- `--log-timestamps` - timestamps в логах.
- `--cache-type-k`, `--cache-type-v` - тип KV cache.
- `--batch-size`, `--ubatch-size` - batch-параметры.
- `--parallel` / `-np` - число server slots.
- `--no-cont-batching` - отключение continuous batching.
- `--no-cache-prompt` - отключение prompt cache.
- `--context-shift` - context shift для длинной генерации.
- `--no-webui` - запуск без встроенного Web UI.

Редкие или экспериментальные параметры можно добавить в поле `Доп. параметры`, например:

```text
--top-p 0.9 --min-p 0.05 --rope-scaling yarn
```

## Профили

Профили сохраняются в `profiles.json`. В профиль входят:

- выбранная модель;
- контекст;
- GPU layers;
- sampling-параметры;
- параметры памяти и KV cache;
- batch/ubatch;
- server flags;
- дополнительные аргументы;
- параметры benchmark.

## Файлы проекта

- `main.py` - основной GUI и логика запуска.
- `requirements.txt` - Python-зависимости.
- `settings.json` - локальные настройки и кэш моделей.
- `profiles.json` - сохраненные профили.
- `ANALYSIS.md` - заметки по анализу и плану улучшений.

## Проверка

Проверить синтаксис:

```powershell
.\.venv\Scripts\Activate.ps1
python -m py_compile main.py
```

Запустить приложение:

```powershell
python main.py
```

Проверить сервер после запуска можно OpenAI-compatible запросом:

```powershell
curl http://127.0.0.1:8080/v1/models
```

## Примечания

- `mmap` в актуальном `llama-server` обычно включен по умолчанию, но переключатель оставлен для явного управления.
- `mlock` полезен только если достаточно RAM и система разрешает блокировку памяти.
- Большой `ctx` резко увеличивает расход памяти KV cache.
- Для максимальной скорости подбирайте `ctx`, `cache-type-k/v`, `batch/ubatch` и `-ngl` под конкретную модель и железо.
