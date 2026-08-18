# Improvement roadmap — статус (что сделано / что осталось)

> Версия: 2026-08-16
> Ветка: `codex/improvement-roadmap`
> Источники: `docs/improvement-roadmap.md`, `uiux_analysis_llama_server_studio.md`

## Что сделано

Коммиты `6f0f8ca` (Implement improvement roadmap stages 0-5) и `b28ac70`
(Fix UI/UX regressions and complete improvement-roadmap UI work) перенесли
большую часть бизнес-логики из `main.py`/`LlamaGUI` в модули `src/core` и
`src/services`, добавили покрытие тестами и исправили регрессии UI.

- **Этап 0 — Safety и база рефакторинга:** выполнен.
  - Markdown-форматирование Runtime Stats вынесено в `src/core/runtime_stats.py`.
  - Memory Visualization больше не дёргает `QApplication.processEvents()` на
    каждую строку лога (throttling через `QTimer`).
  - Введён generation-id для metrics poller (старые сигналы после Stop не
    оживляют мёртвые slots).
  - Логика запуска/остановки, парсинга CLI, реестра параметров, MoE/MTP-советов
    вынесена в `src/core`/`src/services` с unit-тестами.
- **Этапы 1-5 — реализованы** в рамках переноса логики и компоновки:
  - Пресеты (management + README), editable CLI apply mode, импорт/экспорт CLI.
  - Diagnostics, AutoTune manager, HF download coordinator/downloader.
  - MTP fallback, param_registry (единый реестр флагов), runtime_stats KPI.
  - mem_viz_parser, paths_panel, theme.qss, переводы (`translations/llamaserver_ru.ts`).
  - PyInstaller spec для release-сборок.

## Что осталось

### Из `docs/improvement-roadmap.md`
- **Этап 1 (Информационная архитектура):** разбиение левой панели на зоны
  Run/Config/Models/AutoTune, постоянно полезная правая панель, перенос
  Runtime Stats в мониторинговую область, CLI Preview как нижняя панель —
  **требует финальной сборки/полировки layout**.
- **Этап 2 (Monitor и Runtime Stats):** Memory Visualization как вкладка,
  KPI-карточки, экспорт статистики (JSON/Markdown), log search/filter,
  error banner — **частично; экспорт и search не закрыты**.
- **Этап 3 (Запуск и параметры):** Preflight Summary, resolved auto-values
  рядом с полями, MTP conditional UI, diff изменённых параметров для
  «Restart to apply», inline compatibility warnings — **не реализовано**.
- **Этап 4 (Model Manager):** вынос Local Models/HF в широкую область,
  searchable selector, таблица HF-файлов, место на диске — **не реализовано**.
- **Этап 5 (AutoTune и Benchmark):** широкая вкладка, упрощение кнопок по
  состояниям, diff перед Apply Best, подсветка score/OOM/crash, сравнение
  baseline vs best — **частично**.
- **Этап 6 (UX polish и единый стиль):** QSS/design system, унификация языка,
  сохранение состояния splitter/вкладок, поиск по настройкам, toasts вместо
  блокирующих `QMessageBox`, info-popup для сложных расчётов — **не реализовано**.

### Из `uiux_analysis_llama_server_studio.md` (конкретные баги/задачи)
Приоритеты и оценки — в разделе 8 исходного файла. Кратко:

Критические (Спринт 1, быстрые победы):
- Удалить дублированный `_build_left_panel` (мёртвый код).
- Исправить конфликт `placeholderText` у `extra_args`.
- Tooltips `active_time_label`/`current_time_label` через `self.tr()` (i18n).
- Сохранение состояния `CollapsiblePanel` через `QSettings`.
- CLI Preview: `QLineEdit` → `QTextEdit` (читаемость длинных команд).
- Sampling labels: убрать CLI-флаги из заголовков (оставить в тултипах).
- `MemoryVisualizationWidget`: перевести hardcoded русские строки через `tr()`.

Архитектура (Спринт 2):
- Runtime Stats вне scroll-области (фиксированная зона).
- Рефакторинг `_assemble_left_sections` (единая точка порядка виджетов).
- Integration: условная видимость полей по `integration_target`.
- Runtime labels: расшифровать PP/TG (Prompt/Generate).

UX polish (Спринт 3):
- Context Size row: разделить на два ряда, убрать нестандартные 8K/24K/41K.
- Force Stop: автоматическое появление через ~5с после Stop без ответа.
- AutoTune: скрыть колонки параметров по умолчанию + кнопка «Show parameters».
- AutoTune: цветовое кодирование строк (success/running/failed/oom/pending).
- Статус-строка поверх вкладок правой панели (всегда видна).

Стили (Спринт 4):
- Единый QSS stylesheet на уровне `QApplication` (убрать hardcoded тёмные цвета,
  конфликтующие с Windows Light Theme).
- `CollapsiblePanel`/`MemoryBar`: использовать `QPalette`/CSS-переменные.

## Критерии успеха (recap)
- Запуск, импорт/экспорт CLI, пресеты, AutoTune — без регрессий.
- Лог-поток и мониторинг — без reentrancy через `processEvents()`.
- Основные сценарии видны без километрового скролла.
- Runtime/Memory/Logs доступны как мониторинг, а не часть формы настроек.
- Крупные изменения имеют тесты на чистую логику, минимально зависят от PySide UI.
