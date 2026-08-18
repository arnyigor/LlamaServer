# UI/UX Анализ и план улучшений — Llama Server Studio

> Версия: 2026-08-16 | Анализ на основе `source_dump.md` (53 файла, 19 752 строк)

---

## Содержание

1. [Критические баги в UI-коде](#1-критические-баги-в-ui-коде)
2. [Архитектура информации — левая панель](#2-архитектура-информации--левая-панель)
3. [Архитектура информации — правая панель](#3-архитектура-информации--правая-панель)
4. [Проблемы отдельных виджетов](#4-проблемы-отдельных-виджетов)
5. [Стили и тема оформления](#5-стили-и-тема-оформления)
6. [Интернационализация](#6-интернационализация)
7. [Детальный план улучшений](#7-детальный-план-улучшений)
8. [Приоритеты и оценки](#8-приоритеты-и-оценки)

---

## 1. Критические баги в UI-коде

### 1.1 Метод `_build_left_panel` определён ДВАЖДЫ

**Файл:** `src/ui/main_window.py`, строки 15958–15964 и 15965–16065

Первое определение создаёт виджет и layout, но не добавляет в него ничего и ничего не возвращает. Python тихо заменяет его вторым определением. Первая копия — мёртвый код, который вводит в заблуждение при чтении.

```python
# ДУБЛЬ 1 (строки 15958–15964) — ничего не делает, return неявный (None)
def _build_left_panel(self):
    panel = QWidget()
    panel.setMinimumWidth(720)
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(10, 10, 10, 10)
    lay.setSpacing(10)

# ДУБЛЬ 2 (строки 15965–16065) — реальный код
def _build_left_panel(self):
    panel = QWidget()
    ...
```

**Исправление:** Удалить первое определение метода.

---

### 1.2 `force_stop_btn` — невидимая кнопка-дублёр

**Файл:** `src/ui/main_window.py`

`force_stop_btn` создаётся как `QPushButton`, сразу делается `setVisible(False)`. Параллельно создаётся `force_stop_action` в меню `"..."`. Оба указывают на одно действие. Видимая кнопка никогда не показывается, но занимает место в layout и входит в `_runtime_lockable`. Если они не синхронизированы, возможны рассинхронизации состояний.

```python
self.force_stop_btn = QPushButton(self.tr("Force Stop"), enabled=True)
self.force_stop_btn.setVisible(False)           # ← всегда скрыта
...
self.force_stop_action = QAction("Force Stop", self.more_actions_menu)
self.more_actions_menu.addAction(self.force_stop_action)  # ← единственный способ добраться
```

**Исправление:** Выбрать один механизм. Рекомендуется: показывать `force_stop_btn` рядом со Stop при зависании (через таймер), убрать `more_actions_btn`.

---

### 1.3 `placeholderText` у `extra_args` задан дважды (конфликт)

**Файл:** `src/ui/main_window.py`, строки 16978–16979

```python
self.extra_args = QLineEdit(placeholderText="--top-p 0.9 --min-p 0.05 ...")
self.extra_args.setPlaceholderText("--dry-multiplier 0.8 --xtc-probability 0.1 ...")
```

Первый текст задаётся в конструкторе, сразу перезаписывается вторым. Остаток от неполного редактирования.

**Исправление:** Оставить только один `setPlaceholderText` с актуальным текстом.

---

### 1.4 Tooltip `active_time_label` и `current_time_label` — только на русском

**Файл:** `src/ui/main_window.py`, строки 16411–16424

Тексты тултипов для этих двух важных меток написаны по-русски жёсткой строкой (без `self.tr()`), хотя весь остальной интерфейс использует `self.tr()` для локализации. При переключении языка на английский тултипы останутся русскими.

---

### 1.5 Хрупкий порядок вставки виджетов через `insertWidget(N, ...)`

**Файл:** `src/ui/main_window.py`

В методе `_build_performance_section`:
```python
lay.insertWidget(1, g_launch)
lay.insertWidget(2, self.launch_summary_group)
```

Позиция виджета зависит от того, сколько виджетов уже добавлено в `lay` на момент вызова метода. Порядок вызовов `_build_*` методов задаётся в `_build_left_panel`, и стоит изменить этот порядок — вёрстка сломается без ошибки компиляции.

**Исправление:** Использовать `_assemble_left_sections` как единственное место для управления порядком; все методы `_build_*` только создают виджеты, не добавляя их в layout.

---

## 2. Архитектура информации — левая панель

### 2.1 Текущая структура прокрутки (порядок снизу вверх по приоритету использования)

```
[Launch controls]          ← ✅ на виду (правильно)
[Paths panel]              ← редко меняется (не нужен вверху)
[g_launch: Launch settings]← ✅ важно перед запуском
[launch_summary_group]     ← ✅ важно перед запуском
[Model section]            ← ✅ важно перед запуском
[HF Models + downloads]    ← редко используется
[runtime_stats_group]      ← ❌ САМОЕ ВАЖНОЕ ВО ВРЕМЯ РАБОТЫ — в конце прокрутки!
[adv_panel (collapsed)]    ← редко
[sampling_panel (collapsed)]← редко
[server_panel (collapsed)] ← редко
[models_panel (collapsed)] ← редко
[int_panel (collapsed)]    ← редко
[bench_panel (collapsed)]  ← редко
[cli_group]                ← часто нужен для отладки
[stretch]
```

**Проблема:** Runtime Stats (Speed, Tokens, Active/Current time) — самая нужная информация во время работы сервера — находится в середине прокручиваемого списка, после секции HF-загрузок. Чтобы увидеть метрики, пользователь должен прокрутить вниз мимо 5–6 секций.

### 2.2 Предлагаемая структура левой панели

```
┌─────────────────────────────────────────┐
│  [▶ Start Server] [⏹ Stop] [↺ Restart]  │  ← launch controls (фиксированы)
│  [Advanced Settings ▼]                  │
├─────────────────────────────────────────┤
│  RUNTIME STATS (всегда видны, вне scroll│  ← ПЕРЕМЕСТИТЬ из scroll
│  Speed: PP 0 t/s | TG 0 t/s            │
│  Tokens: Total 0 | This task 0         │
│  Time: Active 0:00 | Current 0:00      │
├─────────────────────────────────────────┤  ← QSplitter с scroll ниже
│  [scroll area]                          │
│    ▶ Model & Paths                      │
│    ▶ Launch Settings                    │
│    ▶ Advanced: Performance & Memory     │  ← collapsed by default
│    ▶ Generation: Sampling               │  ← collapsed by default
│    ▶ Server & Diagnostics              │  ← collapsed by default
│    ▶ Model Manager & HF Download       │  ← collapsed by default
│    ▶ Integration                       │  ← collapsed by default
│    ▶ Benchmark                         │  ← collapsed by default
│    CLI Preview                          │
└─────────────────────────────────────────┘
```

**Что изменилось:**
- Runtime stats вынесены из прокрутки в фиксированную зону между кнопками запуска и scroll-областью
- "Advanced Settings" остаётся как toggle для всех collapsed-панелей
- Порядок секций отражает частоту использования

### 2.3 Состояние коллапсируемых панелей не сохраняется

`CollapsiblePanel` (файл `src/ui/widgets.py`) не сохраняет своё состояние. При каждом запуске все панели свёрнуты. Пользователь, предпочитающий работать с раскрытым "Sampling", вынужден разворачивать его каждый раз.

**Исправление:** Добавить `QSettings` в `CollapsiblePanel.__init__` с уникальным ключом на основе заголовка:

```python
def __init__(self, title, parent=None, settings_key=None):
    ...
    self._settings_key = settings_key or f"panel_{title}"
    settings = QSettings("LlamaServerGUI", "UIState")
    is_open = settings.value(self._settings_key, False, type=bool)
    self.toggle_btn.setChecked(is_open)
    self.content_widget.setVisible(is_open)
    self.toggle_btn.clicked.connect(self._save_state)

def _save_state(self):
    settings = QSettings("LlamaServerGUI", "UIState")
    settings.setValue(self._settings_key, self.toggle_btn.isChecked())
```

---

## 3. Архитектура информации — правая панель

### 3.1 Текущие вкладки

| Вкладка | Содержимое | Проблема |
|---|---|---|
| Overview | Статус, модель, 6 карточек метрик, строка настроек | Дублирует левую панель по смыслу, показывает runtime stats которые также есть слева |
| Monitor | Memory visualization (VRAM/RAM bars) | Видна только при открытой вкладке, хотя это важно во время запуска |
| Logs | QTextEdit логов | Хорошо |
| AutoTune | Полный AutoTune widget | Хорошо |

### 3.2 Предложение по реструктуризации вкладок

```
Overview   → переименовать в "Dashboard"
Monitor    → объединить с Overview или сделать mini-strip поверх всех вкладок
Logs       → оставить
AutoTune   → оставить
```

**Идея:** Добавить компактную строку-статус ВВЕРХ правой панели (над TabWidget), которая видна всегда:
```
● Running  |  TG 42.3 t/s  |  VRAM 18.2 / 24.0 GiB (75%)  |  Tokens: 12,430
```

Эта строка обновляется в реальном времени и видна независимо от активной вкладки.

### 3.3 Overview — карточки метрик

6 карточек в сетке 3×2:
```
[Generation]  [Memory]   [Request]
[Context]     [Active]   [Endpoint]
```

**Проблема с "Endpoint":** URL `http://127.0.0.1:8080/v1` — это справочная информация, а не метрика. Занимать целую карточку с `font-size: 20px; font-weight: bold` для неё избыточно.

**Предложение:**
- Убрать "Endpoint" из карточек, перенести в строку `overview_settings`
- Добавить карточку "Uptime" (время с момента старта сервера)
- Переименовать "Active" → "Work Time" для ясности

---

## 4. Проблемы отдельных виджетов

### 4.1 Кнопки управления сервером

**Текущее состояние машины состояний:**

| Состояние | Видимые кнопки | Проблема |
|---|---|---|
| Stopped | [Start Server] [...] [Advanced] | "..." с Force Stop неинтуитивен |
| Running | [Restart] [Stop] [...] [Advanced] | Reload переименован в Restart (setVisible/setEnabled) |
| Starting | Не определено явно | Нет промежуточного состояния |
| Crashed | Не определено | Force Stop может быть нужен |

**Предложение — упрощённая машина состояний:**

```
Stopped:   [▶ Start Server ........]    [Settings ▼]
Starting:  [◉ Starting...       ⏹]    [Settings ▼]
Running:   [↺ Restart] [⏹ Stop]         [Settings ▼]
Stuck:     [↺ Restart] [⏹ Stop] [☠ Kill] [Settings ▼]
```

- `☠ Kill` (Force Stop) появляется автоматически через ~5 сек после нажатия Stop без ответа
- Убирает лишний "..." menu

### 4.2 Ряд Context Size — перегружен

**Текущий ряд (r2):**
```
"Context Size (-c):" [spinbox] [?] "CPU MoE (-ncmoe):" [spinbox] [?] [8K][16K][24K][32K][41K][65K][128K][256K]
```

8 quick-кнопок + 2 spinbox + 2 лейбла + 2 кнопки-помощника в одном `QHBoxLayout` — слишком много для ширины 720–940px.

**Предложения:**
1. Перенести быстрые кнопки контекста на отдельный ряд под spinbox
2. Исправить "41K" → "40K" (40960 = 40×1024, не 41×1000)
3. Визуально сгруппировать: `[8K][16K][32K]` `[64K][128K][256K]` (убрать промежуточные 24K, 41K, или переместить в тултип spinbox)

```
┌──────────────────────────────────────────────────┐
│ Context (-c): [  auto ▲▼] [?]  CPU MoE: [auto▲▼][?] │
│ Quick: [8K] [16K] [32K] [64K] [128K] [256K]     │
└──────────────────────────────────────────────────┘
```

### 4.3 CLI Preview — однострочный QLineEdit

**Файл:** `src/ui/main_window.py`, строка 17113

```python
self.cli_preview = QLineEdit(
    placeholderText="Command will be displayed here...", readOnly=True
)
```

Типичная команда llama-server при 15+ параметрах занимает 300–500 символов. В однострочном поле она нечитаема.

**Исправление:** Заменить на `QTextEdit` с фиксированной высотой (~4 строки), режим word-wrap. При `cli_manual_mode` выключен — `readOnly=True`, включён — `readOnly=False`.

```python
self.cli_preview = QTextEdit()
self.cli_preview.setReadOnly(True)
self.cli_preview.setFixedHeight(80)
self.cli_preview.setFont(QFont("Consolas", 9))
self.cli_preview.setStyleSheet("background-color: #1a1a1a; color: #b5cea8;")
```

### 4.4 Секция Integration — показывает все три пути одновременно

**Файл:** `src/ui/main_window.py`, строки 16983–17046

Всегда показаны три поля: OpenCode JSON, PI JSON, Claude settings JSON — независимо от выбранного Target. Пользователь видит три input-поля, хотя использует только одно.

**Исправление:** Привязать видимость полей к `integration_target.currentIndexChanged`:

```python
def _update_integration_target_visibility(self):
    target = self.integration_target.currentData()
    self.opencode_row.setVisible(target == "opencode")
    self.pi_row.setVisible(target == "pi")
    self.claude_row.setVisible(target == "claude")
```

Это упрощает секцию с 3 input-полей до 1.

### 4.5 HF Downloads — 4 таблицы в одной панели

**Файл:** `src/ui/main_window.py`, строки 16226–16388

Внутри `CollapsiblePanel("Local model manager and download")` размещены:
- `local_models_list` — 130px max (таблица 5 колонок)
- `hf_files` — 120px max (список файлов для скачивания)
- `hf_downloads` — 220px max (активные загрузки)
- `hf_local_files` — 90px max (уже скачанные)

Каждая таблица слишком маленькая, чтобы быть полезной без внутренней прокрутки.

**Предложение:** Объединить в две секции с вложенными вкладками:

```
▶ Model Manager & Downloads
  ┌──────────┬───────────────┐
  │ Library  │ HF Download   │
  └──────────┴───────────────┘
  [Library tab]: одна таблица с моделями, кнопки Refresh/Delete
  [HF Download tab]: поле repo + фильтр + таблица файлов + активные загрузки
```

### 4.6 Метки Runtime Stats — криптичные сокращения

**Текущие метки:**
```
Speed: -
Tokens: total 0 | task 0
Request: -
Saved: 0
Active: 0:00 (PP 0:00 | TG 0:00)
Current: 0:00 (PP 0:00 | TG 0:00)
```

**Проблема:** "PP" и "TG" — это llama.cpp внутренние термины (Prompt Processing, Token Generation). Новый пользователь не понимает значение.

**Предложение — более понятные метки:**
```
Speed: Prompt — t/s  |  Generate — t/s
Tokens: Total 0 | Task 0 | Saved 0
Active work time: 0:00  (Prompt 0:00 | Gen 0:00)
Last request time: 0:00  (Prompt 0:00 | Gen 0:00)
```

Тултипы должны объяснять разницу между "Active" и "Current/Last request".

### 4.7 AutoTune — 26 колонок таблицы

**Файл:** `src/ui/autotune_widget.py`

```python
_COLUMNS = [
    "#", "Status", "Score", "%Best", "ΔTG",
    "Prompt tok/s", "Gen tok/s", "Load sec", "VRAM", "RAM", "Est VRAM", "Risk",
    "ngl", "ncmoe", "ctk", "ctv", "batch", "ubatch", "threads", "threads_batch",
    "np", "flash_attn", "mmproj", "ctx_checkpoints", "cache_ram", "error"
]
```

26 колонок в одной таблице нечитаемы на экране 1920px. Параметры (ngl, ncmoe, ctk...) полезны только при анализе результатов, а не при слежении за прогрессом.

**Предложение — разделить на видимые и скрытые:**

Основные (всегда видны): `#`, `Status`, `Score`, `%Best`, `ΔTG`, `PP t/s`, `TG t/s`, `VRAM`, `Risk`

Детали параметров (показывать в раскрывающейся строке или боковой панели при выборе строки): все остальные

Добавить кнопку "Show parameters" для разворачивания полной таблицы.

**Также:** Добавить цветовое кодирование строк:
- `success` (и best) → светло-зелёный фон
- `running` → светло-синий фон
- `failed` / `oom` → светло-красный фон
- `pending` → серый текст

### 4.8 Sampling Section — избыточность CLI flags в лейблах

```python
("Temperature (--temp):", self.temperature),
("Top K (--top-k):", self.top_k),
...
```

CLI флаги в лейблах занимают место и захламляют интерфейс. Пользователь, работающий через GUI, не должен видеть `--top-k` в интерфейсе.

**Предложение:**
- Лейбл: `Temperature`, `Top K`, `Top P` и т.д.
- В тултипе: `CLI: --temp <value>` + объяснение

### 4.9 CollapsiblePanel — стиль жёстко задан под тёмную тему

**Файл:** `src/ui/widgets.py`, строки 18190–18193

```python
self.toggle_btn.setStyleSheet(
    "text-align: left; font-weight: bold; border: 1px solid #444; "
    "padding: 5px; background: #2a2a2a; color: #ccc; border-radius: 4px;"
)
```

Hardcoded тёмные цвета (#2a2a2a, #444, #ccc) конфликтуют с Windows Light Theme.

**Исправление:** Использовать QPalette или CSS-переменные через QSS:
```python
self.toggle_btn.setStyleSheet(
    "text-align: left; font-weight: bold; padding: 5px; border-radius: 4px;"
)
```
Полный QSS для тёмной/светлой темы определять на уровне QApplication.

### 4.10 MemoryVisualizationWidget — русские строки в paintEvent

**Файл:** `src/ui/mem_viz_widget.py`

```python
painter.drawText(rect, Qt.AlignCenter, "Нет данных")  # в MemoryBar
...
self.model_info.setText("Модель не выбрана")
self.status_label.setText("Ожидание данных...")
self.status_label.setText("Сервер остановлен")
```

Эти строки обходят систему `self.tr()` и не переводятся при выборе английского языка.

---

## 5. Стили и тема оформления

### 5.1 Несогласованность тёмных и системных цветов

| Компонент | Стиль |
|---|---|
| CollapsiblePanel header | `background: #2a2a2a; color: #ccc` (принудительно тёмный) |
| Log view | `background-color: #1e1e1e; color: #d4d4d4` (тёмный, ок для логов) |
| CLI preview | `background-color: #2a2a2a; color: #b5cea8` (тёмный) |
| MemoryBar empty | `QColor(40, 40, 40)` (тёмный) |
| MemoryCategoryWidget util_bar | `background: #2a2a2a` (тёмный) |
| Кнопки Start/Stop/Restart | `background-color: <STATUS_COLOR_*>; color: white` (цветные) |
| QGroupBox, QSpinBox, остальное | системная тема Windows |

Результат: тёмные блоки `CollapsiblePanel` на светлом фоне системной темы выглядят как вставки из другого приложения.

### 5.2 Кнопки действий — встроенные стили

Стили задаются через длинные Python строки:
```python
self.start_btn.setStyleSheet(
    "background-color: " + STATUS_COLOR_RUNNING + "; color: white; font-weight: bold; padding: 8px;"
)
```

**Лучшая практика:** Определить QSS stylesheet на уровне приложения:
```css
QPushButton#start_btn {
    background-color: #2e7d32;
    color: white;
    font-weight: bold;
    padding: 8px;
    border-radius: 4px;
}
QPushButton#stop_btn { background-color: #c62828; ... }
```

И присваивать `self.start_btn.setObjectName("start_btn")`.

---

## 6. Интернационализация

### 6.1 Неполное покрытие `self.tr()`

| Место | Пример | Проблема |
|---|---|---|
| AutoTune widget | `QLabel("Mode:")`, `QLabel("Target:")` | Не оборачивается в `tr()` |
| Tooltips stats labels | `"Active — суммарное время..."` | Только по-русски |
| MemoryVisualizationWidget | `"Нет данных"`, `"Сервер остановлен"` | Только по-русски |
| MemoryBar.paintEvent | `"Нет данных"` | Только по-русски |
| Tooltips в _setup_tooltips | Все на английском | При выборе русского не переводятся |

**Стратегия:** Определить протокол: все user-visible строки через `self.tr()` или `QCoreApplication.translate()`. Провести аудит AutoTune widget — он не наследует от `QMainWindow` и использует `QWidget`, где `tr()` работает через `QObject.tr()`.

---

## 7. Детальный план улучшений

### ЭТАП 1 — Критические исправления (2–4 часа)

#### 1.1 Удалить дублированный `_build_left_panel`
```
Файл: src/ui/main_window.py
Действие: Удалить строки 15958–15964 (первое неполное определение)
Риск: нулевой
```

#### 1.2 Исправить `placeholderText` у `extra_args`
```
Файл: src/ui/main_window.py, ~строка 16978
Действие: Удалить аргумент placeholderText из конструктора,
          оставить только setPlaceholderText с правильным текстом
```

#### 1.3 Перенести tooltip тексты `active_time_label` / `current_time_label` в `_setup_tooltips`
```
Файл: src/ui/main_window.py
Действие: Убрать setToolTip из _build_hf_models_section,
          добавить в _setup_tooltips (и обернуть в self.tr())
```

#### 1.4 Сохранение состояния CollapsiblePanel
```
Файл: src/ui/widgets.py
Действие: В __init__ принять параметр settings_key,
          читать/писать QSettings при toggle_visibility
Влияние: Незначительное, backward-compatible
```

---

### ЭТАП 2 — Layout и информационная архитектура (1–2 дня)

#### 2.1 Вынести Runtime Stats из scroll-области

**Цель:** Runtime Stats всегда видны без прокрутки.

**Реализация:**
- Создать `_build_runtime_bar()` — компактную горизонтальную или вертикальную полосу
- Разместить её между launch controls и `QScrollArea`
- Сделать `runtime_stats_group` не частью scroll
- Обновить `_assemble_left_sections` убрав runtime_stats_group из scroll

```python
# В _setup_ui (упрощённо)
def _build_left_panel(self):
    panel = QWidget()
    lay = QVBoxLayout(panel)

    self._build_launch_controls_section(lay)   # кнопки

    self.runtime_bar = self._build_runtime_bar()
    lay.addWidget(self.runtime_bar)             # ← ФИКСИРОВАННАЯ ЗОНА

    scroll = QScrollArea(...)
    inner = QWidget()
    inner_lay = QVBoxLayout(inner)
    # ... все CollapsiblePanel в inner_lay ...
    scroll.setWidget(inner)
    lay.addWidget(scroll)                       # ← всё остальное в прокрутке
    return panel
```

#### 2.2 Рефакторинг `_assemble_left_sections`

Текущий код: метод добавляет панели, но `g_launch` и `launch_summary_group` добавляются через `insertWidget(N, ...)` в `_build_performance_section`. Это хрупко.

**Цель:** Все `addWidget` / `insertWidget` для левой панели — только в одном месте.

```python
def _build_performance_section(self):
    # Только создаёт виджеты, не добавляет в lay
    self.g_launch = QGroupBox(...)
    self.launch_summary_group = QGroupBox(...)
    self.adv_panel = CollapsiblePanel(...)
    self.sampling_panel = CollapsiblePanel(...)
    self.server_panel = CollapsiblePanel(...)

def _assemble_left_sections(self, scroll_lay):
    scroll_lay.addWidget(self.paths_panel)
    scroll_lay.addWidget(self.g_launch)            # ← явно здесь
    scroll_lay.addWidget(self.launch_summary_group)# ← явно здесь
    scroll_lay.addWidget(self.adv_panel)
    scroll_lay.addWidget(self.sampling_panel)
    scroll_lay.addWidget(self.server_panel)
    scroll_lay.addWidget(self.models_panel)
    scroll_lay.addWidget(self.int_panel)
    scroll_lay.addWidget(self.bench_panel)
    scroll_lay.addWidget(self.cli_group)
    scroll_lay.addStretch()
```

#### 2.3 Перестроить Integration section

```python
# Сделать 3 поля path условно видимыми
self.integration_target.currentIndexChanged.connect(
    self._on_integration_target_changed
)

def _on_integration_target_changed(self):
    target = self.integration_target.currentData()
    self.opencode_row_widget.setVisible(target == "opencode")
    self.pi_row_widget.setVisible(target == "pi")
    self.claude_row_widget.setVisible(target == "claude")
```

---

### ЭТАП 3 — UX улучшения (2–3 дня)

#### 3.1 CLI Preview: QLineEdit → QTextEdit

```python
# Заменить в _build_cli_section:
self.cli_preview = QTextEdit()
self.cli_preview.setReadOnly(True)
self.cli_preview.setMinimumHeight(60)
self.cli_preview.setMaximumHeight(100)
self.cli_preview.setFont(QFont("Consolas", 9))
self.cli_preview.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
self.cli_preview.setStyleSheet(
    "background-color: #1a1a1a; color: #b5cea8;"
)

# Sync с manual mode:
def _on_cli_manual_toggled(self, checked):
    self.cli_preview.setReadOnly(not checked)
    self.cli_apply_btn.setEnabled(checked)
```

#### 3.2 Context Size row — разделить на два ряда

```python
# Ряд 1: spinboxes и help
r2a = QHBoxLayout()
r2a.addWidget(QLabel(self.tr("Context (-c):")))
r2a.addWidget(self.ctx_size)
r2a.addWidget(self.ctx_help_btn)
r2a.addSpacing(16)
r2a.addWidget(QLabel(self.tr("CPU MoE (-ncmoe):")))
r2a.addWidget(self.cpu_moe_layers)
r2a.addWidget(self.ncmoe_help_btn)
r2a.addStretch(1)

# Ряд 2: быстрые кнопки
r2b = QHBoxLayout()
r2b.addWidget(QLabel("Quick:"))
for label, value in [("8K",8192),("16K",16384),("32K",32768),
                      ("64K",65536),("128K",131072),("256K",262144)]:
    btn = QPushButton(label)
    btn.setFixedWidth(44)
    btn.setFixedHeight(22)
    ...
    r2b.addWidget(btn)
r2b.addStretch(1)
```

Убрать "24K", "41K" — нестандартные значения, вызывающие путаницу. Если нужны — пользователь вводит вручную.

#### 3.3 Метки Runtime Stats — расшифровать PP/TG

```python
# active_time_label
self.active_time_label = QLabel("Work time: 0:00  (Prompt 0:00 | Gen 0:00)")

# current_time_label
self.current_time_label = QLabel("Last request: 0:00  (Prompt 0:00 | Gen 0:00)")

# speed_label — добавить единицы
self.speed_label = QLabel("Speed:  Prompt — t/s  |  Gen — t/s")

# tokens_label
self.tokens_label = QLabel("Tokens:  Total 0  |  Task 0")
```

Тултипы через `_setup_tooltips` с `self.tr()`:
```
Work time — total model processing time since server start.
Idle time and queue waiting are not counted.
Prompt = tokenization + KV fill; Gen = token generation.
```

#### 3.4 Force Stop — автоматическое появление

```python
# В методе, вызываемом по Stop
def _on_stop_clicked(self):
    self._stop_requested_at = time.monotonic()
    self._force_stop_timer = QTimer.singleShot(5000, self._show_force_stop)
    # ... нормальный stop ...

def _show_force_stop(self):
    if self._server_still_running():
        self.force_stop_btn.setVisible(True)
        self.force_stop_btn.setEnabled(True)
```

Убрать `more_actions_btn` и `more_actions_menu`.

#### 3.5 Sampling labels — убрать CLI flags из заголовков

```python
# Вместо:
("Temperature (--temp):", self.temperature),
# Делать:
("Temperature:", self.temperature),
# И в тултипе:
self.temperature.setToolTip(
    self.tr("Generation randomness (0.0–2.0).\nCLI: --temp\nauto = server default")
)
```

---

### ЭТАП 4 — AutoTune улучшения (1 день)

#### 4.1 Скрыть колонки параметров по умолчанию

```python
# В _build_ui после создания таблицы:
_HIDDEN_BY_DEFAULT = set(range(12, 25))  # колонки параметров ngl..cache_ram

for col in _HIDDEN_BY_DEFAULT:
    self.table.setColumnHidden(col, True)

# Кнопка "Show parameters"
self.show_params_btn = QPushButton("Show parameters ▶")
self.show_params_btn.setCheckable(True)
self.show_params_btn.toggled.connect(self._toggle_param_columns)

def _toggle_param_columns(self, show: bool):
    for col in _HIDDEN_BY_DEFAULT:
        self.table.setColumnHidden(col, not show)
    self.show_params_btn.setText(
        "Hide parameters ◀" if show else "Show parameters ▶"
    )
```

#### 4.2 Цветовое кодирование строк результатов

```python
_STATUS_COLORS = {
    "success":  QColor(200, 255, 200),   # светло-зелёный
    "running":  QColor(200, 230, 255),   # светло-синий
    "failed":   QColor(255, 210, 210),   # светло-красный
    "oom":      QColor(255, 200, 180),   # светло-оранжевый
    "pending":  QColor(230, 230, 230),   # серый
}

def _set_item(self, row, col, text):
    item = self.table.item(row, col) or QTableWidgetItem()
    item.setText(str(text or ""))
    status = self.table.item(row, 1)
    if status:
        color = _STATUS_COLORS.get(status.text().lower())
        if color:
            item.setBackground(color)
    self.table.setItem(row, col, item)
```

#### 4.3 Детали кандидата при клике на строку

Добавить `QLabel` или `QTextEdit` под таблицей для показа полного набора параметров выбранного кандидата. Это заменяет необходимость показывать все 26 колонок сразу.

---

### ЭТАП 5 — Стили и тема (1–2 дня)

#### 5.1 Единый QSS stylesheet

Создать файл `src/ui/styles.py` или `assets/style.qss`:

```css
/* Кнопки действий */
QPushButton#start_btn {
    background-color: #2e7d32; color: white;
    font-weight: bold; padding: 8px 16px; border-radius: 4px;
}
QPushButton#stop_btn {
    background-color: #c62828; color: white;
    font-weight: bold; padding: 8px 16px; border-radius: 4px;
}
QPushButton#restart_btn {
    background-color: #e65100; color: white;
    font-weight: bold; padding: 8px 16px; border-radius: 4px;
}

/* CollapsiblePanel */
QPushButton[panelToggle="true"] {
    text-align: left; font-weight: bold;
    border: 1px solid palette(mid);
    padding: 5px; border-radius: 4px;
    background: palette(button);
    color: palette(button-text);
}
```

#### 5.2 Применять QSS через objectName

```python
# В create методах:
self.start_btn.setObjectName("start_btn")
self.stop_btn.setObjectName("stop_btn")
self.toggle_btn.setProperty("panelToggle", True)

# В main.py:
with open("assets/style.qss") as f:
    app.setStyleSheet(f.read())
```

---

### ЭТАП 6 — Правая панель: статус-строка (0.5 дня)

Добавить постоянную строку-статус поверх вкладок правой панели:

```python
def _build_right_panel(self):
    panel = QWidget()
    lay = QVBoxLayout(panel)

    # Постоянная строка статуса
    self.status_bar_widget = self._build_status_bar()
    lay.addWidget(self.status_bar_widget)    # ← всегда видна

    self.tabs = QTabWidget()
    lay.addWidget(self.tabs)
    ...
```

```python
def _build_status_bar(self):
    bar = QFrame()
    bar.setFrameShape(QFrame.StyledPanel)
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(8, 4, 8, 4)

    self.status_indicator = QLabel("○")       # ●/○ для running/stopped
    self.status_short = QLabel("Stopped")
    self.status_speed = QLabel("—")           # TG 42.3 t/s
    self.status_vram = QLabel("—")            # 18.2 / 24 GiB

    lay.addWidget(self.status_indicator)
    lay.addWidget(self.status_short)
    lay.addStretch()
    lay.addWidget(QLabel("Speed:"))
    lay.addWidget(self.status_speed)
    lay.addWidget(QLabel("VRAM:"))
    lay.addWidget(self.status_vram)
    return bar
```

---

## 8. Приоритеты и оценки

### Матрица приоритетов

| # | Улучшение | Приоритет | Трудозатраты | Польза |
|---|---|---|---|---|
| 1.1 | Удалить дублированный `_build_left_panel` | 🔴 Critical | 5 мин | Чистота кода |
| 1.2 | Исправить `placeholderText` у `extra_args` | 🔴 Critical | 5 мин | Корректность |
| 1.3 | Tooltips через `tr()` | 🟠 High | 30 мин | i18n |
| 1.4 | Сохранение состояния CollapsiblePanel | 🟠 High | 1 час | UX |
| 2.1 | Runtime Stats вне scroll | 🔴 Critical | 2–4 часа | UX ★★★ |
| 2.2 | Рефакторинг `_assemble_left_sections` | 🟠 High | 2 часа | Maintainability |
| 2.3 | Integration — условная видимость полей | 🟡 Medium | 1 час | UX |
| 3.1 | CLI Preview → QTextEdit | 🟠 High | 30 мин | UX ★★ |
| 3.2 | Context row → два ряда | 🟡 Medium | 1 час | UX |
| 3.3 | Runtime labels — расшифровать PP/TG | 🟠 High | 1 час | UX ★★★ |
| 3.4 | Force Stop — автоматическое появление | 🟡 Medium | 2 часа | UX |
| 3.5 | Sampling labels — убрать CLI flags | 🟡 Medium | 30 мин | UX |
| 4.1 | AutoTune — скрыть parameter columns | 🟡 Medium | 2 часа | UX ★★ |
| 4.2 | AutoTune — color-coding строк | 🟡 Medium | 1 час | UX |
| 4.3 | AutoTune — детали при клике | 🟢 Low | 2 часа | UX |
| 5.1 | Единый QSS stylesheet | 🟡 Medium | 4 часа | Consistency |
| 5.2 | MemoryViz — перевести hardcoded строки | 🟠 High | 30 мин | i18n |
| 6.1 | Статус-строка поверх вкладок | 🟡 Medium | 2 часа | UX ★★ |

### Рекомендуемая последовательность спринтов

**Спринт 1 (быстрые победы, ~полдня):**
1.1, 1.2, 1.3, 1.4, 3.1, 3.5, 5.2

**Спринт 2 (архитектура, ~2 дня):**
2.1, 2.2, 2.3, 3.3

**Спринт 3 (UX polish, ~2 дня):**
3.2, 3.4, 4.1, 4.2, 6.1

**Спринт 4 (стили, ~1–2 дня):**
5.1, 5.2 (продолжение), 4.3

---

*Конец документа. Версия 1.0, 2026-08-16*
