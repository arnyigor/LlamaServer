# План: Редизайн UI — навигационный рейл + логи снизу

> **Ветка:** `ui/navigation-rail-layout`
> **Статус:** 🟡 в разработке (Этап 1 + 1.5 + 1.6 + 1.7 + 2 — каркас, UX-доработки, реорганизация раскладки, очистка UI и лог-док готовы, верификация пройдена)
> **Дата начала:** 2026-08-25
> **Цель:** Перенять паттерн UI из `pytraveler/LlamaServerLauncherAvalonia`
> (иконочный nav-рейл слева, настройки в центре, логи в нижнем доке),
> сохранив все существующие виджеты и связи `main.py` без поломок.

---

## 0. Цель и решения (утверждены пользователем)

1. **Dashboard (Overview) = пункт nav-рейла**, а не отдельная правая панель.
   → центр (страницы настроек) занимает всю ширину.
2. **Реализация поэтапная**, каждый этап проверяется (сборка/запуск).
3. **Автотюнинг** — отдельный независимый блок (уже выделен в
   `AutoTuneWidget`), дорабатывается отдельно; в nav-рейле это просто
   страница-обёртка, интегрируемая «быстро и локально».
4. **Принцип независимых блоков** (на будущее): каждая секция настроек —
   отдельный класс-виджет (как уже сделано с `PathsPanel` и
   `AutoTuneWidget`), чтобы прикручивать/заменять блоки локально, не трогая
   `main.py` и ядро.
5. **Верификация:** можно собирать `.exe` под *другим именем* рядом — конфиги
   подтянутся, т.к. `QSettings("LlamaServerGUI", "UIState")` заданы явно и не
   зависят от имени exe. Новую версию тестируем отдельно.

---

## 1. Принципы архитектуры (независимые блоки)

- **Сохраняемость `self.*`**: `main.py` ссылается на ~200 атрибутов
  `self.ui.*` и коннектит сигналы напрямую. Любой рефакторинг меняет только
  *контейнеры*, но оставляет все виджетные атрибуты (`self.start_btn`,
  `self.ctx_size`, `self.logs`, `self.autotune`, …) на месте.
- **Страницы = классы-виджеты** (Этап 4): каждая секция (`PathsPage`,
  `PerformancePage`, `SamplingPage`, `ServerPage`, `ModelLibraryPage`,
  `IntegrationPage`, `BenchmarkPage`, `DashboardPage`, `AutoTunePage`) —
  отдельный класс, экспонирующий свои виджеты как атрибуты. `MainWindowUI`
  только собирает страницы в `QStackedWidget` и пробрасывает их атрибуты
  наружу (как `PathsPanel` уже делает через реэкспорт).
- **Сигналы наружу**: сложные страницы общаются с `LlamaGUI` через `Signal`,
  как `PathsPanel.browse_exe_requested`. Это держит блоки изолированными.
- **Один источник правды для layout**: порядок/видимость страниц задаётся
  только в одном месте (`_assemble_nav`), как сейчас `_assemble_left_sections`.

---

## 2. Исследование эталона (LlamaServerLauncherAvalonia)

Эталон — Avalonia/C# (MVVM). Код напрямую не переносим, берём **только UX-идеи**:

| Идея эталона | Как применим в PySide6 |
|---|---|
| Сетка 3 зон: шапка / контент / логи | `QVBoxLayout` окна: `[header][content][logdock]` |
| Сворачиваемый icon nav-рейл (48→220px) | `QListWidget` с иконками + `QStackedWidget` (или `QToolBox`/`QTabWidget` без заголовков) |
| Headerless TabControl | `QStackedWidget`, переключаемый рейлом |
| Логи внизу с `GridSplitter` + maximize | `QSplitter` (Vertical) + кнопка разворачивания лог-дока |
| Компактная шапка (профиль + Save flyout + язык) | `QHBoxLayout` сверху: комбо профиля, `QMenu` Save/Clone/Export/Import, комбо языка |
| Полоса управления (гейджи + чипы инстансов + Start/Stop) | Отдельный `QWidget` между контентом и лог-доком |
| Toast overlay | `QWidget` поверх с `Qt.FramelessWindowHint` (опционально, Этап 5) |
| Drag-drop overlay | `dragEnter/Leave` на центральном виджете (опционально, Этап 5) |

**Что НЕ берём** (нет в нашем проекте / избыточно): мульти-инстанс сервера,
MCP, Docker, on-demand proxy, scenarios — это фичи эталона, не относящиеся к
UI-паттерну. Их добавлять не будем.

---

## 3. Текущее состояние проекта

- `src/ui/main_window.py` — `MainWindowUI(QMainWindow)`, ~1712 строк.
  Методы `_build_*` создают виджеты и задают `self.*`. `_assemble_left_sections`
  собирает их в один `QScrollArea`. Правая панель — `QTabWidget` (Overview/Logs).
- Уже реализовано (из старого `uiux_analysis_*`): статус-бар, CLI `QTextEdit`,
  Force-Stop таймер, 2-рядный Context, `tr()`-тултипы, `CollapsiblePanel`
  persistence, лейблы сэмплинга без флагов. **Повторно не делаем.**
- `src/ui/panels/paths_panel.py` — `PathsPanel(CollapsiblePanel)` — эталон
  выделенного блока. Содержит строку языка (перенесём в шапку в Этапе 1).
- `src/ui/autotune_widget.py` — `AutoTuneWidget` — независимый блок, дорабатывается отдельно.
- `src/ui/widgets.py` — `CollapsiblePanel`, `NoWheelValueChangeFilter`.
- `main.py` — `LlamaGUI`, жёстко связан с `self.ui.*`. Не трогаем.

---

## 4. Целевой layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ HEADER: [Llama Server Studio] · Lang[en▾]                              │  Этап 1.5
├──────┬───────────────────────────────────────────────────────────────┤
│ NAV  │  QStackedWidget — страница выбранного раздела (вся ширина)      │
│ рейл │                                                                 │
│(икон-│  Pages:                                                         │
│ ки + │   • Dashboard  (stats: overview cards + live speed/tokens + preflight)      │
│ лейб │   • Paths      (llama.cpp / models / CUDA / update)            │
│ лы)  │   • Launch     (model select + context + vision/mmproj + CUDA + CLI Preview)   │
│      │   • Сэмплинг   (GPU offload + KV cache + attention + batch/threads + MTP + reasoning + sampling grid + Память KV-кэш)                     │
│ Dash │   • Server     (server opts + diagnostics + templates + extra)│
│ Path │   • Library    (local models + HF download)                    │
│ Perf │   • Integration(OpenCode/PI)                                   │
│ Samp │   • Benchmark  (prompt/gen + Test Speed + AutoTune вложен)     │
│ Srv  │   • AutoTune   (обёртка над AutoTuneWidget)                   │
│ Lib  │                                                                 │
│ Int  │                                                                 │
│ Bench│                                                                 │
│ ATune│                                                                 │
├──────┴───────────────────────────────────────────────────────────────┤
│ CONTROL STRIP: гейджи (CPU/RAM/VRAM) · Start/Stop/Restart/Unload · cmd▸│  Этап 3
├──────────────────────────────────────────────────────────────────────┤
│ LOG DOCK (ресайз через QSplitter, кнопка maximize): auto/clear/copy/.. │  Этап 2
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Структура файлов и рефакторинг

Новые/изменённые файлы:

| Файл | Роль | Этап |
|---|---|---|
| `src/ui/main_window.py` | Контейнер: header + nav + stacked + control + logdock. Делегирует страницы. | 1–3 |
| `src/ui/panels/paths_panel.py` | Убрать строку языка (→ шапка). | 1 |
| `src/ui/panels/dashboard_page.py` | **НОВЫЙ** класс-страница (Overview-карточки + preflight). | 1/4 |
| `src/ui/panels/performance_page.py` | **НОВЫЙ** (launch + adv + preflight). | 4 |
| `src/ui/panels/sampling_page.py` | **НОВЫЙ**. | 4 |
| `src/ui/panels/server_page.py` | **НОВЫЙ** (server opts + diagnostics + templates + extra). | 4 |
| `src/ui/panels/library_page.py` | **НОВЫЙ** (local models + HF). | 4 |
| `src/ui/panels/integration_page.py` | **НОВЫЙ**. | 4 |
| `src/ui/panels/benchmark_page.py` | **НОВЫЙ** (prompt/gen + Test + AutoTune). | 4 |
| `src/ui/panels/autotune_page.py` | **НОВЫЙ** — обёртка над `AutoTuneWidget`. | 1/4 |
| `src/ui/header_bar.py` | **НОВЫЙ** — шапка (только язык + брендинг). | 1/1.5 |
| `src/ui/log_dock.py` | **НОВЫЙ** — нижний док логов. | 2 |
| `src/ui/control_strip.py` | **НОВЫЙ** — полоса управления. | 3 |
| `src/ui/nav_rail.py` | **НОВЫЙ** — icon list-widget. | 1 |

> Примечание: Этапы 1–3 можно сделать «лениво» — страницы сначала это просто
> `QWidget`-контейнеры, в которые переносятся существующие группы из
> `main_window.py` (без выноса в классы). Этап 4 выносит логику в классы для
> полной независимости. Это снижает риск: сначала работает layout, потом —
> чистая архитектура.

---

## 6. Этапы реализации (чеклисты)

### ✅ Этап 0 — Ветка и план
- [x] Создана ветка `ui/navigation-rail-layout`
- [x] Записан план в `.zcode/plans/plan-ui-nav-rail.md`
- [x] Зафиксирован план коммитом (только файл плана; чужие изменения не трогаем)

### ✅ Этап 1 — Каркас: шапка + nav-рейл + страницы (Dashboard как пункт рейла)
- [x] `src/ui/header_bar.py`: `HeaderBar(QWidget)` — комбо профиля, `QPushButton`
      Save с `QMenu` (Save/SaveAs/Rename/Clone/Export/Import/Delete), кнопка
      Update llama.cpp, комбо языка. Сигналы: `profile_selected`,
      `save_requested`, `language_changed` (пока заглушки/проброс к main.py).
- [x] `src/ui/nav_rail.py`: `NavRail(QListWidget)` — элементы с иконкой
      (`QStyle.StandardPixmap`) + текстом; `currentRowChanged` →
      `page_selected(int)`.
- [x] `main_window.py`: переписать `_setup_ui` →
      `QVBoxLayout`: `[header][status_bar][launch_controls][splitter: nav|stacked][logdock]`.
- [x] Создать `QStackedWidget`; страницы = `QWidget`-контейнеры (9 шт.), в которые
      перенесены группы из `_build_*` (без выноса в классы — Этап 4).
- [x] **Dashboard** — перенести Overview-карточки + preflight + runtime_stats_group
      из правой панели в страницу Dashboard. Правая `QTabWidget` удалена.
- [x] Убрать строку языка из `paths_panel.py` (переехала в шапку).
- [x] Все `self.*` атрибуты сохранены (имена не меняем) → `main.py` работает.
- [x] `_runtime_lockable` и `_load_ui_state`/`save_ui_state` адаптированы под
      новые контейнеры (`contentSplitterState`, `navIndex`).
- [x] **Верификация Этапа 1**: `verify_phase1.py` (offscreen) — PASSED
      (75 атрибутов, 9 страниц, переключение, advanced toggle, state round-trip).
      Также исправлены: двойной вызов `_build_launch_controls_section`,
      `self.language_combo` реэкспорт из `paths_panel` (→ header), 4 строки
      `self.ui.tabs.setCurrentWidget` в `main.py`,        импорты/иконки под PySide6 6.11.

### ✅ Этап 1.5 — UX-доработки по фидбеку (поверх Этапа 1)
- [x] **Шапка очищена**: убран «Profile» (комбо + Save flyout + сигналы
      `profile_selected`/`save_requested`/`set_profiles`). Оставлена только
      `language_combo` + брендинг-лейбл «Llama Server Studio». `header_bar.py`
      переписан; `main.py` больше не дёргает `header.set_profiles`/коннекты
      профиля (удалены 4 строки + 2 метода `_on_header_save`/`_on_header_profile`).
- [x] **CollapsiblePanel статичные**: добавлен параметр `collapsible=False`
      (заголовок-лейбл, контент всегда виден). Сделаны статичными 6 панелей:
      adv_panel («Память (KV-кэш)»), sampling_panel, server_panel, models_panel,
      int_panel, bench_panel. Чёрные спойлеры убраны (страницы теперь отдельные).
- [x] **NAV_PAGES переименованы на русские** по смыслу: Главная / Пути / Запуск /
      Сэмплинг / Сервер / Модели / Интеграция / Тесты / Автотюн.
- [x] **Preset** (per-model performance preset — единственный механизм
      сохранения) перенесён из performance-секции в постоянную панель запуска
      (`_build_launch_controls_section`, Row1 рядом со Start/Stop/Advanced) +
      добавлен `launch_readout` (QLabel «Model: -», selectable) во Row2.
      Устранено двойное создание `preset_name_combo` (было и в launch-, и в
      performance-секции).
- [x] **MTP-блок** (kv_unified / speculative_mtp / spec_draft_*) перенесён из
      adv_panel («Память») на страницу **Сэмплинг** (sampling_panel).
- [x] **Extra params** (`extra_args`) перенесён из server_panel на страницу
      **Сэмплинг** (sampling_panel).
- [x] **CLI Preview** (`cli_group`: preview + Copy CLI + Import CLI + Apply CLI)
      был осиротел (создан, но не добавлен ни в одну страницу) → возвращён на
      страницу **Запуск** (`_performance_page`). Import CLI сохраняет merge-логику
      (не заменяет параметры — НЕ сломано).
- [x] **Basic/Advanced** теперь прячет только панель «Память (KV-кэш)»
      (`_advanced_panels` возвращает `[adv_panel]`); Sampling/Server и остальные
      страницы остаются видимыми. Тултип `advanced_mode_chk` обновлён.
- [x] **Верификация 1.5**: `py_compile` чистый; `verify_phase1.py` (offscreen) →
      PASSED (75 атрибутов, 9 страниц, переключение, advanced toggle, state
      round-trip); PyInstaller-сборка `dist_next/LlamaServerGUI.exe` →
       `verify_build.py` STARTUP_OK (12s, без краша).

### ✅ Этап 1.6 — Реорганизация раскладки: Запуск / Сэмплинг / Пути
- [x] **Запуск** (`g_launch`) оставлен минимальным: только выбор размера
      контекста (`ctx_size` + быстрые кнопки + CPU MoE `-ncmoe`), vision-модель
      (`use_mmproj` / `mmproj_offload`, перенесены из секции Модель) и CUDA
      (`launch_cuda_version_combo` в `r_cuda`).
- [x] **Vision (mmproj)** перенесён из `_build_model_section` (где был в группе
      Модель на Dashboard) в `_build_performance_section` → `g_launch`.
      Атрибуты `use_mmproj`/`mmproj_offload` сохранены (main.py работает).
- [x] **GPU offload (-ngl)** (`r1`), **KV K/V** (`r3`), **Flash Attention / Fit**
      (`r6`) перенесены с Запуска на страницу **Сэмплинг**, каждый завёрнут в
      свой `QGroupBox`-блок («GPU offload (-ngl)» / «KV cache type» /
      «Attention / Fit») внутри `sampling_panel.content_layout`.
- [x] **Панель «Память (KV-кэш)»** (`adv_panel`) перенесена целиком со страницы
      Запуск на страницу **Сэмплинг** как sibling-блок (`_sampling_page`).
      `_advanced_panels()` и `advanced_mode_chk` остаются рабочими (тоггл
      скрывает/показывает блок Памяти теперь на Сэмплинге).
- [x] **Пути** (`PathsPanel`): убран спойлер — `CollapsiblePanel(collapsible=False)`
      (всегда раскрыта). Версия CUDA для обновления (`cuda_version_combo`,
      источник истины) теперь показана в строке обновления рядом с кнопкой
      «Update llama.cpp».
- [x] **cuda_version_combo** — один экземпляр нельзя поместить в две раскладки,
      поэтому на Запуске показано зеркало `launch_cuda_version_combo` с
      двусторонней синхронизацией (`currentIndexChanged` ↔). Источник
      (`cuda_version_combo`) остаётся там, где `main.py` его читает/блокирует
      (`update_llamacpp`, `setEnabled`) — на вкладке Пути.
- [x] **Верификация 1.6**: `py_compile` чистый; `verify_phase1.py` (offscreen) →
      PASSED (75 атрибутов, 9 страниц, переключение, advanced toggle, state
      round-trip); PyInstaller-сборка `dist_next/LlamaServerGUI_Next.exe` и
       `dist/LlamaServerGUI_Next.exe` → `verify_build.py` STARTUP_OK (12s, без краша).

### ✅ Этап 1.7 — Модель на Запуск, английский UI, Dashboard=stats, удалён «GPU capacity»
- [x] **Выбор и данные модели** (`model_group`: Scan / Found GGUF / Auto setup /
      Model info / Copy path) перенесены с Dashboard на страницу **Запуск**
      (`_performance_page`, первым блоком). Теперь на Запуске сразу видно,
      какая модель выбрана и что запускаем.
- [x] **NAV_PAGES возвращены на английский** (Dashboard / Paths / Launch /
      Sampling / Server / Models / Integration / Benchmark / AutoTune) — проще
      читать для технических программ. (Остальные UI-лейблы и так были
      английскими; русскими были только NAV_PAGES с Этапа 1.5.)
- [x] **Dashboard = stats (onboarding)**: после переноса модели страница
      содержит только статистику — `overview_content_widget` (6 карточек:
      Generation / Memory / Request / Context / Active / Endpoint),
      `runtime_stats_group` (Speed / Tokens / Request / Saved / Active / Current
      time + кнопки экспорта) и `launch_summary_group` (preflight). Порядок:
      карточки → live stats → preflight.
- [x] **Удалён «GPU capacity»** — это была VRAM-оценка в preflight
      (`preflight_vram_bar` + сообщение «GPU capacity not available until
      llama.cpp reports it.»). Она работает только ПОСЛЕ загрузки модели
      (llama.cpp сообщает VRAM только тогда), поэтому до запуска всегда
      показывала «not available» — т.е. «не работает». Удалены виджеты
      `preflight_vram_label`/`preflight_vram_bar` (main_window.py), вся логика
      VRAM-бара и ветка «GPU capacity not available» (main.py), и упоминания из
      `verify_phase1.py` (REQUIRED уменьшен 75→73). Preflight теперь показывает
      model/context/KV/GPU offload/MTP/endpoint + статус готовности.
- [x] **Верификация 1.7**: `py_compile` чистый; `grep preflight_vram` → NONE;
      `verify_phase1.py` (offscreen) → PASSED (73 атрибута, 9 страниц,
      переключение, advanced toggle, state round-trip); PyInstaller-сборка
      `dist_next/LlamaServerGUI_Next.exe` и `dist/LlamaServerGUI_Next.exe` →
      `verify_build.py` STARTUP_OK (12s, без краша).

### ✅ Этап 2 — Лог-док снизу
- [x] `src/ui/log_dock.py`: `LogDock(QWidget)` — `QTextEdit` (self.logs, то же
       имя), заголовок с auto-scroll/clear/copy-last-error/open-diagnostics +
       кнопка maximize (toggle_maximize Signal, set_maximized(on) меняет лейбл).
- [x] Контент (nav|pages `content_splitter`) и `log_dock` обёрнуты в вертикальный
       `QSplitter` (`main_vsplit`) — лог-док ресайзится мышью.
- [x] Кнопка maximize: скрывает контент (`content_splitter.setVisible(False)`) →
       лог-док на всю высоту, и обратно (`setSizes` из сохранённых док-размеров);
       состояние в `QSettings` (`mainVSplitterSizes` + `logDockMaximized`).
- [x] `LogManager(self.ui.logs)` из `main.py` работает без изменений — атрибуты
       `logs`/`autoscroll_logs`/`copy_last_error_btn`/`open_diagnostics_btn`
       реэкспортированы с `log_dock` на `self.ui`.
- [x] **Верификация Этапа 2**: `py_compile` чистый; `verify_phase1.py` (offscreen)
       → PASSED (75 атрибутов, в т.ч. `main_vsplit`/`log_dock`; save/load
       round-trip OK); PyInstaller-сборка `dist_next/LlamaServerGUI_Next.exe` и
       `dist/LlamaServerGUI_Next.exe` → `verify_build.py` STARTUP_OK (12s, без краша).

### ⬜ Этап 3 — Полоса управления сервером
- [ ] `src/ui/control_strip.py`: `ControlStrip(QWidget)` — переиспользовать
      `self.start_btn/stop_btn/reload_btn/force_stop_btn`, гейджи
      (CPU/RAM/VRAM — если есть данные) и свёртку `self.cli_preview`.
- [ ] Разместить между `QStackedWidget` и `LogDock`.
- [ ] **Верификация Этапа 3**.

### ⬜ Этап 4 — Вынос секций в независимые классы (интегрируемость)
- [ ] Создать `src/ui/panels/*.py` (см. раздел 5). Каждый класс:
  - создаёт свои виджеты и задаёт их как атрибуты;
  - экспонирует нужное наружу (как `PathsPanel`);
  - сложные действия — через `Signal`.
- [ ] `MainWindowUI` инстанцирует страницы и реэкспортирует атрибуты
      (`self.ctx_size = self.perf_page.ctx_size`, …) — `main.py` без изменений.
- [ ] `AutoTunePage` — тонкая обёртка над существующим `AutoTuneWidget`
      (блок дорабатывается отдельно, здесь только интеграция).
- [ ] **Верификация Этапа 4** (полный прогон: запуск сервера, HF, autotune).

### ⬜ Этап 5 — Полировка (опционально)
- [ ] Сохранение/восстановление: nav index, log-dock height, maximize state.
- [ ] Toast overlay (переиспользовать лог-сообщения).
- [ ] Drag-drop overlay подсказка.
- [ ] QSS: убрать разрозненные `setStyleSheet` в пользу централизованного
      `theme.qss` (начато в `theme.qss`).
- [ ] **Верификация Этапа 5**.

---

## 7. Процесс верификации (exe под другим именем)

Конфиги (`QSettings("LlamaServerGUI", "UIState")`) не зависят от имени exe →
новая сборка подхватит существующие настройки. Тестируем отдельно от
основной версии.

1. Найти PyInstaller spec / build-скрипт (вероятно `*.spec` или
   `.github/workflows/build.yml`).
2. Собрать с `--name LlamaServerStudioNext` (другое имя) в отдельную папку
   рядом (напр. `dist_next/`).
3. Запустить `dist_next/LlamaServerStudioNext.exe`, проверить:
   - окно открывается, шапка + nav-рейл + страницы отрисованы;
   - переключение пунктов рейла работает;
   - Dashboard показывает метрики;
   - лог-док внизу (после Этапа 2) принимает логи;
   - запуск/остановка сервера работает (связи `main.py` целы);
   - конфиги подтянулись (те же профили/пути).
4. При ошибках — правим, пересобираем, повторяем. Основная ветка/сборка не
   затрагивается.

---

## 8. Журнал прогресса

| Дата | Этап | Что сделано | Как проверено | Статус |
|---|---|---|---|---|
| 2026-08-25 | 0 | Создана ветка `ui/navigation-rail-layout`; записан план | `git branch`, чтение файла | ✅ |
| 2026-08-25 | 1 | Каркас: `HeaderBar` + `NavRail` + `QStackedWidget` (9 страниц, Dashboard как пункт рейла), логи inline снизу, язык в шапку; `verify_phase1.py` | `python verify_phase1.py` (offscreen) → PASSED; `py_compile` чистый | ✅ |
| 2026-08-25 | 1.5 | UX-доработки: шапка без Profile; CollapsiblePanel статичные; NAV_PAGES рус; Preset+readout в панели запуска; MTP+extra_args на Сэмплинг; CLI Preview на Запуск; Basic/Advanced прячет только Память | `verify_phase1.py` PASSED; `py_compile` чистый; `verify_build.py` STARTUP_OK | ✅ |
| 2026-08-25 | 1.6 | Реорганизация раскладки: Запуск = context+vision+CUDA; GPU offload/KV K-V/Flash-Fit/Память → Сэмплинг (QGroupBox-блоки); Пути без спойлера + версия CUDA для обновления; cuda_version_combo-зеркало на Запуске | `verify_phase1.py` PASSED; `py_compile` чистый; `verify_build.py` STARTUP_OK (обе копии Next.exe) | ✅ |
| 2026-08-25 | 1.7 | Модель (выбор+данные) → Запуск; NAV_PAGES на английский; Dashboard=stats; удалён «GPU capacity» (VRAM-оценка preflight, не работала до запуска) | `py_compile` чистый; `grep preflight_vram`→NONE; `verify_phase1.py` PASSED (73); `verify_build.py` STARTUP_OK (обе копии) | ✅ |
| 2026-08-25 | 2 | Лог-док вынесен в `src/ui/log_dock.py`; контент+док в вертикальном `QSplitter` (ресайз); кнопка maximize (скрывает контент, состояние в QSettings) | `py_compile` чистый; `verify_phase1.py` PASSED (75, +main_vsplit/log_dock); `verify_build.py` STARTUP_OK (обе копии) | ✅ |
| 2026-08-25 | 2.1 | Убран лишний заголовок «Performance and Memory:» в панели Память (KV-кэш) на Sampling (дублировал заголовок панели); версия `APP_VERSION` (v1.5.5, `src/core/constants.py`) показана в шапке рядом с брендингом | `py_compile` чистый; `verify_phase1.py` PASSED (75); `verify_build.py` STARTUP_OK | ✅ |
| 2026-08-25 | commit | Фазы 1.6 + 1.7 + 2 + 2.1 зафиксированы коммитом `e9edea4` (8 файлов, +311/−146). Ветка `ui/navigation-rail-layout` НЕ запушена. | — | ✅ |
| | 3 | _ | _ | ⬜ |
| | 4 | _ | _ | ⬜ |
| | 5 | _ | _ | ⬜ |

---

*Конец плана. Версия 1.0, 2026-08-25.*
