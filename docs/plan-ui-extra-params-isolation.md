# План: изоляция основных UI-параметров от Extra params

> Цель: на UI остаются только **main-параметры** (launch settings + sampling +
> reasoning + MTP + mmproj + jinja + enable_thinking). Всё остальное (EXTRA) уходит
> в свободное поле `extra_args` **verbatim** — без вырезания и перезаписи при любом
> изменении виджета. Решает жалобу: «при редактировании CLI параметры перезаписываются».

Репозиторий: `G:\Android\OpenideProjects\LlamaServer\`

---

## Инвариант (подтверждён пользователем)

- **MAIN (виджеты, не трогаем):** launch settings — `host`, `port`, `gpu_auto`,
  `gpu_layers`, `gpu_layers_all`, `cpu_moe_layers`, `ctx_size`, `threads`,
  `threads_batch`, `batch_size`, `ubatch_size`, `parallel_slots`, `cache_type_k/v`,
  `flash_attn`, `fit_off` — **плюс** sampling / reasoning / MTP / mmproj / jinja /
  `enable_thinking`.
- **EXTRA (убираем из видимого UI, только текст):** `ctx_checkpoints`, `cache_ram`,
  `split_mode`, `main_gpu`, `cuda_device`, `use_mlock`, `verbose`, `log_timestamps`,
  `context_shift`, `no_webui`, `use_chat_template`, `chat_template_file`,
  `cuda_visible_devices`, `cuda_module_loading`, `kv_unified`, `cont_batching`,
  `cache_prompt`, `use_mmap`.
- **Единственный источник правды для EXTRA** — текстовое поле `extra_args`, verbatim.
- **🔴 КРИТИЧНО:** EXTRA никогда не перезаписывают и не влияют на команду.

---

## Сделано

- ✅ `src/core/param_registry.py`: 10 флагов → `managed=False`
  (`use_mlock`, `verbose`, `log_timestamps`, `kv_unified`, `ctx_checkpoints`,
  `cache_ram`, `context_shift`, `no_webui`, `use_chat_template`; `cuda_device`/
  `split_mode`/`main_gpu`/`use_mmap`/`cont_batching`/`cache_prompt` уже были
  `managed=False`). Добавлен `MANAGED_FIELD_NAMES` (frozenset имён managed-спек).
- ✅ `src/core/cli_builder.py`: 13 guard'ов на EXTRA-эмиссии (benchmark `-v`,
  `--device`, `--split-mode`, `--main-gpu`, `--kv-unified`, `--ctx-checkpoints`,
  `--cache-ram`, `--mlock`, `--verbose`, `--log-timestamps`, `--context-shift`,
  `--no-webui`, `--chat-template-file`). Импорт `MANAGED_FIELD_NAMES`.
- ✅ `tests/test_cli_builder.py`: 36 тестов OK.

---

## Сделано (продолжение сессии)

- ✅ **Шаг 4** `src/core/cli_parser.py`: в `parse_llama_server_command` после
  получения `spec`/`neg_spec` добавлен сброс в `None`, если `not spec.managed`
  → EXTRA-флаги (pos/neg) остаются в `extra_args`, не попадают в `settings`.
- ✅ **Шаг 5** `src/ui/main_window.py`: в `__init__` после `_setup_ui()` вызван
  `_hide_extra_widgets()` (удаляет 18 EXTRA-виджетов из layout + `hide()`).
- ✅ **Шаг 6** `src/core/config.py`: импорт `PARAM_REGISTRY`; хелпер
  `_extra_flag_tokens(name, value, settings)` (форматирование EXTRA-флагов по
  старой логике эмиссии) + `migrate_extra_fields_to_extra_args(settings)`
  (перенос значимых EXTRA-значений в `extra_args` verbatim + сброс поля в default,
  идемпотентен). Вызов в `load()` (после `extra_args=remaining_extra`) и
  `load_profile()` (перед `apply_to_ui`).
- ✅ **Шаг 7** `config._sanitize_extra_args`: верификация — `MANAGED_EXTRA_FLAGS`
  строится только из `spec.managed`, поэтому EXTRA-флаги НЕ вырезаются.
  Изменений НЕТ.
- ✅ **Баг-фикс** `src/core/param_registry.py`: `cuda_visible_devices` и
  `cuda_module_loading` имели `managed=True` по умолчанию (хотя это EXTRA) →
  миграция их пропускала и значения терялись. Переведены в `managed=False`.
  (`chat_template_file` оставлен `managed=True` — он обрабатывается через
  `use_chat_template` и в миграции пропускается явно.)
- ✅ **Шаг 8** Тесты: `tests/test_param_registry.py` (убраны 10 EXTRA-флагов из
  `_OLD_MANAGED_EXTRA_FLAGS`, diff → `{"--metrics","-ngld"}`); создан
  `tests/__init__.py`; поправлен `test_cli_parser.test_parses_multiline_...`
  (теперь `--device/--split-mode/--main-gpu` → `extra_args`); добавлен
  `tests/test_extra_params_isolation.py` (HARD-инертность build_args, parser→
  extra_args, миграция перенос/сброс/идемпотентность/без-дублей, sanitize не
  режет EXTRA).
- ✅ **Шаг 9** Round-trip проверен скриптом: reference-команда + `--split-mode
  none --mlock --ctx-checkpoints 8 --device CUDA0` → parse кладёт EXTRA в
  `extra_args`, MAIN в `settings`; `build_args` НЕ содержит ни одного EXTRA-флага;
  полная команда = `build_args` + `extra_args` сохраняет EXTRA verbatim.
- ✅ **Шаг 10** Полный прогон: **53 теста OK** (cli_builder 36 + param_registry +
  cli_parser + extra_params_isolation).

---

## Шаг 4 — `cli_parser.py`: EXTRA-флаги остаются в `extra_args` (ОБЯЗАТЕЛЬНО)

**Файл:** `src/core/cli_parser.py`, `parse_llama_server_command` (строки ~262-265).

**Проблема (эмпирически подтверждено):** `parse("... --mlock --ctx-checkpoints 8")`
→ `settings={"use_mlock": True, "ctx_checkpoints": 8}`, `extra_args=""`. Т.е.
EXTRA-флаг съедается в `settings` (т.к. `FLAG_TO_SPEC` строится из ВСЕХ спеков,
включая `managed=False`), а `build_args` его не эмитит (guard) → флаг **исчезает**.

**Фикс:** после `spec = FLAG_TO_SPEC.get(flag)` и
`neg_spec = NEG_FLAG_TO_SPEC.get(flag)` добавить:
```python
if spec is not None and not spec.managed:
    spec = None
if neg_spec is not None and not neg_spec.managed:
    neg_spec = None
```
→ EXTRA-флаг (pos/neg) попадает в ветку «неизвестный» (строки 268-276) и
**остаётся в `extra_args`**, не попадая в `settings`. Потребление value-токена
уже корректно (`consumed_value` выставлен для `_VALUE_FLAGS` до этого).

**Тест:** `parse("... --mlock --ctx-checkpoints 8")` → `extra_args` содержит оба
флага, `settings` их НЕ содержит.

---

## Шаг 5 — UI: EXTRA-виджеты создаём, НЕ кладём в layout

**Файл:** `src/ui/main_window.py`.

- EXTRA-виджеты создавать (атрибуты остаются), но **не добавлять** в видимые
  секции/layout. Визуально — убраны из UI.
- `FIELD_WIDGET_MAP` (`param_registry.py:242`) НЕ трогаем → parity-тест зелёный,
  `config.apply_to_ui`/`read_from_ui` (config.py:544/569) работают с созданными
  виджетами (`getattr(ui, widget_attr, None)` вернёт виджет → `_widget_set` ОК).
- Перф-влияние создания ~18 скрытых виджетов — пренебрежимо.
- Инертность гарантируется: guards (сборка) + parser-fix (парсинг).

**HARD-тест (КРИТИЧНО):** при любых значениях EXTRA-виджетов `build_args` не
содержит ни одного EXTRA-флага.

---

## Шаг 6 — Миграция старых EXTRA-значений → `extra_args` (ОБЯЗАТЕЛЬНО)

**Хелпер:** `migrate_extra_fields_to_extra_args(settings: AppSettings) -> None`
(в `config.py` или `cli_builder.py`).

- Для каждого EXTRA-поля со «значимым» (не-default) значением: сформировать флаг по
  метаданным spec (`cli_flags[0]` + `cli_kind`, reuse guard-условий) → дописать в
  `settings.extra_args` → **сбросить поле в default**.
- Идемпотентность: после сброса поля повторная загрузка не дублирует.
- **Точки вызова:** `SettingsManager.load_profile` (config.py:681-682, после
  `setattr`-цикла, до `apply_to_ui`) и `SettingsManager.load` (после загрузки
  `settings.json`).
- Особые случаи: `use_chat_template`+`chat_template_file` → `--chat-template-file
  <path>`; `cont_batching`/`cache_prompt`/`use_mmap` → `--no-*`/`--*` через
  `cli_neg_flags`; сентинелы (`cache_ram=-1`, отриц. `ctx_checkpoints`) пропускать.
- DRY: маленький `format_extra_flag(spec, settings) -> list[str]` в `cli_builder.py`,
  вызываемый миграцией (и опц. `build_args`).

---

## Шаг 7 — `config._sanitize_extra_args`: верификация (изменений НЕТ)

**Файл:** `src/core/config.py:222-249`. Использует `_MANAGED_EXTRA_FLAGS`
(импорт `MANAGED_EXTRA_FLAGS`). EXTRA-флаги удалены из `MANAGED_EXTRA_FLAGS` →
`_sanitize_extra_args` их не вырезает. Подтвердить тестом.

---

## Шаг 8 — Тесты

- `tests/test_param_registry.py`: убрать 10 EXTRA-флагов из `_OLD_MANAGED_EXTRA_FLAGS`
  (`--cache-ram`, `--chat-template-file`, `--context-shift`, `--ctx-checkpoints`,
  `--kv-unified`, `--log-timestamps`, `--mlock`, `--no-webui`, `--verbose`, `-kvu`);
  ожидаемая разность `set(MANAGED_EXTRA_FLAGS) - _OLD` → `{"--metrics", "-ngld"}`;
  добавить `tests/__init__.py` (пустой) для запуска.
- Новый тест `cli_parser`: EXTRA-флаг → `extra_args`, не `settings`.
- Новый тест миграции: перенос + сброс + идемпотентность.
- HARD-тест инертности EXTRA (Шаг 5).

---

## Шаг 9 — Round-trip reference-команды

Вставить команду пользователя + вручную `--split-mode none` → Apply → смена
MAIN-виджета → флаг сохраняется, не дублируется, не исчезает.

Reference (из запроса; `host` там без `--` — позиционный токен, уходит в extra;
реально нужен `--host`):
```
--host 127.0.0.1 --port 8080 -ngl all -t 16 -c 65536 -tb 16 -b 4096 -ub 2048 -np 1
-ctk q8_0 -ctv q8_0 -ncmoe 18 --fit off --reasoning on --temp 0.6 --top-k 20
--top-p 0.95 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.52
--frequency-penalty 0.0 --flash-attn on --no-mmproj --jinja
```

---

## Шаг 10 — Полный прогон

Окружение: `pytest` НЕ установлен → `python -m unittest`. `PySide6` НЕ установлен →
`python -m unittest discover` падает на `test_token_accumulation.py`. Гонять
изолированно:
```
python -m unittest tests.test_cli_builder tests.test_param_registry tests.test_cli_parser tests.test_extra_params_isolation -v
```

---

## Риски и Rollback

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| Миграция дублирует форматирование флагов | Средняя | Общий `format_extra_flag` |
| Парсер «съедает» value EXTRA-флага | Низкая | unknown-ветка корректно потребляет токен (проверено) |
| Скрытый виджет ломает `apply_to_ui` | Низкая | `getattr` вернёт созданный виджет → `_widget_set` ОК |
| EXTRA перезаписывают команду | Низкая | guards + parser-fix + HARD-тест |

**Rollback:** правки локальны в `param_registry.py`, `cli_builder.py`, `cli_parser.py`,
`config.py`, `main_window.py`, тестах. Откат — `git checkout` по файлам.
