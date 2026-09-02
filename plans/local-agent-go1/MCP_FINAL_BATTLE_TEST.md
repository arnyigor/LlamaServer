# MCP Final Battle Test — local_verify + local_implement

Дата: 2026-09-02
MCP-сервер: `g:\AIModels\MCPs\local-junior` (отдельный пакет, вне репозитория LlamaServer)
Артефакты: `plans/local-agent-go1/mcp-final-battle-20260902/`
Target repo: клон https://github.com/arnyigor/flightlogbook

## Executive Summary

**Гипотеза подтверждена.** Локальная Qwen3.8-27B-IQ4_XS пригодна как субагент старшей
модели — но только в схеме, где runtime, а не модель, отвечает за границы и проверки.

Ключевой результат: junior получил Change Contract (без единой строки кода от senior),
внёс изменение в изолированном worktree, и **runtime сам подтвердил** результат реальной
компиляцией Android-проекта. С первой попытки, 0 ремонтных итераций.

## Что тестировалось

Всё через **настоящий MCP-протокол по stdio** (SDK-клиент → `server.mjs`), а не прямым
вызовом функций — то есть ровно так, как это увидят Claude Code / Codex / OpenCode.

### Шаг 1 — `local_verify`

Проверка конкретного утверждения о миграции 12→13.

| Метрика | Значение |
|---|---|
| verdict | **VERIFIED** |
| wall time | 11 сек |
| steps / tool calls | 2 / 1 |
| files read | 1 |
| budget_limited | false |

Ответ содержал точные строки (14 и 17), процитировал оба SQL-стейтмента и корректно
вывел следствие (колонки останутся NULL). Галлюцинаций нет.

### Шаг 2 — `local_implement`

Change Contract: перенести `ground_time`/`night_time` через миграцию 12→13.
Senior **не писал ни строки Kotlin/SQL** — только goal, allowed_files, changes,
invariants, acceptance.

| Метрика | Значение |
|---|---|
| status | **completed** |
| acceptance.overall | **pass** |
| scope | pass (изменён ровно 1 разрешённый файл) |
| compile `:app:compileDebugKotlin` | **pass, exit_code 0** |
| search (`ground_time` ≥ 3) | pass (3 совпадения) |
| repair_attempts | **0** |
| diff_stat | 1 файл, +1 / −1 |
| wall time | 90 сек |

Полученный diff — ровно требуемое изменение, остальные стейтменты не тронуты:

```diff
-"INSERT INTO main_table_new(date, datetime, log_time, reg_no, ...) SELECT date, datetime, log_time, reg_no, ... FROM main_table"
+"INSERT INTO main_table_new(date, datetime, log_time, ground_time, night_time, reg_no, ...) SELECT date, datetime, log_time, ground_time, night_time, reg_no, ... FROM main_table"
```

Основное рабочее дерево осталось нетронутым (`git status` чистый) — вся работа шла в
`git worktree` под temp-директорией.

## Баги, найденные тестом (все — в моём коде, не в модели)

Первый прогон дал `status: blocked`. Разбор показал три реальных дефекта раннера:

1. **MCP default timeout 60 с** — `local_implement` с реальной сборкой (8 мин) обрывался
   по таймауту протокола. Критично для продакшена: Claude/Codex наткнутся на то же.
   Зафиксировано в README и во всех трёх SKILL как обязательная настройка клиента.
2. **`spawnSync` не запускает `.bat` на Windows** (`EINVAL`) — Node требует `shell: true`
   для `gradlew.bat`. Исправлено.
3. **Repair-loop жёг попытки на инфраструктурной ошибке** — потратил все 3 итерации на
   то, что модель починить не могла в принципе. Добавлен флаг `infrastructure_error`:
   такие сбои больше не ремонтируются, возвращается отдельный статус.

Отдельно стоит отметить поведение junior'а в том провальном прогоне: он **корректно
диагностировал**, что `spawnSync EINVAL` — не проблема кода, и прямо написал, что в
пределах разрешённого файла это неустранимо. Правильное поведение: не стал выдумывать
фиктивные правки, чтобы «пройти» проверку.

## Архитектура

```
Codex / Claude / OpenCode  (senior: design, decide, review)
        │  Skill = policy (когда и как делегировать)
        ▼
     MCP server  (transport/API layer, тонкий)
        │  local_find / local_verify / local_implement / local_cleanup
        ▼
     Direct runtime  (budgets, worktree, filesystem guard, acceptance)
        │
        ▼
     Local Qwen  (junior executor)
```

Реализованные принципы:

- **Бюджеты — в коде, не в промпте**: steps, tool calls, distinct files read, output
  bytes, tokens. Превышение → forced finalize (`tool_choice=none`, thinking off), а не
  падение. Всегда возвращается реальный ответ, при необходимости с `budget_limited`.
- **Записи изолированы**: только `git worktree` под temp, только файлы из
  `allowed_files`. Рабочее дерево пользователя недостижимо.
- **Acceptance — факт runtime, не утверждение модели**: scope сверяется с настоящим
  `git diff`, compile/tests запускает сам runtime.
- **Профиль не дублирует сервер**: `n_ctx`, `model_alias`, sampling-параметры берутся
  live из `GET /props`; в `junior-v1.json` лежит только то, что через API не узнать.

## Deliverables

**MCP-сервер** — `g:\AIModels\MCPs\local-junior` (самостоятельный пакет):
`server.mjs`, `profile.mjs`, `llm-client.mjs`, `agent-loop.mjs`, `readonly-tools.mjs`,
`write-tools.mjs`, `worktree.mjs`, `acceptance.mjs`, `jobs/{find,verify,implement}.mjs`,
`profiles/junior-v1.json`, `test/{smoke,final-battle}.mjs`, `README.md`.

**SKILL** — три варианта одной политики делегирования:
- `skills/claude-code/SKILL.md`
- `skills/codex/SKILL.md`
- `skills/opencode/local-junior.md`

Сервер **нигде не зарегистрирован** — он инертен, пока клиент явно не пропишет его в
своём MCP-конфиге. Запуск локальной модели сам по себе его не активирует.

## Вердикт: можно ли использовать локальную модель как субагента?

**Да — как bounded junior executor, при обязательном machine-enforced контракте.**

| Аспект | Оценка |
|---|---|
| FIND / TRACE / VERIFY | Готово. Быстро (11–90 с), точные file/line, без галлюцинаций |
| Bounded IMPLEMENT | Готово. Контракт выполнен с первой попытки, compile прошёл |
| Соблюдение scope | Готово, и проверяется runtime независимо от модели |
| Честность при сбое | Хорошо: не выдумывает успех, прямо говорит «это не чинится здесь» |
| Самостоятельный аудит / синтез | **Не готово** — подтверждено ещё в battle test #2 |
| Широкие VERIFY-формулировки | **Не готово** — уходит за бюджет; scope обязан быть узким |

Практическое правило (заложено во все три SKILL):

```
IF   поведение решено AND scope известен AND инварианты известны
AND  acceptance объективно проверяем (compile / tests / diff / search)
THEN не писать реализацию самому → local_implement
ELSE senior продолжает думать (можно опереться на local_find / local_verify)
```

## Что дальше

Проверено, что **junior справляется**. Ещё не измерено — **насколько senior реально
экономит**. Следующий шаг: A/B-бенчмарк output-токенов Codex/Claude на одинаковой
правке (сам пишет код vs. Change Contract + `local_implement`). Это единственная
оставшаяся величина, ради которой всё строилось.

Также в бэклоге: асинхронный job-режим (`local_implement` возвращает `job_id` + опрос
статуса) — снимет зависимость от таймаута MCP-клиента, который сейчас приходится
поднимать вручную.
