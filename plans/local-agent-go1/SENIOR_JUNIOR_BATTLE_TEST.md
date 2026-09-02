# Senior → Junior Battle Test #2 (FlightLogbook)

Дата: 2026-09-02
Артефакты запуска: `plans/local-agent-go1/senior-junior-flightlogbook-20260902-105414/`
(`initial-map-*`, `job-a-*`, `job-a2-*`, `job-b-*` … `job-f-*` — output/stderr/trace для каждой job)

## Executive Summary

Схема Senior → Junior сработала лучше первого battle test (единая длинная сессия,
см. [FLIGHTLOGBOOK_BATTLE_TEST.md](FLIGHTLOGBOOK_BATTLE_TEST.md)).

Главный вывод: локальная Qwen/IQ4_XS заметно полезнее как **bounded junior worker**
для FIND / TRACE / VERIFY, чем как самостоятельный senior-аудитор. Большинство коротких
jobs завершились без `finish_reason=length`, вернули конкретный evidence и не пытались
писать большой synthesis. Один migrations job в первой (широкой) формулировке не смог
уложиться в бюджет; после сужения scope тот же тип задачи завершился успешно.

**Итоговый verdict: JUNIOR-READY, но ещё не READY FOR MCP.**
Причина: worker уже полезен для ограниченных задач, но routing/protocol ещё требует
защиты от "раскопал слишком широко и не дал финальный ответ".

## Baseline (Battle Test #1)

| Метрика | Старый battle |
|---|---|
| Sessions | 1 |
| Steps | 24 |
| Tool calls | 68 |
| Files inspected | 45 |
| Duration | ~275 sec |
| Tool output | ~223 KB |
| Result | Evidence хороший, final answer оборван |
| Failure mode | `finish_reason=length`, ранее ошибочно считался completed |

## Новый эксперимент

- Target repo: https://github.com/arnyigor/flightlogbook
- Repo был подготовлен один раз через `direct-agent.mjs --repo ...`, затем все jobs
  запускались через уже подготовленный `--cwd` (без повторного clone):
  `plans/local-agent-go1/senior-junior-flightlogbook-20260902-105414/repo-workdir/flightlogbook`

### Initial Map

| Metric | Value |
|---|---|
| Status | completed |
| Finish reason | stop |
| Steps | 7 |
| Tool calls | 13 |
| Files read | 2 |
| Duration | 40.4 sec |
| Output chars | 2679 |

Результат: Gradle module `:app`, root project `Pilotlogbook`, application class
`FlightLogbookApp`, root DI `AppComponent`, root DB `MainDB`, UI entry
`NavigationActivity`, tests dirs `app/src/test/java`, `app/src/androidTest/java`.
Worker честно пометил uncertainty (например, что `NavigationActivity.kt` и тестовые
файлы не читал, а вывел из manifest/структуры) — это правильное поведение.

### Senior Decomposition → Junior Jobs

| Job | Type | Status | Finish | Duration | Steps | Tool calls | Files read | Completed? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Initial Map | FIND | completed | stop | 40.4s | 7 | 13 | 2 | yes | Хороший стартовый map |
| A1 Migrations | VERIFY | failed | no final | ~210s+ | 10 | 22 | 7 | no | Уперся в max steps (scope слишком широкий) |
| A2 Migrations Tight | VERIFY | completed | stop | 50.5s | 6 | 9 | 2 | yes | Та же тема, но tighter scope |
| B Import Flow | TRACE | completed | stop | 92.6s | 11 | 20 | 8 | yes | Хороший concrete trace |
| C Async Model | FIND/VERIFY | completed | stop | 71.9s | 6 | 10 | 3 | yes | Хороший bounded fact-finding |
| D Tests | FIND | completed | stop | 20.9s | 3 | 5 | 2 | yes | Отлично для простой задачи |
| E DI Facts | TRACE/FIND | completed | stop | 53.7s | 6 | 17 | 10 | yes | Полезно, близко к file budget |
| F Layer Boundaries | FIND/VERIFY | completed | stop | 74.7s | 7 | 13 | 5 | yes | Хорошие concrete examples |

### Pass Criteria Check

| Criterion | Result |
|---|---|
| Все mandatory jobs без `finish_reason=length` | Partially pass: length не встречался, но A1 failed по max steps |
| Ни одна junior session существенно не вышла за scope | Mostly pass; A1 расползся |
| ≥90% checked factual claims VERIFIED/PARTIALLY VERIFIED | Likely pass по проверенной части (полная senior-проверка не завершена из-за лимита) |
| Нет систематических hallucinations файлов/классов | Pass по observed outputs |
| Senior смог сделать вывод без полного самостоятельного reread | Pass |
| Большинство jobs ≤10–15 files | Pass |
| Evidence достаточно конкретен | Pass |
| Local не делает global architecture decisions | Pass |

## Ключевые findings по FlightLogbook

1. **Одномодульный проект** (`:app`) → упрощает reasoning, но слои разделены пакетами,
   а не Gradle-boundaries — boundary violations не ловятся build system.
2. **Стек**: Moxy MVP, Navigation Component, Dagger 2/dagger-android, Room, RxJava 2 —
   зрелый legacy Android stack. Основная цена сейчас не в MVP, а в смешении слоёв,
   слабых транзакционных гарантиях и отсутствии тестов на storage/migrations.
3. **Import flow**: чистит existing flights/custom fields, resetит sqlite sequence,
   затем вставляет новые данные в циклах — high-risk data consistency area без единого
   атомарного boundary.
4. **Migrations 12→13 и 13→14** — потенциально опасные места:
   - 12→13 создаёт `ground_time` / `night_time`, но не копирует их в `INSERT ... SELECT`;
   - 13→14 создаёт unique index на `type_table(reg_no)`;
   - `runMigration` ловит exceptions и только печатает stack trace;
   - `app/schemas` отсутствует, хотя `exportSchema = true`.
   → Persistence layer — главный risk area; стабилизировать schema/migrations/import
   перед любой модернизацией UI.
5. **RxJava** используется широко; coroutine/viewmodel dependencies объявлены, но почти
   не используются — migration smell, не срочный bug. Сначала tests и транзакционная
   безопасность, потом миграция на корутины.
6. **Тесты** — только шаблонные (`ExampleUnitTest.kt`: 2+2, `ExampleInstrumentedTest.kt`:
   package name). Проект фактически без защитной сетки.
7. **Layer boundary leaks**: domain зависит от Android framework/resources; data
   использует utility из presentation/utils; presentation напрямую достаёт
   `Prefs.getInstance`. Мешает тестированию и постепенной модернизации.

## Оценка Senior → Junior схемы

**Лучше стало:**
- Каждая local session держит маленький контекст.
- `finish_reason=length` исчез на successful jobs.
- Worker лучше соблюдает запрет на recommendations.
- Evidence проще проверять.
- Senior не требует от Qwen глобального понимания проекта.
- Простые FIND jobs выполняются быстро и чисто.
- TRACE import flow оказался очень полезным.

**Осталось слабым:**
- Если VERIFY-задача сформулирована слишком широко, Qwen продолжает копать и не
  завершает ответ (см. job A1).
- Worker иногда сам пишет неточные internal STATS в answer, отличающиеся от harness stats.
- Evidence часто без точных line numbers (approximate `~205-215`).
- Нужно enforce `max_files_read` на уровне harness — сейчас только prompt constraint.
- Нужен protocol "stop and summarize now", когда tool budget близок к лимиту.

## Сравнение с Battle Test #1

| Dimension | Старый Battle | Новый Battle |
|---|---|---|
| Context per local session | Очень большой | Маленький |
| Output contract | 17 секций, провал | Короткие reports, в основном pass |
| Tool calls | 68 в одной сессии | 5–20 на job |
| Files per job | 45 total в одном контексте | 2–10 per job |
| Final synthesis | Требовался от Qwen | Делает senior |
| Main failure | truncated final answer | один over-broad job failed |
| Evidence verification | Тяжелее | Существенно проще |
| Practical usefulness | limited | заметно выше |

Новый подход медленнее по wall-clock при последовательном запуске (суммарно ~6–7 мин
с retry), но routing качественнее: каждая задача возвращает отдельный проверяемый artifact.

## Routing Rule (подтверждённая гипотеза)

**LOCAL (junior):** SEARCH / FIND / TRACE / VERIFY / EXTRACT / bounded inspection —
найти файлы/классы, собрать imports/usages, проследить один bounded flow, проверить
наличие тестов, проверить конкретный DAO/query/migration, извлечь Gradle dependencies,
собрать DI wiring facts, сравнить конкретные code fragments.

**SENIOR:** DESIGN / DECIDE / PRIORITIZE / SYNTHESIZE / architecture assessment — общий
architecture review, prioritization, modernization strategy, severity ranking,
ambiguous root-cause analysis, synthesis по нескольким подсистемам, длинный report с
большим output contract, задачи без жёсткого stop condition.

**Исключение**: VERIFY migrations нужно давать очень узко. Формулировка "найди
потенциальные случаи потери данных" всё ещё слишком широкая для junior. Лучше:
"Read MainDB.kt and DatabaseMigrations.kt only. Return max 5 concrete migration risks. Stop."

## Final Verdict

**JUNIOR-READY.** Не READY FOR MCP — для MCP нужен более жёсткий machine-enforced
contract (см. следующий раздел / план).

---

## Следующий шаг: MCP v0.1 + Skill (архитектурный план)

Итог обсуждения: пора перестать делать battle-tests ради проверки Qwen и начать
собирать конечную систему. Нужны оба компонента:

- **MCP** — даёт Codex/Claude доступ к локальной Qwen: изменить файлы, запустить
  compile/tests, вернуть diff.
- **Skill** — учит старшую модель, когда делегировать, как поставить junior-задачу и
  как принять результат.

`MCP = capability. Skill = policy. Direct Agent = runtime. Qwen = junior executor.
Codex/Claude = senior.`

### Целевой flow

```
USER → CODEX/CLAUDE (senior) анализирует
  → если решение определено → Change Contract (~300-800 токенов)
  → MCP local_implement → local Qwen (читает, правит, compile, tests, acceptance checks)
  → короткий отчёт + structured diff
  → senior проверяет контракт → ACCEPT / REVISE / TAKE_OVER
```

### Экономия

Убираем самый бессмысленный расход: дорогая модель печатает сотни строк Kotlin.
Senior пишет Change Contract (500-800 output токенов) + review (300-700), а не полную
реализацию (~4000+ токенов). Точная экономия по input/reasoning не гарантирована, но
output-токены senior сокращаются радикально.

### Этап 1 — довести Direct Runtime (инженерные safeguards, не исследование)

1. **Machine-enforced budgets**: `max_steps`, `max_tool_calls`, `max_files_read`,
   `max_output_bytes`, `timeout` — в коде, не в prompt. При приближении к лимиту —
   `BUDGET_LOW: Stop exploration and finish the requested task`, с зарезервированным
   последним turn на finalization (уже частично сделано: harness теперь возвращает
   `status=incomplete` + exit code 2 при `finish_reason=length`, см.
   [direct-agent.mjs](direct-agent.mjs)).
2. **Runtime-owned statistics**: harness сам считает `steps/tool_calls/files_read/duration_ms`,
   Qwen не должна сама писать эти числа в ответ (сейчас иногда расходятся).
3. **Evidence ledger**: `record_finding` / `record_uncertainty` — findings сохраняются
   по мере появления, даже если final response сорвался.

### Этап 2 — write-mode через worktree

Local модель никогда не пишет в рабочую ветку пользователя напрямую — только во
временный worktree. `local_implement` создаёт worktree → Qwen меняет код → compile →
tests → `git diff` → возвращает результат. Основной working tree untouched.

### Этап 3 — MCP v0.1 (три инструмента)

- **`local_find`** — дешёвая разведка (interface/impl/usages/DI/tests).
- **`local_verify`** — проверка конкретной гипотезы, возвращает
  `VERIFIED / NOT_VERIFIED / INCONCLUSIVE` + evidence.
- **`local_implement`** — самый ценный: senior передаёт Change Contract (goal, scope.allowed_files,
  changes, invariants, acceptance), runtime сам проверяет:
  - **scope**: `git diff --name-only` сверяется с `allowed_files` (не доверять
    утверждению Qwen "я ничего лишнего не трогал");
  - **compile**: runtime сам запускает `./gradlew :module:compileDebugKotlin`, а не
    принимает "всё компилируется" на слово;
  - **tests**: аналогично, runtime запускает и репортит exit code.

  Результат — компактный JSON (status, changed_files, changes, acceptance{scope,compile,
  tests}, deviations, diff_stat), а не полотно текста.

### Этап 4 — Skill `local-junior-delegation` (один канонический skill для Codex и Claude)

Основное правило: если решение уже определено и выражается коротким Change Contract с
объективными acceptance criteria — не писать реализацию самому, вызвать `local_implement`.

Decision gate перед генерацией кода: (1) желаемое поведение известно? (2) scope понятен?
(3) можно описать без полного кода? (4) есть invariants? (5) результат проверяем
compile/test/diff/search? Если (4)-(5) да → DELEGATE, иначе senior продолжает сам.

После результата — только три состояния:
- **ACCEPT** — контракт выполнен, checks passed.
- **REVISE** — senior пишет короткую delta-инструкцию (не переписывает класс целиком)
  и снова вызывает local.
- **TAKE_OVER** — контракт был неверный или нужно архитектурное решение — senior
  забирает задачу обратно.

### Этап 5 — eval для senior-routing (не для Qwen — её уже проверили)

Новый предмет: правильно ли senior-модель делегирует.
- **Тест 1**: задача, которую надо делегировать (протянуть параметр через слои +
  тест) — senior не должен сам выводить весь Kotlin, должен дать Change Contract.
- **Тест 2**: задача, которую нельзя сразу делегировать (расплывчатый баг про stale
  state) — senior должен сначала исследовать (`local_find`/`local_verify`), не звать
  `local_implement` раньше времени.
- **Тест 3**: local ошибся / нарушил invariant — senior должен ответить REVISE с
  короткой delta, не переписывать код сам.

### Целевая структура репозитория

```
local-junior/
├── runtime/        (llama-client, agent-loop, budget-manager, filesystem-guard, evidence-ledger, worktree-manager)
├── tools/          (read, grep, find, edit, run-check)
├── workers/        (find, verify, implement)
├── mcp/server
├── skills/local-junior-delegation/SKILL.md
└── evals/          (flightlogbook, verify, implementation, routing)
```

### Приоритеты

| Приоритет | Что делаем | Зачем |
|---|---|---|
| P0 | Worktree + filesystem guard | безопасный write |
| P0 | Machine budgets | Qwen не уходит в бесконечность |
| P0 | Acceptance runner (compile/tests как runtime-факт) | объективное DONE |
| P1 | `local_implement` | основная экономия output tokens |
| P1 | `local_find` / `local_verify` | дешёвая помощь senior |
| P1 | MCP server | подключение к Codex/Claude |
| P1 | Delegation Skill | правильный routing |
| P2 | Codex eval / Claude eval | проверка автоматического делегирования |
| P3 | evidence ledger | повышает reliability |
| P3 | разные local models по типам задач | оптимизация скорости |

**Следующая разработческая задача**: MCP v0.1 + `local_implement` + первый
`local-junior-delegation` Skill. После этого FlightLogbook снова используется — но уже
для проверки, умеет ли Codex/Claude самостоятельно перейти от анализа к короткому
Change Contract и отдать реализацию локальной модели.
