param(
  [string] $OutDir = "",
  [string] $ModelAlias = "qwen-27b",
  [string] $BaseUrl = "http://127.0.0.1:8080/v1"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "direct-flightlogbook-out-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$task = @'
Проведи глубокий технический аудит этого Android-проекта.

Не делай поверхностный обзор README. Исследуй реальный код и зависимости.

Нужно выяснить:

A. НАЗНАЧЕНИЕ ПРОЕКТА

* Что делает приложение.
* Какие основные пользовательские сценарии реализованы.
* Какие функциональные области существуют.
* Какие части выглядят завершёнными, экспериментальными или заброшенными.

B. КАРТА АРХИТЕКТУРЫ

Построй реальную архитектурную карту проекта:

* application entry point;
* presentation layer;
* domain layer;
* data layer;
* repositories;
* database;
* dependency injection;
* navigation;
* background/utility/file processing code.

Для каждой области указывай конкретные файлы и классы.

Не делай вывод о слое только из имени директории — проверь зависимости в коде.

C. DATA FLOW

Выбери минимум два важных пользовательских сценария, предпочтительно:

1. работа с полётами;
2. статистика или импорт/экспорт/работа с файлами.

Для каждого проследи цепочку настолько далеко, насколько позволяет код:

UI
→ Presenter/ViewModel
→ domain/use case/interactor
→ repository
→ database/file/storage

Укажи конкретные классы и файлы.

D. DATABASE

Проанализируй Room/data storage:

* entities;
* DAO;
* database;
* migrations;
* repository access;
* потенциальные риски миграций;
* потенциальные проблемы consistency/transactions;
* наличие или отсутствие тестов миграций.

E. ASYNC MODEL

Определи, как используются:

* Kotlin Coroutines;
* RxJava;
* synchronous code.

Проверь, существуют ли одновременно несколько asynchronous paradigms.

F. PRESENTATION ARCHITECTURE

Определи фактический UI architecture:

* MVP/Moxy;
* ViewModel;
* Fragment;
* другие подходы.

Проверь, смешиваются ли несколько архитектурных моделей и каким образом.

G. DEPENDENCY INJECTION

Разбери DI:

* используемый framework;
* components/modules;
* scopes;
* injection entry points;
* основные dependency chains.

H. ТЕХНИЧЕСКИЙ ДОЛГ

Найди реальные проблемы, разделив их на CRITICAL, HIGH, MEDIUM, LOW.

Для каждого существенного finding укажи:

* severity;
* файл;
* класс/метод;
* доказательство;
* почему это проблема;
* confidence: high / medium / low.

I. ТЕСТЫ

Определи:

* какие тесты существуют;
* какие важные части покрыты;
* какие критичные области не покрыты;
* какие 5–10 тестов дали бы максимальную отдачу.

J. ЗАВИСИМОСТИ И СТЕК

Составь карту основных технологий и библиотек.

K. МОДЕРНИЗАЦИЯ

Предложи пошаговый modernization plan:

Phase 1 — low-risk cleanup;
Phase 2 — architectural modernization;
Phase 3 — optional major redesign.

L. ЧТО НЕ НУЖНО МЕНЯТЬ

Отдельно найди части старой архитектуры, которые выглядят вполне рабочими и которые не имеет смысла переписывать только ради современности.

M. ИТОГОВАЯ ОЦЕНКА

Поставь оценки 0–10:

* architecture;
* maintainability;
* testability;
* data layer;
* UI architecture;
* dependency management;
* modernization difficulty.

EVIDENCE RULES:

Не выдумывай файлы, классы или связи.
Если утверждение основано на коде, указывай путь к файлу и по возможности строку/символ.
Если что-то не удалось подтвердить, помести это в UNCERTAINTIES.
Не пытайся прочитать каждый файл подряд.
Сначала построй карту, затем углубляйся в наиболее важные ветки.

REQUIRED OUTPUT:

1. Executive summary
2. Project purpose
3. Architecture map
4. Major features
5. Two traced data flows
6. Database/storage analysis
7. Presentation analysis
8. Async/concurrency analysis
9. DI analysis
10. Tests analysis
11. Technical-debt findings
12. Modernization roadmap
13. What should NOT be rewritten
14. Architecture scorecard
15. Relevant files
16. Uncertainties
17. Analysis statistics: tool calls, files inspected, steps, duration if available
'@

$jsonOut = Join-Path $OutDir "direct-output.json"
$traceOut = Join-Path $OutDir "trace.json"
$stderrOut = Join-Path $OutDir "stderr.txt"
$repoWorkdir = Join-Path $OutDir "repo-workdir"

node (Join-Path $PSScriptRoot "direct-agent.mjs") `
  --repo "https://github.com/arnyigor/flightlogbook" `
  --repo-workdir $repoWorkdir `
  --task $task `
  --base-url $BaseUrl `
  --model $ModelAlias `
  --max-steps 24 `
  --max-tool-calls 90 `
  --max-tokens 8192 `
  --trace-out $traceOut `
  1> $jsonOut 2> $stderrOut

$exitCode = $LASTEXITCODE
$raw = Get-Content -LiteralPath $jsonOut -Raw
$parsed = if ([string]::IsNullOrWhiteSpace($raw)) { $null } else { $raw | ConvertFrom-Json }

$summary = [ordered]@{
  outDir = $OutDir
  exitCode = $exitCode
  status = $parsed.status
  answerChars = if ($parsed.answer) { $parsed.answer.Length } else { 0 }
  stats = $parsed.stats
  repo = $parsed.repo
}

$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $OutDir "summary.json") -Encoding UTF8
$summary | ConvertTo-Json -Depth 20
