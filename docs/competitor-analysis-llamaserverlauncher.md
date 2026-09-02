# Конкурентный анализ: LlamaServerLauncherAvalonia

Дата: 2026-08-31
Источник: https://github.com/pytraveler/LlamaServerLauncherAvalonia (MIT, C#/.NET 8/Avalonia, ~70⭐, релизы каждые 1-3 дня)

## Резюме

Зрелый, agressively feature-complete кроссплатформенный лаунчер llama-server.
Сравнение с текущим состоянием Llama Server Studio (этот проект, Windows-only,
PySide) — что стоит перенять, что уже сильнее у нас, и приоритеты.

MIT-лицензия разрешает форк/заимствование идей без ограничений (кроме сохранения
copyright-уведомления в скопированном коде — но код мы не переиспользуем,
стек другой: C#/Avalonia vs Python/PySide).

## Чего у конкурента нет у нас (gap-лист)

1. **Мультиинстанс + Scenarios** — несколько запущенных серверов одновременно
   (разные профили), очередь профилей с таймингами/авто-стартом.
2. **On-Demand Proxy (OpenAI-совместимый)** — прокси сам поднимает нужный
   профиль по полю `model` в запросе, выгружает предыдущий, авто-выгрузка по
   простою. Ценно для сценариев OpenCode/PI — множество моделей через один
   эндпоинт.
3. **MCP-серверы per-profile** — генерация `mcp.json`, импорт из
   Cursor/Claude Desktop, тест инструментов до загрузки модели.
4. **Диагностика крашей с расшифровкой exit code** — `STATUS_ACCESS_VIOLATION`
   и т.п. с вероятной причиной, проверка версий VC++ Redistributable,
   детектор паттерна `--no-mmap`-креша.
5. **Feature detection через `--help`** — парсинг вывода бинарника, гашение в
   UI неподдерживаемых флагов вместо падения на старте.
6. **Hardware Monitor** — CPU/RAM/GPU/VRAM/температура, multi-GPU, AMD
   (rocm-smi) в дополнение к NVIDIA.
7. **Log Stream Server** — WebSocket + HTTP API для удалённого мониторинга
   логов, отдельная HTML-страница просмотра.
8. **Benchmark comparison window** — сравнение прогонов бок-о-бок как
   Markdown-таблицы, именованные наборы сравнения, экспорт отчётов.
9. **Trust/security слой** — build provenance attestation, подписанные
   SHA256SUMS, GPG-ключ в репозитории.
10. **Кроссплатформенность** — Windows/Linux/macOS против нашего Windows-only.
11. **Экспорт профилей** в .bat/.sh/.command + ZIP всех профилей, drag&drop
    произвольных форматов на окно.
12. **Auto-start приложения в ОС**, системный трей с полным набором
    управления по каждому инстансу.

## Где мы уже сильнее

- **CLI import/export** — понимание multi-line Windows `^`-continuation и
  лог-сниппетов `Args:` (у конкурента проще).
- **OpenCode/PI интеграция с авто-инъекцией `limit.context`** — специфично
  под coding-agent workflow, у конкурента такого нет вообще.
- **MTP controls вручную** даже без метаданных модели — гибче их подхода.
- **Детализация runtime-статистики** (current task tokens, saved task
  totals, active model time) — глубже их простого tok/s + slots.
- **VRAM-прогноз до старта** (`llama_autotuner/llama/fit_oracle.py`,
  `tuning/static_memory.py`, `tuning/vram.py`) — если даёт точный прогноз
  без OOM-проб, это сильный дифференциатор против их probe-based HPO.

## Приоритеты (impact vs effort)

| # | Фича | Effort | Impact | Заметки |
|---|------|--------|--------|---------|
| 1 | Crash-advisor с расшифровкой exit code + проверка VC++ Redist | низкий | средний-высокий | ложится на `src/core/diagnostics.py` |
| 2 | Feature-detection по `llama-server --help` | низкий-средний | средний-высокий | ложится на `src/core/param_registry.py` |
| 3 | Подписанные релизы (checksums + attestation) | низкий | средний (доверие) | через GitHub Actions |
| 4 | On-Demand Proxy (OpenAI-совместимый) | высокий | высокий | требует мультиинстанс-основы, которой пока нет в `server_manager.py` |
| 5 | Мультиинстанс | высокий | высокий | предпосылка для #4 и Scenarios |

## Следующий шаг

Начать с #1 (crash-advisor) — минимальный риск, быстрый заметный эффект,
не требует архитектурных изменений.
