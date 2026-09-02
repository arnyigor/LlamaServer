# Flightlogbook Battle Test

Date: 2026-09-02

Branch: `codex/local-agent-go1-poc`

Target repo: `https://github.com/arnyigor/flightlogbook`

Run directory:
`plans/local-agent-go1/direct-flightlogbook-out-20260902-103538`

Prepared repo:
`plans/local-agent-go1/direct-flightlogbook-out-20260902-103538/repo-workdir/flightlogbook`

## Run Summary

The local read-only Qwen subagent was run through `direct-agent.mjs --repo`.
It cloned the repository itself and performed the main investigation before any
independent inspection.

Raw run result:

- exit code: `0`
- reported status: `completed`
- final model finish reason in trace: `length`
- final answer size: `1567` chars
- model steps: `24`
- tool calls: `68`
- files read: `45`
- total tool output: `223252` bytes
- duration: `275539` ms

The final answer was truncated during section 2 and did not contain the required
sections 3-17. This exposed a harness bug: `direct-agent.mjs` treated a final
assistant message with `finish_reason = "length"` as a completed run. The harness
was updated so future runs report `status = "incomplete"`, include
`finish_reason`, and return exit code `2` for this case.

## Claims Checked

Independent verification was intentionally limited to the most important claims.
The cloned repo was indexed with `G:\Android\plugins\ast-index.exe rebuild`, then
selected files were read directly.

1. VERIFIED: single-module Android project.
   Evidence: `settings.gradle.kts` includes only `:app`.

2. VERIFIED: app purpose is a pilot flight logbook with flights, statistics,
   import/export, and dictionaries.
   Evidence: `FlightEntity`, `StatisticInteractor`, `FilesInteractorImpl`,
   `XlsReader`, `JsonReader`, and README-level project metadata read by the
   agent.

3. VERIFIED: stack is Room + Dagger 2/dagger-android + Moxy MVP + Navigation
   Component + RxJava 2.
   Evidence: `app/build.gradle.kts`, `FlightLogbookApp.kt`, `AppComponent.kt`,
   `BaseMvpPresenter.kt`, and `NavigationActivity.kt`.

4. VERIFIED: ViewModel and coroutines are dependencies but not materially used.
   Evidence: `app/build.gradle.kts` declares ViewModel and coroutine artifacts;
   `ast-index` found zero `ViewModel`, zero `CoroutineScope`, and zero
   `androidx.lifecycle` usages in source.

5. VERIFIED: `FlightDAO.queryFlightsByColor` uses `params MATCH :hexColor` on a
   normal Room entity table.
   Evidence: `FlightDAO.kt` line 47 uses `MATCH`; `FlightEntity.kt` line 9 is a
   normal `@Entity(tableName = "main_table")`, not FTS.

6. VERIFIED: XLS/JSON import clears and rebuilds database content without an
   enclosing Room transaction.
   Evidence: `FilesInteractorImpl.readFile` removes flights/custom fields and
   then reinserts records in loops. No transaction boundary is present there.

7. VERIFIED: migration 12 -> 13 drops `ground_time` and `night_time` values.
   Evidence: `DatabaseMigrations.kt` creates these columns in the new table, but
   the `INSERT ... SELECT` statement copies only date, datetime, log_time, reg_no,
   airplane_type, day_night, ifr_vfr, flight_type, and description.

8. VERIFIED: layer violations exist.
   Evidence: `FlightsInteractor.kt` imports `android.graphics.Color` and app
   `R`; `StatisticInteractor.kt` imports app `R`; data/file-reader utilities use
   utility code physically stored under `presentation/utils`.

9. VERIFIED: tests are only default examples.
   Evidence: only `ExampleUnitTest.kt` and `ExampleInstrumentedTest.kt` exist in
   `app/src/test` and `app/src/androidTest`.

10. VERIFIED: schemas are configured but not present in the clone.
    Evidence: `room.schemaLocation` is set to `app/schemas`, `MainDB` has
    `exportSchema = true`, and `app/schemas` does not exist.

## Misses And Hallucination Check

No clear hallucination was found in the truncated part that was independently
checked. The serious problem is not factual quality, but delivery completeness:
the agent had enough evidence and even wrote useful reasoning into the trace, but
the user-visible answer was cut off before the requested audit sections.

Important issues visible in trace or verification but absent from the visible
final answer because of truncation:

- `DatabaseMigrations.runMigration` swallows exceptions after `printStackTrace`.
- migration 13 -> 14 creates a unique index on `type_table(reg_no)`, which can
  fail on duplicate historical data.
- `signing.properties` is loaded unconditionally in the Gradle file.
- release minification is disabled.
- `FlightDAO.queryFlightsWithOrder(orderby)` attempts dynamic ordering through a
  Room bind parameter, which will not behave like raw SQL `ORDER BY`.

## Subagent Quality Score

- Navigation: 8/10. It cloned correctly, explored Gradle, DI, DB, domain,
  presenters, import/export, and tests.
- Completeness: 2/10. The visible final report failed the requested format and
  ended mid-sentence.
- Factual accuracy: 8/10 on checked claims.
- Evidence quality: 5/10. Trace contains good file evidence, but the final answer
  lacks line references and was truncated.
- Architectural reasoning: 7/10. It identified real layering, migration, import,
  and async architecture risks.
- Noise: 4/10. Reasoning trace has many useful notes, but also spends tokens on
  verbose planning and repeated checks.
- Usefulness: 6/10. Useful as an investigator, not yet reliable as a standalone
  final-report subagent.

Verdict: PROMISING.

It is not READY FOR MCP yet. The next gate should require chunked/summarized final
reporting or a two-phase protocol: collect evidence first, then emit a compact
structured answer under a strict output budget.
