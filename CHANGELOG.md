# Changelog

All notable changes to Llama Server Studio are documented here.

The project does not yet use formal semantic versioning; entries are grouped by date and describe the accumulated feature work in each release.

## 2026-08-18

### Added
- Integration context-window injection for OpenCode and PI: the selected server's context window is written into the agent config as `limit.context` (auto-detected from a running server via `/slots`, or set manually in the "Max context" field) so agents compact context correctly instead of abruptly cutting generation.
- Launch-timeout watchdog for benchmark/AutoTune runs: a run is aborted with a clear `failed_timeout` status only if the server process produces zero output within the configured timeout; once output appears the run continues to completion.
- AutoTune controls moved into the Benchmark tab; the standalone AutoTune tab was removed.

### Changed
- Removed the `mmap`, `cache prompt`, and `continuous batching` UI checkboxes. Their registry specs are neutralized (fields and settings are preserved) and the server CLI no longer emits `--mmap` / `--no-mmap` / `--cache-prompt` / `--no-cache-prompt` / `--no-cont-batching`. These flags can now be passed directly through the extra-params field (for example `--load-mode mmap` or `--cache-prompt`); they are no longer stripped from manually entered extra arguments.
- Monitor tab removed; its memory/VRAM breakdown is now shown in the Overview area.
- Integration editing is kept separate from server launch settings.

### Fixed
- Restored signal connections that were accidentally dropped during the Monitor-tab / AutoTune refactor. The following controls are wired again: CLI Import / Apply / Copy buttons and manual-mode toggle, `exe_path` change handlers (bench auto-detect and CLI preview refresh), model-path copy, all browse dialogs (exe, bench, model dir, OpenCode/PI configs, chat template, MTP draft), and preset Add / Delete / Save buttons.
- CLI import now preserves pasted uncommon flags (such as `--mmap`, `--cache-prompt`, `--no-cont-batching`) through the extra-params field instead of dropping them.
