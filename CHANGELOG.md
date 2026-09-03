# Changelog

All notable changes to Llama Server Studio are documented here.

The project does not yet use formal semantic versioning; entries are grouped by date and describe the accumulated feature work in each release.

## 2026-09-03 (v1.6.0)

### Added
- Full UI redesign: replaced the left-panel / right-tab layout with a nav-rail shell — header (Preset save + language), persistent launch controls, an icon nav-rail driving 9 stacked pages (Dashboard, Model Library, Integration, Benchmark, AutoTune, Launch, Sampling, Server, Paths), and a collapsible bottom log dock with a maximize toggle, Copy-logs button, and state persisted in QSettings.
- Bottom server ControlStrip (Start / Restart / Stop / Force Stop) placed between the page content and the log dock.
- Toast overlay for warning/error log messages (transient, top-center, fade-out), reusing the existing log stream.
- `llama_autotuner` v2 engine integrated as the AutoTune backend (vendored models/session/tuning/llama/hardware/benchmark/report modules), replacing the legacy benchmark/report modules.
- Split multi-shard GGUF models (`NAME-NNNNN-of-NNNNN`) are now detected and listed once with the correct combined size, instead of once per shard with only the first shard's size feeding the VRAM/context advisors.
- Passive update check: on startup, the installed CUDA 12 and CUDA 13 llama.cpp builds are compared against the latest GitHub release (no download) and per-build update availability is surfaced with a progress indicator.
- EXTRA (unmanaged) CLI flags are now isolated from the tracked UI fields and preserved verbatim through CLI import/export/save instead of being silently dropped or rewritten.

### Changed
- Header simplified to language-only; the Profile control was removed and Preset is now the sole save mechanism, relocated to the launch bar with a model/params readout.
- Launch/Sampling pages reorganized: context, vision (mmproj), and CUDA moved to Launch; GPU offload, KV cache type, Attention/Fit, and the Memory (KV-cache) panel moved to Sampling.
- Removed the pre-launch "GPU capacity" VRAM bar — llama.cpp only reports VRAM after the model loads, so it was always empty before launch.
- AutoTune GUI simplified to match upstream's one-click workflow: Goal / Search depth / Priority / Max time / Max runs controls were removed (hardcoded to quick / balanced / 8 min / 12 runs), the 3-way degradation choice collapsed into a single "Exact target only" checkbox, the Min PP and Absolute VRAM floor fields were dropped (engine-internal defaults only), and the vision projector override moved to Advanced and now correctly defaults to off instead of silently promoting to REQUIRED whenever an mmproj file happened to be auto-detected.
- `MainWindowUI.set_model_list()` is now the single entry point that repopulates both the Launch model combo and the AutoTune model picker, replacing the previous ad-hoc dual population that could leave one of them stale after a scan.
- `main_window.py` reduced from 1812 to 730 lines by extracting each nav page into its own class under `src/ui/panels/`.

### Fixed
- Maximizing the log dock no longer hides the entire nav-rail and page content — only the content area shrinks, and clicking the already-selected nav item restores normal proportions.
- Multi-shard GGUF models no longer appear duplicated in the model list, and VRAM/context advisors now see the correct combined model size.

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
