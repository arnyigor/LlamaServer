# Llama Server Studio

**A polished desktop control center for running local `llama.cpp` servers on Windows.**

Llama Server Studio helps you launch, tune, monitor, and manage local GGUF models without rebuilding long command lines by hand. It is designed for people who run local coding assistants, agent workflows, research models, long-context models, and experimental `llama.cpp` builds on their own machine.

The app focuses on practical day-to-day operation: select a model, choose the CUDA build, tune performance settings, start the server, watch live stats, save presets, and export the exact CLI when you need reproducibility.

## Highlights

- Native desktop UI for `llama-server.exe`
- GGUF model discovery and metadata inspection
- CUDA 12 / CUDA 13 build selection
- One-click server start, restart, stop, and force-stop
- Full command preview with CLI import/export
- Runtime token, speed, and active-time statistics
- JSON and Markdown stats export
- Model-specific performance presets
- AutoTune-assisted benchmark workflow
- Hugging Face GGUF download manager
- Local model manager with safe deletion
- OpenCode and PI integration helpers (auto context-window injection)
- Diagnostics for failed launches and crashes

## Why It Exists

Running `llama.cpp` directly is powerful, but real-world setups quickly become hard to manage:

- multiple CUDA builds;
- many GGUF models and quantizations;
- long context sizes;
- KV cache tuning;
- speculative decoding experiments;
- custom chat templates;
- tool-calling templates;
- benchmark presets;
- repeated command-line changes;
- runtime token accounting.

Llama Server Studio gives those workflows a stable desktop surface while keeping the generated CLI visible and portable.

## Model Management

The app scans your model directory recursively and lists local `.gguf` models. It reads useful GGUF metadata without loading the full model:

- architecture;
- quantization;
- native context length;
- block/layer count;
- attention metadata;
- embedding size;
- MoE expert metadata;
- QAT and MTP hints when available.

Vision projectors and draft models are handled separately so the main model list stays clean.

## Launch Control

The launch panel covers the settings that usually matter for local server operation:

- host and port;
- CUDA device and split mode;
- GPU layer offload;
- context size;
- threads and batch threads;
- batch and micro-batch size;
- KV cache type;
- parallel slots;
- Flash Attention;
- mlock;
- Jinja and custom chat templates;
- extra uncommon `llama-server` flags (for example `--mmap`, `--cache-prompt`, `--cont-batching`).

Every launch produces a clear `llama-server` command preview.

## CLI Import And Export

The CLI preview is not just a display field.

You can copy a portable command with relative paths, or import an existing `llama-server` command back into the UI. Importing applies launch parameters while keeping the currently selected program path and model path intact, so pasted commands do not accidentally switch your environment.

The importer understands common copied formats, including multi-line Windows commands using `^` continuation and log snippets that start with `Args:`.

## MTP And Speculative Decoding

MTP controls are available manually even when the model name or metadata does not explicitly advertise MTP support. This is useful for experimental GGUF packages and custom builds where capability is known by the user but not visible in the filename.

When enabled, the app emits speculative decoding flags only when the relevant fields are enabled and populated. Empty optional draft fields are omitted.

Supported controls include:

- `--spec-type draft-mtp`;
- draft model path;
- draft token limit;
- draft probability threshold;
- draft GPU layers;
- draft device.

## Presets

Performance presets are saved per model. The default preset follows the selected context size, while named presets can represent workflows such as:

- coding;
- RAG;
- long-context reading;
- fast chat;
- conservative memory mode;
- speculative decoding experiments.

Named presets are managed with explicit Add and Delete actions. The preset selector itself is read-only, which keeps accidental names and typos out of normal operation.

## Runtime Stats

The runtime stats panel tracks live server activity:

- current speed;
- total tokens;
- current task tokens;
- prompt tokens;
- generated tokens;
- request tokens;
- saved task totals;
- active model time;
- current request time.

Stats can be exported as JSON for automation or copied as Markdown for reports, notes, and comparisons.

## AutoTune And Benchmarking

The built-in benchmark workflow helps compare launch parameters against a selected model and context size. AutoTune can generate candidate settings, run benchmarks in the background, show deltas, and save the best result as a preset.

This is intended for practical tuning rather than synthetic leaderboard work: the goal is to find settings that fit your model, GPU, memory budget, and workflow.

## Hugging Face Downloads

The local model manager includes a Hugging Face GGUF download workflow:

- scan a repository;
- filter quantizations;
- include vision projectors when needed;
- download multiple selected files;
- pause or cancel active downloads;
- resume partial `.part` downloads;
- store files in a clean local folder layout.

Downloaded models are compatible with common local model folder structures.

## Integrations

Llama Server Studio can help add the selected local server model to external tool configs:

- OpenCode;
- PI.

When adding a model, the app writes the server's context window into the agent
config as `limit.context` (auto-detected from a running server via `/slots`, or
manually in the "Max context" field). This lets the agent compact context
correctly instead of abruptly cutting generation.

The app keeps integration editing separate from server launch settings, so you can tune local inference first and then publish the selected endpoint to your tools.

## Diagnostics

Failed starts are captured with a readable diagnostic summary and a full report. Diagnostics include the command, environment, process output, exit state, and likely cause when the app can identify it.

The UI also exposes quick actions to copy the last error or open the diagnostics folder.

## Privacy

Llama Server Studio is local-first. Models run through your local `llama.cpp` build, and runtime statistics are collected from your local server endpoints and process logs.

Network access is used only for features that explicitly need it, such as Hugging Face repository scans/downloads or update checks.

## Releases

Use the packaged executable from the project releases. The desktop app is intended to be used directly; building from source is not required for normal use.
