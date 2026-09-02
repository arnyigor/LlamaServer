# Local Agent GO-1 Results

Date: 2026-09-02

Model under test:

- `Qwen3.8-27B-UD-IQ4_XS.gguf`
- served by `llama.cpp` on `127.0.0.1:8080`
- OpenAI-compatible model alias: `qwen-27b`

## Summary

The local subagent idea is validated without requiring Pi.

`llama.cpp + Qwen IQ4_XS` reliably emits native OpenAI-compatible tool calls when the request includes `tools`. A small direct read-only agent loop can execute those tool calls and return grounded results.

## Results

| Test | Result | Notes |
| --- | ---: | --- |
| Direct UUID | 10/10 PASS | Each run used one native `read_file` tool call and returned the random UUID. |
| Direct synthetic repo | 3/3 PASS | Each run used a `list_directory -> grep -> read_file...` chain and found `BUG_ID=FS-REFRESH-42`. |
| Pi SDK UUID | 10/10 PASS | `createAgentSession({ tools: ["read", "ls", "find", "grep"] })` worked; all 20 wire requests contained `tools`. |
| Pi CLI without extension | FAIL | Pi sent no `tools` in the HTTP request and the system prompt showed `Available tools: (none)`. |
| Pi CLI with diagnostic extension | PASS | Registry/active tools were present and wire requests included native tools. |

## Artifacts

- `direct-agent.mjs` implements a direct read-only agent loop over OpenAI-compatible `/v1/chat/completions`.
- `run-direct-uuid.ps1` runs the direct UUID reliability test.
- `run-direct-synthetic.ps1` runs the synthetic repository exploration test.
- `run-pi-sdk-uuid.ps1` runs the Pi SDK UUID reliability test.
- `run-tool-matrix.ps1`, `sdk-tool-check.mjs`, and `tool-debug-extension.mjs` capture Pi CLI/SDK diagnostics.
- `proxy.mjs` is a local HTTP proxy for wire-level request/response inspection.

Generated run output directories are intentionally not required for the PoC. Re-run the scripts above to recreate them.

## Next Step

Wrap the direct agent loop in an MCP tool such as `local_explore()`. Keep the MCP API backend-neutral so the implementation can later switch between Direct and Pi SDK without changing the parent-agent interface.
