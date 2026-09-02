#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";

const DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1";
const DEFAULT_MODEL = "qwen-27b";
let lastFailureTrace = null;

function parseArgs(argv) {
  const parsed = {
    baseUrl: DEFAULT_BASE_URL,
    model: DEFAULT_MODEL,
    maxSteps: 20,
    maxToolCalls: 30,
    maxFilesRead: 15,
    maxToolOutputBytes: 1024 * 1024,
    maxFileBytes: 256 * 1024,
    maxGrepMatches: 100,
    maxTokens: 2048,
    finalizeMaxTokens: 8192,
    traceOut: "",
    repoWorkdir: "",
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };

    if (arg === "--cwd") parsed.cwd = next();
    else if (arg === "--repo") parsed.repo = next();
    else if (arg === "--repo-workdir") parsed.repoWorkdir = next();
    else if (arg === "--task") parsed.task = next();
    else if (arg === "--base-url") parsed.baseUrl = next();
    else if (arg === "--model") parsed.model = next();
    else if (arg === "--max-steps") parsed.maxSteps = Number(next());
    else if (arg === "--max-tool-calls") parsed.maxToolCalls = Number(next());
    else if (arg === "--max-files-read") parsed.maxFilesRead = Number(next());
    else if (arg === "--max-tokens") parsed.maxTokens = Number(next());
    else if (arg === "--finalize-max-tokens") parsed.finalizeMaxTokens = Number(next());
    else if (arg === "--trace-out") parsed.traceOut = next();
    else if (arg === "--help" || arg === "-h") parsed.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }

  return parsed;
}

function usage() {
  return [
    "Usage:",
    "  node direct-agent.mjs --cwd <repo-or-dir> --task <task>",
    "  node direct-agent.mjs --repo <git-url> --task <task>",
    "",
    "Options:",
    "  --base-url <url>        OpenAI-compatible base URL (default: http://127.0.0.1:8080/v1)",
    "  --model <id>            Model id/alias (default: qwen-27b)",
    "  --repo-workdir <dir>    Directory where --repo is cloned",
    "  --max-steps <n>         Maximum model turns (default: 20)",
    "  --max-tool-calls <n>    Maximum total tool calls (default: 30)",
    "  --max-files-read <n>    Maximum distinct files read (default: 15)",
    "  --max-tokens <n>        Maximum tokens per model response (default: 2048)",
    "  --finalize-max-tokens <n>  Token budget for the forced finalize turn (default: 8192);",
    "                          should exceed the server's --reasoning-budget so thinking",
    "                          doesn't crowd out the final answer content.",
    "  --trace-out <file>      Write full trace JSON to this file",
  ].join("\n");
}

async function prepareRepo(options) {
  if (!options.repo) return { cwd: options.cwd, repo: null };
  if (options.cwd) throw new Error("Use either --cwd or --repo, not both");

  const repoUrl = String(options.repo);
  if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\.git)?$/.test(repoUrl)) {
    throw new Error(`Only simple GitHub HTTPS repo URLs are supported by --repo: ${repoUrl}`);
  }

  const parent = options.repoWorkdir
    ? path.resolve(options.repoWorkdir)
    : await fs.mkdtemp(path.join(tmpdir(), "direct-agent-repo-"));
  await fs.mkdir(parent, { recursive: true });

  const repoName = repoUrl.replace(/\.git$/, "").split("/").pop();
  const target = path.join(parent, repoName);
  const relativeTarget = path.relative(parent, target);
  if (relativeTarget === "" || relativeTarget.startsWith("..") || path.isAbsolute(relativeTarget)) {
    throw new Error(`Refusing to clone outside repo workdir: ${target}`);
  }
  try {
    await fs.rm(target, { recursive: true, force: true });
  } catch {
    // Best effort cleanup before cloning into the eval work directory.
  }

  const clone = spawnSync("git", ["clone", "--depth", "1", repoUrl, target], {
    encoding: "utf8",
    timeout: 120_000,
    windowsHide: true,
  });
  if (clone.error) throw clone.error;
  if (clone.status !== 0) {
    throw new Error((clone.stderr || `git clone exited with ${clone.status}`).trim());
  }

  return {
    cwd: target,
    repo: {
      url: repoUrl,
      path: target,
      cloneStdout: clone.stdout,
      cloneStderr: clone.stderr,
    },
  };
}

function truncateUtf8(text, maxBytes) {
  const buffer = Buffer.from(String(text), "utf8");
  if (buffer.length <= maxBytes) return String(text);
  return `${buffer.subarray(0, maxBytes).toString("utf8")}\n[truncated to ${maxBytes} bytes]`;
}

function slash(value) {
  return value.split(path.sep).join("/");
}

function globToRegex(glob) {
  let out = "^";
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i];
    const next = glob[i + 1];
    if (ch === "*" && next === "*") {
      out += ".*";
      i++;
    } else if (ch === "*") {
      out += "[^/]*";
    } else if (ch === "?") {
      out += "[^/]";
    } else {
      out += ch.replace(/[|\\{}()[\]^$+?.]/g, "\\$&");
    }
  }
  out += "$";
  return new RegExp(out);
}

class ReadOnlyTools {
  constructor(root, limits) {
    this.root = path.resolve(root);
    this.limits = limits;
    this.rootReal = null;
    this.totalOutputBytes = 0;
    this.filesRead = new Set();
  }

  async init() {
    this.rootReal = await fs.realpath(this.root);
  }

  async resolveInside(inputPath = ".") {
    const raw = String(inputPath || ".");
    const candidate = path.isAbsolute(raw)
      ? path.resolve(raw)
      : path.resolve(this.rootReal, raw);
    const real = await fs.realpath(candidate);
    const relative = path.relative(this.rootReal, real);
    if (relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative))) {
      return { real, relative: relative || "." };
    }
    throw new Error(`Path escapes cwd: ${raw}`);
  }

  recordOutput(value) {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    this.totalOutputBytes += Buffer.byteLength(text, "utf8");
    if (this.totalOutputBytes > this.limits.maxToolOutputBytes) {
      throw new Error(`Total tool output exceeded ${this.limits.maxToolOutputBytes} bytes`);
    }
    return value;
  }

  async readFile(args) {
    const { real, relative } = await this.resolveInside(args.path);
    const relSlash = slash(relative);
    if (!this.filesRead.has(relSlash) && this.filesRead.size >= this.limits.maxFilesRead) {
      throw new Error(
        `Reached max distinct files read (${this.limits.maxFilesRead}). ` +
        `Stop exploring and answer now using what you already read: ${[...this.filesRead].join(", ")}`
      );
    }
    const stat = await fs.stat(real);
    if (!stat.isFile()) throw new Error(`Not a file: ${args.path}`);
    if (stat.size > this.limits.maxFileBytes) {
      throw new Error(`File exceeds ${this.limits.maxFileBytes} bytes: ${slash(relative)}`);
    }

    const text = await fs.readFile(real, "utf8");
    const lines = text.split(/\r?\n/);
    const offset = Math.max(1, Number(args.offset || 1));
    const limit = args.limit === undefined ? undefined : Math.max(1, Number(args.limit));
    const selected = lines.slice(offset - 1, limit ? offset - 1 + limit : undefined);
    this.filesRead.add(slash(relative));
    return this.recordOutput({
      path: slash(relative),
      offset,
      lineCount: selected.length,
      content: selected.join("\n"),
    });
  }

  async listDirectory(args) {
    const { real, relative } = await this.resolveInside(args.path || ".");
    const stat = await fs.stat(real);
    if (!stat.isDirectory()) throw new Error(`Not a directory: ${args.path || "."}`);
    const limit = Math.max(1, Math.min(1000, Number(args.limit || 500)));
    const entries = await fs.readdir(real, { withFileTypes: true });
    const names = entries
      .map(entry => `${entry.name}${entry.isDirectory() ? "/" : ""}`)
      .sort((a, b) => a.localeCompare(b))
      .slice(0, limit);
    return this.recordOutput({ path: slash(relative), entries: names, truncated: entries.length > limit });
  }

  async findFiles(args) {
    const pattern = String(args.pattern || "*");
    const base = await this.resolveInside(args.path || ".");
    const limit = Math.max(1, Math.min(2000, Number(args.limit || 1000)));
    const regex = globToRegex(slash(pattern));
    const results = [];
    const skipDirs = new Set([".git", ".gradle", "build", "node_modules", ".idea"]);

    const walk = async dir => {
      if (results.length >= limit) return;
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (results.length >= limit) break;
        if (entry.isDirectory() && skipDirs.has(entry.name)) continue;
        const full = path.join(dir, entry.name);
        const rel = slash(path.relative(base.real, full));
        if (entry.isDirectory()) {
          await walk(full);
        } else if (regex.test(rel) || regex.test(entry.name)) {
          results.push(slash(path.relative(this.rootReal, full)));
        }
      }
    };

    await walk(base.real);
    return this.recordOutput({ pattern, path: slash(base.relative), matches: results, truncated: results.length >= limit });
  }

  async grep(args) {
    const pattern = String(args.pattern || "");
    if (!pattern) throw new Error("grep.pattern is required");
    const base = await this.resolveInside(args.path || ".");
    const limit = Math.max(1, Math.min(500, Number(args.limit || this.limits.maxGrepMatches)));
    const rgArgs = ["--line-number", "--no-heading", "--color", "never", "--max-count", String(limit)];
    if (args.ignoreCase) rgArgs.push("--ignore-case");
    if (args.literal) rgArgs.push("--fixed-strings");
    if (args.glob) rgArgs.push("--glob", String(args.glob));
    rgArgs.push(pattern, base.real);

    const result = spawnSync("rg", rgArgs, {
      cwd: this.rootReal,
      encoding: "utf8",
      timeout: 30_000,
      windowsHide: true,
    });
    if (result.error) throw result.error;
    if (result.status !== 0 && result.status !== 1) {
      throw new Error((result.stderr || `rg exited with ${result.status}`).trim());
    }

    const lines = (result.stdout || "")
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(0, limit)
      .map(line => {
        const normalized = line.replaceAll("\\", "/");
        const root = slash(this.rootReal);
        return normalized.startsWith(root) ? normalized.slice(root.length + 1) : normalized;
      });
    return this.recordOutput({ pattern, path: slash(base.relative), matches: lines, truncated: lines.length >= limit });
  }
}

const tools = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a UTF-8 text file inside the allowed cwd.",
      parameters: {
        type: "object",
        required: ["path"],
        properties: {
          path: { type: "string", description: "Relative path inside cwd" },
          offset: { type: "number", description: "1-based first line to read" },
          limit: { type: "number", description: "Maximum number of lines" },
        },
      },
      strict: false,
    },
  },
  {
    type: "function",
    function: {
      name: "list_directory",
      description: "List directory entries inside the allowed cwd.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative directory path, default ." },
          limit: { type: "number", description: "Maximum entries, default 500" },
        },
      },
      strict: false,
    },
  },
  {
    type: "function",
    function: {
      name: "find_files",
      description: "Find files by glob pattern inside the allowed cwd.",
      parameters: {
        type: "object",
        required: ["pattern"],
        properties: {
          pattern: { type: "string", description: "Glob pattern such as *.kt or **/*.gradle.kts" },
          path: { type: "string", description: "Relative search directory, default ." },
          limit: { type: "number", description: "Maximum matches, default 1000" },
        },
      },
      strict: false,
    },
  },
  {
    type: "function",
    function: {
      name: "grep",
      description: "Search file contents with ripgrep inside the allowed cwd.",
      parameters: {
        type: "object",
        required: ["pattern"],
        properties: {
          pattern: { type: "string", description: "Regex or literal search pattern" },
          path: { type: "string", description: "Relative file or directory path, default ." },
          glob: { type: "string", description: "Optional file glob filter" },
          ignoreCase: { type: "boolean", description: "Case-insensitive search" },
          literal: { type: "boolean", description: "Treat pattern as a literal string" },
          limit: { type: "number", description: "Maximum matches, default 100" },
        },
      },
      strict: false,
    },
  },
];

function parseToolArguments(raw) {
  if (!raw) return {};
  if (typeof raw === "object") return raw;
  return JSON.parse(raw);
}

async function callModel(options, messages, { toolChoice = "auto", maxTokens } = {}) {
  const body = {
    model: options.model,
    messages,
    temperature: 0,
    max_tokens: maxTokens ?? options.maxTokens,
    stream: false,
  };
  if (toolChoice === "none") {
    body.tool_choice = "none";
  } else {
    body.tools = tools;
    body.tool_choice = "auto";
  }

  const response = await fetch(`${options.baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: "Bearer local",
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${text}`);
  return JSON.parse(text);
}

async function executeTool(toolRunner, call) {
  const name = call.function?.name;
  const args = parseToolArguments(call.function?.arguments);
  if (name === "read_file") return toolRunner.readFile(args);
  if (name === "list_directory") return toolRunner.listDirectory(args);
  if (name === "find_files") return toolRunner.findFiles(args);
  if (name === "grep") return toolRunner.grep(args);
  throw new Error(`Unknown tool: ${name}`);
}

async function main() {
  const startedAt = Date.now();
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  if ((!options.cwd && !options.repo) || !options.task) throw new Error("--cwd or --repo, and --task are required");

  const prepared = await prepareRepo(options);
  options.cwd = prepared.cwd;

  const toolRunner = new ReadOnlyTools(options.cwd, options);
  await toolRunner.init();

  const messages = [
    {
      role: "system",
      content: [
        "You are a read-only local repository explorer.",
        "Use tools when facts require inspecting files. Do not guess file contents.",
        "All paths are relative to the allowed cwd unless a tool result says otherwise.",
        "For the final answer, be concise and include only conclusions supported by tool results.",
      ].join("\n"),
    },
    { role: "user", content: options.task },
  ];

  const trace = [];
  lastFailureTrace = { trace, options };
  let toolCallCount = 0;
  let finalMessage = null;
  let finalFinishReason = null;
  let budgetLimited = false;
  let forceFinalize = false;

  for (let step = 1; step <= options.maxSteps; step++) {
    const isLastStep = step === options.maxSteps;
    const mustFinalizeNow = forceFinalize || isLastStep;

    if (mustFinalizeNow) {
      budgetLimited = true;
      messages.push({
        role: "user",
        content:
          "BUDGET_LOW: You have reached the exploration budget. " +
          "Answer directly now, with minimal or no step-by-step reasoning. " +
          "Give your final concise answer using only what you already found. Do not call any tools.",
      });
    }

    const response = await callModel(options, messages, {
      toolChoice: mustFinalizeNow ? "none" : "auto",
      maxTokens: mustFinalizeNow ? Math.max(options.maxTokens, options.finalizeMaxTokens) : undefined,
    });
    const choice = response.choices?.[0];
    const assistant = choice?.message;
    if (!assistant) throw new Error("Model response did not include choices[0].message");

    trace.push({
      step,
      finishReason: choice.finish_reason,
      assistant,
      usage: response.usage,
    });
    messages.push(assistant);

    const calls = mustFinalizeNow ? [] : (assistant.tool_calls || []);
    if (calls.length === 0) {
      finalMessage = assistant;
      finalFinishReason = choice.finish_reason;
      break;
    }

    for (const call of calls) {
      toolCallCount++;
      if (toolCallCount > options.maxToolCalls) {
        forceFinalize = true;
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          name: call.function?.name,
          content: JSON.stringify({ ok: false, error: `Reached max tool calls (${options.maxToolCalls}); skipped.` }),
        });
        continue;
      }

      let payload;
      try {
        const result = await executeTool(toolRunner, call);
        payload = { ok: true, result };
      } catch (err) {
        payload = { ok: false, error: err instanceof Error ? err.message : String(err) };
      }

      messages.push({
        role: "tool",
        tool_call_id: call.id,
        name: call.function?.name,
        content: truncateUtf8(JSON.stringify(payload), 128 * 1024),
      });
      trace.push({
        step,
        toolCall: {
          id: call.id,
          name: call.function?.name,
          arguments: call.function?.arguments,
          result: payload,
        },
      });
    }
  }

  if (!finalMessage) {
    finalMessage = { content: "" };
    finalFinishReason = "budget_exhausted";
    budgetLimited = true;
  }

  const content = typeof finalMessage.content === "string"
    ? finalMessage.content
    : JSON.stringify(finalMessage.content ?? "");
  const reasoning = typeof finalMessage.reasoning_content === "string" ? finalMessage.reasoning_content : "";
  const usedReasoningFallback = !content.trim() && Boolean(reasoning.trim());
  const text = usedReasoningFallback ? reasoning : content;
  const output = {
    status: finalFinishReason === "length" || (budgetLimited && !text.trim()) ? "incomplete" : "completed",
    finish_reason: finalFinishReason,
    answer: text.trim(),
    answer_source: usedReasoningFallback ? "reasoning_fallback" : "content",
    stats: {
      steps: trace.filter(entry => entry.assistant).length,
      tool_calls: toolCallCount,
      files_read: toolRunner.filesRead.size,
      files_read_paths: [...toolRunner.filesRead].sort(),
      total_tool_output_bytes: toolRunner.totalOutputBytes,
      duration_ms: Date.now() - startedAt,
      budget_limited: budgetLimited,
    },
    repo: prepared.repo ? { url: prepared.repo.url, path: prepared.repo.path } : undefined,
  };

  if (options.traceOut) {
    await fs.mkdir(path.dirname(path.resolve(options.traceOut)), { recursive: true });
    await fs.writeFile(path.resolve(options.traceOut), JSON.stringify({ output, trace, repo: prepared.repo }, null, 2), "utf8");
  }

  console.log(JSON.stringify(output, null, 2));
  if (output.status === "incomplete") process.exitCode = 2;
}

main().catch(async err => {
  const options = lastFailureTrace?.options;
  if (options?.traceOut) {
    try {
      await fs.mkdir(path.dirname(path.resolve(options.traceOut)), { recursive: true });
      await fs.writeFile(path.resolve(options.traceOut), JSON.stringify({
        output: {
          status: "failed",
          error: err instanceof Error ? err.message : String(err),
        },
        trace: lastFailureTrace.trace,
      }, null, 2), "utf8");
    } catch {
      // Preserve the original failure in stderr.
    }
  }
  console.error(JSON.stringify({
    status: "failed",
    error: err instanceof Error ? err.message : String(err),
  }, null, 2));
  process.exitCode = 1;
});
