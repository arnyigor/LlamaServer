import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function requiredEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function assistantText(messages) {
  const lastAssistant = [...messages].reverse().find(message => message.role === "assistant");
  if (!lastAssistant || !Array.isArray(lastAssistant.content)) return "";
  return lastAssistant.content
    .filter(part => part.type === "text")
    .map(part => part.text)
    .join("");
}

const packageDir = requiredEnv("PI_PACKAGE_DIR");
const agentDir = requiredEnv("PI_AGENT_DIR");
const provider = requiredEnv("PI_PROVIDER");
const modelId = requiredEnv("PI_MODEL");
const cwd = requiredEnv("PI_CWD");
const outPath = requiredEnv("PI_SDK_OUT");
const prompt = requiredEnv("PI_PROMPT");
const toolsValue = process.env.PI_SDK_TOOLS || "";
const tools = toolsValue.trim()
  ? toolsValue.split(",").map(value => value.trim()).filter(Boolean)
  : undefined;

const pi = await import(pathToFileURL(path.join(packageDir, "dist", "index.js")).href);
const modelRuntime = await pi.ModelRuntime.create({
  authPath: path.join(agentDir, "auth.json"),
  modelsPath: path.join(agentDir, "models.json"),
});
const model = modelRuntime.getModel(provider, modelId);
if (!model) throw new Error(`Model not found: ${provider}/${modelId}`);

const { session, modelFallbackMessage } = await pi.createAgentSession({
  cwd,
  agentDir,
  modelRuntime,
  model,
  thinkingLevel: "off",
  sessionManager: pi.SessionManager.inMemory(cwd),
  tools,
});

const events = [];
const unsubscribe = session.subscribe(event => {
  events.push(event);
});

const before = {
  all: session.getAllTools().map(tool => tool.name),
  active: session.getActiveToolNames(),
  allDetails: session.getAllTools().map(tool => ({
    name: tool.name,
    source: tool.sourceInfo?.source,
    hasPromptSnippet: Boolean(tool.promptSnippet),
  })),
};

let error;
try {
  await session.prompt(prompt);
} catch (err) {
  error = err instanceof Error ? { message: err.message, stack: err.stack } : { message: String(err) };
} finally {
  unsubscribe();
  session.dispose();
}

const after = {
  all: session.getAllTools().map(tool => tool.name),
  active: session.getActiveToolNames(),
};

const messages = session.state.messages;
const text = assistantText(messages);
const toolResultCount = messages.filter(message => message.role === "toolResult").length;
const toolCallCount = messages
  .filter(message => message.role === "assistant" && Array.isArray(message.content))
  .flatMap(message => message.content)
  .filter(part => part.type === "toolCall")
  .length;

fs.writeFileSync(outPath, JSON.stringify({
  modelFallbackMessage,
  requestedTools: tools,
  before,
  after,
  error,
  assistantText: text,
  toolCallCount,
  toolResultCount,
  eventTypes: events.map(event => event.type),
}, null, 2), "utf8");

if (error) {
  process.exitCode = 1;
}
