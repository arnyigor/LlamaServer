import fs from "node:fs";

function snapshot(pi, eventType) {
  const allTools = pi.getAllTools().map(tool => ({
    name: tool.name,
    source: tool.sourceInfo?.source,
    hasPromptSnippet: Boolean(tool.promptSnippet),
  }));

  const entry = {
    eventType,
    timestamp: new Date().toISOString(),
    all: allTools.map(tool => tool.name),
    active: pi.getActiveTools(),
    allDetails: allTools,
  };

  const outPath = process.env.PI_TOOL_DEBUG_OUT || "pi-tools-debug.json";
  let previous = [];
  try {
    previous = JSON.parse(fs.readFileSync(outPath, "utf8"));
    if (!Array.isArray(previous)) previous = [previous];
  } catch {
    previous = [];
  }

  previous.push(entry);
  fs.writeFileSync(outPath, JSON.stringify(previous, null, 2), "utf8");
}

export default function (pi) {
  pi.on("session_start", () => {
    snapshot(pi, "session_start");
  });

  pi.on("before_agent_start", () => {
    snapshot(pi, "before_agent_start");
  });
}
