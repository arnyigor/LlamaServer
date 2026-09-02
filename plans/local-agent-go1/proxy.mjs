import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const listenPort = Number(process.env.PROXY_PORT ?? "8081");
const upstreamHost = process.env.UPSTREAM_HOST ?? "127.0.0.1";
const upstreamPort = Number(process.env.UPSTREAM_PORT ?? "8080");
const logDir = process.env.LOG_DIR;

if (!logDir) {
  console.error("LOG_DIR is required");
  process.exit(2);
}

fs.mkdirSync(logDir, { recursive: true });

let counter = 0;

function writeJson(file, value) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2), "utf8");
}

const server = http.createServer((req, res) => {
  const id = String(++counter).padStart(3, "0");
  const chunks = [];

  req.on("data", chunk => chunks.push(chunk));
  req.on("end", () => {
    const rawBody = Buffer.concat(chunks);
    const requestHeaders = { ...req.headers };
    delete requestHeaders.authorization;

    let parsedBody = null;
    try {
      parsedBody = rawBody.length ? JSON.parse(rawBody.toString("utf8")) : null;
    } catch {
      parsedBody = rawBody.toString("utf8");
    }

    writeJson(path.join(logDir, `pi-wire-request-${id}.json`), {
      method: req.method,
      url: req.url,
      headers: requestHeaders,
      body: parsedBody,
    });

    const upstreamReq = http.request(
      {
        host: upstreamHost,
        port: upstreamPort,
        method: req.method,
        path: req.url,
        headers: {
          ...req.headers,
          host: `${upstreamHost}:${upstreamPort}`,
        },
      },
      upstreamRes => {
        res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
        const responsePath = path.join(logDir, `pi-wire-response-${id}.sse`);
        const responseStream = fs.createWriteStream(responsePath, { encoding: "utf8" });

        upstreamRes.on("data", chunk => {
          responseStream.write(chunk);
          res.write(chunk);
        });
        upstreamRes.on("end", () => {
          responseStream.end();
          res.end();
        });
      },
    );

    upstreamReq.on("error", error => {
      writeJson(path.join(logDir, `pi-wire-error-${id}.json`), {
        message: error.message,
        stack: error.stack,
      });
      res.writeHead(502, { "content-type": "text/plain; charset=utf-8" });
      res.end(`Proxy error: ${error.message}`);
    });

    upstreamReq.write(rawBody);
    upstreamReq.end();
  });
});

server.listen(listenPort, "127.0.0.1", () => {
  console.error(`proxy listening on http://127.0.0.1:${listenPort}, forwarding to ${upstreamHost}:${upstreamPort}`);
});

process.on("SIGTERM", () => server.close(() => process.exit(0)));
process.on("SIGINT", () => server.close(() => process.exit(0)));
