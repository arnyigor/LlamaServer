param(
  [string] $OutDir = "",
  [int] $ProxyPort = 8081,
  [int] $UpstreamPort = 8080,
  [string] $ModelAlias = "qwen-27b"
)

$ErrorActionPreference = "Stop"

function Write-Utf8Json {
  param(
    [Parameter(Mandatory = $true)] [string] $Path,
    [Parameter(Mandatory = $true)] $Value
  )
  $Value | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Invoke-JsonPost {
  param(
    [Parameter(Mandatory = $true)] [string] $Uri,
    [Parameter(Mandatory = $true)] $Body,
    [int] $TimeoutSec = 120
  )
  $json = $Body | ConvertTo-Json -Depth 50
  Invoke-RestMethod -Uri $Uri -Method Post -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSec
}

function Invoke-RawSsePost {
  param(
    [Parameter(Mandatory = $true)] [string] $Uri,
    [Parameter(Mandatory = $true)] $Body,
    [Parameter(Mandatory = $true)] [string] $OutFile
  )
  $json = $Body | ConvertTo-Json -Depth 50
  $bodyFile = Join-Path ([System.IO.Path]::GetTempPath()) ("pi-go1-body-" + [guid]::NewGuid().ToString("N") + ".json")
  try {
    Set-Content -LiteralPath $bodyFile -Value $json -Encoding UTF8
    curl.exe -sS -N `
      -H "Content-Type: application/json" `
      -X POST `
      --data-binary "@$bodyFile" `
      $Uri *> $OutFile
  } finally {
    Remove-Item -LiteralPath $bodyFile -ErrorAction SilentlyContinue
  }
}

$root = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $root
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "out-$stamp"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$piVersion = (& pi --version 2>&1 | Out-String).Trim()
$modelsJsonPath = Join-Path $HOME ".pi\agent\models.json"
$settingsJsonPath = Join-Path $HOME ".pi\agent\settings.json"
$modelsJson = Get-Content -LiteralPath $modelsJsonPath -Raw | ConvertFrom-Json
$settingsJson = Get-Content -LiteralPath $settingsJsonPath -Raw | ConvertFrom-Json

$llamacppProvider = $modelsJson.providers.llamacpp
$baseline = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  piVersion = $piVersion
  serverModels = Invoke-RestMethod -Uri "http://127.0.0.1:$UpstreamPort/v1/models" -Method Get -TimeoutSec 10
  globalLlamacppProvider = $llamacppProvider
  globalSettings = [ordered]@{
    defaultProvider = $settingsJson.defaultProvider
    defaultModel = $settingsJson.defaultModel
    defaultThinkingLevel = $settingsJson.defaultThinkingLevel
    enabledModels = $settingsJson.enabledModels
  }
}
Write-Utf8Json -Path (Join-Path $OutDir "baseline.json") -Value $baseline

$toolSchema = @(
  @{
    type = "function"
    function = @{
      name = "read"
      description = "Read a file from disk"
      parameters = @{
        type = "object"
        properties = @{
          path = @{
            type = "string"
            description = "Path to the file"
          }
        }
        required = @("path")
      }
    }
  }
)

$manualBodyBase = [ordered]@{
  model = $ModelAlias
  messages = @(
    @{
      role = "system"
      content = "You are a terse tool-use test endpoint. If a tool is needed, use native tool calls."
    },
    @{
      role = "user"
      content = "Use the read tool to read proof.txt."
    }
  )
  tools = $toolSchema
  tool_choice = "auto"
  temperature = 0
  max_tokens = 256
}

$manualNonStreamBody = [ordered]@{} + $manualBodyBase
$manualNonStreamBody.stream = $false
Write-Utf8Json -Path (Join-Path $OutDir "manual-non-stream-request.json") -Value $manualNonStreamBody
$manualNonStreamResponse = Invoke-JsonPost -Uri "http://127.0.0.1:$UpstreamPort/v1/chat/completions" -Body $manualNonStreamBody
Write-Utf8Json -Path (Join-Path $OutDir "manual-non-stream-response.json") -Value $manualNonStreamResponse

$manualStreamBody = [ordered]@{} + $manualBodyBase
$manualStreamBody.stream = $true
Write-Utf8Json -Path (Join-Path $OutDir "manual-stream-request.json") -Value $manualStreamBody
Invoke-RawSsePost -Uri "http://127.0.0.1:$UpstreamPort/v1/chat/completions" -Body $manualStreamBody -OutFile (Join-Path $OutDir "manual-stream-response.sse")

$isolatedAgentDir = Join-Path $OutDir "pi-agent-dir"
New-Item -ItemType Directory -Force -Path $isolatedAgentDir | Out-Null

$isolatedModels = @{
  providers = @{
    "llamacpp-test" = @{
      baseUrl = "http://127.0.0.1:$ProxyPort/v1"
      api = "openai-completions"
      apiKey = "local"
      compat = @{
        supportsDeveloperRole = $false
        supportsReasoningEffort = $false
        maxTokensField = "max_tokens"
      }
      models = @(
        @{
          id = $ModelAlias
          name = "Qwen3.8 27B IQ4_XS via debug proxy"
          reasoning = $false
          contextWindow = 65536
          maxTokens = 8192
          cost = @{
            input = 0
            output = 0
            cacheRead = 0
            cacheWrite = 0
          }
        }
      )
    }
  }
}
Write-Utf8Json -Path (Join-Path $isolatedAgentDir "models.json") -Value $isolatedModels
Write-Utf8Json -Path (Join-Path $isolatedAgentDir "settings.json") -Value @{
  defaultProvider = "llamacpp-test"
  defaultModel = $ModelAlias
  defaultThinkingLevel = "off"
  enabledModels = @("llamacpp-test/$ModelAlias")
  quietStartup = $true
}

$proofDir = Join-Path $OutDir "proof"
New-Item -ItemType Directory -Force -Path $proofDir | Out-Null
$token = [guid]::NewGuid().ToString()
Set-Content -LiteralPath (Join-Path $proofDir "proof.txt") -Value "SECRET_TOKEN=$token" -NoNewline -Encoding UTF8

$proxyScript = Join-Path $PSScriptRoot "proxy.mjs"
$proxyLogDir = Join-Path $OutDir "proxy"
New-Item -ItemType Directory -Force -Path $proxyLogDir | Out-Null

$nodeCmd = Get-Command node -ErrorAction Stop
$proxyEnv = @{
  LOG_DIR = $proxyLogDir
  PROXY_PORT = [string]$ProxyPort
  UPSTREAM_PORT = [string]$UpstreamPort
}

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $nodeCmd.Source
$psi.Arguments = '"' + $proxyScript + '"'
$psi.WorkingDirectory = $repoRoot
$psi.UseShellExecute = $false
$psi.RedirectStandardError = $true
$psi.RedirectStandardOutput = $true
foreach ($entry in $proxyEnv.GetEnumerator()) {
  $psi.Environment[$entry.Key] = $entry.Value
}

$proxyProcess = [System.Diagnostics.Process]::Start($psi)
Start-Sleep -Milliseconds 700

try {
  Push-Location $proofDir
  try {
    $env:PI_CODING_AGENT_DIR = $isolatedAgentDir
    $prompt = "Open proof.txt using the read tool. Return exactly the SECRET_TOKEN value and nothing else. Do not guess."
    $piOutputPath = Join-Path $OutDir "pi-output.jsonl"
    pi --offline --approve --mode json --provider llamacpp-test --model $ModelAlias --thinking off --tools read,ls,find,grep --no-skills --no-context-files --no-session -p $prompt *> $piOutputPath
    $piExitCode = $LASTEXITCODE
  } finally {
    Remove-Item Env:\PI_CODING_AGENT_DIR -ErrorAction SilentlyContinue
    Pop-Location
  }
} finally {
  if (-not $proxyProcess.HasExited) {
    $proxyProcess.Kill()
    $proxyProcess.WaitForExit()
  }
  $proxyProcess.StandardError.ReadToEnd() | Set-Content -LiteralPath (Join-Path $OutDir "proxy-stderr.txt") -Encoding UTF8
  $proxyProcess.StandardOutput.ReadToEnd() | Set-Content -LiteralPath (Join-Path $OutDir "proxy-stdout.txt") -Encoding UTF8
}

$manualStreamRaw = Get-Content -LiteralPath (Join-Path $OutDir "manual-stream-response.sse") -Raw
$piOutputRaw = Get-Content -LiteralPath (Join-Path $OutDir "pi-output.jsonl") -Raw
$wireRequests = Get-ChildItem -LiteralPath $proxyLogDir -Filter "pi-wire-request-*.json" | Sort-Object Name
$wireResponses = Get-ChildItem -LiteralPath $proxyLogDir -Filter "pi-wire-response-*.sse" | Sort-Object Name

$summary = [ordered]@{
  outDir = $OutDir
  expectedToken = $token
  piExitCode = $piExitCode
  manualNonStreamHasToolCalls = [bool]$manualNonStreamResponse.choices[0].message.tool_calls
  manualNonStreamFinishReason = $manualNonStreamResponse.choices[0].finish_reason
  manualStreamHasDeltaToolCalls = ($manualStreamRaw -match '"tool_calls"')
  manualStreamHasFinishToolCalls = ($manualStreamRaw -match '"finish_reason"\s*:\s*"tool_calls"')
  piOutputContainsToken = ($piOutputRaw -match [regex]::Escape($token))
  piOutputHasToolcallEvents = ($piOutputRaw -match '"toolcall_start"|\"toolcall_end\"')
  proxyRequestCount = $wireRequests.Count
  proxyResponseCount = $wireResponses.Count
  proxySawTools = $false
  proxySawStream = $false
  proxySawNativeToolCalls = $false
  proxySawFinishToolCalls = $false
}

foreach ($request in $wireRequests) {
  $requestJson = Get-Content -LiteralPath $request.FullName -Raw | ConvertFrom-Json
  if ($null -ne $requestJson.body.tools) { $summary.proxySawTools = $true }
  if ($requestJson.body.stream -eq $true) { $summary.proxySawStream = $true }
}
foreach ($response in $wireResponses) {
  $raw = Get-Content -LiteralPath $response.FullName -Raw
  if ($raw -match '"tool_calls"') { $summary.proxySawNativeToolCalls = $true }
  if ($raw -match '"finish_reason"\s*:\s*"tool_calls"') { $summary.proxySawFinishToolCalls = $true }
}

Write-Utf8Json -Path (Join-Path $OutDir "summary.json") -Value $summary
$summary | ConvertTo-Json -Depth 10
