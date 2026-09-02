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
  $Value | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-IsolatedPiConfig {
  param(
    [Parameter(Mandatory = $true)] [string] $AgentDir,
    [Parameter(Mandatory = $true)] [int] $Port,
    [Parameter(Mandatory = $true)] [string] $Model
  )

  New-Item -ItemType Directory -Force -Path $AgentDir | Out-Null
  Write-Utf8Json -Path (Join-Path $AgentDir "models.json") -Value @{
    providers = @{
      "llamacpp-test" = @{
        baseUrl = "http://127.0.0.1:$Port/v1"
        api = "openai-completions"
        apiKey = "local"
        compat = @{
          supportsDeveloperRole = $false
          supportsReasoningEffort = $false
          maxTokensField = "max_tokens"
        }
        models = @(
          @{
            id = $Model
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
  Write-Utf8Json -Path (Join-Path $AgentDir "settings.json") -Value @{
    defaultProvider = "llamacpp-test"
    defaultModel = $Model
    defaultThinkingLevel = "off"
    enabledModels = @("llamacpp-test/$Model")
    quietStartup = $true
  }
}

function Start-DebugProxy {
  param(
    [Parameter(Mandatory = $true)] [string] $LogDir,
    [Parameter(Mandatory = $true)] [int] $Port,
    [Parameter(Mandatory = $true)] [int] $Upstream
  )

  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $nodeCmd = Get-Command node -ErrorAction Stop
  $psi = [System.Diagnostics.ProcessStartInfo]::new()
  $psi.FileName = $nodeCmd.Source
  $psi.Arguments = '"' + (Join-Path $PSScriptRoot "proxy.mjs") + '"'
  $psi.WorkingDirectory = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
  $psi.UseShellExecute = $false
  $psi.RedirectStandardError = $true
  $psi.RedirectStandardOutput = $true
  $psi.Environment["LOG_DIR"] = $LogDir
  $psi.Environment["PROXY_PORT"] = [string]$Port
  $psi.Environment["UPSTREAM_PORT"] = [string]$Upstream

  $process = [System.Diagnostics.Process]::Start($psi)
  Start-Sleep -Milliseconds 700
  return $process
}

function Stop-DebugProxy {
  param(
    [Parameter(Mandatory = $true)] [System.Diagnostics.Process] $Process,
    [Parameter(Mandatory = $true)] [string] $OutDir
  )

  if (-not $Process.HasExited) {
    $Process.Kill()
    $Process.WaitForExit()
  }
  $Process.StandardError.ReadToEnd() | Set-Content -LiteralPath (Join-Path $OutDir "proxy-stderr.txt") -Encoding UTF8
  $Process.StandardOutput.ReadToEnd() | Set-Content -LiteralPath (Join-Path $OutDir "proxy-stdout.txt") -Encoding UTF8
}

function Read-JsonFileOrNull {
  param([Parameter(Mandatory = $true)] [string] $Path)
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-WireSummary {
  param([Parameter(Mandatory = $true)] [string] $ProxyLogDir)

  $requests = @(Get-ChildItem -LiteralPath $ProxyLogDir -Filter "pi-wire-request-*.json" -ErrorAction SilentlyContinue | Sort-Object Name)
  $responses = @(Get-ChildItem -LiteralPath $ProxyLogDir -Filter "pi-wire-response-*.sse" -ErrorAction SilentlyContinue | Sort-Object Name)
  $sawTools = $false
  $sawStream = $false
  $toolNames = [System.Collections.Generic.List[string]]::new()
  $firstSystemPrompt = $null

  foreach ($request in $requests) {
    $requestJson = Get-Content -LiteralPath $request.FullName -Raw | ConvertFrom-Json
    if ($requestJson.body.stream -eq $true) { $sawStream = $true }
    if ($null -ne $requestJson.body.tools) {
      $sawTools = $true
      foreach ($tool in $requestJson.body.tools) {
        if ($tool.function.name) { $toolNames.Add([string]$tool.function.name) }
      }
    }
    if ($null -eq $firstSystemPrompt -and $requestJson.body.messages.Count -gt 0) {
      $firstSystemPrompt = $requestJson.body.messages[0].content
    }
  }

  $sawNativeToolCalls = $false
  $sawFinishToolCalls = $false
  foreach ($response in $responses) {
    $raw = Get-Content -LiteralPath $response.FullName -Raw
    if ($raw -match '"tool_calls"') { $sawNativeToolCalls = $true }
    if ($raw -match '"finish_reason"\s*:\s*"tool_calls"') { $sawFinishToolCalls = $true }
  }

  return [ordered]@{
    requestCount = $requests.Count
    responseCount = $responses.Count
    sawTools = $sawTools
    sawStream = $sawStream
    toolNames = @($toolNames | Select-Object -Unique)
    sawNativeToolCalls = $sawNativeToolCalls
    sawFinishToolCalls = $sawFinishToolCalls
    firstSystemPromptHasNone = ($firstSystemPrompt -match 'Available tools:\s*\(none\)')
  }
}

function New-ProofDir {
  param(
    [Parameter(Mandatory = $true)] [string] $Parent,
    [Parameter(Mandatory = $true)] [string] $Name
  )

  $proofDir = Join-Path $Parent "$Name\proof"
  New-Item -ItemType Directory -Force -Path $proofDir | Out-Null
  $token = [guid]::NewGuid().ToString()
  Set-Content -LiteralPath (Join-Path $proofDir "proof.txt") -Value "SECRET_TOKEN=$token" -NoNewline -Encoding UTF8
  return [ordered]@{
    dir = (Resolve-Path -LiteralPath $proofDir).Path
    token = $token
  }
}

function Invoke-CliCase {
  param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [string[]] $Tools
  )

  $caseDir = Join-Path $OutDir $Name
  New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
  $proxyDir = Join-Path $caseDir "proxy"
  $toolDebugPath = Join-Path $caseDir "pi-tools-debug.json"
  $proof = New-ProofDir -Parent $caseDir -Name "case"
  $proxyProcess = Start-DebugProxy -LogDir $proxyDir -Port $ProxyPort -Upstream $UpstreamPort
  $piExitCode = $null

  try {
    Push-Location $proof.dir
    try {
      $env:PI_CODING_AGENT_DIR = $isolatedAgentDir
      $env:PI_TOOL_DEBUG_OUT = $toolDebugPath
      $prompt = "Open proof.txt using the read tool. Return exactly the SECRET_TOKEN value and nothing else. Do not guess."
      $args = @(
        "--offline", "--approve", "--mode", "json",
        "--provider", "llamacpp-test",
        "--model", $ModelAlias,
        "--thinking", "off",
        "--extension", (Join-Path $PSScriptRoot "tool-debug-extension.mjs"),
        "--no-skills", "--no-context-files", "--no-session"
      )
      if ($Tools -and $Tools.Count -gt 0) {
        $args += @("--tools", ($Tools -join ","))
      }
      $args += @("-p", $prompt)
      & pi @args *> (Join-Path $caseDir "pi-output.jsonl")
      $piExitCode = $LASTEXITCODE
    } finally {
      Remove-Item Env:\PI_CODING_AGENT_DIR -ErrorAction SilentlyContinue
      Remove-Item Env:\PI_TOOL_DEBUG_OUT -ErrorAction SilentlyContinue
      Pop-Location
    }
  } finally {
    Stop-DebugProxy -Process $proxyProcess -OutDir $caseDir
  }

  $outputRaw = Get-Content -LiteralPath (Join-Path $caseDir "pi-output.jsonl") -Raw
  return [ordered]@{
    name = $Name
    kind = "cli"
    requestedTools = $Tools
    expectedToken = $proof.token
    exitCode = $piExitCode
    outputContainsToken = ($outputRaw -match [regex]::Escape($proof.token))
    outputHasToolEvents = ($outputRaw -match '"toolcall_start"|\"toolcall_end\"|"tool_execution_start"|"tool_execution_end"')
    debug = Read-JsonFileOrNull -Path $toolDebugPath
    wire = Get-WireSummary -ProxyLogDir $proxyDir
  }
}

function Invoke-SdkCase {
  param(
    [Parameter(Mandatory = $true)] [string] $Name,
    [string[]] $Tools
  )

  $caseDir = Join-Path $OutDir $Name
  New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
  $proxyDir = Join-Path $caseDir "proxy"
  $sdkOutPath = Join-Path $caseDir "sdk-output.json"
  $proof = New-ProofDir -Parent $caseDir -Name "case"
  $proxyProcess = Start-DebugProxy -LogDir $proxyDir -Port $ProxyPort -Upstream $UpstreamPort
  $sdkExitCode = $null

  try {
    $env:PI_AGENT_DIR = $isolatedAgentDir
    $env:PI_PACKAGE_DIR = $piPackageDir
    $env:PI_PROVIDER = "llamacpp-test"
    $env:PI_MODEL = $ModelAlias
    $env:PI_CWD = $proof.dir
    $env:PI_SDK_OUT = $sdkOutPath
    $env:PI_PROMPT = "Open proof.txt using the read tool. Return exactly the SECRET_TOKEN value and nothing else. Do not guess."
    if ($Tools -and $Tools.Count -gt 0) {
      $env:PI_SDK_TOOLS = ($Tools -join ",")
    } else {
      Remove-Item Env:\PI_SDK_TOOLS -ErrorAction SilentlyContinue
    }
    node (Join-Path $PSScriptRoot "sdk-tool-check.mjs") *> (Join-Path $caseDir "node-output.txt")
    $sdkExitCode = $LASTEXITCODE
  } finally {
    Remove-Item Env:\PI_AGENT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PACKAGE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PROVIDER -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_MODEL -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_CWD -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_SDK_OUT -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PROMPT -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_SDK_TOOLS -ErrorAction SilentlyContinue
    Stop-DebugProxy -Process $proxyProcess -OutDir $caseDir
  }

  $sdkOutput = Read-JsonFileOrNull -Path $sdkOutPath
  return [ordered]@{
    name = $Name
    kind = "sdk"
    requestedTools = $Tools
    expectedToken = $proof.token
    exitCode = $sdkExitCode
    outputContainsToken = ($sdkOutput.assistantText -match [regex]::Escape($proof.token))
    sdk = $sdkOutput
    wire = Get-WireSummary -ProxyLogDir $proxyDir
  }
}

$piPackageDir = "C:\Users\ArnyPC\AppData\Roaming\npm\node_modules\@earendil-works\pi-coding-agent"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "matrix-out-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$isolatedAgentDir = Join-Path $OutDir "pi-agent-dir"
New-IsolatedPiConfig -AgentDir $isolatedAgentDir -Port $ProxyPort -Model $ModelAlias

$baseline = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  piVersion = (& pi --version 2>&1 | Out-String).Trim()
  piPackageDir = $piPackageDir
  serverModels = Invoke-RestMethod -Uri "http://127.0.0.1:$UpstreamPort/v1/models" -Method Get -TimeoutSec 10
}
Write-Utf8Json -Path (Join-Path $OutDir "baseline.json") -Value $baseline

$cases = @(
  (Invoke-CliCase -Name "cli-default"),
  (Invoke-CliCase -Name "cli-explicit-readonly" -Tools @("read", "ls", "find", "grep")),
  (Invoke-SdkCase -Name "sdk-default"),
  (Invoke-SdkCase -Name "sdk-explicit-readonly" -Tools @("read", "ls", "find", "grep"))
)

$summary = [ordered]@{
  outDir = $OutDir
  baseline = $baseline
  cases = $cases
}

Write-Utf8Json -Path (Join-Path $OutDir "summary.json") -Value $summary
$summary | ConvertTo-Json -Depth 80
