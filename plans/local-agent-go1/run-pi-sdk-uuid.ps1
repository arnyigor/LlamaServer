param(
  [int] $Runs = 10,
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

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "pi-sdk-uuid-out-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$piPackageDir = "C:\Users\ArnyPC\AppData\Roaming\npm\node_modules\@earendil-works\pi-coding-agent"
$isolatedAgentDir = Join-Path $OutDir "pi-agent-dir"
$proxyDir = Join-Path $OutDir "proxy"
New-IsolatedPiConfig -AgentDir $isolatedAgentDir -Port $ProxyPort -Model $ModelAlias
$proxyProcess = Start-DebugProxy -LogDir $proxyDir -Port $ProxyPort -Upstream $UpstreamPort

$results = @()
try {
  for ($i = 1; $i -le $Runs; $i++) {
    $caseDir = Join-Path $OutDir ("run-{0:D2}" -f $i)
    $proofDir = Join-Path $caseDir "proof"
    New-Item -ItemType Directory -Force -Path $proofDir | Out-Null
    $token = [guid]::NewGuid().ToString()
    Set-Content -LiteralPath (Join-Path $proofDir "proof.txt") -Value "SECRET_TOKEN=$token" -NoNewline -Encoding UTF8

    $sdkOutPath = Join-Path $caseDir "sdk-output.json"
    $nodeOutPath = Join-Path $caseDir "node-output.txt"
    $prompt = "Open proof.txt using the read tool. Return exactly the SECRET_TOKEN value and nothing else. Do not guess."

    $env:PI_AGENT_DIR = $isolatedAgentDir
    $env:PI_PACKAGE_DIR = $piPackageDir
    $env:PI_PROVIDER = "llamacpp-test"
    $env:PI_MODEL = $ModelAlias
    $env:PI_CWD = $proofDir
    $env:PI_SDK_OUT = $sdkOutPath
    $env:PI_PROMPT = $prompt
    $env:PI_SDK_TOOLS = "read,ls,find,grep"

    node (Join-Path $PSScriptRoot "sdk-tool-check.mjs") *> $nodeOutPath
    $exitCode = $LASTEXITCODE

    Remove-Item Env:\PI_AGENT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PACKAGE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PROVIDER -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_MODEL -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_CWD -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_SDK_OUT -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_PROMPT -ErrorAction SilentlyContinue
    Remove-Item Env:\PI_SDK_TOOLS -ErrorAction SilentlyContinue

    $sdkOutput = Get-Content -LiteralPath $sdkOutPath -Raw | ConvertFrom-Json
    $results += [ordered]@{
      run = $i
      expectedToken = $token
      exitCode = $exitCode
      answer = $sdkOutput.assistantText
      pass = ($exitCode -eq 0 -and $sdkOutput.assistantText -match [regex]::Escape($token))
      active = $sdkOutput.before.active
      toolCalls = $sdkOutput.toolCallCount
      toolResults = $sdkOutput.toolResultCount
    }
  }
} finally {
  Remove-Item Env:\PI_AGENT_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_PACKAGE_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_PROVIDER -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_MODEL -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_CWD -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_SDK_OUT -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_PROMPT -ErrorAction SilentlyContinue
  Remove-Item Env:\PI_SDK_TOOLS -ErrorAction SilentlyContinue
  Stop-DebugProxy -Process $proxyProcess -OutDir $OutDir
}

$wireRequests = @(Get-ChildItem -LiteralPath $proxyDir -Filter "pi-wire-request-*.json" | Sort-Object Name)
$requestsWithTools = 0
foreach ($request in $wireRequests) {
  $requestJson = Get-Content -LiteralPath $request.FullName -Raw | ConvertFrom-Json
  if ($null -ne $requestJson.body.tools) { $requestsWithTools++ }
}

$summary = [ordered]@{
  outDir = $OutDir
  runs = $Runs
  passed = @($results | Where-Object { $_.pass }).Count
  failed = @($results | Where-Object { -not $_.pass }).Count
  wireRequestCount = $wireRequests.Count
  wireRequestsWithTools = $requestsWithTools
  results = $results
}

Write-Utf8Json -Path (Join-Path $OutDir "summary.json") -Value $summary
$summary | ConvertTo-Json -Depth 80
