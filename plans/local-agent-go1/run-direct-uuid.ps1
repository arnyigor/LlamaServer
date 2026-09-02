param(
  [int] $Runs = 10,
  [string] $OutDir = "",
  [string] $ModelAlias = "qwen-27b",
  [string] $BaseUrl = "http://127.0.0.1:8080/v1"
)

$ErrorActionPreference = "Stop"

function Write-Utf8Json {
  param(
    [Parameter(Mandatory = $true)] [string] $Path,
    [Parameter(Mandatory = $true)] $Value
  )
  $Value | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "direct-uuid-out-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
  $caseDir = Join-Path $OutDir ("run-{0:D2}" -f $i)
  $proofDir = Join-Path $caseDir "proof"
  New-Item -ItemType Directory -Force -Path $proofDir | Out-Null
  $token = [guid]::NewGuid().ToString()
  Set-Content -LiteralPath (Join-Path $proofDir "proof.txt") -Value "SECRET_TOKEN=$token" -NoNewline -Encoding UTF8

  $jsonOut = Join-Path $caseDir "direct-output.json"
  $traceOut = Join-Path $caseDir "trace.json"
  $stderrOut = Join-Path $caseDir "stderr.txt"
  $task = "Open proof.txt using the read_file tool. Return exactly the SECRET_TOKEN value and nothing else. Do not guess."

  node (Join-Path $PSScriptRoot "direct-agent.mjs") `
    --cwd $proofDir `
    --task $task `
    --base-url $BaseUrl `
    --model $ModelAlias `
    --max-steps 6 `
    --max-tool-calls 4 `
    --trace-out $traceOut `
    1> $jsonOut 2> $stderrOut
  $exitCode = $LASTEXITCODE

  $parsed = $null
  if (Test-Path -LiteralPath $jsonOut) {
    $raw = Get-Content -LiteralPath $jsonOut -Raw
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
      $parsed = $raw | ConvertFrom-Json
    }
  }

  $trace = $null
  if (Test-Path -LiteralPath $traceOut) {
    $trace = Get-Content -LiteralPath $traceOut -Raw | ConvertFrom-Json
  }

  $results += [ordered]@{
    run = $i
    expectedToken = $token
    exitCode = $exitCode
    answer = $parsed.answer
    pass = ($exitCode -eq 0 -and $parsed.answer -match [regex]::Escape($token))
    toolCalls = $parsed.stats.tool_calls
    filesRead = $parsed.stats.files_read
    nativeToolCallSeen = [bool](@($trace.trace | Where-Object { $_.assistant.tool_calls.Count -gt 0 }).Count)
  }
}

$summary = [ordered]@{
  outDir = $OutDir
  runs = $Runs
  passed = @($results | Where-Object { $_.pass }).Count
  failed = @($results | Where-Object { -not $_.pass }).Count
  results = $results
}

Write-Utf8Json -Path (Join-Path $OutDir "summary.json") -Value $summary
$summary | ConvertTo-Json -Depth 80
