param(
  [int] $Runs = 3,
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

function Write-Text {
  param(
    [Parameter(Mandatory = $true)] [string] $Path,
    [Parameter(Mandatory = $true)] [string] $Value
  )
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  Set-Content -LiteralPath $Path -Value $Value -Encoding UTF8
}

function New-SyntheticRepo {
  param([Parameter(Mandatory = $true)] [string] $Path)

  New-Item -ItemType Directory -Force -Path $Path | Out-Null
  Write-Text -Path (Join-Path $Path "settings.gradle.kts") -Value @"
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
rootProject.name = "SyntheticFirstSteps"
include(":app")
"@
  Write-Text -Path (Join-Path $Path "app\src\main\java\demo\firststeps\FirstStepsWidget.kt") -Value @"
package demo.firststeps

class FirstStepsWidget(private val viewModel: MainViewModel) {
    fun render(): String {
        return viewModel.firstStepsState.joinToString()
    }
}
"@
  Write-Text -Path (Join-Path $Path "app\src\main\java\demo\firststeps\MainViewModel.kt") -Value @"
package demo.firststeps

class MainViewModel(private val repository: FirstStepsRepository) {
    val firstStepsState: List<String>
        get() = repository.cachedFirstSteps

    fun completeStep(id: String) {
        repository.markDone(id)
        // BUG_ID=FS-REFRESH-42
        // The widget observes cachedFirstSteps, but this path never refreshes it after mutation.
    }
}
"@
  Write-Text -Path (Join-Path $Path "app\src\main\java\demo\firststeps\FirstStepsRepository.kt") -Value @"
package demo.firststeps

class FirstStepsRepository {
    var cachedFirstSteps: List<String> = listOf("profile", "import")
        private set

    fun markDone(id: String) {
        Database.completed.add(id)
    }

    fun refreshFirstSteps() {
        cachedFirstSteps = listOf("profile", "import").filterNot { Database.completed.contains(it) }
    }
}
"@
  Write-Text -Path (Join-Path $Path "app\src\test\java\demo\firststeps\MainViewModelTest.kt") -Value @"
package demo.firststeps

class MainViewModelTest {
    fun completeStep_updatesWidgetState() {
        val repository = FirstStepsRepository()
        val viewModel = MainViewModel(repository)
        viewModel.completeStep("profile")
        assert(!viewModel.firstStepsState.contains("profile"))
    }
}
"@
  Write-Text -Path (Join-Path $Path "app\src\main\java\demo\firststeps\Database.kt") -Value @"
package demo.firststeps

object Database {
    val completed = mutableSetOf<String>()
}
"@
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $OutDir = Join-Path $PSScriptRoot "direct-synthetic-out-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
  $caseDir = Join-Path $OutDir ("run-{0:D2}" -f $i)
  $repoDir = Join-Path $caseDir "repo"
  New-SyntheticRepo -Path $repoDir

  $jsonOut = Join-Path $caseDir "direct-output.json"
  $traceOut = Join-Path $caseDir "trace.json"
  $stderrOut = Join-Path $caseDir "stderr.txt"
  $task = "In this Android/Kotlin repository, find why FirstStepsWidget does not show completed steps after MainViewModel.completeStep(). Use search tools, inspect relevant files, and include the BUG_ID in the final answer."

  node (Join-Path $PSScriptRoot "direct-agent.mjs") `
    --cwd $repoDir `
    --task $task `
    --base-url $BaseUrl `
    --model $ModelAlias `
    --max-steps 10 `
    --max-tool-calls 12 `
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

  $toolNames = @()
  if ($trace) {
    $toolNames = @($trace.trace | Where-Object { $_.toolCall } | ForEach-Object { $_.toolCall.name })
  }

  $results += [ordered]@{
    run = $i
    exitCode = $exitCode
    answer = $parsed.answer
    pass = ($exitCode -eq 0 -and $parsed.answer -match "FS-REFRESH-42")
    toolCalls = $parsed.stats.tool_calls
    filesRead = $parsed.stats.files_read
    toolNames = $toolNames
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
