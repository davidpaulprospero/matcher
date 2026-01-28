# Self-Healing Ralph Codebase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After every Ralph iteration, automatically detect codebase errors using a tiered diagnostic system (fast syntax checks, import validation, full pytest). If errors are found, pause the current sprint, create a healing session to fix them, log every fix with thought process, and only resume the sprint when the codebase is clean.

**Architecture:** A new `healing.ps1` module hooks into the post-iteration flow in `Resolve-ClaudeResult`. After each story completes (success OR failure), it runs a **3-tier health check**:
- **Tier 1 (Fast, <2s):** `py_compile` changed files, check for merge conflicts, verify critical files exist, validate config
- **Tier 2 (Medium, 5-15s):** `pytest --collect-only` to catch import/syntax errors without running tests
- **Tier 3 (Full, 30-120s):** Full `pytest` run -- only on cadence (every Nth iteration) or when Tier 1/2 detect issues

If any tier detects errors, it saves the current sprint state to `healing_state.json`, spawns a Claude healing session with error context, validates the fix, logs everything to `healing_log.jsonl`, then resumes the sprint.

**Tech Stack:** PowerShell 5.1+ (Pester tests), Python pytest/py_compile (health checks), git (state detection), JSONL logging, existing Ralph module architecture

---

## Architecture Overview

```
Post-Iteration Hook (claude.ps1 Resolve-ClaudeResult)
    |
    v
Invoke-PostIterationHealing (healing.ps1)
    |
    v
TIER 1: Invoke-FastHealthCheck (<2s, EVERY iteration)
    |-- py_compile changed files (git diff --name-only)
    |-- Check for merge conflicts (git ls-files -u)
    |-- Verify critical files exist (main.py, src/__init__.py, config.yaml)
    |-- Validate config (python main.py --validate-config)
    |-- If errors -> jump to HEALING
    |-- If clean -> continue to Tier 2
    |
    v
TIER 2: Invoke-CollectionHealthCheck (5-15s, EVERY iteration)
    |-- pytest --collect-only -q (catches import errors, no test execution)
    |-- If errors -> jump to HEALING
    |-- If clean -> check if Tier 3 due
    |
    v
TIER 3: Invoke-FullHealthCheck (30-120s, CADENCE-BASED)
    |-- Full pytest tests/ --tb=short -q
    |-- Runs when: every Nth iteration (default: 3), first iteration of session,
    |              or forced by config
    |-- If errors -> jump to HEALING
    |-- If clean -> return, continue sprint
    |
    v (any tier found errors)
Suspend-SprintForHealing
    |-- Save healing_state.json (tier that failed, errors, sprint context)
    |-- Log "healing_started" event
    |
    v
Invoke-HealingSession
    |-- Build healing prompt (tier diagnostics + diff + context)
    |-- Invoke-ClaudeSubprocess (healing mode)
    |-- Validate fix (re-run the SAME tier that failed + all lower tiers)
    |-- Up to maxHealingAttempts (default: 3)
    |
    v
Log-HealingEvent
    |-- Write to healing_log.jsonl
    |-- Record: tier, what broke, why, fix applied, thought process
    |
    v
Resume-SprintFromHealing
    |-- Clear healing_state.json
    |-- Return to story loop
```

**Key Design Decisions:**

1. **Hook location**: Inside `Resolve-ClaudeResult` in `claude.ps1`, after the SUCCESS and FAILURE blocks but before returning. This catches errors introduced by both successful and failed iterations.

2. **Tiered checking**: Fast checks every iteration (< 2s overhead). Full pytest only every Nth iteration or when something smells off. This keeps Ralph fast while still catching everything.

3. **Healing is NOT a story**: Healing iterations don't count against sprint progress, iteration limits, or story metrics. They use a separate "healing" mode in metrics.

4. **Healing log**: `healing_log.jsonl` is a permanent, append-only record. Each entry captures the full diagnosis and fix narrative, including which tier detected the problem.

5. **Sprint suspension**: Uses a `healing_state.json` file (not the queue). The sprint isn't "paused" in the queue sense -- it's frozen in place while healing runs, then resumes exactly where it left off.

6. **Failure escalation**: If healing fails after maxHealingAttempts, the sprint aborts with a clear error. No silent failures.

7. **Validation re-runs the failing tier**: When healing validates a fix, it re-runs the same tier that detected the problem (plus all lower tiers). This avoids false positives from only checking a subset.

---

## Task 1: Create `healing.ps1` Module with Tiered Health Checks

**Files:**
- Create: `scripts/ralph/lib/healing.ps1`
- Create: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing tests

```powershell
# scripts/ralph/tests/Healing.Tests.ps1
BeforeAll {
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions
}

Describe 'Invoke-FastHealthCheck (Tier 1)' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
    }

    It 'returns clean when no changed files' {
        Mock git { return "" } -ParameterFilter { $args -contains 'diff' }

        $result = Invoke-FastHealthCheck -ChangedFiles @()
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 1
    }

    It 'detects syntax errors via py_compile' {
        Mock Start-Process {
            return @{ ExitCode = 1 }
        }
        Mock Invoke-Expression {
            return "SyntaxError: invalid syntax (src/config.py, line 42)"
        } -ParameterFilter { $Command -like '*py_compile*' }

        $result = Invoke-FastHealthCheck -ChangedFiles @("src/config.py")
        $result.HasErrors | Should -BeTrue
        $result.SyntaxErrors.Count | Should -BeGreaterThan 0
    }

    It 'detects merge conflicts via git' {
        Mock git {
            return @("src/pipeline.py")
        } -ParameterFilter { $args -contains 'ls-files' -and $args -contains '-u' }

        $result = Invoke-FastHealthCheck -ChangedFiles @()
        $result.HasErrors | Should -BeTrue
        $result.MergeConflicts.Count | Should -Be 1
    }

    It 'detects missing critical files' {
        # Don't create any files in TestDrive
        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @("main.py")
        $result.HasErrors | Should -BeTrue
        $result.MissingFiles.Count | Should -Be 1
    }

    It 'passes when critical files exist' {
        New-Item -Path (Join-Path $TestDrive "main.py") -ItemType File -Force | Out-Null

        Mock git { return "" } -ParameterFilter { $args -contains 'ls-files' }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @("main.py")
        $result.MissingFiles.Count | Should -Be 0
    }
}

Describe 'Invoke-CollectionHealthCheck (Tier 2)' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
    }

    It 'returns clean when collect-only succeeds' {
        Mock Invoke-Expression {
            return "50 tests collected in 2.1s"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 2
    }

    It 'detects import errors from collect-only' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/test_budget.py - ImportError: cannot import name 'BudgetManager' from 'src.downloader'",
                "ERROR tests/test_cache.py - SyntaxError: invalid syntax",
                "2 errors in 1.3s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeTrue
        $result.ErrorCount | Should -Be 2
        $result.CollectionErrors.Count | Should -Be 2
    }

    It 'handles pytest unavailable gracefully' {
        Mock Invoke-Expression {
            throw "pytest: command not found"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.Skipped | Should -BeTrue
    }
}

Describe 'Invoke-FullHealthCheck (Tier 3)' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
    }

    It 'returns clean when all tests pass' {
        Mock Invoke-Expression {
            return "50 passed in 12.3s"
        } -ParameterFilter { $Command -like '*pytest*' -and $Command -notlike '*--collect-only*' }

        $result = Invoke-FullHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 3
    }

    It 'detects test failures' {
        Mock Invoke-Expression {
            return @(
                "FAILED tests/test_cache.py::test_hit_rate - AssertionError",
                "FAILED tests/test_config.py::test_load - ImportError",
                "2 failed, 10 passed in 3.45s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck
        $result.HasErrors | Should -BeTrue
        $result.FailureCount | Should -Be 2
        $result.Failures.Count | Should -Be 2
    }

    It 'detects collection errors from full run' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/test_budget.py - SyntaxError: invalid syntax",
                "1 error in 0.54s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck
        $result.HasErrors | Should -BeTrue
        $result.ErrorCount | Should -Be 1
    }

    It 'returns clean when no tests found' {
        Mock Invoke-Expression {
            return "no tests ran in 0.01s"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck
        $result.HasErrors | Should -BeFalse
    }
}

Describe 'Invoke-TieredHealthCheck (orchestrator)' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
        $script:State = @{ IterationCount = 5; SessionId = "test" }
    }

    It 'stops at Tier 1 when syntax error found' {
        Mock Invoke-FastHealthCheck {
            return @{ HasErrors = $true; Tier = 1; SyntaxErrors = @(@{File="x.py";Error="bad"}) }
        }

        $result = Invoke-TieredHealthCheck -ChangedFiles @("x.py")
        $result.HasErrors | Should -BeTrue
        $result.FailedTier | Should -Be 1
        Should -Not -Invoke Invoke-CollectionHealthCheck
    }

    It 'escalates to Tier 2 when Tier 1 clean' {
        Mock Invoke-FastHealthCheck {
            return @{ HasErrors = $false; Tier = 1 }
        }
        Mock Invoke-CollectionHealthCheck {
            return @{ HasErrors = $false; Tier = 2 }
        }

        $result = Invoke-TieredHealthCheck -ChangedFiles @()
        $result.HasErrors | Should -BeFalse
        Should -Invoke Invoke-CollectionHealthCheck -Times 1
    }

    It 'runs Tier 3 on cadence (every Nth iteration)' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2 } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3 } }

        # Iteration 3 = cadence hit (every 3)
        $script:State.IterationCount = 3
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }

    It 'skips Tier 3 off cadence' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2 } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3 } }

        # Iteration 4 = not on cadence
        $script:State.IterationCount = 4
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        Should -Invoke Invoke-FullHealthCheck -Times 0
    }

    It 'always runs Tier 3 on first iteration' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2 } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3 } }

        $script:State.IterationCount = 1
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 5
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }
}
```

### Step 2: Run test to verify it fails

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: FAIL with "CommandNotFoundException" because functions don't exist

### Step 3: Write minimal implementation

```powershell
# scripts/ralph/lib/healing.ps1
# Self-healing: tiered health checks, healing sessions, and logging

function Invoke-FastHealthCheck {
    <#
    .SYNOPSIS
        Tier 1: Fast checks (<2s). py_compile changed files, merge conflicts, critical files.
    .PARAMETER ChangedFiles
        List of files changed since last git state (relative paths)
    .PARAMETER CriticalFiles
        Files that must exist (relative to ProjectRoot). Default: main.py, src/__init__.py, config.yaml
    .RETURNS
        Hashtable: HasErrors, Tier=1, SyntaxErrors, MergeConflicts, MissingFiles, ConfigErrors
    #>
    param(
        [string[]]$ChangedFiles = @(),
        [string[]]$CriticalFiles = @("main.py", "src/__init__.py", "config.yaml")
    )

    $result = @{
        HasErrors      = $false
        Tier           = 1
        SyntaxErrors   = @()
        MergeConflicts = @()
        MissingFiles   = @()
        ConfigErrors   = @()
        ChecksRun      = @()
    }

    $projectRoot = if ($script:ProjectRoot) { $script:ProjectRoot } else { Get-Location }

    # --- Check 1: py_compile changed .py files ---
    $pyFiles = @($ChangedFiles | Where-Object { $_ -match '\.py$' })
    if ($pyFiles.Count -gt 0) {
        $result.ChecksRun += "py_compile"
        foreach ($f in $pyFiles) {
            $fullPath = Join-Path $projectRoot $f
            if (Test-Path $fullPath) {
                try {
                    $compileOutput = Invoke-Expression "python -m py_compile `"$fullPath`" 2>&1"
                    if ($LASTEXITCODE -ne 0) {
                        $result.SyntaxErrors += @{ File = $f; Error = ($compileOutput | Out-String).Trim() }
                    }
                } catch {
                    $result.SyntaxErrors += @{ File = $f; Error = $_.Exception.Message }
                }
            }
        }
    }

    # --- Check 2: Merge conflicts ---
    $result.ChecksRun += "merge_conflicts"
    try {
        $conflictFiles = @(git ls-files -u 2>$null | ForEach-Object {
            if ($_ -match '\t(.+)$') { $Matches[1] }
        } | Sort-Object -Unique)
        if ($conflictFiles.Count -gt 0) {
            $result.MergeConflicts = $conflictFiles
        }
    } catch {}

    # --- Check 3: Critical file existence ---
    $result.ChecksRun += "critical_files"
    foreach ($f in $CriticalFiles) {
        $fullPath = Join-Path $projectRoot $f
        if (-not (Test-Path $fullPath)) {
            $result.MissingFiles += $f
        }
    }

    # --- Check 4: Config validation ---
    $result.ChecksRun += "config_validation"
    try {
        $configOutput = Invoke-Expression "python main.py --validate-config 2>&1"
        if ($LASTEXITCODE -ne 0) {
            $result.ConfigErrors += ($configOutput | Out-String).Trim()
        }
    } catch {}

    $result.HasErrors = (
        $result.SyntaxErrors.Count -gt 0 -or
        $result.MergeConflicts.Count -gt 0 -or
        $result.MissingFiles.Count -gt 0 -or
        $result.ConfigErrors.Count -gt 0
    )

    return $result
}

function Invoke-CollectionHealthCheck {
    <#
    .SYNOPSIS
        Tier 2: Medium check (5-15s). pytest --collect-only catches import/syntax errors without running tests.
    .RETURNS
        Hashtable: HasErrors, Tier=2, ErrorCount, CollectionErrors, Summary, Skipped, RawOutput
    #>

    $result = @{
        HasErrors        = $false
        Tier             = 2
        ErrorCount       = 0
        CollectionErrors = @()
        Summary          = ""
        Skipped          = $false
        RawOutput        = ""
    }

    try {
        $output = Invoke-Expression "pytest tests/ --collect-only -q --no-header 2>&1"
        if ($output -is [array]) { $output = $output -join "`n" }
        $result.RawOutput = $output

        # Parse ERROR lines (import/syntax errors during collection)
        $errorLines = @($output -split "`n" | Where-Object { $_ -match '^ERROR ' })
        $result.CollectionErrors = @($errorLines | ForEach-Object {
            if ($_ -match '^ERROR\s+(.+?)\s*-\s*(.+)$') {
                @{ File = $Matches[1]; Error = $Matches[2] }
            } elseif ($_ -match '^ERROR\s+(.+)$') {
                @{ File = $Matches[1]; Error = "Unknown" }
            }
        })
        $result.ErrorCount = $result.CollectionErrors.Count

        if ($output -match '(\d+\s+error.+)$') {
            $result.Summary = $Matches[1]
        }

        $result.HasErrors = ($result.ErrorCount -gt 0)
    } catch {
        $result.Skipped = $true
    }

    return $result
}

function Invoke-FullHealthCheck {
    <#
    .SYNOPSIS
        Tier 3: Full test run (30-120s). Runs pytest with full test execution.
    .PARAMETER PytestArgs
        Override pytest arguments (default: from config or "tests/ --tb=short -q --no-header")
    .RETURNS
        Hashtable: HasErrors, Tier=3, FailureCount, ErrorCount, Failures, CollectionErrors, Summary, Skipped, RawOutput
    #>
    param(
        [string]$PytestArgs = ""
    )

    $result = @{
        HasErrors        = $false
        Tier             = 3
        FailureCount     = 0
        ErrorCount       = 0
        Failures         = @()
        CollectionErrors = @()
        Summary          = ""
        Skipped          = $false
        RawOutput        = ""
    }

    # Resolve pytest args from config if not provided
    if (-not $PytestArgs) {
        try {
            $config = Get-RalphConfig
            if ($config.selfHealing -and $config.selfHealing.pytestArgs) {
                $PytestArgs = $config.selfHealing.pytestArgs
            }
        } catch {}
        if (-not $PytestArgs) {
            $PytestArgs = "tests/ --tb=short -q --no-header"
        }
    }

    try {
        $pytestCmd = "pytest $PytestArgs --non-interactive 2>&1"
        $output = Invoke-Expression $pytestCmd
        if ($output -is [array]) { $output = $output -join "`n" }
        $result.RawOutput = $output

        # Parse FAILED lines
        $failedLines = @($output -split "`n" | Where-Object { $_ -match '^FAILED ' })
        $result.Failures = @($failedLines | ForEach-Object {
            if ($_ -match '^FAILED\s+(.+?)\s*-\s*(.+)$') {
                @{ Test = $Matches[1]; Error = $Matches[2] }
            } elseif ($_ -match '^FAILED\s+(.+)$') {
                @{ Test = $Matches[1]; Error = "Unknown" }
            }
        })
        $result.FailureCount = $result.Failures.Count

        # Parse ERROR lines (collection errors)
        $errorLines = @($output -split "`n" | Where-Object { $_ -match '^ERROR ' })
        $result.CollectionErrors = @($errorLines | ForEach-Object {
            if ($_ -match '^ERROR\s+(.+?)\s*-\s*(.+)$') {
                @{ File = $Matches[1]; Error = $Matches[2] }
            } elseif ($_ -match '^ERROR\s+(.+)$') {
                @{ File = $Matches[1]; Error = "Unknown" }
            }
        })
        $result.ErrorCount = $result.CollectionErrors.Count

        # Parse summary line
        if ($output -match '(\d+\s+(?:failed|passed|error|warning).+)$') {
            $result.Summary = $Matches[1]
        }

        $result.HasErrors = ($result.FailureCount -gt 0) -or ($result.ErrorCount -gt 0)
    } catch {
        $result.Skipped = $true
    }

    return $result
}

function Invoke-TieredHealthCheck {
    <#
    .SYNOPSIS
        Orchestrate tiered health checks. Fast checks every time, full run on cadence.
    .PARAMETER ChangedFiles
        Files changed in this iteration (from git diff)
    .PARAMETER FullRunCadence
        Run Tier 3 every N iterations (default: 3)
    .PARAMETER ForceFullRun
        Force Tier 3 regardless of cadence
    .RETURNS
        Hashtable: HasErrors, FailedTier (0=clean), TierResults (array of tier results), RawDiagnostics (string)
    #>
    param(
        [string[]]$ChangedFiles = @(),
        [int]$FullRunCadence = 3,
        [switch]$ForceFullRun
    )

    $result = @{
        HasErrors      = $false
        FailedTier     = 0
        TierResults    = @()
        RawDiagnostics = ""
    }

    $iteration = if ($script:State) { $script:State.IterationCount } else { 1 }

    # === TIER 1: Fast checks ===
    Write-Host "  Health [T1]: syntax, conflicts, files..." -ForegroundColor DarkGray -NoNewline
    $t1 = Invoke-FastHealthCheck -ChangedFiles $ChangedFiles
    $result.TierResults += $t1

    if ($t1.HasErrors) {
        Write-Host " ERRORS" -ForegroundColor Red
        $result.HasErrors = $true
        $result.FailedTier = 1
        $result.RawDiagnostics = Build-TierDiagnostics -TierResult $t1
        return $result
    }
    Write-Host " OK" -ForegroundColor DarkGray

    # === TIER 2: Collection check ===
    Write-Host "  Health [T2]: import validation..." -ForegroundColor DarkGray -NoNewline
    $t2 = Invoke-CollectionHealthCheck
    $result.TierResults += $t2

    if ($t2.Skipped) {
        Write-Host " SKIPPED" -ForegroundColor DarkGray
    } elseif ($t2.HasErrors) {
        Write-Host " ERRORS" -ForegroundColor Red
        $result.HasErrors = $true
        $result.FailedTier = 2
        $result.RawDiagnostics = Build-TierDiagnostics -TierResult $t2
        return $result
    } else {
        Write-Host " OK" -ForegroundColor DarkGray
    }

    # === TIER 3: Full test run (cadence-based) ===
    $runTier3 = $ForceFullRun -or ($iteration -eq 1) -or ($iteration % $FullRunCadence -eq 0)

    if ($runTier3) {
        Write-Host "  Health [T3]: full test suite..." -ForegroundColor DarkGray -NoNewline
        $t3 = Invoke-FullHealthCheck
        $result.TierResults += $t3

        if ($t3.Skipped) {
            Write-Host " SKIPPED" -ForegroundColor DarkGray
        } elseif ($t3.HasErrors) {
            Write-Host " ERRORS ($($t3.FailureCount)F/$($t3.ErrorCount)E)" -ForegroundColor Red
            $result.HasErrors = $true
            $result.FailedTier = 3
            $result.RawDiagnostics = Build-TierDiagnostics -TierResult $t3
            return $result
        } else {
            Write-Host " OK" -ForegroundColor DarkGray
        }
    }

    return $result
}

function Build-TierDiagnostics {
    <#
    .SYNOPSIS
        Build a human-readable diagnostics string from a tier result.
    #>
    param([Parameter(Mandatory)]$TierResult)

    $lines = @("Tier $($TierResult.Tier) diagnostics:")

    if ($TierResult.SyntaxErrors) {
        foreach ($e in $TierResult.SyntaxErrors) { $lines += "  SYNTAX: $($e.File) -> $($e.Error)" }
    }
    if ($TierResult.MergeConflicts) {
        foreach ($f in $TierResult.MergeConflicts) { $lines += "  CONFLICT: $f" }
    }
    if ($TierResult.MissingFiles) {
        foreach ($f in $TierResult.MissingFiles) { $lines += "  MISSING: $f" }
    }
    if ($TierResult.ConfigErrors) {
        foreach ($e in $TierResult.ConfigErrors) { $lines += "  CONFIG: $e" }
    }
    if ($TierResult.CollectionErrors) {
        foreach ($e in $TierResult.CollectionErrors) { $lines += "  IMPORT: $($e.File) -> $($e.Error)" }
    }
    if ($TierResult.Failures) {
        foreach ($f in $TierResult.Failures) { $lines += "  FAIL: $($f.Test) -> $($f.Error)" }
    }
    if ($TierResult.RawOutput) {
        $lines += ""
        $lines += "Raw output:"
        $lines += $TierResult.RawOutput
    }

    return $lines -join "`n"
}
```

### Step 4: Run test to verify it passes

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add tiered health check system (T1: syntax, T2: imports, T3: full pytest)"
```

---

## Task 2: Add `Log-HealingEvent` and `healing_log.jsonl` Writer

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Log-HealingEvent' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
    }

    It 'creates healing log file on first write' {
        Log-HealingEvent -Event "healing_started" -Data @{
            trigger = "test_failure"
            failedTier = 2
            failureCount = 2
            errors = @("test_cache.py::test_hit_rate", "test_config.py::test_load")
        }

        $script:HealingLogFile | Should -Exist
        $lines = Get-Content $script:HealingLogFile
        $lines.Count | Should -Be 1
        $entry = $lines[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_started"
        $entry.data.failedTier | Should -Be 2
        $entry.timestamp | Should -Not -BeNullOrEmpty
    }

    It 'appends multiple events to log' {
        Log-HealingEvent -Event "healing_started" -Data @{ trigger = "test_failure" }
        Log-HealingEvent -Event "healing_attempt" -Data @{ attempt = 1; prompt = "Fix import error" }
        Log-HealingEvent -Event "healing_resolved" -Data @{ attempt = 1; fix = "Added missing import" }

        $lines = Get-Content $script:HealingLogFile
        $lines.Count | Should -Be 3
    }

    It 'includes thought process in healing_resolved events' {
        Log-HealingEvent -Event "healing_resolved" -Data @{
            attempt = 1
            errorsFixed = @("ImportError in test_cache.py")
            fix = "Added 'from src.cache import BaseCache' to test file"
            thoughtProcess = "The test file was importing BaseCache but the module was refactored. Updated import path."
            filesChanged = @("tests/test_cache.py")
            failedTier = 2
        }

        $entry = (Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.data.thoughtProcess | Should -Not -BeNullOrEmpty
        $entry.data.filesChanged.Count | Should -Be 1
        $entry.data.failedTier | Should -Be 2
    }

    It 'records session context in every event' {
        $script:State = @{ SessionId = "test-session-123"; IterationCount = 5 }
        Log-HealingEvent -Event "healing_started" -Data @{ trigger = "test" }

        $entry = (Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.sessionId | Should -Be "test-session-123"
        $entry.iteration | Should -Be 5
    }
}
```

### Step 2: Run test to verify it fails

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: FAIL - function not defined

### Step 3: Write minimal implementation

Append to `scripts/ralph/lib/healing.ps1`:

```powershell
function Log-HealingEvent {
    <#
    .SYNOPSIS
        Append a structured event to healing_log.jsonl.
        Every fix, diagnosis, and thought process is permanently recorded.
    .PARAMETER Event
        Event type: healing_started, healing_attempt, healing_resolved, healing_failed, healing_skipped
    .PARAMETER Data
        Hashtable of event-specific data (tier, errors, fix description, thought process, etc.)
    #>
    param(
        [Parameter(Mandatory)]
        [ValidateSet("healing_started", "healing_attempt", "healing_resolved", "healing_failed", "healing_skipped")]
        [string]$Event,

        [Parameter(Mandatory)]
        [hashtable]$Data
    )

    $logPath = if ($script:HealingLogFile) {
        $script:HealingLogFile
    } else {
        Join-Path $script:RalphDir "healing_log.jsonl"
    }

    $entry = @{
        timestamp  = (Get-Date -Format "o")
        event      = $Event
        data       = $Data
    }

    # Add session context if available
    if ($script:State) {
        if ($script:State.SessionId) { $entry.sessionId = $script:State.SessionId }
        if ($script:State.IterationCount) { $entry.iteration = $script:State.IterationCount }
    }

    $json = $entry | ConvertTo-Json -Depth 10 -Compress
    try {
        Add-Content -Path $logPath -Value $json -Encoding UTF8
    } catch {
        Start-Sleep -Milliseconds 200
        try { Add-Content -Path $logPath -Value $json -Encoding UTF8 } catch {}
    }
}
```

### Step 4: Run test to verify it passes

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add Log-HealingEvent for healing audit trail"
```

---

## Task 3: Add `Suspend-SprintForHealing` and `Resume-SprintFromHealing`

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Suspend-SprintForHealing' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        $script:State = @{
            SessionId = "test-session"; IterationCount = 7;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }
    }

    It 'creates healing_state.json with tier and sprint context' {
        $healthResult = @{
            HasErrors = $true; FailedTier = 2
            RawDiagnostics = "IMPORT: test_cache.py -> ImportError"
            TierResults = @(
                @{ Tier = 1; HasErrors = $false },
                @{ Tier = 2; HasErrors = $true; ErrorCount = 1;
                   CollectionErrors = @(@{ File = "test_cache.py"; Error = "ImportError" }) }
            )
        }

        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "US-005" -FocusArea "testing"

        $script:HealingStateFile | Should -Exist
        $state = Get-Content $script:HealingStateFile -Raw | ConvertFrom-Json
        $state.paused | Should -BeTrue
        $state.storyId | Should -Be "US-005"
        $state.focusArea | Should -Be "testing"
        $state.failedTier | Should -Be 2
    }

    It 'logs healing_started event with tier info' {
        $healthResult = @{
            HasErrors = $true; FailedTier = 1
            RawDiagnostics = "SYNTAX: config.py -> SyntaxError"
            TierResults = @(@{ Tier = 1; HasErrors = $true; SyntaxErrors = @(@{File="config.py";Error="bad"}) })
        }

        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "US-003" -FocusArea "pipeline"

        $script:HealingLogFile | Should -Exist
        $entry = (Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_started"
        $entry.data.failedTier | Should -Be 1
    }
}

Describe 'Resume-SprintFromHealing' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        $script:State = @{ SessionId = "s1"; IterationCount = 8; ConsecutiveFailures = 0 }
    }

    It 'clears healing_state.json on resume' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Resume-SprintFromHealing -Success $true -AttemptCount 1
        $script:HealingStateFile | Should -Not -Exist
    }

    It 'logs healing_resolved on successful fix' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Resume-SprintFromHealing -Success $true -AttemptCount 2 -FixSummary "Fixed import path"

        $entry = (Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_resolved"
        $entry.data.attempts | Should -Be 2
    }

    It 'logs healing_failed when fix unsuccessful' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Resume-SprintFromHealing -Success $false -AttemptCount 3

        $entry = (Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_failed"
    }
}

Describe 'Test-HealingInProgress' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
    }

    It 'returns false when no healing state file' {
        Test-HealingInProgress | Should -BeFalse
    }

    It 'returns true when healing is paused' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Test-HealingInProgress | Should -BeTrue
    }
}
```

### Step 2: Run test to verify it fails

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: FAIL - functions not defined

### Step 3: Write minimal implementation

Append to `scripts/ralph/lib/healing.ps1`:

```powershell
function Test-HealingInProgress {
    <#
    .SYNOPSIS
        Check if a healing session is currently in progress.
    #>
    $statePath = if ($script:HealingStateFile) { $script:HealingStateFile }
                 else { Join-Path $script:RalphDir "healing_state.json" }

    if (-not (Test-Path $statePath)) { return $false }

    try {
        $state = Get-Content $statePath -Raw | ConvertFrom-Json
        return ($state.paused -eq $true)
    } catch {
        return $false
    }
}

function Suspend-SprintForHealing {
    <#
    .SYNOPSIS
        Pause the current sprint to fix codebase errors.
        Saves tier diagnostics and sprint context to healing_state.json.
    #>
    param(
        [Parameter(Mandatory)][hashtable]$HealthResult,
        [string]$StoryId = "",
        [string]$FocusArea = ""
    )

    $statePath = if ($script:HealingStateFile) { $script:HealingStateFile }
                 else { Join-Path $script:RalphDir "healing_state.json" }

    $healingState = @{
        paused         = $true
        pausedAt       = (Get-Date -Format "o")
        storyId        = $StoryId
        focusArea      = $FocusArea
        failedTier     = $HealthResult.FailedTier
        rawDiagnostics = $HealthResult.RawDiagnostics
    }

    $healingState | ConvertTo-Json -Depth 10 | Set-Content $statePath -Encoding UTF8

    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor Red
    Write-Host "  HEALING MODE: Tier $($HealthResult.FailedTier) errors detected!" -ForegroundColor Red
    Write-Host "  ========================================" -ForegroundColor Red
    Write-Host "  Sprint paused at story: $StoryId" -ForegroundColor Yellow
    Write-Host ""

    Log-HealingEvent -Event "healing_started" -Data @{
        trigger        = "tier_$($HealthResult.FailedTier)_failure"
        storyId        = $StoryId
        focusArea      = $FocusArea
        failedTier     = $HealthResult.FailedTier
        rawDiagnostics = $HealthResult.RawDiagnostics
    }
}

function Resume-SprintFromHealing {
    <#
    .SYNOPSIS
        Resume the sprint after healing completes (success or failure).
    #>
    param(
        [Parameter(Mandatory)][bool]$Success,
        [int]$AttemptCount = 0,
        [string]$FixSummary = ""
    )

    $statePath = if ($script:HealingStateFile) { $script:HealingStateFile }
                 else { Join-Path $script:RalphDir "healing_state.json" }

    if ($Success) {
        Log-HealingEvent -Event "healing_resolved" -Data @{
            attempts   = $AttemptCount
            fixSummary = $FixSummary
        }
        Write-Host "  Healing complete! Resuming sprint..." -ForegroundColor Green
    } else {
        Log-HealingEvent -Event "healing_failed" -Data @{
            attempts = $AttemptCount
            reason   = "Max healing attempts exhausted"
        }
        Write-Host "  Healing FAILED after $AttemptCount attempts. Sprint aborting." -ForegroundColor Red
    }

    if (Test-Path $statePath) {
        Remove-Item $statePath -Force
    }
}
```

### Step 4: Run test to verify it passes

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add sprint suspend/resume for healing mode with tier context"
```

---

## Task 4: Add `Invoke-HealingSession` -- The Core Fix Loop

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Invoke-HealingSession' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:State = @{
            SessionId = "heal-test"; IterationCount = 10;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }

        @{
            paused = $true; storyId = "US-005"; focusArea = "testing"
            failedTier = 2; rawDiagnostics = "IMPORT: test_cache.py -> ImportError"
        } | ConvertTo-Json -Depth 10 | Set-Content $script:HealingStateFile
    }

    It 'attempts healing and succeeds on first try' {
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed the import in test_cache.py by updating path"
                ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }

        # Validation re-check passes
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0; TierResults = @(); RawDiagnostics = "" }
        }

        $result = Invoke-HealingSession
        $result.Success | Should -BeTrue
        $result.AttemptsUsed | Should -Be 1
    }

    It 'fails after max attempts exceeded' {
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Attempted fix"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }

        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $true; FailedTier = 2; RawDiagnostics = "still broken" }
        }

        $result = Invoke-HealingSession -MaxAttempts 2
        $result.Success | Should -BeFalse
        $result.AttemptsUsed | Should -Be 2
    }

    It 'logs each attempt to healing_log.jsonl' {
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed it"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }

        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0 }
        }

        Invoke-HealingSession -MaxAttempts 3

        $lines = Get-Content $script:HealingLogFile
        $lines.Count | Should -BeGreaterOrEqual 1
        $hasAttempt = $lines | Where-Object { ($_ | ConvertFrom-Json).event -eq "healing_attempt" }
        $hasAttempt | Should -Not -BeNullOrEmpty
    }
}
```

### Step 2: Run test to verify it fails

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: FAIL - Invoke-HealingSession not defined

### Step 3: Write minimal implementation

Append to `scripts/ralph/lib/healing.ps1`:

```powershell
function Invoke-HealingSession {
    <#
    .SYNOPSIS
        Run a Claude healing session to fix detected codebase errors.
        Reads healing_state.json for context, invokes Claude with tier-specific
        diagnostics, validates via re-running tiered health check, retries up to MaxAttempts.
    .PARAMETER MaxAttempts
        Maximum number of healing attempts (default: 3)
    .RETURNS
        Hashtable: Success (bool), AttemptsUsed (int), FixSummary (string)
    #>
    param(
        [int]$MaxAttempts = 3
    )

    $statePath = if ($script:HealingStateFile) { $script:HealingStateFile }
                 else { Join-Path $script:RalphDir "healing_state.json" }

    $healingState = $null
    if (Test-Path $statePath) {
        $healingState = Get-Content $statePath -Raw | ConvertFrom-Json
    }

    if (-not $healingState) {
        return @{ Success = $false; AttemptsUsed = 0; FixSummary = "No healing state found" }
    }

    $claudePath = Get-ClaudePath
    $attempt = 0
    $lastOutput = ""

    while ($attempt -lt $MaxAttempts) {
        $attempt++
        $attemptStart = Get-Date
        Write-Host ""
        Write-Host "  Healing attempt $attempt/$MaxAttempts (Tier $($healingState.failedTier) failure)" -ForegroundColor Cyan

        $prompt = Build-HealingPrompt -HealingState $healingState -Attempt $attempt -PreviousOutput $lastOutput

        Log-HealingEvent -Event "healing_attempt" -Data @{
            attempt     = $attempt
            maxAttempts = $MaxAttempts
            failedTier  = [int]$healingState.failedTier
            prompt      = ($prompt.Substring(0, [Math]::Min(500, $prompt.Length)) + "...")
        }

        $claudeArgs = @("--print", "--dangerously-skip-permissions")
        $outFile = Join-Path $script:RalphDir "healing_out_$attempt.log"
        $errFile = Join-Path $script:RalphDir "healing_err_$attempt.log"

        $subResult = Invoke-ClaudeSubprocess `
            -ClaudePath $claudePath `
            -ClaudeArgs $claudeArgs `
            -Prompt $prompt `
            -OutFile $outFile `
            -ErrFile $errFile

        $lastOutput = $subResult.Output

        if ($subResult.TimedOut) {
            Write-Host "  Healing attempt $attempt timed out" -ForegroundColor Yellow
            continue
        }

        if ($subResult.ExitCode -ne 0) {
            Write-Host "  Healing attempt $attempt failed (exit $($subResult.ExitCode))" -ForegroundColor Yellow
            continue
        }

        # Validate fix by re-running tiered health check (force full run to be thorough)
        Write-Host "  Validating fix..." -ForegroundColor Cyan
        $recheck = Invoke-TieredHealthCheck -ForceFullRun

        if (-not $recheck.HasErrors) {
            Write-Host "  Codebase is clean!" -ForegroundColor Green
            return @{
                Success      = $true
                AttemptsUsed = $attempt
                FixSummary   = "Fixed after $attempt attempt(s). Tier $($healingState.failedTier) errors resolved."
            }
        }

        Write-Host "  Still has errors (Tier $($recheck.FailedTier))" -ForegroundColor Yellow

        # Update diagnostics for next attempt
        $healingState.rawDiagnostics = $recheck.RawDiagnostics
        $healingState.failedTier = $recheck.FailedTier
    }

    return @{
        Success      = $false
        AttemptsUsed = $attempt
        FixSummary   = "Failed to fix after $MaxAttempts attempts"
    }
}

function Build-HealingPrompt {
    <#
    .SYNOPSIS
        Build a prompt for Claude to fix detected codebase errors.
        Includes tier-specific diagnostics and previous attempt context.
    #>
    param(
        [Parameter(Mandatory)]$HealingState,
        [int]$Attempt = 1,
        [string]$PreviousOutput = ""
    )

    $tierNames = @{ 1 = "Syntax/File Integrity"; 2 = "Import/Collection"; 3 = "Test Execution" }
    $tierName = $tierNames[[int]$HealingState.failedTier]

    $prompt = @"
# HEALING MODE - Fix Codebase Errors (Tier $($HealingState.failedTier): $tierName)

You are in HEALING MODE. The codebase has errors that must be fixed before the sprint can continue.

## Priority
Fix ALL errors below. Do NOT add new features. Do NOT refactor. ONLY fix what is broken.

## Tier $($HealingState.failedTier) Diagnostics
$($HealingState.rawDiagnostics)

## Context
- Sprint focus area: $($HealingState.focusArea)
- Story in progress: $($HealingState.storyId)
- Failed health check tier: $($HealingState.failedTier) ($tierName)
- This is healing attempt $Attempt

## Instructions
1. Read the failing files to understand what's expected
2. Identify the root cause of each error
3. Make the MINIMAL fix needed -- do not refactor or improve
4. Run ``pytest tests/ --tb=short -q`` to verify ALL tests pass
5. Commit with message: ``fix(healing): <what you fixed>``

IMPORTANT: After fixing, run the full test suite. ALL tests must pass.
"@

    if ($Attempt -gt 1 -and $PreviousOutput) {
        $truncated = $PreviousOutput.Substring(0, [Math]::Min(1000, $PreviousOutput.Length))
        $prompt += @"

## Previous Attempt Output (attempt $($Attempt - 1))
The previous fix attempt did NOT resolve all errors. Here is what was tried:
``````
$truncated
``````

Try a DIFFERENT approach this time.
"@
    }

    return $prompt
}
```

### Step 4: Run test to verify it passes

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add Invoke-HealingSession with tier-aware retry loop"
```

---

## Task 5: Add `Invoke-PostIterationHealing` -- The Top-Level Orchestrator

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Invoke-PostIterationHealing' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        $script:State = @{
            SessionId = "orch-test"; IterationCount = 12;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }
    }

    It 'does nothing when all tiers pass' {
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0; TierResults = @(); RawDiagnostics = "" }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing" -ChangedFiles @()
        $result.HealingNeeded | Should -BeFalse
        $script:HealingStateFile | Should -Not -Exist
    }

    It 'skips healing when disabled in config' {
        Mock Get-RalphConfig {
            return @{ selfHealing = @{ enabled = $false } }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing"
        $result.HealingNeeded | Should -BeFalse
    }

    It 'runs full healing flow when tier 1 errors detected' {
        Mock Invoke-TieredHealthCheck {
            return @{
                HasErrors = $true; FailedTier = 1
                RawDiagnostics = "SYNTAX: config.py -> SyntaxError"
                TierResults = @(@{ Tier = 1; HasErrors = $true })
            }
        }

        Mock Invoke-HealingSession {
            return @{ Success = $true; AttemptsUsed = 1; FixSummary = "Fixed syntax" }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "pipeline" -ChangedFiles @("src/config.py")
        $result.HealingNeeded | Should -BeTrue
        $result.HealingSuccess | Should -BeTrue
        $result.FailedTier | Should -Be 1
        $script:HealingStateFile | Should -Not -Exist  # cleaned up
    }

    It 'returns failure when healing cannot fix errors' {
        Mock Invoke-TieredHealthCheck {
            return @{
                HasErrors = $true; FailedTier = 3
                RawDiagnostics = "FAIL: test.py -> AssertionError"
            }
        }

        Mock Invoke-HealingSession {
            return @{ Success = $false; AttemptsUsed = 3; FixSummary = "Could not fix" }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing"
        $result.HealingNeeded | Should -BeTrue
        $result.HealingSuccess | Should -BeFalse
    }

    It 'skips after failure when runAfterFailure is false' {
        Mock Get-RalphConfig {
            return @{ selfHealing = @{
                enabled = $true; runAfterFailure = $false; runAfterSuccess = $true; maxAttempts = 3
            }}
        }

        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }

        $result = Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -IterationSuccess $false
        Should -Not -Invoke Invoke-TieredHealthCheck
    }

    It 'passes changed files to tiered health check' {
        Mock Invoke-TieredHealthCheck {
            $script:passedChangedFiles = $ChangedFiles
            return @{ HasErrors = $false; FailedTier = 0 }
        }
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true } } }

        $script:passedChangedFiles = $null
        Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -ChangedFiles @("src/foo.py", "src/bar.py")

        $script:passedChangedFiles | Should -Not -BeNullOrEmpty
        $script:passedChangedFiles.Count | Should -Be 2
    }
}
```

### Step 2: Run test to verify it fails

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: FAIL - function not defined

### Step 3: Write minimal implementation

Append to `scripts/ralph/lib/healing.ps1`:

```powershell
function Invoke-PostIterationHealing {
    <#
    .SYNOPSIS
        Post-iteration orchestrator. Runs tiered health check, triggers healing if needed.
        Called from Resolve-ClaudeResult after every story iteration.
    .PARAMETER StoryId
        Current story being worked on
    .PARAMETER FocusArea
        Current focus area
    .PARAMETER ChangedFiles
        Files changed in this iteration (from git diff)
    .PARAMETER IterationSuccess
        Whether the iteration itself succeeded (controls runAfterSuccess/runAfterFailure)
    .RETURNS
        Hashtable: HealingNeeded, HealingSuccess, AttemptsUsed, FailedTier
    #>
    param(
        [string]$StoryId = "",
        [string]$FocusArea = "",
        [string[]]$ChangedFiles = @(),
        [bool]$IterationSuccess = $true
    )

    $result = @{
        HealingNeeded  = $false
        HealingSuccess = $false
        AttemptsUsed   = 0
        FailedTier     = 0
    }

    # Check config
    $config = Get-RalphConfig
    $shConfig = $config.selfHealing

    if ($shConfig -and $shConfig.enabled -eq $false) { return $result }
    if ($IterationSuccess -and $shConfig -and $shConfig.runAfterSuccess -eq $false) { return $result }
    if (-not $IterationSuccess -and $shConfig -and $shConfig.runAfterFailure -eq $false) { return $result }

    $maxAttempts = if ($shConfig -and $shConfig.maxAttempts) { $shConfig.maxAttempts } else { 3 }
    $fullRunCadence = if ($shConfig -and $shConfig.fullRunCadence) { $shConfig.fullRunCadence } else { 3 }

    # Run tiered health check
    $health = Invoke-TieredHealthCheck -ChangedFiles $ChangedFiles -FullRunCadence $fullRunCadence

    if (-not $health.HasErrors) {
        return $result
    }

    # Errors detected -- begin healing
    $result.HealingNeeded = $true
    $result.FailedTier = $health.FailedTier

    Suspend-SprintForHealing -HealthResult $health -StoryId $StoryId -FocusArea $FocusArea

    $healResult = Invoke-HealingSession -MaxAttempts $maxAttempts
    $result.AttemptsUsed = $healResult.AttemptsUsed
    $result.HealingSuccess = $healResult.Success

    Resume-SprintFromHealing -Success $healResult.Success -AttemptCount $healResult.AttemptsUsed -FixSummary $healResult.FixSummary

    return $result
}
```

### Step 4: Run test to verify it passes

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add Invoke-PostIterationHealing orchestrator with tiered checks"
```

---

## Task 6: Wire Healing into `Resolve-ClaudeResult` and `ralph.ps1`

**Files:**
- Modify: `scripts/ralph/lib/claude.ps1:448` (before final return in Resolve-ClaudeResult)
- Modify: `scripts/ralph/ralph.ps1` (dot-source order + file path init)

### Step 1: Write integration test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Resolve-ClaudeResult healing integration' {
    It 'calls Invoke-PostIterationHealing after successful story' {
        Mock Invoke-PostIterationHealing {
            $script:healingCalledWith = @{ StoryId = $StoryId; FocusArea = $FocusArea }
            return @{ HealingNeeded = $false; HealingSuccess = $false; AttemptsUsed = 0; FailedTier = 0 }
        }
        Mock Log-StateTransition {}
        Mock Get-GitDiffStats { return @{ Added = 5; Deleted = 2 } }
        Mock Get-DiffQualityScore { return @{ testRatio = 0.5; churnRisk = "low"; sizeAppropriate = $true; warnings = @() } }
        Mock Compare-TestBaseline { return @{ hasRegression = $false } }
        Mock Record-Metric {}
        Mock Log-StoryVerification {}
        Mock Invoke-CodeReview { return $null }
        Mock Update-TestBaseline {}
        Mock Save-StoryProgress {}
        Mock Update-LearningDb {}
        Mock Get-SprintTokenBudget {}
        Mock Append-SessionTimeline {}
        Mock Get-FileOperations { return @() }

        $script:State = @{ IterationCount = 1; ConsecutiveFailures = 0; CurrentRetryCount = 1; CurrentMode = "Standard" }
        $script:healingCalledWith = $null

        Resolve-ClaudeResult -SubResult @{ TimedOut = $false; ExitCode = 0 } -Ctx @{
            TransitionContext = @{}; IsStoryWork = $true; StoryId = "US-005"
            FocusAreaId = "testing"; Identifier = "US-005"
            StoryObj = @{ acceptanceCriteria = @("test passes") }
            IterationDuration = [TimeSpan]::FromMinutes(2)
            TokensUsed = 1000; TestResults = @{ passed = 5; failed = 0 }
            PhaseTimings = @{ read_ms = 0; analyze_ms = 0; implement_ms = 0; test_ms = 0; commit_ms = 0 }
            GitStateBefore = @{ hash = "abc123" }; FileOps = @(); Commits = @(); ClaudeOutput = "done"
        }

        $script:healingCalledWith | Should -Not -BeNullOrEmpty
        $script:healingCalledWith.StoryId | Should -Be "US-005"
    }

    It 'aborts sprint when healing fails' {
        Mock Invoke-PostIterationHealing {
            return @{ HealingNeeded = $true; HealingSuccess = $false; AttemptsUsed = 3; FailedTier = 3 }
        }
        Mock Log-StateTransition {}
        Mock Get-GitDiffStats { return @{ Added = 0; Deleted = 0 } }
        Mock Get-DiffQualityScore { return @{ testRatio = 0; churnRisk = "low"; sizeAppropriate = $true; warnings = @() } }
        Mock Compare-TestBaseline { return $null }
        Mock Record-Metric {}
        Mock Append-SessionTimeline {}
        Mock Log-StoryVerification {}
        Mock Update-TestBaseline {}
        Mock Save-StoryProgress {}
        Mock Update-LearningDb {}
        Mock Get-SprintTokenBudget {}
        Mock Get-FileOperations { return @() }

        $script:State = @{ IterationCount = 1; ConsecutiveFailures = 0; CurrentRetryCount = 1; CurrentMode = "Standard" }

        $result = Resolve-ClaudeResult -SubResult @{ TimedOut = $false; ExitCode = 0 } -Ctx @{
            TransitionContext = @{}; IsStoryWork = $true; StoryId = "US-005"
            FocusAreaId = "testing"; Identifier = "US-005"
            StoryObj = @{ acceptanceCriteria = @() }
            IterationDuration = [TimeSpan]::FromMinutes(2)
            TokensUsed = 1000; TestResults = $null
            PhaseTimings = @{ read_ms = 0; analyze_ms = 0; implement_ms = 0; test_ms = 0; commit_ms = 0 }
            GitStateBefore = @{ hash = "abc123" }; FileOps = @(); Commits = @(); ClaudeOutput = "done"
        }

        $result.Success | Should -BeFalse
    }
}
```

### Step 2: Modify `ralph.ps1`

**A) Add dot-source** (after `quality.ps1`, before `prompts.ps1`):

```powershell
. (Join-Path $libDir "healing.ps1")
```

**B) Add file path initialization** (with other `$script:*File` declarations):

```powershell
$script:HealingLogFile = Join-Path $script:RalphDir "healing_log.jsonl"
$script:HealingStateFile = Join-Path $script:RalphDir "healing_state.json"
```

### Step 3: Modify `Resolve-ClaudeResult` in `claude.ps1`

Insert BEFORE the final `return` statement (around line 448):

```powershell
    # === POST-ITERATION HEALING ===
    # After any story iteration, run tiered health check and heal if needed
    if ($Ctx.IsStoryWork) {
        # Get changed files from this iteration for Tier 1 targeted compile check
        $changedFiles = @()
        if ($Ctx.FileOps) {
            $changedFiles = @($Ctx.FileOps | ForEach-Object {
                if ($_.path) { $_.path } elseif ($_.file) { $_.file }
            } | Where-Object { $_ })
        }

        $healResult = Invoke-PostIterationHealing `
            -StoryId $Ctx.StoryId `
            -FocusArea $Ctx.FocusAreaId `
            -ChangedFiles $changedFiles `
            -IterationSuccess $success

        if ($healResult.HealingNeeded -and -not $healResult.HealingSuccess) {
            $success = $false
            $iterationStatus = "healing_failed"
            Write-Host "  Sprint aborted: Tier $($healResult.FailedTier) errors could not be healed" -ForegroundColor Red
        }
    }

    return @{
        Success         = $success
        IterationStatus = $iterationStatus
    }
```

### Step 4: Run tests

Run: `Invoke-Pester -Path 'scripts/ralph/tests/Healing.Tests.ps1' -Output Detailed`
Expected: All tests PASS

### Step 5: Commit

```bash
git add scripts/ralph/lib/claude.ps1 scripts/ralph/lib/healing.ps1 scripts/ralph/ralph.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): wire tiered self-healing into post-iteration flow"
```

---

## Task 7: Add `selfHealing` Config Section to `ralph-config.json`

**Files:**
- Modify: `scripts/ralph/ralph-config.json`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Self-healing config' {
    It 'ralph-config.json has selfHealing section with tiered settings' {
        $configPath = Join-Path $PSScriptRoot '..' 'ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json

        $config.selfHealing | Should -Not -BeNullOrEmpty
        $config.selfHealing.enabled | Should -BeOfType [bool]
        $config.selfHealing.maxAttempts | Should -BeGreaterThan 0
        $config.selfHealing.runAfterSuccess | Should -BeOfType [bool]
        $config.selfHealing.runAfterFailure | Should -BeOfType [bool]
        $config.selfHealing.fullRunCadence | Should -BeGreaterThan 0
        $config.selfHealing.criticalFiles | Should -Not -BeNullOrEmpty
    }
}
```

### Step 2: Add config section

Add to `ralph-config.json` (after the `"regression"` section):

```json
    "selfHealing": {
        "enabled": true,
        "maxAttempts": 3,
        "runAfterSuccess": true,
        "runAfterFailure": true,
        "healingTimeout": 300,
        "fullRunCadence": 3,
        "pytestArgs": "tests/ --tb=short -q --no-header",
        "criticalFiles": ["main.py", "src/__init__.py", "config.yaml", "src/config.py", "src/pipeline.py"]
    },
```

### Step 3: Run tests, commit

```bash
git add scripts/ralph/ralph-config.json scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add selfHealing config with tiered check settings"
```

---

## Task 8: Add Healing Metrics to `Record-Metric`

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1` (Invoke-HealingSession -- add Record-Metric calls)
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Healing metrics recording' {
    It 'records healing iteration as mode=Healing in metrics CSV' {
        Mock Record-Metric { $script:capturedMetricArgs = $PSBoundParameters }
        Mock Invoke-ClaudeSubprocess {
            return @{ Exited = $true; ExitCode = 0; TimedOut = $false; Output = "Fixed";
                      ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date) }
        }
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0 }
        }

        $script:RalphDir = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        $script:State = @{ SessionId = "m-test"; IterationCount = 5; CurrentMode = "Standard"; ConsecutiveFailures = 0 }
        $script:capturedMetricArgs = $null

        @{ paused = $true; storyId = "US-005"; focusArea = "testing"; failedTier = 2;
           rawDiagnostics = "IMPORT error" } |
            ConvertTo-Json -Depth 10 | Set-Content $script:HealingStateFile

        Invoke-HealingSession -MaxAttempts 1

        $script:capturedMetricArgs | Should -Not -BeNullOrEmpty
        $script:capturedMetricArgs.Mode | Should -Be "Healing"
    }
}
```

### Step 2: Add Record-Metric call inside Invoke-HealingSession

After the validation check succeeds (codebase is clean), add:

```powershell
Record-Metric -StoryId "HEALING-$($healingState.storyId)" -Mode "Healing" `
    -DurationMin ([math]::Round(((Get-Date) - $attemptStart).TotalMinutes, 0)) `
    -Success $true -Timeout $false -TokensUsed 0 `
    -ErrorCategory "" -TestResults $null `
    -RetryCount $attempt `
    -LinesAdded 0 -LinesDeleted 0
```

### Step 3: Run tests, commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): record healing iterations in metrics CSV"
```

---

## Task 9: Add `Get-HealingSummary` for Sprint Reports

**Files:**
- Modify: `scripts/ralph/lib/healing.ps1`
- Modify: `scripts/ralph/tests/Healing.Tests.ps1`

### Step 1: Write the failing test

Append to `Healing.Tests.ps1`:

```powershell
Describe 'Get-HealingSummary' {
    It 'summarizes healing activity from log' {
        $script:RalphDir = $TestDrive
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"

        @{ timestamp = "2026-01-28T10:00:00"; event = "healing_started"; data = @{ failedTier = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile
        @{ timestamp = "2026-01-28T10:01:00"; event = "healing_resolved"; data = @{ attempts = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile
        @{ timestamp = "2026-01-28T10:05:00"; event = "healing_started"; data = @{ failedTier = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile
        @{ timestamp = "2026-01-28T10:06:00"; event = "healing_resolved"; data = @{ attempts = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 2
        $summary.TotalResolved | Should -Be 2
        $summary.TotalFailed | Should -Be 0
        $summary.TotalAttempts | Should -Be 3
    }

    It 'returns empty summary when no log exists' {
        $script:RalphDir = $TestDrive
        $script:HealingLogFile = Join-Path $TestDrive "nonexistent.jsonl"

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 0
    }
}
```

### Step 2: Implement

Append to `scripts/ralph/lib/healing.ps1`:

```powershell
function Get-HealingSummary {
    <#
    .SYNOPSIS
        Summarize healing activity for sprint reports.
    #>
    $logPath = if ($script:HealingLogFile) { $script:HealingLogFile }
               else { Join-Path $script:RalphDir "healing_log.jsonl" }

    $summary = @{
        TotalHealingSessions = 0
        TotalResolved        = 0
        TotalFailed          = 0
        TotalAttempts        = 0
        TierBreakdown        = @{ 1 = 0; 2 = 0; 3 = 0 }
        Events               = @()
    }

    if (-not (Test-Path $logPath)) { return $summary }

    $lines = Get-Content $logPath
    foreach ($line in $lines) {
        try {
            $entry = $line | ConvertFrom-Json
            $summary.Events += $entry

            switch ($entry.event) {
                "healing_started" {
                    $summary.TotalHealingSessions++
                    $tier = if ($entry.data.failedTier) { [int]$entry.data.failedTier } else { 0 }
                    if ($tier -ge 1 -and $tier -le 3) { $summary.TierBreakdown[$tier]++ }
                }
                "healing_resolved" {
                    $summary.TotalResolved++
                    if ($entry.data.attempts) { $summary.TotalAttempts += $entry.data.attempts }
                }
                "healing_failed" {
                    $summary.TotalFailed++
                    if ($entry.data.attempts) { $summary.TotalAttempts += $entry.data.attempts }
                }
            }
        } catch {}
    }

    return $summary
}
```

### Step 3: Run tests, commit

```bash
git add scripts/ralph/lib/healing.ps1 scripts/ralph/tests/Healing.Tests.ps1
git commit -m "feat(ralph): add Get-HealingSummary with tier breakdown"
```

---

## Task 10: Wire `Get-HealingSummary` into `Save-SprintArchive`

**Files:**
- Modify: `scripts/ralph/lib/sprint.ps1` (New-SprintReport function)

### Step 1: Find sprint report generation in `New-SprintReport`

Add a "Self-Healing" section after existing report sections:

```powershell
# Inside New-SprintReport, after existing sections:
$healingSummary = Get-HealingSummary
if ($healingSummary.TotalHealingSessions -gt 0) {
    $report += "`n## Self-Healing`n"
    $report += "- Healing sessions: $($healingSummary.TotalHealingSessions)`n"
    $report += "- Resolved: $($healingSummary.TotalResolved)`n"
    $report += "- Failed: $($healingSummary.TotalFailed)`n"
    $report += "- Total attempts: $($healingSummary.TotalAttempts)`n"
    $report += "- Tier breakdown: T1=$($healingSummary.TierBreakdown[1]), T2=$($healingSummary.TierBreakdown[2]), T3=$($healingSummary.TierBreakdown[3])`n"
}
```

### Step 2: Run all tests

Run: `Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed`
Expected: All tests PASS

### Step 3: Commit

```bash
git add scripts/ralph/lib/sprint.ps1
git commit -m "feat(ralph): include tiered healing summary in sprint reports"
```

---

## Task 11: Update CLAUDE.md with Self-Healing Documentation

**Files:**
- Modify: `CLAUDE.md`

### Step 1: Add documentation

In the Ralph Loop section, after the module structure table, add:

```markdown
**Self-Healing:** After every iteration, Ralph runs a 3-tier health check:

| Tier | What | Speed | When |
|------|------|-------|------|
| T1 | `py_compile` changed files, merge conflicts, critical file existence, config validation | <2s | Every iteration |
| T2 | `pytest --collect-only` (import/syntax errors without running tests) | 5-15s | Every iteration |
| T3 | Full `pytest` run | 30-120s | Every Nth iteration (default: 3), first iteration, or forced |

If any tier detects errors: sprint pauses, Claude healing session spawns (up to 3 attempts), fixes are validated by re-running the failing tier, everything is logged to `healing_log.jsonl` with tier info, diagnostics, and thought process.

Config: `ralph-config.json` -> `selfHealing` section. Logs: `scripts/ralph/healing_log.jsonl`. State: `scripts/ralph/healing_state.json`.
```

Also add `healing.ps1` to the module structure table:

```markdown
| `healing.ps1` | Self-healing: tiered health checks, healing sessions, logging |
```

### Step 2: Commit

```bash
git add CLAUDE.md
git commit -m "docs: add tiered self-healing Ralph documentation to CLAUDE.md"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Tiered health checks: `Invoke-FastHealthCheck` (T1), `Invoke-CollectionHealthCheck` (T2), `Invoke-FullHealthCheck` (T3), `Invoke-TieredHealthCheck` (orchestrator) | healing.ps1, Healing.Tests.ps1 |
| 2 | `Log-HealingEvent` -- JSONL audit trail with tier info | healing.ps1 |
| 3 | `Suspend/Resume-SprintForHealing` + `Test-HealingInProgress` | healing.ps1 |
| 4 | `Invoke-HealingSession` -- Claude fix loop with tier-aware validation | healing.ps1 |
| 5 | `Invoke-PostIterationHealing` -- top-level orchestrator | healing.ps1 |
| 6 | Wire into `Resolve-ClaudeResult` + dot-source + file paths | claude.ps1, ralph.ps1 |
| 7 | `selfHealing` config section with tiered settings | ralph-config.json |
| 8 | Healing metrics in CSV | healing.ps1 |
| 9 | `Get-HealingSummary` with tier breakdown | healing.ps1 |
| 10 | Sprint report integration | sprint.ps1 |
| 11 | CLAUDE.md docs | CLAUDE.md |

**Total: 11 tasks, ~12 functions, 1 new file (`healing.ps1`), 4 modified files, 1 test file**

**Tier overhead per iteration:**
- T1 + T2 (every iteration): ~5-17s
- T3 (every 3rd iteration): +30-120s
- Average per iteration: ~15-50s additional (negligible vs. 5-10min Claude iterations)
