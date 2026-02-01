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
        $mainPy = Join-Path $projectRoot "main.py"
        $configOutput = Invoke-Expression "python `"$mainPy`" --validate-config 2>&1"
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

function Log-HealingEvent {
    <#
    .SYNOPSIS
        Append a structured event to healing_log.jsonl.
        Every fix, diagnosis, and thought process is permanently recorded.
    .PARAMETER Event
        Event type: healing_started, healing_attempt, healing_resolved, healing_failed, healing_skipped,
                    stall_detected, stall_timeout, stall_recovery
    .PARAMETER Data
        Hashtable of event-specific data (tier, errors, fix description, thought process, etc.)
    #>
    param(
        [Parameter(Mandatory)]
        [ValidateSet("healing_started", "healing_attempt", "healing_resolved", "healing_failed", "healing_skipped", "stall_detected", "stall_timeout", "stall_recovery")]
        [string]$Event,

        [Parameter(Mandatory)]
        [hashtable]$Data
    )

    $logPath = if ($script:HealingLogFile) {
        $script:HealingLogFile
    } else {
        if ($script:Paths) { $script:Paths.HealingLogFile } else { Join-Path $script:RalphDir "session\healing_log.jsonl" }
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

# Track stall state across heartbeat calls
$script:StallState = @{
    StallStartTime = $null
    StallWarningIssued = $false
    ConsecutiveStalls = 0
    LastActivityTime = $null
}

function Test-ClaudeStall {
    <#
    .SYNOPSIS
        Check if Claude subprocess appears stalled and log healing events.
        Called periodically during subprocess wait loop.
    .PARAMETER WaitingSeconds
        Total seconds spent waiting for Claude
    .PARAMETER TimeSinceActivity
        Seconds since last detected activity (output, file changes)
    .PARAMETER StallThreshold
        Seconds without activity before considered stalled (default: 120)
    .RETURNS
        Hashtable: IsStalled, ShouldWarn, ShouldKill
    #>
    param(
        [Parameter(Mandatory)][int]$WaitingSeconds,
        [Parameter(Mandatory)][int]$TimeSinceActivity,
        [int]$StallThreshold = 180,   # Increased from 120 - complex tasks need more time
        [int]$KillThreshold = 360     # Increased from 300 - give more breathing room
    )

    $result = @{
        IsStalled = $false
        ShouldWarn = $false
        ShouldKill = $false
    }

    # Not stalled if recent activity
    if ($TimeSinceActivity -lt $StallThreshold) {
        $script:StallState.StallStartTime = $null
        $script:StallState.StallWarningIssued = $false
        $script:StallState.LastActivityTime = Get-Date
        return $result
    }

    # Stall detected
    $result.IsStalled = $true

    # Record stall start time
    if (-not $script:StallState.StallStartTime) {
        $script:StallState.StallStartTime = Get-Date

        # Log stall detection to healing log
        Log-HealingEvent -Event "stall_detected" -Data @{
            waitingSeconds = $WaitingSeconds
            timeSinceActivity = $TimeSinceActivity
            storyId = $script:State.CurrentStoryId
            focusArea = $script:State.CurrentFocusArea
        }

        Write-Host "  [STALL] No activity for ${TimeSinceActivity}s - monitoring..." -ForegroundColor Yellow
    }

    # Issue warning once
    if (-not $script:StallState.StallWarningIssued -and $TimeSinceActivity -ge ($StallThreshold + 60)) {
        $script:StallState.StallWarningIssued = $true
        $result.ShouldWarn = $true
        Write-Host "  [STALL] Extended stall (${TimeSinceActivity}s) - Claude may be hung" -ForegroundColor Red
    }

    # Recommend kill if stall exceeds kill threshold
    if ($TimeSinceActivity -ge $KillThreshold) {
        $result.ShouldKill = $true
    }

    return $result
}

function Invoke-StallRecovery {
    <#
    .SYNOPSIS
        Handle recovery after a Claude stall/timeout.
        Logs the event and updates stall tracking state.
    .PARAMETER Reason
        Why recovery was triggered: "timeout", "manual_kill", "stall_threshold"
    .PARAMETER WaitingSeconds
        How long Claude was running before recovery
    .PARAMETER ProcessId
        The Claude process ID that was killed (if applicable)
    #>
    param(
        [Parameter(Mandatory)][string]$Reason,
        [int]$WaitingSeconds = 0,
        [int]$ProcessId = 0
    )

    $script:StallState.ConsecutiveStalls++

    # Log to healing system
    Log-HealingEvent -Event "stall_recovery" -Data @{
        reason = $Reason
        waitingSeconds = $WaitingSeconds
        processId = $ProcessId
        consecutiveStalls = $script:StallState.ConsecutiveStalls
        storyId = $script:State.CurrentStoryId
        focusArea = $script:State.CurrentFocusArea
    }

    # Update heartbeat with stall info
    Write-Heartbeat -Phase "stall_recovery" -Details @{
        reason = $Reason
        consecutiveStalls = $script:StallState.ConsecutiveStalls
    }

    Write-SessionLog -Event "stall_recovery" -Message "Claude stall recovery: $Reason after ${WaitingSeconds}s" -Data @{
        consecutiveStalls = $script:StallState.ConsecutiveStalls
    }

    # Reset stall tracking
    $script:StallState.StallStartTime = $null
    $script:StallState.StallWarningIssued = $false

    # If too many consecutive stalls, recommend pause
    if ($script:StallState.ConsecutiveStalls -ge 3) {
        Write-Host ""
        Write-Host "  [HEALING] $($script:StallState.ConsecutiveStalls) consecutive stalls detected" -ForegroundColor Red
        Write-Host "  Consider: Network issues, Claude API problems, or resource exhaustion" -ForegroundColor Yellow
        Write-Host ""
        return @{ ShouldPause = $true }
    }

    return @{ ShouldPause = $false }
}

function Reset-StallTracking {
    <#
    .SYNOPSIS
        Reset stall tracking after successful iteration.
    #>
    $script:StallState.ConsecutiveStalls = 0
    $script:StallState.StallStartTime = $null
    $script:StallState.StallWarningIssued = $false
    $script:StallState.LastActivityTime = Get-Date
}

function Test-HealingInProgress {
    <#
    .SYNOPSIS
        Check if a healing session is currently in progress.
    #>
    $statePath = if ($script:HealingStateFile) { $script:HealingStateFile }
                 else { if ($script:Paths) { $script:Paths.HealingStateFile } else { Join-Path $script:RalphDir "state\healing_state.json" } }

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
                 else { if ($script:Paths) { $script:Paths.HealingStateFile } else { Join-Path $script:RalphDir "state\healing_state.json" } }

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
                 else { if ($script:Paths) { $script:Paths.HealingStateFile } else { Join-Path $script:RalphDir "state\healing_state.json" } }

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
                 else { if ($script:Paths) { $script:Paths.HealingStateFile } else { Join-Path $script:RalphDir "state\healing_state.json" } }

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
            try {
                Record-Metric -StoryId "HEALING-$($healingState.storyId)" -Mode "Healing" `
                    -DurationMin ([math]::Round(((Get-Date) - $attemptStart).TotalMinutes, 1)) `
                    -Success $true -Timeout $false -TokensUsed 0 `
                    -ErrorCategory "" -TestResults "" `
                    -RetryCount $attempt `
                    -LinesAdded 0 -LinesDeleted 0
            } catch {}
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

function Get-HealingSummary {
    <#
    .SYNOPSIS
        Summarize healing activity for sprint reports.
    #>
    $logPath = if ($script:HealingLogFile) { $script:HealingLogFile }
               else { if ($script:Paths) { $script:Paths.HealingLogFile } else { Join-Path $script:RalphDir "session\healing_log.jsonl" } }

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
