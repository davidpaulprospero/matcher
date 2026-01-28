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
