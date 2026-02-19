# Ralph Loop Progress Watcher
# Usage: .\scripts\ralph\watch.ps1 [-Interval 5]
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1
#
# Run in a separate terminal to monitor Ralph loop progress.
# Press Ctrl+C to stop.

param(
    [int]$Interval = 5
)

# Import library functions
$ScriptDir = (Split-Path -Parent $MyInvocation.MyCommand.Path) -replace '\\', '/'
. (Join-Path $ScriptDir "lib/watch.ps1") -replace '\\', '/'

# Paths - state files in state/, session-volatile files in session/
$StateDir = (Join-Path $ScriptDir "state") -replace '\\', '/'
$SessionDir = (Join-Path $ScriptDir "session") -replace '\\', '/'
$PrdPath = (Join-Path $StateDir "prd.json") -replace '\\', '/'
$ProgressPath = (Join-Path $SessionDir "progress.txt") -replace '\\', '/'
$MetricsPath = (Join-Path $SessionDir "metrics.csv") -replace '\\', '/'
$BlockedPath = (Join-Path $ScriptDir "BLOCKED.md") -replace '\\', '/'
$QueuePath = (Join-Path $StateDir "queue.json") -replace '\\', '/'
$LogsDir = (Join-Path $ScriptDir "logs") -replace '\\', '/'

$loopCount = 0

Write-Host "Watching Ralph progress (refresh every ${Interval}s)... Ctrl+C to stop" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    # Clear viewport and scrollback to prevent log bleed from previous refresh
    [Console]::Clear()
    $loopCount++

    # Header
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  RALPH LOOP MONITOR - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "  (refreshes every ${Interval}s)" -ForegroundColor DarkGray
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""

    # Queue status
    $queueActive = Show-QueueStatus -QueuePath $QueuePath

    # Sprint info
    $sprintInfo = Show-SprintInfo -PrdPath $PrdPath
    if (-not $sprintInfo) {
        Start-Sleep -Seconds $Interval
        continue
    }

    # Blocked check
    Show-BlockedStatus -BlockedPath $BlockedPath | Out-Null

    # Session resolution
    $currentSession = Get-CurrentSession -QueuePath $QueuePath -MetricsPath $MetricsPath

    # Metrics Dashboard
    Write-Host ""
    Write-Host "  METRICS DASHBOARD" -ForegroundColor Yellow
    Write-Host "  ---------------------------------------------------------------------" -ForegroundColor DarkGray

    if (Test-Path $MetricsPath) {
        try {
            # Non-locking read to avoid blocking concurrent writers
            $fs = [System.IO.FileStream]::new($MetricsPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $sr = [System.IO.StreamReader]::new($fs, [System.Text.Encoding]::UTF8)
            try { $csvText = $sr.ReadToEnd() } finally { $sr.Close(); $fs.Close() }
            $metrics = $csvText | ConvertFrom-Csv

            # Get session metrics
            if ($currentSession) {
                $sessionMetrics = @($metrics | Where-Object { $_.session -eq $currentSession })
            } else {
                $sessions = $metrics | Group-Object session | Sort-Object { $_.Group[0].timestamp } -Descending
                if ($sessions.Count -gt 0) {
                    $sessionMetrics = @($sessions[0].Group)
                    $currentSession = $sessions[0].Name
                } else {
                    $sessionMetrics = @()
                }
            }

            if ($sessionMetrics.Count -gt 0) {
                # Session Stats
                $totalIterations = $sessionMetrics.Count
                $successes = @($sessionMetrics | Where-Object { $_.success -eq 'true' }).Count
                $timeouts = @($sessionMetrics | Where-Object { $_.timeout -eq 'true' }).Count
                $successRate = if ($totalIterations -gt 0) { [math]::Round(($successes / $totalIterations) * 100) } else { 0 }

                Write-Host "  Session: $currentSession" -ForegroundColor White
                $successColor = if ($successRate -ge 80) { 'Green' } elseif ($successRate -ge 50) { 'Yellow' } else { 'Red' }
                Write-Host "  Iterations: $totalIterations | Success: $successRate% | Timeouts: $timeouts" -ForegroundColor $successColor
                Write-Host ""

                # Check for extended metrics columns
                $hasTokens = $null -ne $sessionMetrics[0].PSObject.Properties['tokens_used']
                $hasErrors = $null -ne $sessionMetrics[0].PSObject.Properties['error_category']
                $hasLines = $null -ne $sessionMetrics[0].PSObject.Properties['lines_added']

                # Cost Attribution
                if ($hasTokens) {
                    Show-CostAttribution -SessionMetrics $sessionMetrics
                }

                # Anomaly Detection
                Show-AnomalyCheck -SessionMetrics $sessionMetrics

                # Error Breakdown
                if ($hasErrors) {
                    Show-ErrorBreakdown -SessionMetrics $sessionMetrics
                }

                # Productivity
                if ($hasLines) {
                    Show-ProductivityStats -SessionMetrics $sessionMetrics
                }

                # Timing Stats
                Show-TimingStats -SessionMetrics $sessionMetrics

                # Focus Area Health
                Show-FocusAreaHealth -SessionMetrics $sessionMetrics

                # Phase Breakdown (if available)
                $hasPhaseTimings = $null -ne $sessionMetrics[0].PSObject.Properties['phase_read_ms']
                if ($hasPhaseTimings) {
                    $withTimings = @($sessionMetrics | Where-Object {
                        $_.phase_read_ms -and $_.phase_read_ms -ne '' -and $_.phase_read_ms -ne '0'
                    })
                    if ($withTimings.Count -gt 0) {
                        Write-Host ""
                        Write-Host "  PHASE BREAKDOWN (avg):" -ForegroundColor Cyan

                        $avgRead = [math]::Round(($withTimings | ForEach-Object { [int]$_.phase_read_ms } | Measure-Object -Average).Average / 1000, 1)
                        $avgAnalyze = [math]::Round(($withTimings | ForEach-Object { [int]$_.phase_analyze_ms } | Measure-Object -Average).Average / 1000, 1)
                        $avgImpl = [math]::Round(($withTimings | ForEach-Object { [int]$_.phase_implement_ms } | Measure-Object -Average).Average / 1000, 1)
                        $avgTest = [math]::Round(($withTimings | ForEach-Object { [int]$_.phase_test_ms } | Measure-Object -Average).Average / 1000, 1)
                        $avgCommit = [math]::Round(($withTimings | ForEach-Object { [int]$_.phase_commit_ms } | Measure-Object -Average).Average / 1000, 1)
                        $totalAvg = $avgRead + $avgAnalyze + $avgImpl + $avgTest + $avgCommit

                        if ($totalAvg -gt 0) {
                            $pctRead = [math]::Round(($avgRead / $totalAvg) * 100)
                            $pctAnalyze = [math]::Round(($avgAnalyze / $totalAvg) * 100)
                            $pctImpl = [math]::Round(($avgImpl / $totalAvg) * 100)
                            $pctTest = [math]::Round(($avgTest / $totalAvg) * 100)
                            $pctCommit = [math]::Round(($avgCommit / $totalAvg) * 100)

                            Write-Host "    R:${pctRead}% A:${pctAnalyze}% I:${pctImpl}% T:${pctTest}% C:${pctCommit}%" -ForegroundColor Gray
                            Write-Host "    (Read:${avgRead}s Analyze:${avgAnalyze}s Impl:${avgImpl}s Test:${avgTest}s Commit:${avgCommit}s)" -ForegroundColor DarkGray
                        }
                    }
                }

                # Sprint diagnostics
                $diagFile = Join-Path $SessionDir "diagnostics.json"
                if (Test-Path $diagFile) {
                    try {
                        $diag = Get-Content $diagFile -Raw | ConvertFrom-Json
                        $flagList = @()
                        if ($diag.active_flags) {
                            $diag.active_flags.PSObject.Properties | Where-Object { $_.Value -eq $true } | ForEach-Object { $flagList += $_.Name }
                        }
                        $flagStr = if ($flagList.Count -gt 0) { $flagList -join ' | ' } else { 'none' }
                        $firstTry = if ($diag.metrics.first_attempt_success_rate) { [math]::Round($diag.metrics.first_attempt_success_rate * 100) } else { '?' }
                        $delta = ''
                        if ($diag.comparison -and $diag.comparison.vs_last_3_sprints -and $diag.comparison.vs_last_3_sprints.first_attempt_rate_delta) {
                            $delta = " ($($diag.comparison.vs_last_3_sprints.first_attempt_rate_delta))"
                        }
                        $heal = if ($null -ne $diag.metrics.healing_sessions) { $diag.metrics.healing_sessions } else { '?' }
                        $regress = if ($null -ne $diag.metrics.regressions_caught) { $diag.metrics.regressions_caught } else { '?' }
                        Write-Host ""
                        Write-Host "  [DIAG] Flags: $flagStr | 1st-try: ${firstTry}%${delta} | Heal: $heal | Regress: $regress" -ForegroundColor DarkCyan
                    }
                    catch {}
                }
            } else {
                Write-Host "  No metrics data for current session" -ForegroundColor DarkGray
            }
        } catch {
            Write-Host "  Error loading metrics: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "  No metrics file found" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "  ---------------------------------------------------------------------" -ForegroundColor DarkGray

    # Recent activity
    if (Test-Path $ProgressPath) {
        Write-Host ""
        Write-Host "  Recent activity:" -ForegroundColor DarkGray
        Get-Content $ProgressPath | Select-Object -Last 5 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor DarkGray
        }
    }

    # Logs and timeline
    Show-LatestLogs -LogsDir $LogsDir

    # Git status
    Show-GitStatus

    # Cleanup
    if ($loopCount % 50 -eq 0) {
        [System.GC]::Collect()
    }

    Start-Sleep -Seconds $Interval
}
