# scripts/ralph/lib/watch.ps1
# Reusable display functions for the watch dashboard
#
# Moved from watch-lib.ps1 to lib/watch.ps1 per convention

# ============================================================================
# QUEUE STATUS
# ============================================================================

function Show-QueueStatus {
    <#
    .SYNOPSIS
        Display queue mode progress
    .PARAMETER QueuePath
        Path to queue.json
    .RETURNS
        $true if queue is active, $false otherwise
    #>
    param([string]$QueuePath)

    if (-not (Test-Path $QueuePath)) { return $false }

    try {
        $queue = Get-Content $QueuePath -Raw | ConvertFrom-Json

        if (-not $queue.focusAreas) {
            Write-Host "  [!] Legacy queue format detected - run interview.ps1 to create new format" -ForegroundColor Yellow
            return $false
        }

        $allAreas = $queue.focusAreas
        $queueTotal = $allAreas.Count
        $completed = @($allAreas | Where-Object { $_.completed -eq $true })
        $queueCompleted = $completed.Count
        $currentArea = $allAreas | Where-Object { -not $_.completed } | Select-Object -First 1

        if ($queueTotal -eq 0) { return $false }

        # Progress bar
        $queuePct = [math]::Round(($queueCompleted / $queueTotal) * 100)
        $qBar = ("=" * [math]::Round(($queueCompleted / $queueTotal) * 30)) + ("-" * (30 - [math]::Round(($queueCompleted / $queueTotal) * 30)))

        Write-Host "  QUEUE MODE" -ForegroundColor Magenta
        Write-Host "  [$qBar] $queuePct% ($queueCompleted/$queueTotal areas)" -ForegroundColor Magenta

        if ($queue.interviewContext) {
            Write-Host "  Context: $($queue.interviewContext)" -ForegroundColor DarkGray
        }
        Write-Host ""

        # Show areas
        foreach ($area in $allAreas) {
            if ($area.completed) {
                Write-Host "    [DONE] $($area.id)" -ForegroundColor Green
            } elseif ($currentArea -and $area.id -eq $currentArea.id) {
                Write-Host "    [>>  ] $($area.id) (current)" -ForegroundColor Cyan
            } else {
                Write-Host "    [    ] $($area.id)" -ForegroundColor DarkGray
            }
        }
        Write-Host ""

        # Retry status
        if ($queue.networkRetries -and $queue.networkRetries.currentRetryCount -gt 0) {
            Write-Host "    Network retries: $($queue.networkRetries.currentRetryCount)/$($queue.networkRetries.maxRetries)" -ForegroundColor Yellow
        }
        if ($queue.apiRetries -and $queue.apiRetries.currentRetryCount -gt 0) {
            Write-Host "    API retries: $($queue.apiRetries.currentRetryCount)/$($queue.apiRetries.maxRetries)" -ForegroundColor Yellow
        }

        Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host ""
        return $true
    }
    catch { return $false }
}

# ============================================================================
# SPRINT INFO
# ============================================================================

function Show-SprintInfo {
    <#
    .SYNOPSIS
        Display sprint and story status
    .PARAMETER PrdPath
        Path to prd.json
    .RETURNS
        Hashtable with passed/failed/nextStory, or $null if no PRD
    #>
    param([string]$PrdPath)

    if (-not (Test-Path $PrdPath)) {
        Write-Host "  Waiting for PRD file..." -ForegroundColor Yellow
        return $null
    }

    try {
        $prd = Get-Content $PrdPath -Raw | ConvertFrom-Json
    } catch {
        Write-Host "  Error reading PRD (file updating)..." -ForegroundColor Yellow
        return $null
    }

    Write-Host "  Sprint: $($prd.sprintNumber) | Branch: $($prd.branchName)" -ForegroundColor Cyan
    if ($prd.focusArea) {
        Write-Host "  Focus: $($prd.focusArea)" -ForegroundColor Cyan
    }
    Write-Host ""

    $passed = 0
    $failed = 0
    $nextStory = $null

    foreach ($story in $prd.userStories) {
        if ($story.passes) {
            $passed++
            Write-Host "  [DONE] $($story.id): $($story.title)" -ForegroundColor Green
        } else {
            $failed++
            Write-Host "  [    ] $($story.id): $($story.title)" -ForegroundColor Yellow
            if ($null -eq $nextStory) { $nextStory = $story }
        }
    }

    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan

    # Progress bar
    $total = $passed + $failed
    if ($total -gt 0) {
        $pct = [math]::Round(($passed / $total) * 100)
        $bar = ("=" * [math]::Round(($passed / $total) * 50)) + ("-" * (50 - [math]::Round(($passed / $total) * 50)))

        if ($failed -eq 0) {
            Write-Host "  [$bar] $pct% ($passed/$total)" -ForegroundColor Green
            Write-Host ""
            Write-Host "  ALL COMPLETE!" -ForegroundColor Green
        } else {
            Write-Host "  [$bar] $pct% ($passed/$total)" -ForegroundColor Yellow
            Write-Host ""
            if ($nextStory) {
                Write-Host "  Current: $($nextStory.id) - $($nextStory.title)" -ForegroundColor White
            }
        }
    }

    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan

    return @{ Passed = $passed; Failed = $failed; NextStory = $nextStory; Prd = $prd }
}

# ============================================================================
# BLOCKED CHECK
# ============================================================================

function Show-BlockedStatus {
    param([string]$BlockedPath)

    if (Test-Path $BlockedPath) {
        Write-Host ""
        Write-Host "  !! BLOCKED !!" -ForegroundColor Red
        Write-Host ""
        Get-Content $BlockedPath | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        Write-Host ""
        return $true
    }
    return $false
}

# ============================================================================
# SESSION RESOLUTION
# ============================================================================

function Get-CurrentSession {
    param(
        [string]$QueuePath,
        [string]$MetricsPath
    )

    $currentSession = ""

    if (Test-Path $QueuePath) {
        try {
            $queueData = Get-Content $QueuePath -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($queueData.session.id) { $currentSession = $queueData.session.id }
            elseif ($queueData.sessionId) { $currentSession = $queueData.sessionId }
        } catch {}
    }

    if (-not $currentSession -and (Test-Path $MetricsPath)) {
        try {
            $lastMetric = Import-Csv $MetricsPath | Select-Object -Last 1
            if ($lastMetric) { $currentSession = $lastMetric.session }
        } catch {}
    }

    return $currentSession
}

# ============================================================================
# COST ATTRIBUTION
# ============================================================================

function Show-CostAttribution {
    param([array]$SessionMetrics)

    $tokensWithValues = @($SessionMetrics | Where-Object { $_.tokens_used -and $_.tokens_used -ne '' })
    if ($tokensWithValues.Count -eq 0) { return }

    $totalTokens = ($tokensWithValues | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Sum).Sum
    $avgTokens = [math]::Round(($tokensWithValues | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Average).Average)
    $estimatedCost = [math]::Round($totalTokens * 0.000003, 2)

    Write-Host "  COST ATTRIBUTION:" -ForegroundColor Cyan
    Write-Host "    Total tokens: $($totalTokens.ToString('N0'))" -ForegroundColor White
    Write-Host "    Avg/iteration: $($avgTokens.ToString('N0'))" -ForegroundColor White
    $costColor = if ($estimatedCost -lt 1) { 'Green' } elseif ($estimatedCost -lt 5) { 'Yellow' } else { 'Red' }
    Write-Host "    Est. total: `$$estimatedCost" -ForegroundColor $costColor

    # By focus area
    $focusCosts = $tokensWithValues | Group-Object focus_area | ForEach-Object {
        $fTokens = ($_.Group | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Sum).Sum
        @{ Name = $_.Name; Cost = [math]::Round($fTokens * 0.000003, 3); Pct = [math]::Round(($fTokens / $totalTokens) * 100) }
    } | Sort-Object { $_.Cost } -Descending

    if ($focusCosts.Count -gt 0) {
        Write-Host "    By Focus Area:" -ForegroundColor DarkGray
        foreach ($fc in $focusCosts | Select-Object -First 4) {
            Write-Host "      $($fc.Name): `$$($fc.Cost) ($($fc.Pct)%)" -ForegroundColor Gray
        }
    }
    Write-Host ""
}

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

function Show-AnomalyCheck {
    param([array]$SessionMetrics)

    Write-Host "  ANOMALY CHECK:" -ForegroundColor Cyan
    $anomalies = @()

    # Duration anomaly
    $durations = @($SessionMetrics | Where-Object { $_.duration_min -and $_.duration_min -ne '' } | ForEach-Object { [double]$_.duration_min })
    if ($durations.Count -gt 2) {
        $avgDur = ($durations | Measure-Object -Average).Average
        $maxDur = ($durations | Measure-Object -Maximum).Maximum
        if ($maxDur -gt ($avgDur * 3) -and $avgDur -gt 0) {
            $anomalies += "Duration spike: max $([math]::Round($maxDur, 1)) min (avg $([math]::Round($avgDur, 1)) min)"
        }
    }

    # Error rate anomaly
    $lastFive = @($SessionMetrics | Select-Object -Last 5)
    if ($lastFive.Count -ge 5) {
        $recentFailures = @($lastFive | Where-Object { $_.success -eq 'false' }).Count
        if ($recentFailures -ge 4) {
            $anomalies += "High failure rate: $recentFailures/5 recent iterations failed"
        }
    }

    # Timeout trend
    $recentTimeouts = @($SessionMetrics | Select-Object -Last 10 | Where-Object { $_.timeout -eq 'true' }).Count
    if ($recentTimeouts -ge 3) {
        $anomalies += "Timeout trend: $recentTimeouts timeouts in last 10 iterations"
    }

    if ($anomalies.Count -gt 0) {
        foreach ($anomaly in $anomalies) {
            Write-Host "    ! $anomaly" -ForegroundColor Red
        }
    } else {
        Write-Host "    No anomalies detected" -ForegroundColor Green
    }
    Write-Host ""
}

# ============================================================================
# ERROR BREAKDOWN
# ============================================================================

function Show-ErrorBreakdown {
    param([array]$SessionMetrics)

    $errors = @($SessionMetrics | Where-Object { $_.success -eq 'false' -and $_.error_category -and $_.error_category -ne '' })
    if ($errors.Count -eq 0) { return }

    Write-Host "  ERRORS:" -ForegroundColor Red
    $errorGroups = $errors | Group-Object error_category | Sort-Object Count -Descending
    foreach ($eg in $errorGroups) {
        Write-Host "    $($eg.Name): $($eg.Count)" -ForegroundColor Yellow
    }
    Write-Host ""
}

# ============================================================================
# PRODUCTIVITY STATS
# ============================================================================

function Show-ProductivityStats {
    param([array]$SessionMetrics)

    $withLines = @($SessionMetrics | Where-Object { $_.lines_added -or $_.lines_deleted })
    if ($withLines.Count -eq 0) { return }

    $linesAdded = ($withLines | ForEach-Object { if ($_.lines_added) { [int]$_.lines_added } else { 0 } } | Measure-Object -Sum).Sum
    $linesDeleted = ($withLines | ForEach-Object { if ($_.lines_deleted) { [int]$_.lines_deleted } else { 0 } } | Measure-Object -Sum).Sum
    $netLines = $linesAdded - $linesDeleted

    Write-Host "  PRODUCTIVITY:" -ForegroundColor Cyan
    Write-Host "    Lines added: +$linesAdded" -ForegroundColor Green
    Write-Host "    Lines deleted: -$linesDeleted" -ForegroundColor Red
    $netPrefix = if ($netLines -ge 0) { '+' } else { '' }
    Write-Host "    Net change: $netPrefix$netLines" -ForegroundColor White
    Write-Host ""
}

# ============================================================================
# TIMING STATS
# ============================================================================

function Show-TimingStats {
    param([array]$SessionMetrics)

    $withDuration = @($SessionMetrics | Where-Object { $_.duration_min -and $_.duration_min -ne '' })
    if ($withDuration.Count -eq 0) { return }

    $durations = $withDuration | ForEach-Object { [double]$_.duration_min }
    $totalDuration = ($durations | Measure-Object -Sum).Sum
    $avgDuration = [math]::Round(($durations | Measure-Object -Average).Average, 1)
    $maxDuration = ($durations | Measure-Object -Maximum).Maximum

    Write-Host "  TIMING:" -ForegroundColor Cyan
    Write-Host "    Total time: $([math]::Round($totalDuration, 0)) min" -ForegroundColor White
    Write-Host "    Avg/story: $avgDuration min" -ForegroundColor White
    Write-Host "    Longest: $maxDuration min" -ForegroundColor $(if ($maxDuration -le 15) { 'Green' } elseif ($maxDuration -le 30) { 'Yellow' } else { 'Red' })
}

# ============================================================================
# FOCUS AREA HEALTH
# ============================================================================

function Show-FocusAreaHealth {
    param([array]$SessionMetrics)

    $focusGroups = $SessionMetrics | Group-Object focus_area | Sort-Object Count -Descending
    if ($focusGroups.Count -eq 0) { return }

    Write-Host ""
    Write-Host "  FOCUS AREA HEALTH:" -ForegroundColor Cyan
    foreach ($fg in $focusGroups | Select-Object -First 5) {
        $areaSuccesses = @($fg.Group | Where-Object { $_.success -eq 'true' }).Count
        $areaTotal = $fg.Group.Count
        $areaRate = [math]::Round(($areaSuccesses / $areaTotal) * 100)
        $areaColor = if ($areaRate -ge 80) { 'Green' } elseif ($areaRate -ge 50) { 'Yellow' } else { 'Red' }

        $areaDurations = @($fg.Group | Where-Object { $_.duration_min -and $_.duration_min -ne '' } | ForEach-Object { [double]$_.duration_min })
        $areaAvgMin = if ($areaDurations.Count -gt 0) { [math]::Round(($areaDurations | Measure-Object -Average).Average, 1) } else { 0 }

        $areaTokens = @($fg.Group | Where-Object { $_.tokens_used -and $_.tokens_used -ne '' } | ForEach-Object { [int]$_.tokens_used })
        $areaCost = if ($areaTokens.Count -gt 0) { [math]::Round((($areaTokens | Measure-Object -Sum).Sum * 0.000003), 3) } else { 0 }

        Write-Host "    $($fg.Name): $areaRate% ($areaSuccesses/$areaTotal) | ${areaAvgMin} min avg | `$$areaCost" -ForegroundColor $areaColor
    }
}

# ============================================================================
# LOGS AND TIMELINE
# ============================================================================

function Show-LatestLogs {
    param([string]$LogsDir)

    if (-not (Test-Path $LogsDir)) { return }

    $latestLogDir = Get-ChildItem $LogsDir -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latestLogDir) { return }

    # Latest log file
    $latestLog = Get-ChildItem $latestLogDir.FullName -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        $age = [math]::Round(((Get-Date) - $latestLog.LastWriteTime).TotalMinutes, 1)
        Write-Host ""
        Write-Host "  Latest log: $($latestLog.Name) ($age min ago)" -ForegroundColor DarkGray
    }

    # Latest manifest
    $latestManifest = Get-ChildItem $latestLogDir.FullName -Filter "iteration_*_manifest.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestManifest) {
        Write-Host ""
        Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "  LATEST ITERATION MANIFEST" -ForegroundColor Yellow
        Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
        try {
            $manifest = Get-Content $latestManifest.FullName -Raw | ConvertFrom-Json
            Write-Host "    Iteration: $($manifest.iteration) | Story: $($manifest.storyId) | Status: $($manifest.status)" -ForegroundColor $(if ($manifest.status -eq 'completed') { 'Green' } else { 'Red' })
            Write-Host "    Duration: $($manifest.timestamps.durationSec)s | Tokens: $($manifest.metrics.tokensEstimated)" -ForegroundColor Gray
            Write-Host "    Git: +$($manifest.git.linesAdded)/-$($manifest.git.linesDeleted) lines | $($manifest.git.commits.Count) commit(s)" -ForegroundColor Gray
            Write-Host "    Tests: $($manifest.tests.passed) passed, $($manifest.tests.failed) failed" -ForegroundColor $(if ($manifest.tests.failed -eq 0) { 'Green' } else { 'Red' })
        }
        catch {
            Write-Host "    (manifest loading...)" -ForegroundColor DarkGray
        }
    }

    # Timeline
    $timelineFile = Join-Path $latestLogDir.FullName "session_timeline.jsonl"
    if (Test-Path $timelineFile) {
        Write-Host ""
        Write-Host "  TIMELINE (last 5 events):" -ForegroundColor Cyan
        $timelineLines = Get-Content $timelineFile -Tail 5 -ErrorAction SilentlyContinue
        if ($timelineLines) {
            foreach ($line in $timelineLines) {
                try {
                    $event = $line | ConvertFrom-Json
                    $eventTime = ([datetime]$event.ts).ToString("HH:mm:ss")
                    $eventName = $event.event
                    $details = @()
                    if ($event.iteration) { $details += "iter=$($event.iteration)" }
                    if ($event.storyId) { $details += "story=$($event.storyId)" }
                    if ($event.status) { $details += "$($event.status)" }
                    $detailStr = if ($details.Count -gt 0) { " ($($details -join ', '))" } else { "" }

                    $color = switch -Regex ($eventName) {
                        "complete" { "Green" }
                        "start" { "Cyan" }
                        "error|fail" { "Red" }
                        default { "Gray" }
                    }
                    Write-Host "    $eventTime $eventName$detailStr" -ForegroundColor $color
                }
                catch {}
            }
        }
    }

    # State transitions
    $stateFile = Join-Path $latestLogDir.FullName "state_transitions.jsonl"
    if (Test-Path $stateFile) {
        $stateLines = Get-Content $stateFile -Tail 3 -ErrorAction SilentlyContinue
        if ($stateLines -and $stateLines.Count -gt 0) {
            Write-Host ""
            Write-Host "  STATE MACHINE (last 3):" -ForegroundColor Cyan
            foreach ($line in $stateLines) {
                try {
                    $state = $line | ConvertFrom-Json
                    $stateTime = ([datetime]$state.timestamp).ToString("HH:mm:ss")
                    $transition = "$($state.from) -> $($state.to)"
                    $reason = if ($state.reason.Length -gt 30) { $state.reason.Substring(0, 27) + "..." } else { $state.reason }
                    $color = switch ($state.to) {
                        "completed" { "Green" }
                        "running" { "Cyan" }
                        "failed" { "Red" }
                        default { "Gray" }
                    }
                    Write-Host "    $stateTime [$transition] $reason" -ForegroundColor $color
                }
                catch {}
            }
        }
    }
}

# ============================================================================
# GIT STATUS
# ============================================================================

function Show-GitStatus {
    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Git status:" -ForegroundColor DarkGray
    $gitStatus = git status --short 2>$null | Select-Object -First 5
    if ($gitStatus) {
        $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    } else {
        Write-Host "    (clean)" -ForegroundColor DarkGray
    }
}
