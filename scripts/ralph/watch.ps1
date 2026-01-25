# Ralph Loop Progress Watcher
# Usage: .\scripts\ralph\watch.ps1 [-Interval 5]
#
# Run in a separate terminal to monitor Ralph loop progress.
# Press Ctrl+C to stop.

param(
    [int]$Interval = 5
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PrdPath = Join-Path $ScriptDir "prd.json"
$ProgressPath = Join-Path $ScriptDir "progress.txt"
$MetricsPath = Join-Path $ScriptDir "metrics.csv"
$BlockedPath = Join-Path $ScriptDir "BLOCKED.md"
$QueuePath = Join-Path $ScriptDir "queue.json"
$LogsDir = Join-Path $ScriptDir "logs"

$loopCount = 0

Write-Host "Watching Ralph progress (refresh every ${Interval}s)... Ctrl+C to stop" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host
    $loopCount++

    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  RALPH LOOP MONITOR - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "  (refreshes every ${Interval}s)" -ForegroundColor DarkGray
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""

    # Queue status (if active)
    $queueActive = $false
    if (Test-Path $QueuePath) {
        try {
            $queue = Get-Content $QueuePath -Raw | ConvertFrom-Json

            # Check for interview format (focusAreas) or legacy format (queue)
            $isInterviewFormat = $null -ne $queue.focusAreas
            $isLegacyFormat = $null -ne $queue.queue

            if ($isInterviewFormat -or $isLegacyFormat) {
                $queueActive = $true

                if ($isInterviewFormat) {
                    # Interview format: focusAreas array with completed flag
                    $allAreas = $queue.focusAreas
                    $queueTotal = $allAreas.Count
                    $completed = @($allAreas | Where-Object { $_.completed -eq $true })
                    $queueCompleted = $completed.Count
                    $queueSkipped = 0  # Interview format doesn't track skipped separately

                    # Find current (first incomplete)
                    $currentArea = $allAreas | Where-Object { -not $_.completed } | Select-Object -First 1
                    $queueCurrent = if ($currentArea) {
                        $idx = 0
                        for ($i = 0; $i -lt $allAreas.Count; $i++) {
                            if ($allAreas[$i].id -eq $currentArea.id) { $idx = $i; break }
                        }
                        $idx + 1
                    } else {
                        $queueTotal
                    }
                } else {
                    # Legacy format: queue array + completedAreas/skippedAreas arrays
                    $queueTotal = $queue.queue.Count
                    $queueCompleted = if ($queue.completedAreas) { $queue.completedAreas.Count } else { 0 }
                    $queueSkipped = if ($queue.skippedAreas) { $queue.skippedAreas.Count } else { 0 }
                    $queueCurrent = $queue.currentIndex + 1
                }

                if ($queueTotal -gt 0) {
                    # Queue progress bar
                    $queuePct = [math]::Round(($queueCompleted / $queueTotal) * 100)
                    $qBarWidth = 30
                    $qFilledWidth = [math]::Round(($queueCompleted / $queueTotal) * $qBarWidth)
                    $qEmptyWidth = $qBarWidth - $qFilledWidth
                    $qBar = ("=" * $qFilledWidth) + ("-" * $qEmptyWidth)

                    Write-Host "  QUEUE MODE" -ForegroundColor Magenta
                    Write-Host "  [$qBar] $queuePct% ($queueCompleted/$queueTotal areas)" -ForegroundColor Magenta

                    # Show interview context if available
                    if ($queue.interviewContext) {
                        Write-Host "  Context: $($queue.interviewContext)" -ForegroundColor DarkGray
                    }
                    Write-Host ""

                    # Show queue status based on format
                    if ($isInterviewFormat) {
                        foreach ($area in $allAreas) {
                            if ($area.completed) {
                                Write-Host "    [DONE] $($area.id)" -ForegroundColor Green
                            } elseif ($currentArea -and $area.id -eq $currentArea.id) {
                                Write-Host "    [>>  ] $($area.id) (current)" -ForegroundColor Cyan
                            } else {
                                Write-Host "    [    ] $($area.id)" -ForegroundColor DarkGray
                            }
                        }
                    } else {
                        # Legacy format display
                        foreach ($i in 0..($queueTotal - 1)) {
                            $area = $queue.queue[$i]
                            if ($queue.completedAreas -contains $area) {
                                Write-Host "    [DONE] $area" -ForegroundColor Green
                            } elseif ($queue.skippedAreas | Where-Object { $_.area -eq $area }) {
                                $skipInfo = $queue.skippedAreas | Where-Object { $_.area -eq $area } | Select-Object -First 1
                                Write-Host "    [SKIP] $area ($($skipInfo.reason))" -ForegroundColor Yellow
                            } elseif ($i -eq $queue.currentIndex) {
                                Write-Host "    [>>  ] $area (current)" -ForegroundColor Cyan
                            } else {
                                Write-Host "    [    ] $area" -ForegroundColor DarkGray
                            }
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
                    if ($queue.retryCount -gt 0) {
                        Write-Host "    Sprint retries: $($queue.retryCount)" -ForegroundColor Yellow
                    }

                    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
                    Write-Host ""
                }
            }
        } catch {}
    }

    # Check for PRD
    if (-not (Test-Path $PrdPath)) {
        Write-Host "  Waiting for PRD file..." -ForegroundColor Yellow
        Start-Sleep -Seconds $Interval
        continue
    }

    # Load PRD
    try {
        $prd = Get-Content $PrdPath -Raw | ConvertFrom-Json
    } catch {
        Write-Host "  Error reading PRD (file updating)..." -ForegroundColor Yellow
        Start-Sleep -Seconds $Interval
        continue
    }

    # Sprint info
    Write-Host "  Sprint: $($prd.sprintNumber) | Branch: $($prd.branchName)" -ForegroundColor Cyan
    if ($prd.focusArea) {
        Write-Host "  Focus: $($prd.focusArea)" -ForegroundColor Cyan
    }
    Write-Host ""

    # Story status
    $passed = 0
    $failed = 0
    $nextStory = $null

    foreach ($story in $prd.userStories) {
        if ($story.passes) {
            $passed++
            $status = "[DONE]"
            $color = "Green"
        } else {
            $failed++
            $status = "[    ]"
            $color = "Yellow"
            if ($null -eq $nextStory) { $nextStory = $story }
        }
        Write-Host "  $status $($story.id): $($story.title)" -ForegroundColor $color
    }

    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan

    # Progress bar
    $total = $passed + $failed
    if ($total -gt 0) {
        $pct = [math]::Round(($passed / $total) * 100)
        $barWidth = 50
        $filledWidth = [math]::Round(($passed / $total) * $barWidth)
        $emptyWidth = $barWidth - $filledWidth
        $bar = ("=" * $filledWidth) + ("-" * $emptyWidth)

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

    # Check for BLOCKED
    if (Test-Path $BlockedPath) {
        Write-Host ""
        Write-Host "  !! BLOCKED !!" -ForegroundColor Red
        Write-Host ""
        Get-Content $BlockedPath | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        Write-Host ""
    }

    # Get current session ID
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

    # Metrics Dashboard
    Write-Host ""
    Write-Host "  METRICS DASHBOARD" -ForegroundColor Yellow
    Write-Host "  ---------------------------------------------------------------------" -ForegroundColor DarkGray

    if (Test-Path $MetricsPath) {
        try {
            $metrics = Import-Csv $MetricsPath

            # Get session metrics - use current session or latest
            if ($currentSession) {
                $sessionMetrics = @($metrics | Where-Object { $_.session -eq $currentSession })
            } else {
                # Fallback to latest session
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

                # Check if we have the extended metrics columns
                $hasTokens = $null -ne $sessionMetrics[0].PSObject.Properties['tokens_used']
                $hasErrors = $null -ne $sessionMetrics[0].PSObject.Properties['error_category']
                $hasTests = $null -ne $sessionMetrics[0].PSObject.Properties['test_results']
                $hasLines = $null -ne $sessionMetrics[0].PSObject.Properties['lines_added']
                $hasRetries = $null -ne $sessionMetrics[0].PSObject.Properties['retry_count']
                $hasHour = $null -ne $sessionMetrics[0].PSObject.Properties['hour_of_day']

                # Token Usage (Cost Tracking)
                if ($hasTokens) {
                    $tokensWithValues = @($sessionMetrics | Where-Object { $_.tokens_used -and $_.tokens_used -ne '' })
                    if ($tokensWithValues.Count -gt 0) {
                        $totalTokens = ($tokensWithValues | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Sum).Sum
                        $avgTokens = [math]::Round(($tokensWithValues | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Average).Average)
                        $estimatedCost = [math]::Round($totalTokens * 0.000003, 2)  # ~$3/1M tokens

                        Write-Host "  COST:" -ForegroundColor Cyan
                        Write-Host "    Total tokens: $($totalTokens.ToString('N0'))" -ForegroundColor White
                        Write-Host "    Avg/iteration: $($avgTokens.ToString('N0'))" -ForegroundColor White
                        $costColor = if ($estimatedCost -lt 1) { 'Green' } else { 'Yellow' }
                        Write-Host "    Est. cost: `$$estimatedCost" -ForegroundColor $costColor
                        Write-Host ""
                    }
                }

                # Error Breakdown
                if ($hasErrors) {
                    $errors = @($sessionMetrics | Where-Object { $_.success -eq 'false' -and $_.error_category -and $_.error_category -ne '' })
                    if ($errors.Count -gt 0) {
                        Write-Host "  ERRORS:" -ForegroundColor Red
                        $errorGroups = $errors | Group-Object error_category | Sort-Object Count -Descending
                        foreach ($eg in $errorGroups) {
                            Write-Host "    $($eg.Name): $($eg.Count)" -ForegroundColor Yellow
                        }
                        Write-Host ""
                    }
                }

                # Test Results Summary
                if ($hasTests) {
                    $withTests = @($sessionMetrics | Where-Object { $_.test_results -and $_.test_results -ne '' })
                    if ($withTests.Count -gt 0) {
                        Write-Host "  TESTS:" -ForegroundColor Cyan
                        $passOnly = @($withTests | Where-Object { $_.test_results -notmatch 'fail' }).Count
                        Write-Host "    Iterations with tests: $($withTests.Count)" -ForegroundColor White
                        $testColor = if ($passOnly -eq $withTests.Count) { 'Green' } else { 'Yellow' }
                        Write-Host "    All tests passing: $passOnly/$($withTests.Count)" -ForegroundColor $testColor

                        # Show last test result
                        $lastTest = $withTests | Select-Object -Last 1
                        $truncatedResult = if ($lastTest.test_results.Length -gt 50) { $lastTest.test_results.Substring(0, 47) + "..." } else { $lastTest.test_results }
                        Write-Host "    Latest: $truncatedResult" -ForegroundColor Gray
                        Write-Host ""
                    }
                }

                # Productivity (Lines Changed)
                if ($hasLines) {
                    $withLines = @($sessionMetrics | Where-Object { $_.lines_added -or $_.lines_deleted })
                    if ($withLines.Count -gt 0) {
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
                }

                # Retry Analysis
                if ($hasRetries) {
                    $withRetries = @($sessionMetrics | Where-Object { $_.retry_count -and $_.retry_count -ne '' })
                    if ($withRetries.Count -gt 0) {
                        $retryValues = $withRetries | ForEach-Object { [int]$_.retry_count }
                        $maxRetries = ($retryValues | Measure-Object -Maximum).Maximum
                        $avgRetries = [math]::Round(($retryValues | Measure-Object -Average).Average, 1)

                        Write-Host "  RESILIENCE:" -ForegroundColor Cyan
                        $avgColor = if ($avgRetries -le 1.5) { 'Green' } elseif ($avgRetries -le 3) { 'Yellow' } else { 'Red' }
                        Write-Host "    Avg retries: $avgRetries" -ForegroundColor $avgColor
                        $maxColor = if ($maxRetries -le 3) { 'Green' } else { 'Yellow' }
                        Write-Host "    Max retries: $maxRetries" -ForegroundColor $maxColor
                        Write-Host ""
                    }
                }

                # Hour of Day Pattern (mini histogram)
                if ($hasHour) {
                    $withHour = @($sessionMetrics | Where-Object { $_.hour_of_day -and $_.hour_of_day -ne '' })
                    if ($withHour.Count -gt 0) {
                        Write-Host "  TIME PATTERN:" -ForegroundColor Cyan
                        $hourGroups = $withHour | Group-Object hour_of_day | Sort-Object { [int]$_.Name }
                        $maxHourCount = ($hourGroups | Measure-Object -Property Count -Maximum).Maximum
                        foreach ($hg in $hourGroups) {
                            $barLength = if ($maxHourCount -gt 0) { [math]::Round(($hg.Count / $maxHourCount) * 20) } else { 0 }
                            $bar = [string]::new([char]0x2588, $barLength)  # Unicode block character
                            $hour = $hg.Name.PadLeft(2, '0')
                            Write-Host "    ${hour}h: $bar $($hg.Count)" -ForegroundColor Gray
                        }
                        Write-Host ""
                    }
                }

                # Duration Stats (always available in current schema)
                $withDuration = @($sessionMetrics | Where-Object { $_.duration_min -and $_.duration_min -ne '' })
                if ($withDuration.Count -gt 0) {
                    $durations = $withDuration | ForEach-Object { [double]$_.duration_min }
                    $totalDuration = ($durations | Measure-Object -Sum).Sum
                    $avgDuration = [math]::Round(($durations | Measure-Object -Average).Average, 1)
                    $maxDuration = ($durations | Measure-Object -Maximum).Maximum

                    Write-Host "  TIMING:" -ForegroundColor Cyan
                    Write-Host "    Total time: $([math]::Round($totalDuration, 0)) min" -ForegroundColor White
                    Write-Host "    Avg/story: $avgDuration min" -ForegroundColor White
                    Write-Host "    Longest: $maxDuration min" -ForegroundColor $(if ($maxDuration -le 15) { 'Green' } elseif ($maxDuration -le 30) { 'Yellow' } else { 'Red' })
                }

                # Focus Area Summary (always available)
                $focusGroups = $sessionMetrics | Group-Object focus_area | Sort-Object Count -Descending
                if ($focusGroups.Count -gt 0) {
                    Write-Host ""
                    Write-Host "  FOCUS AREAS:" -ForegroundColor Cyan
                    foreach ($fg in $focusGroups | Select-Object -First 5) {
                        $areaSuccesses = @($fg.Group | Where-Object { $_.success -eq 'true' }).Count
                        $areaTotal = $fg.Group.Count
                        $areaRate = [math]::Round(($areaSuccesses / $areaTotal) * 100)
                        $areaColor = if ($areaRate -ge 80) { 'Green' } elseif ($areaRate -ge 50) { 'Yellow' } else { 'Red' }
                        Write-Host "    $($fg.Name): $areaRate% ($areaSuccesses/$areaTotal)" -ForegroundColor $areaColor
                    }
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

    # Recent progress.txt
    if (Test-Path $ProgressPath) {
        Write-Host ""
        Write-Host "  Recent activity:" -ForegroundColor DarkGray
        Get-Content $ProgressPath | Select-Object -Last 5 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor DarkGray
        }
    }

    # Latest log file activity
    if (Test-Path $LogsDir) {
        $latestLogDir = Get-ChildItem $LogsDir -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latestLogDir) {
            $latestLog = Get-ChildItem $latestLogDir.FullName -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestLog) {
                $age = [math]::Round(((Get-Date) - $latestLog.LastWriteTime).TotalMinutes, 1)
                Write-Host ""
                Write-Host "  Latest log: $($latestLog.Name) ($age min ago)" -ForegroundColor DarkGray

                # Show last few lines
                $lastLines = Get-Content $latestLog.FullName -Tail 3 2>$null
                if ($lastLines) {
                    $lastLines | ForEach-Object {
                        $truncated = if ($_.Length -gt 70) { $_.Substring(0, 67) + "..." } else { $_ }
                        Write-Host "    $truncated" -ForegroundColor DarkGray
                    }
                }
            }
        }
    }

    Write-Host ""
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Git status:" -ForegroundColor DarkGray
    $gitStatus = git status --short 2>$null | Select-Object -First 5
    if ($gitStatus) {
        $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    } else {
        Write-Host "    (clean)" -ForegroundColor DarkGray
    }

    # Cleanup
    if ($loopCount % 50 -eq 0) {
        [System.GC]::Collect()
    }

    Start-Sleep -Seconds $Interval
}
