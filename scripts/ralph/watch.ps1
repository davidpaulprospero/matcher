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

    # Metrics summary
    if (Test-Path $MetricsPath) {
        try {
            $metrics = Import-Csv $MetricsPath
            # Get latest session
            $sessions = $metrics | Group-Object session | Sort-Object { $_.Group[0].timestamp } -Descending
            if ($sessions.Count -gt 0) {
                $latestSession = $sessions[0].Group
                $successes = @($latestSession | Where-Object { $_.success -eq 'true' }).Count
                $timeouts = @($latestSession | Where-Object { $_.timeout -eq 'true' }).Count
                $totalMetrics = $latestSession.Count

                Write-Host ""
                Write-Host "  Session: $($sessions[0].Name)" -ForegroundColor DarkGray
                Write-Host "  Stats: $successes/$totalMetrics successful, $timeouts timeouts" -ForegroundColor DarkGray
            }
        } catch {}
    }

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
