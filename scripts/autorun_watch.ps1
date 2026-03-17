# Autorun Watch script - monitors degold_autorun.py
# Started as Claude Code background task
# Monitors autorun health and auto-restarts on exit or failure

param(
    [Parameter(Mandatory=$false)]
    [int]$Duration = 5,  # Total duration in minutes before exit

    [Parameter(Mandatory=$false)]
    [int]$CheckInterval = 30  # Check every 30 seconds
)

$logFile = "D:\_Projects\voiceover-matcher-stable\logs\degold_autorun.log"
$stateFile = "D:\_Projects\voiceover-matcher-stable\Degold\degold_autorun_state.json"
$lockFile = "D:\_Projects\voiceover-matcher-stable\Degold\degold_autorun.lock"
$queueStopFile = "D:\_Projects\voiceover-matcher-stable\Degold\queue_stop.txt"
$queueStateFile = "D:\_Projects\voiceover-matcher-stable\Degold\pipeline_queue_state.json"

$iteration = 0
$startTime = Get-Date
$prevRunningCard = ""
$prevPipelineLogFile = ""

Write-Host "[AutorunWatch] Starting: duration=${Duration}min interval=${CheckInterval}s"

# Wait a bit for autorun to start up
Start-Sleep -Seconds 5

while ($true) {
    $iteration++

    # Check for queue stop file
    if (Test-Path $queueStopFile) {
        $stopContent = (Get-Content $queueStopFile -Raw -ErrorAction SilentlyContinue).Trim().ToLower()
        if ($stopContent -eq "true" -or $stopContent -eq "1" -or $stopContent -eq "stop") {
            Write-Host "AUTORUN_WATCH_EXIT_QUEUE_STOP iteration=$iteration"
            break
        }
    }

    # Check if duration has expired
    $elapsed = (Get-Date) - $startTime
    if ($elapsed.TotalMinutes -ge $Duration) {
        Write-Host "AUTORUN_WATCH_EXIT_DURATION_EXPIRED iteration=$iteration elapsed=$([math]::Round($elapsed.TotalMinutes,1))min"
        break
    }

    # Check if autorun is running
    $autorunRunning = $false
    $status = "unknown"
    $lockPid = ""

    # Check lock file
    if (Test-Path $lockFile) {
        try {
            $lockContent = Get-Content $lockFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue | ConvertFrom-Json
            $lockPid = $lockContent.pid
            $heartbeatAt = $lockContent.heartbeat_at

            # Check if process is running
            $process = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
            if ($process) {
                $autorunRunning = $true
                $status = "running"
            } else {
                $status = "dead_pid"
            }

            # Check heartbeat staleness
            if ($heartbeatAt -and $autorunRunning) {
                $heartbeatTime = [DateTime]::Parse($heartbeatAt)
                $timeSinceHeartbeat = (Get-Date) - $heartbeatTime
                $staleSeconds = 90
                if ($timeSinceHeartbeat.TotalSeconds -gt $staleSeconds) {
                    $status = "stale_heartbeat"
                    $autorunRunning = $false
                }
            }
        } catch {
            $status = "lock_error"
        }
    } else {
        $status = "no_lock"
    }

    # Check log file for recent activity
    $logActivity = "none"
    if (Test-Path $logFile) {
        $lastLogTime = (Get-Item $logFile).LastWriteTime
        $timeSinceLog = (Get-Date) - $lastLogTime

        if ($timeSinceLog.TotalMinutes -lt 1) {
            $logActivity = "active"
        } elseif ($timeSinceLog.TotalMinutes -lt 5) {
            $logActivity = "recent"
        } else {
            $logActivity = "stale"
        }
    }

    # Get state info if available
    $targetCards = @()
    $currentStep = ""
    $lastLaunched = ""
    $launchOutcome = ""
    $pipelineStage = ""
    $suppressedCards = @()
    $consecutiveFailures = 0
    $uptime = ""
    $runningCard = ""
    if (Test-Path $stateFile) {
        try {
            $stateContent = Get-Content $stateFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($stateContent.runtime) {
                $targetCards = @($stateContent.runtime.target_card_ids)
                $currentStep = $stateContent.runtime.current_step
                if ($stateContent.runtime.started_at) {
                    try {
                        $runtimeStart = [DateTime]::Parse($stateContent.runtime.started_at)
                        $runtimeElapsed = (Get-Date) - $runtimeStart
                        $uptime = "$([math]::Floor($runtimeElapsed.TotalHours))h$([math]::Floor($runtimeElapsed.Minutes))m"
                    } catch {}
                }
            }
            if ($stateContent.last_cycle) {
                $lastLaunched = $stateContent.last_cycle.launched_card_id
                $launchOutcome = $stateContent.last_cycle.launch_outcome
                $pipelineStage = $stateContent.last_cycle.pipeline_stage
                $consecutiveFailures = $stateContent.last_cycle.consecutive_failures
            }
            if ($stateContent.suppressed_card_ids) {
                $suppressedCards = @($stateContent.suppressed_card_ids)
            }
        } catch {
            # Ignore
        }
    }

    # Get queue counts
    $readyCount = 0
    $runningCount = 0
    $completedCount = 0
    if (Test-Path $queueStateFile) {
        try {
            $queueContent = Get-Content $queueStateFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue | ConvertFrom-Json
            $queueData = $queueContent.queue
            if ($queueData) {
                $readyCount = @($queueData.ready_card_ids).Count
                $completedCount = @($queueData.completed_card_ids).Count
            }
            # Check running pipelines from runtime_summary (authoritative)
            if ($queueContent.runtime_summary) {
                $runningCount = [int]($queueContent.runtime_summary.running_count)
                $runningIds = @($queueContent.runtime_summary.running_card_ids)
                if ($runningIds.Count -gt 0) {
                    $runningCard = $runningIds -join ','
                }
            }
            # Fallback: check individual pipeline entries
            if ($runningCount -eq 0 -and $queueContent.pipelines) {
                foreach ($prop in $queueContent.pipelines.PSObject.Properties) {
                    $pipeline = $prop.Value
                    if ($pipeline.pipeline_runtime_state -eq "running") {
                        $runningCount++
                        $runningCard = $prop.Value.card_id
                    }
                }
            }
        } catch {}
    }

    # Fallback: if queue says 0 running but we had a pipeline last iteration,
    # check if its log is still being written to (queue state file can be stale during autorun sync)
    if ($runningCount -eq 0 -and $prevRunningCard -and $prevPipelineLogFile) {
        if (Test-Path $prevPipelineLogFile) {
            $prevLogAge = ((Get-Date) - (Get-Item $prevPipelineLogFile).LastWriteTime).TotalSeconds
            if ($prevLogAge -lt 120) {
                # Log still active — pipeline is still running, queue state is stale
                $runningCard = $prevRunningCard
                $runningCount = 1
            }
        }
    }

    # Read checkpoint stage and find pipeline log for the running pipeline
    $checkpointStage = ""
    $pipelineLogFile = ""
    $pipelineProjDir = ""
    if ($runningCard -and $runningCard -notmatch ',') {
        # Find project dir from queue state
        if ($queueContent.pipelines) {
            foreach ($prop in $queueContent.pipelines.PSObject.Properties) {
                $pl = $prop.Value
                if ($pl.card_id -eq $runningCard -and $pl.project -and $pl.project.local_project_dirs) {
                    $pipelineProjDir = @($pl.project.local_project_dirs)[0]
                    $cpFile = Join-Path $pipelineProjDir "checkpoint.json"
                    if (Test-Path $cpFile) {
                        try {
                            $cpBytes = [System.IO.File]::ReadAllBytes($cpFile)
                            # Check for gzip (0x1f 0x8b)
                            if ($cpBytes.Length -ge 2 -and $cpBytes[0] -eq 0x1f -and $cpBytes[1] -eq 0x8b) {
                                $ms = New-Object System.IO.MemoryStream(,$cpBytes)
                                $gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Decompress)
                                $sr = New-Object System.IO.StreamReader($gz)
                                $cpJson = $sr.ReadToEnd() | ConvertFrom-Json
                                $sr.Close(); $gz.Close(); $ms.Close()
                            } else {
                                $cpJson = [System.Text.Encoding]::UTF8.GetString($cpBytes) | ConvertFrom-Json
                            }
                            $checkpointStage = $cpJson.last_completed_stage
                        } catch {}
                    }
                    # Find most recent verbose log
                    $logsDir = Join-Path $pipelineProjDir "logs"
                    if (Test-Path $logsDir) {
                        $latestLog = Get-ChildItem $logsDir -Filter "run_*_verbose.log" -ErrorAction SilentlyContinue |
                            Sort-Object LastWriteTime -Descending | Select-Object -First 1
                        if ($latestLog) {
                            $pipelineLogFile = $latestLog.FullName
                        }
                    }
                    break
                }
            }
        }
    }

    # Derive current stage from last_completed_stage
    $stageOrder = @("ANALYZE", "STOCK_FOOTAGE", "VIDEO_SEARCH", "CAPTION", "MATCH", "DOWNLOAD_SEGMENTS", "OUTPUT")
    $currentStage = ""
    if ($checkpointStage) {
        $idx = [array]::IndexOf($stageOrder, $checkpointStage)
        if ($idx -ge 0 -and $idx -lt ($stageOrder.Count - 1)) {
            $currentStage = $stageOrder[$idx + 1]
        } elseif ($idx -eq ($stageOrder.Count - 1)) {
            $currentStage = "COMPLETED"
        }
    }

    # Extract progress from log tail based on current stage
    $stageProgress = ""
    if ($pipelineLogFile -and (Test-Path $pipelineLogFile) -and $currentStage) {
        try {
            $fs = [System.IO.File]::Open($pipelineLogFile, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $progressTailBytes = 65536
            if ($fs.Length -gt $progressTailBytes) {
                $fs.Seek(-$progressTailBytes, [System.IO.SeekOrigin]::End) | Out-Null
            }
            $psr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
            if ($fs.Length -gt $progressTailBytes) { $psr.ReadLine() | Out-Null }
            $progressLines = @()
            while ($null -ne ($pline = $psr.ReadLine())) { $progressLines += $pline }
            $psr.Close(); $fs.Close()

            switch ($currentStage) {
                "DOWNLOAD_SEGMENTS" {
                    # Look for: [DOWNLOAD_SEGMENTS] Progress: 35% (176/493)
                    for ($pi = $progressLines.Count - 1; $pi -ge 0; $pi--) {
                        if ($progressLines[$pi] -match '\[DOWNLOAD_SEGMENTS\] Progress: (\d+)% \((\d+)/(\d+)\).*ok=(\d+), failed=(\d+), cached=(\d+)') {
                            $stageProgress = "$($Matches[1])% ($($Matches[2])/$($Matches[3])) ok=$($Matches[4]) fail=$($Matches[5]) cached=$($Matches[6])"
                            break
                        } elseif ($progressLines[$pi] -match '\[DOWNLOAD_SEGMENTS\] Progress: (\d+)% \((\d+)/(\d+)\)') {
                            $stageProgress = "$($Matches[1])% ($($Matches[2])/$($Matches[3]))"
                            break
                        }
                    }
                }
                "VIDEO_SEARCH" {
                    # Look for: [N/M] Searching:
                    for ($pi = $progressLines.Count - 1; $pi -ge 0; $pi--) {
                        if ($progressLines[$pi] -match '\[(\d+)/(\d+)\] Searching: (.+)') {
                            $stageProgress = "$($Matches[1])/$($Matches[2]) keywords"
                            break
                        }
                    }
                }
                "CAPTION" {
                    # Get total from non-verbose log (smaller, has Stage started line)
                    $captionTotal = 0
                    $captionDone = 0
                    $nonVerboseLog = $pipelineLogFile -replace '_verbose\.log$', '.log'
                    if ($nonVerboseLog -ne $pipelineLogFile -and (Test-Path $nonVerboseLog)) {
                        try {
                            $nvFs = [System.IO.File]::Open($nonVerboseLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
                            $nvSr = New-Object System.IO.StreamReader($nvFs, [System.Text.Encoding]::UTF8)
                            while ($null -ne ($nvLine = $nvSr.ReadLine())) {
                                if ($nvLine -match '\[CAPTION\] Stage started - video_count=(\d+)') {
                                    $captionTotal = [int]$Matches[1]
                                }
                                if ($nvLine -match 'Stream state:') {
                                    $captionDone++
                                }
                            }
                            $nvSr.Close(); $nvFs.Close()
                        } catch {}
                    }
                    # Fallback: count from verbose tail
                    if ($captionTotal -eq 0) {
                        foreach ($pline in $progressLines) {
                            if ($pline -match '\[CAPTION\] Stage started - video_count=(\d+)') {
                                $captionTotal = [int]$Matches[1]
                            }
                            if ($pline -match 'Stream state:') {
                                $captionDone++
                            }
                        }
                    }
                    if ($captionTotal -gt 0) {
                        $captionPct = [math]::Round(($captionDone / $captionTotal) * 100)
                        $stageProgress = "${captionPct}% (~${captionDone}/${captionTotal} checked)"
                    } elseif ($captionDone -gt 0) {
                        $stageProgress = "${captionDone} checked"
                    }
                }
                "MATCH" {
                    # Look for: VOICEOVER lines (each is one match attempt)
                    $matchCount = 0
                    foreach ($pline in $progressLines) {
                        if ($pline -match 'VOICEOVER:') { $matchCount++ }
                    }
                    if ($matchCount -gt 0) {
                        $stageProgress = "${matchCount} segments matched (tail)"
                    }
                }
                "OUTPUT" {
                    $stageProgress = "generating timeline"
                }
            }
        } catch {}
    }

    # Build output line with key details
    $details = "[AutorunWatch] iteration=$iteration status=$status log=$logActivity pid=$lockPid"
    $details += " uptime=$uptime"
    $details += " queue=${readyCount}ready/${runningCount}running/${completedCount}done"

    if ($runningCard) {
        $details += " active_pipeline=$runningCard"
    }
    if ($currentStage) {
        $details += " stage=$currentStage"
    } elseif ($checkpointStage) {
        $details += " stage=$checkpointStage"
    } elseif ($pipelineStage) {
        $details += " stage=$pipelineStage"
    }
    if ($stageProgress) {
        $details += " progress=[$stageProgress]"
    }
    if ($lastLaunched -and $launchOutcome) {
        $details += " last=${lastLaunched}:${launchOutcome}"
    }
    if ($suppressedCards.Count -gt 0) {
        $details += " suppressed=$($suppressedCards -join ',')"
    }
    if ($consecutiveFailures -gt 0) {
        $details += " failures=$consecutiveFailures"
    }

    Write-Host $details

    # Show pipeline log tail and recent errors for the running pipeline
    if ($pipelineLogFile -and (Test-Path $pipelineLogFile)) {
        try {
            $logItem = Get-Item $pipelineLogFile
            $logAge = ((Get-Date) - $logItem.LastWriteTime).TotalSeconds
            $logSizeKB = [math]::Round($logItem.Length / 1024)
            Write-Host "  --- Pipeline log: $($logItem.Name) (${logSizeKB}KB, ${logAge}s ago) ---"

            # Read only the tail of the file with shared access (seek to last ~32KB)
            $fs = [System.IO.File]::Open($pipelineLogFile, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $tailBytes = 32768
            if ($fs.Length -gt $tailBytes) {
                $fs.Seek(-$tailBytes, [System.IO.SeekOrigin]::End) | Out-Null
            }
            $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
            # Skip partial first line if we seeked
            if ($fs.Length -gt $tailBytes) { $sr.ReadLine() | Out-Null }
            $tailLines = @()
            while ($null -ne ($line = $sr.ReadLine())) { $tailLines += $line }
            $sr.Close(); $fs.Close()

            # Show last 15 lines
            $showCount = [math]::Min(15, $tailLines.Count)
            $last15 = $tailLines[($tailLines.Count - $showCount)..($tailLines.Count - 1)]
            foreach ($line in $last15) {
                $trimmed = if ($line.Length -gt 180) { $line.Substring(0, 180) + "..." } else { $line }
                Write-Host "  | $trimmed"
            }

            # Scan tail for recent errors (show unique error/warning lines)
            $scanLines = $tailLines
            $errors = @()
            foreach ($line in $scanLines) {
                if ($line -match " - ERROR - " -or $line -match " - CRITICAL - " -or $line -match "Traceback" -or $line -match "Exception:" -or $line -match "HTTP 429|HTTP 403|HTTP 500") {
                    $trimmed = if ($line.Length -gt 160) { $line.Substring(0, 160) + "..." } else { $line }
                    if ($errors -notcontains $trimmed) {
                        $errors += $trimmed
                    }
                }
            }
            if ($errors.Count -gt 0) {
                Write-Host "  --- Recent errors ($($errors.Count)) ---"
                # Show up to 5 unique errors
                $showCount = [math]::Min(5, $errors.Count)
                for ($ei = 0; $ei -lt $showCount; $ei++) {
                    Write-Host "  ! $($errors[$ei])"
                }
                if ($errors.Count -gt 5) {
                    Write-Host "  ! ... and $($errors.Count - 5) more"
                }
            }
        } catch {
            Write-Host "  [warn] Could not read pipeline log: $_"
        }
    }

    # If autorun is not running but we expect it to be, try to restart it
    if (-not $autorunRunning -and $status -ne "no_lock") {
        Write-Host "AUTORUN_WATCH_EXIT_AUTORUN_STOPPED iteration=$iteration status=$status"
        # Don't exit - let the skill restart autorun
        break
    }

    # If log is stale but autorun is running, that's concerning
    if ($autorunRunning -and $logActivity -eq "stale") {
        Write-Host "AUTORUN_WATCH_LOG_STALE iteration=$iteration log=$logActivity failures=$consecutiveFailures"
        # Don't exit yet - give it more time
    }

    # Remember running pipeline for next iteration (handles stale queue state during sync)
    if ($runningCard) {
        $prevRunningCard = $runningCard
        if ($pipelineLogFile) { $prevPipelineLogFile = $pipelineLogFile }
    }

    Start-Sleep -Seconds $CheckInterval
}

Write-Host "AUTORUN_WATCH_COMPLETE iteration=$iteration"
