# Watch script - runs with duration-based auto-restart loop
# Started as Claude Code background task
# Loops: check pipeline -> sleep interval -> repeat until duration expires
# Exits when duration expires OR pipeline completes OR error
# Claude Code will detect exit and auto-restart

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectPath = 'E:\Edit Job\test',

    [Parameter(Mandatory=$false)]
    [int]$Duration = 5  # Total duration in minutes before exit

)

$statusMarker = "$ProjectPath\watch_status.txt"
$runningMarker = "$ProjectPath\watch_running.txt"

# Write running marker
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"running|$timestamp" | Set-Content -Path $runningMarker

$iteration = 0
$startTime = Get-Date
$checkInterval = 60  # Check every 60 seconds
$stallThreshold = 10  # Minutes of inactivity before considering stalled

Write-Host "[Watch] Starting: project=$ProjectPath duration=${Duration}min interval=${checkInterval}s"

while ($true) {
    $iteration++

    # Check if duration has expired
    $elapsed = (Get-Date) - $startTime
    if ($elapsed.TotalMinutes -ge $Duration) {
        Write-Host "WATCH_EXIT_DURATION_EXPIRED iteration=$iteration elapsed=$([math]::Round($elapsed.TotalMinutes,1))min duration=${Duration}min"
        break
    }

    # Check if pipeline is still running by checking log activity
    $logFile = Get-ChildItem "$ProjectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $pipelineRunning = $true

    if ($logFile) {
        $lastWrite = $logFile.LastWriteTime
        $timeSinceLastWrite = (Get-Date) - $lastWrite

        # If no log activity for stallThreshold+ minutes, check if pipeline actually finished
        if ($timeSinceLastWrite.TotalMinutes -gt $stallThreshold) {
            # Check checkpoint for completion (try multiple backup files)
            $checkpointFound = $false
            $lastStage = "UNKNOWN"

            foreach ($cpFile in @("$ProjectPath\checkpoint.json", "$ProjectPath\checkpoint.backup.json", "$ProjectPath\checkpoint.backup.1.json")) {
                if (Test-Path $cpFile) {
                    try {
                        # Try to read as gzipped first, then as plain JSON
                        $cpContent = ""
                        try {
                            # Try gzipped
                            $cpBytes = [System.IO.File]::ReadAllBytes($cpFile)
                            $cpMemStream = New-Object System.IO.MemoryStream(, $cpBytes)
                            $cpGzipStream = New-Object System.IO.Compression.GZipStream($cpMemStream, [System.IO.Compression.CompressionMode]::Decompress)
                            $cpReader = New-Object System.IO.StreamReader($cpGzipStream)
                            $cpContent = $cpReader.ReadToEnd()
                            $cpReader.Close()
                            $cpGzipStream.Close()
                            $cpMemStream.Close()
                        } catch {
                            # Fall back to plain text
                            $cpContent = Get-Content $cpFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
                        }

                        if ($cpContent) {
                            $checkpoint = $cpContent | ConvertFrom-Json
                            $lastStage = $checkpoint.last_completed_stage
                            $checkpointFound = $true
                            break
                        }
                    } catch {
                        # Try next file
                    }
                }
            }

            if ($checkpointFound) {
                # Check if OUTPUT stage completed (pipeline finished)
                if ($lastStage -eq "OUTPUT") {
                    Write-Host "WATCH_EXIT_PIPELINE_COMPLETE iteration=$iteration stage=$lastStage"
                    $pipelineRunning = $false
                } else {
                    Write-Host "WATCH_EXIT_PIPELINE_STALLED iteration=$iteration lastStage=$lastStage timeSinceWrite=$([math]::Round($timeSinceLastWrite.TotalMinutes, 1))min"
                    $pipelineRunning = $false
                }
            } else {
                Write-Host "WATCH_EXIT_NO_LOG_ACTIVITY iteration=$iteration timeSinceWrite=$([math]::Round($timeSinceLastWrite.TotalMinutes, 1))min"
                $pipelineRunning = $false
            }

            if (-not $pipelineRunning) {
                break
            }
        }
    }

    # Write status
    $statusFile = "$ProjectPath\PIPELINE_STATUS.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # Read checkpoint for stage info (handle gzipped files)
    $stage = "UNKNOWN"
    $videoCount = "N/A"
    $checkpointPath = "$ProjectPath\checkpoint.json"

    # Try multiple checkpoint files
    foreach ($cpFile in @("$ProjectPath\checkpoint.json", "$ProjectPath\checkpoint.backup.json", "$ProjectPath\checkpoint.backup.1.json")) {
        if (Test-Path $cpFile) {
            try {
                $cpContent = ""
                try {
                    # Try gzipped first
                    $cpBytes = [System.IO.File]::ReadAllBytes($cpFile)
                    $cpMemStream = New-Object System.IO.MemoryStream(, $cpBytes)
                    $cpGzipStream = New-Object System.IO.Compression.GZipStream($cpMemStream, [System.IO.Compression.CompressionMode]::Decompress)
                    $cpReader = New-Object System.IO.StreamReader($cpGzipStream)
                    $cpContent = $cpReader.ReadToEnd()
                    $cpReader.Close()
                    $cpGzipStream.Close()
                    $cpMemStream.Close()
                } catch {
                    # Fall back to plain text
                    $cpContent = Get-Content $cpFile -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
                }

                if ($cpContent) {
                    $checkpoint = $cpContent | ConvertFrom-Json
                    $stage = $checkpoint.last_completed_stage
                    if ($checkpoint.stages -and $checkpoint.stages.VIDEO_SEARCH) {
                        $videoCount = $checkpoint.stages.VIDEO_SEARCH.video_count
                    }
                    break
                }
            } catch { }
        }
    }

    # Read latest log
    $logContent = ""
    $latestLog = Get-ChildItem "$ProjectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        $lastLines = Get-Content $latestLog.FullName -Tail 30
        $logContent = $lastLines -join "`n"
    }

    # Count progress
    $streamStates = 0
    $captionsFetched = 0
    if ($latestLog) {
        $streamStates = (Select-String -Path $latestLog.FullName -Pattern "Stream state:" -AllMatches).Matches.Count
        $captionsFetched = (Select-String -Path $latestLog.FullName -Pattern "captions fetched" -AllMatches).Matches.Count
    }

    # Use the log file's actual LastWriteTime, not timestamps from log content
    $lastActivity = if ($logFile) { $logFile.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') } else { 'N/A' }
    $timeSinceLastWriteStr = if ($logFile) { "$([math]::Round($timeSinceLastWrite.TotalMinutes, 1)) min" } else { 'N/A' }

    $elapsedStr = "$([math]::Round($elapsed.TotalMinutes, 1)) min"
    $remaining = [math]::Max(0, $Duration - $elapsed.TotalMinutes)
    $remainingStr = "$([math]::Round($remaining, 1)) min"

    $content = @"
# Pipeline Status

**Last Updated:** $timestamp
**Iteration:** $iteration
**Current Stage:** $stage
**Elapsed:** $elapsedStr / ${Duration}min (remaining: $remainingStr)

## Progress

| Metric | Count |
|--------|-------|
| Stage | $stage |
| Video Candidates | $videoCount |
| Stream States Checked | $streamStates |
| Captions Fetched | $captionsFetched |
| Last Activity | $lastActivity |
| Time Since Last Write | $timeSinceLastWriteStr |

## Recent Activity (last 30 log lines)

```
$logContent
```

## Status

Background watch active - checks every ${checkInterval}s, duration ${Duration}min.
"@

    Set-Content -Path $statusFile -Value $content

    # Update running marker
    "$iteration|$timestamp" | Set-Content -Path $statusMarker

    Write-Host "[Watch $iteration] stage=$stage streamStates=$streamStates captions=$captionsFetched lastActivity=$lastActivity"

    # Sleep check interval
    Start-Sleep -Seconds $checkInterval
}

Write-Host "WATCH_LOOP_EXITED iteration=$iteration"
Remove-Item -Path $runningMarker -ErrorAction SilentlyContinue
