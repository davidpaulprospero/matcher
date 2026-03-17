$projectPath = 'E:\Edit Job\test'
$checkScript = 'D:\_Projects\voiceover-matcher-subtitle\scripts\watch_check.ps1'
$statusMarker = "$projectPath\watch_status.txt"

# Run checks in a loop, exiting when pipeline stops
$iteration = 0
while ($true) {
    $iteration++

    # Run check script
    $output = & $checkScript 2>&1 | Out-String

    # Write iteration marker
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$iteration|$timestamp" | Set-Content -Path $statusMarker

    # Check for pipeline stop
    if ($output -match "PIPELINE_STOPPED") {
        Write-Host "WATCH_EXIT_PIPELINE_DONE"
        break
    }

    # Check if we should continue (log still updating)
    $logFile = Get-ChildItem "$projectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($logFile) {
        $lastWrite = $logFile.LastWriteTime
        $timeSinceLastWrite = (Get-Date) - $lastWrite
        if ($timeSinceLastWrite.TotalMinutes -gt 10) {
            Write-Host "WATCH_EXIT_NO_ACTIVITY"
            break
        }
    }

    # Wait 5 minutes
    Start-Sleep -Seconds 300
}

Write-Host "WATCH_ITERATION=$iteration"
