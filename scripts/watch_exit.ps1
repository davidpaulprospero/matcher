$projectPath = 'E:\Edit Job\test'

# Single check with 5-min sleep, then exit

# Check if pipeline is still running
$logFile = Get-ChildItem "$projectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($logFile) {
    $lastWrite = $logFile.LastWriteTime
    $timeSinceLastWrite = (Get-Date) - $lastWrite
    if ($timeSinceLastWrite.TotalMinutes -gt 10) {
        Write-Host "WATCH_EXIT_PIPELINE_DONE"
        exit 0
    }
}

# Write status
$statusFile = "$projectPath\PIPELINE_STATUS.md"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Read latest log
$logContent = ""
$latestLog = Get-ChildItem "$projectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
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

$content = @"
# Pipeline Status

**Last Updated:** $timestamp

## Progress

| Metric | Count |
|--------|-------|
| Stream States Checked | $streamStates |
| Captions Fetched | $captionsFetched |
| Last Activity | $($lastWrite.ToString('HH:mm:ss')) |

## Recent Activity (last 30 log lines)

```
$logContent
```

## Next Check

Monitoring will update in 5 minutes.
"@

Set-Content -Path $statusFile -Value $content

Write-Host "WATCH_CHECK_COMPLETE streamStates=$streamStates captions=$captionsFetched lastActivity=$($lastWrite.ToString('HH:mm:ss'))"

# Sleep 5 minutes then exit (this is the key part - task will complete and exit)
Start-Sleep -Seconds 300

# After sleep, exit - Claude Code will see this exit and can re-run
Write-Host "WATCH_SLEEP_COMPLETE_EXITING"
exit 0
