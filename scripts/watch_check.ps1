$projectPath = 'E:\Edit Job\test'
$interval = 5 * 60  # 5 minutes in seconds

# Single iteration - check status and write PIPELINE_STATUS.md

# Check if pipeline is still running
$logFile = Get-ChildItem "$projectPath\logs\run_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($logFile) {
    $lastWrite = $logFile.LastWriteTime
    $timeSinceLastWrite = (Get-Date) - $lastWrite
    if ($timeSinceLastWrite.TotalMinutes -gt 10) {
        Write-Host "PIPELINE_STOPPED"
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

Run /watch to check again.
"@

Set-Content -Path $statusFile -Value $content

Write-Host "STREAM_STATES=$streamStates"
Write-Host "CAPTIONS_FETCHED=$captionsFetched"
Write-Host "LAST_WRITE=$($lastWrite.ToString('HH:mm:ss'))"
