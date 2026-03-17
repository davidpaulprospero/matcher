# Cleanup old Ralph logs
# Moves logs older than 7 days to archive/logs/
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1

param(
    [int]$DaysToKeep = 7,
    [switch]$WhatIf
)

$logsDir = Join-Path $PSScriptRoot "logs"
$archiveDir = Join-Path $PSScriptRoot "archive\logs"

# Create archive if needed
if (-not (Test-Path $archiveDir)) {
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
}

$cutoffDate = (Get-Date).AddDays(-$DaysToKeep)
$logDirs = Get-ChildItem -Path $logsDir -Directory | Where-Object {
    # Parse date from folder name (format: 2026-01-24_145800)
    if ($_.Name -match '^(\d{4}-\d{2}-\d{2})_') {
        $folderDate = [datetime]::ParseExact($matches[1], 'yyyy-MM-dd', $null)
        return $folderDate -lt $cutoffDate
    }
    return $false
}

$count = ($logDirs | Measure-Object).Count
Write-Host "Found $count log directories older than $DaysToKeep days"

foreach ($dir in $logDirs) {
    $dest = Join-Path $archiveDir $dir.Name
    if ($WhatIf) {
        Write-Host "  Would move: $($dir.Name)" -ForegroundColor Yellow
    } else {
        Move-Item -Path $dir.FullName -Destination $dest -Force
        Write-Host "  Archived: $($dir.Name)" -ForegroundColor Green
    }
}

Write-Host "`nDone. Use -WhatIf to preview without moving."
