# Ralph Loop Status Check
# Usage: .\scripts\ralph\status.ps1
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateDir = Join-Path $ScriptDir "state"
$SessionDir = Join-Path $ScriptDir "session"
$PrdPath = Join-Path $StateDir "prd.json"
$MetricsPath = Join-Path $SessionDir "metrics.csv"
$BlockedPath = Join-Path $ScriptDir "BLOCKED.md"

if (-not (Test-Path $PrdPath)) {
    Write-Host "No PRD file found. Run ralph.ps1 to create a sprint." -ForegroundColor Yellow
    exit 0
}

$prd = Get-Content $PrdPath -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  RALPH LOOP STATUS - voiceover-matcher-subtitle" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
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
        $status = "[DONE]"
        $color = "Green"
    } else {
        $failed++
        $status = "[    ]"
        $color = "Yellow"
        if ($null -eq $nextStory) { $nextStory = $story }
    }

    Write-Host "  $status $($story.id): $($story.title)" -ForegroundColor $color
    if ($story.notes -and $story.notes.Length -gt 0) {
        $truncated = if ($story.notes.Length -gt 70) { $story.notes.Substring(0, 67) + "..." } else { $story.notes }
        Write-Host "         $truncated" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan

$total = $passed + $failed
$pct = if ($total -gt 0) { [math]::Round(($passed / $total) * 100) } else { 0 }

if ($failed -eq 0) {
    Write-Host "  PROGRESS: $passed/$total complete ($pct%)" -ForegroundColor Green
    Write-Host "  STATUS: All stories complete!" -ForegroundColor Green
} else {
    Write-Host "  PROGRESS: $passed/$total complete ($pct%)" -ForegroundColor Yellow
    Write-Host "  REMAINING: $failed stories" -ForegroundColor Yellow
}

if ($nextStory) {
    Write-Host ""
    Write-Host "  NEXT: $($nextStory.id) - $($nextStory.title)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Acceptance Criteria:" -ForegroundColor DarkGray
    foreach ($criteria in $nextStory.acceptanceCriteria) {
        Write-Host "    - $criteria" -ForegroundColor DarkGray
    }
}

Write-Host "======================================================================" -ForegroundColor Cyan

# Check for BLOCKED
if (Test-Path $BlockedPath) {
    Write-Host ""
    Write-Host "  !! BLOCKED !!" -ForegroundColor Red
    Write-Host ""
    Get-Content $BlockedPath | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ""
}

# Metrics summary
if (Test-Path $MetricsPath) {
    try {
        $metrics = Import-Csv $MetricsPath
        $totalRuns = $metrics.Count
        $successes = @($metrics | Where-Object { $_.success -eq 'true' }).Count
        $timeouts = @($metrics | Where-Object { $_.timeout -eq 'true' }).Count

        Write-Host ""
        Write-Host "  Lifetime Stats: $successes/$totalRuns successful, $timeouts timeouts" -ForegroundColor DarkGray
    } catch {}
}

Write-Host ""
