# Ralph Loop Operations Playbook

This guide covers common operational scenarios for the Ralph Loop autonomous development system.

## Table of Contents
1. [Recovering from Partial Failure](#recovering-from-partial-failure)
2. [Resuming from Specific Iteration](#resuming-from-specific-iteration)
3. [Skipping a Failing Story](#skipping-a-failing-story)
4. [Force-Retrying a Focus Area](#force-retrying-a-focus-area)
5. [Merging Interrupted Sessions](#merging-interrupted-sessions)
6. [Emergency Stop Procedures](#emergency-stop-procedures)

---

## Recovering from Partial Failure

When Ralph crashes mid-sprint:

### Step 1: Check Current State
```powershell
# Check if PRD exists and is valid
Get-Content scripts/ralph/prd.json | ConvertFrom-Json | Select-Object sprintNumber, focusArea

# Check how many stories passed
$prd = Get-Content scripts/ralph/prd.json | ConvertFrom-Json
$prd.userStories | Select-Object id, title, passes
```

### Step 2: Resume from Checkpoint
```powershell
# Resume with existing PRD
.\scripts\ralph\ralph.ps1 -Resume

# If queue mode, check queue state first
Get-Content scripts/ralph/queue.json | ConvertFrom-Json
```

### Step 3: If PRD is Corrupted
```powershell
# Delete corrupted PRD and restart
Remove-Item scripts/ralph/prd.json
.\scripts\ralph\ralph.ps1 -FocusArea "testing"
```

---

## Resuming from Specific Iteration

### Using Session Logs
```powershell
# List available sessions
Get-ChildItem scripts/ralph/logs -Directory | Sort-Object Name -Descending

# Check iteration manifests for a session
Get-ChildItem "scripts/ralph/logs/2026-01-25_120000" -Filter "iteration_*_manifest.json" |
    ForEach-Object { Get-Content $_.FullName | ConvertFrom-Json | Select-Object iteration, storyId, status }
```

### Manual Story Reset
```powershell
# Reset a specific story to incomplete
$prd = Get-Content scripts/ralph/prd.json | ConvertFrom-Json
$story = $prd.userStories | Where-Object { $_.id -eq "US-005" }
$story.passes = $false
$prd | ConvertTo-Json -Depth 10 | Set-Content scripts/ralph/prd.json
```

---

## Skipping a Failing Story

When a story repeatedly fails and blocks progress:

### Step 1: Mark Story as Passed (Skip)
```powershell
$prd = Get-Content scripts/ralph/prd.json | ConvertFrom-Json
$story = $prd.userStories | Where-Object { $_.id -eq "US-007" }
$story.passes = $true
$prd | ConvertTo-Json -Depth 10 | Set-Content scripts/ralph/prd.json
```

### Step 2: Log the Skip (for audit)
```powershell
# Add to skips file manually
$skip = @{
    ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    session = "manual"
    itemId = "US-007"
    itemType = "story"
    reason = "Repeated failures - skipping to unblock"
    blockerType = "manual"
    resolved = $false
} | ConvertTo-Json -Compress

Add-Content "scripts/ralph/logs/latest_session/skips_blockers.jsonl" $skip
```

### Step 3: Resume
```powershell
.\scripts\ralph\ralph.ps1 -Resume
```

---

## Force-Retrying a Focus Area

When a focus area needs to be re-run:

### In Queue Mode
```powershell
# Reset completion flag for a focus area
$queue = Get-Content scripts/ralph/queue.json | ConvertFrom-Json
$area = $queue.focusAreas | Where-Object { $_.id -eq "testing" }
$area.completed = $false
$queue | ConvertTo-Json -Depth 10 | Set-Content scripts/ralph/queue.json

# Resume
.\scripts\ralph\ralph.ps1 -Queue -Resume
```

### Single Focus Area Mode
```powershell
# Delete PRD and restart with specific focus
Remove-Item scripts/ralph/prd.json -ErrorAction SilentlyContinue
.\scripts\ralph\ralph.ps1 -FocusArea "testing"
```

---

## Merging Interrupted Sessions

When you need to combine metrics from multiple interrupted sessions:

### Export Combined Metrics
```powershell
# Get all session metrics
$allMetrics = Get-ChildItem "scripts/ralph/logs" -Directory |
    ForEach-Object {
        $sessionId = $_.Name
        $manifests = Get-ChildItem $_.FullName -Filter "iteration_*_manifest.json"
        $manifests | ForEach-Object {
            $data = Get-Content $_.FullName | ConvertFrom-Json
            $data | Add-Member -NotePropertyName "session" -NotePropertyValue $sessionId -PassThru
        }
    }

# Export to CSV
$allMetrics | Select-Object session, iteration, storyId, status, @{N='duration';E={$_.timestamps.durationSec}} |
    Export-Csv "scripts/ralph/combined_metrics.csv" -NoTypeInformation
```

### Merge Timeline Events
```powershell
# Combine all timeline files
Get-ChildItem "scripts/ralph/logs" -Recurse -Filter "session_timeline.jsonl" |
    ForEach-Object { Get-Content $_.FullName } |
    Set-Content "scripts/ralph/combined_timeline.jsonl"
```

---

## Emergency Stop Procedures

### Graceful Stop
```powershell
# Create BLOCKED.md to pause after current iteration
@"
# Blocked: Manual Stop

Operator requested stop at $(Get-Date)

## Next Steps
- Review progress in watch dashboard
- Check for any uncommitted changes
- Resume when ready: .\scripts\ralph\ralph.ps1 -Resume
"@ | Set-Content scripts/ralph/BLOCKED.md
```

### Hard Stop
```powershell
# Kill all Claude processes
Get-Process | Where-Object { $_.Name -match "claude|node" } | Stop-Process -Force

# Check git status for any half-committed changes
git status
git diff
```

### Recovery After Hard Stop
```powershell
# Check what was in progress
Get-ChildItem "scripts/ralph/logs" -Directory | Sort-Object Name -Descending | Select-Object -First 1 |
    ForEach-Object { Get-Content "$($_.FullName)/session_timeline.jsonl" -Tail 10 }

# Verify git state
git log --oneline -5
git status

# Resume if safe
.\scripts\ralph\ralph.ps1 -Resume
```

---

## Quick Reference

| Scenario | Command |
|----------|---------|
| Resume after crash | `.\ralph.ps1 -Resume` |
| Start fresh | `.\ralph.ps1 -FocusArea "testing"` |
| Check progress | `.\watch.ps1` |
| View session logs | `Get-ChildItem scripts/ralph/logs -Directory` |
| Check PRD status | `Get-Content prd.json \| ConvertFrom-Json` |
| Kill stuck process | `Get-Process claude* \| Stop-Process -Force` |
