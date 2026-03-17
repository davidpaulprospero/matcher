# Ralph Interview Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an interactive interview mode where Ralph asks guided questions, suggests focus areas, and spawns autonomous work in new terminals.

**Architecture:** PowerShell interview script that conducts guided Q&A, saves context to queue.json, then spawns Ralph loop + watch in separate terminal windows. Integrates with existing queue persistence.

**Tech Stack:** PowerShell, JSON (queue.json), Windows Terminal/cmd spawning

---

## Task 1: Create Interview PowerShell Script

**Files:**
- Create: `scripts/ralph/interview.ps1`

**Step 1: Create the base interview script structure**

```powershell
# scripts/ralph/interview.ps1
param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"

# Load config
$config = Get-Content $script:ConfigFile | ConvertFrom-Json

function Write-RalphHeader {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "     'Hi, Super Nintendo Chalmers!' - Ralph" -ForegroundColor Yellow
    Write-Host "     Interview Mode" -ForegroundColor Cyan
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-ExistingContext {
    if (Test-Path $script:QueueFile) {
        $queue = Get-Content $script:QueueFile | ConvertFrom-Json
        if ($queue.interviewContext -and $queue.focusAreas -and $queue.focusAreas.Count -gt 0) {
            $completed = ($queue.focusAreas | Where-Object { $_.completed }).Count
            $total = $queue.focusAreas.Count
            if ($completed -lt $total) {
                return @{
                    HasContext = $true
                    Context = $queue.interviewContext
                    FocusAreas = $queue.focusAreas
                    Completed = $completed
                    Total = $total
                }
            }
        }
    }
    return @{ HasContext = $false }
}

# Entry point
Write-RalphHeader
```

**Step 2: Verify script runs without errors**

Run: `powershell -ExecutionPolicy Bypass -File scripts/ralph/interview.ps1`
Expected: Header displays, no errors

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add interview.ps1 base structure

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Add Smart Resume Detection

**Files:**
- Modify: `scripts/ralph/interview.ps1`

**Step 1: Add resume detection and prompt**

Add after `Get-ExistingContext` function:

```powershell
function Show-ResumePrompt {
    param($ExistingContext)

    Write-Host "  Found interrupted session:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    Context: $($ExistingContext.Context)" -ForegroundColor White
    Write-Host "    Progress: $($ExistingContext.Completed)/$($ExistingContext.Total) focus areas complete" -ForegroundColor White
    Write-Host ""
    Write-Host "    Remaining:" -ForegroundColor Cyan
    $ExistingContext.FocusAreas | Where-Object { -not $_.completed } | ForEach-Object {
        Write-Host "      - $($_.id)" -ForegroundColor Gray
    }
    Write-Host ""

    $choice = Read-Host "  Resume this session? [Y]es / [N]ew interview"
    return $choice -match "^[Yy]"
}

# Check for existing context
$existing = Get-ExistingContext
if ($existing.HasContext -and -not $Resume) {
    $shouldResume = Show-ResumePrompt -ExistingContext $existing
    if ($shouldResume) {
        Write-Host ""
        Write-Host "  Resuming previous session..." -ForegroundColor Green
        # Will spawn windows in Task 6
        $script:ResumeMode = $true
        $script:FocusAreas = $existing.FocusAreas | Where-Object { -not $_.completed }
    }
}
```

**Step 2: Test resume detection with mock queue.json**

Create test queue.json with interviewContext, run interview.ps1, verify prompt appears.

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add smart resume detection to interview mode

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add Guided Question Flow

**Files:**
- Modify: `scripts/ralph/interview.ps1`

**Step 1: Add question functions**

```powershell
function Ask-WorkType {
    Write-Host "  What kind of work?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    [B] Bug fix      - Something's broken" -ForegroundColor White
    Write-Host "    [F] Feature      - Add new functionality" -ForegroundColor White
    Write-Host "    [I] Improvement  - Make something better" -ForegroundColor White
    Write-Host "    [C] Client       - Based on client feedback" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "  Choice"
    switch -Regex ($choice) {
        "^[Bb]" { return "bug" }
        "^[Ff]" { return "feature" }
        "^[Ii]" { return "improvement" }
        "^[Cc]" { return "client" }
        default { return "improvement" }
    }
}

function Ask-Details {
    param([string]$WorkType)

    $prompts = @{
        "bug" = "Describe the bug (or paste error message)"
        "feature" = "What feature do you want?"
        "improvement" = "What should be improved?"
        "client" = "What was the client feedback?"
    }

    Write-Host ""
    Write-Host "  $($prompts[$WorkType]):" -ForegroundColor Cyan
    $details = Read-Host "  "
    return $details
}

function Ask-Area {
    param([string]$WorkType, [string]$Details)

    # Load focus areas from config
    $areas = $config.focusAreas | ForEach-Object { $_.id }

    Write-Host ""
    Write-Host "  Which area? (optional - press Enter to let Ralph decide)" -ForegroundColor Cyan
    Write-Host "    Available: $($areas -join ', ')" -ForegroundColor Gray
    Write-Host ""

    $area = Read-Host "  Area"
    return $area
}

function Ask-Client {
    Write-Host ""
    Write-Host "  Which client? (optional - press Enter to skip)" -ForegroundColor Cyan

    # Check if clients.json exists
    $clientsFile = Join-Path $script:RalphDir "clients.json"
    if (Test-Path $clientsFile) {
        $clients = Get-Content $clientsFile | ConvertFrom-Json
        if ($clients.clients) {
            $clientNames = $clients.clients.PSObject.Properties.Name
            Write-Host "    Known clients: $($clientNames -join ', ')" -ForegroundColor Gray
        }
    }

    $client = Read-Host "  Client"
    return $client
}

function Ask-Priority {
    Write-Host ""
    Write-Host "  Priority?" -ForegroundColor Cyan
    Write-Host "    [H] High - Do this first" -ForegroundColor White
    Write-Host "    [N] Normal - Regular priority (default)" -ForegroundColor White
    Write-Host "    [L] Low - When you get to it" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "  Choice"
    switch -Regex ($choice) {
        "^[Hh]" { return "high" }
        "^[Ll]" { return "low" }
        default { return "normal" }
    }
}
```

**Step 2: Test each question function individually**

Run interview.ps1, verify each question displays correctly and accepts input.

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add guided question functions to interview

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Add Dynamic Question Flow Controller

**Files:**
- Modify: `scripts/ralph/interview.ps1`

**Step 1: Add the main interview conductor**

```powershell
function Start-Interview {
    $context = @{
        workType = $null
        details = $null
        area = $null
        client = $null
        priority = "normal"
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }

    # Question 1: Always ask work type
    $context.workType = Ask-WorkType

    # Question 2: Always ask for details
    $context.details = Ask-Details -WorkType $context.workType

    # Dynamic: If details are vague (< 20 chars), ask for area
    if ($context.details.Length -lt 20) {
        $context.area = Ask-Area -WorkType $context.workType -Details $context.details
    }

    # Dynamic: If client-related or high-stakes, ask for client
    if ($context.workType -eq "client" -or $context.details -match "client|theresa|stu|feedback") {
        $context.client = Ask-Client
    }

    # Dynamic: If bug or details mention "urgent|critical|asap", ask priority
    if ($context.workType -eq "bug" -or $context.details -match "urgent|critical|asap|broken") {
        $context.priority = Ask-Priority
    }

    return $context
}

# Main flow (add after resume check)
if (-not $script:ResumeMode) {
    Write-Host "  Let's figure out what you need." -ForegroundColor White
    Write-Host ""

    $interviewContext = Start-Interview

    Write-Host ""
    Write-Host "  Got it. Let me suggest some focus areas..." -ForegroundColor Green
    Write-Host ""
}
```

**Step 2: Test the dynamic flow**

Run interview.ps1 with different inputs:
- Short bug description → should ask for area
- Client feedback → should ask for client name
- Urgent issue → should ask for priority

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add dynamic question flow controller

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Add Focus Area Suggestion Logic

**Files:**
- Modify: `scripts/ralph/interview.ps1`

**Step 1: Add suggestion generator**

```powershell
function Get-SuggestedFocusAreas {
    param($Context)

    $suggestions = @()
    $allAreas = $config.focusAreas

    # If user specified an area, include it first
    if ($Context.area) {
        $match = $allAreas | Where-Object { $_.id -eq $Context.area }
        if ($match) {
            $suggestions += $match.id
        }
    }

    # Keyword matching from details
    $details = $Context.details.ToLower()

    $keywordMap = @{
        "otio|timeline|edl|xml|davinci|resolve" = "otio"
        "download|youtube|yt-dlp|429|rate limit|cookie" = "rate-limiting"
        "match|confidence|score|quality|poor" = "quality"
        "caption|subtitle|srt|transcript" = "caption"
        "config|yaml|setting" = "config"
        "test|coverage|pytest" = "testing"
        "speed|slow|fast|performance|cache" = "speed"
        "client|theresa|stu|preset|learn" = "client-learning"
        "heal|recover|retry|error|fail" = "agents"
        "compile|compilation|topic|keyword" = "compilation"
    }

    foreach ($pattern in $keywordMap.Keys) {
        if ($details -match $pattern) {
            $areaId = $keywordMap[$pattern]
            if ($areaId -notin $suggestions) {
                $match = $allAreas | Where-Object { $_.id -eq $areaId }
                if ($match) {
                    $suggestions += $areaId
                }
            }
        }
    }

    # If still not enough, add related areas based on work type
    if ($suggestions.Count -lt 3) {
        $typeDefaults = @{
            "bug" = @("agents", "testing")
            "feature" = @("pipeline", "config")
            "improvement" = @("quality", "speed")
            "client" = @("client-learning", "quality")
        }

        foreach ($default in $typeDefaults[$Context.workType]) {
            if ($default -notin $suggestions -and $suggestions.Count -lt 5) {
                $suggestions += $default
            }
        }
    }

    # Cap at 5
    return $suggestions | Select-Object -First 5
}

function Show-Suggestions {
    param($Suggestions)

    Write-Host "  Suggested focus areas:" -ForegroundColor Cyan
    Write-Host ""

    $i = 1
    foreach ($area in $Suggestions) {
        $areaInfo = $config.focusAreas | Where-Object { $_.id -eq $area }
        Write-Host "    [$i] $area" -ForegroundColor White -NoNewline
        if ($areaInfo.name) {
            Write-Host " - $($areaInfo.name)" -ForegroundColor Gray
        } else {
            Write-Host ""
        }
        $i++
    }

    Write-Host ""
    Write-Host "  Options:" -ForegroundColor Cyan
    Write-Host "    [A] Approve all" -ForegroundColor White
    Write-Host "    [1-5] Remove specific area" -ForegroundColor White
    Write-Host "    [+area] Add an area (e.g., +testing)" -ForegroundColor White
    Write-Host "    [R] Restart interview" -ForegroundColor White
    Write-Host ""
}

function Get-ApprovedAreas {
    param($Suggestions)

    $approved = [System.Collections.ArrayList]@($Suggestions)

    while ($true) {
        Show-Suggestions -Suggestions $approved
        $choice = Read-Host "  Choice"

        if ($choice -match "^[Aa]") {
            return $approved
        }
        elseif ($choice -match "^[Rr]") {
            return $null  # Signal to restart
        }
        elseif ($choice -match "^\+(.+)") {
            $newArea = $Matches[1].Trim()
            if ($newArea -notin $approved) {
                $approved.Add($newArea) | Out-Null
                Write-Host "    Added: $newArea" -ForegroundColor Green
            }
        }
        elseif ($choice -match "^[1-5]$") {
            $index = [int]$choice - 1
            if ($index -lt $approved.Count) {
                $removed = $approved[$index]
                $approved.RemoveAt($index)
                Write-Host "    Removed: $removed" -ForegroundColor Yellow
            }
        }
    }
}
```

**Step 2: Test suggestion logic**

Run interview.ps1 with:
- "OTIO crashes" → should suggest `otio`
- "Match quality poor for theresa" → should suggest `quality`, `client-learning`

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add focus area suggestion logic

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Add Queue Save and Window Spawning

**Files:**
- Modify: `scripts/ralph/interview.ps1`

**Step 1: Add queue save and spawn functions**

```powershell
function Save-InterviewQueue {
    param($Context, $FocusAreas)

    $queue = @{
        interviewContext = "$($Context.workType): $($Context.details)"
        interviewDetails = $Context
        focusAreas = $FocusAreas | ForEach-Object {
            @{
                id = $_
                completed = $false
                startedAt = $null
                completedAt = $null
            }
        }
        createdAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        sessionId = [guid]::NewGuid().ToString().Substring(0, 8)
    }

    $queue | ConvertTo-Json -Depth 10 | Set-Content $script:QueueFile
    Write-Host "  Saved queue with $($FocusAreas.Count) focus areas" -ForegroundColor Green

    return $queue
}

function Start-RalphWindows {
    param($FocusAreas)

    $areaList = $FocusAreas -join ","

    Write-Host ""
    Write-Host "  Launching Ralph..." -ForegroundColor Cyan
    Write-Host ""

    # Spawn Ralph loop in new window
    $ralphCmd = "Set-Location '$script:ProjectRoot'; .\scripts\ralph\ralph.ps1 -Queue -SkipPlanApproval"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$ralphCmd}" -WindowStyle Normal

    # Brief pause to let Ralph start
    Start-Sleep -Seconds 2

    # Spawn Watch in new window
    $watchCmd = "Set-Location '$script:ProjectRoot'; .\scripts\ralph\watch.ps1"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$watchCmd}" -WindowStyle Normal

    Write-Host "  Ralph loop and watch windows launched!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  You can close this window now." -ForegroundColor Gray
    Write-Host ""
}

# Complete the main flow
if (-not $script:ResumeMode) {
    $suggestions = Get-SuggestedFocusAreas -Context $interviewContext
    $approved = Get-ApprovedAreas -Suggestions $suggestions

    if ($null -eq $approved) {
        # User chose to restart
        Write-Host "  Restarting interview..." -ForegroundColor Yellow
        & $PSCommandPath
        exit
    }

    Save-InterviewQueue -Context $interviewContext -FocusAreas $approved
    Start-RalphWindows -FocusAreas $approved
}
else {
    # Resume mode - just spawn windows
    Start-RalphWindows -FocusAreas ($script:FocusAreas | ForEach-Object { $_.id })
}
```

**Step 2: Test full flow end-to-end**

Run interview.ps1, complete interview, verify:
- queue.json is created with context
- Two new PowerShell windows spawn
- Ralph loop starts in queue mode

**Step 3: Commit**

```bash
git add scripts/ralph/interview.ps1
git commit -m "feat: add queue save and window spawning

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Update ralph.ps1 Queue Mode for Interview Context

**Files:**
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Add interview context awareness to queue processing**

Find the queue processing section and update to handle interview focus areas:

```powershell
# In the Initialize-Queue or queue loading section, add:

function Get-InterviewFocusAreas {
    if (Test-Path $script:QueueFile) {
        $queue = Get-Content $script:QueueFile | ConvertFrom-Json
        if ($queue.focusAreas) {
            $remaining = $queue.focusAreas | Where-Object { -not $_.completed }
            return $remaining
        }
    }
    return $null
}

function Update-InterviewProgress {
    param([string]$FocusAreaId, [bool]$Completed)

    if (Test-Path $script:QueueFile) {
        $queue = Get-Content $script:QueueFile | ConvertFrom-Json
        if ($queue.focusAreas) {
            $area = $queue.focusAreas | Where-Object { $_.id -eq $FocusAreaId }
            if ($area) {
                $area.completed = $Completed
                $area.completedAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                $queue | ConvertTo-Json -Depth 10 | Set-Content $script:QueueFile
            }
        }
    }
}

# In the main loop, check for interview mode:
$interviewAreas = Get-InterviewFocusAreas
if ($interviewAreas -and $Queue) {
    Write-Host "  Running interview queue: $($interviewAreas.Count) focus areas" -ForegroundColor Cyan
    foreach ($area in $interviewAreas) {
        $script:CurrentFocusArea = $area.id
        # ... run the focus area ...
        Update-InterviewProgress -FocusAreaId $area.id -Completed $true
    }
}
```

**Step 2: Test queue mode reads interview context**

Create queue.json with interview data, run `ralph.ps1 -Queue`, verify it processes the focus areas.

**Step 3: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat: update ralph.ps1 to handle interview queue

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Add Completion Choice (Loop Back vs TrueAuto)

**Files:**
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Add completion handler**

```powershell
function Show-CompletionChoice {
    param($CompletedAreas)

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Green
    Write-Host "     Interview queue complete!" -ForegroundColor Green
    Write-Host "  =====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Completed: $($CompletedAreas -join ', ')" -ForegroundColor White
    Write-Host ""
    Write-Host "  What next?" -ForegroundColor Cyan
    Write-Host "    [I] New interview - Give me more direction" -ForegroundColor White
    Write-Host "    [T] TrueAuto - Continue improving on your own" -ForegroundColor White
    Write-Host "    [S] Stop - I'll check results later" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "  Choice"

    switch -Regex ($choice) {
        "^[Ii]" {
            # Clear old queue and launch interview
            Remove-Item $script:QueueFile -Force -ErrorAction SilentlyContinue
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {Set-Location '$script:ProjectRoot'; .\scripts\ralph\interview.ps1}"
            return "interview"
        }
        "^[Tt]" {
            # Clear interview context, continue in TrueAuto
            $queue = Get-Content $script:QueueFile | ConvertFrom-Json
            $queue.PSObject.Properties.Remove('interviewContext')
            $queue.PSObject.Properties.Remove('interviewDetails')
            $queue.PSObject.Properties.Remove('focusAreas')
            $queue | ConvertTo-Json -Depth 10 | Set-Content $script:QueueFile
            return "trueauto"
        }
        default {
            return "stop"
        }
    }
}

# Add at end of interview queue processing:
if ($interviewAreas -and $allComplete) {
    $completedIds = $interviewAreas | ForEach-Object { $_.id }
    $nextAction = Show-CompletionChoice -CompletedAreas $completedIds

    if ($nextAction -eq "trueauto") {
        Write-Host "  Falling into TrueAuto mode..." -ForegroundColor Cyan
        $TrueAuto = $true
        # Continue main loop
    }
    elseif ($nextAction -eq "interview") {
        Write-Host "  Interview window launched. Goodbye!" -ForegroundColor Green
        exit
    }
    else {
        Write-Host "  Stopping. Run 1-im-learnding.bat to check results!" -ForegroundColor Green
        exit
    }
}
```

**Step 2: Test completion flow**

Run ralph.ps1 with a small interview queue (1 focus area), complete it, verify choice prompt appears.

**Step 3: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat: add completion choice (interview/trueauto/stop)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Create Interview Batch File

**Files:**
- Create: `scripts/ralph/7-hi-super-nintendo-chalmers.bat`

**Step 1: Create the batch file**

```batch
@echo off
REM ============================================================================
REM   "Hi, Super Nintendo Chalmers!" - Ralph Wiggum
REM
REM   Interview Mode: Tell Ralph what you want, he'll make it happen
REM ============================================================================

cd /d "%~dp0..\.."

echo.
echo   =====================================================
echo      "Hi, Super Nintendo Chalmers!" - Ralph
echo      Interview Mode
echo   =====================================================
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0interview.ps1" %*
```

**Step 2: Test batch file launches interview**

Run: `scripts\ralph\7-hi-super-nintendo-chalmers.bat`
Expected: Interview script starts

**Step 3: Commit**

```bash
git add scripts/ralph/7-hi-super-nintendo-chalmers.bat
git commit -m "feat: add interview batch file (hi-super-nintendo-chalmers)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Update Morning Check-in with Interview Option

**Files:**
- Modify: `scripts/ralph/1-im-learnding.bat`

**Step 1: Add interview option to menu**

Find the "What's next?" menu section and update:

```batch
echo   -----------------------------------------------------
echo   What's next?
echo   -----------------------------------------------------
echo     1. Continue Ralph (interactive)
echo     2. Start TrueAuto mode
echo     3. Interview mode (give specific direction)
echo     4. View detailed logs
echo     5. Just exit
echo.

set /p NEXT="   Choose (1-5): "

if "%NEXT%"=="1" (
    start "Ralph Loop" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\ralph.ps1}"
    start "Ralph Watch" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\watch.ps1}"
)
if "%NEXT%"=="2" (
    start "Ralph Loop" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\ralph.ps1 -TrueAuto -SkipPlanApproval}"
    start "Ralph Watch" powershell -NoExit -Command "& {Set-Location 'D:\_Projects\voiceover-matcher-subtitle'; .\scripts\ralph\watch.ps1}"
)
if "%NEXT%"=="3" (
    call "%~dp07-hi-super-nintendo-chalmers.bat"
)
if "%NEXT%"=="4" (
    call "%~dp04-tastes-like-burning.bat"
)
```

**Step 2: Test morning check-in menu**

Run: `scripts\ralph\1-im-learnding.bat`
Choose option 3, verify interview starts.

**Step 3: Commit**

```bash
git add scripts/ralph/1-im-learnding.bat
git commit -m "feat: add interview option to morning check-in

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Update README Documentation

**Files:**
- Modify: `scripts/ralph/README.md`

**Step 1: Add interview mode section**

Add after the Quick Start section:

```markdown
## Interview Mode

Give Ralph specific direction before he starts working.

```powershell
# Start interview
.\scripts\ralph\7-hi-super-nintendo-chalmers.bat

# Or from morning check-in
.\scripts\ralph\1-im-learnding.bat
# Choose option 3
```

### Interview Flow

1. Ralph asks what kind of work (bug/feature/improvement/client)
2. You describe what you want
3. Ralph asks follow-up questions if needed
4. Ralph suggests 3-5 focus areas
5. You approve/modify the list
6. Ralph spawns loop + watch windows and gets to work

### After Completion

When all focus areas are done, Ralph asks:
- **[I]nterview** - Give more direction
- **[T]rueAuto** - Continue improving on his own
- **[S]top** - Check results later

### Crash Recovery

If interrupted, next interview start offers to resume:
```
Found interrupted session:
  Context: bug: OTIO crashes on unicode
  Progress: 2/4 focus areas complete

  Resume this session? [Y]es / [N]ew interview
```
```

**Step 2: Commit**

```bash
git add scripts/ralph/README.md
git commit -m "docs: add interview mode documentation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Create interview.ps1 base structure |
| 2 | Add smart resume detection |
| 3 | Add guided question functions |
| 4 | Add dynamic question flow controller |
| 5 | Add focus area suggestion logic |
| 6 | Add queue save and window spawning |
| 7 | Update ralph.ps1 for interview queue |
| 8 | Add completion choice (loop/trueauto/stop) |
| 9 | Create interview batch file |
| 10 | Update morning check-in with interview option |
| 11 | Update README documentation |

**Total commits:** 11
