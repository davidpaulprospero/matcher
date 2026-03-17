# Ralph Startup Experience Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Ralph Loop startup reliable, fast, and user-friendly with clear feedback.

**Architecture:** Add environment validation at startup, configurable Claude CLI path, streamlined sprint selection menu, and real-time status feedback. Replace hidden `cmd` process with direct PowerShell invocation for better error handling.

**Tech Stack:** PowerShell, JSON configuration, Claude CLI

---

## Current Problems

1. **Claude CLI not found** - Uses `cmd /c ... | claude` which fails if `claude` not in PATH for cmd.exe
2. **Confusing menu flow** - Multiple prompts before starting, unclear sprint state
3. **Silent failures** - Errors go to log files, user sees "no progress"
4. **Slow feedback** - No indication of what's happening during startup
5. **Hidden process** - `WindowStyle Hidden` makes debugging impossible

---

## Task 1: Add Claude CLI Path Configuration

**Files:**
- Modify: `scripts/ralph/ralph-config.json`
- Modify: `scripts/ralph/ralph.ps1:636-642`

**Step 1: Add claudePath to config**

In `scripts/ralph/ralph-config.json`, add to the root object:

```json
{
  "claudePath": {
    "auto": true,
    "path": null,
    "description": "Path to claude CLI. Set auto=false and provide path if not in PATH."
  },
  ...existing config...
}
```

**Step 2: Add Get-ClaudePath function to ralph.ps1**

After line 100 (after PRD MANAGEMENT section header), add:

```powershell
# ============================================================================
# CLAUDE CLI RESOLUTION
# ============================================================================

function Get-ClaudePath {
    # Check config first
    if ($Config.claudePath -and -not $Config.claudePath.auto -and $Config.claudePath.path) {
        if (Test-Path $Config.claudePath.path) {
            return $Config.claudePath.path
        }
        Write-Log "  WARNING: Configured claudePath not found: $($Config.claudePath.path)" -Color Yellow
    }

    # Try common locations
    $commonPaths = @(
        "$env:APPDATA\npm\claude.cmd",
        "$env:LOCALAPPDATA\Programs\claude\claude.exe",
        "$env:USERPROFILE\.claude\local\claude.exe",
        "C:\Program Files\Claude\claude.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    # Try PATH via Get-Command
    $cmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    return $null
}

function Test-ClaudeAvailable {
    $claudePath = Get-ClaudePath
    if (-not $claudePath) {
        return @{ Available = $false; Path = $null; Error = "Claude CLI not found" }
    }

    # Test it works
    try {
        $version = & $claudePath --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            return @{ Available = $true; Path = $claudePath; Version = $version }
        }
        return @{ Available = $false; Path = $claudePath; Error = "Claude CLI failed: $version" }
    } catch {
        return @{ Available = $false; Path = $claudePath; Error = $_.Exception.Message }
    }
}
```

**Step 3: Test manually**

```powershell
cd D:\_Projects\voiceover-matcher-subtitle
. .\scripts\ralph\ralph.ps1
Get-ClaudePath
Test-ClaudeAvailable
```

Expected: Returns path to claude.cmd and `Available = $true`

**Step 4: Commit**

```bash
git add scripts/ralph/ralph-config.json scripts/ralph/ralph.ps1
git commit -m "feat(ralph): add Claude CLI path resolution"
```

---

## Task 2: Add Startup Validation

**Files:**
- Modify: `scripts/ralph/ralph.ps1:888-910` (MAIN section)

**Step 1: Add validation function**

After the `Test-ClaudeAvailable` function, add:

```powershell
function Test-RalphEnvironment {
    $issues = @()
    $warnings = @()

    # Check Claude CLI
    $claude = Test-ClaudeAvailable
    if (-not $claude.Available) {
        $issues += "Claude CLI: $($claude.Error)"
        $issues += "  Fix: Install via 'npm install -g @anthropic-ai/claude-code'"
        $issues += "  Or set claudePath in ralph-config.json"
    }

    # Check config file
    if (-not (Test-Path $ConfigPath)) {
        $issues += "Config file missing: $ConfigPath"
    }

    # Check prompt file
    $promptPath = Join-Path $ScriptDir "prompt.md"
    if (-not (Test-Path $promptPath)) {
        $issues += "Prompt file missing: $promptPath"
    }

    # Check git
    $gitStatus = git status --porcelain 2>&1
    if ($LASTEXITCODE -ne 0) {
        $warnings += "Not in a git repository"
    }

    # Check for stale locks
    $lockFiles = Get-ChildItem "$env:USERPROFILE\.claude\*.lock" -ErrorAction SilentlyContinue
    if ($lockFiles) {
        $warnings += "Found $($lockFiles.Count) Claude lock file(s) - may indicate crashed session"
    }

    return @{
        Valid = ($issues.Count -eq 0)
        Issues = $issues
        Warnings = $warnings
        ClaudePath = $claude.Path
        ClaudeVersion = $claude.Version
    }
}
```

**Step 2: Add validation call at startup**

Replace the MAIN section header (around line 888-910) with:

```powershell
# ============================================================================
# MAIN
# ============================================================================

# Validate environment before anything else
Write-Host ""
Write-Host "  Checking environment..." -ForegroundColor DarkGray

$envCheck = Test-RalphEnvironment

if (-not $envCheck.Valid) {
    Write-Host ""
    Write-Host "  STARTUP FAILED - Environment issues:" -ForegroundColor Red
    Write-Host ""
    foreach ($issue in $envCheck.Issues) {
        Write-Host "    $issue" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

if ($envCheck.Warnings.Count -gt 0) {
    foreach ($warning in $envCheck.Warnings) {
        Write-Host "  WARNING: $warning" -ForegroundColor Yellow
    }
}

Write-Host "  Claude CLI: $($envCheck.ClaudePath)" -ForegroundColor DarkGreen
Write-Host "  Version: $($envCheck.ClaudeVersion)" -ForegroundColor DarkGray

# Store for later use
$script:ClaudePath = $envCheck.ClaudePath

# Check for incompatible modes
if ($Queue -and $TrueAuto) {
    Write-Host "Error: -Queue and -TrueAuto cannot be used together" -ForegroundColor Red
    exit 1
}
```

**Step 3: Test startup validation**

```powershell
.\scripts\ralph\ralph.ps1 -DryRun
```

Expected: Shows "Checking environment...", Claude path, and version before menu

**Step 4: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat(ralph): add startup environment validation"
```

---

## Task 3: Use Resolved Claude Path in Process Calls

**Files:**
- Modify: `scripts/ralph/ralph.ps1:636-642` (New-Sprint function)
- Modify: `scripts/ralph/ralph.ps1:778-782` (Invoke-Iteration function)

**Step 1: Update New-Sprint to use resolved path**

Replace lines 638-640:

```powershell
# OLD:
$Process = Start-Process -FilePath "cmd" `
    -ArgumentList "/c", "type `"$tempPrompt`" | claude --print --dangerously-skip-permissions > `"$sprintLogFile`" 2>&1" `
    -PassThru -Wait:$false -WindowStyle Hidden
```

With:

```powershell
# NEW: Use resolved Claude path and PowerShell for better error handling
$claudeArgs = "--print --dangerously-skip-permissions"
$Process = Start-Process -FilePath "powershell" `
    -ArgumentList "-NoProfile", "-Command", "Get-Content '$tempPrompt' | & '$($script:ClaudePath)' $claudeArgs > '$sprintLogFile' 2>&1" `
    -PassThru -Wait:$false -WindowStyle Hidden
```

**Step 2: Update Invoke-Iteration to use resolved path**

Replace lines 780-782:

```powershell
# OLD:
$Process = Start-Process -FilePath "cmd" `
    -ArgumentList "/c", "type `"$tempPrompt`" | claude --print --dangerously-skip-permissions > `"$iterationLogFile`" 2>&1" `
    -PassThru -Wait:$false -WindowStyle Hidden
```

With:

```powershell
# NEW: Use resolved Claude path
$claudeArgs = "--print --dangerously-skip-permissions"
$Process = Start-Process -FilePath "powershell" `
    -ArgumentList "-NoProfile", "-Command", "Get-Content '$tempPrompt' | & '$($script:ClaudePath)' $claudeArgs > '$iterationLogFile' 2>&1" `
    -PassThru -Wait:$false -WindowStyle Hidden
```

**Step 3: Test with dry run first**

```powershell
.\scripts\ralph\ralph.ps1 -DryRun -FocusArea pipeline
```

**Step 4: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat(ralph): use resolved Claude path in process calls"
```

---

## Task 4: Add Quick Start Mode

**Files:**
- Modify: `scripts/ralph/ralph.ps1:13-24` (param block)
- Modify: `scripts/ralph/ralph.ps1` (after validation, before menu)

**Step 1: Add -Resume parameter**

Update the param block:

```powershell
param(
    [string]$FocusArea = "",
    [string]$Task = "",
    [int]$MaxIterations = 15,
    [int]$TimeoutMinutes = 30,
    [int]$MaxTotalMinutes = 120,
    [string]$Granularity = "medium",
    [switch]$TrueAuto,
    [switch]$Queue,
    [switch]$SkipPlanApproval,
    [switch]$DryRun,
    [switch]$Resume  # NEW: Auto-resume last sprint without prompts
)
```

**Step 2: Add quick resume logic after validation**

After the environment validation block, before `$script:QueueMode = $Queue`:

```powershell
# Quick Resume Mode - skip all prompts, continue where we left off
if ($Resume) {
    $status = Get-StoryStatus
    if ($status.Total -gt 0 -and $status.Incomplete -gt 0) {
        $FocusArea = $status.Prd.focusArea
        Write-Log ""
        Write-Log "  QUICK RESUME" -Color Green
        Write-Log "  Sprint: $($status.Prd.branchName)" -Color Cyan
        Write-Log "  Focus: $FocusArea" -Color Cyan
        Write-Log "  Progress: $($status.Complete)/$($status.Total) complete" -Color Yellow
        Write-Log "  Next: $($status.NextStory.id) - $($status.NextStory.title)" -Color White
        Write-Log ""

        # Switch to sprint branch if needed
        $currentBranch = git branch --show-current 2>$null
        if ($status.Prd.branchName -and $currentBranch -ne $status.Prd.branchName) {
            git checkout $status.Prd.branchName 2>$null
        }

        # Skip all menus, go straight to main loop
        $script:QueueMode = $false
        $SkipPlanApproval = $true
        # Jump to iteration loop (handled by existing logic)
    } else {
        Write-Log "  No active sprint to resume" -Color Yellow
        Write-Log "  Starting interactive mode..." -Color DarkGray
        $Resume = $false
    }
}
```

**Step 3: Update README with new flag**

In `scripts/ralph/README.md`, add to Parameters table:

```markdown
| `-Resume` | false | Quick resume last sprint (no prompts) |
```

**Step 4: Test quick resume**

```powershell
# First, ensure there's an active sprint in prd.json, then:
.\scripts\ralph\ralph.ps1 -Resume -DryRun
```

**Step 5: Commit**

```bash
git add scripts/ralph/ralph.ps1 scripts/ralph/README.md
git commit -m "feat(ralph): add -Resume flag for quick sprint continuation"
```

---

## Task 5: Improve Sprint Selection UX

**Files:**
- Modify: `scripts/ralph/ralph.ps1:968-1000` (sprint mismatch handling)

**Step 1: Replace verbose sprint selection with streamlined version**

Replace the sprint mismatch handling (around lines 968-1000):

```powershell
# Check if we need to create a sprint
$status = Get-StoryStatus
$existingFocus = $status.Prd.focusArea
$needNewSprint = $false

if ($status.Total -eq 0 -or $status.Incomplete -eq 0) {
    $needNewSprint = $true
} elseif ($existingFocus -and $existingFocus -ne $FocusArea -and $FocusArea -ne "custom" -and -not $Resume) {
    # Streamlined sprint conflict UI
    Write-Log ""
    Write-Host "  +-----------------------------------------+" -ForegroundColor Cyan
    Write-Host "  |  EXISTING SPRINT FOUND                  |" -ForegroundColor Cyan
    Write-Host "  +-----------------------------------------+" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Current: " -NoNewline -ForegroundColor White
    Write-Host "$existingFocus" -ForegroundColor Yellow -NoNewline
    Write-Host " ($($status.Incomplete) remaining)" -ForegroundColor DarkGray
    Write-Host "  Selected: " -NoNewline -ForegroundColor White
    Write-Host "$FocusArea" -ForegroundColor Green
    Write-Host ""
    Write-Host "  [C] Continue '$existingFocus' sprint" -ForegroundColor White
    Write-Host "  [N] New '$FocusArea' sprint" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "  Choice"

    switch ($choice.ToUpper()) {
        'N' {
            $needNewSprint = $true
        }
        'Q' {
            Write-Log "  Exiting." -Color Yellow
            exit 0
        }
        default {
            $FocusArea = $existingFocus
        }
    }
}
```

**Step 2: Test the new UI**

Create a sprint with one focus, then try to start with different focus:

```powershell
.\scripts\ralph\ralph.ps1 -FocusArea otio -DryRun
# Should show the streamlined conflict UI
```

**Step 3: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat(ralph): streamline sprint selection UX"
```

---

## Task 6: Add Startup Progress Indicator

**Files:**
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Add spinner function**

After the logging functions (around line 90), add:

```powershell
function Write-Progress-Inline {
    param(
        [string]$Activity,
        [string]$Status = ""
    )
    $spinChars = @('|', '/', '-', '\')
    $script:SpinIndex = if ($script:SpinIndex) { ($script:SpinIndex + 1) % 4 } else { 0 }
    $spin = $spinChars[$script:SpinIndex]

    $line = "  $spin $Activity"
    if ($Status) { $line += " - $Status" }

    Write-Host "`r$line" -NoNewline -ForegroundColor DarkGray
}

function Write-Progress-Complete {
    param(
        [string]$Activity,
        [bool]$Success = $true
    )
    $symbol = if ($Success) { "[OK]" } else { "[FAIL]" }
    $color = if ($Success) { "Green" } else { "Red" }
    Write-Host "`r  $symbol $Activity                    " -ForegroundColor $color
}
```

**Step 2: Use progress indicators in startup**

Update the environment check section:

```powershell
Write-Host ""
Write-Progress-Inline "Checking Claude CLI"
Start-Sleep -Milliseconds 100

$envCheck = Test-RalphEnvironment

if (-not $envCheck.Valid) {
    Write-Progress-Complete "Checking Claude CLI" -Success $false
    # ... rest of error handling
}

Write-Progress-Complete "Checking Claude CLI" -Success $true
Write-Host "    Path: $($envCheck.ClaudePath)" -ForegroundColor DarkGray
```

**Step 3: Test startup visuals**

```powershell
.\scripts\ralph\ralph.ps1 -DryRun
```

**Step 4: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat(ralph): add startup progress indicators"
```

---

## Task 7: Create Startup Batch File for Convenience

**Files:**
- Create: `scripts/ralph/start.bat`
- Create: `scripts/ralph/resume.bat`

**Step 1: Create start.bat**

```batch
@echo off
REM Ralph Loop Quick Start
REM Usage: start.bat [focus-area]
REM Example: start.bat quality

cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0ralph.ps1" %*
```

**Step 2: Create resume.bat**

```batch
@echo off
REM Ralph Loop Quick Resume
REM Resumes the last active sprint without prompts

cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0ralph.ps1" -Resume %*
```

**Step 3: Test batch files**

```cmd
cd D:\_Projects\voiceover-matcher-subtitle\scripts\ralph
resume.bat -DryRun
```

**Step 4: Update README**

Add to Quick Start section:

```markdown
## Quick Start (Batch Files)

```cmd
# Start with interactive menu
scripts\ralph\start.bat

# Quick resume last sprint
scripts\ralph\resume.bat

# Start specific focus area
scripts\ralph\start.bat -FocusArea quality
```
```

**Step 5: Commit**

```bash
git add scripts/ralph/start.bat scripts/ralph/resume.bat scripts/ralph/README.md
git commit -m "feat(ralph): add convenience batch files for startup"
```

---

## Task 8: Add Queue State Persistence for Recovery

**Files:**
- Modify: `scripts/ralph/ralph.ps1` (queue management section)
- Modify: `scripts/ralph/queue.json` (add session tracking)

**Problem:** If Claude runs out of tokens mid-queue, the queue state is lost and user must restart from scratch.

**Step 1: Add session tracking to queue state**

Update the `Initialize-Queue` function to include session info:

```powershell
function Initialize-Queue {
    param([array]$FocusAreas)
    $q = @{
        queue = $FocusAreas
        currentIndex = 0
        completedAreas = @()
        skippedAreas = @()
        createdAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        retryCount = 0
        # NEW: Session tracking for recovery
        session = @{
            id = $SessionTimestamp
            startedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            lastActivityAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            currentStoryId = $null
            iterationCount = 0
        }
        networkRetries = @{ currentRetryCount = 0; maxRetries = 10; lastError = $null; lastRetryAt = $null; backoffSeconds = 60 }
        apiRetries = @{ currentRetryCount = 0; maxRetries = 5; lastError = $null; lastRetryAt = $null; backoffSeconds = 30 }
    }
    Save-Queue $q
    return $q
}
```

**Step 2: Add queue activity update function**

After `Save-Queue` function, add:

```powershell
function Update-QueueActivity {
    param(
        [string]$StoryId = $null,
        [int]$Iteration = 0
    )
    $q = Get-Queue
    if ($q.session) {
        $q.session.lastActivityAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        if ($StoryId) { $q.session.currentStoryId = $StoryId }
        if ($Iteration -gt 0) { $q.session.iterationCount = $Iteration }
        Save-Queue $q
    }
}
```

**Step 3: Call Update-QueueActivity in main loop**

In the main loop, after incrementing `$script:IterationCount`, add:

```powershell
$script:IterationCount++
$sprintIterations++

# Update queue state for recovery
if ($script:QueueMode) {
    Update-QueueActivity -StoryId $status.NextStory.id -Iteration $script:IterationCount
}
```

**Step 4: Add queue recovery detection at startup**

After the `Test-QueueActive` check (around line 913), add:

```powershell
# Check for interrupted queue session
if ($Queue -and (Test-QueueActive)) {
    $existingQueue = Get-Queue
    if ($existingQueue.session -and $existingQueue.session.id -ne $SessionTimestamp) {
        $lastActivity = $existingQueue.session.lastActivityAt
        $currentFocus = Get-CurrentQueueFocus

        Write-Log ""
        Write-Host "  +-----------------------------------------+" -ForegroundColor Yellow
        Write-Host "  |  INTERRUPTED QUEUE DETECTED             |" -ForegroundColor Yellow
        Write-Host "  +-----------------------------------------+" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Previous session: $($existingQueue.session.id)" -ForegroundColor DarkGray
        Write-Host "  Last activity: $lastActivity" -ForegroundColor DarkGray
        Write-Host "  Current focus: $currentFocus" -ForegroundColor Cyan
        Write-Host "  Progress: $($existingQueue.currentIndex + 1)/$($existingQueue.queue.Count)" -ForegroundColor Yellow
        Write-Host "  Story: $($existingQueue.session.currentStoryId)" -ForegroundColor White
        Write-Host ""
        Write-Host "  [R] Resume interrupted queue" -ForegroundColor Green
        Write-Host "  [N] Start fresh queue" -ForegroundColor White
        Write-Host "  [Q] Quit" -ForegroundColor DarkGray
        Write-Host ""

        $choice = Read-Host "  Choice"

        switch ($choice.ToUpper()) {
            'R' {
                Write-Log "  Resuming queue from $currentFocus..." -Color Green
                $FocusArea = $currentFocus
                # Update session to current
                $existingQueue.session.id = $SessionTimestamp
                $existingQueue.session.startedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                Save-Queue $existingQueue
            }
            'N' {
                Reset-Queue
                $selectedAreas = Select-QueueFocusAreas
                Initialize-Queue -FocusAreas $selectedAreas
            }
            'Q' {
                exit 0
            }
            default {
                # Default to resume
                $FocusArea = $currentFocus
                $existingQueue.session.id = $SessionTimestamp
                Save-Queue $existingQueue
            }
        }
    }
}
```

**Step 5: Test queue recovery**

```powershell
# Start queue, let it run one iteration, then kill the window
.\scripts\ralph\ralph.ps1 -Queue

# Restart - should detect interrupted queue
.\scripts\ralph\ralph.ps1 -Queue
```

**Step 6: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "feat(ralph): add queue state persistence for crash recovery"
```

---

## Task 9: Add Queue Mode Batch File

**Files:**
- Create: `scripts/ralph/queue.bat`
- Modify: `scripts/ralph/README.md`

**Step 1: Create queue.bat**

```batch
@echo off
REM Ralph Loop Queue Mode
REM Runs multiple focus areas in sequence
REM Usage: queue.bat [additional flags]
REM Example: queue.bat -SkipPlanApproval

cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0ralph.ps1" -Queue %*
```

**Step 2: Update README with queue batch file**

Add to Quick Start section:

```markdown
# Queue mode (multiple focus areas)
scripts\ralph\queue.bat

# Queue mode, skip plan approval
scripts\ralph\queue.bat -SkipPlanApproval
```

**Step 3: Test**

```cmd
scripts\ralph\queue.bat -DryRun
```

**Step 4: Commit**

```bash
git add scripts/ralph/queue.bat scripts/ralph/README.md
git commit -m "feat(ralph): add queue mode batch file"
```

---

## Task 10: Add Fast-Fail Detection for Stuck Iterations

**Files:**
- Modify: `scripts/ralph/ralph-config.json`
- Modify: `scripts/ralph/ralph.ps1` (Invoke-Iteration and main loop)

**Problem:** If Claude fails quickly (within 2 minutes) 3 times in a row, something is fundamentally broken. Don't waste time retrying - skip to next focus area or abort.

**Step 1: Add fast-fail config**

In `scripts/ralph/ralph-config.json`, add to `autonomy` section:

```json
"autonomy": {
    "onUncertainty": "document_and_continue",
    "maxConsecutiveErrors": 5,
    "maxConsecutiveTimeouts": 3,
    "autoCreateSprint": true,
    "canModifyConfig": true,
    "canModifyClients": true,
    "canCreateTestProjects": true,
    "fastFail": {
        "enabled": true,
        "thresholdMinutes": 2,
        "maxConsecutiveFastFails": 3,
        "action": "skip_focus_area"
    }
}
```

**Step 2: Add fast-fail tracking variables**

After the `$consecutiveErrors` declarations (around line 1037), add:

```powershell
$sprintIterations = 0
$consecutiveErrors = 0
$consecutiveTimeouts = 0
$consecutiveFastFails = 0  # NEW: Track iterations that fail within threshold
$fastFailThreshold = if ($Config.autonomy.fastFail.thresholdMinutes) {
    $Config.autonomy.fastFail.thresholdMinutes
} else { 2 }
$maxFastFails = if ($Config.autonomy.fastFail.maxConsecutiveFastFails) {
    $Config.autonomy.fastFail.maxConsecutiveFastFails
} else { 3 }
```

**Step 3: Update Invoke-Iteration to return duration**

The function already returns `Duration` in the result. No change needed.

**Step 4: Add fast-fail detection in main loop**

After the result handling (around line 1351, after `if ($result.Success)`), update the else block:

```powershell
if ($result.Success) {
    $consecutiveErrors = 0
    $consecutiveTimeouts = 0
    $consecutiveFastFails = 0  # Reset on success
    # ... existing success handling
} else {
    $consecutiveErrors++

    # NEW: Fast-fail detection
    if ($result.Duration -lt $fastFailThreshold -and -not $result.Timeout) {
        $consecutiveFastFails++
        Write-Log "  FAST FAIL: Iteration failed in $([math]::Round($result.Duration, 1)) min ($consecutiveFastFails/$maxFastFails)" -Color Magenta

        if ($consecutiveFastFails -ge $maxFastFails) {
            Write-Log ""
            Write-Log "  FAST-FAIL LIMIT REACHED" -Color Red
            Write-Log "  $maxFastFails consecutive failures within $fastFailThreshold minutes" -Color Red
            Write-Log "  This suggests a fundamental issue (missing dependencies, bad prompt, etc.)" -Color Yellow

            if ($script:QueueMode) {
                # Skip to next focus area
                Write-Log "  Queue: Skipping '$FocusArea' due to fast-fails" -Color Yellow
                $nextFocus = Advance-Queue -Completed $false -Reason "fast_fail_limit"
                if ($nextFocus) {
                    Write-Log "  Queue: Moving to '$nextFocus'..." -Color Cyan
                    $FocusArea = $nextFocus
                    $created = New-Sprint -Focus $FocusArea
                    if ($created) {
                        $sprintIterations = 0
                        $consecutiveErrors = 0
                        $consecutiveTimeouts = 0
                        $consecutiveFastFails = 0
                        Reset-RetryCounters
                        continue
                    }
                } else {
                    Write-LogSection "QUEUE EXHAUSTED (fast-fail)"
                    $q = Get-Queue
                    Write-Log "  Completed: $($q.completedAreas -join ', ')" -Color Green
                    Write-Log "  Skipped: $(($q.skippedAreas | ForEach-Object { $_.area }) -join ', ')" -Color Yellow
                    Reset-Queue
                    Write-SessionReport -ExitReason "FAST_FAIL_EXHAUSTED" -FinalStatus $status
                    exit 1
                }
            } else {
                # Non-queue mode: exit
                Write-SessionReport -ExitReason "FAST_FAIL_LIMIT" -FinalStatus $status
                exit 1
            }
        }
    } else {
        # Reset fast-fail counter if iteration took normal time
        $consecutiveFastFails = 0
    }

    # ... existing consecutive error handling
```

**Step 5: Add fast-fail info to session report**

In the `Write-SessionReport` function, add to the summary object:

```powershell
$summary = @{
    # ... existing fields
    fast_fails = $consecutiveFastFails
    fast_fail_threshold_min = $fastFailThreshold
}
```

**Step 6: Test fast-fail detection**

This is hard to test directly, but you can verify the logic:

```powershell
# Verify config loaded
.\scripts\ralph\ralph.ps1 -DryRun
# Check that fastFail settings appear in startup
```

**Step 7: Commit**

```bash
git add scripts/ralph/ralph-config.json scripts/ralph/ralph.ps1
git commit -m "feat(ralph): add fast-fail detection for stuck iterations"
```

---

## Summary

| Task | Description | Estimated Effort |
|------|-------------|------------------|
| 1 | Claude CLI path configuration | 5 min |
| 2 | Startup environment validation | 5 min |
| 3 | Use resolved path in process calls | 3 min |
| 4 | Quick resume mode (-Resume flag) | 5 min |
| 5 | Streamlined sprint selection UI | 5 min |
| 6 | Startup progress indicators | 3 min |
| 7 | Convenience batch files (start, resume) | 3 min |
| 8 | Queue state persistence for crash recovery | 8 min |
| 9 | Queue mode batch file | 2 min |
| 10 | Fast-fail detection (3 fails < 2min = skip) | 8 min |

**Total:** ~47 minutes

## Testing Checklist

After all tasks:

- [ ] `.\scripts\ralph\ralph.ps1 -DryRun` shows Claude path and version
- [ ] Starting without Claude in PATH shows clear error message
- [ ] `-Resume` flag continues last sprint without prompts
- [ ] Sprint conflict shows streamlined [C]/[N]/[Q] menu
- [ ] `resume.bat` works from any directory
- [ ] Lock file warning appears if stale locks exist
- [ ] `queue.bat` starts queue mode correctly
- [ ] Killing Ralph mid-queue, then restarting shows "INTERRUPTED QUEUE DETECTED"
- [ ] Fast-fail detection triggers after 3 quick failures
- [ ] Fast-fail in queue mode skips to next focus area
