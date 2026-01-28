# Ralph Loop Unified Launcher
# Combines mode selection + grouped focus area selection
# Usage: .\launcher.ps1 [-Mode <mode>] [-FocusArea <area>]
#
# Modes: standard, trueauto, resume, smartqueue, ralphschoice, ralphschoiceauto, status, morning, logs, recovery, watch

param(
    [ValidateSet("standard", "trueauto", "resume", "smartqueue", "ralphschoice", "ralphschoiceauto", "status", "morning", "logs", "recovery", "watch")]
    [string]$Mode,
    [string]$FocusArea
)

$script:RalphDir = $PSScriptRoot
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $script:RalphDir)
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"

# Load config
function Get-RalphConfig {
    if (Test-Path $script:ConfigFile) {
        return Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
    }
    return $null
}

function Show-Banner {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host "     `"Sleep! That's where I'm a Viking!`" - Ralph" -ForegroundColor Magenta
    Write-Host "     Unified Launcher" -ForegroundColor Magenta
    Write-Host "  =====================================================" -ForegroundColor Magenta
}

function Show-ModeSelection {
    Write-Host ""
    Write-Host "  MODE SELECTION" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    [1] Standard        Work through stories, pause on sprint complete" -ForegroundColor White
    Write-Host "    [2] TrueAuto        Continuous improvement, auto-generate new sprints" -ForegroundColor White
    Write-Host "    [3] Resume          Continue where you left off" -ForegroundColor White
    Write-Host "    [4] Smart Queue     Describe what you want, Ralph picks focus areas" -ForegroundColor Green
    Write-Host ""
    Write-Host "    [A] Ralph's Choice      Ralph decides focus areas (confirm each)" -ForegroundColor Magenta
    Write-Host "    [Z] Ralph's Choice Auto Ralph decides (fully autonomous)" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "    [5] Status          Quick status check" -ForegroundColor DarkGray
    Write-Host "    [6] Morning         Morning check-in with commits" -ForegroundColor DarkGray
    Write-Host "    [7] Logs            View logs and reports" -ForegroundColor DarkGray
    Write-Host "    [8] Recovery        Emergency stop and recovery" -ForegroundColor DarkGray
    Write-Host "    [9] Watch Only      Just the watch dashboard" -ForegroundColor DarkGray
    Write-Host ""

    $selection = Read-Host "  Enter mode (1-9, A, Z)"

    switch -Regex ($selection) {
        "^1$" { return "standard" }
        "^2$" { return "trueauto" }
        "^3$" { return "resume" }
        "^4$" { return "smartqueue" }
        "^[Aa]$" { return "ralphschoice" }
        "^[Zz]$" { return "ralphschoiceauto" }
        "^5$" { return "status" }
        "^6$" { return "morning" }
        "^7$" { return "logs" }
        "^8$" { return "recovery" }
        "^9$" { return "watch" }
        default { return "standard" }
    }
}

function Get-EmojiForCategory {
    param([string]$CategoryId)

    # Map category IDs to display emojis (using simple chars for PS compatibility)
    $emojiMap = @{
        "core"         = "[C]"
        "acquisition"  = "[A]"
        "processing"   = "[P]"
        "output"       = "[O]"
        "intelligence" = "[I]"
        "meta"         = "[M]"
    }

    $emoji = $emojiMap[$CategoryId]
    if ($emoji) { return $emoji }
    return "[?]"
}

function Show-CategorizedFocusAreaSelection {
    $config = Get-RalphConfig
    if (-not $config) {
        Write-Host "  ERROR: Could not load ralph-config.json" -ForegroundColor Red
        return $null
    }

    Write-Host ""
    Write-Host "  FOCUS AREA SELECTION" -ForegroundColor Yellow

    $areaList = @()
    $categoryOrder = if ($config.categoryOrder) { $config.categoryOrder } else { @("core", "acquisition", "processing", "output", "intelligence", "meta") }

    foreach ($catId in $categoryOrder) {
        $category = $config.focusAreaCategories.$catId
        if (-not $category) { continue }

        Write-Host ""
        $prefix = Get-EmojiForCategory -CategoryId $catId
        Write-Host "  $prefix $($category.name.ToUpper())" -ForegroundColor Cyan

        # Get areas in this category
        $categoryAreas = $config.focusAreas | Where-Object { $_.category -eq $catId }
        foreach ($area in $categoryAreas) {
            $areaList += $area.id
            $num = "[$($areaList.Count)]".PadRight(5)
            $id = $area.id.PadRight(20)
            Write-Host "    $num $id $($area.description)" -ForegroundColor White
        }
    }

    Write-Host ""
    $selection = Read-Host "  Enter focus area (1-$($areaList.Count)), or press Enter to use current"

    if ([string]::IsNullOrWhiteSpace($selection)) {
        return $null  # Use current/default
    }

    $index = 0
    if ([int]::TryParse($selection, [ref]$index)) {
        $index--
        if ($index -ge 0 -and $index -lt $areaList.Count) {
            return $areaList[$index]
        }
    }
    Write-Host "  Invalid selection: $selection" -ForegroundColor Yellow
    return $null
}

function Start-RalphLoop {
    param(
        [string]$SelectedMode,
        [string]$Focus
    )

    $ralphScript = Join-Path $script:RalphDir "ralph.ps1"
    $watchScript = Join-Path $script:RalphDir "watch.ps1"
    $ralphArgs = @()

    switch ($SelectedMode) {
        "trueauto" {
            $ralphArgs += "-TrueAuto"
            $ralphArgs += "-SkipPlanApproval"
        }
        "resume" {
            $ralphArgs += "-Resume"
        }
        "smartqueue" {
            Start-SmartQueue
            return
        }
        "ralphschoice" {
            $ralphArgs += "-RalphsChoice"
        }
        "ralphschoiceauto" {
            $ralphArgs += "-RalphsChoiceAuto"
        }
        "status" {
            $statusScript = Join-Path $script:RalphDir "status.ps1"
            if (Test-Path $statusScript) {
                & $statusScript
            } else {
                Write-Host "  status.ps1 not found" -ForegroundColor Yellow
            }
            Read-Host "`n  Press Enter to continue"
            return
        }
        "morning" {
            Show-MorningCheckin
            return
        }
        "logs" {
            Show-LogViewer
            return
        }
        "recovery" {
            Show-EmergencyRecovery
            return
        }
        "watch" {
            Write-Host ""
            Write-Host "  Starting Watch Dashboard only..." -ForegroundColor Green
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$script:ProjectRoot'; .\scripts\ralph\watch.ps1"
            return
        }
    }

    if ($Focus) {
        $ralphArgs += "-FocusArea"
        $ralphArgs += $Focus
    }

    Write-Host ""
    Write-Host "  Starting Ralph Loop..." -ForegroundColor Green
    if ($ralphArgs.Count -gt 0) {
        Write-Host "  Args: $($ralphArgs -join ' ')" -ForegroundColor DarkGray
    }

    # Escape path for safe interpolation inside single-quoted PowerShell strings
    $escapedPath = $script:ProjectRoot -replace "'", "''"

    # Launch Watch in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$escapedPath'; .\scripts\ralph\watch.ps1"

    # Small delay to let watch window open
    Start-Sleep -Milliseconds 500

    # Launch Ralph in new window
    $argString = $ralphArgs -join ' '
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$escapedPath'; .\scripts\ralph\ralph.ps1 $argString"
}

function Show-MorningCheckin {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "     `"I'm learnding!`" - Ralph" -ForegroundColor Cyan
    Write-Host "     Morning Check-in" -ForegroundColor Cyan
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host ""

    # Show status
    $statusScript = Join-Path $script:RalphDir "status.ps1"
    if (Test-Path $statusScript) {
        & $statusScript
    }

    Write-Host ""
    Write-Host "  Recent commits:" -ForegroundColor Yellow
    Set-Location $script:ProjectRoot
    git log --oneline -8 2>$null

    Write-Host ""
    $next = Read-Host "  [1] Standard  [2] TrueAuto  [3] Resume  [Enter] Exit"
    switch ($next) {
        "1" {
            $focus = Show-CategorizedFocusAreaSelection
            Start-RalphLoop -SelectedMode "standard" -Focus $focus
        }
        "2" {
            $focus = Show-CategorizedFocusAreaSelection
            Start-RalphLoop -SelectedMode "trueauto" -Focus $focus
        }
        "3" {
            Start-RalphLoop -SelectedMode "resume"
        }
    }
}

function Show-LogViewer {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Red
    Write-Host "     `"It tastes like burning!`" - Ralph" -ForegroundColor Red
    Write-Host "     Log Viewer" -ForegroundColor Red
    Write-Host "  =====================================================" -ForegroundColor Red
    Write-Host ""

    $logsDir = Join-Path $script:RalphDir "logs"

    if (-not (Test-Path $logsDir)) {
        Write-Host "  No logs directory found." -ForegroundColor Yellow
        Read-Host "`n  Press Enter to continue"
        return
    }

    $sessions = Get-ChildItem $logsDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending

    if ($sessions.Count -eq 0) {
        Write-Host "  No log sessions found." -ForegroundColor Yellow
        Read-Host "`n  Press Enter to continue"
        return
    }

    $latestSession = $sessions[0]
    Write-Host "  Latest session: $($latestSession.Name)" -ForegroundColor White
    Write-Host ""

    # Show REPORT.md if exists
    $reportPath = Join-Path $latestSession.FullName "REPORT.md"
    if (Test-Path $reportPath) {
        Write-Host "  === REPORT.md ===" -ForegroundColor Cyan
        Get-Content $reportPath | Select-Object -First 50
        if ((Get-Content $reportPath).Count -gt 50) {
            Write-Host "  ... (truncated)" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  No REPORT.md found in this session." -ForegroundColor Yellow

        # List other files
        Write-Host ""
        Write-Host "  Files in session:" -ForegroundColor White
        Get-ChildItem $latestSession.FullName | ForEach-Object {
            Write-Host "    - $($_.Name)" -ForegroundColor DarkGray
        }
    }

    Write-Host ""
    Write-Host "  Other sessions:" -ForegroundColor Yellow
    $sessions | Select-Object -Skip 1 -First 5 | ForEach-Object {
        Write-Host "    - $($_.Name)" -ForegroundColor DarkGray
    }

    Read-Host "`n  Press Enter to continue"
}

function Show-EmergencyRecovery {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host "     `"I bent my Wookie!`" - Ralph" -ForegroundColor Magenta
    Write-Host "     Emergency Recovery" -ForegroundColor Magenta
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host ""

    Set-Location $script:ProjectRoot

    Write-Host "  Recent commits (rollback targets):" -ForegroundColor Yellow
    git log --oneline -10 2>$null

    Write-Host ""
    Write-Host "  Uncommitted changes:" -ForegroundColor Yellow
    git status --short 2>$null

    Write-Host ""
    Write-Host "  Current branch:" -ForegroundColor Yellow
    git branch --show-current 2>$null

    Write-Host ""
    Write-Host "  Recovery commands:" -ForegroundColor Cyan
    Write-Host "    git revert HEAD             - Undo last commit (safe)" -ForegroundColor White
    Write-Host "    git reset --soft HEAD~1     - Undo commit, keep changes" -ForegroundColor White
    Write-Host "    git stash                   - Stash uncommitted changes" -ForegroundColor White
    Write-Host "    git checkout -- .           - Discard all changes (DANGER)" -ForegroundColor Red

    Write-Host ""
    Write-Host "  Ralph files:" -ForegroundColor Cyan
    Write-Host "    prd.json      - Current sprint definition" -ForegroundColor White
    Write-Host "    queue.json    - Focus area queue" -ForegroundColor White
    Write-Host "    progress.txt  - Progress tracking" -ForegroundColor White

    # Check graceful stop status
    $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
    $hasSignal = Test-Path $signalFile

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "  Graceful Stop Options:" -ForegroundColor Cyan
    Write-Host "  =====================================================" -ForegroundColor Cyan
    if ($hasSignal) {
        Write-Host "    [G] Cancel graceful stop (currently PENDING)" -ForegroundColor Yellow
    }
    else {
        Write-Host "    [G] Request graceful stop (safe stop after sprint)" -ForegroundColor White
    }
    Write-Host "    [Enter] Return to menu" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "  Selection"

    if ($choice -eq "G" -or $choice -eq "g") {
        $gracefulStopScript = Join-Path $script:RalphDir "graceful-stop.ps1"
        if (Test-Path $gracefulStopScript) {
            if ($hasSignal) {
                & $gracefulStopScript -Cancel
            }
            else {
                & $gracefulStopScript
            }
        }
        else {
            # Inline fallback if script not found
            if ($hasSignal) {
                Remove-Item $signalFile -Force -ErrorAction SilentlyContinue
                Write-Host "  Graceful stop cancelled" -ForegroundColor Green
            }
            else {
                @{ requestedAt = (Get-Date).ToString("o"); reason = "User requested via recovery menu" } |
                    ConvertTo-Json | Set-Content $signalFile -Encoding UTF8
                Write-Host "  Graceful stop requested - Ralph will stop after current sprint" -ForegroundColor Cyan
            }
        }
        Read-Host "`n  Press Enter to continue"
    }
}

function Start-SmartQueue {
    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Green
    Write-Host "     Smart Queue Creation" -ForegroundColor Green
    Write-Host "  =====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Describe what you want to work on (2-3 sentences):" -ForegroundColor Yellow
    Write-Host ""

    $userInput = Read-Host "  >"

    if ([string]::IsNullOrWhiteSpace($userInput)) {
        Write-Host "  No input provided. Returning to menu." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "  Analyzing your request..." -ForegroundColor DarkGray

    # Build the analysis prompt
    $analysisPrompt = @"
Analyze this user request and identify which focus areas from the Ralph Loop system are relevant.

User request: "$userInput"

Available focus areas:
- pipeline: Core pipeline stages and orchestration
- config: Config management and validation
- rate-limiting: YouTube download resilience: cookie rotation, circuit breaker, impersonation escalation, extractor-args bypass
- caption: Caption-first mode and transcript handling
- download: Core download logic, yt-dlp integration, browser impersonation (curl_cffi), TLS fingerprint bypass
- quality: Improve matching accuracy and confidence
- speed: Optimize pipeline speed and caching
- compilation: Keyword compilation and montage features
- otio: OTIO/EDL/XML timeline generation
- agents: Self-healing agents and error recovery
- client-learning: Cross-project learning and presets
- testing: General test coverage and reliability
- unit-tests: Isolated function tests with mocking
- integration-tests: Component interaction tests
- mutation-tests: Verify test quality catches changes
- documentation: API docs, README, inline comments
- ux: Prompts, progress feedback, error messages

Respond with ONLY valid JSON (no markdown, no explanation):
{"focusAreas":["area1","area2"],"reasoning":{"area1":"brief reason","area2":"brief reason"},"storyContext":"summarized intent"}
"@

    try {
        # Invoke Claude to analyze
        $claudeOutput = & claude --print $analysisPrompt 2>$null

        # Try to parse JSON from output
        $result = $null
        if ($claudeOutput) {
            # Extract JSON from output (Claude might add extra text)
            $jsonMatch = [regex]::Match($claudeOutput, '\{[\s\S]*"focusAreas"[\s\S]*\}')
            if ($jsonMatch.Success) {
                $result = $jsonMatch.Value | ConvertFrom-Json
            } else {
                # Try parsing full output
                $result = $claudeOutput | ConvertFrom-Json -ErrorAction SilentlyContinue
            }
        }

        if ($result -and $result.focusAreas -and $result.focusAreas.Count -gt 0) {
            Write-Host ""
            Write-Host "  Ralph identified these focus areas:" -ForegroundColor Cyan
            foreach ($area in $result.focusAreas) {
                $reason = if ($result.reasoning.$area) { $result.reasoning.$area } else { "matches request" }
                Write-Host "    + $($area.PadRight(20)) ($reason)" -ForegroundColor White
            }

            # Create queue.json with identified focus areas
            $queue = @{
                focusAreas = @($result.focusAreas | ForEach-Object { @{ id = $_; completed = $false } })
                sessionId = Get-Date -Format "yyyy-MM-dd_HHmmss"
                createdAt = (Get-Date).ToString("o")
                interviewContext = "Smart Queue: $userInput"
                interviewDetails = if ($result.storyContext) { $result.storyContext } else { $userInput }
            }

            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            Write-Host ""
            Write-Host "  Queue created with $($result.focusAreas.Count) focus areas." -ForegroundColor Green
            Write-Host "  Your input will be used as context for story generation." -ForegroundColor DarkGray
            Write-Host ""

            # Start the queue in standard mode
            Start-RalphLoop -SelectedMode "standard"
        } else {
            Write-Host "  Could not identify focus areas from your request." -ForegroundColor Yellow
            Write-Host "  Please try being more specific, or select focus areas manually." -ForegroundColor Yellow

            $focus = Show-CategorizedFocusAreaSelection
            if ($focus) {
                Start-RalphLoop -SelectedMode "standard" -Focus $focus
            }
        }
    }
    catch {
        Write-Host "  Error analyzing request: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  Falling back to manual selection..." -ForegroundColor Yellow

        $focus = Show-CategorizedFocusAreaSelection
        if ($focus) {
            Start-RalphLoop -SelectedMode "standard" -Focus $focus
        }
    }
}

# ============================================================================
# MAIN
# ============================================================================

Show-Banner

# Determine mode
$selectedMode = if ($Mode) { $Mode } else { Show-ModeSelection }

# For modes that don't need focus area selection, handle directly
# Ralph's Choice modes skip focus area selection - Ralph picks the focus area
if ($selectedMode -in @("status", "morning", "logs", "recovery", "watch", "smartqueue", "resume", "ralphschoice", "ralphschoiceauto")) {
    Start-RalphLoop -SelectedMode $selectedMode
    exit 0
}

# For standard/trueauto, show focus area selection
$selectedFocus = if ($FocusArea) { $FocusArea } else { Show-CategorizedFocusAreaSelection }

Start-RalphLoop -SelectedMode $selectedMode -Focus $selectedFocus
