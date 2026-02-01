# Ralph Loop - Unified Entry Point
# Usage: .\scripts\ralph\interview.ps1 [options]
#
# This is the ONLY entry point for Ralph. All other launchers have been removed.
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1
#
# Modes:
#   (default)          Interactive interview flow
#   -Mode <mode>       Skip interview, use specified mode
#   -FocusArea <area>  Skip interview, work on specific area
#   -Status            Quick status check and exit
#   -Watch             Open watch dashboard and exit
#   -Stop              Request graceful stop and exit
#   -Logs              View logs and exit
#   -Recovery          Emergency recovery menu
#   -Morning           Morning check-in with commits

param(
    [switch]$Resume,
    [switch]$NoLaunch,      # Skip launching Ralph/Watch windows (used when called from Ralph)

    # Mode selection (skips interview if provided)
    [ValidateSet("standard", "trueauto", "ralphschoice", "ralphschoiceauto", "overnight", "smartqueue")]
    [string]$Mode,

    # Direct focus area (skips interview if provided)
    [string]$FocusArea,
    [string[]]$FocusAreas,  # Multiple areas for overnight mode

    # Utility commands (execute and exit)
    [switch]$Status,
    [switch]$Watch,
    [switch]$Stop,
    [switch]$Logs,
    [switch]$Recovery,
    [switch]$Morning,
    [switch]$Queue,         # Skip work type, go directly to queue mode

    # Overnight options
    [int]$MaxHours = 12
)

# Set up paths
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "state\queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"

# Load domain modules for shared state access
$script:LibPath = Join-Path $script:RalphDir 'lib'
if (Test-Path $script:LibPath) {
    . "$script:LibPath\sprint.ps1"   # Read-JsonFile, Save-StateFile
    . "$script:LibPath\queue.ps1"    # Get-Queue, Save-Queue
    . "$script:LibPath\interview.ps1" # Improve-InterviewContext, Get-SuggestedAreas
}

# Load config if it exists
$config = Read-JsonFile -Path $script:ConfigFile

# Also load paths module if available (for state directory paths)
$pathsModule = Join-Path $script:LibPath "paths.ps1"
if (Test-Path $pathsModule) {
    . $pathsModule
    Initialize-RalphPaths -RalphDir $script:RalphDir | Out-Null
}

# ============================================================================
# UTILITY COMMAND HANDLERS (Execute and Exit)
# ============================================================================
# These commands are executed immediately and exit - no interview flow

if ($Status) {
    $statusScript = Join-Path $script:RalphDir "status.ps1"
    if (Test-Path $statusScript) {
        & $statusScript
    } else {
        Write-Host "  status.ps1 not found" -ForegroundColor Yellow
    }
    exit 0
}

if ($Watch) {
    Write-Host ""
    Write-Host "  Starting Watch Dashboard..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$script:ProjectRoot'; .\scripts\ralph\watch.ps1"
    exit 0
}

if ($Stop) {
    $gracefulStopScript = Join-Path $script:RalphDir "graceful-stop.ps1"
    if (Test-Path $gracefulStopScript) {
        & $gracefulStopScript
    } else {
        # Inline fallback
        $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
        @{ requestedAt = (Get-Date).ToString("o"); reason = "User requested via interview.ps1 -Stop" } |
            ConvertTo-Json | Set-Content $signalFile -Encoding UTF8
        Write-Host "  Graceful stop requested - Ralph will stop after current sprint" -ForegroundColor Cyan
    }
    exit 0
}

# ============================================================================
# MODE SELECTION UI
# ============================================================================

function Show-ModeSelection {
    <#
    .SYNOPSIS
        Asks user about autonomous behavior in a conversational way
    .RETURNS
        Mode string: standard, trueauto
    #>
    Write-Host ""
    Write-Host "  Should Ralph work autonomously?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    [Y] Yes - let Ralph decide and keep going" -ForegroundColor White
    Write-Host "    [N] No  - pause after each sprint for review (default)" -ForegroundColor White
    Write-Host ""

    $selection = Read-Host "  "

    if ($selection -match "^[Yy]") {
        return "trueauto"
    }
    return "standard"
}

function Get-EmojiForCategory {
    param([string]$CategoryId)

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
    <#
    .SYNOPSIS
        Shows focus areas organized by category (from launcher.ps1)
    .RETURNS
        Selected focus area ID, or $null for default
    #>
    if (-not $config -or -not $config.focusAreas) {
        Write-Host "  ERROR: Could not load focus areas from config" -ForegroundColor Red
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

        $categoryAreas = $config.focusAreas | Where-Object { $_.category -eq $catId }
        foreach ($area in $categoryAreas) {
            $areaList += $area.id
            $num = "[$($areaList.Count)]".PadRight(5)
            $id = $area.id.PadRight(20)
            Write-Host "    $num $id $($area.description)" -ForegroundColor White
        }
    }

    Write-Host ""
    $selection = Read-Host "  Enter focus area (1-$($areaList.Count)), or press Enter for Ralph's Choice"

    if ([string]::IsNullOrWhiteSpace($selection)) {
        return $null
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

# ============================================================================
# UTILITY MENUS (Logs, Recovery, Morning)
# ============================================================================

function Show-LogViewer {
    <#
    .SYNOPSIS
        Display recent logs and session reports (from launcher.ps1)
    #>
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

    $reportPath = Join-Path $latestSession.FullName "REPORT.md"
    if (Test-Path $reportPath) {
        Write-Host "  === REPORT.md ===" -ForegroundColor Cyan
        Get-Content $reportPath | Select-Object -First 50
        if ((Get-Content $reportPath).Count -gt 50) {
            Write-Host "  ... (truncated)" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  No REPORT.md found in this session." -ForegroundColor Yellow
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
    <#
    .SYNOPSIS
        Emergency recovery menu with git status and graceful stop options (from launcher.ps1)
    #>
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

    $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
    $hasSignal = Test-Path $signalFile

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Cyan
    Write-Host "  Graceful Stop Options:" -ForegroundColor Cyan
    Write-Host "  =====================================================" -ForegroundColor Cyan
    if ($hasSignal) {
        Write-Host "    [G] Cancel graceful stop (currently PENDING)" -ForegroundColor Yellow
    } else {
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
            } else {
                & $gracefulStopScript
            }
        } else {
            if ($hasSignal) {
                Remove-Item $signalFile -Force -ErrorAction SilentlyContinue
                Write-Host "  Graceful stop cancelled" -ForegroundColor Green
            } else {
                @{ requestedAt = (Get-Date).ToString("o"); reason = "User requested via recovery menu" } |
                    ConvertTo-Json | Set-Content $signalFile -Encoding UTF8
                Write-Host "  Graceful stop requested - Ralph will stop after current sprint" -ForegroundColor Cyan
            }
        }
        Read-Host "`n  Press Enter to continue"
    }
}

function Show-MorningCheckin {
    <#
    .SYNOPSIS
        Morning check-in with status and recent commits (from launcher.ps1)
    #>
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
    $next = Read-Host "  Ready to work? [Y]es / [N]o"
    if ($next -match "^[Yy]") {
        $focus = Show-CategorizedFocusAreaSelection
        $autonomous = Read-Host "  Work autonomously? [Y]es / [N]o"
        $mode = if ($autonomous -match "^[Yy]") { "trueauto" } else { "standard" }
        return @{ mode = $mode; focusArea = $focus }
    }
    return $null
}

# Handle -Logs, -Recovery, -Morning as utility commands
if ($Logs) {
    Show-LogViewer
    exit 0
}

if ($Recovery) {
    Show-EmergencyRecovery
    exit 0
}

if ($Morning) {
    $result = Show-MorningCheckin
    if ($result) {
        # Continue to Ralph spawn with selected mode
        $Mode = $result.mode
        $FocusArea = $result.focusArea
        # Fall through to main flow
    } else {
        exit 0
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Test-FocusAreaExists {
    <#
    .SYNOPSIS
        Validates if a focus area ID exists in the config
    .PARAMETER AreaId
        The focus area ID to validate
    .RETURNS
        $true if valid or can't validate, $false if definitely invalid
    #>
    param([string]$AreaId)

    if (-not $config -or -not $config.focusAreas) {
        return $true  # Can't validate, assume valid
    }

    $validAreas = $config.focusAreas | ForEach-Object {
        if ($_.id) { $_.id } else { $_ }
    }
    return $AreaId -in $validAreas
}

# Write-InterviewLog is provided by lib/interview.ps1

function Write-RalphHeader {
    Write-Host ""
    Write-Host "  Ralph Loop" -ForegroundColor Yellow
    Write-Host ""
}

# Get-ExistingContext and Show-ResumePrompt have been moved to lib/queue.ps1
# as Test-ExistingQueue, Get-QueueSummary, and Show-QueueContinuationPrompt
# for shared use across launcher.ps1 and interview.ps1

# ============================================================================
# QUESTION FUNCTIONS
# ============================================================================

function Ask-WorkType {
    <#
    .SYNOPSIS
        Asks user what kind of work they want to do
    .RETURNS
        "bug", "feature", "improvement", "client", or "queue"
    #>

    Write-Host "  What do you want to work on?" -ForegroundColor Cyan
    Write-Host "  (bug, feature, improvement, client feedback, or just pick areas)" -ForegroundColor DarkGray
    Write-Host ""

    $response = Read-Host "  "

    # Parse natural language input
    switch -Regex ($response.ToLower()) {
        "bug|fix|broken|error|crash" { return "bug" }
        "feature|add|new|create" { return "feature" }
        "client|theresa|stu|feedback" { return "client" }
        "area|pick|queue|select" { return "queue" }
        "^$" { return "queue" }  # Empty = go to area selection
        default { return "improvement" }
    }
}

function Ask-Details {
    <#
    .SYNOPSIS
        Asks for details based on work type
    .PARAMETER WorkType
        The type of work (bug, feature, improvement, client)
    .RETURNS
        User's text input describing the work
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$WorkType
    )

    $prompts = @{
        "bug" = "Describe the bug (or paste error message)"
        "feature" = "What feature do you want?"
        "improvement" = "What should be improved?"
        "client" = "What was the client feedback?"
    }

    $prompt = $prompts[$WorkType]
    if (-not $prompt) {
        $prompt = "Describe what you need"
    }

    Write-Host ""
    Write-Host "  $prompt" -ForegroundColor Cyan
    Write-Host ""

    $response = Read-Host "  "
    return $response
}

function Ask-Area {
    <#
    .SYNOPSIS
        Asks which focus area to work on
    .PARAMETER WorkType
        The type of work
    .PARAMETER Details
        The work details (for context)
    .RETURNS
        User's input (can be empty string to let Ralph decide)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$WorkType,
        [Parameter(Mandatory=$true)]
        [string]$Details
    )

    Write-Host ""
    Write-Host "  Which area? (optional - press Enter to let Ralph decide)" -ForegroundColor Cyan

    # Show available areas from config if available
    if ($config -and $config.focusAreas) {
        Write-Host ""
        Write-Host "  Available areas:" -ForegroundColor Gray
        foreach ($area in $config.focusAreas) {
            $areaId = if ($area.id) { $area.id } else { $area }
            Write-Host "    - $areaId" -ForegroundColor Gray
        }
    }

    Write-Host ""
    $response = Read-Host "  "

    # If user pressed Enter without input, show Ralph's Choice preview
    if ([string]::IsNullOrWhiteSpace($response)) {
        Show-RalphsChoicePreview | Out-Null
        Write-Host "  (Ralph will use these to pick the best focus area)" -ForegroundColor DarkGray
    }

    return $response
}

function Ask-Client {
    <#
    .SYNOPSIS
        Asks which client this work is for
    .RETURNS
        User's input (can be empty string to skip)
    #>

    Write-Host ""
    Write-Host "  Which client? (optional - press Enter to skip)" -ForegroundColor Cyan

    # Show known clients if clients.json exists
    $clientsFile = Join-Path $script:RalphDir "clients.json"
    if (Test-Path $clientsFile) {
        try {
            $clientsData = Read-JsonFile -Path $clientsFile
            if ($clientsData.clients) {
                $clientNames = @()
                $clientsData.clients.PSObject.Properties | ForEach-Object {
                    if ($_.Name -ne "default") {
                        $clientNames += $_.Name
                    }
                }
                if ($clientNames.Count -gt 0) {
                    Write-Host ""
                    Write-Host "  Known clients: $($clientNames -join ', ')" -ForegroundColor Gray
                }
            }
        }
        catch {
            # Ignore parse errors
        }
    }

    Write-Host ""
    $response = Read-Host "  "
    return $response
}

function Ask-Priority {
    <#
    .SYNOPSIS
        Asks for work priority
    .RETURNS
        "high", "normal", or "low"
    #>

    Write-Host ""
    Write-Host "  Priority?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [H] High" -ForegroundColor White
    Write-Host "  [N] Normal (default)" -ForegroundColor White
    Write-Host "  [L] Low" -ForegroundColor White
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Hh]" { return "high" }
        "^[Ll]" { return "low" }
        default { return "normal" }
    }
}

# ============================================================================
# INTERVIEW FLOW CONTROLLER
# ============================================================================

function Start-Interview {
    <#
    .SYNOPSIS
        Orchestrates the interview flow with dynamic questions
    .DESCRIPTION
        Always asks: work type and details
        Dynamically asks based on context:
        - Area: if details are vague (< 20 chars)
        - Client: if work type is "client" OR details mention client patterns
        - Priority: if work type is "bug" OR details mention urgency
    .RETURNS
        Hashtable with: workType, details, area, client, priority, timestamp
    #>

    # Initialize context with defaults
    $context = @{
        workType  = ""
        details   = ""
        area      = ""
        client    = ""
        priority  = "normal"
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }

    # ---- ALWAYS ASK: Work Type ----
    $context.workType = Ask-WorkType
    Write-InterviewLog "Work type selected: $($context.workType)"

    # ---- ALWAYS ASK: Details ----
    $context.details = Ask-Details -WorkType $context.workType
    Write-InterviewLog "Details provided: $($context.details)"

    # ---- DYNAMIC: Area (if vague description) ----
    if ($context.details.Length -lt 20) {
        $context.area = Ask-Area -WorkType $context.workType -Details $context.details
        Write-InterviewLog "Area specified: $($context.area)"
    }

    # ---- DYNAMIC: Client (if client work type or client-related keywords) ----
    $clientPattern = "client|theresa|stu|feedback"
    if ($context.workType -eq "client" -or $context.details -match $clientPattern) {
        $context.client = Ask-Client
        Write-InterviewLog "Client specified: $($context.client)"
    }

    # ---- DYNAMIC: Priority (if bug or urgency keywords) ----
    $urgencyPattern = "urgent|critical|asap|broken"
    if ($context.workType -eq "bug" -or $context.details -match $urgencyPattern) {
        $context.priority = Ask-Priority
        Write-InterviewLog "Priority set: $($context.priority)"
    }

    return $context
}

function Start-QueueMode {
    <#
    .SYNOPSIS
        Direct queue mode - skip interview, pick focus areas directly
    .DESCRIPTION
        Shows all available focus areas with descriptions and lets
        users select which ones to queue one at a time, showing a
        running list of selections.
    .RETURNS
        Array of selected focus area IDs
    #>

    # Load focus areas from config
    $configPath = Join-Path $PSScriptRoot "ralph-config.json"
    $focusConfig = Read-JsonFile -Path $configPath

    $selected = [System.Collections.ArrayList]::new()

    while ($true) {
        # Clear and show header
        Write-Host ""
        Write-Host "  Available focus areas:" -ForegroundColor Cyan
        Write-Host ""

        # Display numbered list with selection status
        $i = 1
        foreach ($area in $focusConfig.focusAreas) {
            $num = $i.ToString().PadLeft(2)
            $isSelected = $selected -contains $area.id

            if ($isSelected) {
                Write-Host "  [$num] " -ForegroundColor DarkGray -NoNewline
                Write-Host "$($area.id)" -ForegroundColor Green -NoNewline
                Write-Host " - $($area.description)" -ForegroundColor DarkGray -NoNewline
                Write-Host " [queued]" -ForegroundColor Green
            } else {
                Write-Host "  [$num] $($area.id)" -ForegroundColor White -NoNewline
                Write-Host " - $($area.description)" -ForegroundColor DarkGray
            }
            $i++
        }

        # Show current queue
        Write-Host ""
        if ($selected.Count -gt 0) {
            Write-Host "  Queue: " -ForegroundColor Yellow -NoNewline
            Write-Host "$($selected -join ' -> ')" -ForegroundColor Green
        } else {
            Write-Host "  Queue: (empty)" -ForegroundColor DarkGray
        }

        Write-Host ""
        Write-Host "  Enter number or name to add/remove, [D]one when finished" -ForegroundColor Yellow
        Write-Host ""

        $response = Read-Host "  "

        # Check for done
        if ($response -match '^[Dd]' -or ($response -eq '' -and $selected.Count -gt 0)) {
            break
        }

        # Skip empty input when queue is empty
        if ($response -eq '') {
            Write-Host "    Add at least one area first" -ForegroundColor Red
            continue
        }

        $input = $response.Trim()

        # Check if it's a number
        if ($input -match '^\d+$') {
            $idx = [int]$input - 1
            if ($idx -ge 0 -and $idx -lt $focusConfig.focusAreas.Count) {
                $areaId = $focusConfig.focusAreas[$idx].id
                if ($selected -contains $areaId) {
                    # Toggle off
                    [void]$selected.Remove($areaId)
                    Write-Host "    Removed: $areaId" -ForegroundColor Yellow
                } else {
                    # Add
                    [void]$selected.Add($areaId)
                    Write-Host "    Added: $areaId" -ForegroundColor Green
                }
            } else {
                Write-Host "    Invalid number: $input" -ForegroundColor Red
            }
        } else {
            # Treat as area name
            $areaId = $input.ToLower()
            $match = $focusConfig.focusAreas | Where-Object { $_.id -eq $areaId }
            if ($match) {
                if ($selected -contains $areaId) {
                    # Toggle off
                    [void]$selected.Remove($areaId)
                    Write-Host "    Removed: $areaId" -ForegroundColor Yellow
                } else {
                    # Add
                    [void]$selected.Add($areaId)
                    Write-Host "    Added: $areaId" -ForegroundColor Green
                }
            } else {
                Write-Host "    Unknown area: $input" -ForegroundColor Red
            }
        }

        Start-Sleep -Milliseconds 300
    }

    if ($selected.Count -eq 0) {
        Write-Host ""
        Write-Host "  No areas selected. Defaulting to 'pipeline'." -ForegroundColor Yellow
        $selected = [System.Collections.ArrayList]@("pipeline")
    }

    Write-InterviewLog "Queue mode selected areas: $($selected -join ', ')"

    return $selected.ToArray()
}

# ============================================================================
# RALPH'S CHOICE SCORING (LIGHTWEIGHT VERSION FOR INTERVIEW)
# ============================================================================

function Get-InterviewSprintHistory {
    <#
    .SYNOPSIS
        Load sprint history for scoring
    #>
    $historyFile = Join-Path $script:RalphDir "sprint_history.json"
    $history = Read-JsonFile -Path $historyFile
    if ($history) { return $history }

    return @{
        totalSprintsCompleted = 0
        focusAreaBreakdown = @{}
        sprints = @()
    }
}

function Get-RalphsChoiceRecommendations {
    <#
    .SYNOPSIS
        Get Ralph's Choice recommendations for interview mode
    .DESCRIPTION
        Lightweight scoring based on neglected areas and category balance.
        Skips git activity analysis for speed.
    .RETURNS
        Array of top 5 focus area IDs sorted by score
    #>

    $sprintHistory = Get-InterviewSprintHistory

    if (-not $config -or -not $config.focusAreas) {
        return @("pipeline", "testing", "quality", "agents", "caption")
    }

    $scores = @()
    $totalSprints = [int]$sprintHistory.totalSprintsCompleted

    foreach ($area in $config.focusAreas) {
        $areaId = $area.id

        # Neglected score (0-1): areas with 0 sprints get 1.0
        $sprintsDone = 0
        if ($sprintHistory.focusAreaBreakdown -and $sprintHistory.focusAreaBreakdown.$areaId) {
            $sprintsDone = [int]$sprintHistory.focusAreaBreakdown.$areaId.sprints
        }

        $neglectedScore = if ($sprintsDone -eq 0) { 1.0 } else {
            # Find days since last sprint
            $daysSince = 14
            $areaSprints = $sprintHistory.sprints | Where-Object { $_.focusArea -eq $areaId } | Sort-Object completedAt -Descending
            if ($areaSprints -and $areaSprints.Count -gt 0) {
                $lastSprint = $areaSprints | Select-Object -First 1
                if ($lastSprint.completedAt) {
                    $daysSince = [math]::Min(((Get-Date) - [datetime]$lastSprint.completedAt).Days, 14)
                }
            }
            $daysSince / 14.0
        }

        # Category balance score (0-1)
        $areaCategory = $area.category
        $categorySprintCount = 0

        if ($areaCategory -and $config.focusAreaCategories.$areaCategory) {
            foreach ($catArea in $config.focusAreaCategories.$areaCategory.areas) {
                if ($sprintHistory.focusAreaBreakdown -and $sprintHistory.focusAreaBreakdown.$catArea) {
                    $categorySprintCount += [int]$sprintHistory.focusAreaBreakdown.$catArea.sprints
                }
            }
        }

        $categoryRatio = if ($totalSprints -gt 0) { $categorySprintCount / $totalSprints } else { 0 }
        $balanceScore = 1.0 - $categoryRatio

        # Combined score (weight neglected higher since we skip git activity)
        $total = ($neglectedScore * 0.6) + ($balanceScore * 0.4)

        $scores += @{
            areaId = $areaId
            total = [math]::Round($total, 2)
            sprintsDone = $sprintsDone
        }
    }

    # Sort by score descending and return top 5 IDs
    $sorted = $scores | Sort-Object -Property total -Descending | Select-Object -First 5
    return @($sorted | ForEach-Object { $_.areaId })
}

function Show-RalphsChoicePreview {
    <#
    .SYNOPSIS
        Show a brief Ralph's Choice preview in interview mode
    #>
    $recommendations = Get-RalphsChoiceRecommendations

    Write-Host ""
    Write-Host "  Ralph's Choice recommendations:" -ForegroundColor Magenta
    $i = 1
    foreach ($areaId in $recommendations) {
        $area = $config.focusAreas | Where-Object { $_.id -eq $areaId } | Select-Object -First 1
        $name = if ($area -and $area.name) { $area.name } else { $areaId }
        Write-Host "    $i. $areaId ($name)" -ForegroundColor White
        $i++
    }
    Write-Host ""

    return $recommendations
}

# ============================================================================
# FOCUS AREA SUGGESTION FUNCTIONS
# ============================================================================

function Get-SuggestedFocusAreas {
    <#
    .SYNOPSIS
        Suggests focus areas based on interview context
    .PARAMETER Context
        The interview context hashtable from Start-Interview
    .PARAMETER UseRalphsChoice
        If true, use Ralph's Choice scoring instead of keyword matching
    .RETURNS
        Array of suggested focus area IDs (max 5)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Context,
        [switch]$UseRalphsChoice
    )

    $suggestions = @()

    # If user specified an area, add it first
    if ($Context.area -and $Context.area.Trim() -ne "") {
        $suggestions += $Context.area.Trim().ToLower()
    }

    # If no area specified and UseRalphsChoice, use Ralph's Choice scoring
    if ($suggestions.Count -eq 0 -and $UseRalphsChoice) {
        Write-Host "  Using Ralph's Choice to find best focus areas..." -ForegroundColor Magenta
        $ralphSuggestions = Get-RalphsChoiceRecommendations
        return $ralphSuggestions
    }

    # Keyword matching from details (lowercase)
    $keywordMap = @{
        "otio|timeline|edl|xml|davinci|resolve" = "otio"
        "download|youtube|yt-dlp|429|rate limit|cookie" = "rate-limiting"
        "impersonate|curl_cffi|tls|fingerprint|bypass|403|bot detect" = "download"
        "match|confidence|score|quality|poor" = "quality"
        "caption|subtitle|srt|transcript" = "caption"
        "config|yaml|setting" = "config"
        "test|coverage|pytest" = "testing"
        "speed|slow|fast|performance|cache" = "speed"
        "client|theresa|stu|preset|learn" = "client-learning"
        "heal|recover|retry|error|fail" = "agents"
        "compile|compilation|topic|keyword" = "compilation"
    }

    $detailsLower = $Context.details.ToLower()

    foreach ($pattern in $keywordMap.Keys) {
        if ($detailsLower -match $pattern) {
            $areaId = $keywordMap[$pattern]
            if ($suggestions -notcontains $areaId) {
                $suggestions += $areaId
            }
        }
    }

    # If < 3 suggestions, use Ralph's Choice to fill in
    if ($suggestions.Count -lt 3) {
        $ralphSuggestions = Get-RalphsChoiceRecommendations
        foreach ($area in $ralphSuggestions) {
            if ($suggestions -notcontains $area -and $suggestions.Count -lt 5) {
                $suggestions += $area
            }
        }
    }

    # Cap at 5 suggestions
    if ($suggestions.Count -gt 5) {
        $suggestions = $suggestions[0..4]
    }

    return $suggestions
}

function Show-Suggestions {
    <#
    .SYNOPSIS
        Displays the suggested focus areas with options
    .PARAMETER Suggestions
        Array of focus area IDs to display
    #>
    param(
        [Parameter(Mandatory=$true)]
        [array]$Suggestions
    )

    Write-Host "  Suggested focus areas:" -ForegroundColor Cyan
    Write-Host ""

    $index = 1
    foreach ($area in $Suggestions) {
        # Try to get a friendly name from config
        $areaName = $area
        if ($config -and $config.focusAreas) {
            foreach ($configArea in $config.focusAreas) {
                $configId = if ($configArea.id) { $configArea.id } else { $configArea }
                if ($configId -eq $area -and $configArea.name) {
                    $areaName = "$area ($($configArea.name))"
                    break
                }
            }
        }
        Write-Host "  [$index] $areaName" -ForegroundColor White
        $index++
    }

    Write-Host ""
    Write-Host "  [A] Approve all" -ForegroundColor Green
    Write-Host "  [1-$($Suggestions.Count)] Remove specific" -ForegroundColor Yellow
    Write-Host "  [+area] Add area (e.g., +testing)" -ForegroundColor Yellow
    Write-Host "  [R] Restart interview" -ForegroundColor Red
    Write-Host ""
}

function Get-ApprovedAreas {
    <#
    .SYNOPSIS
        Interactive loop to let user approve/modify suggested areas
    .PARAMETER Suggestions
        Initial array of suggested focus area IDs
    .RETURNS
        ArrayList of approved focus areas, or $null to restart
    #>
    param(
        [Parameter(Mandatory=$true)]
        [array]$Suggestions
    )

    # Create mutable ArrayList from suggestions
    $approved = [System.Collections.ArrayList]::new()
    foreach ($s in $Suggestions) {
        [void]$approved.Add($s)
    }

    while ($true) {
        Show-Suggestions -Suggestions $approved.ToArray()

        $choice = Read-Host "  Choice"

        # [A] Approve all
        if ($choice -match "^[Aa]$") {
            Write-InterviewLog "User approved areas: $($approved -join ', ')"
            return $approved
        }

        # [R] Restart
        if ($choice -match "^[Rr]$") {
            Write-InterviewLog "User chose to restart interview"
            return $null
        }

        # [+area] Add area
        if ($choice -match "^\+(.+)$") {
            $newArea = $Matches[1].Trim().ToLower()
            if ($approved -notcontains $newArea) {
                # Validate against known focus areas
                if (-not (Test-FocusAreaExists -AreaId $newArea)) {
                    Write-Host "    Warning: '$newArea' is not a known focus area" -ForegroundColor Yellow
                    if ($config -and $config.focusAreas) {
                        $knownAreas = $config.focusAreas | ForEach-Object {
                            if ($_.id) { $_.id } else { $_ }
                        }
                        Write-Host "    Known areas: $($knownAreas -join ', ')" -ForegroundColor DarkGray
                    }
                    $confirm = Read-Host "    Add anyway? [Y]es / [N]o"
                    if ($confirm -notmatch "^[Yy]") {
                        Write-InterviewLog "User declined to add unknown area: $newArea"
                        continue
                    }
                    Write-InterviewLog "User added unknown area (confirmed): $newArea" "WARN"
                }
                [void]$approved.Add($newArea)
                Write-Host "  Added: $newArea" -ForegroundColor Green
                Write-InterviewLog "User added area: $newArea"
            } else {
                Write-Host "  Already in list: $newArea" -ForegroundColor DarkGray
            }
            continue
        }

        # [number] Remove specific
        if ($choice -match "^\d+$") {
            $removeIndex = [int]$choice - 1
            if ($removeIndex -ge 0 -and $removeIndex -lt $approved.Count) {
                $removed = $approved[$removeIndex]
                $approved.RemoveAt($removeIndex)
                Write-Host "  Removed: $removed" -ForegroundColor Yellow
                Write-InterviewLog "User removed area: $removed"
            } else {
                Write-Host "  Invalid index" -ForegroundColor Red
            }
            continue
        }

        Write-Host "  Invalid choice. Try again." -ForegroundColor Red
    }
}

# ============================================================================
# WINDOW SPAWNING
# ============================================================================
# Save-InterviewQueue is provided by lib/interview.ps1

function Start-RalphWindows {
    <#
    .SYNOPSIS
        Spawns Ralph loop and watch windows with mode support
    .PARAMETER FocusAreas
        Array of focus areas (for display purposes)
    .PARAMETER SelectedMode
        Execution mode: standard, trueauto, ralphschoice, ralphschoiceauto, overnight, resume
    .PARAMETER FocusArea
        Single focus area for direct mode (not queue mode)
    .PARAMETER MaxHours
        Max hours for overnight mode (default: 12)
    #>
    param(
        $FocusAreas = @(),
        [string]$SelectedMode = "standard",
        [string]$FocusArea = "",
        [int]$MaxHours = 12
    )

    Write-Host ""
    Write-Host "  Launching Ralph..." -ForegroundColor Cyan

    # Escape path for safe interpolation
    $escapedPath = $script:ProjectRoot -replace "'", "''"

    # Build Ralph arguments based on mode
    $ralphArgs = @()

    switch ($SelectedMode) {
        "standard" {
            # Queue mode uses -Queue flag
            if ($FocusAreas.Count -gt 0) {
                $ralphArgs += "-Queue"
                $ralphArgs += "-SkipPlanApproval"
            } elseif ($FocusArea) {
                $ralphArgs += "-FocusArea"
                $ralphArgs += $FocusArea
            }
        }
        "trueauto" {
            $ralphArgs += "-TrueAuto"
            $ralphArgs += "-SkipPlanApproval"
            if ($FocusAreas.Count -gt 0) {
                $ralphArgs += "-Queue"
            } elseif ($FocusArea) {
                $ralphArgs += "-FocusArea"
                $ralphArgs += $FocusArea
            }
        }
        "resume" {
            $ralphArgs += "-Resume"
        }
        "ralphschoice" {
            $ralphArgs += "-RalphsChoice"
        }
        "ralphschoiceauto" {
            $ralphArgs += "-RalphsChoiceAuto"
        }
        "overnight" {
            $ralphArgs += "-Overnight"
            $ralphArgs += "-MaxHours"
            $ralphArgs += $MaxHours
            if ($FocusAreas.Count -gt 0) {
                $ralphArgs += "-Queue"
            }
        }
    }

    $argString = $ralphArgs -join ' '
    $displayMode = if ($SelectedMode) { $SelectedMode } else { "queue" }
    $displayAreas = if ($FocusAreas.Count -gt 0) { $FocusAreas -join ', ' } elseif ($FocusArea) { $FocusArea } else { "Ralph's Choice" }

    Write-InterviewLog "Spawning Ralph - Mode: $displayMode, Areas: $displayAreas, Args: $argString"

    # Spawn Watch in new window first
    $watchCmd = "Set-Location '$escapedPath'; .\scripts\ralph\watch.ps1"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$watchCmd}"

    Start-Sleep -Milliseconds 500

    # Spawn Ralph loop in new window
    $ralphCmd = "Set-Location '$escapedPath'; .\scripts\ralph\ralph.ps1 $argString"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$ralphCmd}"

    Write-Host ""
    Write-Host "  Launched. Working on: $displayAreas" -ForegroundColor Green
    Write-Host "  You can close this window." -ForegroundColor Gray
    Write-InterviewLog "Ralph loop and watch windows launched successfully"
}

# ============================================================================
# ENTRY POINT
# ============================================================================

# Entry point
Write-RalphHeader
Write-InterviewLog "Interview session started"

# ============================================================================
# DIRECT MODE HANDLING (Skip interview if -Mode or -FocusArea provided)
# ============================================================================

# Track selected mode for later use
$script:SelectedMode = $Mode

# Direct focus area mode - skip interview entirely
if ($FocusArea) {
    # Validate focus area exists
    if (-not (Test-FocusAreaExists -AreaId $FocusArea)) {
        Write-Host "  Warning: '$FocusArea' may not be a known focus area" -ForegroundColor Yellow
        if ($config -and $config.focusAreas) {
            $knownAreas = $config.focusAreas | ForEach-Object { $_.id }
            Write-Host "  Known areas: $($knownAreas -join ', ')" -ForegroundColor DarkGray
        }
    }

    Write-Host "  Focus area: $FocusArea" -ForegroundColor Cyan

    # Default to standard mode if not provided
    if (-not $script:SelectedMode) {
        $script:SelectedMode = "standard"
    }

    Write-InterviewLog "Direct focus area: $FocusArea, Mode: $script:SelectedMode"

    if (-not $NoLaunch) {
        Start-RalphWindows -FocusArea $FocusArea -SelectedMode $script:SelectedMode -MaxHours $MaxHours
    } else {
        Write-Host "  Configuration ready. Returning to caller..." -ForegroundColor Green
    }
    exit 0
}

# Multiple focus areas mode (e.g., overnight with specific areas)
if ($FocusAreas -and $FocusAreas.Count -gt 0) {
    Write-Host "  Multiple focus areas mode: $($FocusAreas -join ', ')" -ForegroundColor Cyan

    # Default to overnight if multiple areas
    if (-not $script:SelectedMode) {
        $script:SelectedMode = "overnight"
    }

    # Create queue with focus areas
    $context = @{
        workType  = "queue"
        details   = "Direct CLI invocation with focus areas: $($FocusAreas -join ', ')"
        area      = ""
        client    = ""
        priority  = "normal"
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }

    Save-InterviewQueue -Context $context -FocusAreas $FocusAreas | Out-Null
    Write-InterviewLog "Direct focus areas: $($FocusAreas -join ', '), Mode: $script:SelectedMode"

    if (-not $NoLaunch) {
        Start-RalphWindows -FocusAreas $FocusAreas -SelectedMode $script:SelectedMode -MaxHours $MaxHours
    } else {
        Write-Host "  Queue saved. Returning to caller..." -ForegroundColor Green
    }
    exit 0
}

# Direct mode without focus area - show focus area selection, then launch
if ($Mode -and $Mode -notin @("resume", "ralphschoice", "ralphschoiceauto")) {
    Write-Host "  Mode: $Mode (direct)" -ForegroundColor Cyan

    if ($Mode -eq "smartqueue") {
        # Smart queue mode - use Claude to pick focus areas
        Write-Host ""
        Write-Host "  Describe what you want to work on (2-3 sentences):" -ForegroundColor Yellow
        $userInput = Read-Host "  >"

        if (-not [string]::IsNullOrWhiteSpace($userInput)) {
            # Try to use lib/interview.ps1's Get-SuggestedAreas
            $suggested = Get-SuggestedAreas -Problem $userInput -NoLLM:$false
            if ($suggested -and $suggested.Count -gt 0) {
                Write-Host ""
                Write-Host "  Suggested focus areas:" -ForegroundColor Cyan
                $suggested | ForEach-Object { Write-Host "    - $_" -ForegroundColor White }

                $context = @{
                    workType  = "smartqueue"
                    details   = $userInput
                    area      = ""
                    client    = ""
                    priority  = "normal"
                    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                }
                Save-InterviewQueue -Context $context -FocusAreas $suggested | Out-Null
            }
        }
        $script:SelectedMode = "standard"
    } else {
        # Show focus area selection
        $selectedFocus = Show-CategorizedFocusAreaSelection
        if ($selectedFocus) {
            if (-not $NoLaunch) {
                Start-RalphWindows -FocusArea $selectedFocus -SelectedMode $Mode -MaxHours $MaxHours
            }
            exit 0
        }
    }
}

# Modes that don't need focus area selection
if ($Mode -in @("resume", "ralphschoice", "ralphschoiceauto")) {
    Write-Host "  Mode: $Mode" -ForegroundColor Cyan
    Write-InterviewLog "Direct mode: $Mode"

    if (-not $NoLaunch) {
        Start-RalphWindows -SelectedMode $Mode -MaxHours $MaxHours
    }
    exit 0
}

# ============================================================================
# QUEUE CONTINUATION CHECK (Interactive mode only)
# ============================================================================

$queueInfo = Test-ExistingQueue
if ($queueInfo.exists -and $queueInfo.incompleteCount -gt 0 -and -not $Resume) {
    $queueChoice = Show-QueueContinuationPrompt -QueueInfo $queueInfo

    switch ($queueChoice) {
        "continue" {
            Write-Host "  Resuming previous session..." -ForegroundColor Green
            $script:ResumeMode = $true
            $script:FocusAreas = $queueInfo.queue.focusAreas | Where-Object { -not $_.completed }
        }
        "new" {
            Write-Host "  Starting new interview..." -ForegroundColor Cyan
            Clear-QueueForFresh
            $script:ResumeMode = $false
        }
        "cancel" {
            Write-Host "  Exiting." -ForegroundColor DarkGray
            exit 0
        }
    }
} elseif ($Resume -and $queueInfo.exists -and $queueInfo.incompleteCount -gt 0) {
    # -Resume flag was passed explicitly
    Write-Host "  Resuming previous session (via -Resume flag)..." -ForegroundColor Green
    $script:ResumeMode = $true
    $script:FocusAreas = $queueInfo.queue.focusAreas | Where-Object { -not $_.completed }
} else {
    # No existing context or starting fresh
    $script:ResumeMode = $false
}

# ============================================================================
# MAIN INTERVIEW FLOW
# ============================================================================

if (-not $script:ResumeMode) {
    # Simplified flow: just ask what they want to do
    Write-Host ""
    Write-Host "  What do you want to work on?" -ForegroundColor Cyan
    Write-Host "  (Press Enter to use Ralph's Choice)" -ForegroundColor DarkGray
    Write-Host ""
    $userInput = Read-Host "  "
    Write-InterviewLog "User input: $userInput"

    # Create context
    $interviewContext = @{
        workType  = "improvement"
        details   = $userInput
        area      = ""
        client    = ""
        priority  = "normal"
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    }

    # ---- CONTEXT IMPROVEMENT: Scan codebase and improve vague input ----
    $improved = $null
    if (-not [string]::IsNullOrWhiteSpace($userInput)) {
        $improved = Improve-InterviewContext -RawInput $userInput
        if ($improved.improvedContext -ne $userInput) {
            $confirmResult = Show-ImprovedContext -ImprovedResult $improved
            if ($confirmResult.approved) {
                $interviewContext.details = $confirmResult.context
                $interviewContext.improvedContext = $improved
                Write-InterviewLog "Context improved: $($interviewContext.details)"
            }
        }
    } else {
        # Empty input - use Ralph's Choice
        Write-Host "  Using Ralph's Choice to select focus areas..." -ForegroundColor Magenta
        $interviewContext.details = "Ralph's Choice - autonomous area selection"
    }

    # Get suggested focus areas from codebase analysis or keywords
    $autoDetectedAreas = if ($improved -and $improved.detectedAreas) { $improved.detectedAreas } else { @() }
    if ($autoDetectedAreas.Count -gt 0) {
        Write-Host "  Using areas detected from codebase analysis..." -ForegroundColor Magenta
        $suggestions = $autoDetectedAreas
    } else {
        $suggestions = Get-SuggestedFocusAreas -Context $interviewContext -UseRalphsChoice
    }

    # Let user approve/modify the suggestions
    $approvedAreas = Get-ApprovedAreas -Suggestions $suggestions

    # If user chose to restart, re-run the script
    if ($null -eq $approvedAreas) {
        Write-Host ""
        Write-Host "  Restarting interview..." -ForegroundColor Yellow
        Write-Host ""
        & $PSCommandPath -NoLaunch:$NoLaunch
        return
    }

    # Save the queue (use $savedQueue to avoid collision with [switch]$Queue script parameter)
    $savedQueue = Save-InterviewQueue -Context $interviewContext -FocusAreas $approvedAreas

    # Ask about autonomous mode (unless already provided via parameter)
    Write-Host ""
    Write-Host "  Focus areas: $($approvedAreas -join ', ')" -ForegroundColor Green

    if (-not $script:SelectedMode) {
        $script:SelectedMode = Show-ModeSelection
    }

    Write-InterviewLog "Mode selected: $script:SelectedMode"

    # Launch Ralph windows (unless -NoLaunch was specified)
    if (-not $NoLaunch) {
        Start-RalphWindows -FocusAreas $approvedAreas -SelectedMode $script:SelectedMode -MaxHours $MaxHours
    } else {
        Write-Host ""
        Write-Host "  Queue saved. Returning to caller..." -ForegroundColor Green
    }
} else {
    # Resume mode - just launch Ralph windows with remaining focus areas
    $remainingAreas = @()
    foreach ($area in $script:FocusAreas) {
        $areaId = if ($area.id) { $area.id } else { $area }
        $remainingAreas += $areaId
    }

    Write-Host ""
    Write-Host "  Resuming with $($remainingAreas.Count) remaining focus areas:" -ForegroundColor Cyan
    foreach ($area in $remainingAreas) {
        Write-Host "    - $area" -ForegroundColor Gray
    }

    # Default to standard for resume, no extra prompts
    if (-not $script:SelectedMode) {
        $script:SelectedMode = "standard"
    }

    Write-InterviewLog "Resume mode: $script:SelectedMode"

    if (-not $NoLaunch) {
        Start-RalphWindows -FocusAreas $remainingAreas -SelectedMode $script:SelectedMode -MaxHours $MaxHours
    } else {
        Write-Host ""
        Write-Host "  Queue ready. Returning to caller..." -ForegroundColor Green
    }
}
