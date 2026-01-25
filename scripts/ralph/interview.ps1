# Ralph Interview Mode
# Usage: .\scripts\ralph\interview.ps1 [-Resume]

param(
    [switch]$Resume
)

# Set up paths
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"

# Load config if it exists
$config = $null
if (Test-Path $script:ConfigFile) {
    $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
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

# ============================================================================
# LOGGING FUNCTION
# ============================================================================

function Write-InterviewLog {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"

    # Create logs directory if needed
    $logsDir = Join-Path $script:RalphDir "logs"
    if (-not (Test-Path $logsDir)) {
        New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    }

    # Log to interview-specific log file
    $logFile = Join-Path $logsDir "interview.log"
    Add-Content -Path $logFile -Value $logLine

    # Also write to verbose stream for debugging
    Write-Verbose $logLine
}

function Write-RalphHeader {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "   'Hi, Super Nintendo Chalmers!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Interview Mode" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-ExistingContext {
    <#
    .SYNOPSIS
        Checks if there's resumable interview context in queue.json
    .DESCRIPTION
        Returns a hashtable with:
        - HasContext: $true if resumable context exists
        - Context: The interview context string
        - FocusAreas: Array of focus area objects
        - Completed: Number of completed focus areas
        - Total: Total number of focus areas
    #>

    # Check if queue.json exists
    if (-not (Test-Path $script:QueueFile)) {
        return @{ HasContext = $false }
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Check for interviewContext
        if (-not $queue.interviewContext) {
            return @{ HasContext = $false }
        }

        # Check for focusAreas with incomplete items
        if (-not $queue.focusAreas -or $queue.focusAreas.Count -eq 0) {
            return @{ HasContext = $false }
        }

        # Count completed vs incomplete focus areas
        $completed = 0
        $incomplete = 0

        foreach ($area in $queue.focusAreas) {
            if ($area.completed -eq $true) {
                $completed++
            } else {
                $incomplete++
            }
        }

        # Only return context if there are incomplete items
        if ($incomplete -eq 0) {
            return @{ HasContext = $false }
        }

        return @{
            HasContext = $true
            Context = $queue.interviewContext
            FocusAreas = $queue.focusAreas
            Completed = $completed
            Total = $queue.focusAreas.Count
        }
    }
    catch {
        # If we can't parse the file, no resumable context
        return @{ HasContext = $false }
    }
}

function Show-ResumePrompt {
    <#
    .SYNOPSIS
        Displays resume prompt for interrupted interview sessions
    .PARAMETER ExistingContext
        The context hashtable from Get-ExistingContext
    .RETURNS
        $true if user wants to resume, $false for new interview
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$ExistingContext
    )

    Write-Host "  Found interrupted session:" -ForegroundColor Yellow
    Write-Host ""

    # Display context (truncate if too long)
    $contextText = $ExistingContext.Context
    if ($contextText.Length -gt 200) {
        $contextText = $contextText.Substring(0, 197) + "..."
    }
    Write-Host "  Context: $contextText" -ForegroundColor Gray
    Write-Host ""

    # Show progress
    $completed = $ExistingContext.Completed
    $total = $ExistingContext.Total
    Write-Host "  Progress: $completed/$total focus areas complete" -ForegroundColor Cyan
    Write-Host ""

    # List remaining focus areas
    $remaining = $ExistingContext.FocusAreas | Where-Object { -not $_.completed }
    if ($remaining.Count -gt 0) {
        Write-Host "  Remaining focus areas:" -ForegroundColor White
        foreach ($area in $remaining) {
            $areaName = if ($area.name) { $area.name } else { $area.area }
            Write-Host "    - $areaName" -ForegroundColor Gray
        }
        Write-Host ""
    }

    # Prompt user
    Write-Host "  Resume this session? [Y]es / [N]ew interview" -ForegroundColor Yellow -NoNewline
    $response = Read-Host " "

    $result = ($response -match "^[Yy]")
    Write-InterviewLog "Resume prompt shown - Context: $($ExistingContext.Context)"
    Write-InterviewLog "User chose to resume: $result"

    return $result
}

# ============================================================================
# QUESTION FUNCTIONS
# ============================================================================

function Ask-WorkType {
    <#
    .SYNOPSIS
        Asks user what kind of work they want to do
    .RETURNS
        "bug", "feature", "improvement", or "client"
    #>

    Write-Host "  What kind of work?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [B] Bug fix" -ForegroundColor White
    Write-Host "  [F] Feature" -ForegroundColor White
    Write-Host "  [I] Improvement (default)" -ForegroundColor White
    Write-Host "  [C] Client feedback" -ForegroundColor White
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Bb]" { return "bug" }
        "^[Ff]" { return "feature" }
        "^[Cc]" { return "client" }
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
            $clientsData = Get-Content $clientsFile -Raw | ConvertFrom-Json
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

# ============================================================================
# FOCUS AREA SUGGESTION FUNCTIONS
# ============================================================================

function Get-SuggestedFocusAreas {
    <#
    .SYNOPSIS
        Suggests focus areas based on interview context
    .PARAMETER Context
        The interview context hashtable from Start-Interview
    .RETURNS
        Array of suggested focus area IDs (max 5)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Context
    )

    $suggestions = @()

    # If user specified an area, add it first
    if ($Context.area -and $Context.area.Trim() -ne "") {
        $suggestions += $Context.area.Trim().ToLower()
    }

    # Keyword matching from details (lowercase)
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

    $detailsLower = $Context.details.ToLower()

    foreach ($pattern in $keywordMap.Keys) {
        if ($detailsLower -match $pattern) {
            $areaId = $keywordMap[$pattern]
            if ($suggestions -notcontains $areaId) {
                $suggestions += $areaId
            }
        }
    }

    # If < 3 suggestions, add defaults based on work type
    if ($suggestions.Count -lt 3) {
        $defaults = @{
            "bug" = @("agents", "testing")
            "feature" = @("pipeline", "config")
            "improvement" = @("quality", "speed")
            "client" = @("client-learning", "quality")
        }

        $workTypeDefaults = $defaults[$Context.workType]
        if ($workTypeDefaults) {
            foreach ($defaultArea in $workTypeDefaults) {
                if ($suggestions -notcontains $defaultArea -and $suggestions.Count -lt 5) {
                    $suggestions += $defaultArea
                }
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

        # [1-5] Remove specific
        if ($choice -match "^[1-5]$") {
            $removeIndex = [int]$choice - 1
            if ($removeIndex -lt $approved.Count) {
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
# QUEUE SAVE AND WINDOW SPAWNING
# ============================================================================

function Save-InterviewQueue {
    <#
    .SYNOPSIS
        Saves the interview context and focus areas to queue.json
    .PARAMETER Context
        The interview context hashtable from Start-Interview
    .PARAMETER FocusAreas
        Array or ArrayList of approved focus area IDs
    .RETURNS
        The queue hashtable that was saved
    #>
    param(
        [Parameter(Mandatory=$true)]
        [hashtable]$Context,
        [Parameter(Mandatory=$true)]
        $FocusAreas
    )

    # Build focus area objects with tracking fields
    $focusAreaObjects = @()
    foreach ($area in $FocusAreas) {
        $focusAreaObjects += @{
            id = $area
            completed = $false
            startedAt = $null
            completedAt = $null
        }
    }

    # Create the queue structure
    $queue = @{
        interviewContext = "$($Context.workType): $($Context.details)"
        interviewDetails = $Context
        focusAreas = $focusAreaObjects
        createdAt = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        sessionId = [guid]::NewGuid().ToString().Substring(0, 8)
    }

    # Save to queue.json
    $queue | ConvertTo-Json -Depth 10 | Set-Content -Path $script:QueueFile -Encoding UTF8

    Write-Host "  Saved queue with $($FocusAreas.Count) focus areas" -ForegroundColor Green
    Write-InterviewLog "Queue saved - Session: $($queue.sessionId), Areas: $($FocusAreas.Count)"

    return $queue
}

function Start-RalphWindows {
    <#
    .SYNOPSIS
        Spawns Ralph loop and watch windows
    .PARAMETER FocusAreas
        Array of focus areas (for display purposes)
    #>
    param(
        [Parameter(Mandatory=$true)]
        $FocusAreas
    )

    Write-Host ""
    Write-Host "  Launching Ralph..." -ForegroundColor Cyan
    Write-InterviewLog "Spawning Ralph windows for areas: $($FocusAreas -join ', ')"

    # Spawn Ralph loop in new window
    $ralphCmd = "Set-Location '$script:ProjectRoot'; .\scripts\ralph\ralph.ps1 -Queue -SkipPlanApproval"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$ralphCmd}"

    Start-Sleep -Seconds 2

    # Spawn Watch in new window
    $watchCmd = "Set-Location '$script:ProjectRoot'; .\scripts\ralph\watch.ps1"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {$watchCmd}"

    Write-Host ""
    Write-Host "  Ralph loop and watch windows launched!" -ForegroundColor Green
    Write-Host "  You can close this window now." -ForegroundColor Gray
    Write-InterviewLog "Ralph loop and watch windows launched successfully"
}

# ============================================================================
# ENTRY POINT
# ============================================================================

# Entry point
Write-RalphHeader
Write-InterviewLog "Interview session started"

# Check for existing session to resume
$existing = Get-ExistingContext
if ($existing.HasContext -and -not $Resume) {
    $shouldResume = Show-ResumePrompt -ExistingContext $existing
    if ($shouldResume) {
        Write-Host "  Resuming previous session..." -ForegroundColor Green
        $script:ResumeMode = $true
        $script:FocusAreas = $existing.FocusAreas | Where-Object { -not $_.completed }
    } else {
        Write-Host "  Starting new interview..." -ForegroundColor Cyan
        $script:ResumeMode = $false
    }
} elseif ($Resume -and $existing.HasContext) {
    # -Resume flag was passed explicitly
    Write-Host "  Resuming previous session (via -Resume flag)..." -ForegroundColor Green
    $script:ResumeMode = $true
    $script:FocusAreas = $existing.FocusAreas | Where-Object { -not $_.completed }
} else {
    # No existing context or starting fresh
    $script:ResumeMode = $false
}

# ============================================================================
# MAIN INTERVIEW FLOW
# ============================================================================

if (-not $script:ResumeMode) {
    Write-Host "  Let's figure out what you need." -ForegroundColor White
    Write-Host ""

    $interviewContext = Start-Interview

    Write-Host ""
    Write-Host "  Got it. Let me suggest some focus areas..." -ForegroundColor Green
    Write-Host ""

    # Get suggested focus areas based on interview context
    $suggestions = Get-SuggestedFocusAreas -Context $interviewContext

    # Let user approve/modify the suggestions
    $approvedAreas = Get-ApprovedAreas -Suggestions $suggestions

    # If user chose to restart, re-run the script
    if ($null -eq $approvedAreas) {
        Write-Host ""
        Write-Host "  Restarting interview..." -ForegroundColor Yellow
        Write-Host ""
        & $PSCommandPath
        return
    }

    # Save the queue
    $queue = Save-InterviewQueue -Context $interviewContext -FocusAreas $approvedAreas

    # Launch Ralph windows
    Start-RalphWindows -FocusAreas $approvedAreas
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

    Start-RalphWindows -FocusAreas $remainingAreas
}
