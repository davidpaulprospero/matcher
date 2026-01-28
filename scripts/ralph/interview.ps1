# Ralph Interview Mode
# Usage: .\scripts\ralph\interview.ps1 [-Resume] [-NoLaunch]

param(
    [switch]$Resume,
    [switch]$NoLaunch  # Skip launching Ralph/Watch windows (used when called from Ralph)
)

# Set up paths
$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"

# Load domain modules for shared state access
$script:LibPath = Join-Path $script:RalphDir 'lib'
if (Test-Path $script:LibPath) {
    . "$script:LibPath\sprint.ps1"   # Read-JsonFile, Save-StateFile
    . "$script:LibPath\queue.ps1"    # Get-Queue, Save-Queue
}

# Load config if it exists
$config = Read-JsonFile -Path $script:ConfigFile

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

    $queue = Get-Queue
    if (-not $queue) {
        return @{ HasContext = $false }
    }

    try {
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
        "bug", "feature", "improvement", "client", or "queue"
    #>

    Write-Host "  What kind of work?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [B] Bug fix" -ForegroundColor White
    Write-Host "  [F] Feature" -ForegroundColor White
    Write-Host "  [I] Improvement (default)" -ForegroundColor White
    Write-Host "  [C] Client feedback" -ForegroundColor White
    Write-Host "  [Q] Queue focus areas" -ForegroundColor Magenta
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Bb]" { return "bug" }
        "^[Ff]" { return "feature" }
        "^[Cc]" { return "client" }
        "^[Qq]" { return "queue" }
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

    # Save to queue.json (atomic write)
    Save-StateFile -Path $script:QueueFile -Data $queue

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

    # First ask work type to determine flow
    $workType = Ask-WorkType
    Write-InterviewLog "Work type selected: $workType"

    if ($workType -eq "queue") {
        # Queue mode - direct area selection, skip interview
        Write-Host ""
        $selectedAreas = Start-QueueMode

        Write-Host ""
        Write-Host "  Confirm your selections:" -ForegroundColor Green
        Write-Host ""

        # Let user approve/modify the selections
        $approvedAreas = Get-ApprovedAreas -Suggestions $selectedAreas

        # If user chose to restart, re-run the script
        if ($null -eq $approvedAreas) {
            Write-Host ""
            Write-Host "  Restarting interview..." -ForegroundColor Yellow
            Write-Host ""
            & $PSCommandPath -NoLaunch:$NoLaunch
            return
        }

        # Ask for optional work description
        Write-Host ""
        Write-Host "  What should Ralph work on? (Enter to use default)" -ForegroundColor Cyan
        Write-Host "  Default: Find issues, add tests, improve code quality" -ForegroundColor DarkGray
        Write-Host ""
        $queueDescription = Read-Host "  "

        if ([string]::IsNullOrWhiteSpace($queueDescription)) {
            $queueDescription = "Find and fix issues, add missing tests, improve code quality and documentation"
        }
        Write-InterviewLog "Queue description: $queueDescription"

        # Create context for queue mode with actual work description
        $interviewContext = @{
            workType  = "queue"
            details   = $queueDescription
            area      = ""
            client    = ""
            priority  = "normal"
            timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        }
    } else {
        # Normal interview flow - continue with remaining questions
        # Create context with the already-answered workType
        $interviewContext = @{
            workType  = $workType
            details   = ""
            area      = ""
            client    = ""
            priority  = "normal"
            timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        }

        # ---- ALWAYS ASK: Details ----
        $interviewContext.details = Ask-Details -WorkType $workType
        Write-InterviewLog "Details provided: $($interviewContext.details)"

        # ---- DYNAMIC: Area (if vague description) ----
        if ($interviewContext.details.Length -lt 20) {
            $interviewContext.area = Ask-Area -WorkType $workType -Details $interviewContext.details
            Write-InterviewLog "Area specified: $($interviewContext.area)"
        }

        # ---- DYNAMIC: Client (if client work type or client-related keywords) ----
        $clientPattern = "client|theresa|stu|feedback"
        if ($workType -eq "client" -or $interviewContext.details -match $clientPattern) {
            $interviewContext.client = Ask-Client
            Write-InterviewLog "Client specified: $($interviewContext.client)"
        }

        # ---- DYNAMIC: Priority (if bug or urgency keywords) ----
        $urgencyPattern = "urgent|critical|asap|broken"
        if ($workType -eq "bug" -or $interviewContext.details -match $urgencyPattern) {
            $interviewContext.priority = Ask-Priority
            Write-InterviewLog "Priority set: $($interviewContext.priority)"
        }

        Write-Host ""
        Write-Host "  Got it. Let me suggest some focus areas..." -ForegroundColor Green
        Write-Host ""

        # Get suggested focus areas based on interview context
        # Use Ralph's Choice scoring when user didn't specify an area
        $useRalphsChoice = [string]::IsNullOrWhiteSpace($interviewContext.area)
        $suggestions = Get-SuggestedFocusAreas -Context $interviewContext -UseRalphsChoice:$useRalphsChoice

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
    }

    # Save the queue
    $queue = Save-InterviewQueue -Context $interviewContext -FocusAreas $approvedAreas

    # Launch Ralph windows (unless -NoLaunch was specified)
    if (-not $NoLaunch) {
        Start-RalphWindows -FocusAreas $approvedAreas
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

    if (-not $NoLaunch) {
        Start-RalphWindows -FocusAreas $remainingAreas
    } else {
        Write-Host ""
        Write-Host "  Queue ready. Returning to caller..." -ForegroundColor Green
    }
}
