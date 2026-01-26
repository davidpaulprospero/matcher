# Ralph Loop - Main Execution Script
# Usage: .\scripts\ralph\ralph.ps1 [-Queue] [-SkipPlanApproval] [-TrueAuto] [-Resume] [-FocusArea <area>]
#
# Modes:
#   -Queue             Process focus areas from queue.json (interview mode)
#   -SkipPlanApproval  Skip plan approval prompts
#   -TrueAuto          Continuous improvement mode (no exit on sprint complete)
#   -Resume            Resume previous sprint instead of starting new
#   -FocusArea <area>  Override focus area for this session
#   -RalphsChoice      Ralph decides focus areas, user confirms each decision
#   -RalphsChoiceAuto  Ralph decides and continues autonomously

param(
    [switch]$Queue,
    [switch]$SkipPlanApproval,
    [switch]$TrueAuto,
    [switch]$Resume,
    [switch]$RalphsChoice,
    [switch]$RalphsChoiceAuto,
    [string]$FocusArea = "",
    [string]$Task = ""
)

# ============================================================================
# SETUP
# ============================================================================

$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"
$script:QueueFile = Join-Path $script:RalphDir "queue.json"
$script:ConfigFile = Join-Path $script:RalphDir "ralph-config.json"
$script:PrdFile = Join-Path $script:RalphDir "prd.json"
$script:ProgressFile = Join-Path $script:RalphDir "progress.txt"
$script:MetricsFile = Join-Path $script:RalphDir "metrics.csv"
$script:PromptFile = Join-Path $script:RalphDir "prompt.md"
$script:LogDir = Join-Path $script:RalphDir "logs"
$script:ArchiveDir = Join-Path $script:RalphDir "archive"
$script:SprintHistoryFile = Join-Path $script:RalphDir "sprint_history.json"
$script:ExplorationContextFile = Join-Path $script:RalphDir "exploration_context.md"

# Session tracking
$script:SessionId = Get-Date -Format "yyyy-MM-dd_HHmmss"
$script:IterationCount = 0
$script:ConsecutiveFailures = 0
$script:SessionStartTime = Get-Date
$script:CurrentMode = "Standard"  # "Interview", "Standard", or "TrueAuto"

# Retry tracking
$script:CurrentRetryCount = 0      # Attempts on current focus area/story
$script:LastFocusAreaId = ""       # Track when focus area changes
$script:LastStoryId = ""           # Track when story changes

# Exploration tracking
$script:StoriesSinceExploration = 0        # Counter for periodic exploration
$script:LastExplorationSummary = ""        # Cached exploration summary
$script:LastExplorationTime = $null        # When last exploration ran
$script:SprintExplorationContext = ""      # Sprint-start exploration context
$script:LastExplorationCommit = (git rev-parse HEAD 2>$null)  # Commit hash at last exploration (init to current HEAD)

# Ensure logs directory exists
if (-not (Test-Path $script:LogDir)) {
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
}

# Create session log directory
$script:SessionLogDir = Join-Path $script:LogDir $script:SessionId
New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null

# Ensure archive directory exists
if (-not (Test-Path $script:ArchiveDir)) {
    New-Item -ItemType Directory -Path $script:ArchiveDir -Force | Out-Null
}

# ============================================================================
# CONFIG LOADING
# ============================================================================

function Get-RalphConfig {
    <#
    .SYNOPSIS
        Loads ralph-config.json with defaults
    #>
    $defaults = @{
        claudePath = "claude"
        maxIterations = 10
        iterationTimeout = 600
        focusAreas = @()
        autonomy = @{
            maxIterations = 10
            fastFail = @{
                consecutiveFailures = 3
                minIterationTime = 120
            }
        }
    }

    if (Test-Path $script:ConfigFile) {
        try {
            $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
            return $config
        }
        catch {
            Write-Host "  Warning: Could not parse ralph-config.json, using defaults" -ForegroundColor Yellow
            return [PSCustomObject]$defaults
        }
    }

    return [PSCustomObject]$defaults
}

$script:Config = Get-RalphConfig

# ============================================================================
# SPRINT ARCHIVE FUNCTIONS
# ============================================================================

function Get-SprintHistory {
    <#
    .SYNOPSIS
        Load sprint history or return empty structure
    #>
    if (Test-Path $script:SprintHistoryFile) {
        try {
            return Get-Content $script:SprintHistoryFile -Raw | ConvertFrom-Json
        } catch {
            Write-Host "  Warning: Could not parse sprint_history.json" -ForegroundColor Yellow
        }
    }

    return @{
        version = 1
        totalSprintsCompleted = 0
        totalStoriesCompleted = 0
        focusAreaBreakdown = @{}
        sprints = @()
    }
}

function Update-SprintHistory {
    <#
    .SYNOPSIS
        Update cumulative sprint history
    .PARAMETER SprintData
        Hashtable with sprintNumber, focusArea, storiesCompleted, storiesTotal, archiveFile
    #>
    param([hashtable]$SprintData)

    $history = Get-SprintHistory

    # Ensure sprints array exists
    if (-not $history.sprints) {
        $history | Add-Member -NotePropertyName "sprints" -NotePropertyValue @() -Force
    }

    # Update totals
    $history.totalSprintsCompleted++
    $history.totalStoriesCompleted += $SprintData.storiesCompleted

    # Update focus area breakdown
    $focusArea = $SprintData.focusArea

    # Ensure focusAreaBreakdown exists as an object
    if (-not $history.focusAreaBreakdown) {
        $history | Add-Member -NotePropertyName "focusAreaBreakdown" -NotePropertyValue ([PSCustomObject]@{}) -Force
    }

    # Check if this focus area exists in breakdown
    $existingArea = $history.focusAreaBreakdown.PSObject.Properties[$focusArea]
    if (-not $existingArea) {
        # Add new focus area entry
        $history.focusAreaBreakdown | Add-Member -NotePropertyName $focusArea -NotePropertyValue ([PSCustomObject]@{ sprints = 0; stories = 0 }) -Force
    }

    # Update the counts
    $areaStats = $history.focusAreaBreakdown.$focusArea
    $newSprints = [int]$areaStats.sprints + 1
    $newStories = [int]$areaStats.stories + [int]$SprintData.storiesCompleted

    # Replace the area stats with updated values
    $history.focusAreaBreakdown | Add-Member -NotePropertyName $focusArea -NotePropertyValue ([PSCustomObject]@{
        sprints = $newSprints
        stories = $newStories
    }) -Force

    # Add sprint record
    $sprintRecord = @{
        sprintNumber = $SprintData.sprintNumber
        focusArea = $SprintData.focusArea
        startedAt = $SprintData.startedAt
        completedAt = (Get-Date).ToString("o")
        storiesCompleted = $SprintData.storiesCompleted
        storiesTotal = $SprintData.storiesTotal
        archiveFile = $SprintData.archiveFile
    }
    $history.sprints += $sprintRecord

    # Save
    $history | ConvertTo-Json -Depth 10 | Set-Content $script:SprintHistoryFile -Encoding UTF8
}

function Save-SprintArchive {
    <#
    .SYNOPSIS
        Archive current PRD before overwriting
    .PARAMETER Reason
        Why the sprint is being archived: complete, superseded, abandoned
    #>
    param(
        [ValidateSet("complete", "superseded", "abandoned")]
        [string]$Reason = "complete"
    )

    if (-not (Test-Path $script:PrdFile)) {
        return  # Nothing to archive
    }

    try {
        $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
    } catch {
        Write-Host "  Warning: Could not parse prd.json for archiving" -ForegroundColor Yellow
        return
    }

    # Check if PRD has stories
    if (-not $prd.userStories -or $prd.userStories.Count -eq 0) {
        return  # Empty PRD, skip archive
    }

    # Create archive directory
    if (-not (Test-Path $script:ArchiveDir)) {
        New-Item -ItemType Directory -Path $script:ArchiveDir -Force | Out-Null
    }

    # Determine archive filename
    $sprintNum = if ($prd.sprintNumber) { $prd.sprintNumber } else { 1 }
    $archiveFile = "sprint-$sprintNum.json"
    $archivePath = Join-Path $script:ArchiveDir $archiveFile

    # Handle duplicate sprint numbers with timestamp
    if (Test-Path $archivePath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $archiveFile = "sprint-$sprintNum-$timestamp.json"
        $archivePath = Join-Path $script:ArchiveDir $archiveFile
    }

    # Count completed stories
    $completedStories = 0
    $totalStories = $prd.userStories.Count
    foreach ($story in $prd.userStories) {
        if ($story.passes -eq $true) {
            $completedStories++
        }
    }

    # Add archive metadata to PRD
    $prd | Add-Member -NotePropertyName "_archiveMetadata" -NotePropertyValue @{
        archivedAt = (Get-Date).ToString("o")
        completedStories = $completedStories
        totalStories = $totalStories
        reason = $Reason
    } -Force

    # Save archive
    $prd | ConvertTo-Json -Depth 10 | Set-Content $archivePath -Encoding UTF8

    Write-Host "  Archived sprint $sprintNum to $archiveFile ($completedStories/$totalStories stories)" -ForegroundColor DarkGray

    # Update sprint history
    $sprintData = @{
        sprintNumber = $sprintNum
        focusArea = if ($prd.focusArea) { $prd.focusArea } else { "unknown" }
        startedAt = if ($prd.startedAt) { $prd.startedAt } else { (Get-Date).AddHours(-1).ToString("o") }
        storiesCompleted = $completedStories
        storiesTotal = $totalStories
        archiveFile = $archiveFile
    }
    Update-SprintHistory -SprintData $sprintData

    # Story 1.7: Generate sprint report
    try {
        New-SprintReport -SprintData $sprintData -Prd $prd | Out-Null
    }
    catch {
        Write-Host "  Warning: Could not generate sprint report: $_" -ForegroundColor Yellow
    }

    # Story 2.5: Sprint retrospective
    try {
        $retro = Get-SprintRetrospective -Prd $prd
        if ($retro.lessons.Count -gt 0 -or $retro.recommendations.Count -gt 0) {
            $retroFile = Join-Path $script:RalphDir "last_retrospective.json"
            Write-JsonNoBom -Path $retroFile -Content ($retro | ConvertTo-Json -Depth 5)
            Write-Host "  Retrospective: $($retro.lessons.Count) lessons, $($retro.recommendations.Count) recommendations saved" -ForegroundColor DarkGray
        }
    }
    catch {
        Write-Host "  Warning: Could not generate retrospective: $_" -ForegroundColor Yellow
    }

    # Story 3.2: Codebase health snapshot at sprint end
    try {
        $health = Measure-CodebaseHealth
        if ($health) {
            $comparison = Compare-HealthMetrics -Current $health
            if ($comparison -and -not $comparison.isBaseline -and $comparison.trends.Count -gt 0) {
                foreach ($trend in $comparison.trends) {
                    $arrow = if ($trend.direction -eq "up") { "+" } else { "-" }
                    $color = if ($trend.metric -eq "techDebt" -and $trend.direction -eq "up") { "Yellow" }
                             elseif ($trend.direction -eq "up") { "Green" }
                             else { "Red" }
                    Write-Host "  Health: $($trend.metric) $arrow ($($trend.previous) -> $($trend.current))" -ForegroundColor $color
                }
            }
        }
    }
    catch {
        Write-Host "  Warning: Could not measure codebase health: $_" -ForegroundColor Yellow
    }
}

function Test-ShouldGenerateNewPRD {
    <#
    .SYNOPSIS
        Determine if a new PRD should be generated
    .PARAMETER NewFocusArea
        The focus area being requested
    .OUTPUTS
        Hashtable with ShouldGenerate (bool) and Reason (string)
    #>
    param([string]$NewFocusArea)

    # No PRD exists -> generate new
    if (-not (Test-Path $script:PrdFile)) {
        return @{ ShouldGenerate = $true; Reason = "No PRD exists" }
    }

    try {
        $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
    } catch {
        return @{ ShouldGenerate = $true; Reason = "PRD is corrupted" }
    }

    $currentFocusArea = if ($prd.focusArea) { $prd.focusArea } else { "" }

    # Check sprint completion
    $isComplete = $true
    $hasStories = ($prd.userStories -and $prd.userStories.Count -gt 0)
    if ($hasStories) {
        foreach ($story in $prd.userStories) {
            if ($story.passes -ne $true) {
                $isComplete = $false
                break
            }
        }
    } else {
        $isComplete = $false  # No stories = not complete
    }

    # Same focus area
    if ($NewFocusArea -eq $currentFocusArea) {
        if ($isComplete) {
            return @{ ShouldGenerate = $true; Reason = "Sprint complete, same focus area - generating new stories" }
        } else {
            return @{ ShouldGenerate = $false; Reason = "Continuing incomplete sprint" }
        }
    }

    # Different focus area
    if ($isComplete) {
        return @{ ShouldGenerate = $true; Reason = "Sprint complete, switching to $NewFocusArea" }
    } else {
        return @{ ShouldGenerate = $true; Reason = "Superseding incomplete sprint with $NewFocusArea" }
    }
}

# ============================================================================
# RALPH'S CHOICE - SCORING FUNCTIONS
# ============================================================================

function Get-GitActivityByArea {
    <#
    .SYNOPSIS
        Count git commits per focus area in the last N days
    .PARAMETER WindowDays
        Number of days to look back (default: 7)
    .RETURNS
        Hashtable mapping area ID to commit count
    #>
    param(
        [int]$WindowDays = 7
    )

    $config = Get-RalphConfig
    $relevantFiles = $config.relevantFiles

    if (-not $relevantFiles) {
        return @{}
    }

    $activityByArea = @{}
    $since = (Get-Date).AddDays(-$WindowDays).ToString("yyyy-MM-dd")

    # Get list of changed files in the window
    $changedFiles = @()
    try {
        Push-Location $script:ProjectRoot
        $gitLog = git log --since="$since" --name-only --pretty=format:"" 2>$null
        if ($gitLog) {
            $changedFiles = $gitLog | Where-Object { $_ -ne "" } | Select-Object -Unique
        }
    }
    catch {
        # Git not available or not a repo
    }
    finally {
        Pop-Location
    }

    # Map changed files to focus areas
    foreach ($prop in $relevantFiles.PSObject.Properties) {
        $areaId = $prop.Name
        $patterns = $prop.Value
        $commitCount = 0

        foreach ($file in $changedFiles) {
            foreach ($pattern in $patterns) {
                # Convert glob pattern to regex
                $regexPattern = $pattern -replace '\*\*/', '.*' -replace '\*', '[^/]*' -replace '\.', '\.'
                if ($file -match $regexPattern) {
                    $commitCount++
                    break  # Don't double-count same file
                }
            }
        }

        $activityByArea[$areaId] = $commitCount
    }

    return $activityByArea
}

function Get-FocusAreaScore {
    <#
    .SYNOPSIS
        Calculate weighted score for a single focus area
    .PARAMETER AreaId
        The focus area ID to score
    .PARAMETER ActivityByArea
        Hashtable of git activity counts (from Get-GitActivityByArea)
    .PARAMETER SprintHistory
        Sprint history object (from Get-SprintHistory)
    .RETURNS
        Hashtable with score breakdown: activityScore, neglectedScore, balanceScore, total
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$AreaId,
        [Parameter(Mandatory=$true)]
        [hashtable]$ActivityByArea,
        [Parameter(Mandatory=$true)]
        $SprintHistory
    )

    $config = Get-RalphConfig
    $weights = if ($config.ralphsChoice -and $config.ralphsChoice.weights) {
        $config.ralphsChoice.weights
    } else {
        @{ recentActivity = 0.5; neglectedArea = 0.3; categoryBalance = 0.2 }
    }

    # 1. Recent Activity Score (0-1)
    $commits = if ($ActivityByArea.ContainsKey($AreaId)) { $ActivityByArea[$AreaId] } else { 0 }
    $activityScore = [math]::Min($commits / 10.0, 1.0)

    # 2. Neglected Score (0-1)
    $sprintsDone = 0
    $daysSinceLastSprint = 14  # Default to max

    if ($SprintHistory.focusAreaBreakdown -and $SprintHistory.focusAreaBreakdown.$AreaId) {
        $areaStats = $SprintHistory.focusAreaBreakdown.$AreaId
        $sprintsDone = [int]$areaStats.sprints
    }

    if ($sprintsDone -eq 0) {
        $neglectedScore = 1.0
    } else {
        # Find most recent sprint for this area
        $areasSprints = $SprintHistory.sprints | Where-Object { $_.focusArea -eq $AreaId } | Sort-Object completedAt -Descending
        if ($areasSprints -and $areasSprints.Count -gt 0) {
            $lastSprint = $areasSprints | Select-Object -First 1
            if ($lastSprint.completedAt) {
                $daysSince = ((Get-Date) - [datetime]$lastSprint.completedAt).Days
                $daysSinceLastSprint = $daysSince
            }
        }
        $neglectedScore = [math]::Min($daysSinceLastSprint / 14.0, 1.0)
    }

    # 3. Category Balance Score (0-1)
    # Get area's category
    $areaCategory = ""
    foreach ($area in $config.focusAreas) {
        if ($area.id -eq $AreaId) {
            $areaCategory = $area.category
            break
        }
    }

    $categorySprintCount = 0
    $totalSprints = [int]$SprintHistory.totalSprintsCompleted

    if ($areaCategory -and $config.focusAreaCategories.$areaCategory) {
        $categoryAreas = $config.focusAreaCategories.$areaCategory.areas
        foreach ($catArea in $categoryAreas) {
            if ($SprintHistory.focusAreaBreakdown -and $SprintHistory.focusAreaBreakdown.$catArea) {
                $categorySprintCount += [int]$SprintHistory.focusAreaBreakdown.$catArea.sprints
            }
        }
    }

    $categoryRatio = if ($totalSprints -gt 0) { $categorySprintCount / $totalSprints } else { 0 }
    $balanceScore = 1.0 - $categoryRatio

    # Final weighted score
    $total = ($activityScore * $weights.recentActivity) +
             ($neglectedScore * $weights.neglectedArea) +
             ($balanceScore * $weights.categoryBalance)

    return @{
        areaId = $AreaId
        activityScore = [math]::Round($activityScore, 2)
        neglectedScore = [math]::Round($neglectedScore, 2)
        balanceScore = [math]::Round($balanceScore, 2)
        total = [math]::Round($total, 2)
        commits = $commits
        sprintsDone = $sprintsDone
    }
}

function Get-AllFocusAreaScores {
    <#
    .SYNOPSIS
        Score all focus areas and return sorted list
    .RETURNS
        Array of score hashtables, sorted by total score descending
    #>

    $config = Get-RalphConfig
    $sprintHistory = Get-SprintHistory
    $windowDays = if ($config.ralphsChoice.activityWindowDays) { $config.ralphsChoice.activityWindowDays } else { 7 }
    $activityByArea = Get-GitActivityByArea -WindowDays $windowDays

    $scores = @()

    foreach ($area in $config.focusAreas) {
        $areaId = $area.id
        $score = Get-FocusAreaScore -AreaId $areaId -ActivityByArea $activityByArea -SprintHistory $sprintHistory
        $score.areaName = $area.name
        $score.category = $area.category
        $scores += $score
    }

    # Sort by total score descending
    $sorted = $scores | Sort-Object -Property total -Descending

    return $sorted
}

function Get-StayOrSwitchDecision {
    <#
    .SYNOPSIS
        After sprint complete, determine if should stay or switch focus areas
    .PARAMETER CurrentArea
        The current focus area ID
    .RETURNS
        Hashtable with: decision (stay/switch), newArea, stayReasons, switchReasons
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$CurrentArea
    )

    $config = Get-RalphConfig
    $sprintHistory = Get-SprintHistory
    $maxSprintsBeforeRotate = if ($config.ralphsChoice.maxSprintsBeforeRotate) { $config.ralphsChoice.maxSprintsBeforeRotate } else { 3 }

    $stayReasons = @()
    $switchReasons = @()

    # Check consecutive sprints in current area
    $consecutiveSprints = 0
    if ($sprintHistory.sprints -and $sprintHistory.sprints.Count -gt 0) {
        $recentSprints = $sprintHistory.sprints | Sort-Object completedAt -Descending
        foreach ($sprint in $recentSprints) {
            if ($sprint.focusArea -eq $CurrentArea) {
                $consecutiveSprints++
            } else {
                break
            }
        }
    }

    # Check for recent commits (last day)
    $recentActivity = Get-GitActivityByArea -WindowDays 1
    $recentCommits = if ($recentActivity.ContainsKey($CurrentArea)) { $recentActivity[$CurrentArea] } else { 0 }

    # Run tests to check for failures
    $testsFailing = $false
    # TODO: Could integrate with test patterns from config to check actual test status

    # Check if PRD has unfinished stories
    $unfinishedStories = $false
    if (Test-Path $script:PrdFile) {
        try {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            if ($prd.userStories) {
                $incomplete = $prd.userStories | Where-Object { $_.passes -ne $true }
                if ($incomplete -and $incomplete.Count -gt 0) {
                    $unfinishedStories = $true
                }
            }
        } catch {}
    }

    # STAY reasons
    if ($testsFailing) {
        $stayReasons += "Tests failing in current area"
    }
    if ($recentCommits -ge 3) {
        $stayReasons += "Recent commits still flowing ($recentCommits commits today)"
    }
    if ($unfinishedStories) {
        $stayReasons += "Sprint has unfinished stories"
    }

    # SWITCH reasons
    if ($consecutiveSprints -ge $maxSprintsBeforeRotate) {
        $switchReasons += "Hit max sprints in area ($consecutiveSprints sprints)"
    }

    # Get all scores to check if another area has significantly higher score
    $allScores = Get-AllFocusAreaScores
    $matchedArea = $allScores | Where-Object { $_.areaId -eq $CurrentArea }
    $currentScore = if ($matchedArea -and $matchedArea.total) { $matchedArea.total } else { 0 }
    $topScore = $allScores[0]

    if ($topScore.areaId -ne $CurrentArea) {
        $scoreDelta = $topScore.total - $currentScore
        if ($scoreDelta -gt 0.3) {
            $switchReasons += "$($topScore.areaId) has significantly higher score ($([math]::Round($scoreDelta, 2)) delta)"
        }
        if (-not $testsFailing -and $recentCommits -lt 3) {
            $switchReasons += "Area stable: tests passing, low recent activity"
        }
    }

    # Decision rule
    $decision = "stay"
    $newArea = $CurrentArea

    if ($switchReasons.Count -gt 0 -and $stayReasons.Count -eq 0) {
        $decision = "switch"
        $newArea = $topScore.areaId
    }

    return @{
        decision = $decision
        newArea = $newArea
        currentArea = $CurrentArea
        stayReasons = $stayReasons
        switchReasons = $switchReasons
        scores = $allScores | Select-Object -First 5  # Top 5 scores
        consecutiveSprints = $consecutiveSprints
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Write-JsonNoBom {
    <#
    .SYNOPSIS
        Write JSON to file without UTF-8 BOM
    .PARAMETER Path
        File path to write to
    .PARAMETER Content
        JSON string content
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path,
        [Parameter(Mandatory=$true)]
        [string]$Content
    )

    # Use .NET to write without BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

# ============================================================================
# BANNER AND DISPLAY
# ============================================================================

function Write-RalphBanner {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host "   'Me fail English? That's unpossible!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Ralph Loop v2.0" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""

    if ($Queue) {
        Write-Host "  Mode: Interview Queue" -ForegroundColor Green
    }
    elseif ($TrueAuto) {
        Write-Host "  Mode: TrueAuto (continuous improvement)" -ForegroundColor Magenta
    }
    elseif ($RalphsChoice) {
        Write-Host "  Mode: Ralph's Choice (Ralph decides, you confirm)" -ForegroundColor Magenta
    }
    elseif ($RalphsChoiceAuto) {
        Write-Host "  Mode: Ralph's Choice Auto (fully autonomous)" -ForegroundColor Magenta
    }
    else {
        Write-Host "  Mode: Standard" -ForegroundColor White
    }

    Write-Host "  Session: $script:SessionId" -ForegroundColor DarkGray
    Write-Host ""
}

function Write-IterationBanner {
    param(
        [int]$Iteration,
        [string]$FocusArea,
        [string]$StoryId = ""
    )

    Write-Host ""
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Iteration $Iteration" -ForegroundColor Cyan
    if ($FocusArea) {
        Write-Host "  Focus: $FocusArea" -ForegroundColor Yellow
    }
    if ($StoryId) {
        Write-Host "  Story: $StoryId" -ForegroundColor Green
    }
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# QUEUE MANAGEMENT (INTERVIEW MODE)
# ============================================================================

function Get-QueueData {
    <#
    .SYNOPSIS
        Read and parse queue.json. Returns $null if file doesn't exist or is invalid.
    #>
    if (-not (Test-Path $script:QueueFile)) { return $null }
    try { return Get-Content $script:QueueFile -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Get-InterviewFocusAreas {
    <#
    .SYNOPSIS
        Read incomplete focus areas from queue.json
    .RETURNS
        Array of incomplete focus area objects, or empty array if none
    #>
    $queue = Get-QueueData
    if (-not $queue) { return @() }

    # Interview format: focusAreas array with completed flag
    if ($queue.focusAreas) {
        return @($queue.focusAreas | Where-Object { -not $_.completed })
    }

    # Legacy format: queue array with completedAreas
    if ($queue.queue -and $queue.completedAreas) {
        return @($queue.queue | Where-Object { $queue.completedAreas -notcontains $_ } | ForEach-Object { @{ id = $_; completed = $false } })
    }

    return @()
}

function Get-InterviewContext {
    <#
    .SYNOPSIS
        Get the interview context string from queue.json
    #>
    $queue = Get-QueueData
    if ($queue -and $queue.interviewContext) { return $queue.interviewContext }
    return ""
}

function Get-NextQueuedFocusArea {
    <#
    .SYNOPSIS
        Get the next incomplete focus area ID
    .RETURNS
        Focus area ID string, or $null if none remaining
    #>
    $areas = Get-InterviewFocusAreas
    if ($areas.Count -gt 0) {
        $first = $areas | Select-Object -First 1
        $result = if ($first.id) { $first.id } else { $first }
        return $result
    }
    return $null
}

function Update-QueueProgress {
    <#
    .SYNOPSIS
        Mark a focus area as completed in queue.json (handles both formats)
    .PARAMETER AreaId
        The focus area ID to mark as completed
    .PARAMETER Silent
        If set, don't print success message
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$AreaId,
        [switch]$Silent
    )

    $queue = Get-QueueData
    if (-not $queue) { return }

    try {
        $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

        # Update interview format (focusAreas array)
        if ($queue.focusAreas) {
            foreach ($area in $queue.focusAreas) {
                if ($area.id -eq $AreaId) {
                    $area.completed = $true
                    $area.completedAt = $timestamp
                }
            }
        }

        # Update legacy format (completedAreas array)
        if ($null -ne $queue.completedAreas -and $queue.completedAreas -notcontains $AreaId) {
            $queue.completedAreas += $AreaId
        }

        # Update currentIndex for legacy format
        if ($queue.queue) {
            $idx = [array]::IndexOf($queue.queue, $AreaId)
            if ($idx -ge 0) { $queue.currentIndex = $idx + 1 }
        }

        # Update session info
        if ($queue.session) {
            $queue.session.lastActivityAt = $timestamp
            $queue.session.iterationCount = $script:IterationCount
        }

        Write-JsonNoBom -Path $script:QueueFile -Content ($queue | ConvertTo-Json -Depth 10)

        if (-not $Silent) {
            Write-Host "  Marked '$AreaId' as completed" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "  Warning: Could not update queue.json" -ForegroundColor Yellow
    }
}

# Aliases for backward compatibility
function Update-InterviewProgress { param([string]$AreaId) Update-QueueProgress -AreaId $AreaId }
function Update-LegacyQueueProgress { param([string]$CompletedArea) Update-QueueProgress -AreaId $CompletedArea -Silent }

# ============================================================================
# SEED PRD CREATION
# ============================================================================

function New-SeedPRD {
    <#
    .SYNOPSIS
        Create a minimal PRD with just US-001 to generate stories for a focus area
    .DESCRIPTION
        Instead of generating all stories upfront, this creates a seed PRD where
        US-001's job is to generate the remaining stories. This allows resume to
        work naturally and makes story generation trackable.
    .PARAMETER FocusAreaId
        The focus area to create a seed PRD for
    .PARAMETER Context
        Optional context from interview
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FocusAreaId,
        [string]$Context = ""
    )

    $prdPath = Join-Path $script:RalphDir "prd.json"

    # Archive current PRD before overwriting (if exists)
    if (Test-Path $prdPath) {
        Save-SprintArchive -Reason "superseded"
    }

    # Read current PRD to get sprint number
    $currentSprint = 0
    if (Test-Path $prdPath) {
        try {
            $currentPrd = Get-Content $prdPath -Raw | ConvertFrom-Json
            $currentSprint = $currentPrd.sprintNumber
        } catch {}
    }

    $newSprint = $currentSprint + 1

    # Build context string for the story
    $contextNote = if ($Context) { "`nUser context: $Context" } else { "" }

    # Create seed PRD with just US-001
    $seedPrd = @{
        branchName = "ralph/sprint-$newSprint"
        sprintNumber = $newSprint
        focusArea = $FocusAreaId
        projectContext = @{
            description = "voiceover-matcher-subtitle: AI-powered pipeline for matching voiceover to stock footage"
            testFramework = "pytest"
            testCommand = "pytest tests/ -v --tb=short"
            sourceDir = "src/"
            testsDir = "tests/"
        }
        userStories = @(
            @{
                id = "US-001"
                title = "Generate sprint stories for $FocusAreaId focus area"
                acceptanceCriteria = @(
                    "Read scripts/ralph/ralph-config.json to understand the '$FocusAreaId' focus area"
                    "Read scripts/ralph/prompt.md for project context and architecture notes"
                    "Read scripts/ralph/queue.json for interview details (story outline, architecture decisions, key files)"
                    "Read CLAUDE.md for project conventions and known issues"
                    "Analyze the codebase to find 8-12 specific improvements for '$FocusAreaId'"
                    "Add stories US-002 through US-012 to scripts/ralph/prd.json with clear acceptance criteria"
                    "Each story should have 4-6 testable acceptance criteria"
                    "Mark this story (US-001) as passes: true when done"
                )
                priority = "high"
                passes = $false
                notes = $contextNote.Trim()
            }
        )
    }

    # Write the seed PRD
    $seedPrd | ConvertTo-Json -Depth 10 | Set-Content $prdPath -Encoding UTF8

    Write-Host "  Created seed PRD for $FocusAreaId (Sprint $newSprint)" -ForegroundColor Green
    Write-Host "  US-001 will generate the remaining stories" -ForegroundColor DarkGray

    return $true
}

# ============================================================================
# COMPLETION CHOICE
# ============================================================================

function Show-CompletionChoice {
    <#
    .SYNOPSIS
        After interview queue complete, prompt for next action
    .PARAMETER CompletedAreas
        Array of completed focus area IDs
    .RETURNS
        "interview", "trueauto", or "stop"
    #>
    param(
        [array]$CompletedAreas
    )

    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "   'I bent my Wookie!' - Ralph" -ForegroundColor Yellow
    Write-Host "   Interview Queue Complete!" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""

    # Show completed areas
    Write-Host "  Completed focus areas:" -ForegroundColor Cyan
    foreach ($area in $CompletedAreas) {
        $areaId = if ($area.id) { $area.id } else { $area }
        Write-Host "    [x] $areaId" -ForegroundColor Green
    }

    # Show session stats
    $duration = (Get-Date) - $script:SessionStartTime
    Write-Host ""
    Write-Host "  Session Stats:" -ForegroundColor DarkGray
    Write-Host "    Iterations: $script:IterationCount" -ForegroundColor DarkGray
    Write-Host "    Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor DarkGray

    Write-Host ""
    Write-Host "  What's next?" -ForegroundColor White
    Write-Host ""
    Write-Host "  [I] New interview - Start fresh with new focus areas" -ForegroundColor Cyan
    Write-Host "  [T] TrueAuto mode - Continue with general improvements" -ForegroundColor Magenta
    Write-Host "  [S] Stop - Exit Ralph loop" -ForegroundColor Yellow
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Ii]" { return "interview" }
        "^[Tt]" { return "trueauto" }
        default { return "stop" }
    }
}

# ============================================================================
# COMPREHENSIVE LOGGING FUNCTIONS (Phase 1)
# ============================================================================

function Get-GitState {
    <#
    .SYNOPSIS
        Capture current git state (hash, branch, clean status)
    .RETURNS
        Hashtable with git state info
    #>
    $state = @{
        hash = ""
        branch = ""
        clean = $true
        modifiedFiles = @()
    }

    try {
        $state.hash = (git rev-parse HEAD 2>$null)
        $state.branch = (git rev-parse --abbrev-ref HEAD 2>$null)
        $status = git status --porcelain 2>$null
        if ($status) {
            $state.clean = $false
            $state.modifiedFiles = @($status | ForEach-Object { $_.Substring(3) })
        }
    }
    catch {}

    return $state
}

function Get-FileOperations {
    <#
    .SYNOPSIS
        Get file operations between two git states
    .PARAMETER BeforeHash
        Git commit hash before the operation
    .RETURNS
        Hashtable with files created, modified, deleted
    #>
    param([string]$BeforeHash)

    $ops = @{
        filesCreated = @()
        filesModified = @()
        filesDeleted = @()
        totalFilesChanged = 0
    }

    try {
        # Get diff stats
        $diffOutput = git diff --name-status $BeforeHash HEAD 2>$null
        if ($diffOutput) {
            foreach ($line in $diffOutput) {
                if ($line -match "^([AMDRC])\s+(.+)$") {
                    $status = $Matches[1]
                    $file = $Matches[2]
                    switch ($status) {
                        "A" { $ops.filesCreated += @{ path = $file; size = (Get-Item $file -ErrorAction SilentlyContinue).Length } }
                        "M" { $ops.filesModified += @{ path = $file } }
                        "D" { $ops.filesDeleted += $file }
                    }
                }
            }
        }

        # Also check unstaged changes
        $statusOutput = git status --porcelain 2>$null
        if ($statusOutput) {
            foreach ($line in $statusOutput) {
                if ($line -match "^\?\?\s+(.+)$") {
                    $file = $Matches[1]
                    $ops.filesCreated += @{ path = $file; size = (Get-Item $file -ErrorAction SilentlyContinue).Length }
                }
            }
        }

        $ops.totalFilesChanged = $ops.filesCreated.Count + $ops.filesModified.Count + $ops.filesDeleted.Count
    }
    catch {}

    return $ops
}

function Get-GitCommits {
    <#
    .SYNOPSIS
        Get commits made since a specific hash
    .PARAMETER SinceHash
        Git commit hash to start from
    .RETURNS
        Array of commit objects
    #>
    param([string]$SinceHash)

    $commits = @()

    try {
        $logOutput = git log --format="%H|%s|%ai|%an" "$SinceHash..HEAD" 2>$null
        if ($logOutput) {
            foreach ($line in $logOutput) {
                $parts = $line -split '\|'
                if ($parts.Count -ge 4) {
                    # Get diff stats for this commit
                    $stats = git diff --shortstat "$($parts[0])^" $parts[0] 2>$null
                    $insertions = 0
                    $deletions = 0
                    $filesChanged = 0
                    if ($stats -match "(\d+) files? changed") { $filesChanged = [int]$Matches[1] }
                    if ($stats -match "(\d+) insertions?") { $insertions = [int]$Matches[1] }
                    if ($stats -match "(\d+) deletions?") { $deletions = [int]$Matches[1] }

                    $commits += @{
                        hash = $parts[0]
                        message = $parts[1]
                        timestamp = $parts[2]
                        author = $parts[3]
                        filesChanged = $filesChanged
                        insertions = $insertions
                        deletions = $deletions
                    }
                }
            }
        }
    }
    catch {}

    return $commits
}

# Helper for JSONL append operations
function Append-Jsonl {
    <#
    .SYNOPSIS
        Append an entry to a JSONL file with standard fields (timestamp, session)
    .PARAMETER File
        Path to the JSONL file
    .PARAMETER Data
        Hashtable of data to write
    .PARAMETER SkipSessionCheck
        If set, skip session directory check
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$File,
        [Parameter(Mandatory=$true)]
        [hashtable]$Data,
        [switch]$SkipSessionCheck
    )

    if (-not $SkipSessionCheck -and (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir))) {
        return
    }

    $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); session = $script:SessionId }
    foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
    $entry | ConvertTo-Json -Compress | Add-Content -Path $File -Encoding UTF8
}

function Log-ClaudeInvocation {
    <#
    .SYNOPSIS
        Log detailed Claude CLI invocation information
    .PARAMETER Iteration
        Iteration number
    .PARAMETER ClaudePath
        Path to Claude executable
    .PARAMETER Arguments
        Array of CLI arguments
    .PARAMETER PromptFile
        Path to prompt file
    .PARAMETER PromptType
        Type of prompt (prd_generation, story_work)
    .PARAMETER ProcessId
        Process ID of Claude process
    .PARAMETER StartTime
        When execution started
    .PARAMETER EndTime
        When execution ended
    .PARAMETER ExitCode
        Process exit code
    .PARAMETER TimedOut
        Whether the process timed out
    #>
    param(
        [int]$Iteration,
        [string]$ClaudePath,
        [array]$Arguments,
        [string]$PromptFile,
        [string]$PromptType,
        [int]$ProcessId,
        [datetime]$StartTime,
        [datetime]$EndTime,
        [int]$ExitCode,
        [bool]$TimedOut
    )

    # Input validation
    if (-not $script:SessionLogDir) {
        Write-Warning "Log-ClaudeInvocation: SessionLogDir not set, skipping"
        return
    }

    $invocationFile = Join-Path $script:SessionLogDir "claude_invocation_$Iteration.json"

    # Read prompt metadata
    $promptContent = ""
    $promptLines = 0
    $promptChars = 0
    if (Test-Path $PromptFile) {
        $promptContent = Get-Content $PromptFile -Raw -ErrorAction SilentlyContinue
        if ($promptContent) {
            $promptLines = ($promptContent -split "`n").Count
            $promptChars = $promptContent.Length
        }
    }

    # Get hash of prompt for tracking
    $promptHash = ""
    if ($promptContent) {
        $md5 = [System.Security.Cryptography.MD5]::Create()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($promptContent)
        $hashBytes = $md5.ComputeHash($bytes)
        $promptHash = [BitConverter]::ToString($hashBytes) -replace '-', ''
    }

    $invocation = @{
        iteration = $Iteration
        timestamp = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
        command = @{
            executable = $ClaudePath
            resolvedPath = (Resolve-Path $ClaudePath -ErrorAction SilentlyContinue).Path
            arguments = $Arguments
            workingDirectory = $script:ProjectRoot
        }
        prompt = @{
            file = (Split-Path $PromptFile -Leaf)
            type = $PromptType
            lineCount = $promptLines
            charCount = $promptChars
            hash = $promptHash.Substring(0, [Math]::Min(16, $promptHash.Length))
        }
        execution = @{
            processId = $ProcessId
            startedAt = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            endedAt = $EndTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            durationMs = [int](($EndTime - $StartTime).TotalMilliseconds)
            exitCode = $ExitCode
            timedOut = $TimedOut
        }
    }

    Write-JsonNoBom -Path $invocationFile -Content ($invocation | ConvertTo-Json -Depth 5)
}

function Log-IterationManifest {
    <#
    .SYNOPSIS
        Create structured iteration manifest
    .PARAMETER Iteration
        Iteration number
    .PARAMETER StoryId
        Story ID being worked on
    .PARAMETER FocusArea
        Focus area
    .PARAMETER Status
        Iteration status (completed, failed, timeout)
    .PARAMETER StartTime
        When iteration started
    .PARAMETER EndTime
        When iteration ended
    .PARAMETER PromptFile
        Path to prompt file
    .PARAMETER GitBefore
        Git state before iteration
    .PARAMETER GitAfter
        Git state after iteration
    .PARAMETER FileOps
        File operations during iteration
    .PARAMETER Commits
        Git commits made during iteration
    .PARAMETER TestResults
        Test results string
    .PARAMETER TokensEstimated
        Estimated token count
    .PARAMETER RetryCount
        Retry count
    #>
    param(
        [int]$Iteration,
        [string]$StoryId,
        [string]$FocusArea,
        [string]$Status,
        [datetime]$StartTime,
        [datetime]$EndTime,
        [string]$PromptFile,
        [hashtable]$GitBefore,
        [hashtable]$GitAfter,
        [hashtable]$FileOps,
        [array]$Commits,
        [string]$TestResults,
        [int]$TokensEstimated,
        [int]$RetryCount
    )

    # Input validation
    if (-not $script:SessionLogDir) {
        Write-Warning "Log-IterationManifest: SessionLogDir not set, skipping"
        return
    }
    if ($Iteration -lt 1) {
        Write-Warning "Log-IterationManifest: Invalid iteration number: $Iteration"
        $Iteration = 1
    }

    $manifestFile = Join-Path $script:SessionLogDir "iteration_${Iteration}_manifest.json"

    # Parse test results
    $testsPassed = 0
    $testsFailed = 0
    $testsSkipped = 0
    if ($TestResults -match "(\d+)\s*passed") { $testsPassed = [int]$Matches[1] }
    if ($TestResults -match "(\d+)\s*failed") { $testsFailed = [int]$Matches[1] }
    if ($TestResults -match "(\d+)\s*skipped") { $testsSkipped = [int]$Matches[1] }

    # Get PRD info
    $sprint = 0
    $prdBranch = ""
    if (Test-Path $script:PrdFile) {
        try {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $sprint = $prd.sprintNumber
            $prdBranch = $prd.branchName
        }
        catch {}
    }

    # Use git branch as source of truth, fall back to PRD branch
    $actualBranch = if ($GitAfter -and $GitAfter.branch) { $GitAfter.branch } else { $prdBranch }

    # Calculate lines from commits
    $linesAdded = 0
    $linesDeleted = 0
    foreach ($commit in $Commits) {
        $linesAdded += $commit.insertions
        $linesDeleted += $commit.deletions
    }

    # Determine if this is focus area work vs story work
    # StoryId should be null/empty for focus area work (PRD generation)
    $isStoryWork = $StoryId -and $StoryId -match "^US-\d+"
    $effectiveStoryId = if ($isStoryWork) { $StoryId } else { $null }

    $manifest = @{
        iteration = $Iteration
        storyId = $effectiveStoryId
        focusArea = $FocusArea
        sprint = $sprint
        branch = $actualBranch
        status = $Status
        timestamps = @{
            started = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            completed = $EndTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            durationSec = [int](($EndTime - $StartTime).TotalSeconds)
        }
        prompt = @{
            file = (Split-Path $PromptFile -Leaf)
        }
        output = @{
            stdout = "claude_out_$Iteration.log"
            stderr = "claude_err_$Iteration.log"
        }
        git = @{
            beforeCommit = if ($GitBefore) { $GitBefore.hash } else { "" }
            afterCommit = if ($GitAfter) { $GitAfter.hash } else { "" }
            branch = $actualBranch
            filesCreated = if ($FileOps -and $FileOps.filesCreated) { $FileOps.filesCreated.Count } else { 0 }
            filesModified = if ($FileOps -and $FileOps.filesModified) { $FileOps.filesModified.Count } else { 0 }
            filesDeleted = if ($FileOps -and $FileOps.filesDeleted) { $FileOps.filesDeleted.Count } else { 0 }
            linesAdded = $linesAdded
            linesDeleted = $linesDeleted
            commits = $Commits
        }
        tests = @{
            passed = $testsPassed
            failed = $testsFailed
            skipped = $testsSkipped
            raw = $TestResults
        }
        metrics = @{
            tokensEstimated = $TokensEstimated
            retryCount = $RetryCount
        }
    }

    Write-JsonNoBom -Path $manifestFile -Content ($manifest | ConvertTo-Json -Depth 10)
}

function Log-FileOperations {
    <#
    .SYNOPSIS
        Log file operations to a dedicated file
    .PARAMETER Iteration
        Iteration number
    .PARAMETER FileOps
        File operations hashtable
    #>
    param(
        [int]$Iteration,
        [hashtable]$FileOps
    )

    $fileOpsFile = Join-Path $script:SessionLogDir "file_operations_$Iteration.json"

    Write-JsonNoBom -Path $fileOpsFile -Content ($FileOps | ConvertTo-Json -Depth 5)
}

function Log-GitOperations {
    <#
    .SYNOPSIS
        Log git operations to a dedicated file
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Branch
        Current branch
    .PARAMETER Commits
        Array of commits
    .PARAMETER BeforeState
        Git state before
    .PARAMETER AfterState
        Git state after
    #>
    param(
        [int]$Iteration,
        [string]$Branch,
        [array]$Commits,
        [hashtable]$BeforeState,
        [hashtable]$AfterState
    )

    $gitOpsFile = Join-Path $script:SessionLogDir "git_operations_$Iteration.json"

    $gitOps = @{
        iteration = $Iteration
        branch = $Branch
        commits = $Commits
        beforeState = @{
            hash = $BeforeState.hash
            clean = $BeforeState.clean
        }
        afterState = @{
            hash = $AfterState.hash
            clean = $AfterState.clean
        }
        totalCommits = $Commits.Count
    }

    Write-JsonNoBom -Path $gitOpsFile -Content ($gitOps | ConvertTo-Json -Depth 5)
}

function Log-StoryVerification {
    <#
    .SYNOPSIS
        Log story verification with evidence-based acceptance criteria checking (Story 2.4)
    .DESCRIPTION
        Searches Claude output and git diff for per-criterion evidence instead of
        rubber-stamping all criteria as passed. Uses Search-CriterionEvidence for
        keyword matching, completion signal detection, and diff analysis.
    .PARAMETER StoryId
        Story ID
    .PARAMETER Story
        Full story object from PRD
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Passed
        Whether story passed (exit code 0)
    .PARAMETER ClaudeOutput
        Full Claude stdout+stderr for evidence searching
    .PARAMETER DiffOutput
        Git diff output for evidence searching
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [int]$Iteration,
        [bool]$Passed,
        [string]$ClaudeOutput = "",
        [string]$DiffOutput = ""
    )

    $verificationFile = Join-Path $script:SessionLogDir "story_${StoryId}_verification.json"

    # Build acceptance criteria verification with evidence search
    $criteriaVerification = @()
    $criteriaMetCount = 0
    $criteriaTotalCount = 0

    if ($Story -and $Story.acceptanceCriteria) {
        foreach ($criterion in $Story.acceptanceCriteria) {
            $criteriaTotalCount++

            if ($Passed -and ($ClaudeOutput -or $DiffOutput)) {
                # Story 2.4: Evidence-based verification
                $evidenceResult = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $ClaudeOutput -DiffOutput $DiffOutput

                $verified = $evidenceResult.found
                $evidence = $evidenceResult.evidence
                $confidence = $evidenceResult.confidence

                if ($verified) { $criteriaMetCount++ }
            }
            elseif ($Passed) {
                # Fallback: passed but no output to search
                $verified = $true
                $evidence = "Story passed (exit code 0) but no output available for evidence search"
                $confidence = 0.5
                $criteriaMetCount++
            }
            else {
                $verified = $false
                $evidence = "Story not yet complete"
                $confidence = 0.0
            }

            $criteriaVerification += @{
                criterion = $criterion
                verified = $verified
                evidence = $evidence
                confidence = $confidence
            }
        }
    }

    $verification = @{
        storyId = $StoryId
        title = if ($Story) { $Story.title } else { "" }
        verifiedAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        iteration = $Iteration
        acceptanceCriteria = $criteriaVerification
        criteriaMet = $criteriaMetCount
        criteriaTotal = $criteriaTotalCount
        overallVerified = $Passed
        evidenceBased = ($ClaudeOutput -ne "" -or $DiffOutput -ne "")
    }

    Write-JsonNoBom -Path $verificationFile -Content ($verification | ConvertTo-Json -Depth 5)

    if ($Passed -and $criteriaTotalCount -gt 0) {
        $pct = [math]::Round(($criteriaMetCount / $criteriaTotalCount) * 100, 0)
        $color = if ($pct -ge 80) { "Green" } elseif ($pct -ge 50) { "Yellow" } else { "Red" }
        Write-Host "  Evidence: $criteriaMetCount/$criteriaTotalCount criteria verified ($pct%)" -ForegroundColor $color
    }
}

function Append-SessionTimeline {
    <#
    .SYNOPSIS
        Append an event to the session timeline
    #>
    param([string]$Event, [hashtable]$Data = @{})
    $Data.event = $Event
    Append-Jsonl -File (Join-Path $script:SessionLogDir "session_timeline.jsonl") -Data $Data
}

# ============================================================================
# QUALITY GATES & INTELLIGENCE (Phase 1)
# ============================================================================

function Invoke-QualityReview {
    <#
    .SYNOPSIS
        Invoke a separate Claude session to review story quality (Story 1.1)
    .DESCRIPTION
        When llmAsJudgeQuality flag is enabled, spawns a read-only Claude session
        that reviews the git diff against acceptance criteria and returns a score.
    .PARAMETER StoryId
        The story ID being reviewed
    .PARAMETER Story
        The story object from PRD (with acceptanceCriteria)
    .PARAMETER DiffOutput
        The git diff to review
    .RETURNS
        Hashtable with score, passed, and review details; or $null if disabled/failed
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$DiffOutput
    )

    $config = Get-RalphConfig

    # Check if review is enabled via flag AND/OR config
    $flagEnabled = $config.flags.llmAsJudgeQuality -eq $true
    $reviewEnabled = $config.review -and $config.review.enabled -eq $true
    if (-not $flagEnabled -and -not $reviewEnabled) {
        return $null
    }

    Write-Host "  Quality review starting..." -ForegroundColor Cyan

    $reviewPromptPath = Join-Path $script:RalphDir "review-prompt.md"
    if (-not (Test-Path $reviewPromptPath)) {
        Write-Host "  Warning: review-prompt.md not found, skipping review" -ForegroundColor Yellow
        return $null
    }

    $reviewInstructions = Get-Content $reviewPromptPath -Raw

    # Build criteria list
    $criteriaText = "No acceptance criteria found."
    if ($Story -and $Story.acceptanceCriteria) {
        $criteriaText = ($Story.acceptanceCriteria | ForEach-Object { "- $_" }) -join "`n"
    }

    # Truncate diff if too large (keep first 3000 lines)
    $diffLines = ($DiffOutput -split "`n")
    $truncated = ""
    if ($diffLines.Count -gt 3000) {
        $DiffOutput = ($diffLines[0..2999] -join "`n")
        $truncated = "`n[TRUNCATED: showing first 3000 of $($diffLines.Count) lines]"
    }

    # Build the review request
    $reviewRequest = @"
$reviewInstructions

## Story Being Reviewed

**ID:** $StoryId
**Title:** $(if ($Story) { $Story.title } else { "Unknown" })

### Acceptance Criteria
$criteriaText

### Git Diff
$DiffOutput$truncated

Please output your review as JSON.
"@

    # Save review prompt for debugging
    $tempPromptFile = Join-Path $script:SessionLogDir "review_prompt_$StoryId.txt"
    $reviewRequest | Set-Content $tempPromptFile -Encoding UTF8

    # Get review settings
    $timeout = if ($config.review -and $config.review.timeout) { $config.review.timeout } else { 180 }
    $minScore = if ($config.review -and $config.review.minScoreToPass) { $config.review.minScoreToPass } else { 6 }
    $model = if ($config.review -and $config.review.model) { $config.review.model } else { "sonnet" }

    try {
        $claudePath = Get-ClaudePath

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model $model -p `"$reviewRequest`""
        $psi.RedirectStandardInput = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WorkingDirectory = $script:ProjectRoot

        $process = [System.Diagnostics.Process]::Start($psi)
        $exited = $process.WaitForExit($timeout * 1000)
        $output = $process.StandardOutput.ReadToEnd()

        if (-not $exited) {
            $process.Kill()
            Write-Host "  Review timed out after ${timeout}s" -ForegroundColor Yellow
            return $null
        }

        # Try to parse JSON from output
        $jsonMatch = [regex]::Match($output, '\{[\s\S]*?"overallScore"[\s\S]*?\}')
        if ($jsonMatch.Success) {
            try {
                $review = $jsonMatch.Value | ConvertFrom-Json
            }
            catch {
                Write-Host "  Could not parse review JSON" -ForegroundColor Yellow
                $rawFile = Join-Path $script:SessionLogDir "review_${StoryId}_raw.txt"
                $output | Set-Content $rawFile -Encoding UTF8
                return $null
            }

            $score = [int]$review.overallScore
            $passed = $score -ge $minScore
            $recommendation = if ($review.recommendation) { $review.recommendation } else { if ($passed) { "pass" } else { "revise" } }

            # Log review
            $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"
            $reviewData = @{
                storyId = $StoryId
                reviewedAt = (Get-Date).ToString("o")
                overallScore = $score
                scores = $review.scores
                criteriaResults = $review.criteriaResults
                issues = $review.issues
                recommendation = $recommendation
                passed = $passed
                minScoreRequired = $minScore
            }
            Write-JsonNoBom -Path $reviewFile -Content ($reviewData | ConvertTo-Json -Depth 5)

            $scoreColor = if ($passed) { "Green" } elseif ($score -ge 4) { "Yellow" } else { "Red" }
            Write-Host "  Review score: $score/10 ($recommendation)" -ForegroundColor $scoreColor

            if ($review.issues) {
                $errorCount = @($review.issues | Where-Object { $_.severity -eq "error" }).Count
                $warnCount = @($review.issues | Where-Object { $_.severity -eq "warning" }).Count
                if ($errorCount -gt 0) { Write-Host "    Errors: $errorCount" -ForegroundColor Red }
                if ($warnCount -gt 0) { Write-Host "    Warnings: $warnCount" -ForegroundColor Yellow }
            }

            Append-SessionTimeline -Event "quality_review" -Data @{
                storyId = $StoryId
                score = $score
                passed = $passed
                recommendation = $recommendation
            }

            return @{
                score = $score
                passed = $passed
                recommendation = $recommendation
                review = $reviewData
            }
        }
        else {
            Write-Host "  Could not find review JSON in output" -ForegroundColor Yellow
            $rawFile = Join-Path $script:SessionLogDir "review_${StoryId}_raw.txt"
            $output | Set-Content $rawFile -Encoding UTF8
            return $null
        }
    }
    catch {
        Write-Host "  Review failed: $_" -ForegroundColor Yellow
        return $null
    }
}

function Get-StoryFailureContext {
    <#
    .SYNOPSIS
        Build failure context for retry prompts (Story 1.3)
    .DESCRIPTION
        When a story is being retried, extracts the error category and
        last N lines from the previous attempt's output to help Claude
        understand what went wrong.
    .PARAMETER StoryId
        The story being retried
    .PARAMETER RetryCount
        Current retry attempt number
    .RETURNS
        String with failure context to prepend to prompt, or empty string
    #>
    param(
        [string]$StoryId,
        [int]$RetryCount
    )

    if ($RetryCount -le 1) {
        return ""
    }

    $context = @()
    $context += "RETRY CONTEXT (attempt $RetryCount):"
    $context += "This story has been attempted before and failed."

    # Check for last output file
    $prevIteration = $script:IterationCount  # Current iteration (we're building prompt for next)
    $outFile = Join-Path $script:SessionLogDir "claude_stdout_$prevIteration.txt"
    $errFile = Join-Path $script:SessionLogDir "claude_stderr_$prevIteration.txt"

    $lastOutput = ""
    if (Test-Path $outFile) {
        $lastOutput = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
    }
    if (Test-Path $errFile) {
        $lastOutput += "`n" + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
    }

    # Get last 50 lines of previous output
    if ($lastOutput) {
        $lines = ($lastOutput -split "`n") | Where-Object { $_.Trim() }
        $tailLines = if ($lines.Count -gt 50) { $lines[-50..-1] } else { $lines }
        $tailText = ($tailLines -join "`n").Trim()
        if ($tailText) {
            $context += ""
            $context += "Last 50 lines from previous attempt:"
            $context += "---"
            $context += $tailText
            $context += "---"
        }
    }

    # Check for previous error category in metrics
    if (Test-Path $script:MetricsFile) {
        try {
            $metrics = Import-Csv $script:MetricsFile -ErrorAction SilentlyContinue
            $storyMetrics = $metrics | Where-Object { $_.story_id -eq $StoryId -and $_.success -eq "false" } | Select-Object -Last 1
            if ($storyMetrics -and $storyMetrics.error_category) {
                $context += "Previous error category: $($storyMetrics.error_category)"
            }
        }
        catch {}
    }

    # Check for previous verification logs
    $prevVerificationFile = Join-Path $script:SessionLogDir "story_${StoryId}_verification.json"
    if (Test-Path $prevVerificationFile) {
        try {
            $prevVerification = Get-Content $prevVerificationFile -Raw | ConvertFrom-Json
            if ($prevVerification.acceptanceCriteria) {
                $unmet = @($prevVerification.acceptanceCriteria | Where-Object { -not $_.verified })
                if ($unmet.Count -gt 0) {
                    $context += ""
                    $context += "Unmet criteria from previous attempt:"
                    foreach ($c in $unmet) {
                        $context += "  - $($c.criterion)"
                    }
                }
            }
        }
        catch {}
    }

    # Check for review feedback
    $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"
    if (Test-Path $reviewFile) {
        try {
            $review = Get-Content $reviewFile -Raw | ConvertFrom-Json
            if ($review.issues) {
                $context += ""
                $context += "Issues from quality review (score: $($review.overallScore)/10):"
                foreach ($issue in $review.issues) {
                    $context += "  [$($issue.severity)] $($issue.description)"
                }
            }
        }
        catch {}
    }

    $context += ""
    $context += "Please address these issues in this attempt."
    $context += ""

    return ($context -join "`n")
}

function Get-DiffQualityScore {
    <#
    .SYNOPSIS
        Compute quality metrics from git diff (Story 1.4)
    .DESCRIPTION
        Analyzes the git diff to compute testRatio, churnRisk, and sizeAppropriate.
    .PARAMETER DiffOutput
        Raw git diff output
    .RETURNS
        Hashtable with quality metrics
    #>
    param(
        [string]$DiffOutput
    )

    $result = @{
        testRatio = 0.0
        churnRisk = "low"
        sizeAppropriate = $true
        totalLinesChanged = 0
        testLinesChanged = 0
        implLinesChanged = 0
        filesChanged = 0
        warnings = @()
    }

    if (-not $DiffOutput) {
        return $result
    }

    $config = Get-RalphConfig
    $maxDiffLines = if ($config.quality -and $config.quality.maxDiffLines) { $config.quality.maxDiffLines } else { 800 }
    $minTestRatio = if ($config.quality -and $config.quality.minTestRatio) { $config.quality.minTestRatio } else { 0.2 }

    # Parse diff to count lines by category
    $testLines = 0
    $implLines = 0
    $currentFile = ""
    $filesChanged = 0

    foreach ($line in ($DiffOutput -split "`n")) {
        if ($line -match '^diff --git a/(.+) b/') {
            $currentFile = $Matches[1]
            $filesChanged++
        }
        elseif ($line -match '^\+[^+]' -or $line -match '^\-[^-]') {
            $isTest = $currentFile -match '(test_|\.Tests\.|_test\.|tests/|spec/)'
            if ($isTest) {
                $testLines++
            }
            else {
                $implLines++
            }
        }
    }

    $totalLines = $testLines + $implLines
    $result.totalLinesChanged = $totalLines
    $result.testLinesChanged = $testLines
    $result.implLinesChanged = $implLines
    $result.filesChanged = $filesChanged

    # Test ratio
    if ($implLines -gt 0) {
        $result.testRatio = [math]::Round($testLines / $implLines, 2)
    }
    elseif ($testLines -gt 0) {
        $result.testRatio = 1.0  # Pure test changes
    }

    # Size appropriateness
    if ($totalLines -gt $maxDiffLines) {
        $result.sizeAppropriate = $false
        $result.warnings += "Diff size ($totalLines lines) exceeds maximum ($maxDiffLines)"
    }

    # Churn risk
    if ($filesChanged -gt 10) {
        $result.churnRisk = "high"
        $result.warnings += "High file count ($filesChanged files changed)"
    }
    elseif ($filesChanged -gt 5) {
        $result.churnRisk = "medium"
    }

    # Test ratio warning
    if ($implLines -gt 20 -and $result.testRatio -lt $minTestRatio) {
        $result.warnings += "Low test ratio ($($result.testRatio)) for $implLines impl lines (minimum: $minTestRatio)"
    }

    return $result
}

function Get-TestBaseline {
    <#
    .SYNOPSIS
        Capture current test suite baseline (Story 1.5)
    .DESCRIPTION
        Runs pytest --collect-only and a quick test run to establish baseline.
        Saves to test_baseline.json.
    .RETURNS
        Hashtable with test counts, or $null on failure
    #>

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    try {
        # Quick test collection count
        $collectOutput = & python -m pytest tests/ --collect-only -q 2>&1
        $collectText = $collectOutput -join "`n"

        $totalTests = 0
        if ($collectText -match '(\d+)\s+tests?\s+collected') {
            $totalTests = [int]$Matches[1]
        }
        elseif ($collectText -match '(\d+)\s+item') {
            $totalTests = [int]$Matches[1]
        }

        # Quick test run for pass/fail baseline
        $testOutput = & python -m pytest tests/ -v --tb=no -q 2>&1
        $testText = $testOutput -join "`n"

        $passed = 0
        $failed = 0
        if ($testText -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }
        if ($testText -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }

        $baseline = @{
            capturedAt = (Get-Date).ToString("o")
            totalTests = $totalTests
            passed = $passed
            failed = $failed
            errors = 0
        }

        if ($testText -match '(\d+)\s+error') {
            $baseline.errors = [int]$Matches[1]
        }

        Write-JsonNoBom -Path $baselineFile -Content ($baseline | ConvertTo-Json -Depth 3)
        Write-Host "  Test baseline: $passed passed, $failed failed of $totalTests tests" -ForegroundColor DarkGray

        return $baseline
    }
    catch {
        Write-Host "  Could not capture test baseline: $_" -ForegroundColor Yellow
        return $null
    }
}

function Compare-TestBaseline {
    <#
    .SYNOPSIS
        Compare current test results against baseline (Story 1.5)
    .PARAMETER CurrentResults
        String like "41/41 pass" from Get-TestResults
    .RETURNS
        Hashtable with regression info, or $null if no baseline
    #>
    param(
        [string]$CurrentResults
    )

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    if (-not (Test-Path $baselineFile)) {
        return $null
    }

    $config = Get-RalphConfig
    $regressionEnabled = -not $config.regression -or $config.regression.enabled -ne $false

    if (-not $regressionEnabled) {
        return $null
    }

    try {
        $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }

    # Parse current results
    $currentPassed = 0
    $currentFailed = 0
    if ($CurrentResults -match '(\d+)/(\d+)') {
        $currentPassed = [int]$Matches[1]
    }
    if ($CurrentResults -match '(\d+)\s+fail') {
        $currentFailed = [int]$Matches[1]
    }

    $baselinePassed = [int]$baseline.passed
    $baselineFailed = [int]$baseline.failed

    $passedDelta = $currentPassed - $baselinePassed
    $failedDelta = $currentFailed - $baselineFailed

    $hasRegression = ($failedDelta -gt 0) -or ($passedDelta -lt 0 -and $baselinePassed -gt 0)

    $comparison = @{
        baselinePassed = $baselinePassed
        baselineFailed = $baselineFailed
        currentPassed = $currentPassed
        currentFailed = $currentFailed
        passedDelta = $passedDelta
        failedDelta = $failedDelta
        hasRegression = $hasRegression
        capturedAt = $baseline.capturedAt
    }

    if ($hasRegression) {
        Write-Host "  REGRESSION DETECTED: $baselinePassed -> $currentPassed passed, $baselineFailed -> $currentFailed failed" -ForegroundColor Red
    }
    else {
        $deltaStr = if ($passedDelta -gt 0) { " (+$passedDelta)" } else { "" }
        Write-Host "  Test baseline OK: $currentPassed passed$deltaStr" -ForegroundColor DarkGreen
    }

    return $comparison
}

function Update-TestBaseline {
    <#
    .SYNOPSIS
        Update test baseline after successful story (Story 1.5)
    .PARAMETER TestResults
        Current test results string from Get-TestResults
    #>
    param(
        [string]$TestResults
    )

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    if (-not $TestResults) { return }

    $passed = 0
    $failed = 0
    $total = 0

    if ($TestResults -match '(\d+)/(\d+)') {
        $passed = [int]$Matches[1]
        $total = [int]$Matches[2]
    }
    if ($TestResults -match '(\d+)\s+fail') {
        $failed = [int]$Matches[1]
    }

    if ($total -eq 0 -and $passed -eq 0) { return }

    $baseline = @{
        capturedAt = (Get-Date).ToString("o")
        totalTests = $total
        passed = $passed
        failed = $failed
        errors = 0
    }

    Write-JsonNoBom -Path $baselineFile -Content ($baseline | ConvertTo-Json -Depth 3)
}

function Import-HumanFeedback {
    <#
    .SYNOPSIS
        Read human feedback file (Story 1.6)
    .RETURNS
        Array of feedback entries, or empty array
    #>

    $feedbackFile = Join-Path $script:RalphDir "feedback.json"

    if (-not (Test-Path $feedbackFile)) {
        return @()
    }

    try {
        $data = Get-Content $feedbackFile -Raw | ConvertFrom-Json
        if ($data.entries) {
            return @($data.entries)
        }
        return @()
    }
    catch {
        Write-Host "  Warning: Could not parse feedback.json: $_" -ForegroundColor Yellow
        return @()
    }
}

function Get-FeedbackForStory {
    <#
    .SYNOPSIS
        Get relevant feedback entries for a specific story (Story 1.6)
    .PARAMETER StoryId
        The story ID
    .PARAMETER Story
        The story object
    .PARAMETER FocusArea
        Current focus area
    .RETURNS
        String with feedback context, or empty string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$FocusArea
    )

    $feedback = Import-HumanFeedback
    if ($feedback.Count -eq 0) { return "" }

    $relevant = @()

    foreach ($entry in $feedback) {
        $isRelevant = $false

        # Match by story ID
        if ($entry.storyId -and $entry.storyId -eq $StoryId) {
            $isRelevant = $true
        }
        # Match by focus area
        elseif ($entry.focusArea -and $entry.focusArea -eq $FocusArea) {
            $isRelevant = $true
        }
        # Match by keywords in story title
        elseif ($entry.keywords -and $Story -and $Story.title) {
            foreach ($kw in $entry.keywords) {
                if ($Story.title -match [regex]::Escape($kw)) {
                    $isRelevant = $true
                    break
                }
            }
        }
        # Global feedback (no filters)
        elseif (-not $entry.storyId -and -not $entry.focusArea -and -not $entry.keywords) {
            $isRelevant = $true
        }

        if ($isRelevant) {
            $relevant += $entry
        }
    }

    if ($relevant.Count -eq 0) { return "" }

    $context = @()
    $context += "HUMAN FEEDBACK (please incorporate):"
    foreach ($entry in $relevant) {
        $line = "- $($entry.feedback)"
        if ($entry.priority) { $line += " [Priority: $($entry.priority)]" }
        $context += $line
    }
    $context += ""

    return ($context -join "`n")
}

function New-SprintReport {
    <#
    .SYNOPSIS
        Generate sprint report markdown (Story 1.7)
    .PARAMETER SprintData
        Sprint metadata hashtable
    .PARAMETER Prd
        The PRD object with user stories
    .RETURNS
        Path to generated report, or $null
    #>
    param(
        [hashtable]$SprintData,
        [object]$Prd
    )

    if (-not $Prd -or -not $Prd.userStories) { return $null }

    $reportDir = $script:ArchiveDir
    $sprintNum = if ($SprintData -and $SprintData.sprintNumber) { $SprintData.sprintNumber } else { 1 }
    $reportPath = Join-Path $reportDir "sprint-${sprintNum}-REPORT.md"

    $completed = @($Prd.userStories | Where-Object { $_.passes -eq $true })
    $incomplete = @($Prd.userStories | Where-Object { $_.passes -ne $true })

    # Gather metrics from CSV
    $metricsData = @()
    if (Test-Path $script:MetricsFile) {
        $metricsData = @(Import-Csv $script:MetricsFile -ErrorAction SilentlyContinue)
    }

    # Filter to stories in this sprint
    $storyIds = @($Prd.userStories | ForEach-Object { $_.id })
    $sprintMetrics = @($metricsData | Where-Object { $storyIds -contains $_.story_id })

    # Calculate stats
    $totalDuration = ($sprintMetrics | Measure-Object -Property duration_min -Sum).Sum
    if (-not $totalDuration) { $totalDuration = 0 }
    $totalTokens = ($sprintMetrics | Measure-Object -Property tokens_used -Sum).Sum
    if (-not $totalTokens) { $totalTokens = 0 }
    $avgDuration = if ($sprintMetrics.Count -gt 0) { [math]::Round($totalDuration / $sprintMetrics.Count, 1) } else { 0 }
    $successCount = @($sprintMetrics | Where-Object { $_.success -eq "true" }).Count
    $successRate = if ($sprintMetrics.Count -gt 0) { [math]::Round($successCount / $sprintMetrics.Count * 100) } else { 0 }
    $retries = @($sprintMetrics | Where-Object { [int]$_.retry_count -gt 1 }).Count

    # Collect quality review scores from session logs
    $qualityScores = @()
    if ($script:SessionLogDir -and (Test-Path $script:SessionLogDir)) {
        Get-ChildItem -Path $script:SessionLogDir -Filter "review_US-*.json" -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                $review = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $qualityScores += $review
            }
            catch {}
        }
    }

    # Build report
    $report = @()
    $report += "# Sprint $sprintNum Report"
    $report += ""
    $report += "**Focus Area:** $($Prd.focusArea)"
    $report += "**Completed:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    $report += "**Stories:** $($completed.Count)/$($Prd.userStories.Count) completed"
    $report += ""
    $report += "## Summary"
    $report += ""
    $report += "| Metric | Value |"
    $report += "|--------|-------|"
    $report += "| Total iterations | $($sprintMetrics.Count) |"
    $report += "| Total duration | $totalDuration min |"
    $report += "| Avg per story | $avgDuration min |"
    $report += "| Success rate | $successRate% |"
    $report += "| Retries needed | $retries |"
    $report += "| Est. tokens used | $totalTokens |"
    $report += ""

    # Quality scores section
    if ($qualityScores.Count -gt 0) {
        $avgScore = [math]::Round(($qualityScores | Measure-Object -Property overallScore -Average).Average, 1)
        $report += "## Quality Scores"
        $report += ""
        $report += "Average review score: **$avgScore/10**"
        $report += ""
        $report += "| Story | Score | Recommendation |"
        $report += "|-------|-------|----------------|"
        foreach ($qs in $qualityScores) {
            $report += "| $($qs.storyId) | $($qs.overallScore)/10 | $($qs.recommendation) |"
        }
        $report += ""
    }

    # Completed stories
    $report += "## Completed Stories"
    $report += ""
    foreach ($story in $completed) {
        $report += "### $($story.id): $($story.title)"
        if ($story.acceptanceCriteria) {
            foreach ($c in $story.acceptanceCriteria) {
                $report += "- [x] $c"
            }
        }
        $report += ""
    }

    # Incomplete stories
    if ($incomplete.Count -gt 0) {
        $report += "## Incomplete Stories"
        $report += ""
        foreach ($story in $incomplete) {
            $report += "### $($story.id): $($story.title)"
            if ($story.acceptanceCriteria) {
                foreach ($c in $story.acceptanceCriteria) {
                    $report += "- [ ] $c"
                }
            }
            $report += ""
        }
    }

    # Test baseline
    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"
    if (Test-Path $baselineFile) {
        try {
            $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
            $report += "## Test Baseline"
            $report += ""
            $report += "| Metric | Value |"
            $report += "|--------|-------|"
            $report += "| Tests passing | $($baseline.passed) |"
            $report += "| Tests failing | $($baseline.failed) |"
            $report += "| Total tests | $($baseline.totalTests) |"
            $report += "| Last updated | $($baseline.capturedAt) |"
            $report += ""
        }
        catch {}
    }

    # Token budget
    $budgetStatus = Get-SprintTokenBudget
    if ($budgetStatus) {
        $report += "## Token Budget"
        $report += ""
        $report += "| Metric | Value |"
        $report += "|--------|-------|"
        $report += "| Tokens used | $($budgetStatus.totalUsed) |"
        $report += "| Budget limit | $($budgetStatus.maxTokens) |"
        $report += "| Usage | $($budgetStatus.percentUsed)% |"
        $report += ""
    }

    $reportContent = $report -join "`n"
    $reportContent | Set-Content $reportPath -Encoding UTF8

    Write-Host "  Sprint report: $reportPath" -ForegroundColor DarkGray

    return $reportPath
}

function Get-SprintTokenBudget {
    <#
    .SYNOPSIS
        Calculate token budget status for current sprint (Story 1.8)
    .RETURNS
        Hashtable with budget status, or $null if budget tracking disabled
    #>

    $config = Get-RalphConfig

    if (-not $config.budget -or $config.budget.enabled -ne $true) {
        return $null
    }

    $maxTokens = if ($config.budget.maxTokensPerSprint) { $config.budget.maxTokensPerSprint } else { 500000 }
    $warnPercent = if ($config.budget.warnAtPercent) { $config.budget.warnAtPercent } else { 80 }

    # Sum tokens from current session metrics
    $totalUsed = 0
    if (Test-Path $script:MetricsFile) {
        try {
            $metrics = @(Import-Csv $script:MetricsFile -ErrorAction SilentlyContinue)
            $sessionMetrics = @($metrics | Where-Object { $_.session -eq $script:SessionId })
            $sumResult = ($sessionMetrics | Measure-Object -Property tokens_used -Sum).Sum
            if ($sumResult) { $totalUsed = [int]$sumResult }
        }
        catch {
            $totalUsed = 0
        }
    }

    $percentUsed = if ($maxTokens -gt 0) { [math]::Round($totalUsed / $maxTokens * 100, 1) } else { 0 }
    $remaining = [math]::Max(0, $maxTokens - $totalUsed)
    $budgetExceeded = $totalUsed -ge $maxTokens
    $budgetWarning = $percentUsed -ge $warnPercent

    $status = @{
        totalUsed = $totalUsed
        maxTokens = $maxTokens
        remaining = $remaining
        percentUsed = $percentUsed
        exceeded = $budgetExceeded
        warning = $budgetWarning
    }

    if ($budgetWarning) {
        $warnColor = if ($budgetExceeded) { "Red" } else { "Yellow" }
        $warnMsg = if ($budgetExceeded) { "TOKEN BUDGET EXCEEDED" } else { "Token budget warning" }
        Write-Host "  ${warnMsg}: $totalUsed / $maxTokens ($percentUsed%)" -ForegroundColor $warnColor
    }

    return $status
}

# ============================================================================
# CORE QUALITY GATE (Phase 2)
# ============================================================================

function Format-ReviewPrompt {
    <#
    .SYNOPSIS
        Build a comprehensive review prompt for the code review agent (Story 2.1)
    .DESCRIPTION
        Creates a structured prompt with diff, acceptance criteria, file context,
        and review instructions for independent code review.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Full story object from PRD
    .PARAMETER DiffOutput
        Git diff output string
    .PARAMETER FileOps
        File operations hashtable from Get-FileOperations
    .RETURNS
        Formatted review prompt string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$DiffOutput,
        [hashtable]$FileOps = @{}
    )

    $reviewPromptFile = Join-Path $script:RalphDir "review-prompt.md"
    $basePrompt = ""
    if (Test-Path $reviewPromptFile) {
        $basePrompt = Get-Content $reviewPromptFile -Raw
    }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine($basePrompt)
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Story Under Review")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("**Story ID:** $StoryId")
    if ($Story) {
        [void]$sb.AppendLine("**Title:** $($Story.title)")
        if ($Story.acceptanceCriteria) {
            [void]$sb.AppendLine("")
            [void]$sb.AppendLine("### Acceptance Criteria")
            $i = 1
            foreach ($criterion in $Story.acceptanceCriteria) {
                [void]$sb.AppendLine("$i. $criterion")
                $i++
            }
        }
    }

    # File context
    if ($FileOps -and $FileOps.totalFilesChanged -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Files Changed")
        if ($FileOps.filesCreated) {
            foreach ($f in $FileOps.filesCreated) {
                $path = if ($f -is [hashtable]) { $f.path } else { "$f" }
                [void]$sb.AppendLine("- **Created:** $path")
            }
        }
        if ($FileOps.filesModified) {
            foreach ($f in $FileOps.filesModified) {
                $path = if ($f -is [hashtable]) { $f.path } else { "$f" }
                [void]$sb.AppendLine("- **Modified:** $path")
            }
        }
        if ($FileOps.filesDeleted) {
            foreach ($f in $FileOps.filesDeleted) {
                [void]$sb.AppendLine("- **Deleted:** $f")
            }
        }
    }

    # Diff
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("### Git Diff")
    [void]$sb.AppendLine('```diff')
    if ($DiffOutput) {
        # Truncate very large diffs
        $maxDiffChars = 15000
        if ($DiffOutput.Length -gt $maxDiffChars) {
            [void]$sb.AppendLine($DiffOutput.Substring(0, $maxDiffChars))
            [void]$sb.AppendLine("... [TRUNCATED - diff too large] ...")
        } else {
            [void]$sb.AppendLine($DiffOutput)
        }
    } else {
        [void]$sb.AppendLine("(no diff available)")
    }
    [void]$sb.AppendLine('```')

    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Review Focus Areas")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("In addition to the standard scoring criteria, pay special attention to:")
    [void]$sb.AppendLine("1. **Architecture**: Do changes follow existing patterns in the codebase?")
    [void]$sb.AppendLine("2. **Security**: Any injection risks, hardcoded secrets, unsafe operations?")
    [void]$sb.AppendLine("3. **Style**: Consistent naming, formatting, and conventions?")
    [void]$sb.AppendLine("4. **Completeness**: Are all acceptance criteria fully addressed with evidence?")
    [void]$sb.AppendLine("5. **Regressions**: Could these changes break existing functionality?")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("Output your review as the JSON format specified above.")

    return $sb.ToString()
}

function Invoke-CodeReview {
    <#
    .SYNOPSIS
        Independent code review agent (Story 2.1)
    .DESCRIPTION
        Invokes a separate Claude session with read-only tools to perform
        comprehensive code review of story changes. Uses Format-ReviewPrompt
        for structured review. Returns parsed review result.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Full story object from PRD
    .PARAMETER DiffOutput
        Git diff output string
    .PARAMETER FileOps
        File operations hashtable
    .PARAMETER ClaudeOutput
        Full Claude output from story execution
    .RETURNS
        Hashtable with score, passed, issues, or $null if review disabled/failed
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$DiffOutput,
        [hashtable]$FileOps = @{},
        [string]$ClaudeOutput = ""
    )

    $config = Get-RalphConfig

    # Check if review is enabled (use Phase 1 flag OR review.enabled)
    $reviewEnabled = $false
    if ($config.flags -and $config.flags.llmAsJudgeQuality) { $reviewEnabled = $true }
    if ($config.review -and $config.review.enabled) { $reviewEnabled = $true }

    if (-not $reviewEnabled) {
        return $null
    }

    $timeout = if ($config.review -and $config.review.timeout) { $config.review.timeout } else { 180 }
    $minScore = if ($config.review -and $config.review.minScoreToPass) { $config.review.minScoreToPass } else { 6 }
    $model = if ($config.review -and $config.review.model) { $config.review.model } else { "sonnet" }

    # Build comprehensive review prompt
    $reviewPrompt = Format-ReviewPrompt -StoryId $StoryId -Story $Story -DiffOutput $DiffOutput -FileOps $FileOps

    Write-Host "  Code review: Starting independent review..." -ForegroundColor DarkCyan

    $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"

    try {
        # Write prompt to temp file
        $promptFile = Join-Path $env:TEMP "ralph_review_${StoryId}.md"
        $reviewPrompt | Set-Content $promptFile -Encoding UTF8

        # Invoke separate Claude session (read-only)
        $reviewCmd = "claude --model $model --print `"Review the code changes described in $promptFile and output ONLY the JSON review object, no other text.`""

        $reviewProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $reviewCmd" `
            -RedirectStandardOutput (Join-Path $env:TEMP "ralph_review_out_${StoryId}.txt") `
            -RedirectStandardError (Join-Path $env:TEMP "ralph_review_err_${StoryId}.txt") `
            -NoNewWindow -PassThru

        $reviewProcess.WaitForExit($timeout * 1000) | Out-Null

        if (-not $reviewProcess.HasExited) {
            $reviewProcess.Kill()
            Write-Host "  Code review: Timed out after ${timeout}s" -ForegroundColor Yellow
            return $null
        }

        $reviewOutput = Get-Content (Join-Path $env:TEMP "ralph_review_out_${StoryId}.txt") -Raw -ErrorAction SilentlyContinue

        # Clean up temp files
        Remove-Item $promptFile -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $env:TEMP "ralph_review_out_${StoryId}.txt") -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $env:TEMP "ralph_review_err_${StoryId}.txt") -ErrorAction SilentlyContinue

        if (-not $reviewOutput) {
            Write-Host "  Code review: No output from review agent" -ForegroundColor Yellow
            return $null
        }

        # Extract JSON from output
        $jsonMatch = [regex]::Match($reviewOutput, '\{[\s\S]*"overallScore"[\s\S]*\}')
        if (-not $jsonMatch.Success) {
            Write-Host "  Code review: Could not parse JSON from review output" -ForegroundColor Yellow
            return $null
        }

        $review = $jsonMatch.Value | ConvertFrom-Json

        $score = if ($review.overallScore) { $review.overallScore } else { 0 }
        $passed = $score -ge $minScore

        $result = @{
            score = $score
            passed = $passed
            minScore = $minScore
            scores = if ($review.scores) { $review.scores } else { @{} }
            criteriaResults = if ($review.criteriaResults) { $review.criteriaResults } else { @() }
            issues = if ($review.issues) { $review.issues } else { @() }
            recommendation = if ($review.recommendation) { $review.recommendation } else { if ($passed) { "pass" } else { "revise" } }
        }

        # Save review result
        Write-JsonNoBom -Path $reviewFile -Content ($result | ConvertTo-Json -Depth 5)

        $statusColor = if ($passed) { "Green" } else { "Red" }
        $statusText = if ($passed) { "PASSED" } else { "NEEDS REVISION" }
        Write-Host "  Code review: Score $score/$minScore - $statusText" -ForegroundColor $statusColor

        if ($result.issues.Count -gt 0) {
            $errors = @($result.issues | Where-Object { $_.severity -eq "error" })
            $warnings = @($result.issues | Where-Object { $_.severity -eq "warning" })
            if ($errors.Count -gt 0) {
                Write-Host "  Code review: $($errors.Count) error(s), $($warnings.Count) warning(s)" -ForegroundColor Yellow
            }
        }

        return $result
    }
    catch {
        Write-Host "  Code review: Error - $_" -ForegroundColor Yellow
        return $null
    }
}

function Get-OptimalNextStory {
    <#
    .SYNOPSIS
        Smart story ordering (Story 2.2)
    .DESCRIPTION
        Replaces simple "first incomplete" ordering with intelligent selection
        considering failure history, dependencies, test-first preference, and
        file conflict risk.
    .PARAMETER Stories
        Array of story objects from PRD
    .PARAMETER Metrics
        Optional metrics CSV data for failure history
    .RETURNS
        The optimal next story object, or $null if all complete
    #>
    param(
        [array]$Stories,
        [object]$Metrics = $null
    )

    $config = Get-RalphConfig
    $smartEnabled = $config.ordering -and $config.ordering.smart

    if (-not $smartEnabled) {
        # Fallback: first incomplete story (original behavior)
        foreach ($story in $Stories) {
            if (-not $story.passes) {
                return $story
            }
        }
        return $null
    }

    $preferTestsFirst = -not $config.ordering -or $config.ordering.preferTestsFirst -ne $false

    # Build scoring for each incomplete story
    $candidates = @()
    foreach ($story in $Stories) {
        if ($story.passes) { continue }

        $storyId = $story.id
        $score = 100  # Base score

        # Factor 1: Failure history (penalize stories that fail repeatedly)
        $failureCount = 0
        if ($Metrics) {
            $storyMetrics = @($Metrics | Where-Object { $_.story_id -eq $storyId })
            $failureCount = @($storyMetrics | Where-Object { $_.success -eq "False" }).Count
            $score -= ($failureCount * 15)  # -15 per failure
        }

        # Factor 2: Test-first preference
        if ($preferTestsFirst) {
            $title = if ($story.title) { $story.title.ToLower() } else { "" }
            $isTestStory = $title -match '\btest\b|testing|pester|pytest|unit test|spec\b'
            if ($isTestStory) {
                $score += 20
            }
        }

        # Factor 3: Story position (slight preference for document order)
        $idx = [array]::IndexOf($Stories, $story)
        $score -= $idx  # -1 per position

        # Factor 4: File conflict risk (check against recently completed stories)
        $conflictRisk = Test-FileConflict -StoryId $storyId -Story $story
        if ($conflictRisk -and $conflictRisk.hasConflict) {
            $score -= 10  # Penalize conflicting stories
        }

        # Factor 5: Retry count (deprioritize stories being retried)
        if ($story.retryCount -and $story.retryCount -gt 0) {
            $score -= ($story.retryCount * 5)
        }

        $candidates += @{
            story = $story
            score = $score
            failureCount = $failureCount
        }
    }

    if ($candidates.Count -eq 0) {
        return $null
    }

    # Sort by score descending
    $sorted = $candidates | Sort-Object { $_.score } -Descending
    $best = $sorted[0]

    if ($best.failureCount -gt 0) {
        Write-Host "  Smart ordering: Selected $($best.story.id) (score: $($best.score), prior failures: $($best.failureCount))" -ForegroundColor DarkGray
    }

    return $best.story
}

function Get-StoryFileTouches {
    <#
    .SYNOPSIS
        Get files touched by each completed story (Story 2.3)
    .DESCRIPTION
        Reads session log data to build a map of which stories touched which files.
    .RETURNS
        Hashtable mapping storyId -> array of file paths
    #>

    $touchMap = @{}

    if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) {
        return $touchMap
    }

    # Read iteration logs for file operations
    $logFiles = Get-ChildItem -Path $script:SessionLogDir -Filter "iteration_*.json" -ErrorAction SilentlyContinue
    foreach ($logFile in $logFiles) {
        try {
            $log = Get-Content $logFile.FullName -Raw | ConvertFrom-Json
            $storyId = $log.storyId
            if (-not $storyId) { continue }

            if (-not $touchMap.ContainsKey($storyId)) {
                $touchMap[$storyId] = @()
            }

            if ($log.fileOperations) {
                if ($log.fileOperations.filesCreated) {
                    foreach ($f in $log.fileOperations.filesCreated) {
                        $path = if ($f -is [PSCustomObject] -or $f -is [hashtable]) { $f.path } else { "$f" }
                        if ($path -and $touchMap[$storyId] -notcontains $path) {
                            $touchMap[$storyId] += $path
                        }
                    }
                }
                if ($log.fileOperations.filesModified) {
                    foreach ($f in $log.fileOperations.filesModified) {
                        $path = if ($f -is [PSCustomObject] -or $f -is [hashtable]) { $f.path } else { "$f" }
                        if ($path -and $touchMap[$storyId] -notcontains $path) {
                            $touchMap[$storyId] += $path
                        }
                    }
                }
            }
        }
        catch {
            # Skip malformed log files
        }
    }

    return $touchMap
}

function Test-FileConflict {
    <#
    .SYNOPSIS
        Check if a story has file overlap with recently completed stories (Story 2.3)
    .DESCRIPTION
        Uses Get-StoryFileTouches to detect potential conflicts.
    .PARAMETER StoryId
        Story ID to check
    .PARAMETER Story
        Story object (used for keyword-based file prediction)
    .RETURNS
        Hashtable with hasConflict, overlappingFiles, overlappingStories
    #>
    param(
        [string]$StoryId,
        [object]$Story = $null
    )

    $result = @{
        hasConflict = $false
        overlappingFiles = @()
        overlappingStories = @()
    }

    $touchMap = Get-StoryFileTouches
    if ($touchMap.Count -eq 0) {
        return $result
    }

    # Get all files touched by completed stories (excluding self)
    $allTouchedFiles = @{}
    foreach ($sid in $touchMap.Keys) {
        if ($sid -eq $StoryId) { continue }
        foreach ($file in $touchMap[$sid]) {
            if (-not $allTouchedFiles.ContainsKey($file)) {
                $allTouchedFiles[$file] = @()
            }
            $allTouchedFiles[$file] += $sid
        }
    }

    # If this story has already touched files, check overlap
    if ($touchMap.ContainsKey($StoryId)) {
        foreach ($file in $touchMap[$StoryId]) {
            if ($allTouchedFiles.ContainsKey($file)) {
                $result.hasConflict = $true
                $result.overlappingFiles += $file
                foreach ($sid in $allTouchedFiles[$file]) {
                    if ($result.overlappingStories -notcontains $sid) {
                        $result.overlappingStories += $sid
                    }
                }
            }
        }
    }

    # Predictive: check if story title/criteria mention files that were recently modified
    if (-not $result.hasConflict -and $Story) {
        $storyText = ""
        if ($Story.title) { $storyText += $Story.title + " " }
        if ($Story.acceptanceCriteria) { $storyText += ($Story.acceptanceCriteria -join " ") }
        $storyText = $storyText.ToLower()

        foreach ($file in $allTouchedFiles.Keys) {
            $fileName = [System.IO.Path]::GetFileNameWithoutExtension($file).ToLower()
            if ($fileName.Length -gt 3 -and $storyText -match [regex]::Escape($fileName)) {
                $result.hasConflict = $true
                $result.overlappingFiles += "$file (predicted)"
                foreach ($sid in $allTouchedFiles[$file]) {
                    if ($result.overlappingStories -notcontains $sid) {
                        $result.overlappingStories += $sid
                    }
                }
            }
        }
    }

    if ($result.hasConflict) {
        Write-Host "  Conflict: $StoryId overlaps with $($result.overlappingStories -join ', ') on $($result.overlappingFiles.Count) file(s)" -ForegroundColor Yellow
    }

    return $result
}

function Search-CriterionEvidence {
    <#
    .SYNOPSIS
        Search Claude output and git diff for evidence of a criterion being met (Story 2.4)
    .PARAMETER Criterion
        Acceptance criterion text
    .PARAMETER ClaudeOutput
        Full Claude stdout+stderr
    .PARAMETER DiffOutput
        Git diff output
    .RETURNS
        Hashtable with found (bool), evidence (string), confidence (float)
    #>
    param(
        [string]$Criterion,
        [string]$ClaudeOutput = "",
        [string]$DiffOutput = ""
    )

    $result = @{
        found = $false
        evidence = ""
        confidence = 0.0
    }

    if (-not $Criterion) { return $result }

    $criterionLower = $Criterion.ToLower()
    $evidenceParts = @()
    $confidenceScore = 0.0

    # Extract keywords from criterion (words > 3 chars, excluding common words)
    $stopWords = @('should', 'must', 'shall', 'that', 'when', 'with', 'from', 'have', 'this', 'been', 'were', 'they', 'each', 'which', 'their', 'will', 'would', 'could', 'into', 'more', 'also', 'than', 'only', 'other', 'does', 'such')
    $keywords = @()
    foreach ($word in ($criterionLower -split '\W+')) {
        if ($word.Length -gt 3 -and $stopWords -notcontains $word) {
            $keywords += $word
        }
    }

    if ($keywords.Count -eq 0) { return $result }

    # Search in Claude output
    if ($ClaudeOutput) {
        $outputLower = $ClaudeOutput.ToLower()
        $matchedKeywords = 0
        foreach ($kw in $keywords) {
            if ($outputLower -match [regex]::Escape($kw)) {
                $matchedKeywords++
            }
        }
        $keywordRatio = if ($keywords.Count -gt 0) { $matchedKeywords / $keywords.Count } else { 0 }

        if ($keywordRatio -ge 0.5) {
            $evidenceParts += "Keywords found in Claude output ($matchedKeywords/$($keywords.Count))"
            $confidenceScore += $keywordRatio * 0.4
        }

        # Check for explicit completion signals
        $completionPatterns = @(
            'acceptance criteria.*met',
            'criterion.*satisfied',
            'implemented.*successfully',
            'all tests pass',
            'verified.*criterion',
            'criterion.*verified',
            'completed.*requirement'
        )
        foreach ($pattern in $completionPatterns) {
            if ($outputLower -match $pattern) {
                $evidenceParts += "Completion signal: $($Matches[0])"
                $confidenceScore += 0.2
                break
            }
        }
    }

    # Search in git diff
    if ($DiffOutput) {
        $diffLower = $DiffOutput.ToLower()
        $diffMatches = 0
        foreach ($kw in $keywords) {
            if ($diffLower -match [regex]::Escape($kw)) {
                $diffMatches++
            }
        }
        $diffRatio = if ($keywords.Count -gt 0) { $diffMatches / $keywords.Count } else { 0 }

        if ($diffRatio -ge 0.3) {
            $evidenceParts += "Keywords found in diff ($diffMatches/$($keywords.Count))"
            $confidenceScore += $diffRatio * 0.3
        }

        # Check for test-related evidence
        if ($criterionLower -match 'test|spec|assert|verify|check') {
            if ($diffLower -match 'def test_|it\s*\(|describe\s*\(|test\s*\(|\.Tests\.|assert|should') {
                $evidenceParts += "Test code found in diff"
                $confidenceScore += 0.2
            }
        }

        # Check for function/class creation evidence
        if ($criterionLower -match 'creat|add|implement|build|write|function|class|method') {
            if ($DiffOutput -match '^\+\s*(def |function |class |const |export )') {
                $evidenceParts += "New code definitions found in diff"
                $confidenceScore += 0.15
            }
        }
    }

    $confidenceScore = [math]::Min($confidenceScore, 1.0)

    if ($evidenceParts.Count -gt 0) {
        $result.found = $confidenceScore -ge 0.3
        $result.evidence = $evidenceParts -join "; "
        $result.confidence = [math]::Round($confidenceScore, 2)
    }
    else {
        $result.evidence = "No evidence found in output or diff"
        $result.confidence = 0.0
    }

    return $result
}

function Get-SprintRetrospective {
    <#
    .SYNOPSIS
        Analyze sprint for patterns and lessons (Story 2.5)
    .DESCRIPTION
        Reads metrics CSV and story verification data to identify:
        - Common failure patterns
        - High-churn stories
        - Retry patterns
        - Quality trends
    .PARAMETER Prd
        PRD object
    .RETURNS
        Hashtable with patterns, lessons, recommendations
    #>
    param(
        [object]$Prd = $null
    )

    $retro = @{
        failurePatterns = @()
        highChurnStories = @()
        retryPatterns = @()
        qualityTrends = @()
        lessons = @()
        recommendations = @()
    }

    $metricsFile = Join-Path $script:RalphDir "metrics.csv"
    if (-not (Test-Path $metricsFile)) {
        return $retro
    }

    try {
        $metrics = Import-Csv $metricsFile
    }
    catch {
        return $retro
    }

    if ($metrics.Count -eq 0) { return $retro }

    # Analyze failure patterns
    $failures = @($metrics | Where-Object { $_.success -eq "False" })
    $errorCats = @{}
    foreach ($f in $failures) {
        $cat = if ($f.error_category) { $f.error_category } else { "unknown" }
        if (-not $errorCats.ContainsKey($cat)) { $errorCats[$cat] = 0 }
        $errorCats[$cat]++
    }
    foreach ($cat in $errorCats.Keys) {
        if ($errorCats[$cat] -ge 2) {
            $retro.failurePatterns += @{
                category = $cat
                count = $errorCats[$cat]
                lesson = "Recurring '$cat' errors ($($errorCats[$cat])x) - consider adding targeted error handling"
            }
        }
    }

    # Analyze retry patterns
    $storyRetries = @{}
    foreach ($m in $metrics) {
        $sid = $m.story_id
        if (-not $sid) { continue }
        if (-not $storyRetries.ContainsKey($sid)) { $storyRetries[$sid] = @{ total = 0; failures = 0 } }
        $storyRetries[$sid].total++
        if ($m.success -eq "False") { $storyRetries[$sid].failures++ }
    }
    foreach ($sid in $storyRetries.Keys) {
        $data = $storyRetries[$sid]
        if ($data.failures -ge 3) {
            $retro.retryPatterns += @{
                storyId = $sid
                attempts = $data.total
                failures = $data.failures
                lesson = "Story $sid required $($data.failures) retries - may need story decomposition"
            }
        }
    }

    # High churn detection
    foreach ($m in $metrics) {
        $added = if ($m.lines_added) { [int]$m.lines_added } else { 0 }
        $deleted = if ($m.lines_deleted) { [int]$m.lines_deleted } else { 0 }
        $churn = $added + $deleted
        if ($churn -gt 500) {
            $retro.highChurnStories += @{
                storyId = $m.story_id
                linesAdded = $added
                linesDeleted = $deleted
                totalChurn = $churn
            }
        }
    }

    # Generate lessons
    if ($retro.failurePatterns.Count -gt 0) {
        $topFailure = ($retro.failurePatterns | Sort-Object { $_.count } -Descending)[0]
        $retro.lessons += "Most common failure: $($topFailure.category) ($($topFailure.count)x)"
    }

    if ($retro.retryPatterns.Count -gt 0) {
        $retro.lessons += "Stories requiring excessive retries: $(($retro.retryPatterns | ForEach-Object { $_.storyId }) -join ', ')"
    }

    $totalStories = if ($Prd -and $Prd.userStories) { $Prd.userStories.Count } else { 0 }
    $passedStories = if ($Prd -and $Prd.userStories) { @($Prd.userStories | Where-Object { $_.passes }).Count } else { 0 }
    if ($totalStories -gt 0) {
        $passRate = [math]::Round(($passedStories / $totalStories) * 100, 0)
        $retro.lessons += "Sprint pass rate: ${passRate}% ($passedStories/$totalStories)"

        if ($passRate -lt 60) {
            $retro.recommendations += "Low pass rate (${passRate}%) - consider smaller stories or more specific acceptance criteria"
        }
    }

    $timeouts = @($metrics | Where-Object { $_.timeout -eq "True" })
    if ($timeouts.Count -ge 2) {
        $retro.recommendations += "$($timeouts.Count) timeouts detected - consider increasing timeout or breaking stories into smaller tasks"
    }

    return $retro
}

function Get-RetrospectiveContext {
    <#
    .SYNOPSIS
        Format retrospective data into prompt context (Story 2.5)
    .DESCRIPTION
        Converts retrospective analysis into a string that can be injected
        into story prompts for the next sprint.
    .PARAMETER Retro
        Retrospective hashtable from Get-SprintRetrospective
    .RETURNS
        Formatted context string, or empty string if no useful data
    #>
    param(
        [hashtable]$Retro
    )

    if (-not $Retro) { return "" }

    $hasData = ($Retro.lessons.Count -gt 0) -or ($Retro.recommendations.Count -gt 0) -or ($Retro.failurePatterns.Count -gt 0)
    if (-not $hasData) { return "" }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Lessons from Previous Sprint")
    [void]$sb.AppendLine("")

    if ($Retro.lessons.Count -gt 0) {
        foreach ($lesson in $Retro.lessons) {
            [void]$sb.AppendLine("- $lesson")
        }
    }

    if ($Retro.failurePatterns.Count -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Common Failure Patterns (avoid these)")
        foreach ($pattern in $Retro.failurePatterns) {
            [void]$sb.AppendLine("- **$($pattern.category)** ($($pattern.count)x): $($pattern.lesson)")
        }
    }

    if ($Retro.recommendations.Count -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Recommendations")
        foreach ($rec in $Retro.recommendations) {
            [void]$sb.AppendLine("- $rec")
        }
    }

    return $sb.ToString()
}

# ============================================================================
# INTELLIGENCE LAYER (Phase 3)
# ============================================================================

function Get-RelevantFilesForStory {
    <#
    .SYNOPSIS
        Find files relevant to a story based on keywords (Story 3.3)
    .DESCRIPTION
        Parses story title and acceptance criteria for keywords, searches the
        codebase for matching files. Returns file paths as "start here" hints.
    .PARAMETER Story
        Story object from PRD
    .PARAMETER MaxFiles
        Maximum files to return (default 10)
    .RETURNS
        Array of file paths relevant to the story
    #>
    param(
        [object]$Story,
        [int]$MaxFiles = 10
    )

    if (-not $Story) { return @() }

    # Extract keywords from story
    $storyText = ""
    if ($Story.title) { $storyText += $Story.title + " " }
    if ($Story.acceptanceCriteria) { $storyText += ($Story.acceptanceCriteria -join " ") }

    $stopWords = @('should', 'must', 'shall', 'that', 'when', 'with', 'from', 'have', 'this', 'been', 'were', 'they', 'each', 'which', 'their', 'will', 'would', 'could', 'into', 'more', 'also', 'than', 'only', 'other', 'does', 'such', 'make', 'ensure', 'test', 'create', 'update', 'implement', 'write', 'read', 'file', 'code', 'function')
    $keywords = @()
    foreach ($word in ($storyText.ToLower() -split '\W+')) {
        if ($word.Length -gt 3 -and $stopWords -notcontains $word) {
            $keywords += $word
        }
    }

    if ($keywords.Count -eq 0) { return @() }

    $relevantFiles = @{}

    # Search Python and PowerShell source files
    $extensions = @('*.py', '*.ps1', '*.yaml', '*.json')
    $searchDirs = @('src', 'tests', 'scripts/ralph', '.')

    foreach ($dir in $searchDirs) {
        $fullDir = Join-Path $script:ProjectRoot $dir
        if (-not (Test-Path $fullDir)) { continue }

        foreach ($ext in $extensions) {
            $files = Get-ChildItem -Path $fullDir -Filter $ext -Recurse -ErrorAction SilentlyContinue -Depth 3
            foreach ($file in $files) {
                $relPath = $file.FullName.Replace($script:ProjectRoot, '').TrimStart('\', '/')
                $score = 0

                # Check filename against keywords
                $fileName = $file.BaseName.ToLower()
                foreach ($kw in $keywords) {
                    if ($fileName -match [regex]::Escape($kw)) {
                        $score += 3
                    }
                }

                # Quick content scan (first 50 lines) for keyword matches
                if ($score -gt 0 -or $keywords.Count -le 5) {
                    try {
                        $content = Get-Content $file.FullName -TotalCount 50 -ErrorAction SilentlyContinue
                        if ($content) {
                            $contentText = ($content -join " ").ToLower()
                            foreach ($kw in $keywords) {
                                if ($contentText -match [regex]::Escape($kw)) {
                                    $score += 1
                                }
                            }
                        }
                    }
                    catch {}
                }

                if ($score -gt 0) {
                    $relevantFiles[$relPath] = $score
                }
            }
        }
    }

    # Sort by score descending, return top N
    $sorted = $relevantFiles.GetEnumerator() | Sort-Object { $_.Value } -Descending | Select-Object -First $MaxFiles
    return @($sorted | ForEach-Object { $_.Key })
}

function Measure-CodebaseHealth {
    <#
    .SYNOPSIS
        Track codebase health metrics (Story 3.2)
    .DESCRIPTION
        Collects test count, pass rate, file count, complexity indicators,
        and tech debt markers. Saves to health_metrics.json.
    .RETURNS
        Hashtable with health metrics
    #>

    $config = Get-RalphConfig
    $healthEnabled = -not $config.health -or $config.health.enabled -ne $false

    if (-not $healthEnabled) { return $null }

    $health = @{
        measuredAt = (Get-Date).ToString("o")
        tests = @{ total = 0; passed = 0; failed = 0; passRate = 0.0 }
        codebase = @{ pyFiles = 0; ps1Files = 0; totalLines = 0 }
        techDebt = @{ todoCount = 0; fixmeCount = 0; hackCount = 0; complexFunctions = 0 }
    }

    # Test metrics
    try {
        $testsDir = Join-Path $script:ProjectRoot 'tests'
        $testOutput = & python -m pytest $testsDir --tb=no -q 2>&1
        $testText = $testOutput -join "`n"
        if ($testText -match '(\d+)\s+passed') { $health.tests.passed = [int]$Matches[1] }
        if ($testText -match '(\d+)\s+failed') { $health.tests.failed = [int]$Matches[1] }
        $health.tests.total = $health.tests.passed + $health.tests.failed
        $health.tests.passRate = if ($health.tests.total -gt 0) {
            [math]::Round($health.tests.passed / $health.tests.total, 2)
        } else { 0.0 }
    }
    catch {}

    # File metrics
    try {
        $pyFiles = @(Get-ChildItem -Path (Join-Path $script:ProjectRoot 'src') -Filter '*.py' -Recurse -ErrorAction SilentlyContinue)
        $ps1Files = @(Get-ChildItem -Path (Join-Path $script:ProjectRoot 'scripts/ralph') -Filter '*.ps1' -ErrorAction SilentlyContinue)
        $health.codebase.pyFiles = $pyFiles.Count
        $health.codebase.ps1Files = $ps1Files.Count

        $totalLines = 0
        foreach ($f in ($pyFiles + $ps1Files)) {
            try {
                $lineCount = (Get-Content $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
                $totalLines += $lineCount
            }
            catch {}
        }
        $health.codebase.totalLines = $totalLines
    }
    catch {}

    # Tech debt indicators
    $trackTechDebt = -not $config.health -or $config.health.trackTechDebt -ne $false
    if ($trackTechDebt) {
        try {
            $srcDir = Join-Path $script:ProjectRoot 'src'
            if (Test-Path $srcDir) {
                Push-Location $script:ProjectRoot
                try {
                    $todoCount = (git grep -c 'TODO' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $fixmeCount = (git grep -c 'FIXME' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $hackCount = (git grep -c 'HACK\|WORKAROUND' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $health.techDebt.todoCount = $todoCount
                    $health.techDebt.fixmeCount = $fixmeCount
                    $health.techDebt.hackCount = $hackCount
                }
                finally {
                    Pop-Location
                }
            }
        }
        catch {}
    }

    # Save metrics
    $healthFile = Join-Path $script:RalphDir "health_metrics.json"
    Write-JsonNoBom -Path $healthFile -Content ($health | ConvertTo-Json -Depth 5)

    return $health
}

function Compare-HealthMetrics {
    <#
    .SYNOPSIS
        Compare current health against previous snapshot (Story 3.2)
    .PARAMETER Current
        Current health metrics hashtable
    .RETURNS
        Hashtable with deltas and trend indicators
    #>
    param(
        [hashtable]$Current
    )

    if (-not $Current) { return $null }

    $healthFile = Join-Path $script:RalphDir "health_metrics.json"
    if (-not (Test-Path $healthFile)) {
        return @{ isBaseline = $true; trends = @() }
    }

    try {
        $previous = Get-Content $healthFile -Raw | ConvertFrom-Json
    }
    catch {
        return @{ isBaseline = $true; trends = @() }
    }

    $trends = @()

    # Test pass rate trend
    $prevPassRate = if ($previous.tests -and $previous.tests.passRate) { $previous.tests.passRate } else { 0 }
    $currPassRate = $Current.tests.passRate
    if ($currPassRate -lt $prevPassRate) {
        $trends += @{ metric = "testPassRate"; direction = "down"; previous = $prevPassRate; current = $currPassRate }
    }
    elseif ($currPassRate -gt $prevPassRate) {
        $trends += @{ metric = "testPassRate"; direction = "up"; previous = $prevPassRate; current = $currPassRate }
    }

    # Test count trend
    $prevTotal = if ($previous.tests -and $previous.tests.total) { $previous.tests.total } else { 0 }
    $currTotal = $Current.tests.total
    if ($currTotal -lt $prevTotal) {
        $trends += @{ metric = "testCount"; direction = "down"; previous = $prevTotal; current = $currTotal }
    }
    elseif ($currTotal -gt $prevTotal) {
        $trends += @{ metric = "testCount"; direction = "up"; previous = $prevTotal; current = $currTotal }
    }

    # Tech debt trend
    $prevDebt = if ($previous.techDebt) {
        ($previous.techDebt.todoCount + $previous.techDebt.fixmeCount + $previous.techDebt.hackCount)
    } else { 0 }
    $currDebt = $Current.techDebt.todoCount + $Current.techDebt.fixmeCount + $Current.techDebt.hackCount
    if ($currDebt -gt $prevDebt) {
        $trends += @{ metric = "techDebt"; direction = "up"; previous = $prevDebt; current = $currDebt }
    }

    return @{ isBaseline = $false; trends = $trends }
}

function Test-TokenBudget {
    <#
    .SYNOPSIS
        Check if token budget allows continuing (Story 3.4)
    .DESCRIPTION
        Blocks story start if sprint token budget exceeded.
        Used alongside Test-MaxIterations in loop conditions.
    .RETURNS
        $true if budget exceeded (should stop), $false if OK to continue
    #>

    $config = Get-RalphConfig
    $budgetEnabled = $config.budget -and $config.budget.enabled

    if (-not $budgetEnabled) {
        return $false  # No budget enforcement
    }

    $budgetStatus = Get-SprintTokenBudget

    if (-not $budgetStatus) {
        return $false
    }

    if ($budgetStatus.exceeded) {
        Write-Host ""
        Write-Host "  TOKEN BUDGET EXCEEDED: $($budgetStatus.totalUsed) / $($budgetStatus.maxTokens) tokens" -ForegroundColor Red
        Write-Host "  Sprint budget depleted. Stopping execution." -ForegroundColor Red
        Write-Host ""
        return $true
    }

    return $false
}

function Get-PromptEffectivenessHistory {
    <#
    .SYNOPSIS
        Read prompt effectiveness data from session logs (Story 3.5)
    .DESCRIPTION
        Reads prompt_effectiveness.jsonl to identify which prompt patterns
        correlate with successful outcomes.
    .RETURNS
        Array of effectiveness records
    #>

    $records = @()

    if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) {
        return $records
    }

    $effectivenessFile = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"
    if (-not (Test-Path $effectivenessFile)) {
        return $records
    }

    try {
        $lines = Get-Content $effectivenessFile -ErrorAction SilentlyContinue
        foreach ($line in $lines) {
            if ($line.Trim()) {
                try {
                    $record = $line | ConvertFrom-Json
                    $records += $record
                }
                catch {}
            }
        }
    }
    catch {}

    return $records
}

function Get-PromptRecommendation {
    <#
    .SYNOPSIS
        Recommend prompt sections based on effectiveness history (Story 3.5)
    .DESCRIPTION
        Analyzes prompt_effectiveness.jsonl to determine which sections
        (failure context, feedback, criteria, retrospective) correlate
        with success and should be emphasized.
    .RETURNS
        Hashtable with section weights and recommendations
    #>

    $config = Get-RalphConfig
    $adaptiveEnabled = $config.prompts -and $config.prompts.adaptive

    if (-not $adaptiveEnabled) {
        return @{
            useFailureContext = $true
            useFeedback = $true
            useCriteria = $true
            useRetrospective = $true
            useFileHints = $true
            maxLength = if ($config.prompts -and $config.prompts.maxLength) { $config.prompts.maxLength } else { 4000 }
        }
    }

    $history = Get-PromptEffectivenessHistory

    $recommendation = @{
        useFailureContext = $true
        useFeedback = $true
        useCriteria = $true
        useRetrospective = $true
        useFileHints = $true
        maxLength = if ($config.prompts -and $config.prompts.maxLength) { $config.prompts.maxLength } else { 4000 }
    }

    if ($history.Count -lt 5) {
        # Not enough data - use defaults
        return $recommendation
    }

    # Analyze effectiveness by prompt type
    $typeStats = @{}
    foreach ($record in $history) {
        $type = if ($record.promptType) { $record.promptType } else { "unknown" }
        if (-not $typeStats.ContainsKey($type)) {
            $typeStats[$type] = @{ total = 0; totalEffectiveness = 0.0 }
        }
        $typeStats[$type].total++
        $effectiveness = if ($record.effectiveness) { [double]$record.effectiveness } else { 0.5 }
        $typeStats[$type].totalEffectiveness += $effectiveness
    }

    # Calculate average effectiveness per type
    foreach ($type in $typeStats.Keys) {
        $avg = $typeStats[$type].totalEffectiveness / $typeStats[$type].total
        if ($avg -lt 0.3 -and $typeStats[$type].total -ge 3) {
            # This prompt type is consistently ineffective
            Write-Host "  Adaptive: '$type' prompts averaging $([math]::Round($avg, 2)) effectiveness" -ForegroundColor DarkYellow
        }
    }

    return $recommendation
}

function Build-StoryPrompt {
    <#
    .SYNOPSIS
        Adaptive prompt builder for stories (Story 3.1)
    .DESCRIPTION
        Replaces static one-liner prompt with multi-section prompt builder.
        Combines: failure context, review feedback, retrospective, file hints,
        human feedback, conflict warnings, acceptance criteria.
        Respects prompt length limits and adaptive settings.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Story object from PRD
    .PARAMETER FocusArea
        Focus area name
    .PARAMETER RetryCount
        Current retry count
    .RETURNS
        Formatted prompt string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$FocusArea = "",
        [int]$RetryCount = 0
    )

    $config = Get-RalphConfig
    $recommendation = Get-PromptRecommendation
    $maxLength = $recommendation.maxLength
    $promptParts = @()

    # Section 1: Failure context (on retries)
    if ($recommendation.useFailureContext -and $RetryCount -gt 0) {
        $failureContext = Get-StoryFailureContext -StoryId $StoryId -RetryCount $RetryCount
        if ($failureContext) {
            $promptParts += $failureContext
        }
    }

    # Section 2: Human feedback
    if ($recommendation.useFeedback) {
        $feedbackContext = Get-FeedbackForStory -StoryId $StoryId -Story $Story -FocusArea $FocusArea
        if ($feedbackContext) {
            $promptParts += $feedbackContext
        }
    }

    # Section 3: Retrospective context
    if ($recommendation.useRetrospective) {
        $retroFile = Join-Path $script:RalphDir "last_retrospective.json"
        if (Test-Path $retroFile) {
            try {
                $retro = Get-Content $retroFile -Raw | ConvertFrom-Json
                $retroContext = Get-RetrospectiveContext -Retro @{
                    lessons = @($retro.lessons)
                    failurePatterns = @($retro.failurePatterns)
                    recommendations = @($retro.recommendations)
                }
                if ($retroContext) {
                    $promptParts += $retroContext
                }
            }
            catch {}
        }
    }

    # Section 4: Conflict warnings
    $conflictResult = Test-FileConflict -StoryId $StoryId -Story $Story
    if ($conflictResult -and $conflictResult.hasConflict) {
        $promptParts += ""
        $promptParts += "## File Conflict Warning"
        $promptParts += "The following files were recently modified by other stories: $($conflictResult.overlappingFiles -join ', ')"
        $promptParts += "Take extra care when modifying these files to avoid regressions."
    }

    # Section 5: Relevant file hints
    if ($recommendation.useFileHints -and $Story) {
        $relevantFiles = Get-RelevantFilesForStory -Story $Story -MaxFiles 8
        if ($relevantFiles.Count -gt 0) {
            $promptParts += ""
            $promptParts += "## Relevant Files (start here)"
            foreach ($f in $relevantFiles) {
                $promptParts += "- $f"
            }
        }
    }

    # Section 6: Resume context from checkpoints (Phase 4, Story 4.1)
    if ($RetryCount -gt 0 -and $StoryId) {
        try {
            $storyProgress = Get-StoryProgress -StoryId $StoryId
            if ($storyProgress -and ($storyProgress.milestones.testsCreated -or $storyProgress.milestones.implementationStarted -or $storyProgress.milestones.committed)) {
                $resumeContext = Build-ResumePrompt -StoryId $StoryId -Progress $storyProgress -Story $Story
                if ($resumeContext) {
                    $promptParts += $resumeContext
                }
            }
        }
        catch {}
    }

    # Section 7: Acceptance criteria (when backpressure enabled)
    if ($recommendation.useCriteria -and $config.flags.acceptanceDrivenBackpressure -and $Story) {
        $promptParts += ""
        $promptParts += "STORY: $StoryId - $($Story.title)"
        $promptParts += ""
        if ($Story.acceptanceCriteria) {
            $promptParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
            foreach ($criterion in $Story.acceptanceCriteria) {
                $promptParts += "  [ ] $criterion"
            }
            $promptParts += ""
        }
        $promptParts += "Work on this story from scripts/ralph/prd.json. Read scripts/ralph/prompt.md for instructions."
        $promptParts += "Before marking the story as passed, verify EACH acceptance criterion above is met."
    }
    else {
        $promptParts += "Work on story $StoryId from scripts/ralph/prd.json. Read scripts/ralph/prompt.md for instructions."
    }

    $prompt = $promptParts -join "`n"

    # Enforce max length
    if ($maxLength -gt 0 -and $prompt.Length -gt $maxLength) {
        $prompt = $prompt.Substring(0, $maxLength - 50) + "`n`n[Prompt truncated to $maxLength chars]"
    }

    return $prompt
}

# ============================================================================
# ADVANCED CAPABILITIES (Phase 4)
# ============================================================================

function Get-StoryProgress {
    <#
    .SYNOPSIS
        Get current progress milestones for a story (Story 4.1)
    .DESCRIPTION
        Checks git history and session logs to determine what milestones
        have been reached: tests created, implementation started, committed.
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        Hashtable with milestone states and last checkpoint data
    #>
    param(
        [string]$StoryId
    )

    $progressFile = Join-Path $script:RalphDir "story_progress.json"
    $progress = @{
        storyId = $StoryId
        milestones = @{
            testsCreated = $false
            implementationStarted = $false
            committed = $false
            reviewPassed = $false
        }
        lastCheckpoint = $null
        lastAttemptOutput = ""
    }

    # Check saved progress
    if (Test-Path $progressFile) {
        try {
            $saved = Get-Content $progressFile -Raw | ConvertFrom-Json
            if ($saved.$StoryId) {
                $storyData = $saved.$StoryId
                if ($storyData.milestones) {
                    if ($storyData.milestones.testsCreated) { $progress.milestones.testsCreated = $true }
                    if ($storyData.milestones.implementationStarted) { $progress.milestones.implementationStarted = $true }
                    if ($storyData.milestones.committed) { $progress.milestones.committed = $true }
                    if ($storyData.milestones.reviewPassed) { $progress.milestones.reviewPassed = $true }
                }
                if ($storyData.lastCheckpoint) { $progress.lastCheckpoint = $storyData.lastCheckpoint }
            }
        }
        catch {}
    }

    # Check git log for recent commits mentioning this story
    try {
        $recentCommits = git -C $script:ProjectRoot log --oneline -10 --grep="$StoryId" 2>$null
        if ($recentCommits) {
            $progress.milestones.committed = $true
            $commitText = ($recentCommits -join " ").ToLower()
            if ($commitText -match 'test') { $progress.milestones.testsCreated = $true }
            if ($commitText -match 'implement|add|create|feat') { $progress.milestones.implementationStarted = $true }
        }
    }
    catch {}

    return $progress
}

function Save-StoryProgress {
    <#
    .SYNOPSIS
        Save story progress milestones (Story 4.1)
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Milestone
        Milestone name (testsCreated, implementationStarted, committed, completed)
    .PARAMETER Data
        Optional hashtable of additional data
    #>
    param(
        [string]$StoryId,
        [string]$Milestone,
        [hashtable]$Data = @{}
    )

    $progressFile = Join-Path $script:RalphDir "story_progress.json"
    $allProgress = @{}

    if (Test-Path $progressFile) {
        try {
            $existing = Get-Content $progressFile -Raw | ConvertFrom-Json
            # Convert PSCustomObject to hashtable
            foreach ($prop in $existing.PSObject.Properties) {
                $allProgress[$prop.Name] = $prop.Value
            }
        }
        catch {}
    }

    # Get or create story entry
    $storyEntry = $null
    if ($allProgress.ContainsKey($StoryId) -and $allProgress[$StoryId]) {
        $storyEntry = $allProgress[$StoryId]
    }

    $milestones = @{
        testsCreated = $false
        implementationStarted = $false
        committed = $false
        reviewPassed = $false
    }

    # Preserve existing milestones
    if ($storyEntry -and $storyEntry.milestones) {
        $m = $storyEntry.milestones
        if ($m.testsCreated) { $milestones.testsCreated = $true }
        if ($m.implementationStarted) { $milestones.implementationStarted = $true }
        if ($m.committed) { $milestones.committed = $true }
        if ($m.reviewPassed) { $milestones.reviewPassed = $true }
    }

    # Set the new milestone
    if ($Milestone -eq 'completed') {
        $milestones.testsCreated = $true
        $milestones.implementationStarted = $true
        $milestones.committed = $true
        $milestones.reviewPassed = $true
    }
    elseif ($milestones.ContainsKey($Milestone)) {
        $milestones[$Milestone] = $true
    }

    $allProgress[$StoryId] = @{
        milestones = $milestones
        lastCheckpoint = $Milestone
        data = $Data
    }

    Write-JsonNoBom -Path $progressFile -Content ($allProgress | ConvertTo-Json -Depth 5)
}

function Build-ResumePrompt {
    <#
    .SYNOPSIS
        Build a resume-aware prompt for partially completed stories (Story 4.1)
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Story object
    .PARAMETER Progress
        Story progress from Get-StoryProgress
    .RETURNS
        Additional prompt context for resuming, or empty string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [hashtable]$Progress
    )

    if (-not $Progress -or -not $Progress.milestones) { return "" }

    $milestones = $Progress.milestones
    $completedSteps = @()
    $remainingSteps = @()

    if ($milestones.testsCreated) { $completedSteps += "Tests already created" }
    else { $remainingSteps += "Create tests" }

    if ($milestones.implementationStarted) { $completedSteps += "Implementation started" }
    else { $remainingSteps += "Implement the feature" }

    if ($milestones.committed) { $completedSteps += "Changes committed" }
    else { $remainingSteps += "Commit changes" }

    if ($completedSteps.Count -eq 0) { return "" }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Resume Context")
    [void]$sb.AppendLine("This story was partially completed in a previous attempt.")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("### Completed Steps")
    foreach ($step in $completedSteps) {
        [void]$sb.AppendLine("- [x] $step")
    }
    if ($remainingSteps.Count -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Remaining Steps")
        foreach ($step in $remainingSteps) {
            [void]$sb.AppendLine("- [ ] $step")
        }
    }
    if ($Progress.lastCheckpoint) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("Last checkpoint: $($Progress.lastCheckpoint)")
    }

    return $sb.ToString()
}

function Get-IndependentStories {
    <#
    .SYNOPSIS
        Find stories that can run in parallel (Story 4.2)
    .DESCRIPTION
        Analyzes story titles and acceptance criteria to identify stories
        with no file overlap or dependency conflicts.
    .PARAMETER Stories
        Array of incomplete story objects
    .PARAMETER MaxConcurrent
        Maximum parallel stories (default from config)
    .RETURNS
        Array of story groups that can run concurrently
    #>
    param(
        [array]$Stories,
        [int]$MaxConcurrent = 0
    )

    $config = Get-RalphConfig
    $parallelEnabled = $config.parallel -and $config.parallel.enabled

    if (-not $parallelEnabled) { return @() }

    if ($MaxConcurrent -eq 0) {
        $MaxConcurrent = if ($config.parallel.maxConcurrent) { $config.parallel.maxConcurrent } else { 2 }
    }

    $incomplete = @($Stories | Where-Object { -not $_.passes })
    if ($incomplete.Count -lt 2) { return @() }

    # Simple heuristic: stories that don't share keywords are independent
    $groups = @()
    $used = @{}

    for ($i = 0; $i -lt $incomplete.Count -and $groups.Count -lt $MaxConcurrent; $i++) {
        $story = $incomplete[$i]
        if ($used.ContainsKey($story.id)) { continue }

        # Check against already-grouped stories
        $hasConflict = $false
        foreach ($group in $groups) {
            $conflict = Test-FileConflict -StoryId $story.id -Story $story
            if ($conflict -and $conflict.hasConflict) {
                $hasConflict = $true
                break
            }
        }

        if (-not $hasConflict) {
            $groups += $story
            $used[$story.id] = $true
        }
    }

    return $groups
}

function Update-LearningDb {
    <#
    .SYNOPSIS
        Update cross-session learning database (Story 4.3)
    .DESCRIPTION
        Records failure patterns, file affinities, quality correlations
        across sessions for long-term learning.
    .PARAMETER Entry
        Learning entry hashtable with type, data, session info
    #>
    param(
        [hashtable]$Entry
    )

    $dbFile = Join-Path $script:RalphDir "learning_db.json"

    $db = @{ entries = @(); lastUpdated = "" }
    if (Test-Path $dbFile) {
        try {
            $existing = Get-Content $dbFile -Raw | ConvertFrom-Json
            if ($existing.entries) {
                $db.entries = @($existing.entries)
            }
        }
        catch {}
    }

    $Entry.recordedAt = (Get-Date).ToString("o")
    $db.entries += $Entry
    $db.lastUpdated = (Get-Date).ToString("o")

    # Keep last 500 entries to prevent unbounded growth
    if ($db.entries.Count -gt 500) {
        $db.entries = $db.entries | Select-Object -Last 500
    }

    Write-JsonNoBom -Path $dbFile -Content ($db | ConvertTo-Json -Depth 10)
}

function Get-LearningContext {
    <#
    .SYNOPSIS
        Get relevant learning context for current story (Story 4.3)
    .DESCRIPTION
        Queries learning_db.json for patterns relevant to the current
        story's focus area, error category, or file patterns.
    .PARAMETER FocusArea
        Current focus area
    .PARAMETER ErrorCategory
        Optional error category to find similar past failures
    .RETURNS
        Array of relevant learning entries
    #>
    param(
        [string]$FocusArea = "",
        [string]$ErrorCategory = ""
    )

    $dbFile = Join-Path $script:RalphDir "learning_db.json"
    if (-not (Test-Path $dbFile)) { return @() }

    try {
        $db = Get-Content $dbFile -Raw | ConvertFrom-Json
    }
    catch { return @() }

    if (-not $db.entries -or $db.entries.Count -eq 0) { return @() }

    $relevant = @()

    foreach ($entry in $db.entries) {
        $isRelevant = $false

        # Match by focus area
        if ($FocusArea -and $entry.focusArea -eq $FocusArea) {
            $isRelevant = $true
        }

        # Match by error category
        if ($ErrorCategory -and $entry.errorCategory -eq $ErrorCategory) {
            $isRelevant = $true
        }

        # Match by type (failure patterns are always relevant)
        if ($entry.type -eq 'failure_pattern') {
            $isRelevant = $true
        }

        if ($isRelevant) {
            $relevant += $entry
        }
    }

    # Return most recent 10
    return @($relevant | Select-Object -Last 10)
}

function Test-CanRollback {
    <#
    .SYNOPSIS
        Check if a story's changes can be safely rolled back (Story 4.4)
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        Hashtable with canRollback (bool), commitHash, reason
    #>
    param(
        [string]$StoryId
    )

    $result = @{
        canRollback = $false
        commitHash = ""
        reason = ""
    }

    $config = Get-RalphConfig
    $autoRollback = $config.regression -and $config.regression.autoRollback

    if (-not $autoRollback) {
        $result.reason = "Auto-rollback disabled in config"
        return $result
    }

    # Find the commit for this story
    try {
        $commit = git log --oneline -1 --grep="$StoryId" --format="%H" 2>$null
        if (-not $commit) {
            $result.reason = "No commit found for story $StoryId"
            return $result
        }

        $result.commitHash = $commit.Trim()

        # Check if this is the most recent commit (can only revert HEAD safely)
        $headHash = (git rev-parse HEAD 2>$null).Trim()
        if ($result.commitHash -ne $headHash) {
            $result.reason = "Story commit is not HEAD - cannot safely revert"
            return $result
        }

        # Check for uncommitted changes
        $status = git status --porcelain 2>$null
        if ($status) {
            $result.reason = "Uncommitted changes present - cannot safely revert"
            return $result
        }

        $result.canRollback = $true
        $result.reason = "Can safely revert commit $($result.commitHash.Substring(0, 7))"
    }
    catch {
        $result.reason = "Error checking rollback: $_"
    }

    return $result
}

function Invoke-StoryRollback {
    <#
    .SYNOPSIS
        Rollback a story's changes via git revert (Story 4.4)
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        $true if rollback succeeded, $false otherwise
    #>
    param(
        [string]$StoryId
    )

    $canRollback = Test-CanRollback -StoryId $StoryId

    if (-not $canRollback.canRollback) {
        Write-Host "  Rollback: Cannot rollback $StoryId - $($canRollback.reason)" -ForegroundColor Yellow
        return $false
    }

    Write-Host "  Rollback: Reverting commit for $StoryId..." -ForegroundColor Yellow

    try {
        $revertOutput = git revert --no-edit HEAD 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Rollback: Successfully reverted $StoryId" -ForegroundColor Green

            # Record in learning DB
            Update-LearningDb -Entry @{
                type = 'rollback'
                storyId = $StoryId
                commitHash = $canRollback.commitHash
                reason = 'Regression detected'
            }

            return $true
        }
        else {
            Write-Host "  Rollback: git revert failed: $revertOutput" -ForegroundColor Red
            # Abort the failed revert
            git revert --abort 2>$null
            return $false
        }
    }
    catch {
        Write-Host "  Rollback: Error - $_" -ForegroundColor Red
        git revert --abort 2>$null
        return $false
    }
}

function Build-DependencyGraph {
    <#
    .SYNOPSIS
        Build a dependency graph for stories (Story 4.5)
    .DESCRIPTION
        Parses stories for explicit dependencies (dependsOn field) and
        implicit dependencies (shared keywords, file patterns).
    .PARAMETER Stories
        Array of story objects from PRD
    .RETURNS
        Hashtable mapping storyId -> array of dependency storyIds
    #>
    param(
        [array]$Stories
    )

    $graph = @{}

    if (-not $Stories -or $Stories.Count -eq 0) { return $graph }

    foreach ($story in $Stories) {
        $storyId = $story.id
        $graph[$storyId] = @()

        # Explicit dependencies
        if ($story.dependsOn) {
            foreach ($dep in $story.dependsOn) {
                if ($graph[$storyId] -notcontains $dep) {
                    $graph[$storyId] += $dep
                }
            }
        }

        # Implicit: if story title mentions another story's output
        $titleLower = if ($story.title) { $story.title.ToLower() } else { "" }
        foreach ($other in $Stories) {
            if ($other.id -eq $storyId) { continue }
            $otherTitle = if ($other.title) { $other.title.ToLower() } else { "" }

            # Check if this story's title references concepts from another story
            # Simple heuristic: if story mentions "test" and another creates the function
            if ($titleLower -match 'test.*for|verify|validate' -and $otherTitle -match 'create|implement|add|build') {
                # Extract key nouns
                $testSubject = ($titleLower -replace 'test.*for\s*', '' -replace 'verify\s*', '' -replace 'validate\s*', '').Trim()
                if ($testSubject.Length -gt 3 -and $otherTitle -match [regex]::Escape($testSubject.Substring(0, [math]::Min($testSubject.Length, 10)))) {
                    if ($graph[$storyId] -notcontains $other.id) {
                        $graph[$storyId] += $other.id
                    }
                }
            }
        }
    }

    return $graph
}

function Get-ExecutableStories {
    <#
    .SYNOPSIS
        Get stories whose dependencies are all met (Story 4.5)
    .DESCRIPTION
        Uses Build-DependencyGraph to find stories that can be executed
        (all dependencies are passed).
    .PARAMETER Stories
        Array of story objects
    .RETURNS
        Array of executable story objects
    #>
    param(
        [array]$Stories
    )

    if (-not $Stories -or $Stories.Count -eq 0) { return @() }

    $graph = Build-DependencyGraph -Stories $Stories
    $passedIds = @($Stories | Where-Object { $_.passes } | ForEach-Object { $_.id })

    $executable = @()
    foreach ($story in $Stories) {
        if ($story.passes) { continue }

        $deps = $graph[$story.id]
        $allDepsMet = $true

        if ($deps -and $deps.Count -gt 0) {
            foreach ($dep in $deps) {
                if ($passedIds -notcontains $dep) {
                    $allDepsMet = $false
                    break
                }
            }
        }

        if ($allDepsMet) {
            $executable += $story
        }
    }

    return $executable
}

# ============================================================================
# STATE MACHINE LOGGING (Phase 2 - Task 2.5)
# ============================================================================

function Log-StateTransition {
    <#
    .SYNOPSIS
        Log state machine transitions for debugging
    #>
    param([string]$From, [string]$To, [string]$Reason, [hashtable]$Context = @{})
    $Context.from = $From; $Context.to = $To; $Context.reason = $Reason; $Context.iteration = $script:IterationCount
    Append-Jsonl -File (Join-Path $script:SessionLogDir "state_transitions.jsonl") -Data $Context
}

# ============================================================================
# PHASE TIMING (Phase 2 - Task 2.1)
# ============================================================================

function Measure-PhaseTimings {
    <#
    .SYNOPSIS
        Estimate phase timings from Claude output
    .PARAMETER Output
        Claude's output text
    .PARAMETER TotalDurationMs
        Total execution time in milliseconds
    .RETURNS
        Hashtable with estimated phase timings
    #>
    param(
        [string]$Output,
        [int]$TotalDurationMs
    )

    $timings = @{
        read_ms = 0
        analyze_ms = 0
        implement_ms = 0
        test_ms = 0
        commit_ms = 0
    }

    if (-not $Output -or $TotalDurationMs -le 0) {
        return $timings
    }

    # Estimate based on output content patterns
    $hasReadOps = $Output -match "Read tool|Reading file|file_path"
    $hasAnalysis = $Output -match "analy|understand|plan|think|consider"
    $hasImplement = $Output -match "Edit tool|Write tool|Editing|Writing|implement"
    $hasTests = $Output -match "pytest|test.*pass|test.*fail|running tests"
    $hasGit = $Output -match "git commit|git add|Bash.*git"

    # Count pattern occurrences to weight phases
    $readWeight = if ($hasReadOps) { ([regex]::Matches($Output, "Read tool|Reading file")).Count + 1 } else { 0 }
    $analyzeWeight = if ($hasAnalysis) { 2 } else { 1 }  # Analysis always happens
    $implementWeight = if ($hasImplement) { ([regex]::Matches($Output, "Edit tool|Write tool")).Count + 1 } else { 0 }
    $testWeight = if ($hasTests) { 3 } else { 0 }  # Tests take significant time
    $gitWeight = if ($hasGit) { 1 } else { 0 }

    $totalWeight = [math]::Max(1, $readWeight + $analyzeWeight + $implementWeight + $testWeight + $gitWeight)

    # Distribute time based on weights
    $timings.read_ms = [int](($readWeight / $totalWeight) * $TotalDurationMs)
    $timings.analyze_ms = [int](($analyzeWeight / $totalWeight) * $TotalDurationMs)
    $timings.implement_ms = [int](($implementWeight / $totalWeight) * $TotalDurationMs)
    $timings.test_ms = [int](($testWeight / $totalWeight) * $TotalDurationMs)
    $timings.commit_ms = [int](($gitWeight / $totalWeight) * $TotalDurationMs)

    return $timings
}

# ============================================================================
# ERROR EVOLUTION (Phase 2 - Task 2.4)
# ============================================================================

function Log-ErrorEvolution {
    <#
    .SYNOPSIS
        Track error patterns over time
    #>
    param([string]$ErrorCategory, [string]$ErrorDetails = "", [int]$Iteration)
    Append-Jsonl -File (Join-Path $script:SessionLogDir "error_evolution.jsonl") -Data @{
        category = $ErrorCategory; details = $ErrorDetails; iteration = $Iteration
    }
}

# ============================================================================
# PHASE 3: DETAILED TRACKING
# ============================================================================

# Task 3.1: Configuration Change Audit Trail
function Log-ConfigChange {
    <#
    .SYNOPSIS
        Log configuration changes for audit trail
    #>
    param([string]$Field, $OldValue, $NewValue, [string]$Reason = "manual")
    Append-Jsonl -File (Join-Path $script:RalphDir "config_audit.jsonl") -SkipSessionCheck -Data @{
        field = $Field; old = $OldValue; new = $NewValue; reason = $Reason
    }
}

# Task 3.2: Test Failure Detail Logging
function Log-TestDetails {
    <#
    .SYNOPSIS
        Parse and log detailed test results from Claude output
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Output
        Claude's output containing test results
    #>
    param(
        [int]$Iteration,
        [string]$Output
    )

    $testDetailsFile = Join-Path $script:SessionLogDir "test_details_$Iteration.json"

    # Parse pytest output for individual test results
    $testResults = @()
    $passed = 0
    $failed = 0
    $skipped = 0
    $errors = 0

    # Match pytest verbose output: test_file.py::test_name PASSED/FAILED
    $testMatches = [regex]::Matches($Output, '([\w\/]+\.py::\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[\s*(\d+)%\])?')
    foreach ($match in $testMatches) {
        $testName = $match.Groups[1].Value
        $status = $match.Groups[2].Value.ToLower()

        $testResults += @{
            name = $testName
            status = $status
        }

        switch ($status) {
            "passed" { $passed++ }
            "failed" { $failed++ }
            "skipped" { $skipped++ }
            "error" { $errors++ }
        }
    }

    # Extract duration if present
    $duration = 0
    if ($Output -match "passed.*in\s+([\d.]+)s") {
        $duration = [double]$Matches[1]
    }

    $testDetails = @{
        iteration = $Iteration
        testRun = @{
            command = "pytest"
            duration = $duration
            exitCode = if ($failed -eq 0 -and $errors -eq 0) { 0 } else { 1 }
        }
        results = $testResults
        summary = @{
            passed = $passed
            failed = $failed
            skipped = $skipped
            errors = $errors
            total = $passed + $failed + $skipped + $errors
        }
    }

    Write-JsonNoBom -Path $testDetailsFile -Content ($testDetails | ConvertTo-Json -Depth 5)
    return $testDetails
}

# Task 3.3: Resource Usage Monitoring
function Get-ProcessMetrics {
    <#
    .SYNOPSIS
        Get resource usage metrics for a process
    .PARAMETER ProcessId
        Process ID to monitor
    .RETURNS
        Hashtable with CPU, memory, handles, threads
    #>
    param([int]$ProcessId)

    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            return @{
                cpu = [math]::Round($proc.CPU, 2)
                memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
                handles = $proc.HandleCount
                threads = $proc.Threads.Count
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
            }
        }
    }
    catch {}

    return @{ cpu = 0; memoryMB = 0; handles = 0; threads = 0; timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ") }
}

function Log-ResourceUsage {
    <#
    .SYNOPSIS
        Log resource usage for an iteration
    .PARAMETER Iteration
        Iteration number
    .PARAMETER ProcessId
        Process ID that was monitored
    .PARAMETER Samples
        Array of resource samples
    #>
    param(
        [int]$Iteration,
        [int]$ProcessId,
        [array]$Samples
    )

    $resourceFile = Join-Path $script:SessionLogDir "resource_usage_$Iteration.json"

    # Calculate averages and peaks
    $avgCpu = if ($Samples.Count -gt 0) { [math]::Round(($Samples | ForEach-Object { $_.cpu } | Measure-Object -Average).Average, 2) } else { 0 }
    $avgMem = if ($Samples.Count -gt 0) { [math]::Round(($Samples | ForEach-Object { $_.memoryMB } | Measure-Object -Average).Average, 2) } else { 0 }
    $peakMem = if ($Samples.Count -gt 0) { ($Samples | ForEach-Object { $_.memoryMB } | Measure-Object -Maximum).Maximum } else { 0 }

    $usage = @{
        iteration = $Iteration
        processId = $ProcessId
        sampleCount = $Samples.Count
        averages = @{
            cpuSeconds = $avgCpu
            memoryMB = $avgMem
        }
        peaks = @{
            memoryMB = $peakMem
        }
        samples = $Samples
    }

    Write-JsonNoBom -Path $resourceFile -Content ($usage | ConvertTo-Json -Depth 5)
}

# Task 3.4: Prompt Effectiveness Scoring
function Get-PromptEffectiveness {
    <#
    .SYNOPSIS
        Calculate prompt effectiveness score
    .PARAMETER Success
        Whether iteration succeeded
    .PARAMETER RetryCount
        Number of retries attempted
    .RETURNS
        Effectiveness score (0.0 to 1.0)
    #>
    param(
        [bool]$Success,
        [int]$RetryCount
    )

    if (-not $Success) {
        return 0.0
    }

    if ($RetryCount -le 1) {
        return 1.0  # First try success
    }
    elseif ($RetryCount -le 3) {
        return 0.5  # Success after few retries
    }
    else {
        return 0.25  # Success after many retries
    }
}

function Log-PromptEffectiveness {
    <#
    .SYNOPSIS
        Log prompt effectiveness for analysis
    #>
    param([int]$Iteration, [string]$PromptType, [double]$Effectiveness, [string]$PromptHash)
    Append-Jsonl -File (Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl") -Data @{
        iteration = $Iteration; promptType = $PromptType; effectiveness = $Effectiveness; promptHash = $PromptHash
    }
}

# Task 3.5: Skip/Blocker Tracking
function Log-Skip {
    <#
    .SYNOPSIS
        Log when a story/focus area is skipped
    #>
    param([string]$ItemId, [string]$ItemType = "story", [string]$Reason, [string]$BlockerType = "manual")
    Append-Jsonl -File (Join-Path $script:SessionLogDir "skips_blockers.jsonl") -Data @{
        itemId = $ItemId; itemType = $ItemType; reason = $Reason; blockerType = $BlockerType; resolved = $false
    }
    Append-SessionTimeline -Event "item_skipped" -Data @{ itemId = $ItemId; reason = $Reason }
}

# ============================================================================
# CLAUDE INVOCATION
# ============================================================================

function Invoke-ClaudeExploration {
    <#
    .SYNOPSIS
        Invoke Claude for exploration with read-only or full tools
    .PARAMETER Prompt
        The exploration prompt to send
    .PARAMETER FullExplore
        If set, uses full exploration tools (Task, Read, Glob, Grep, Bash, Write)
        Otherwise uses lightweight read-only tools (Read, Glob, Grep)
    .RETURNS
        The exploration output string
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,
        [switch]$FullExplore
    )

    $claudePath = Get-ClaudePath
    $tempFile = [System.IO.Path]::GetTempFileName()

    try {
        $Prompt | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline

        $claudeArgs = @("--print", "--dangerously-skip-permissions")

        if ($FullExplore) {
            # Full exploration with Task/Explore agent access
            $claudeArgs += "--allowedTools=Task,Read,Glob,Grep,Bash,Write"
        } else {
            # Light exploration - read-only, fast
            # Use haiku for speed/cost efficiency if configured
            $explorationConfig = $script:Config.exploration
            if ($explorationConfig -and $explorationConfig.periodic -and $explorationConfig.periodic.useHaiku) {
                $claudeArgs += "--model"
                $claudeArgs += "haiku"
            }
            $claudeArgs += "--allowedTools=Read,Glob,Grep"
        }

        # Create process
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = $claudeArgs -join ' '
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        # Async output capture (prevents pipe buffer deadlock)
        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            # Send prompt via stdin
            $process.StandardInput.Write($Prompt)
            $process.StandardInput.Close()

            # Wait with timeout (5 minutes for exploration)
            $timeoutMs = 300000
            $completed = $process.WaitForExit($timeoutMs)

            if (-not $completed) {
                $process.Kill()
                Write-Host "  Exploration timed out after 5 minutes" -ForegroundColor Yellow
            }

            # Small delay to let async handlers flush
            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()
            $stderr = $errBuilder.ToString()

            if ($stderr) {
                $output += "`n$stderr"
            }

            return $output
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Host "  Exploration failed: $_" -ForegroundColor Red
        return ""
    }
    finally {
        Remove-Item $tempFile -ErrorAction SilentlyContinue
    }
}

function Invoke-FocusAreaExploration {
    <#
    .SYNOPSIS
        Perform exploration for a focus area and cache results
    .PARAMETER FocusArea
        The focus area ID to explore
    .PARAMETER Reason
        Why exploration is being triggered: "interval", "git_changes", "sprint_start"
    .PARAMETER FullExplore
        If set, uses full exploration (for sprint start)
    .RETURNS
        The exploration summary string
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FocusArea,
        [Parameter(Mandatory=$true)]
        [string]$Reason,
        [switch]$FullExplore
    )

    $config = $script:Config

    # Get relevant files for this focus area from config
    $relevantFiles = @()
    if ($config.relevantFiles -and $config.relevantFiles.$FocusArea) {
        $relevantFiles = $config.relevantFiles.$FocusArea
    }

    # Get test patterns for this focus area
    $testPatterns = @()
    if ($config.testPatterns -and $config.testPatterns.$FocusArea) {
        $testPatterns = $config.testPatterns.$FocusArea
    }

    # Get recent git changes in these files
    $recentChanges = ""
    try {
        $filePatterns = $relevantFiles -join " "
        if ($filePatterns) {
            $recentChanges = git diff --stat HEAD~5 -- $relevantFiles 2>$null
            if (-not $recentChanges) { $recentChanges = "(no recent changes)" }
        }
    }
    catch {
        $recentChanges = "(could not retrieve git changes)"
    }

    # Build exploration prompt
    if ($FullExplore) {
        # Full sprint-start exploration
        $explorationPrompt = @"
MANDATORY EXPLORATION - Focus Area: $FocusArea
Reason: $Reason

You MUST explore this focus area before generating stories.

## Step 1: Explore relevant files
Use the Task tool with subagent_type=Explore to scan these files/patterns:
$($relevantFiles -join "`n")

## Step 2: Check recent changes
Run: git log --oneline -10 -- $($relevantFiles -join " ")

Recent diff stats:
$recentChanges

## Step 3: Run relevant tests to see current state
$(if ($testPatterns.Count -gt 0) { "Run: pytest tests/$($testPatterns[0]) -v --tb=short 2>&1 | head -50" } else { "No specific test patterns configured for this focus area" })

## Step 4: Summarize findings
After exploration, write a summary to: scripts/ralph/exploration_context.md

Include:
- Current implementation state (what exists, what's missing)
- Recent changes and their impact
- Test status (passing/failing, coverage gaps)
- Technical debt or issues noticed
- Patterns to follow when implementing stories

This exploration context will be used for PRD generation.
Keep the summary focused and actionable (under 1000 words).
"@
    }
    else {
        # Lightweight periodic exploration
        $explorationPrompt = @"
EXPLORATION PHASE - Focus Area: $FocusArea
Reason: $Reason

Quickly scan these files for the current state:
$($relevantFiles -join "`n")

Recent changes:
$recentChanges

Summarize in 3-5 bullet points:
- Current implementation state
- Any issues or TODOs noticed
- Key patterns to follow
- Anything that might affect upcoming stories

Keep response under 500 words. This is context-gathering, not implementation.
"@
    }

    Write-Host "  Running exploration ($Reason)..." -ForegroundColor Cyan

    # Invoke Claude
    $result = Invoke-ClaudeExploration -Prompt $explorationPrompt -FullExplore:$FullExplore

    # Cache the summary for use in story prompts
    $script:LastExplorationSummary = $result
    $script:LastExplorationTime = Get-Date

    # For sprint-start, also update the context file
    if ($FullExplore -and $result) {
        # Try to read what Claude wrote to the file
        if (Test-Path $script:ExplorationContextFile) {
            $script:SprintExplorationContext = Get-Content $script:ExplorationContextFile -Raw -ErrorAction SilentlyContinue
        }
        if (-not $script:SprintExplorationContext) {
            # Fall back to the output if file wasn't written
            $script:SprintExplorationContext = $result
        }
    }

    Write-Host "  Exploration complete" -ForegroundColor Green

    return $result
}

function Test-ShouldExplore {
    <#
    .SYNOPSIS
        Check if periodic exploration should be triggered
    .PARAMETER FocusArea
        The current focus area to check for git changes
    .RETURNS
        Hashtable with ShouldExplore (bool) and Reason (string)
    #>
    param(
        [string]$FocusArea
    )

    $config = $script:Config
    $explorationConfig = $config.exploration

    # Check if exploration is enabled
    if (-not $explorationConfig -or -not $explorationConfig.enabled) {
        return @{ ShouldExplore = $false; Reason = "exploration_disabled" }
    }

    if (-not $explorationConfig.periodic -or -not $explorationConfig.periodic.enabled) {
        return @{ ShouldExplore = $false; Reason = "periodic_disabled" }
    }

    # Trigger 1: Interval-based (every N stories)
    $intervalStories = $explorationConfig.periodic.intervalStories
    if (-not $intervalStories) { $intervalStories = 3 }

    if ($script:StoriesSinceExploration -ge $intervalStories) {
        return @{ ShouldExplore = $true; Reason = "interval" }
    }

    # Trigger 2: Git changes in focus area files (since last exploration)
    if ($explorationConfig.periodic.triggerOnGitChanges -and $FocusArea) {
        $relevantFiles = @()
        if ($config.relevantFiles -and $config.relevantFiles.$FocusArea) {
            $relevantFiles = $config.relevantFiles.$FocusArea
        }

        if ($relevantFiles.Count -gt 0 -and $script:LastExplorationCommit) {
            try {
                # Compare against the commit from last exploration, not HEAD~1
                # This prevents triggering on every iteration since Ralph commits after each story
                $changedFiles = git diff --name-only $script:LastExplorationCommit HEAD 2>$null
                if ($changedFiles) {
                    foreach ($changed in ($changedFiles -split "`n")) {
                        foreach ($pattern in $relevantFiles) {
                            if ($changed -like $pattern -or $changed -like "*$pattern*") {
                                return @{ ShouldExplore = $true; Reason = "git_changes" }
                            }
                        }
                    }
                }
            }
            catch {
                # Ignore git errors
            }
        }
    }

    return @{ ShouldExplore = $false; Reason = "not_needed" }
}

function Invoke-PeriodicExplorationIfNeeded {
    <#
    .SYNOPSIS
        Check and perform periodic exploration after story completion
    .PARAMETER FocusArea
        The current focus area
    .RETURNS
        $true if exploration was performed, $false otherwise
    #>
    param(
        [string]$FocusArea
    )

    # Increment story counter
    $script:StoriesSinceExploration++

    # Check if exploration needed
    $check = Test-ShouldExplore -FocusArea $FocusArea

    if ($check.ShouldExplore) {
        Write-Host ""
        Write-Host ">>> Periodic Exploration triggered ($($check.Reason))" -ForegroundColor Cyan
        Write-Host ""

        # Perform lightweight exploration
        $result = Invoke-FocusAreaExploration -FocusArea $FocusArea -Reason $check.Reason

        # Reset counter and update git baseline
        $script:StoriesSinceExploration = 0
        $script:LastExplorationCommit = (git rev-parse HEAD 2>$null)

        return $true
    }

    return $false
}

function Invoke-ClaudeProcess {
    <#
    .SYNOPSIS
        Core Claude process execution with logging and monitoring.
        Shared by Invoke-ClaudeForFocusArea and Invoke-ClaudeForStory.
    .PARAMETER Prompt
        The prompt to send to Claude
    .PARAMETER PromptType
        Type of prompt for logging (prd_generation, focus_area_work, story_work)
    .PARAMETER Identifier
        The story ID or focus area ID being worked on
    .PARAMETER FocusArea
        The focus area context (optional, for story work)
    .PARAMETER StoryObj
        The story object from PRD (optional, for story verification)
    .PARAMETER AllowedTools
        If set, pass --allowedTools flag to Claude
    .RETURNS
        $true if iteration succeeded, $false otherwise
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,
        [Parameter(Mandatory=$true)]
        [string]$PromptType,
        [Parameter(Mandatory=$true)]
        [string]$Identifier,
        [string]$FocusArea = "",
        [object]$StoryObj = $null,
        [switch]$AllowedTools
    )

    $isStoryWork = $PromptType -eq "story_work"
    $storyId = if ($isStoryWork) { $Identifier } else { $null }
    $focusAreaId = if ($isStoryWork) { $FocusArea } else { $Identifier }

    # Track retries
    $trackingId = if ($isStoryWork) { $storyId } else { $focusAreaId }
    $lastTrackingVar = if ($isStoryWork) { 'LastStoryId' } else { 'LastFocusAreaId' }
    $lastTracking = Get-Variable -Name $lastTrackingVar -Scope Script -ErrorAction SilentlyContinue
    if (-not $lastTracking -or $lastTracking.Value -ne $trackingId) {
        $script:CurrentRetryCount = 0
        Set-Variable -Name $lastTrackingVar -Value $trackingId -Scope Script
    }
    $script:CurrentRetryCount++

    $script:IterationCount++
    $iterationStart = Get-Date

    Write-IterationBanner -Iteration $script:IterationCount -FocusArea $focusAreaId -StoryId $storyId

    # Get Claude path
    $claudePath = Get-ClaudePath

    # Build arguments
    $claudeArgs = @("--print", "--dangerously-skip-permissions")
    if ($AllowedTools) {
        $claudeArgs += "--allowedTools=Bash,Read,Write,Edit,Glob,Grep,WebSearch"
    }

    $displayPrompt = if ($isStoryWork) { "Work on $storyId" } else { "Focus on $focusAreaId" }
    Write-Host "  Invoking Claude..." -ForegroundColor Cyan
    Write-Host "  Prompt: $displayPrompt" -ForegroundColor DarkGray

    # Capture git state BEFORE Claude runs
    $gitStateBefore = Get-GitState

    # Log state transition
    $transitionContext = @{ focusArea = $focusAreaId; promptType = $PromptType }
    if ($storyId) { $transitionContext.storyId = $storyId }
    Log-StateTransition -From "idle" -To "running" -Reason "Starting: $displayPrompt" -Context $transitionContext

    # Log timeline event: iteration start
    $timelineData = @{ iteration = $script:IterationCount; focusArea = $focusAreaId; promptType = $PromptType }
    if ($storyId) { $timelineData.storyId = $storyId }
    Append-SessionTimeline -Event "iteration_start" -Data $timelineData

    try {
        # Get timeout from config
        $timeout = 600
        if ($script:Config.iterationTimeout) { $timeout = $script:Config.iterationTimeout }
        if ($script:Config.autonomy -and $script:Config.autonomy.iterationTimeout) {
            $timeout = $script:Config.autonomy.iterationTimeout
        }

        # Output file paths
        $outFile = Join-Path $script:SessionLogDir "claude_out_$($script:IterationCount).log"
        $errFile = Join-Path $script:SessionLogDir "claude_err_$($script:IterationCount).log"
        $promptFile = Join-Path $script:SessionLogDir "prompt_$($script:IterationCount).txt"

        # Write prompt to file
        $Prompt | Out-File -FilePath $promptFile -Encoding UTF8 -NoNewline

        $flagsString = ($claudeArgs -join ' ')
        $executionStart = Get-Date

        # Create process with proper stdin redirection
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = $flagsString
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        # Async output capture
        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            $process.StandardInput.Write($Prompt)
            $process.StandardInput.Close()

            # Activity-based timeout monitoring
            $resourceSamples = @()
            $checkIntervalSec = 5
            $timeSinceProgress = 0
            $totalElapsed = 0
            $maxTotalMinutes = 60
            $lastMinuteShown = -1

            $lastPrdTime = (Get-Item $script:PrdFile -ErrorAction SilentlyContinue).LastWriteTime
            $lastProgressTime = (Get-Item $script:ProgressFile -ErrorAction SilentlyContinue).LastWriteTime
            $lastGitStatus = (git status --porcelain 2>$null | Measure-Object -Line).Lines
            $lastLogSize = if (Test-Path $outFile) { (Get-Item $outFile).Length } else { 0 }

            while (-not $process.HasExited -and $timeSinceProgress -lt $timeout -and $totalElapsed -lt ($maxTotalMinutes * 60)) {
                Start-Sleep -Seconds $checkIntervalSec
                $timeSinceProgress += $checkIntervalSec
                $totalElapsed += $checkIntervalSec

                if (-not $process.HasExited) {
                    $sample = Get-ProcessMetrics -ProcessId $process.Id
                    $resourceSamples += $sample

                    $mins = [math]::Floor($totalElapsed / 60)
                    $currentPrdTime = (Get-Item $script:PrdFile -ErrorAction SilentlyContinue).LastWriteTime
                    $currentProgressTime = (Get-Item $script:ProgressFile -ErrorAction SilentlyContinue).LastWriteTime
                    $currentGitStatus = (git status --porcelain 2>$null | Measure-Object -Line).Lines
                    $currentLogSize = if (Test-Path $outFile) { (Get-Item $outFile).Length } else { 0 }

                    $prdUpdated = $currentPrdTime -and $lastPrdTime -and ($currentPrdTime -gt $lastPrdTime)
                    $progressUpdated = $currentProgressTime -and $lastProgressTime -and ($currentProgressTime -gt $lastProgressTime)
                    $gitChanged = $currentGitStatus -ne $lastGitStatus
                    $logGrowing = $currentLogSize -gt $lastLogSize

                    if ($prdUpdated -or $progressUpdated -or $gitChanged -or $logGrowing) {
                        $reason = if ($prdUpdated) { "prd.json" } elseif ($progressUpdated) { "progress.txt" } elseif ($gitChanged) { "git changes" } else { "log output" }
                        Write-Host "  [$mins min] Activity detected ($reason)" -ForegroundColor DarkGreen
                        $timeSinceProgress = 0
                        $lastPrdTime = $currentPrdTime
                        $lastProgressTime = $currentProgressTime
                        $lastGitStatus = $currentGitStatus
                        $lastLogSize = $currentLogSize
                    }
                    elseif ($mins -gt $lastMinuteShown) {
                        Write-Host "  [$mins min] Running..." -ForegroundColor DarkGray
                        $lastMinuteShown = $mins
                    }
                }
            }

            $exited = $process.HasExited
            $executionEnd = Get-Date

            $exitCode = $null
            if ($exited) {
                $process.WaitForExit()
                $exitCode = $process.ExitCode
            }
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 100
            $outBuilder.ToString() | Set-Content $outFile -ErrorAction SilentlyContinue
            $errBuilder.ToString() | Set-Content $errFile -ErrorAction SilentlyContinue
        }

        $iterationDuration = (Get-Date) - $iterationStart
        $claudeOutput = $outBuilder.ToString() + $errBuilder.ToString()

        # Calculate metrics
        $tokensUsed = Get-EstimatedTokens -Output $claudeOutput
        $testResults = Get-TestResults -Output $claudeOutput

        # Capture git state AFTER
        $gitStateAfter = Get-GitState
        $fileOps = Get-FileOperations -BeforeHash $gitStateBefore.hash
        $commits = Get-GitCommits -SinceHash $gitStateBefore.hash

        # Phase timings
        $executionDurationMs = [int](($executionEnd - $executionStart).TotalMilliseconds)
        $phaseTimings = Measure-PhaseTimings -Output $claudeOutput -TotalDurationMs $executionDurationMs

        # Determine status
        $iterationStatus = "completed"
        $success = $false
        $timedOut = $false

        if (-not $exited) {
            Write-Host "  Timeout after $timeout seconds" -ForegroundColor Yellow
            $process.Kill()
            $iterationStatus = "timeout"
            $timedOut = $true
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $true

            Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Timeout after ${timeout}s" -Iteration $script:IterationCount
            Log-StateTransition -From "running" -To "failed" -Reason "Timeout after ${timeout}s" -Context $transitionContext

            Record-Metric -StoryId $Identifier -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $true -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0 -PhaseReadMs $phaseTimings.read_ms -PhaseAnalyzeMs $phaseTimings.analyze_ms -PhaseImplementMs $phaseTimings.implement_ms -PhaseTestMs $phaseTimings.test_ms -PhaseCommitMs $phaseTimings.commit_ms

            if ($StoryObj) { Log-StoryVerification -StoryId $storyId -Story $StoryObj -Iteration $script:IterationCount -Passed $false }
            $script:ConsecutiveFailures++
        }
        elseif ($exitCode -eq 0) {
            $successMsg = if ($isStoryWork) { "Story completed successfully" } else { "Iteration completed successfully" }
            Write-Host "  $successMsg" -ForegroundColor Green
            $iterationStatus = "completed"
            $success = $true

            Log-StateTransition -From "running" -To "completed" -Reason "Success" -Context $transitionContext

            $gitStats = Get-GitDiffStats

            # Story 1.4: Diff quality scoring
            $diffOutput = git diff HEAD~1 2>$null
            $diffQuality = Get-DiffQualityScore -DiffOutput $diffOutput
            if ($diffQuality.warnings.Count -gt 0) {
                foreach ($warn in $diffQuality.warnings) {
                    Write-Host "  Quality: $warn" -ForegroundColor Yellow
                }
            }

            # Story 1.5: Test regression detection
            $regressionResult = $null
            if ($testResults) {
                $regressionResult = Compare-TestBaseline -CurrentResults $testResults
            }

            # Story 4.4: Auto-rollback on regression
            if ($regressionResult -and $regressionResult.hasRegression -and $storyId) {
                $rollbackCheck = Test-CanRollback -StoryId $storyId
                if ($rollbackCheck.canRollback) {
                    Write-Host "  Auto-rollback: Reverting regression in $storyId" -ForegroundColor Yellow
                    $rollbackOk = Invoke-StoryRollback -StoryId $storyId -Reason "Test regression detected"
                    if ($rollbackOk) {
                        $success = $false
                        $iterationStatus = "rolled_back"
                        Write-Host "  Rollback complete. Story will retry." -ForegroundColor Yellow
                    }
                }
            }

            Record-Metric -StoryId $Identifier -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $true -Timeout $false -TokensUsed $tokensUsed -ErrorCategory "" -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded $gitStats.Added -LinesDeleted $gitStats.Deleted -PhaseReadMs $phaseTimings.read_ms -PhaseAnalyzeMs $phaseTimings.analyze_ms -PhaseImplementMs $phaseTimings.implement_ms -PhaseTestMs $phaseTimings.test_ms -PhaseCommitMs $phaseTimings.commit_ms

            if ($StoryObj) {
                Log-StoryVerification -StoryId $storyId -Story $StoryObj -Iteration $script:IterationCount -Passed $true -ClaudeOutput $claudeOutput -DiffOutput $diffOutput
                Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $storyId; passed = $true }

                # Story 2.1: Independent code review (enhanced from Story 1.1)
                if ($isStoryWork -and $diffOutput) {
                    $reviewResult = Invoke-CodeReview -StoryId $storyId -Story $StoryObj -DiffOutput $diffOutput -FileOps $fileOps -ClaudeOutput $claudeOutput
                    if ($reviewResult) {
                        Append-SessionTimeline -Event "code_review" -Data @{
                            storyId = $storyId
                            testRatio = $diffQuality.testRatio
                            churnRisk = $diffQuality.churnRisk
                            reviewScore = $reviewResult.score
                            reviewPassed = $reviewResult.passed
                            issueCount = $reviewResult.issues.Count
                        }

                        # Story 2.3: Log file conflict data
                        $conflictResult = Test-FileConflict -StoryId $storyId -Story $StoryObj
                        if ($conflictResult -and $conflictResult.hasConflict) {
                            Append-SessionTimeline -Event "file_conflict" -Data @{
                                storyId = $storyId
                                overlappingFiles = $conflictResult.overlappingFiles
                                overlappingStories = $conflictResult.overlappingStories
                            }
                        }
                    }
                }

                # Story 1.5: Update baseline after successful story
                if ($testResults) {
                    Update-TestBaseline -TestResults $testResults
                }

                # Story 1.8: Token budget check
                Get-SprintTokenBudget | Out-Null

                # Story 4.1: Save story progress (completed milestone)
                try {
                    Save-StoryProgress -StoryId $storyId -Milestone "completed" -Data @{
                        iteration = $script:IterationCount
                        retryCount = $script:CurrentRetryCount
                        tokensUsed = $tokensUsed
                    }
                } catch {}

                # Story 4.3: Update learning database
                try {
                    Update-LearningDb -Entry @{
                        type = "story_success"
                        storyId = $storyId
                        focusArea = $focusAreaId
                        retryCount = $script:CurrentRetryCount
                        tokensUsed = $tokensUsed
                        linesAdded = $gitStats.Added
                        linesDeleted = $gitStats.Deleted
                        testRatio = if ($diffQuality) { $diffQuality.testRatio } else { 0 }
                        reviewScore = if ($reviewResult) { $reviewResult.score } else { $null }
                    }
                } catch {}
            }
            $script:ConsecutiveFailures = 0
        }
        else {
            $exitCodeStr = if ($null -ne $exitCode) { $exitCode } else { "unknown" }
            $failMsg = if ($isStoryWork) { "Story failed with exit code $exitCodeStr" } else { "Iteration failed with exit code $exitCodeStr" }
            Write-Host "  $failMsg" -ForegroundColor Red
            $iterationStatus = "failed"
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $false

            Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Exit code: $exitCodeStr" -Iteration $script:IterationCount
            Log-StateTransition -From "running" -To "failed" -Reason "Exit code: $exitCodeStr" -Context $transitionContext

            Record-Metric -StoryId $Identifier -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $false -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0 -PhaseReadMs $phaseTimings.read_ms -PhaseAnalyzeMs $phaseTimings.analyze_ms -PhaseImplementMs $phaseTimings.implement_ms -PhaseTestMs $phaseTimings.test_ms -PhaseCommitMs $phaseTimings.commit_ms

            if ($StoryObj) { Log-StoryVerification -StoryId $storyId -Story $StoryObj -Iteration $script:IterationCount -Passed $false }

            # Story 4.3: Update learning database on failure
            if ($storyId) {
                try {
                    Update-LearningDb -Entry @{
                        type = "story_failure"
                        storyId = $storyId
                        focusArea = $focusAreaId
                        errorCategory = $errorCategory
                        retryCount = $script:CurrentRetryCount
                        exitCode = $exitCodeStr
                    }
                } catch {}
            }

            $script:ConsecutiveFailures++
        }

        # === COMPREHENSIVE LOGGING ===
        Log-ClaudeInvocation -Iteration $script:IterationCount -ClaudePath $claudePath -Arguments $claudeArgs -PromptFile $promptFile -PromptType $PromptType -ProcessId $process.Id -StartTime $executionStart -EndTime $executionEnd -ExitCode $(if ($null -ne $exitCode) { $exitCode } else { -1 }) -TimedOut $timedOut

        Log-IterationManifest -Iteration $script:IterationCount -StoryId $storyId -FocusArea $focusAreaId -Status $iterationStatus -StartTime $iterationStart -EndTime (Get-Date) -PromptFile $promptFile -GitBefore $gitStateBefore -GitAfter $gitStateAfter -FileOps $fileOps -Commits $commits -TestResults $testResults -TokensEstimated $tokensUsed -RetryCount $script:CurrentRetryCount

        Log-FileOperations -Iteration $script:IterationCount -FileOps $fileOps
        Log-GitOperations -Iteration $script:IterationCount -Branch $gitStateAfter.branch -Commits $commits -BeforeState $gitStateBefore -AfterState $gitStateAfter

        $completeData = @{ iteration = $script:IterationCount; status = $iterationStatus; success = $success; durationSec = [int]$iterationDuration.TotalSeconds }
        if ($storyId) { $completeData.storyId = $storyId }
        Append-SessionTimeline -Event "iteration_complete" -Data $completeData

        # Phase 3 logging
        if ($claudeOutput) { Log-TestDetails -Iteration $script:IterationCount -Output $claudeOutput }
        if ($resourceSamples.Count -gt 0) { Log-ResourceUsage -Iteration $script:IterationCount -ProcessId $process.Id -Samples $resourceSamples }

        $effectiveness = Get-PromptEffectiveness -Success $success -RetryCount $script:CurrentRetryCount
        $promptContent = Get-Content $promptFile -Raw -ErrorAction SilentlyContinue
        $promptHashShort = ""
        if ($promptContent) {
            $md5 = [System.Security.Cryptography.MD5]::Create()
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($promptContent)
            $hashBytes = $md5.ComputeHash($bytes)
            $promptHashShort = ([BitConverter]::ToString($hashBytes) -replace '-', '').Substring(0, 16)
        }
        Log-PromptEffectiveness -Iteration $script:IterationCount -PromptType $PromptType -Effectiveness $effectiveness -PromptHash $promptHashShort

        return $success
    }
    catch {
        Write-Host "  Error invoking Claude: $_" -ForegroundColor Red
        $errorCategory = Get-ErrorCategory -Output $_.ToString() -TimedOut $false

        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails $_.ToString() -Iteration $script:IterationCount
        Log-StateTransition -From "running" -To "error" -Reason $_.ToString() -Context $transitionContext

        Record-Metric -StoryId $Identifier -Mode $script:CurrentMode -DurationMin 0 -Success $false -Timeout $false -TokensUsed 0 -ErrorCategory $errorCategory -TestResults "" -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0 -PhaseReadMs 0 -PhaseAnalyzeMs 0 -PhaseImplementMs 0 -PhaseTestMs 0 -PhaseCommitMs 0

        $errorData = @{ iteration = $script:IterationCount; error = $_.ToString() }
        if ($storyId) { $errorData.storyId = $storyId }
        Append-SessionTimeline -Event "iteration_error" -Data $errorData

        $script:ConsecutiveFailures++
        return $false
    }
}

function Get-ClaudePath {
    <#
    .SYNOPSIS
        Resolve the path to Claude CLI
    .RETURNS
        Full path to claude executable
    #>

    # Check config first
    $claudePath = $script:Config.claudePath
    if (-not $claudePath) {
        $claudePath = "claude"
    }

    # If it's just "claude", try to find it
    if ($claudePath -eq "claude") {
        # Try common locations
        $possiblePaths = @(
            "$env:LOCALAPPDATA\Programs\claude-code\claude.exe",
            "$env:APPDATA\npm\claude.cmd",
            "$env:USERPROFILE\.npm-global\claude.cmd",
            "C:\Program Files\Claude Code\claude.exe"
        )

        foreach ($path in $possiblePaths) {
            if (Test-Path $path) {
                return $path
            }
        }

        # Fall back to hoping it's in PATH
        return "claude"
    }

    return $claudePath
}

function Invoke-ClaudeForFocusArea {
    <#
    .SYNOPSIS
        Spawn Claude Code to work on a focus area
    .PARAMETER FocusAreaId
        The focus area ID to work on
    .PARAMETER Context
        Additional context from interview (optional)
    .PARAMETER GeneratePRD
        If specified, generate a new PRD for this focus area instead of working on stories
    .RETURNS
        $true if iteration succeeded, $false otherwise
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FocusAreaId,
        [string]$Context = "",
        [switch]$GeneratePRD
    )

    # === SPRINT-START EXPLORATION ===
    # Run mandatory exploration before PRD generation if enabled
    if ($GeneratePRD) {
        $explorationConfig = $script:Config.exploration
        $sprintStartEnabled = $explorationConfig -and $explorationConfig.enabled -and `
                              $explorationConfig.sprintStart -and $explorationConfig.sprintStart.enabled

        if ($sprintStartEnabled) {
            Write-Host ""
            Write-Host ">>> Sprint-Start Exploration: $FocusAreaId" -ForegroundColor Cyan
            Write-Host ""

            # Run full exploration
            $explorationResult = Invoke-FocusAreaExploration -FocusArea $FocusAreaId -Reason "sprint_start" -FullExplore

            # Reset stories counter since we're starting fresh
            $script:StoriesSinceExploration = 0

            Write-Host ""
        }
    }

    # Build the prompt
    if ($GeneratePRD) {
        # Build exploration context section for PRD prompt
        $explorationSection = ""
        if ($script:SprintExplorationContext) {
            $explorationSection = @"

## Exploration Context (Fresh Scan)
$script:SprintExplorationContext

Use this exploration context to inform story generation. Prioritize:
- Issues discovered during exploration
- Test failures that need fixing
- Technical debt identified
- Missing functionality noted

"@
        }

        $prompt = @"
You are generating a new sprint PRD for focus area: $FocusAreaId

INSTRUCTIONS:
1. Read scripts/ralph/ralph-config.json to understand the focus area
2. Read scripts/ralph/prompt.md for context about the project
3. Read CLAUDE.md for project conventions
4. Read scripts/ralph/queue.json for interview details (story outline, architecture decisions, key files)
5. Analyze the codebase to find improvement opportunities for '$FocusAreaId'
6. Update scripts/ralph/prd.json with:
   - focusArea: "$FocusAreaId"
   - sprintNumber: increment from current
   - branchName: "ralph/sprint-N" (matching sprintNumber)
   - 8-12 specific user stories with:
     - Clear acceptance criteria (4-6 items each)
     - passes: false for all stories
     - Action verbs in titles (Add, Create, Update, Fix, etc.)

$(if ($Context) { "Context from user: $Context" } else { "" })
$explorationSection
Start by reading the config and prompt files, then generate the PRD.
"@
        $promptType = "prd_generation"
    }
    else {
        $prompt = "Focus on: $FocusAreaId`n`n"
        if ($Context) { $prompt += "Context: $Context`n`n" }
        $prompt += "Read scripts/ralph/prompt.md for instructions. Work on ONE user story from prd.json that aligns with the focus area. If no stories exist for this focus area, generate appropriate stories first."
        $promptType = "focus_area_work"
    }

    # Update progress file
    $progressEntry = "`n## Focus Area: $FocusAreaId - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n- Status: In Progress...`n"
    Add-Content -Path $script:ProgressFile -Value $progressEntry

    # Invoke the common process handler
    $useTools = $GeneratePRD -or $SkipPlanApproval
    return Invoke-ClaudeProcess -Prompt $prompt -PromptType $promptType -Identifier $FocusAreaId -AllowedTools:$useTools
}

function Invoke-BatchPreFlight {
    <#
    .SYNOPSIS
        Batch pre-flight check at sprint start: show done stories and scan incomplete ones for existing commits
    .DESCRIPTION
        Runs once at the beginning of a Ralph Loop session. Shows already-done stories,
        then iterates through incomplete stories and checks if a matching git commit exists.
        Phase 0: Display stories already marked as passes=true (done).
        Phase 1: Find candidates by story ID match in git log (cheap).
        Phase 2: LLM-verify that commit semantically matches the story (prevents cross-sprint false positives).
        Phase 3: Auto-complete only LLM-confirmed matches.
    .RETURNS
        Number of stories auto-completed
    #>

    if (-not (Test-Path $script:PrdFile)) {
        return 0
    }

    try {
        $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
    }
    catch {
        return 0
    }

    $allStories = @($prd.userStories)
    $doneStories = @($allStories | Where-Object { $_.passes })
    $incompleteStories = @($allStories | Where-Object { -not $_.passes })

    Write-Host ""

    # Show done stories
    if ($doneStories.Count -gt 0) {
        Write-Host "  Pre-flight: $($doneStories.Count)/$($allStories.Count) stories already done" -ForegroundColor Green
        foreach ($done in $doneStories) {
            Write-Host "    $($done.id): $($done.title)" -ForegroundColor DarkGreen
        }
    }

    if ($incompleteStories.Count -eq 0) {
        Write-Host "  Pre-flight: all stories complete" -ForegroundColor Green
        Write-Host ""
        return 0
    }

    Write-Host "  Pre-flight scan: checking $($incompleteStories.Count) incomplete stories against git..." -ForegroundColor Cyan

    # Phase 1: Find candidates (stories with matching commit IDs)
    $candidates = @()
    foreach ($story in $incompleteStories) {
        if (-not $story.id -or -not $story.title) { continue }

        try {
            $gitLog = git log --oneline --all --grep="\[$($story.id)\]" 2>$null
            if (-not $gitLog) {
                $gitLog = git log --oneline --all --grep="($($story.id))" 2>$null
            }
            if ($gitLog) {
                $commitLine = ($gitLog -split "`n" | Where-Object { $_ } | Select-Object -First 1)
                $candidates += @{
                    storyId    = $story.id
                    storyTitle = $story.title
                    commitMsg  = $commitLine
                    storyObj   = $story
                }
                Write-Host "    $($story.id): found commit candidate" -ForegroundColor DarkGray
            }
        }
        catch {}
    }

    if ($candidates.Count -eq 0) {
        Write-Host "  Pre-flight: no prior commits found, all stories need implementation" -ForegroundColor DarkGray
        Write-Host ""
        return 0
    }

    Write-Host "  Pre-flight: $($candidates.Count) candidates found, verifying with LLM..." -ForegroundColor Cyan

    # Phase 2: LLM verification (single call for all candidates)
    $matchedIds = Confirm-CommitMatchesStory -Candidates $candidates

    # Phase 3: Auto-complete verified matches
    $autoCompleted = 0
    foreach ($id in $matchedIds) {
        $candidate = $candidates | Where-Object { $_.storyId -eq $id } | Select-Object -First 1
        if ($candidate) {
            Write-Host "    Pre-flight: LLM confirmed $id matches commit" -ForegroundColor Green
            Write-Host "      $($candidate.commitMsg)" -ForegroundColor DarkCyan
            Complete-StoryAutomatically -StoryId $id -Story $candidate.storyObj -Reason "git-commit-detected"
            $autoCompleted++
        }
    }

    # Log rejections
    $rejectedCandidates = $candidates | Where-Object { $_.storyId -notin $matchedIds }
    foreach ($r in $rejectedCandidates) {
        Write-Host "    Pre-flight: LLM rejected $($r.storyId) - different sprint's work" -ForegroundColor DarkYellow
        Write-Host "      Story: $($r.storyTitle)" -ForegroundColor DarkYellow
        Write-Host "      Commit: $($r.commitMsg)" -ForegroundColor DarkYellow
    }

    if ($autoCompleted -gt 0) {
        Write-Host "  Pre-flight: auto-completed $autoCompleted/$($incompleteStories.Count) stories from prior commits" -ForegroundColor Green
    } else {
        Write-Host "  Pre-flight: LLM found no genuine matches among $($candidates.Count) candidates" -ForegroundColor DarkGray
    }
    Write-Host ""

    return $autoCompleted
}

function Confirm-CommitMatchesStory {
    <#
    .SYNOPSIS
        LLM-verify that git commits semantically match their user stories
    .DESCRIPTION
        Sends a single LLM call with all candidate story/commit pairs.
        The LLM determines if each commit actually implements the described story
        (not just a coincidental ID match from a different sprint).
    .PARAMETER Candidates
        Array of hashtables with: storyId, storyTitle, commitMsg
    .RETURNS
        Array of story IDs that the LLM confirms as genuine matches
    #>
    param(
        [array]$Candidates = @()
    )

    if (-not $Candidates -or $Candidates.Count -eq 0) { return @() }

    $claudePath = Get-ClaudePath
    if (-not $claudePath) {
        Write-Host "    Pre-flight: Claude not available, skipping LLM verification" -ForegroundColor DarkYellow
        return @()
    }

    # Build verification prompt
    $pairsList = ""
    foreach ($c in $Candidates) {
        $pairsList += "- $($c.storyId): Story=`"$($c.storyTitle)`" | Commit=`"$($c.commitMsg)`"`n"
    }

    $prompt = @"
You are verifying if git commits implement specific user stories.
IMPORTANT: Different sprints reuse story IDs (US-001, US-002, etc.), so the commit might be from a DIFFERENT sprint with COMPLETELY DIFFERENT work despite having the same ID.

A commit matches a story ONLY if the commit message describes the SAME WORK as the story title.
Example MATCH: Story="Add retry logic to downloader" | Commit="feat: [US-003] Add retry logic to downloader"
Example NO MATCH: Story="Add retry logic to downloader" | Commit="feat: [US-003] Wire impersonation into core.py"

For each pair, reply MATCH or NO_MATCH followed by the story ID. Nothing else.

$pairsList
"@

    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model haiku"
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        # Async output capture (prevents pipe buffer deadlock)
        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            $process.StandardInput.Write($prompt)
            $process.StandardInput.Close()

            # 60 second timeout for simple classification
            $completed = $process.WaitForExit(60000)

            if (-not $completed) {
                $process.Kill()
                Write-Host "    Pre-flight: LLM verification timed out" -ForegroundColor DarkYellow
                return @()
            }

            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()

            # Parse response for MATCH lines (exclude NO_MATCH)
            $matchedIds = @()
            $candidateIds = @($Candidates | ForEach-Object { $_.storyId })
            foreach ($line in ($output -split "`n")) {
                $trimmed = $line.Trim()
                # Skip NO_MATCH lines, then check for MATCH
                if ($trimmed -match '^NO_MATCH') { continue }
                if ($trimmed -match 'MATCH\s+(US-\d+)') {
                    $id = $Matches[1]
                    # Only accept IDs that are actual candidates (safety)
                    if ($id -in $candidateIds) {
                        $matchedIds += $id
                    }
                }
            }

            return $matchedIds
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Host "    Pre-flight: LLM verification failed: $_" -ForegroundColor DarkYellow
        return @()
    }
}

function Test-StoryAlreadyCommitted {
    <#
    .SYNOPSIS
        Per-story pre-flight check (used as guard in Invoke-ClaudeForStory)
    .DESCRIPTION
        Checks if a story was already committed by finding the commit via ID match,
        then LLM-verifying the commit semantically matches the story title.
        Falls back conservatively (returns false) if LLM is unavailable.
    .PARAMETER StoryId
        Story identifier (e.g., US-005)
    .PARAMETER Story
        Story object from PRD (required for verification)
    .RETURNS
        $true if story is confirmed already committed, $false otherwise
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$StoryId,
        [object]$Story = $null
    )

    if (-not $Story -or -not $Story.title) {
        return $false
    }

    try {
        $gitLog = git log --oneline --all --grep="\[$StoryId\]" 2>$null
        if (-not $gitLog) {
            $gitLog = git log --oneline --all --grep="($StoryId)" 2>$null
        }
        if (-not $gitLog) {
            return $false
        }

        $commitLine = ($gitLog -split "`n" | Where-Object { $_ } | Select-Object -First 1)

        # LLM-verify the match
        $matchedIds = Confirm-CommitMatchesStory -Candidates @(
            @{
                storyId    = $StoryId
                storyTitle = $Story.title
                commitMsg  = $commitLine
            }
        )

        if ($StoryId -in $matchedIds) {
            Write-Host "    Pre-flight: LLM confirmed $StoryId already committed" -ForegroundColor Green
            Write-Host "      $commitLine" -ForegroundColor DarkCyan
            return $true
        }
        else {
            Write-Host "    Pre-flight: commit for $StoryId is from a different sprint - skipping" -ForegroundColor DarkYellow
        }
    }
    catch {
        Write-Host "    Pre-flight: git check failed, proceeding normally" -ForegroundColor DarkYellow
    }

    return $false
}

function Complete-StoryAutomatically {
    <#
    .SYNOPSIS
        Auto-complete a story that was already committed but not marked in prd.json
    .DESCRIPTION
        Updates prd.json (passes: true), appends to progress.txt, and records
        a metrics entry. Called when pre-flight check detects the work is done.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Story object from PRD
    .PARAMETER Reason
        Why auto-completed (e.g., "git-commit-detected")
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$StoryId,
        [object]$Story = $null,
        [string]$Reason = "pre-flight-detected"
    )

    try {
        # Update prd.json
        if (Test-Path $script:PrdFile) {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $storyToUpdate = $prd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
            if ($storyToUpdate) {
                $storyToUpdate.passes = $true
                $commitInfo = git log --oneline -1 --all --grep="\[$StoryId\]" 2>$null
                if (-not $commitInfo) {
                    $commitInfo = git log --oneline -1 --all --grep="($StoryId)" 2>$null
                }
                $noteText = "Auto-completed by pre-flight check ($Reason). Commit: $commitInfo"
                if ($storyToUpdate.PSObject.Properties['notes']) {
                    $storyToUpdate.notes = $noteText
                }
                else {
                    $storyToUpdate | Add-Member -NotePropertyName 'notes' -NotePropertyValue $noteText -Force
                }
                $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
                Write-Host "    Updated prd.json: $StoryId -> passes: true" -ForegroundColor Green
            }
        }

        # Append to progress.txt
        $progressFile = Join-Path $script:RalphDir "progress.txt"
        $title = if ($Story) { $Story.title } else { "Unknown" }
        $progressEntry = @"

## Pre-flight Auto-Complete - [$StoryId] $title
- Status: COMPLETE (auto-detected from git commit, $Reason)
- Story was implemented in a previous session but prd.json was not updated (likely timeout)
- Pre-flight check in ralph.ps1 detected the existing commit and auto-marked as complete
"@
        $progressEntry | Out-File -FilePath $progressFile -Append -Encoding UTF8

        # Record metrics
        $metricsFile = Join-Path $script:RalphDir "metrics.csv"
        if (Test-Path $metricsFile) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $sessionId = if ($script:SessionId) { $script:SessionId } else { "preflight" }
            $sprintName = "sprint-$($script:SprintNumber)"
            $focusArea = ""
            if (Test-Path $script:PrdFile) {
                try {
                    $prdData = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
                    $focusArea = $prdData.focusArea
                } catch {}
            }
            $metricsLine = "$timestamp,$sessionId,$sprintName,$StoryId,PreFlight,0,true,false,$focusArea,0,,$(Get-Date -Format 'HH'),,0,0,0,0,0,0,0,0,false,,0"
            $metricsLine | Out-File -FilePath $metricsFile -Append -Encoding UTF8
        }
    }
    catch {
        Write-Host "    Warning: Auto-complete update failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Invoke-ClaudeForStory {
    <#
    .SYNOPSIS
        Spawn Claude to work on a specific story
    .PARAMETER StoryId
        The story ID to work on (e.g., US-001)
    .RETURNS
        $true if iteration succeeded, $false otherwise
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$StoryId
    )

    # Get focus area and story object from PRD
    $focusArea = ""
    $storyObj = $null
    if (Test-Path $script:PrdFile) {
        try {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $focusArea = $prd.focusArea
            $storyObj = $prd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
        }
        catch {}
    }

    # Pre-flight: skip stories already committed in git
    if ($storyObj -and -not $storyObj.passes) {
        $alreadyDone = Test-StoryAlreadyCommitted -StoryId $StoryId -Story $storyObj
        if ($alreadyDone) {
            Write-Host "    Pre-flight: $StoryId already committed in git - auto-completing" -ForegroundColor Green
            Complete-StoryAutomatically -StoryId $StoryId -Story $storyObj -Reason "git-commit-detected"
            return $true
        }
    }

    # Story 3.1: Adaptive prompt builder (consolidates Stories 1.2, 1.3, 1.6, 2.5, 3.3)
    $prompt = Build-StoryPrompt -StoryId $StoryId -Story $storyObj -FocusArea $focusArea -RetryCount $script:CurrentRetryCount

    # Invoke the common process handler
    return Invoke-ClaudeProcess -Prompt $prompt -PromptType "story_work" -Identifier $StoryId -FocusArea $focusArea -StoryObj $storyObj
}

# ============================================================================
# METRICS
# ============================================================================

function Get-GitDiffStats {
    <#
    .SYNOPSIS
    Gets lines added/deleted since last commit using git diff
    #>

    try {
        # Get diff stats for staged and unstaged changes
        $diffOutput = git diff --numstat HEAD~1 2>$null

        if (-not $diffOutput) {
            return @{ Added = 0; Deleted = 0 }
        }

        $totalAdded = 0
        $totalDeleted = 0

        foreach ($line in $diffOutput -split "`n") {
            if ($line -match '^(\d+)\s+(\d+)\s+') {
                $totalAdded += [int]$Matches[1]
                $totalDeleted += [int]$Matches[2]
            }
        }

        return @{ Added = $totalAdded; Deleted = $totalDeleted }
    }
    catch {
        return @{ Added = 0; Deleted = 0 }
    }
}

function Get-TestResults {
    <#
    .SYNOPSIS
        Parse pytest output to extract pass/fail counts
    .PARAMETER Output
        The output text to analyze
    .RETURNS
        String like "41/41 pass" or "38/41 pass, 3 fail", or empty string if no test results found
    #>
    param(
        [string]$Output
    )

    if (-not $Output) {
        return ""
    }

    $passed = 0
    $failed = 0

    # Pattern: "X passed" or "X passed,"
    if ($Output -match '(\d+)\s+passed') {
        $passed = [int]$Matches[1]
    }

    # Pattern: "X failed"
    if ($Output -match '(\d+)\s+failed') {
        $failed = [int]$Matches[1]
    }

    # Pattern: "X error" (collection errors)
    if ($Output -match '(\d+)\s+error') {
        $failed += [int]$Matches[1]
    }

    $total = $passed + $failed

    if ($total -eq 0) {
        return ""  # No test results found
    }

    if ($failed -eq 0) {
        return "$passed/$total pass"
    }
    else {
        return "$passed/$total pass, $failed fail"
    }
}

function Get-ErrorCategory {
    <#
    .SYNOPSIS
        Categorize error type from Claude output
    .PARAMETER Output
        The output text to analyze
    .PARAMETER TimedOut
        Whether the iteration timed out
    .RETURNS
        Error category string
    #>
    param(
        [string]$Output,
        [bool]$TimedOut
    )

    if ($TimedOut) { return "Timeout" }
    if ($Output -match "SyntaxError|parse error|unexpected token") { return "SyntaxError" }
    if ($Output -match "FAILED|AssertionError|test.*failed") { return "TestFailure" }
    if ($Output -match "cannot be loaded|compilation|ImportError") { return "CompileError" }
    if ($Output -match "ValidationError|schema") { return "ValidationError" }
    if ($Output -match "API|rate.?limit|quota|429") { return "APIError" }
    return "Unknown"
}

function Get-EstimatedTokens {
    <#
    .SYNOPSIS
        Estimate token count from output length (Story 1.8: improved estimation)
    .DESCRIPTION
        First tries to parse actual token usage from Claude's stderr output.
        Falls back to character-based estimation (~4 chars per token).
    .PARAMETER Output
        The output text to estimate tokens from
    .RETURNS
        Estimated token count (capped at 200000)
    #>
    param(
        [string]$Output
    )

    if (-not $Output -or $Output.Length -eq 0) {
        return 0
    }

    # Try to parse actual token usage from Claude output
    # Claude CLI may output usage info like "Input tokens: 1234" or "Total tokens: 5678"
    if ($Output -match 'total[_\s]?tokens[:\s]+(\d+)') {
        return [int]$Matches[1]
    }
    if ($Output -match 'input[_\s]?tokens[:\s]+(\d+)') {
        $inputTokens = [int]$Matches[1]
        # Also check for output tokens
        $outputTokens = 0
        if ($Output -match 'output[_\s]?tokens[:\s]+(\d+)') {
            $outputTokens = [int]$Matches[1]
        }
        return $inputTokens + $outputTokens
    }

    # Fallback: estimate from output length (~4 chars per token)
    $estimatedTokens = [math]::Round($Output.Length / 4)
    # Cap at reasonable max for a single interaction
    return [math]::Min($estimatedTokens, 200000)
}

function Record-Metric {
    param(
        [string]$Session,
        [string]$Sprint,
        [string]$StoryId,
        [string]$Mode,
        [double]$DurationMin,
        [bool]$Success = $true,
        [bool]$Timeout = $false,
        [string]$FocusArea,
        [int]$TokensUsed = 0,
        [string]$ErrorCategory = "",
        [int]$HourOfDay = -1,
        [string]$TestResults = "",
        [int]$RetryCount = 0,
        [int]$LinesAdded = 0,
        [int]$LinesDeleted = 0,
        # Phase 2 - Task 2.1: Phase timing breakdown
        [int]$PhaseReadMs = 0,
        [int]$PhaseAnalyzeMs = 0,
        [int]$PhaseImplementMs = 0,
        [int]$PhaseTestMs = 0,
        [int]$PhaseCommitMs = 0,
        # Exploration tracking
        [bool]$ExplorationTriggered = $false,
        [string]$ExplorationReason = "",
        [int]$ExplorationTokens = 0
    )

    # V3 schema: 24 columns including exploration tracking
    $v3Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms,exploration_triggered,exploration_reason,exploration_tokens"

    # Ensure metrics file exists with v3 header
    if (-not (Test-Path $script:MetricsFile)) {
        $v3Header | Set-Content $script:MetricsFile -Encoding UTF8
    }
    else {
        # Check if we need to migrate schema
        $header = Get-Content $script:MetricsFile -First 1
        $headerCols = ($header -split ',').Count

        if ($headerCols -lt 24) {
            # Migration: rewrite with v3 header and pad old rows
            $lines = Get-Content $script:MetricsFile
            $lines[0] = $v3Header
            # Add empty values to existing rows (pad to 24 columns)
            for ($i = 1; $i -lt $lines.Count; $i++) {
                $rowCols = ($lines[$i] -split ',').Count
                $padding = 24 - $rowCols
                if ($padding -gt 0) {
                    # Pad with appropriate defaults: false,, 0 for new exploration columns
                    if ($rowCols -eq 21) {
                        # Coming from v2, add 3 new exploration columns
                        $lines[$i] = $lines[$i] + ",false,,0"
                    } else {
                        # Coming from older version, pad with zeros
                        $lines[$i] = $lines[$i] + (',' + '0' * $padding -replace '0', ',0').Substring(1)
                    }
                }
            }
            $lines | Set-Content $script:MetricsFile -Encoding UTF8
            Write-Host "  Migrated metrics.csv to v3 schema (24 columns with exploration)" -ForegroundColor DarkGray
        }
    }

    # Use defaults from script variables if not provided
    if (-not $Session) { $Session = $script:SessionId }
    if (-not $Mode) { $Mode = $script:CurrentMode }
    if (-not $FocusArea -or -not $Sprint) {
        if (Test-Path $script:PrdFile) {
            try {
                $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
                if (-not $FocusArea) { $FocusArea = $prd.focusArea }
                if (-not $Sprint) { $Sprint = "sprint-$($prd.sprintNumber)" }
            }
            catch {}
        }
    }

    # Calculate hour_of_day if not provided
    if ($HourOfDay -eq -1) {
        $HourOfDay = (Get-Date).Hour
    }

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $row = "$timestamp,$Session,$Sprint,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),$($Timeout.ToString().ToLower()),$FocusArea,$TokensUsed,$ErrorCategory,$HourOfDay,$TestResults,$RetryCount,$LinesAdded,$LinesDeleted,$PhaseReadMs,$PhaseAnalyzeMs,$PhaseImplementMs,$PhaseTestMs,$PhaseCommitMs,$($ExplorationTriggered.ToString().ToLower()),$ExplorationReason,$ExplorationTokens"
    Add-Content -Path $script:MetricsFile -Value $row
}

# ============================================================================
# FAST-FAIL DETECTION
# ============================================================================

function Test-ShouldAbort {
    <#
    .SYNOPSIS
        Check if we should abort due to consecutive failures
    .RETURNS
        $true if should abort, $false otherwise
    #>

    $maxFailures = 3
    if ($script:Config.autonomy -and $script:Config.autonomy.fastFail) {
        $maxFailures = $script:Config.autonomy.fastFail.consecutiveFailures
    }

    if ($script:ConsecutiveFailures -ge $maxFailures) {
        Write-Host ""
        Write-Host "  FAST-FAIL: $script:ConsecutiveFailures consecutive failures" -ForegroundColor Red
        Write-Host "  Aborting to prevent wasted iterations" -ForegroundColor Red
        Write-Host ""
        return $true
    }

    return $false
}

function Test-MaxIterations {
    <#
    .SYNOPSIS
        Check if we've hit max iterations
    .RETURNS
        $true if at max, $false otherwise
    #>

    $maxIterations = 10
    if ($script:Config.autonomy -and $script:Config.autonomy.maxIterations) {
        $maxIterations = $script:Config.autonomy.maxIterations
    }
    if ($script:Config.maxIterations) {
        $maxIterations = $script:Config.maxIterations
    }

    if ($script:IterationCount -ge $maxIterations) {
        Write-Host ""
        Write-Host "  MAX ITERATIONS: Reached $maxIterations iterations" -ForegroundColor Yellow
        Write-Host ""
        return $true
    }

    return $false
}

# ============================================================================
# GRACEFUL STOP
# ============================================================================

function Test-GracefulStopRequested {
    <#
    .SYNOPSIS
        Check if a graceful stop has been requested
    .DESCRIPTION
        Looks for graceful_stop.signal file in the Ralph directory.
        File-based approach avoids race conditions and works from any terminal.
    .RETURNS
        $true if stop requested, $false otherwise
    #>

    $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
    if (Test-Path $signalFile) {
        # Read metadata if present for logging
        try {
            $meta = Get-Content $signalFile -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
            $reason = if ($meta -and $meta.reason) { $meta.reason } else { "User requested" }
            $requestedAt = if ($meta -and $meta.requestedAt) { $meta.requestedAt } else { "Unknown" }
            Write-Host ""
            Write-Host "  =====================================================" -ForegroundColor Cyan
            Write-Host "     GRACEFUL STOP REQUESTED" -ForegroundColor Cyan
            Write-Host "  =====================================================" -ForegroundColor Cyan
            Write-Host "  Reason: $reason" -ForegroundColor DarkGray
            Write-Host "  Requested at: $requestedAt" -ForegroundColor DarkGray
            Write-Host ""
        }
        catch {
            Write-Host ""
            Write-Host "  GRACEFUL STOP REQUESTED" -ForegroundColor Cyan
            Write-Host ""
        }
        return $true
    }
    return $false
}

function Clear-GracefulStopSignal {
    <#
    .SYNOPSIS
        Clear any existing graceful stop signal
    .DESCRIPTION
        Removes the graceful_stop.signal file if it exists.
        Called after honoring a stop request or at startup to clear stale signals.
    #>

    $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
    if (Test-Path $signalFile) {
        Remove-Item $signalFile -Force -ErrorAction SilentlyContinue
    }
}

function Request-GracefulStop {
    <#
    .SYNOPSIS
        Request a graceful stop (for programmatic use)
    .PARAMETER Reason
        Human-readable reason for the stop request
    #>
    param(
        [string]$Reason = "User requested graceful stop"
    )

    $signalFile = Join-Path $script:RalphDir "graceful_stop.signal"
    @{
        requestedAt = (Get-Date).ToString("o")
        reason = $Reason
    } | ConvertTo-Json | Set-Content $signalFile -Encoding UTF8
}

# ============================================================================
# SPRINT STATUS
# ============================================================================

function Get-SprintStatus {
    <#
    .SYNOPSIS
        Get current sprint completion status
    .RETURNS
        Hashtable with passed, failed, total, nextStory
    #>

    if (-not (Test-Path $script:PrdFile)) {
        return @{ passed = 0; failed = 0; total = 0; nextStory = $null; complete = $true }
    }

    try {
        $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json

        $passed = 0
        $failed = 0

        foreach ($story in $prd.userStories) {
            if ($story.passes) {
                $passed++
            }
            else {
                $failed++
            }
        }

        # Story 2.2: Smart story ordering
        $nextStory = $null
        if ($failed -gt 0) {
            $metricsFile = Join-Path $script:RalphDir "metrics.csv"
            $metricsData = $null
            if (Test-Path $metricsFile) {
                try { $metricsData = Import-Csv $metricsFile } catch {}
            }
            $nextStory = Get-OptimalNextStory -Stories $prd.userStories -Metrics $metricsData
        }

        return @{
            passed = $passed
            failed = $failed
            total = $passed + $failed
            nextStory = $nextStory
            complete = ($failed -eq 0)
            focusArea = $prd.focusArea
        }
    }
    catch {
        return @{ passed = 0; failed = 0; total = 0; nextStory = $null; complete = $true }
    }
}

# ============================================================================
# MAIN LOOP
# ============================================================================

function Start-InterviewQueueLoop {
    <#
    .SYNOPSIS
        Process focus areas from interview queue
    #>

    $script:CurrentMode = "Interview"
    $context = Get-InterviewContext
    $focusAreas = Get-InterviewFocusAreas

    if ($focusAreas.Count -eq 0) {
        Write-Host "  No pending focus areas in queue" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Starting interview to add focus areas..." -ForegroundColor Cyan
        Write-Host ""

        # Launch interview inline (with -NoLaunch to prevent spawning new windows)
        $interviewPath = Join-Path $script:RalphDir "interview.ps1"
        if (Test-Path $interviewPath) {
            & $interviewPath -NoLaunch

            # Re-check for focus areas after interview
            $focusAreas = Get-InterviewFocusAreas
            if ($focusAreas.Count -eq 0) {
                Write-Host "  No focus areas added. Exiting." -ForegroundColor Yellow
                return
            }
            # Update context after interview
            $context = Get-InterviewContext
        } else {
            Write-Host "  Interview script not found at: $interviewPath" -ForegroundColor Red
            return
        }
    }

    Write-Host "  Processing $($focusAreas.Count) focus areas from interview queue" -ForegroundColor Cyan
    if ($context) {
        Write-Host "  Context: $context" -ForegroundColor DarkGray
    }
    Write-Host ""

    $completedAreas = @()

    foreach ($area in $focusAreas) {
        $areaId = if ($area.id) { $area.id } else { $area }

        Write-Host "  Starting focus area: $areaId" -ForegroundColor Cyan

        # Check if PRD already has stories for this focus area (resume scenario)
        $currentStatus = Get-SprintStatus
        if ($currentStatus.focusArea -eq $areaId -and -not $currentStatus.complete) {
            Write-Host "  Resuming existing sprint for $areaId..." -ForegroundColor Yellow
        } else {
            # Create seed PRD - US-001 will generate the rest of the stories
            Write-Host "  Creating seed PRD for $areaId..." -ForegroundColor Yellow
            $seedCreated = New-SeedPRD -FocusAreaId $areaId -Context $context

            if (-not $seedCreated) {
                Write-Host "  Failed to create seed PRD for $areaId, skipping..." -ForegroundColor Red
                continue
            }
        }

        Start-Sleep -Seconds 2

        # Pre-flight: batch check for stories already committed in git
        $preFlightCompleted = Invoke-BatchPreFlight

        # Work through stories until sprint complete (US-001 generates the rest)
        $sprintComplete = $false
        while (-not $sprintComplete -and -not (Test-MaxIterations) -and -not (Test-TokenBudget) -and -not (Test-ShouldAbort)) {
            $status = Get-SprintStatus

            if ($status.complete) {
                Write-Host "  Sprint complete for $areaId!" -ForegroundColor Green
                $sprintComplete = $true
                Update-InterviewProgress -AreaId $areaId
                $completedAreas += $area

                # Check for graceful stop before moving to next focus area
                if (Test-GracefulStopRequested) {
                    Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                    Clear-GracefulStopSignal
                    $script:GracefulStopTriggered = $true
                    break
                }
                break
            }

            if ($status.nextStory) {
                Write-Host "  Next story: $($status.nextStory.id) - $($status.nextStory.title)" -ForegroundColor White
                $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

                if (-not $success) {
                    Write-Host "  Story failed, continuing..." -ForegroundColor Yellow
                }
            }
            else {
                Write-Host "  No stories found in PRD for $areaId" -ForegroundColor Yellow
                break
            }

            Start-Sleep -Seconds 2
        }

        # Check abort conditions
        if (Test-ShouldAbort) {
            break
        }

        if (Test-MaxIterations) {
            break
        }

        # Check if graceful stop was triggered
        if ($script:GracefulStopTriggered) {
            break
        }

        # Brief pause between focus areas
        Start-Sleep -Seconds 2
    }

    # Check if graceful stop was triggered - show appropriate message
    if ($script:GracefulStopTriggered) {
        Write-Host ""
        Write-Host "  Graceful stop completed. Sprint work finished cleanly." -ForegroundColor Green
        Write-Host "  Run with -Resume to continue remaining focus areas." -ForegroundColor DarkGray
        return
    }

    # Check if all areas completed
    $remainingAreas = Get-InterviewFocusAreas
    if ($remainingAreas.Count -eq 0) {
        # All done - show completion choice
        $choice = Show-CompletionChoice -CompletedAreas $completedAreas

        switch ($choice) {
            "interview" {
                Write-Host ""
                Write-Host "  Starting new interview..." -ForegroundColor Cyan
                # Launch interview.ps1
                $interviewPath = Join-Path $script:RalphDir "interview.ps1"
                & $interviewPath
            }
            "trueauto" {
                Write-Host ""
                Write-Host "  Switching to TrueAuto mode..." -ForegroundColor Magenta
                Start-TrueAutoLoop
            }
            default {
                Write-Host ""
                Write-Host "  'My cat's breath smells like cat food.' - Ralph" -ForegroundColor Yellow
                Write-Host "  Goodbye!" -ForegroundColor Cyan
            }
        }
    }
    else {
        Write-Host ""
        Write-Host "  $($remainingAreas.Count) focus areas remaining" -ForegroundColor Yellow
        Write-Host "  Run with -Resume to continue" -ForegroundColor DarkGray
    }
}

function Start-TrueAutoLoop {
    <#
    .SYNOPSIS
        Continuous improvement mode - work through stories until max iterations
    #>

    $script:CurrentMode = "TrueAuto"
    Write-Host "  TrueAuto mode: Continuous improvement" -ForegroundColor Magenta
    Write-Host ""

    # Pre-flight: batch check for stories already committed in git
    $preFlightCompleted = Invoke-BatchPreFlight

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget)) {
        $status = Get-SprintStatus

        if ($status.complete) {
            Write-Host ""
            Write-Host "  Sprint complete!" -ForegroundColor Green

            # Archive the completed sprint
            Save-SprintArchive -Reason "complete"

            # Check for graceful stop BEFORE generating new sprint
            if (Test-GracefulStopRequested) {
                Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                Clear-GracefulStopSignal
                break
            }

            Write-Host "  TrueAuto will generate new stories..." -ForegroundColor Magenta

            # Use -FocusArea parameter if provided, otherwise use Ralph's Choice scoring
            if ($FocusArea) {
                $focusTarget = $FocusArea
                Write-Host "  Focus: $focusTarget (from parameter)" -ForegroundColor Yellow
            } else {
                # Use Ralph's Choice scoring to pick the best area
                $scores = Get-AllFocusAreaScores
                $topArea = $scores[0]
                $focusTarget = $topArea.areaId
                Write-Host "  Focus: $focusTarget (Ralph's Choice score: $($topArea.total))" -ForegroundColor Yellow
            }

            # Generate new PRD using Invoke-ClaudeForFocusArea (archive happens in New-SeedPRD)
            $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $focusTarget -Context "" -GeneratePRD

            # Pre-flight: check new PRD for already-committed stories
            Invoke-BatchPreFlight | Out-Null

            Start-Sleep -Seconds 2
            continue
        }

        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Check for periodic exploration after story completion
            if ($success) {
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
        }
        else {
            Write-Host "  No next story found" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "  TrueAuto session complete" -ForegroundColor Magenta
    Write-Host "  Iterations: $script:IterationCount" -ForegroundColor DarkGray
}

function Start-StandardLoop {
    <#
    .SYNOPSIS
        Standard mode - work through stories until sprint complete
        After sprint complete, checks queue.json for pending focus areas and advances
    #>

    $script:CurrentMode = "Standard"

    # Check if we need to generate a new PRD for the queued focus area
    $queuedArea = Get-NextQueuedFocusArea
    if ($queuedArea) {
        # Check if this is a fresh queue (no completed areas)
        $queueData = $null
        if (Test-Path $script:QueueFile) {
            $queueData = Get-Content $script:QueueFile -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        }
        $completedCount = 0
        if ($queueData -and $queueData.focusAreas) {
            $completedCount = ($queueData.focusAreas | Where-Object { $_.completed }).Count
        }
        $isFreshQueue = ($completedCount -eq 0)

        # Use Test-ShouldGenerateNewPRD for decision
        $decision = Test-ShouldGenerateNewPRD -NewFocusArea $queuedArea

        # Fresh queue always generates new PRD (overrides continue decision)
        if ($isFreshQueue -and -not $decision.ShouldGenerate) {
            $decision = @{ ShouldGenerate = $true; Reason = "Fresh queue detected - generating new sprint" }
        }

        if ($decision.ShouldGenerate) {
            Write-Host "  $($decision.Reason)" -ForegroundColor Yellow
            Write-Host "  Generating new sprint PRD for: $queuedArea..." -ForegroundColor Yellow
            Write-Host ""

            $context = Get-InterviewContext
            $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $queuedArea -Context $context -GeneratePRD

            if (-not $prdGenerated) {
                Write-Host "  Failed to generate PRD for $queuedArea" -ForegroundColor Red
                return
            }
            Write-Host "  PRD generated. Starting work on $queuedArea" -ForegroundColor Green
            Write-Host ""
        } else {
            Write-Host "  $($decision.Reason)" -ForegroundColor DarkGray
        }
    }

    # Pre-flight: batch check for stories already committed in git
    $preFlightCompleted = Invoke-BatchPreFlight

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget)) {
        $status = Get-SprintStatus

        if ($status.complete) {
            Write-Host ""
            Write-Host "=====================================================" -ForegroundColor Green
            Write-Host "   SPRINT COMPLETE!" -ForegroundColor Green
            Write-Host "=====================================================" -ForegroundColor Green
            Write-Host ""
            Write-Host "  All $($status.total) stories passed" -ForegroundColor Green
            Write-Host "  Focus area: $($status.focusArea)" -ForegroundColor Cyan
            Write-Host ""

            # Archive the completed sprint
            Save-SprintArchive -Reason "complete"

            # Check for graceful stop BEFORE queue processing
            if (Test-GracefulStopRequested) {
                Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                Clear-GracefulStopSignal
                break
            }

            # Check for pending queue items (legacy format)
            $nextArea = Get-NextQueuedFocusArea
            if ($nextArea) {
                Write-Host "  Queue has more focus areas. Next: $nextArea" -ForegroundColor Cyan
                Write-Host ""

                # Update queue to mark current area as complete
                if ($status.focusArea) {
                    Update-LegacyQueueProgress -CompletedArea $status.focusArea
                }

                # Use Test-ShouldGenerateNewPRD for decision
                $decision = Test-ShouldGenerateNewPRD -NewFocusArea $nextArea
                Write-Host "  $($decision.Reason)" -ForegroundColor DarkGray

                # Generate new PRD for next focus area
                Write-Host "  Generating PRD for focus area: $nextArea..." -ForegroundColor Yellow
                $context = Get-InterviewContext
                $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $nextArea -Context $context -GeneratePRD

                if ($prdGenerated) {
                    Write-Host "  PRD generated. Continuing with $nextArea" -ForegroundColor Green
                    # Pre-flight: check new PRD for already-committed stories
                    Invoke-BatchPreFlight | Out-Null
                    # Continue the loop - don't break
                    Start-Sleep -Seconds 2
                    continue
                }
                else {
                    Write-Host "  Failed to generate PRD for $nextArea" -ForegroundColor Red
                    break
                }
            }
            else {
                # Mark final focus area as completed before exiting
                if ($status.focusArea) {
                    Update-InterviewProgress -AreaId $status.focusArea
                    Update-LegacyQueueProgress -CompletedArea $status.focusArea
                }
                Write-Host "  Queue complete! All focus areas done." -ForegroundColor Green
                break
            }
        }

        if ($status.nextStory) {
            Write-Host "  Next story: $($status.nextStory.id) - $($status.nextStory.title)" -ForegroundColor White

            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Check for periodic exploration after story completion
            if ($success) {
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
        }
        else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 2
    }
}

# ============================================================================
# RALPH'S CHOICE - LOOP FUNCTIONS
# ============================================================================

function Show-RalphsReasoning {
    <#
    .SYNOPSIS
        Display scoring breakdown and recommendation
    .PARAMETER Scores
        Array of score hashtables from Get-AllFocusAreaScores
    .PARAMETER Decision
        Optional stay/switch decision hashtable
    .PARAMETER ShowAllScores
        If true, show all scores instead of top 5
    #>
    param(
        [Parameter(Mandatory=$true)]
        $Scores,
        $Decision = $null,
        [switch]$ShowAllScores
    )

    $config = Get-RalphConfig
    $sprintHistory = Get-SprintHistory
    $windowDays = if ($config.ralphsChoice.activityWindowDays) { $config.ralphsChoice.activityWindowDays } else { 7 }
    $activityByArea = Get-GitActivityByArea -WindowDays $windowDays

    Write-Host ""
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host "     Ralph's Choice - Analyzing project state..." -ForegroundColor Magenta
    Write-Host "  =====================================================" -ForegroundColor Magenta
    Write-Host ""

    # Git Activity summary
    Write-Host "  Git Activity (last $windowDays days):" -ForegroundColor Yellow
    $topActivity = $activityByArea.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5
    foreach ($entry in $topActivity) {
        if ($entry.Value -gt 0) {
            Write-Host "    * $($entry.Key.PadRight(20)) $($entry.Value) commits" -ForegroundColor White
        }
    }
    if (-not $topActivity -or ($topActivity | Measure-Object).Count -eq 0) {
        Write-Host "    (no recent activity)" -ForegroundColor DarkGray
    }
    Write-Host ""

    # Sprint History summary
    Write-Host "  Sprint History:" -ForegroundColor Yellow
    $totalSprints = [int]$sprintHistory.totalSprintsCompleted
    $totalStories = [int]$sprintHistory.totalStoriesCompleted
    Write-Host "    Total: $totalSprints sprints, $totalStories stories" -ForegroundColor White

    if ($sprintHistory.focusAreaBreakdown) {
        foreach ($prop in $sprintHistory.focusAreaBreakdown.PSObject.Properties) {
            $areaId = $prop.Name
            $stats = $prop.Value
            $sprints = [int]$stats.sprints
            $stories = [int]$stats.stories

            # Find last sprint time
            $lastTime = "never"
            $areaSprints = $sprintHistory.sprints | Where-Object { $_.focusArea -eq $areaId } | Sort-Object completedAt -Descending
            if ($areaSprints -and $areaSprints.Count -gt 0) {
                $lastSprint = $areaSprints | Select-Object -First 1
                if ($lastSprint.completedAt) {
                    $hoursAgo = [math]::Round(((Get-Date) - [datetime]$lastSprint.completedAt).TotalHours, 0)
                    if ($hoursAgo -lt 24) {
                        $lastTime = "${hoursAgo}h ago"
                    } else {
                        $daysAgo = [math]::Floor($hoursAgo / 24)
                        $lastTime = "${daysAgo}d ago"
                    }
                }
            }

            Write-Host "    * ${areaId}: $sprints sprints ($stories stories) - last: $lastTime" -ForegroundColor White
        }
    }
    Write-Host ""

    # Category Coverage
    Write-Host "  Category Coverage:" -ForegroundColor Yellow
    foreach ($catId in $config.categoryOrder) {
        $category = $config.focusAreaCategories.$catId
        if (-not $category) { continue }

        $catSprints = 0
        foreach ($areaId in $category.areas) {
            if ($sprintHistory.focusAreaBreakdown -and $sprintHistory.focusAreaBreakdown.$areaId) {
                $catSprints += [int]$sprintHistory.focusAreaBreakdown.$areaId.sprints
            }
        }

        $percentage = if ($totalSprints -gt 0) { [math]::Round(($catSprints / $totalSprints) * 100, 0) } else { 0 }
        $status = if ($percentage -gt 50) { "(over-represented)" } elseif ($percentage -eq 0) { "(untouched)" } else { "" }
        Write-Host "    * ${catId}: ${percentage}% of work $status" -ForegroundColor White
    }
    Write-Host ""

    # Scores
    Write-Host "  Scores:" -ForegroundColor Yellow
    $displayScores = if ($ShowAllScores) { $Scores } else { $Scores | Select-Object -First 5 }
    foreach ($score in $displayScores) {
        $dots = "." * (25 - $score.areaId.Length)
        Write-Host "    $($score.areaId) $dots $($score.total)" -ForegroundColor White
    }
    Write-Host ""

    # Recommendation
    $topScore = $Scores[0]
    Write-Host "  > Ralph recommends: " -ForegroundColor Cyan -NoNewline
    Write-Host $topScore.areaId -ForegroundColor Green
    Write-Host "    Reason: " -ForegroundColor Cyan -NoNewline

    $reasons = @()
    if ($topScore.commits -gt 0) {
        $reasons += "Git activity ($($topScore.commits) commits)"
    }
    if ($topScore.sprintsDone -eq 0) {
        $reasons += "never worked on"
    }
    if ($topScore.balanceScore -gt 0.5) {
        $reasons += "category needs balance"
    }

    if ($reasons.Count -eq 0) {
        $reasons += "highest overall score"
    }

    Write-Host ($reasons -join " + ") -ForegroundColor White
    Write-Host ""

    # If this is a stay/switch decision, show that context
    if ($Decision) {
        Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Stay vs Switch Analysis:" -ForegroundColor Yellow
        Write-Host ""

        if ($Decision.stayReasons.Count -gt 0) {
            Write-Host "  Reasons to STAY in $($Decision.currentArea):" -ForegroundColor White
            foreach ($reason in $Decision.stayReasons) {
                Write-Host "    + $reason" -ForegroundColor Green
            }
        } else {
            Write-Host "  Reasons to STAY:" -ForegroundColor White
            Write-Host "    (none)" -ForegroundColor DarkGray
        }
        Write-Host ""

        if ($Decision.switchReasons.Count -gt 0) {
            Write-Host "  Reasons to SWITCH:" -ForegroundColor White
            foreach ($reason in $Decision.switchReasons) {
                Write-Host "    + $reason" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Reasons to SWITCH:" -ForegroundColor White
            Write-Host "    (none)" -ForegroundColor DarkGray
        }
        Write-Host ""

        $decisionText = if ($Decision.decision -eq "switch") { "SWITCH to $($Decision.newArea)" } else { "STAY in $($Decision.currentArea)" }
        Write-Host "  > Ralph recommends: " -ForegroundColor Cyan -NoNewline
        Write-Host $decisionText -ForegroundColor $(if ($Decision.decision -eq "switch") { "Yellow" } else { "Green" })
        Write-Host ""
    }
}

function Get-RalphsChoiceUserInput {
    <#
    .SYNOPSIS
        Get user input for Ralph's Choice decision
    .PARAMETER Scores
        Array of score hashtables
    .PARAMETER IsStaySwitch
        If true, this is a between-sprint decision
    .PARAMETER CurrentArea
        Current focus area (for stay/switch)
    .RETURNS
        Hashtable with: action (accept/manual/stay/showAll), selectedArea
    #>
    param(
        [Parameter(Mandatory=$true)]
        $Scores,
        [switch]$IsStaySwitch,
        [string]$CurrentArea = ""
    )

    Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
    if ($IsStaySwitch) {
        Write-Host "  [Enter] Accept   [S] See all scores   [K] Keep $CurrentArea   [M] Manual" -ForegroundColor White
    } else {
        Write-Host "  [Enter] Accept   [S] See all scores   [M] Manual override" -ForegroundColor White
    }
    Write-Host "  -----------------------------------------------------" -ForegroundColor DarkGray
    Write-Host ""

    $response = Read-Host "  Choice"

    switch -Regex ($response) {
        "^[Ss]$" {
            return @{ action = "showAll"; selectedArea = $null }
        }
        "^[Kk]$" {
            if ($IsStaySwitch -and $CurrentArea) {
                return @{ action = "stay"; selectedArea = $CurrentArea }
            }
            return @{ action = "accept"; selectedArea = $Scores[0].areaId }
        }
        "^[Mm]$" {
            Write-Host ""
            Write-Host "  Enter focus area ID:" -ForegroundColor Cyan
            $manualArea = Read-Host "  "
            return @{ action = "manual"; selectedArea = $manualArea }
        }
        default {
            return @{ action = "accept"; selectedArea = $Scores[0].areaId }
        }
    }
}

function Write-RalphsChoiceLog {
    <#
    .SYNOPSIS
        Log Ralph's Choice decisions to file
    .PARAMETER Decision
        The decision made (area selected, stay/switch)
    .PARAMETER Reason
        Reason for the decision
    .PARAMETER Scores
        Score breakdown at time of decision
    #>
    param(
        [string]$Decision,
        [string]$Reason,
        $Scores
    )

    $config = Get-RalphConfig
    $logFile = if ($config.ralphsChoice.autoMode.logFile) { $config.ralphsChoice.autoMode.logFile } else { "ralphs_choices.log" }
    $logPath = Join-Path $script:RalphDir $logFile

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $topScores = ($Scores | Select-Object -First 3 | ForEach-Object { "$($_.areaId):$($_.total)" }) -join ", "

    $logLine = "[$timestamp] Decision: $Decision | Reason: $Reason | TopScores: $topScores"
    Add-Content -Path $logPath -Value $logLine
}

function Start-RalphsChoiceLoop {
    <#
    .SYNOPSIS
        Ralph's Choice mode - Ralph decides focus areas, user confirms each
    #>

    $script:CurrentMode = "RalphsChoice"
    Write-Host "  Ralph's Choice mode: Ralph decides, you confirm" -ForegroundColor Magenta
    Write-Host ""

    $sprintCount = 0
    $config = Get-RalphConfig

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget)) {
        $status = Get-SprintStatus

        if ($status.complete -or $sprintCount -eq 0) {
            if ($sprintCount -gt 0) {
                Write-Host ""
                Write-Host "=====================================================" -ForegroundColor Green
                Write-Host "   SPRINT $sprintCount COMPLETE!" -ForegroundColor Green
                Write-Host "=====================================================" -ForegroundColor Green
                Write-Host ""

                # Archive the completed sprint
                Save-SprintArchive -Reason "complete"

                # Check for graceful stop
                if (Test-GracefulStopRequested) {
                    Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                    Clear-GracefulStopSignal
                    break
                }

                # Get stay/switch decision
                $decision = Get-StayOrSwitchDecision -CurrentArea $status.focusArea
                $scores = $decision.scores

                # Show reasoning with stay/switch context
                Show-RalphsReasoning -Scores (Get-AllFocusAreaScores) -Decision $decision

                # Get user input
                $userChoice = Get-RalphsChoiceUserInput -Scores $scores -IsStaySwitch -CurrentArea $status.focusArea

                while ($userChoice.action -eq "showAll") {
                    Show-RalphsReasoning -Scores (Get-AllFocusAreaScores) -Decision $decision -ShowAllScores
                    $userChoice = Get-RalphsChoiceUserInput -Scores $scores -IsStaySwitch -CurrentArea $status.focusArea
                }

                $selectedArea = if ($userChoice.action -eq "stay") {
                    $status.focusArea
                } elseif ($userChoice.action -eq "manual") {
                    $userChoice.selectedArea
                } else {
                    $decision.newArea
                }
            } else {
                # First sprint - just pick
                $scores = Get-AllFocusAreaScores
                Show-RalphsReasoning -Scores $scores

                $userChoice = Get-RalphsChoiceUserInput -Scores $scores

                while ($userChoice.action -eq "showAll") {
                    Show-RalphsReasoning -Scores $scores -ShowAllScores
                    $userChoice = Get-RalphsChoiceUserInput -Scores $scores
                }

                $selectedArea = if ($userChoice.action -eq "manual") {
                    $userChoice.selectedArea
                } else {
                    $scores[0].areaId
                }
            }

            Write-Host ""
            Write-Host "  Selected focus area: $selectedArea" -ForegroundColor Green
            Write-Host ""

            # Generate PRD for selected area
            $context = Get-InterviewContext
            $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $selectedArea -Context $context -GeneratePRD

            if (-not $prdGenerated) {
                Write-Host "  Failed to generate PRD for $selectedArea" -ForegroundColor Red
                break
            }

            # Pre-flight: check new PRD for already-committed stories
            Invoke-BatchPreFlight | Out-Null

            $sprintCount++
            Start-Sleep -Seconds 2
            continue
        }

        # Work on current story
        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Check for periodic exploration after story completion
            if ($success) {
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
        } else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "  Ralph's Choice session complete" -ForegroundColor Magenta
    Write-Host "  Sprints completed: $sprintCount" -ForegroundColor DarkGray
}

function Start-RalphsChoiceAutoLoop {
    <#
    .SYNOPSIS
        Ralph's Choice Auto mode - fully autonomous with countdown
    #>

    $script:CurrentMode = "RalphsChoiceAuto"
    Write-Host "  Ralph's Choice Auto: Fully autonomous" -ForegroundColor Magenta
    Write-Host ""

    $sprintCount = 0
    $config = Get-RalphConfig
    $continueDelay = if ($config.ralphsChoice.autoMode.continueDelaySeconds) { $config.ralphsChoice.autoMode.continueDelaySeconds } else { 10 }
    $maxSprints = if ($config.ralphsChoice.autoMode.maxConsecutiveSprints) { $config.ralphsChoice.autoMode.maxConsecutiveSprints } else { 20 }

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget) -and $sprintCount -lt $maxSprints) {
        $status = Get-SprintStatus

        if ($status.complete -or $sprintCount -eq 0) {
            if ($sprintCount -gt 0) {
                Write-Host ""
                Write-Host "=====================================================" -ForegroundColor Green
                Write-Host "   SPRINT $sprintCount COMPLETE!" -ForegroundColor Green
                Write-Host "=====================================================" -ForegroundColor Green
                Write-Host ""

                # Archive the completed sprint
                Save-SprintArchive -Reason "complete"

                # Check for graceful stop
                if (Test-GracefulStopRequested) {
                    Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                    Clear-GracefulStopSignal
                    break
                }
            }

            # Get decision
            $scores = Get-AllFocusAreaScores
            $selectedArea = $scores[0].areaId
            $reason = "Highest score"

            if ($sprintCount -gt 0 -and $status.focusArea) {
                $decision = Get-StayOrSwitchDecision -CurrentArea $status.focusArea
                if ($decision.decision -eq "stay") {
                    $selectedArea = $decision.currentArea
                    $reason = "Staying: " + ($decision.stayReasons -join ", ")
                } else {
                    $selectedArea = $decision.newArea
                    $reason = "Switching: " + ($decision.switchReasons -join ", ")
                }
            }

            # Log the decision
            Write-RalphsChoiceLog -Decision $selectedArea -Reason $reason -Scores $scores

            # Show brief status
            Write-Host ""
            Write-Host "  -----------------------------------------------------" -ForegroundColor Magenta
            Write-Host "  Ralph's Choice Auto" -ForegroundColor Magenta
            Write-Host "  -----------------------------------------------------" -ForegroundColor Magenta
            Write-Host ""
            Write-Host "  Decision: " -ForegroundColor Cyan -NoNewline
            Write-Host $selectedArea -ForegroundColor Green
            Write-Host "  Reason: $reason" -ForegroundColor DarkGray
            Write-Host ""

            # Countdown with interrupt option
            Write-Host "  Continuing in ${continueDelay}s... [Press any key to pause]" -ForegroundColor Yellow

            $interrupted = $false
            for ($i = $continueDelay; $i -gt 0; $i--) {
                if ([Console]::KeyAvailable) {
                    [Console]::ReadKey($true) | Out-Null
                    $interrupted = $true
                    break
                }
                Write-Host "`r  Continuing in ${i}s... [Press any key to pause]  " -ForegroundColor Yellow -NoNewline
                Start-Sleep -Seconds 1
            }
            Write-Host ""

            if ($interrupted) {
                Write-Host ""
                Write-Host "  Paused! Options:" -ForegroundColor Cyan
                Write-Host "    [C] Continue with $selectedArea" -ForegroundColor White
                Write-Host "    [M] Manual override" -ForegroundColor White
                Write-Host "    [S] Show full analysis" -ForegroundColor White
                Write-Host "    [Q] Quit" -ForegroundColor White
                Write-Host ""

                $pauseChoice = Read-Host "  Choice"

                switch -Regex ($pauseChoice) {
                    "^[Qq]$" {
                        Write-Host "  Exiting Ralph's Choice Auto" -ForegroundColor Yellow
                        break
                    }
                    "^[Mm]$" {
                        Write-Host "  Enter focus area ID:" -ForegroundColor Cyan
                        $selectedArea = Read-Host "  "
                    }
                    "^[Ss]$" {
                        Show-RalphsReasoning -Scores $scores -ShowAllScores
                        Write-Host "  Press Enter to continue with $selectedArea, or type new area:" -ForegroundColor Yellow
                        $override = Read-Host "  "
                        if ($override -and $override.Trim() -ne "") {
                            $selectedArea = $override.Trim()
                        }
                    }
                    # Default: continue with selected area
                }

                if ($pauseChoice -match "^[Qq]$") {
                    break
                }
            }

            Write-Host ""
            Write-Host "  Starting sprint for: $selectedArea" -ForegroundColor Green
            Write-Host ""

            # Generate PRD for selected area
            $context = Get-InterviewContext
            $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $selectedArea -Context $context -GeneratePRD

            if (-not $prdGenerated) {
                Write-Host "  Failed to generate PRD for $selectedArea" -ForegroundColor Red
                break
            }

            # Pre-flight: check new PRD for already-committed stories
            Invoke-BatchPreFlight | Out-Null

            $sprintCount++
            Start-Sleep -Seconds 2
            continue
        }

        # Work on current story
        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Check for periodic exploration after story completion
            if ($success) {
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
        } else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 2
    }

    if ($sprintCount -ge $maxSprints) {
        Write-Host ""
        Write-Host "  Reached max consecutive sprints limit ($maxSprints)" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  Ralph's Choice Auto session complete" -ForegroundColor Magenta
    Write-Host "  Sprints completed: $sprintCount" -ForegroundColor DarkGray
}

# ============================================================================
# ENTRY POINT
# ============================================================================

Write-RalphBanner

# Initialize graceful stop state
$script:GracefulStopTriggered = $false

# Clear stale graceful stop signals from previous sessions
$staleSignal = Join-Path $script:RalphDir "graceful_stop.signal"
if (Test-Path $staleSignal) {
    Write-Host "  Clearing stale graceful stop signal" -ForegroundColor DarkGray
    Remove-Item $staleSignal -Force -ErrorAction SilentlyContinue
}

# Validate Claude is available
$claudePath = Get-ClaudePath
Write-Host "  Claude path: $claudePath" -ForegroundColor DarkGray

# Check for Claude availability
try {
    $versionCheck = & $claudePath --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Warning: Claude may not be available" -ForegroundColor Yellow
    }
    else {
        Write-Host "  Claude version: $versionCheck" -ForegroundColor DarkGray
    }
}
catch {
    Write-Host "  Warning: Could not verify Claude installation" -ForegroundColor Yellow
    Write-Host "  Error: $_" -ForegroundColor Red
}

Write-Host ""

# Determine mode for logging
$sessionMode = if ($Queue) { "Queue" }
    elseif ($TrueAuto) { "TrueAuto" }
    elseif ($RalphsChoice) { "RalphsChoice" }
    elseif ($RalphsChoiceAuto) { "RalphsChoiceAuto" }
    else { "Standard" }

# Log session start event
Append-SessionTimeline -Event "session_start" -Data @{
    mode = $sessionMode
    claudePath = $claudePath
    projectRoot = $script:ProjectRoot
}

# Route to appropriate loop based on flags
# Note: each mode function runs its own pre-flight at the right time (after PRD generation)
if ($Queue) {
    Start-InterviewQueueLoop
}
elseif ($TrueAuto) {
    Start-TrueAutoLoop
}
elseif ($RalphsChoice) {
    Start-RalphsChoiceLoop
}
elseif ($RalphsChoiceAuto) {
    Start-RalphsChoiceAutoLoop
}
else {
    Start-StandardLoop
}

# Session summary
$duration = (Get-Date) - $script:SessionStartTime

# Log session end event
Append-SessionTimeline -Event "session_end" -Data @{
    iterations = $script:IterationCount
    durationMin = [math]::Round($duration.TotalMinutes, 1)
    consecutiveFailures = $script:ConsecutiveFailures
}
Write-Host ""
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Session Summary" -ForegroundColor Cyan
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Session ID: $script:SessionId" -ForegroundColor DarkGray
Write-Host "  Iterations: $script:IterationCount" -ForegroundColor DarkGray
Write-Host "  Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor DarkGray
Write-Host "  Logs: $script:SessionLogDir" -ForegroundColor DarkGray
Write-Host ""
