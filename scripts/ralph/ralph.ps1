# Ralph Loop - Main Execution Engine (INTERNAL USE ONLY)
# =====================================================================
# DO NOT CALL THIS SCRIPT DIRECTLY - Use interview.ps1 instead!
#
# This is the core execution engine that processes sprints and stories.
# It is spawned by interview.ps1 with the appropriate parameters.
#
# User entry point: .\scripts\ralph\interview.ps1
# =====================================================================
#
# Usage (internal): .\ralph.ps1 [-Queue] [-TrueAuto] [-Resume] [-FocusArea <area>]
#
# STANDALONE SCRIPT - Do not define functions here that are called from lib/
# All shared functions belong in lib/*.ps1
#
# Modes (set by interview.ps1):
#   -Queue             Process focus areas from queue.json (interview mode)
#   -SkipPlanApproval  Skip plan approval prompts
#   -TrueAuto          Continuous improvement mode (no exit on sprint complete)
#   -Resume            Resume previous sprint instead of starting new
#   -FocusArea <area>  Override focus area for this session
#   -RalphsChoice      Ralph decides focus areas, user confirms each decision
#   -RalphsChoiceAuto  Ralph decides and continues autonomously
#   -Overnight         Adaptive multi-focus overnight mode with area rotation
#   -MaxHours <hours>  Max hours for overnight mode (default: 12)

param(
    [switch]$Queue,
    [switch]$SkipPlanApproval,
    [switch]$TrueAuto,
    [switch]$Resume,
    [switch]$RalphsChoice,
    [switch]$RalphsChoiceAuto,
    [switch]$Overnight,
    [int]$MaxHours = 12,
    [string]$FocusArea = "",
    [string[]]$FocusAreas = @(),
    [string]$Task = ""
)

# ============================================================================
# SETUP
# ============================================================================

$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$script:RalphDir = Join-Path $script:ProjectRoot "scripts\ralph"

# Load paths module first (provides centralized path definitions)
$script:LibPath = Join-Path $PSScriptRoot 'lib'
. "$script:LibPath\paths.ps1"

# Initialize paths and get references (creates directories if needed)
$paths = Initialize-RalphPaths -RalphDir $script:RalphDir

# Set legacy path variables for backward compatibility
# These will be gradually replaced with Resolve-RalphPath calls
$script:QueueFile = Resolve-RalphPath -PathKey 'QueueFile'
$script:ConfigFile = Resolve-RalphPath -PathKey 'ConfigFile'
$script:PrdFile = Resolve-RalphPath -PathKey 'PrdFile'
$script:ProgressFile = Resolve-RalphPath -PathKey 'ProgressFile'
$script:MetricsFile = Resolve-RalphPath -PathKey 'MetricsFile'
$script:PromptFile = Resolve-RalphPath -PathKey 'PromptFile'
$script:LogDir = $script:Paths.LogsDir
$script:ArchiveDir = $script:Paths.ArchiveDir
$script:SprintHistoryFile = Resolve-RalphPath -PathKey 'SprintHistoryFile'
$script:ExplorationContextFile = Resolve-RalphPath -PathKey 'ExplorationContextFile'
$script:HealingLogFile = Resolve-RalphPath -PathKey 'HealingLogFile'
$script:HealingStateFile = Resolve-RalphPath -PathKey 'HealingStateFile'

# Mutable session state (consolidated hashtable)
$script:State = @{
    SessionId               = (Get-Date -Format 'yyyy-MM-dd_HHmmss')
    IterationCount          = 0
    ConsecutiveFailures     = 0
    SessionStartTime        = Get-Date
    CurrentMode             = 'Standard'   # "Interview", "Standard", or "TrueAuto"
    CurrentRetryCount       = 0            # Attempts on current focus area/story
    LastFocusAreaId          = ''           # Track when focus area changes
    LastStoryId             = ''           # Track when story changes
    StoriesSinceExploration = 0            # Counter for periodic exploration
    LastExplorationSummary  = ''           # Cached exploration summary
    LastExplorationTime     = $null        # When last exploration ran
    SprintExplorationContext = ''          # Sprint-start exploration context
    LastExplorationCommit   = (git rev-parse HEAD 2>$null)  # Commit hash at last exploration
    # Heartbeat tracking
    CurrentStoryId          = ''           # Active story being worked on
    CurrentFocusArea        = ''           # Active focus area
    CurrentSprintNumber     = 0            # Current sprint number
    # Branch tracking
    LastEnsuredBranch       = ''           # Track which branch we've ensured exists
}

# Ensure logs directory exists
if (-not (Test-Path $script:LogDir)) {
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
}

# Create session log directory
$script:SessionLogDir = Join-Path $script:LogDir $script:State.SessionId
New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null

# Ensure archive directory exists
if (-not (Test-Path $script:ArchiveDir)) {
    New-Item -ItemType Directory -Path $script:ArchiveDir -Force | Out-Null
}

# Load domain modules (paths.ps1 already loaded above)
. "$script:LibPath\sprint.ps1"
. "$script:LibPath\scoring.ps1"
. "$script:LibPath\queue.ps1"
. "$script:LibPath\metrics.ps1"
. "$script:LibPath\quality.ps1"
. "$script:LibPath\healing.ps1"
. "$script:LibPath\prompts.ps1"
. "$script:LibPath\claude.ps1"
. "$script:LibPath\display.ps1"
. "$script:LibPath\loops.ps1"
. "$script:LibPath\heartbeat.ps1"
. "$script:LibPath\interview.ps1"
. "$script:LibPath\learning.ps1"
. "$script:LibPath\reporting.ps1"

# ============================================================================
# CONFIG LOADING
# ============================================================================

$script:Config = Get-RalphConfig

# ============================================================================
# CRASH RECOVERY TRAP
# ============================================================================

# Register crash handler to save state before exit
$script:CrashHandlerRegistered = $false

trap {
    Write-Host ""
    Write-Host "  *** UNEXPECTED ERROR ***" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""

    # Save crash state for recovery
    try {
        $crashInfo = @{
            crashTime = (Get-Date).ToString("o")
            sessionId = $script:State.SessionId
            lastSprint = $script:State.CurrentSprintNumber
            lastStory = $script:State.CurrentStoryId
            lastFocusArea = $script:State.CurrentFocusArea
            nextStory = ""
            error = $_.Exception.Message
            stackTrace = $_.ScriptStackTrace
            iterationCount = $script:State.IterationCount
            mode = $script:State.CurrentMode
        }

        # Try to get next story
        try {
            $status = Get-SprintStatus
            if ($status -and $status.nextStory) {
                $crashInfo.nextStory = $status.nextStory.id
            }
        }
        catch {}

        $crashFile = Join-Path $script:RalphDir "crash_recovery.json"
        $crashInfo | ConvertTo-Json -Depth 5 | Set-Content $crashFile -Encoding UTF8

        Write-Host "  Crash state saved. Run with -Resume to continue." -ForegroundColor Yellow
    }
    catch {
        Write-Host "  Warning: Could not save crash state" -ForegroundColor Yellow
    }

    # Attempt partial report
    try {
        Generate-PartialReport -Emergency
    }
    catch {}

    break
}

# ============================================================================
# CLAUDE PROCESS EXECUTION
# ============================================================================

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

    # Track retries via State hashtable
    $trackingId = if ($isStoryWork) { $storyId } else { $focusAreaId }
    $lastTrackingKey = if ($isStoryWork) { 'LastStoryId' } else { 'LastFocusAreaId' }
    if ($script:State[$lastTrackingKey] -ne $trackingId) {
        $script:State.CurrentRetryCount = 0
        $script:State[$lastTrackingKey] = $trackingId
    }
    $script:State.CurrentRetryCount++

    $script:State.IterationCount++
    $iterationStart = Get-Date

    # Save crash recovery state before each iteration
    Save-CrashRecoveryState

    Write-IterationBanner -Iteration $script:State.IterationCount -FocusArea $focusAreaId -StoryId $storyId

    # Get Claude path
    $claudePath = Get-ClaudePath

    # Build arguments
    $claudeArgs = @("--print", "--dangerously-skip-permissions")
    if ($AllowedTools) {
        $claudeArgs += "--allowedTools=Bash,Read,Write,Edit,Glob,Grep,WebSearch"
    }

    $displayPrompt = if ($isStoryWork) { "Work on $storyId" } elseif ($PromptType -eq "prd_generation") { "Generate PRD for $focusAreaId" } else { "Focus on $focusAreaId" }
    Write-Host "  Invoking Claude..." -ForegroundColor Cyan
    Write-Host "  Prompt: $displayPrompt" -ForegroundColor DarkGray

    # Capture git state BEFORE Claude runs
    $gitStateBefore = Get-GitState

    # Log state transition
    $transitionContext = @{ focusArea = $focusAreaId; promptType = $PromptType }
    if ($storyId) { $transitionContext.storyId = $storyId }
    Log-StateTransition -From "idle" -To "running" -Reason "Starting: $displayPrompt" -Context $transitionContext

    # Log timeline event: iteration start
    $timelineData = @{ iteration = $script:State.IterationCount; focusArea = $focusAreaId; promptType = $PromptType }
    if ($storyId) { $timelineData.storyId = $storyId }
    Append-SessionTimeline -Event "iteration_start" -Data $timelineData

    try {
        # Output file paths
        $outFile = Join-Path $script:SessionLogDir "claude_out_$($script:State.IterationCount).log"
        $errFile = Join-Path $script:SessionLogDir "claude_err_$($script:State.IterationCount).log"
        $promptFile = Join-Path $script:SessionLogDir "prompt_$($script:State.IterationCount).txt"

        # Write prompt to file
        $Prompt | Out-File -FilePath $promptFile -Encoding UTF8 -NoNewline

        # === SUBPROCESS WITH INFINITE RETRY ===
        # Track story start time for 30-minute limit
        if (-not $script:State.CurrentStoryStartTime) {
            $script:State.CurrentStoryStartTime = Get-Date
        }

        $subResult = Invoke-ClaudeWithInfiniteRetry `
            -ClaudePath $claudePath `
            -ClaudeArgs $claudeArgs `
            -Prompt $Prompt `
            -OutFile $outFile `
            -ErrFile $errFile `
            -StoryId $(if ($storyId) { $storyId } else { $Identifier }) `
            -StoryStartTime $script:State.CurrentStoryStartTime `
            -FocusArea $focusAreaId

        # === COMPUTE METRICS ===
        $iterationDuration = (Get-Date) - $iterationStart
        $claudeOutput = $subResult.Output

        $tokensUsed = Get-EstimatedTokens -Output $claudeOutput
        $testResults = Get-TestResults -Output $claudeOutput

        $gitStateAfter = Get-GitState
        $fileOps = Get-FileOperations -BeforeHash $gitStateBefore.hash
        $commits = Get-GitCommits -SinceHash $gitStateBefore.hash

        $executionDurationMs = [int](($subResult.ExecutionEnd - $subResult.ExecutionStart).TotalMilliseconds)
        $phaseTimings = Measure-PhaseTimings -Output $claudeOutput -TotalDurationMs $executionDurationMs

        # === RESOLVE RESULT ===
        $resolution = Resolve-ClaudeResult -SubResult $subResult -Ctx @{
            TransitionContext = $transitionContext
            IsStoryWork       = $isStoryWork
            StoryId           = $storyId
            FocusAreaId       = $focusAreaId
            Identifier        = $Identifier
            StoryObj          = $StoryObj
            IterationDuration = $iterationDuration
            TokensUsed        = $tokensUsed
            TestResults       = $testResults
            PhaseTimings      = $phaseTimings
            GitStateBefore    = $gitStateBefore
            FileOps           = $fileOps
            Commits           = $commits
            ClaudeOutput      = $claudeOutput
        }

        # === LOG ===
        Record-IterationLog -Ctx @{
            Iteration       = $script:State.IterationCount
            ClaudePath      = $claudePath
            ClaudeArgs      = $claudeArgs
            PromptFile      = $promptFile
            PromptType      = $PromptType
            ProcessId       = $subResult.ProcessId
            ExecutionStart  = $subResult.ExecutionStart
            ExecutionEnd    = $subResult.ExecutionEnd
            ExitCode        = $subResult.ExitCode
            TimedOut        = $subResult.TimedOut
            StoryId         = $storyId
            FocusAreaId     = $focusAreaId
            IterationStatus = $resolution.IterationStatus
            IterationStart  = $iterationStart
            GitStateBefore  = $gitStateBefore
            GitStateAfter   = $gitStateAfter
            FileOps         = $fileOps
            Commits         = $commits
            TestResults     = $testResults
            TokensUsed      = $tokensUsed
            IterationDuration = $iterationDuration
            Success         = $resolution.Success
            ClaudeOutput    = $claudeOutput
            ResourceSamples = $subResult.ResourceSamples
        }

        return $resolution.Success
    }
    catch {
        Write-Host "  Error invoking Claude: $_" -ForegroundColor Red
        $errorCategory = Get-ErrorCategory -Output $_.ToString() -TimedOut $false

        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails $_.ToString() -Iteration $script:State.IterationCount
        Log-StateTransition -From "running" -To "error" -Reason $_.ToString() -Context $transitionContext

        Record-Metric -StoryId $Identifier -Mode $script:State.CurrentMode -DurationMin 0 -Success $false -Timeout $false -TokensUsed 0 -ErrorCategory $errorCategory -TestResults "" -RetryCount $script:State.CurrentRetryCount -LinesAdded 0 -LinesDeleted 0 -PhaseReadMs 0 -PhaseAnalyzeMs 0 -PhaseImplementMs 0 -PhaseTestMs 0 -PhaseCommitMs 0

        $errorData = @{ iteration = $script:State.IterationCount; error = $_.ToString() }
        if ($storyId) { $errorData.storyId = $storyId }
        Append-SessionTimeline -Event "iteration_error" -Data $errorData

        $script:State.ConsecutiveFailures++
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

# Invoke-ClaudeForFocusArea moved to lib/claude.ps1

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
        $prd = Get-Sprint
        if ($prd) {
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
                Save-StateFile -Path $script:PrdFile -Data $prd
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
            $sessionId = if ($script:State.SessionId) { $script:State.SessionId } else { "preflight" }
            $focusArea = ""
            $prdData = Get-Sprint
            if ($prdData) {
                $focusArea = $prdData.focusArea
            }
            $sprintNum = if ($prdData -and $prdData.sprintNumber) { $prdData.sprintNumber } else { "0" }
            $sprintName = "sprint-$sprintNum"
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
    $prd = Get-Sprint
    if ($prd) {
        $focusArea = $prd.focusArea
        $storyObj = $prd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
    }

    # Update state for heartbeat tracking
    $script:State.CurrentStoryId = $StoryId
    $script:State.CurrentFocusArea = $focusArea
    $script:State.CurrentSprintNumber = if ($prd.sprintNumber) { $prd.sprintNumber } else { 0 }

    # Reset story start time for 30-minute limit tracking
    $script:State.CurrentStoryStartTime = Get-Date

    # Log story start
    Write-SessionLog -Event "story_start" -Message "Starting story $StoryId" -Data @{
        focusArea = $focusArea
        title = if ($storyObj) { $storyObj.title } else { "unknown" }
    }
    Write-Heartbeat -Phase "story_starting" -Details @{
        title = if ($storyObj) { $storyObj.title } else { "unknown" }
    }

    # Pre-flight: skip stories already committed in git
    # Guard: track auto-completed stories to prevent infinite loop if prd.json write fails
    if (-not $script:AutoCompletedStories) { $script:AutoCompletedStories = @{} }
    if ($storyObj -and -not $storyObj.passes) {
        if ($script:AutoCompletedStories.ContainsKey($StoryId)) {
            Write-Host "    Pre-flight: $StoryId already auto-completed this session - skipping" -ForegroundColor Yellow
            return $true
        }
        $alreadyDone = Test-StoryAlreadyCommitted -StoryId $StoryId -Story $storyObj
        if ($alreadyDone) {
            Write-Host "    Pre-flight: $StoryId already committed in git - auto-completing" -ForegroundColor Green
            Complete-StoryAutomatically -StoryId $StoryId -Story $storyObj -Reason "git-commit-detected"
            $script:AutoCompletedStories[$StoryId] = $true
            return $true
        }
    }

    # Story 3.1: Adaptive prompt builder (consolidates Stories 1.2, 1.3, 1.6, 2.5, 3.3)
    $prompt = Build-StoryPrompt -StoryId $StoryId -Story $storyObj -FocusArea $focusArea -RetryCount $script:State.CurrentRetryCount

    # Track errors for hard story detection
    if (-not $script:State.CurrentStoryErrors) {
        $script:State.CurrentStoryErrors = @()
    }

    # Invoke the common process handler
    $success = Invoke-ClaudeProcess -Prompt $prompt -PromptType "story_work" -Identifier $StoryId -FocusArea $focusArea -StoryObj $storyObj

    # Post-success validation: verify prd.json was actually updated
    # Prevents infinite loop when Claude exits 0 but doesn't set passes: true
    if ($success) {
        $freshPrd = Get-Sprint
        if ($freshPrd) {
            $freshStory = $freshPrd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
            if ($freshStory -and -not $freshStory.passes) {
                Write-Host "  [PHANTOM] Story $StoryId reported success but passes is still false in prd.json" -ForegroundColor Yellow

                # Track phantom successes per story
                if (-not $script:PhantomSuccesses) { $script:PhantomSuccesses = @{} }
                if (-not $script:PhantomSuccesses.ContainsKey($StoryId)) { $script:PhantomSuccesses[$StoryId] = 0 }
                $script:PhantomSuccesses[$StoryId]++

                $phantomCount = $script:PhantomSuccesses[$StoryId]
                Write-Host "  [PHANTOM] Phantom success count for $StoryId`: $phantomCount" -ForegroundColor Yellow

                # Treat as failure so hard-story detection kicks in
                $success = $false

                if ($phantomCount -ge 2) {
                    Write-Host "  [PHANTOM] $StoryId has $phantomCount phantom successes - marking as hard story" -ForegroundColor Red
                    Mark-AsHardStory `
                        -StoryId $StoryId `
                        -Errors @(@{ ErrorType = "phantom_success"; ErrorMessage = "Story reports success but passes remains false ($phantomCount times)"; Timestamp = (Get-Date).ToString("o") }) `
                        -Reason "phantom_success" `
                        -FocusArea $focusArea `
                        -StoryTitle $(if ($storyObj) { $storyObj.title } else { "Unknown" })

                    Update-StoryStatus -StoryId $StoryId -Passes $false -Notes "HARD STORY: Phantom success - exits 0 but never sets passes: true"

                    # Reset tracking
                    $script:PhantomSuccesses.Remove($StoryId)
                    $script:State.CurrentStoryErrors = @()
                    $script:State.CurrentStoryStartTime = $null
                    return $false
                }
            }
        }
    }

    # Check for hard story conditions
    if (-not $success) {
        # Track error
        $script:State.CurrentStoryErrors += @{
            ErrorType = "failure"
            ErrorMessage = "Story failed on attempt $($script:State.CurrentRetryCount)"
            Timestamp = (Get-Date).ToString("o")
        }

        # Check if we should mark as hard (3 failures or 30-min timeout)
        $maxRetries = 3
        if ($script:Config.autonomy -and $script:Config.autonomy.fastFail) {
            $maxRetries = $script:Config.autonomy.fastFail.consecutiveFailures
        }

        $shouldMarkHard = $false
        $reason = "3_failures"

        # Check 30-minute limit
        if ($script:State.CurrentStoryStartTime) {
            $elapsed = (Get-Date) - $script:State.CurrentStoryStartTime
            if ($elapsed.TotalMinutes -gt 30) {
                $shouldMarkHard = $true
                $reason = "30_minute_limit"
            }
        }

        # Check failure count
        if ($script:State.CurrentRetryCount -ge $maxRetries) {
            $shouldMarkHard = $true
        }

        if ($shouldMarkHard) {
            Write-Host "  [HARD STORY] $StoryId marked as hard ($reason)" -ForegroundColor Yellow
            Mark-AsHardStory `
                -StoryId $StoryId `
                -Errors $script:State.CurrentStoryErrors `
                -Reason $reason `
                -FocusArea $focusArea `
                -StoryTitle $(if ($storyObj) { $storyObj.title } else { "Unknown" })

            # Reset error tracking for next story
            $script:State.CurrentStoryErrors = @()
            $script:State.CurrentStoryStartTime = $null

            # Update PRD to mark story as skipped (special status)
            Update-StoryStatus -StoryId $StoryId -Passes $false -Notes "HARD STORY: Skipped after $reason - pending decomposition"
        }
    }
    else {
        # Success - reset error tracking
        $script:State.CurrentStoryErrors = @()
        $script:State.CurrentStoryStartTime = $null
    }

    return $success
}

# ============================================================================
# ABORT / ITERATION CONTROL
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

    if ($script:State.ConsecutiveFailures -ge $maxFailures) {
        Write-Host ""
        Write-Host "  FAST-FAIL: $($script:State.ConsecutiveFailures) consecutive failures" -ForegroundColor Red
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

    if ($script:State.IterationCount -ge $maxIterations) {
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
        $meta = Read-JsonFile -Path $signalFile -Silent
        $reason = if ($meta -and $meta.reason) { $meta.reason } else { "User requested" }
        $requestedAt = if ($meta -and $meta.requestedAt) { $meta.requestedAt } else { "Unknown" }
        Write-Host ""
        Write-Host "  =====================================================" -ForegroundColor Cyan
        Write-Host "     GRACEFUL STOP REQUESTED" -ForegroundColor Cyan
        Write-Host "  =====================================================" -ForegroundColor Cyan
        Write-Host "  Reason: $reason" -ForegroundColor DarkGray
        Write-Host "  Requested at: $requestedAt" -ForegroundColor DarkGray
        Write-Host ""
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

    $prd = Get-Sprint
    if (-not $prd) {
        return @{ passed = 0; failed = 0; total = 0; nextStory = $null; complete = $true }
    }

    # Ensure we're on the correct branch (once per sprint)
    $targetBranch = $prd.branchName
    if ($targetBranch -and $targetBranch -ne $script:State.LastEnsuredBranch) {
        Ensure-SprintBranch | Out-Null
        $script:State.LastEnsuredBranch = $targetBranch
    }

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

# ============================================================================
# MAIN LOOP
# ============================================================================

# ============================================================================
# ENTRY POINT
# ============================================================================

Write-RalphBanner -Queue:$Queue -TrueAuto:$TrueAuto -RalphsChoice:$RalphsChoice -RalphsChoiceAuto:$RalphsChoiceAuto

# Check for crash recovery
$crashRecovery = Test-CrashRecovery
if ($crashRecovery -and $Resume) {
    $recoveryResult = Invoke-CrashRecovery
    if ($recoveryResult.Resume) {
        Write-Host "  Resuming from crash recovery state..." -ForegroundColor Green
        # The resume logic in the loops will pick up from the current PRD state
    }
}
elseif ($crashRecovery -and -not $Resume) {
    Write-Host ""
    Write-Host "  [!] Previous session may have crashed" -ForegroundColor Yellow
    Write-Host "  Run with -Resume to continue from crash state" -ForegroundColor DarkGray
    Write-Host "  Crash state: Sprint $($crashRecovery.lastSprint), Story $($crashRecovery.lastStory)" -ForegroundColor DarkGray
    Write-Host ""
}

# Show diagnosis of last session (helps identify silent hangs)
Show-LastSessionDiagnosis

# Rotate session log if too large
Clear-SessionLog

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
    elseif ($Overnight) { "Overnight" }
    else { "Standard" }

# Log session start event
Append-SessionTimeline -Event "session_start" -Data @{
    mode = $sessionMode
    claudePath = $claudePath
    projectRoot = $script:ProjectRoot
}

# Write initial heartbeat and session log
Write-SessionLog -Event "loop_start" -Message "Ralph Loop starting in $sessionMode mode"
Write-Heartbeat -Phase "loop_starting" -Details @{ mode = $sessionMode }

# Route to appropriate loop based on flags
# Note: each mode function runs its own pre-flight at the right time (after PRD generation)
# Wrapped in try-finally to ensure session_end event is always written
# (session 2026-01-27_231506 crashed without writing session_end)
try {
    if ($Queue) {
        Start-InterviewQueueLoop
    }
    elseif ($TrueAuto) {
        Start-TrueAutoLoop -FocusArea $FocusArea
    }
    elseif ($RalphsChoice) {
        Start-RalphsChoiceLoop
    }
    elseif ($RalphsChoiceAuto) {
        Start-RalphsChoiceAutoLoop
    }
    elseif ($Overnight) {
        # Overnight mode: adaptive multi-focus with rotation
        $areasToUse = @()
        if ($FocusAreas -and $FocusAreas.Count -gt 0) {
            $areasToUse = $FocusAreas
        }
        elseif ($FocusArea) {
            $areasToUse = @($FocusArea)
        }
        Start-AdaptiveOvernightLoop -FocusAreas $areasToUse -MaxHours $MaxHours
    }
    else {
        Start-StandardLoop
    }
}
finally {
    # Session summary - always runs even if loop throws unhandled exception
    $duration = (Get-Date) - $script:State.SessionStartTime

    # Clear crash recovery file on successful completion
    Clear-CrashRecovery

    # Log session end event
    Append-SessionTimeline -Event "session_end" -Data @{
        iterations = $script:State.IterationCount
        durationMin = [math]::Round($duration.TotalMinutes, 1)
        consecutiveFailures = $script:State.ConsecutiveFailures
    }

    # Final heartbeat and session log
    Write-SessionLog -Event "loop_end" -Message "Ralph Loop ended after $($script:State.IterationCount) iterations"
    Write-Heartbeat -Phase "loop_ended" -Details @{
        iterations = $script:State.IterationCount
        durationMin = [math]::Round($duration.TotalMinutes, 1)
    }
    Write-Host ""
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Session Summary" -ForegroundColor Cyan
    Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  Session ID: $($script:State.SessionId)" -ForegroundColor DarkGray
    Write-Host "  Iterations: $($script:State.IterationCount)" -ForegroundColor DarkGray
    Write-Host "  Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor DarkGray
    Write-Host "  Logs: $script:SessionLogDir" -ForegroundColor DarkGray
    Write-Host ""
}
