# Ralph Loop - Main Execution Script
# Usage: .\scripts\ralph\ralph.ps1 [-Queue] [-SkipPlanApproval] [-TrueAuto] [-Resume] [-FocusArea <area>]
#
# Modes:
#   -Queue             Process focus areas from queue.json (interview mode)
#   -SkipPlanApproval  Skip plan approval prompts
#   -TrueAuto          Continuous improvement mode (no exit on sprint complete)
#   -Resume            Resume previous sprint instead of starting new
#   -FocusArea <area>  Override focus area for this session

param(
    [switch]$Queue,
    [switch]$SkipPlanApproval,
    [switch]$TrueAuto,
    [switch]$Resume,
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

# Ensure logs directory exists
if (-not (Test-Path $script:LogDir)) {
    New-Item -ItemType Directory -Path $script:LogDir -Force | Out-Null
}

# Create session log directory
$script:SessionLogDir = Join-Path $script:LogDir $script:SessionId
New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null

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

function Get-InterviewFocusAreas {
    <#
    .SYNOPSIS
        Read incomplete focus areas from queue.json
    .RETURNS
        Array of incomplete focus area objects, or empty array if none
    #>

    if (-not (Test-Path $script:QueueFile)) {
        return @()
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Check for focusAreas array (interview mode format)
        if ($queue.focusAreas) {
            $incomplete = @($queue.focusAreas | Where-Object { -not $_.completed })
            return $incomplete
        }

        # Legacy format: queue array with completedAreas
        if ($queue.queue -and $queue.completedAreas) {
            $incomplete = @()
            foreach ($area in $queue.queue) {
                if ($queue.completedAreas -notcontains $area) {
                    $incomplete += @{ id = $area; completed = $false }
                }
            }
            return $incomplete
        }

        return @()
    }
    catch {
        Write-Host "  Warning: Could not parse queue.json" -ForegroundColor Yellow
        return @()
    }
}

function Update-InterviewProgress {
    <#
    .SYNOPSIS
        Mark a focus area as completed in queue.json
    .PARAMETER AreaId
        The focus area ID to mark as completed
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$AreaId
    )

    if (-not (Test-Path $script:QueueFile)) {
        return
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Update focusAreas format
        if ($queue.focusAreas) {
            foreach ($area in $queue.focusAreas) {
                if ($area.id -eq $AreaId) {
                    $area.completed = $true
                    $area.completedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                }
            }
        }

        # Update legacy format
        if ($queue.completedAreas -and $queue.completedAreas -notcontains $AreaId) {
            $queue.completedAreas += $AreaId
        }

        # Update session info
        if ($queue.session) {
            $queue.session.lastActivityAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
            $queue.session.iterationCount = $script:IterationCount
        }

        $queue | ConvertTo-Json -Depth 10 | Set-Content -Path $script:QueueFile -Encoding UTF8

        Write-Host "  Marked '$AreaId' as completed" -ForegroundColor Green
    }
    catch {
        Write-Host "  Warning: Could not update queue.json" -ForegroundColor Yellow
    }
}

function Get-InterviewContext {
    <#
    .SYNOPSIS
        Get the interview context string from queue.json
    .RETURNS
        Interview context string, or empty string if not found
    #>

    if (-not (Test-Path $script:QueueFile)) {
        return ""
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
        if ($queue.interviewContext) {
            return $queue.interviewContext
        }
        return ""
    }
    catch {
        return ""
    }
}

function Get-NextQueuedFocusArea {
    <#
    .SYNOPSIS
        Get the next incomplete focus area from the legacy queue
    .RETURNS
        Focus area ID string, or $null if none remaining
    #>

    if (-not (Test-Path $script:QueueFile)) {
        return $null
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Legacy format: queue array + completedAreas
        if ($queue.queue -and $queue.completedAreas) {
            foreach ($area in $queue.queue) {
                if ($queue.completedAreas -notcontains $area) {
                    return $area
                }
            }
        }

        # Interview format: focusAreas with completed flag
        if ($queue.focusAreas) {
            $next = $queue.focusAreas | Where-Object { -not $_.completed } | Select-Object -First 1
            if ($next) {
                return $next.id
            }
        }
    }
    catch {
        return $null
    }

    return $null
}

function Update-LegacyQueueProgress {
    <#
    .SYNOPSIS
        Mark a focus area as completed in the legacy queue format
    .PARAMETER CompletedArea
        The focus area ID to mark as completed
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$CompletedArea
    )

    if (-not (Test-Path $script:QueueFile)) {
        return
    }

    try {
        $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json

        # Legacy format: add to completedAreas if not already present
        if ($queue.completedAreas -and $queue.completedAreas -notcontains $CompletedArea) {
            $queue.completedAreas += $CompletedArea
        }

        # Update currentIndex if queue array exists
        if ($queue.queue) {
            $idx = [array]::IndexOf($queue.queue, $CompletedArea)
            if ($idx -ge 0) {
                $queue.currentIndex = $idx + 1
            }
        }

        # Also update interview format if present
        if ($queue.focusAreas) {
            foreach ($area in $queue.focusAreas) {
                if ($area.id -eq $CompletedArea) {
                    $area.completed = $true
                    $area.completedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
                }
            }
        }

        $queue | ConvertTo-Json -Depth 10 | Set-Content -Path $script:QueueFile -Encoding UTF8
        Write-Host "  Updated queue progress: $CompletedArea completed" -ForegroundColor DarkGray
    }
    catch {
        Write-Host "  Warning: Could not update queue progress" -ForegroundColor Yellow
    }
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
# CLAUDE INVOCATION
# ============================================================================

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

    # Track retries per focus area
    if ($FocusAreaId -ne $script:LastFocusAreaId) {
        $script:CurrentRetryCount = 0
        $script:LastFocusAreaId = $FocusAreaId
    }
    $script:CurrentRetryCount++

    $script:IterationCount++
    $iterationStart = Get-Date

    Write-IterationBanner -Iteration $script:IterationCount -FocusArea $FocusAreaId

    # Build the prompt
    if ($GeneratePRD) {
        # Generate a new PRD for this focus area
        $prompt = @"
You are generating a new sprint PRD for focus area: $FocusAreaId

INSTRUCTIONS:
1. Read scripts/ralph/ralph-config.json to understand the focus area
2. Read scripts/ralph/prompt.md for context about the project
3. Read CLAUDE.md for project conventions
4. Analyze the codebase to find improvement opportunities for '$FocusAreaId'
5. Update scripts/ralph/prd.json with:
   - focusArea: "$FocusAreaId"
   - sprintNumber: increment from current
   - branchName: "ralph/sprint-N" (matching sprintNumber)
   - 8-12 specific user stories with:
     - Clear acceptance criteria (4-6 items each)
     - passes: false for all stories
     - Action verbs in titles (Add, Create, Update, Fix, etc.)

$(if ($Context) { "Context from user: $Context" } else { "" })

Start by reading the config and prompt files, then generate the PRD.
"@
    }
    else {
        # Work on existing stories
        $prompt = "Focus on: $FocusAreaId`n`n"
        if ($Context) {
            $prompt += "Context: $Context`n`n"
        }
        $prompt += "Read scripts/ralph/prompt.md for instructions. Work on ONE user story from prd.json that aligns with the focus area. If no stories exist for this focus area, generate appropriate stories first."
    }

    # Log iteration start
    $iterationLog = Join-Path $script:SessionLogDir "iteration_$($script:IterationCount).log"
    "Iteration $($script:IterationCount) - Focus: $FocusAreaId - Started: $(Get-Date -Format 'HH:mm:ss')" | Out-File $iterationLog

    # Update progress file to show activity
    $progressEntry = "`n## Iteration $script:IterationCount - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n- Focus: $FocusAreaId`n- Status: In Progress...`n"
    Add-Content -Path $script:ProgressFile -Value $progressEntry

    # Get Claude path
    $claudePath = Get-ClaudePath

    # Build arguments
    $claudeArgs = @(
        "--print",
        "--dangerously-skip-permissions",
        $prompt
    )

    # Always allow tools for PRD generation (needs write access to prd.json)
    # Also allow if -SkipPlanApproval was passed
    if ($GeneratePRD -or $SkipPlanApproval) {
        $claudeArgs += "--allowedTools"
        $claudeArgs += "Bash,Read,Write,Edit,Glob,Grep,WebSearch"
    }

    Write-Host "  Invoking Claude..." -ForegroundColor Cyan
    Write-Host "  Prompt: Focus on $FocusAreaId" -ForegroundColor DarkGray

    try {
        # Get timeout from config
        $timeout = 600  # Default 10 minutes
        if ($script:Config.iterationTimeout) {
            $timeout = $script:Config.iterationTimeout
        }
        if ($script:Config.autonomy -and $script:Config.autonomy.iterationTimeout) {
            $timeout = $script:Config.autonomy.iterationTimeout
        }

        # Output file paths for reading after completion
        $outFile = Join-Path $script:SessionLogDir "claude_out_$($script:IterationCount).log"
        $errFile = Join-Path $script:SessionLogDir "claude_err_$($script:IterationCount).log"

        # Run Claude with timeout
        $process = Start-Process -FilePath $claudePath `
            -ArgumentList $claudeArgs `
            -WorkingDirectory $script:ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput $outFile `
            -RedirectStandardError $errFile

        $exited = $process.WaitForExit($timeout * 1000)

        $iterationDuration = (Get-Date) - $iterationStart

        # Read output for metrics
        $claudeOutput = ""
        if (Test-Path $outFile) {
            $claudeOutput = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
        }
        if (Test-Path $errFile) {
            $claudeOutput += Get-Content $errFile -Raw -ErrorAction SilentlyContinue
        }

        # Calculate metrics
        $tokensUsed = Get-EstimatedTokens -Output $claudeOutput
        $testResults = Get-TestResults -Output $claudeOutput

        if (-not $exited) {
            Write-Host "  Timeout after $timeout seconds" -ForegroundColor Yellow
            $process.Kill()

            # Log timeout
            "Timeout after $timeout seconds" | Add-Content $iterationLog
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $true
            Record-Metric -StoryId $FocusAreaId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $true -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0

            $script:ConsecutiveFailures++
            return $false
        }

        if ($process.ExitCode -eq 0) {
            Write-Host "  Iteration completed successfully" -ForegroundColor Green
            "Completed successfully in $([math]::Round($iterationDuration.TotalSeconds)) seconds" | Add-Content $iterationLog

            # Capture git diff stats on success
            $gitStats = Get-GitDiffStats
            Record-Metric -StoryId $FocusAreaId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $true -Timeout $false -TokensUsed $tokensUsed -ErrorCategory "" -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded $gitStats.Added -LinesDeleted $gitStats.Deleted

            $script:ConsecutiveFailures = 0
            return $true
        }
        else {
            Write-Host "  Iteration failed with exit code $($process.ExitCode)" -ForegroundColor Red
            "Failed with exit code $($process.ExitCode)" | Add-Content $iterationLog
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $false
            Record-Metric -StoryId $FocusAreaId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $false -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0

            $script:ConsecutiveFailures++
            return $false
        }
    }
    catch {
        Write-Host "  Error invoking Claude: $_" -ForegroundColor Red
        "Error: $_" | Add-Content $iterationLog
        $errorCategory = Get-ErrorCategory -Output $_.ToString() -TimedOut $false
        Record-Metric -StoryId $FocusAreaId -Mode $script:CurrentMode -DurationMin 0 -Success $false -Timeout $false -TokensUsed 0 -ErrorCategory $errorCategory -TestResults "" -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0

        $script:ConsecutiveFailures++
        return $false
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

    # Track retries per story
    if ($StoryId -ne $script:LastStoryId) {
        $script:CurrentRetryCount = 0
        $script:LastStoryId = $StoryId
    }
    $script:CurrentRetryCount++

    $script:IterationCount++
    $iterationStart = Get-Date

    # Get focus area from PRD
    $focusArea = ""
    if (Test-Path $script:PrdFile) {
        try {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $focusArea = $prd.focusArea
        }
        catch {}
    }

    Write-IterationBanner -Iteration $script:IterationCount -FocusArea $focusArea -StoryId $StoryId

    # Build the prompt
    $prompt = "Work on story $StoryId from scripts/ralph/prd.json. Read scripts/ralph/prompt.md for instructions."

    # Get Claude path
    $claudePath = Get-ClaudePath

    # Build arguments
    $claudeArgs = @(
        "--print",
        "--dangerously-skip-permissions",
        $prompt
    )

    Write-Host "  Invoking Claude for $StoryId..." -ForegroundColor Cyan

    try {
        $timeout = 600
        if ($script:Config.iterationTimeout) {
            $timeout = $script:Config.iterationTimeout
        }

        # Output file paths for reading after completion
        $outFile = Join-Path $script:SessionLogDir "claude_out_$($script:IterationCount).log"
        $errFile = Join-Path $script:SessionLogDir "claude_err_$($script:IterationCount).log"

        $process = Start-Process -FilePath $claudePath `
            -ArgumentList $claudeArgs `
            -WorkingDirectory $script:ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput $outFile `
            -RedirectStandardError $errFile

        $exited = $process.WaitForExit($timeout * 1000)
        $iterationDuration = (Get-Date) - $iterationStart

        # Read output for metrics
        $claudeOutput = ""
        if (Test-Path $outFile) {
            $claudeOutput = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
        }
        if (Test-Path $errFile) {
            $claudeOutput += Get-Content $errFile -Raw -ErrorAction SilentlyContinue
        }

        # Calculate metrics
        $tokensUsed = Get-EstimatedTokens -Output $claudeOutput
        $testResults = Get-TestResults -Output $claudeOutput

        if (-not $exited) {
            Write-Host "  Timeout after $timeout seconds" -ForegroundColor Yellow
            $process.Kill()
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $true
            Record-Metric -StoryId $StoryId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $true -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0
            $script:ConsecutiveFailures++
            return $false
        }

        if ($process.ExitCode -eq 0) {
            Write-Host "  Story completed successfully" -ForegroundColor Green

            # Capture git diff stats on success
            $gitStats = Get-GitDiffStats
            Record-Metric -StoryId $StoryId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $true -Timeout $false -TokensUsed $tokensUsed -ErrorCategory "" -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded $gitStats.Added -LinesDeleted $gitStats.Deleted
            $script:ConsecutiveFailures = 0
            return $true
        }
        else {
            Write-Host "  Story failed with exit code $($process.ExitCode)" -ForegroundColor Red
            $errorCategory = Get-ErrorCategory -Output $claudeOutput -TimedOut $false
            Record-Metric -StoryId $StoryId -Mode $script:CurrentMode -DurationMin ([math]::Round($iterationDuration.TotalMinutes, 0)) -Success $false -Timeout $false -TokensUsed $tokensUsed -ErrorCategory $errorCategory -TestResults $testResults -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0
            $script:ConsecutiveFailures++
            return $false
        }
    }
    catch {
        Write-Host "  Error invoking Claude: $_" -ForegroundColor Red
        $errorCategory = Get-ErrorCategory -Output $_.ToString() -TimedOut $false
        Record-Metric -StoryId $StoryId -Mode $script:CurrentMode -DurationMin 0 -Success $false -Timeout $false -TokensUsed 0 -ErrorCategory $errorCategory -TestResults "" -RetryCount $script:CurrentRetryCount -LinesAdded 0 -LinesDeleted 0
        $script:ConsecutiveFailures++
        return $false
    }
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
        Estimate token count from output length
    .PARAMETER Output
        The output text to estimate tokens from
    .RETURNS
        Estimated token count (capped at 50000)
    #>
    param(
        [string]$Output
    )

    if (-not $Output -or $Output.Length -eq 0) {
        return 0
    }

    # Estimate tokens from output length (~4 chars per token)
    $estimatedTokens = [math]::Round($Output.Length / 4)
    # Cap at reasonable max
    return [math]::Min($estimatedTokens, 50000)
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
        [int]$LinesDeleted = 0
    )

    # Ensure metrics file exists with header
    if (-not (Test-Path $script:MetricsFile)) {
        "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted" | Set-Content $script:MetricsFile
    }

    # Use defaults from script variables if not provided
    if (-not $Session) { $Session = $script:SessionId }
    if (-not $Mode) { $Mode = $script:CurrentMode }
    if (-not $FocusArea) {
        if (Test-Path $script:PrdFile) {
            try {
                $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
                $FocusArea = $prd.focusArea
            }
            catch {}
        }
    }

    # Calculate hour_of_day if not provided
    if ($HourOfDay -eq -1) {
        $HourOfDay = (Get-Date).Hour
    }

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $row = "$timestamp,$Session,$Sprint,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),$($Timeout.ToString().ToLower()),$FocusArea,$TokensUsed,$ErrorCategory,$HourOfDay,$TestResults,$RetryCount,$LinesAdded,$LinesDeleted"
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
        $nextStory = $null

        foreach ($story in $prd.userStories) {
            if ($story.passes) {
                $passed++
            }
            else {
                $failed++
                if ($null -eq $nextStory) {
                    $nextStory = $story
                }
            }
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
        return
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

        # Run Claude for this focus area
        $success = Invoke-ClaudeForFocusArea -FocusAreaId $areaId -Context $context

        if ($success) {
            Update-InterviewProgress -AreaId $areaId
            $completedAreas += $area
        }

        # Check abort conditions
        if (Test-ShouldAbort) {
            break
        }

        if (Test-MaxIterations) {
            break
        }

        # Brief pause between focus areas
        Start-Sleep -Seconds 2
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

    while (-not (Test-MaxIterations)) {
        $status = Get-SprintStatus

        if ($status.complete) {
            Write-Host ""
            Write-Host "  Sprint complete!" -ForegroundColor Green
            Write-Host "  TrueAuto will generate new stories..." -ForegroundColor Magenta

            # Invoke Claude to generate new stories
            $prompt = "All stories in prd.json are complete. Generate 8-12 new user stories for continued improvement. Focus on: testing, performance, code quality. Update prd.json with new stories."

            $claudePath = Get-ClaudePath
            $claudeArgs = @("--print", "--dangerously-skip-permissions", $prompt)

            $process = Start-Process -FilePath $claudePath `
                -ArgumentList $claudeArgs `
                -WorkingDirectory $script:ProjectRoot `
                -NoNewWindow `
                -PassThru `
                -Wait

            Start-Sleep -Seconds 2
            continue
        }

        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
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
    while (-not (Test-MaxIterations)) {
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

            # Check for pending queue items (legacy format)
            $nextArea = Get-NextQueuedFocusArea
            if ($nextArea) {
                Write-Host "  Queue has more focus areas. Next: $nextArea" -ForegroundColor Cyan
                Write-Host ""

                # Update queue to mark current area as complete
                if ($status.focusArea) {
                    Update-LegacyQueueProgress -CompletedArea $status.focusArea
                }

                # Generate new PRD for next focus area
                Write-Host "  Generating PRD for focus area: $nextArea..." -ForegroundColor Yellow
                $context = Get-InterviewContext
                $prdGenerated = Invoke-ClaudeForFocusArea -FocusAreaId $nextArea -Context $context -GeneratePRD

                if ($prdGenerated) {
                    Write-Host "  PRD generated. Continuing with $nextArea" -ForegroundColor Green
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
        }
        else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 2
    }
}

# ============================================================================
# ENTRY POINT
# ============================================================================

Write-RalphBanner

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

# Route to appropriate loop based on flags
if ($Queue) {
    Start-InterviewQueueLoop
}
elseif ($TrueAuto) {
    Start-TrueAutoLoop
}
else {
    Start-StandardLoop
}

# Session summary
$duration = (Get-Date) - $script:SessionStartTime
Write-Host ""
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Session Summary" -ForegroundColor Cyan
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Session ID: $script:SessionId" -ForegroundColor DarkGray
Write-Host "  Iterations: $script:IterationCount" -ForegroundColor DarkGray
Write-Host "  Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor DarkGray
Write-Host "  Logs: $script:SessionLogDir" -ForegroundColor DarkGray
Write-Host ""
