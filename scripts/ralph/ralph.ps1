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

# Load domain modules
$script:LibPath = Join-Path $PSScriptRoot 'lib'
. "$script:LibPath\sprint.ps1"
. "$script:LibPath\scoring.ps1"
. "$script:LibPath\queue.ps1"
. "$script:LibPath\metrics.ps1"
. "$script:LibPath\quality.ps1"
. "$script:LibPath\prompts.ps1"

# ============================================================================
# CONFIG LOADING
# ============================================================================

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
    $result = Invoke-ClaudeProcess -Prompt $prompt -PromptType $promptType -Identifier $FocusAreaId -AllowedTools:$useTools

    # After PRD generation, update queue context from the new PRD
    if ($GeneratePRD -and $result) {
        Update-ContextFromPRD
    }

    return $result
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
        $contextRefreshed = $false
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

                # After first story (US-001 generates full PRD), update context
                if (-not $contextRefreshed) {
                    Update-ContextFromPRD
                    $contextRefreshed = $true
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
    # Read queue directly (inline) for reliability - Get-NextQueuedFocusArea has been unreliable
    $queuedArea = $null
    if (Test-Path $script:QueueFile) {
        try {
            $inlineQueue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            if ($inlineQueue.focusAreas) {
                $inlineIncomplete = @($inlineQueue.focusAreas | Where-Object { -not $_.completed })
                Write-Host "  Queue: $($inlineQueue.focusAreas.Count) areas, $($inlineIncomplete.Count) incomplete" -ForegroundColor DarkGray
                if ($inlineIncomplete.Count -gt 0) {
                    $queuedArea = $inlineIncomplete[0].id
                }
            }
        } catch {
            Write-Host "  Queue parse error: $_" -ForegroundColor Red
        }
    }
    Write-Host "  Queue next area: '$queuedArea'" -ForegroundColor DarkGray
    if ($queuedArea) {
        # Check current PRD focus area
        $currentPrdFocus = ""
        if (Test-Path $script:PrdFile) {
            try {
                $currentPrd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
                $currentPrdFocus = if ($currentPrd.focusArea) { $currentPrd.focusArea } else { "" }
            } catch {}
        }
        Write-Host "  Current PRD focus: '$currentPrdFocus', Queue next: '$queuedArea'" -ForegroundColor DarkGray

        # Check if this is a fresh queue (no completed areas) - reuse already-parsed data
        $completedCount = @($inlineQueue.focusAreas | Where-Object { $_.completed }).Count
        $isFreshQueue = ($completedCount -eq 0)

        # Use Test-ShouldGenerateNewPRD for decision
        $decision = Test-ShouldGenerateNewPRD -NewFocusArea $queuedArea
        Write-Host "  PRD decision: ShouldGenerate=$($decision.ShouldGenerate), Reason='$($decision.Reason)'" -ForegroundColor DarkGray

        # Fresh queue always generates new PRD (overrides continue decision)
        if ($isFreshQueue -and -not $decision.ShouldGenerate) {
            $decision = @{ ShouldGenerate = $true; Reason = "Fresh queue detected - generating new sprint" }
        }

        # Force generate if queue area differs from PRD focus area and sprint is done
        if (-not $decision.ShouldGenerate -and $queuedArea -ne $currentPrdFocus) {
            $decision = @{ ShouldGenerate = $true; Reason = "Queue area '$queuedArea' differs from PRD '$currentPrdFocus' - generating new sprint" }
            Write-Host "  Forced PRD generation: area mismatch" -ForegroundColor Yellow
        }

        if ($decision.ShouldGenerate) {
            Write-Host "  $($decision.Reason)" -ForegroundColor Yellow
            Write-Host "  Generating new sprint PRD for: $queuedArea..." -ForegroundColor Yellow
            Write-Host ""

            # Read context inline (using already-parsed queue data)
            $context = ""
            if ($inlineQueue -and $inlineQueue.interviewContext) { $context = $inlineQueue.interviewContext }
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
    } else {
        Write-Host "  No queued focus areas found" -ForegroundColor DarkGray
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

            # Mark current area as complete BEFORE checking for next
            if ($status.focusArea) {
                Update-LegacyQueueProgress -CompletedArea $status.focusArea
            }

            # Check for pending queue items (inline for reliability)
            $nextArea = $null
            if (Test-Path $script:QueueFile) {
                try {
                    $loopQueue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
                    if ($loopQueue.focusAreas) {
                        $loopIncomplete = @($loopQueue.focusAreas | Where-Object { -not $_.completed })
                        if ($loopIncomplete.Count -gt 0) {
                            $nextArea = $loopIncomplete[0].id
                        }
                    }
                } catch {}
            }
            Write-Host "  Next queued area: '$nextArea'" -ForegroundColor DarkGray
            if ($nextArea) {
                Write-Host "  Queue has more focus areas. Next: $nextArea" -ForegroundColor Cyan
                Write-Host ""

                # Use Test-ShouldGenerateNewPRD for decision
                $decision = Test-ShouldGenerateNewPRD -NewFocusArea $nextArea
                Write-Host "  $($decision.Reason)" -ForegroundColor DarkGray

                # Generate new PRD for next focus area
                Write-Host "  Generating PRD for focus area: $nextArea..." -ForegroundColor Yellow
                # Read context inline for reliability
                $context = ""
                if ($loopQueue -and $loopQueue.interviewContext) { $context = $loopQueue.interviewContext }
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
                # Update interview progress (queue already marked above)
                if ($status.focusArea) {
                    Update-InterviewProgress -AreaId $status.focusArea
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
