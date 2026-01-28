# claude.ps1 - Claude process execution and result handling
# Extracted from ralph.ps1 Invoke-ClaudeProcess

function Get-PromptHash {
    <#
    .SYNOPSIS
        Compute a short MD5 hash of prompt content for tracking.
    .PARAMETER Content
        The prompt text to hash
    .RETURNS
        16-char hex string, or empty string if content is null/empty
    #>
    param([string]$Content)
    if (-not $Content) { return "" }
    $md5 = [System.Security.Cryptography.MD5]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Content)
    $hashBytes = $md5.ComputeHash($bytes)
    return ([BitConverter]::ToString($hashBytes) -replace '-', '').Substring(0, 16)
}

function Invoke-ClaudeSubprocess {
    <#
    .SYNOPSIS
        Spawn Claude CLI process with stdin pipe, async output capture, and
        activity-based timeout monitoring. Kills process on timeout.
    .PARAMETER ClaudePath
        Full path to claude executable
    .PARAMETER ClaudeArgs
        Arguments array for claude CLI
    .PARAMETER Prompt
        Prompt text to pipe via stdin
    .PARAMETER OutFile
        Path to write stdout log
    .PARAMETER ErrFile
        Path to write stderr log
    .RETURNS
        Hashtable: Exited, ExitCode, Output, ResourceSamples, ExecutionStart, ExecutionEnd, TimedOut
    #>
    param(
        [Parameter(Mandatory)][string]$ClaudePath,
        [Parameter(Mandatory)][string[]]$ClaudeArgs,
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$OutFile,
        [Parameter(Mandatory)][string]$ErrFile
    )

    # Get timeout from config
    $timeout = 600
    if ($script:Config.iterationTimeout) { $timeout = $script:Config.iterationTimeout }
    if ($script:Config.autonomy -and $script:Config.autonomy.iterationTimeout) {
        $timeout = $script:Config.autonomy.iterationTimeout
    }

    $flagsString = ($ClaudeArgs -join ' ')
    $executionStart = Get-Date

    # Create process with proper stdin redirection
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $ClaudePath
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

    $resourceSamples = @()

    try {
        $process.Start() | Out-Null
        $processId = $process.Id  # Cache before Dispose() in finally block
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()

        $process.StandardInput.Write($Prompt)
        $process.StandardInput.Close()

        # Activity-based timeout monitoring
        $checkIntervalSec = 5
        $timeSinceProgress = 0
        $totalElapsed = 0
        $maxTotalMinutes = 60
        $lastMinuteShown = -1

        $lastPrdTime = (Get-Item $script:PrdFile -ErrorAction SilentlyContinue).LastWriteTime
        $lastProgressTime = (Get-Item $script:ProgressFile -ErrorAction SilentlyContinue).LastWriteTime
        $lastGitStatus = (git status --porcelain 2>$null | Measure-Object -Line).Lines
        $lastBufferLength = $outBuilder.Length

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
                $currentBufferLength = $outBuilder.Length

                $prdUpdated = $currentPrdTime -and $lastPrdTime -and ($currentPrdTime -gt $lastPrdTime)
                $progressUpdated = $currentProgressTime -and $lastProgressTime -and ($currentProgressTime -gt $lastProgressTime)
                $gitChanged = $currentGitStatus -ne $lastGitStatus
                $bufferGrowing = $currentBufferLength -gt $lastBufferLength

                if ($prdUpdated -or $progressUpdated -or $gitChanged -or $bufferGrowing) {
                    $reason = if ($prdUpdated) { "prd.json" } elseif ($progressUpdated) { "progress.txt" } elseif ($gitChanged) { "git changes" } else { "claude output" }
                    Write-Host "  [$mins min] Activity detected ($reason)" -ForegroundColor DarkGreen
                    $timeSinceProgress = 0
                    $lastPrdTime = $currentPrdTime
                    $lastProgressTime = $currentProgressTime
                    $lastGitStatus = $currentGitStatus
                    $lastBufferLength = $currentBufferLength
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
            # Do NOT call parameterless WaitForExit() — it deadlocks on .NET Framework
            # when child processes hold stdout/stderr pipe handles open.
            # HasExited is already true, so just cancel async readers and read exit code.
            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
            Start-Sleep -Milliseconds 200
            $exitCode = $process.ExitCode
        }
    }
    finally {
        Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
        Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
        Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 100
        try {
            $outBuilder.ToString() | Set-Content $OutFile -ErrorAction Stop
        } catch {
            Write-Host "  Warning: Failed to write Claude output to $OutFile : $_" -ForegroundColor Yellow
        }
        try {
            $errBuilder.ToString() | Set-Content $ErrFile -ErrorAction Stop
        } catch {
            Write-Host "  Warning: Failed to write Claude stderr to $ErrFile : $_" -ForegroundColor Yellow
        }
        # Ensure process is terminated on any exit path (timeout, Ctrl+C, error)
        if ($process -and -not $process.HasExited) {
            Write-Host "  Terminating Claude process..." -ForegroundColor Yellow
            # Kill entire process tree to avoid orphaned node subagents
            $treePid = $process.Id
            try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
            # Fallback if taskkill didn't work
            if (-not $process.HasExited) {
                try { $process.Kill() } catch {}
            }
        }
        if ($process) { $process.Dispose() }
    }

    $timedOut = -not $exited
    if ($timedOut) {
        Write-Host "  Timeout after $timeout seconds" -ForegroundColor Yellow
        # Process already killed in finally block
    }

    return @{
        Exited         = $exited
        ExitCode       = $exitCode
        Output         = $outBuilder.ToString() + $errBuilder.ToString()
        ResourceSamples = $resourceSamples
        ExecutionStart = $executionStart
        ExecutionEnd   = $executionEnd
        TimedOut       = $timedOut
        Timeout        = $timeout
        ProcessId      = $processId
    }
}

function Record-IterationLog {
    <#
    .SYNOPSIS
        Write comprehensive iteration logs: invocation, manifest, file ops,
        git ops, timeline, test details, resource usage, prompt effectiveness.
    .PARAMETER Ctx
        Hashtable with keys: Iteration, ClaudePath, ClaudeArgs, PromptFile,
        PromptType, ProcessId, ExecutionStart, ExecutionEnd, ExitCode, TimedOut,
        StoryId, FocusAreaId, IterationStatus, IterationStart, GitStateBefore,
        GitStateAfter, FileOps, Commits, TestResults, TokensUsed,
        IterationDuration, Success, ClaudeOutput, ResourceSamples
    #>
    param([Parameter(Mandatory)][hashtable]$Ctx)

    Log-ClaudeInvocation `
        -Iteration $Ctx.Iteration `
        -ClaudePath $Ctx.ClaudePath `
        -Arguments $Ctx.ClaudeArgs `
        -PromptFile $Ctx.PromptFile `
        -PromptType $Ctx.PromptType `
        -ProcessId $Ctx.ProcessId `
        -StartTime $Ctx.ExecutionStart `
        -EndTime $Ctx.ExecutionEnd `
        -ExitCode $(if ($null -ne $Ctx.ExitCode) { $Ctx.ExitCode } else { -1 }) `
        -TimedOut $Ctx.TimedOut

    Log-IterationManifest `
        -Iteration $Ctx.Iteration `
        -StoryId $Ctx.StoryId `
        -FocusArea $Ctx.FocusAreaId `
        -Status $Ctx.IterationStatus `
        -StartTime $Ctx.IterationStart `
        -EndTime (Get-Date) `
        -PromptFile $Ctx.PromptFile `
        -GitBefore $Ctx.GitStateBefore `
        -GitAfter $Ctx.GitStateAfter `
        -FileOps $Ctx.FileOps `
        -Commits $Ctx.Commits `
        -TestResults $Ctx.TestResults `
        -TokensEstimated $Ctx.TokensUsed `
        -RetryCount $script:State.CurrentRetryCount

    Log-FileOperations -Iteration $Ctx.Iteration -FileOps $Ctx.FileOps
    Log-GitOperations -Iteration $Ctx.Iteration -Branch $Ctx.GitStateAfter.branch -Commits $Ctx.Commits -BeforeState $Ctx.GitStateBefore -AfterState $Ctx.GitStateAfter

    $completeData = @{ iteration = $Ctx.Iteration; status = $Ctx.IterationStatus; success = $Ctx.Success; durationSec = [int]$Ctx.IterationDuration.TotalSeconds }
    if ($Ctx.StoryId) { $completeData.storyId = $Ctx.StoryId }
    Append-SessionTimeline -Event "iteration_complete" -Data $completeData

    # Phase 3 logging
    if ($Ctx.ClaudeOutput) { Log-TestDetails -Iteration $Ctx.Iteration -Output $Ctx.ClaudeOutput }
    if ($Ctx.ResourceSamples.Count -gt 0) { Log-ResourceUsage -Iteration $Ctx.Iteration -ProcessId $Ctx.ProcessId -Samples $Ctx.ResourceSamples }

    $effectiveness = Get-PromptEffectiveness -Success $Ctx.Success -RetryCount $script:State.CurrentRetryCount
    $promptContent = Get-Content $Ctx.PromptFile -Raw -ErrorAction SilentlyContinue
    $promptHashShort = Get-PromptHash -Content $promptContent
    Log-PromptEffectiveness -Iteration $Ctx.Iteration -PromptType $Ctx.PromptType -Effectiveness $effectiveness -PromptHash $promptHashShort
}

function Resolve-ClaudeResult {
    <#
    .SYNOPSIS
        Route on Claude process exit status (timeout/success/failure).
        Handles quality checks, regression detection, code review,
        learning DB updates, and metric recording.
    .PARAMETER SubResult
        Hashtable from Invoke-ClaudeSubprocess (TimedOut, ExitCode, Output, Timeout)
    .PARAMETER Ctx
        Hashtable with keys: TransitionContext, IsStoryWork, StoryId, FocusAreaId,
        Identifier, StoryObj, IterationDuration, TokensUsed, TestResults,
        PhaseTimings, GitStateBefore, FileOps, Commits, ClaudeOutput
    .RETURNS
        Hashtable: Success (bool), IterationStatus (string)
    #>
    param(
        [Parameter(Mandatory)][hashtable]$SubResult,
        [Parameter(Mandatory)][hashtable]$Ctx
    )

    $iterationStatus = "completed"
    $success = $false

    if ($SubResult.TimedOut) {
        # === TIMEOUT ===
        $iterationStatus = "timeout"
        $errorCategory = Get-ErrorCategory -Output $Ctx.ClaudeOutput -TimedOut $true

        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Timeout after $($SubResult.Timeout)s" -Iteration $script:State.IterationCount
        Log-StateTransition -From "running" -To "failed" -Reason "Timeout after $($SubResult.Timeout)s" -Context $Ctx.TransitionContext

        Record-Metric -StoryId $Ctx.Identifier -Mode $script:State.CurrentMode `
            -DurationMin ([math]::Round($Ctx.IterationDuration.TotalMinutes, 0)) `
            -Success $false -Timeout $true -TokensUsed $Ctx.TokensUsed `
            -ErrorCategory $errorCategory -TestResults $Ctx.TestResults `
            -RetryCount $script:State.CurrentRetryCount `
            -LinesAdded 0 -LinesDeleted 0 `
            -PhaseReadMs $Ctx.PhaseTimings.read_ms -PhaseAnalyzeMs $Ctx.PhaseTimings.analyze_ms `
            -PhaseImplementMs $Ctx.PhaseTimings.implement_ms -PhaseTestMs $Ctx.PhaseTimings.test_ms `
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms

        if ($Ctx.StoryObj) { Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }
        $script:State.ConsecutiveFailures++
    }
    elseif ($SubResult.ExitCode -eq 0) {
        # === SUCCESS ===
        $successMsg = if ($Ctx.IsStoryWork) { "Story completed successfully" } else { "Iteration completed successfully" }
        Write-Host "  $successMsg" -ForegroundColor Green
        $iterationStatus = "completed"
        $success = $true

        Log-StateTransition -From "running" -To "completed" -Reason "Success" -Context $Ctx.TransitionContext

        $gitStats = Get-GitDiffStats

        # Diff quality scoring
        $diffOutput = git diff HEAD~1 2>$null
        $diffQuality = Get-DiffQualityScore -DiffOutput $diffOutput
        if ($diffQuality.warnings.Count -gt 0) {
            foreach ($warn in $diffQuality.warnings) {
                Write-Host "  Quality: $warn" -ForegroundColor Yellow
            }
        }

        # Test regression detection
        $regressionResult = $null
        if ($Ctx.TestResults) {
            $regressionResult = Compare-TestBaseline -CurrentResults $Ctx.TestResults
        }

        # Auto-rollback on regression
        if ($regressionResult -and $regressionResult.hasRegression -and $Ctx.StoryId) {
            $rollbackCheck = Test-CanRollback -StoryId $Ctx.StoryId
            if ($rollbackCheck.canRollback) {
                Write-Host "  Auto-rollback: Reverting regression in $($Ctx.StoryId)" -ForegroundColor Yellow
                $rollbackOk = Invoke-StoryRollback -StoryId $Ctx.StoryId -Reason "Test regression detected"
                if ($rollbackOk) {
                    $success = $false
                    $iterationStatus = "rolled_back"
                    Write-Host "  Rollback complete. Story will retry." -ForegroundColor Yellow
                }
            }
        }

        Record-Metric -StoryId $Ctx.Identifier -Mode $script:State.CurrentMode `
            -DurationMin ([math]::Round($Ctx.IterationDuration.TotalMinutes, 0)) `
            -Success $true -Timeout $false -TokensUsed $Ctx.TokensUsed `
            -ErrorCategory "" -TestResults $Ctx.TestResults `
            -RetryCount $script:State.CurrentRetryCount `
            -LinesAdded $gitStats.Added -LinesDeleted $gitStats.Deleted `
            -PhaseReadMs $Ctx.PhaseTimings.read_ms -PhaseAnalyzeMs $Ctx.PhaseTimings.analyze_ms `
            -PhaseImplementMs $Ctx.PhaseTimings.implement_ms -PhaseTestMs $Ctx.PhaseTimings.test_ms `
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms

        if ($Ctx.StoryObj) {
            Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $true -ClaudeOutput $Ctx.ClaudeOutput -DiffOutput $diffOutput
            Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $true }

            # Independent code review
            if ($Ctx.IsStoryWork -and $diffOutput) {
                $reviewResult = Invoke-CodeReview -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -DiffOutput $diffOutput -FileOps $Ctx.FileOps -ClaudeOutput $Ctx.ClaudeOutput
                if ($reviewResult) {
                    Append-SessionTimeline -Event "code_review" -Data @{
                        storyId = $Ctx.StoryId
                        testRatio = $diffQuality.testRatio
                        churnRisk = $diffQuality.churnRisk
                        reviewScore = $reviewResult.score
                        reviewPassed = $reviewResult.passed
                        issueCount = $reviewResult.issues.Count
                    }

                    # File conflict detection
                    $conflictResult = Test-FileConflict -StoryId $Ctx.StoryId -Story $Ctx.StoryObj
                    if ($conflictResult -and $conflictResult.hasConflict) {
                        Append-SessionTimeline -Event "file_conflict" -Data @{
                            storyId = $Ctx.StoryId
                            overlappingFiles = $conflictResult.overlappingFiles
                            overlappingStories = $conflictResult.overlappingStories
                        }
                    }
                }
            }

            # Update baseline after successful story
            if ($Ctx.TestResults) {
                Update-TestBaseline -TestResults $Ctx.TestResults
            }

            # Token budget check
            Get-SprintTokenBudget | Out-Null

            # Save story progress
            try {
                Save-StoryProgress -StoryId $Ctx.StoryId -Milestone "completed" -Data @{
                    iteration = $script:State.IterationCount
                    retryCount = $script:State.CurrentRetryCount
                    tokensUsed = $Ctx.TokensUsed
                }
            } catch {}

            # Update learning database
            try {
                Update-LearningDb -Entry @{
                    type = "story_success"
                    storyId = $Ctx.StoryId
                    focusArea = $Ctx.FocusAreaId
                    retryCount = $script:State.CurrentRetryCount
                    tokensUsed = $Ctx.TokensUsed
                    linesAdded = $gitStats.Added
                    linesDeleted = $gitStats.Deleted
                    testRatio = if ($diffQuality) { $diffQuality.testRatio } else { 0 }
                    reviewScore = if ($reviewResult) { $reviewResult.score } else { $null }
                }
            } catch {}
        }
        $script:State.ConsecutiveFailures = 0
    }
    else {
        # === FAILURE ===
        $exitCodeStr = if ($null -ne $SubResult.ExitCode) { $SubResult.ExitCode } else { "unknown" }
        $failMsg = if ($Ctx.IsStoryWork) { "Story failed with exit code $exitCodeStr" } else { "Iteration failed with exit code $exitCodeStr" }
        Write-Host "  $failMsg" -ForegroundColor Red
        $iterationStatus = "failed"
        $errorCategory = Get-ErrorCategory -Output $Ctx.ClaudeOutput -TimedOut $false

        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Exit code: $exitCodeStr" -Iteration $script:State.IterationCount
        Log-StateTransition -From "running" -To "failed" -Reason "Exit code: $exitCodeStr" -Context $Ctx.TransitionContext

        Record-Metric -StoryId $Ctx.Identifier -Mode $script:State.CurrentMode `
            -DurationMin ([math]::Round($Ctx.IterationDuration.TotalMinutes, 0)) `
            -Success $false -Timeout $false -TokensUsed $Ctx.TokensUsed `
            -ErrorCategory $errorCategory -TestResults $Ctx.TestResults `
            -RetryCount $script:State.CurrentRetryCount `
            -LinesAdded 0 -LinesDeleted 0 `
            -PhaseReadMs $Ctx.PhaseTimings.read_ms -PhaseAnalyzeMs $Ctx.PhaseTimings.analyze_ms `
            -PhaseImplementMs $Ctx.PhaseTimings.implement_ms -PhaseTestMs $Ctx.PhaseTimings.test_ms `
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms

        if ($Ctx.StoryObj) { Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }

        # Update learning database on failure
        if ($Ctx.StoryId) {
            try {
                Update-LearningDb -Entry @{
                    type = "story_failure"
                    storyId = $Ctx.StoryId
                    focusArea = $Ctx.FocusAreaId
                    errorCategory = $errorCategory
                    retryCount = $script:State.CurrentRetryCount
                    exitCode = $exitCodeStr
                }
            } catch {}
        }

        $script:State.ConsecutiveFailures++
    }

    # === POST-ITERATION HEALING ===
    # After any story iteration, run tiered health check and heal if needed
    if ($Ctx.IsStoryWork) {
        $changedFiles = @()
        if ($Ctx.FileOps) {
            $changedFiles += @($Ctx.FileOps.filesCreated | ForEach-Object { $_.path })
            $changedFiles += @($Ctx.FileOps.filesModified | ForEach-Object { $_.path })
            $changedFiles = @($changedFiles | Where-Object { $_ })
        }

        $healResult = Invoke-PostIterationHealing `
            -StoryId $Ctx.StoryId `
            -FocusArea $Ctx.FocusAreaId `
            -ChangedFiles $changedFiles `
            -IterationSuccess $success

        if ($healResult.HealingNeeded -and -not $healResult.HealingSuccess) {
            $success = $false
            $iterationStatus = "healing_failed"
            Write-Host "  Sprint aborted: Tier $($healResult.FailedTier) errors could not be healed" -ForegroundColor Red
        }
    }

    return @{
        Success         = $success
        IterationStatus = $iterationStatus
    }
}
