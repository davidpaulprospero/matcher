# claude.ps1 - Claude process execution and result handling
# Extracted from ralph.ps1 Invoke-ClaudeProcess

# ============================================================================
# STALL DETECTION THRESHOLDS
# ============================================================================

function Get-StallThresholds {
    <#
    .SYNOPSIS
        Get stall detection thresholds for a focus area.
        Testing-related focus areas get extended timeouts since tests can take a while.
    .PARAMETER FocusArea
        The focus area ID (e.g., "testing", "unit-tests", "download")
    .RETURNS
        Hashtable with: stallThreshold, killThreshold, isExtended
    #>
    param([string]$FocusArea = "")

    # Default thresholds
    $defaults = @{
        stallThreshold = 180
        killThreshold = 360
    }

    # Check config for overrides
    if ($script:Config.stallDetection) {
        if ($script:Config.stallDetection.default) {
            $defaults.stallThreshold = $script:Config.stallDetection.default.stallThreshold
            $defaults.killThreshold = $script:Config.stallDetection.default.killThreshold
        }

        # Check for focus-area-specific override
        if ($FocusArea -and $script:Config.stallDetection.focusAreaOverrides) {
            $override = $script:Config.stallDetection.focusAreaOverrides.$FocusArea
            if ($override) {
                return @{
                    stallThreshold = $override.stallThreshold
                    killThreshold = $override.killThreshold
                    isExtended = $true
                    focusArea = $FocusArea
                }
            }
        }
    }

    return @{
        stallThreshold = $defaults.stallThreshold
        killThreshold = $defaults.killThreshold
        isExtended = $false
        focusArea = $FocusArea
    }
}

# ============================================================================
# API TIMEOUT RECOVERY (Phase 1 - P0)
# ============================================================================

function Log-APITimeoutRetry {
    <#
    .SYNOPSIS
        Log an API timeout retry attempt to api_timeouts.jsonl
    .PARAMETER StoryId
        The story being worked on
    .PARAMETER Attempt
        Retry attempt number
    .PARAMETER Backoff
        Backoff seconds before retry
    .PARAMETER ErrorType
        Type of error (timeout, exception)
    #>
    param(
        [string]$StoryId = "",
        [int]$Attempt = 1,
        [int]$Backoff = 30,
        [string]$ErrorType = "timeout"
    )

    $logFile = if ($script:Paths) { $script:Paths.ApiTimeoutsFile } else { Join-Path $script:RalphDir "session\api_timeouts.jsonl" }
    $entry = @{
        timestamp = (Get-Date).ToString("o")
        storyId = $StoryId
        attempt = $Attempt
        backoff = $Backoff
        errorType = $ErrorType
    }

    try {
        $entry | ConvertTo-Json -Compress | Out-File -FilePath $logFile -Append -Encoding UTF8
    }
    catch {
        Write-Host "  Warning: Could not log timeout: $_" -ForegroundColor Yellow
    }
}

function Get-RecentTimeouts {
    <#
    .SYNOPSIS
        Get count of recent API timeouts for circuit breaker
    .PARAMETER Minutes
        Time window in minutes (default: 10)
    .RETURNS
        Array of recent timeout entries
    #>
    param(
        [int]$Minutes = 10
    )

    $logFile = if ($script:Paths) { $script:Paths.ApiTimeoutsFile } else { Join-Path $script:RalphDir "session\api_timeouts.jsonl" }
    if (-not (Test-Path $logFile)) {
        return @()
    }

    $cutoff = (Get-Date).AddMinutes(-$Minutes)
    $recent = @()

    try {
        $lines = Get-Content $logFile -ErrorAction SilentlyContinue
        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $entry = $line | ConvertFrom-Json
                if ($entry.timestamp) {
                    $entryTime = [datetime]$entry.timestamp
                    if ($entryTime -gt $cutoff) {
                        $recent += $entry
                    }
                }
            }
            catch {}
        }
    }
    catch {}

    return $recent
}

function Test-APIHealth {
    <#
    .SYNOPSIS
        Circuit breaker: check if API appears to be down
    .DESCRIPTION
        If >10 timeouts in last 10 minutes, pause for 10 minutes
    .PARAMETER FocusAreaOnly
        If true, only pause this focus area (not global)
    .RETURNS
        $true if API is healthy, $false if circuit breaker tripped
    #>
    param(
        [switch]$FocusAreaOnly
    )

    $recentTimeouts = Get-RecentTimeouts -Minutes 10

    if ($recentTimeouts.Count -gt 10) {
        Write-Host ""
        Write-Host "  [CIRCUIT BREAKER] API appears to be experiencing issues" -ForegroundColor Red
        Write-Host "  $($recentTimeouts.Count) timeouts in last 10 minutes" -ForegroundColor Yellow
        Write-Host "  Pausing for 10 minutes before retry..." -ForegroundColor Yellow
        Write-Host ""

        # Log the circuit breaker trip
        Write-SessionLog -Event "circuit_breaker" -Message "API circuit breaker tripped" -Data @{
            recentTimeouts = $recentTimeouts.Count
            pauseMinutes = 10
        }

        # Show countdown
        for ($i = 600; $i -gt 0; $i -= 30) {
            $mins = [math]::Floor($i / 60)
            $secs = $i % 60
            Write-Host "  Resuming in ${mins}:$($secs.ToString('00'))..." -ForegroundColor DarkGray -NoNewline
            Start-Sleep -Seconds 30
            Write-Host "`r                                    `r" -NoNewline
        }

        Write-Host "  Circuit breaker released. Retrying..." -ForegroundColor Green
        return $false
    }

    return $true
}

function Invoke-ClaudeWithInfiniteRetry {
    <#
    .SYNOPSIS
        Invoke Claude subprocess with infinite retry on API timeout
    .DESCRIPTION
        Keeps retrying forever on API timeout (not just 3 times) with exponential backoff.
        Has a 30-minute max per story to prevent infinite loops.
        Properly handles non-timeout exceptions.
    .PARAMETER ClaudePath
        Path to Claude executable
    .PARAMETER ClaudeArgs
        Arguments for Claude CLI
    .PARAMETER Prompt
        Prompt to send
    .PARAMETER OutFile
        Output file path
    .PARAMETER ErrFile
        Error file path
    .PARAMETER StoryId
        Story ID for logging
    .PARAMETER StoryStartTime
        When this story started (for 30-min limit)
    .PARAMETER FocusArea
        Focus area ID (for stall threshold override - testing areas get longer timeouts)
    .RETURNS
        Hashtable with: TimedOut, Decompose, Reason, Attempts, and all Invoke-ClaudeSubprocess fields
    #>
    param(
        [Parameter(Mandatory)][string]$ClaudePath,
        [Parameter(Mandatory)][string[]]$ClaudeArgs,
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$OutFile,
        [Parameter(Mandatory)][string]$ErrFile,
        [string]$StoryId = "",
        [datetime]$StoryStartTime = (Get-Date),
        [string]$FocusArea = ""
    )

    $attempt = 0
    $backoffSeconds = 30
    $maxBackoff = 300  # Cap at 5 minutes
    $maxStoryMinutes = 30

    while ($true) {
        $attempt++
        $elapsed = (Get-Date) - $StoryStartTime

        # Check 30-minute story limit
        if ($elapsed.TotalMinutes -gt $maxStoryMinutes) {
            Write-Host "  [TIMEOUT] Story exceeded $maxStoryMinutes minutes - marking for decomposition" -ForegroundColor Red
            Write-SessionLog -Event "story_timeout" -Message "Story $StoryId exceeded 30-minute limit" -Data @{
                attempts = $attempt
                elapsedMinutes = [math]::Round($elapsed.TotalMinutes, 1)
            }
            return @{
                TimedOut = $true
                Decompose = $true
                Reason = "30_minute_limit"
                Attempts = $attempt
                Exited = $false
                ExitCode = $null
                Output = ""
            }
        }

        # Check circuit breaker before each attempt
        $apiHealthy = Test-APIHealth
        # Circuit breaker handles its own pause, so we continue regardless

        # Try to invoke Claude
        try {
            $result = Invoke-ClaudeSubprocess `
                -ClaudePath $ClaudePath `
                -ClaudeArgs $ClaudeArgs `
                -Prompt $Prompt `
                -OutFile $OutFile `
                -ErrFile $ErrFile `
                -FocusArea $FocusArea

            # Success or non-timeout failure
            if (-not $result.TimedOut) {
                $result.Attempts = $attempt
                $result.Decompose = $false
                return $result
            }

            # API timeout - log and retry
            Write-Host "  [RETRY] API timeout (attempt $attempt) - retrying in ${backoffSeconds}s..." -ForegroundColor Yellow

            Log-APITimeoutRetry -StoryId $StoryId -Attempt $attempt -Backoff $backoffSeconds -ErrorType "timeout"

            # Wait with progress indicator
            $remaining = $backoffSeconds
            while ($remaining -gt 0) {
                $waitChunk = [math]::Min($remaining, 10)
                Write-Host "    Waiting ${remaining}s..." -ForegroundColor DarkGray -NoNewline
                Start-Sleep -Seconds $waitChunk
                $remaining -= $waitChunk
                Write-Host "`r                              `r" -NoNewline
            }
            Write-Host ""

            # Exponential backoff with cap
            $backoffSeconds = [math]::Min($backoffSeconds * 2, $maxBackoff)
        }
        catch {
            # Non-timeout exception - log and return error
            Write-Host "  [ERROR] Exception during Claude invocation: $_" -ForegroundColor Red
            Log-APITimeoutRetry -StoryId $StoryId -Attempt $attempt -Backoff 0 -ErrorType "exception"

            return @{
                TimedOut = $false
                Decompose = $false
                Reason = "exception"
                Attempts = $attempt
                Exited = $false
                ExitCode = -1
                Output = $_.ToString()
                ErrorMessage = $_.ToString()
                ErrorType = "exception"
            }
        }
    }
}

# ============================================================================
# ORIGINAL FUNCTIONS
# ============================================================================

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
        [Parameter(Mandatory)][string]$ErrFile,
        [string]$FocusArea = ""
    )

    # Get timeout from config
    $timeout = 600
    if ($script:Config.iterationTimeout) { $timeout = $script:Config.iterationTimeout }
    if ($script:Config.autonomy -and $script:Config.autonomy.iterationTimeout) {
        $timeout = $script:Config.autonomy.iterationTimeout
    }

    $flagsString = ($ClaudeArgs -join ' ')
    $executionStart = Get-Date

    # Log that we're about to call Claude
    Write-SessionLog -Event "claude_call" -Message "Starting Claude subprocess" -Data @{
        timeout = $timeout
        argsCount = $ClaudeArgs.Count
    }
    Write-Heartbeat -Phase "starting_claude" -Details @{
        timeout = $timeout
        startTime = $executionStart.ToString("HH:mm:ss")
    }

    # Get focus-area-specific stall thresholds
    $stallThresholds = Get-StallThresholds -FocusArea $FocusArea
    if ($stallThresholds.isExtended) {
        $stallMins = [math]::Round($stallThresholds.stallThreshold / 60, 1)
        $killMins = [math]::Round($stallThresholds.killThreshold / 60, 1)
        Write-Host "  [INFO] Testing focus area - extended stall timeout: ${stallMins}min warning, ${killMins}min kill" -ForegroundColor Cyan
    }

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
        $lastHeartbeatUpdate = 0

        # CPU activity tracking (5th signal for stall detection)
        $lastCpuTime = 0
        $cpuActivityConfig = $script:Config.stallDetection.cpuActivity
        $cpuActivityEnabled = if ($cpuActivityConfig -and $null -ne $cpuActivityConfig.enabled) { $cpuActivityConfig.enabled } else { $true }
        $cpuActivityThreshold = if ($cpuActivityConfig -and $cpuActivityConfig.threshold) { $cpuActivityConfig.threshold } else { 0.5 }

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

                # CPU activity detection (5th signal)
                $cpuActive = $false
                $cpuDelta = 0
                if ($cpuActivityEnabled -and $sample.cpu -gt 0) {
                    $cpuDelta = $sample.cpu - $lastCpuTime
                    $cpuActive = $cpuDelta -gt $cpuActivityThreshold
                    $lastCpuTime = $sample.cpu
                }

                if ($prdUpdated -or $progressUpdated -or $gitChanged -or $bufferGrowing -or $cpuActive) {
                    $reason = if ($prdUpdated) { "prd.json" } `
                        elseif ($progressUpdated) { "progress.txt" } `
                        elseif ($gitChanged) { "git changes" } `
                        elseif ($cpuActive) { "CPU active (+$([math]::Round($cpuDelta, 1))s)" } `
                        else { "claude output" }
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

                # Update heartbeat every 30 seconds (6 check intervals)
                if ($totalElapsed - $lastHeartbeatUpdate -ge 30) {
                    Write-Heartbeat -Phase "waiting_for_claude" -Details @{
                        waitingSeconds = $totalElapsed
                        timeSinceActivity = $timeSinceProgress
                        outputBytes = $currentBufferLength
                        processId = $process.Id
                    }
                    $lastHeartbeatUpdate = $totalElapsed

                    # Check for stall condition and log to healing system (use focus-area-specific thresholds)
                    $stallCheck = Test-ClaudeStall -WaitingSeconds $totalElapsed -TimeSinceActivity $timeSinceProgress `
                        -StallThreshold $stallThresholds.stallThreshold -KillThreshold $stallThresholds.killThreshold
                    if ($stallCheck.ShouldKill) {
                        Write-Host "  [HEALING] Stall threshold exceeded (${timeSinceProgress}s without activity) - killing process" -ForegroundColor Red
                        # Kill immediately instead of just breaking the loop
                        $treePid = $process.Id
                        try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                        if (-not $process.HasExited) {
                            try { $process.Kill() } catch {}
                        }
                        # Brief wait for process termination
                        Start-Sleep -Milliseconds 500
                        break  # Now exit loop - process is dead
                    }
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
            Start-Sleep -Milliseconds 500  # Give async event handlers time to process final chunks
            $exitCode = $process.ExitCode
        }
    }
    finally {
        Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
        Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
        Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue

        # Poll until output buffer stabilizes (max 2 seconds)
        # This ensures async event handlers have fully drained before we read the buffer
        $maxWait = 2000
        $waited = 0
        $lastLen = $outBuilder.Length
        while ($waited -lt $maxWait) {
            Start-Sleep -Milliseconds 200
            $waited += 200
            $currentLen = $outBuilder.Length
            if ($currentLen -eq $lastLen) { break }  # Buffer stable
            $lastLen = $currentLen
        }

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
        Write-SessionLog -Event "claude_timeout" -Message "Claude subprocess timed out after ${timeout}s" -Data @{
            totalElapsed = $totalElapsed
            outputLength = $outBuilder.Length
        }
        Write-Heartbeat -Phase "claude_timeout" -Details @{ timeout = $timeout; elapsed = $totalElapsed }

        # Invoke stall recovery and log to healing system
        $recoveryResult = Invoke-StallRecovery -Reason "timeout" -WaitingSeconds $totalElapsed -ProcessId $processId
        if ($recoveryResult.ShouldPause) {
            Write-Host "  [HEALING] Too many stalls - consider pausing session" -ForegroundColor Red
        }
    }
    else {
        $resultType = if ($exitCode -eq 0) { "success" } else { "failure" }
        Write-SessionLog -Event "claude_return" -Message "Claude subprocess exited ($resultType)" -Data @{
            exitCode = $exitCode
            durationSec = $totalElapsed
            outputLength = $outBuilder.Length
        }
        Write-Heartbeat -Phase "claude_returned" -Details @{ exitCode = $exitCode; durationSec = $totalElapsed }

        # Reset stall tracking on successful completion
        if ($exitCode -eq 0) {
            Reset-StallTracking
        }
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
        [Console]::Out.Flush()  # Force immediate display instead of buffering
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

# ============================================================================
# FOCUS AREA EXECUTION
# ============================================================================

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
            [void](Invoke-FocusAreaExploration -FocusArea $FocusAreaId -Reason "sprint_start" -FullExplore)

            # Reset stories counter since we're starting fresh
            $script:State.StoriesSinceExploration = 0

            Write-Host ""
        }
    }

    # Archive existing incomplete sprint before generating new one
    # (completed sprints are already archived by the calling loop)
    if ($GeneratePRD -and (Test-Path $script:PrdFile)) {
        $existingPrd = Get-Sprint
        if ($existingPrd -and $existingPrd.userStories) {
            $incompleteStories = @($existingPrd.userStories | Where-Object { $_.passes -ne $true })
            if ($incompleteStories.Count -gt 0) {
                Save-SprintArchive -Reason "superseded"
            }
        }
    }

    # Build the prompt
    if ($GeneratePRD) {
        # Build exploration context section for PRD prompt
        $explorationSection = ""
        if ($script:State.SprintExplorationContext) {
            $explorationSection = @"

## Exploration Context (Fresh Scan)
$script:State.SprintExplorationContext

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
   - sprintNumber: increment from current (check current prd.json first)
   - branchName: "ralph/sprint-N" (matching sprintNumber)
   - 8-12 specific user stories with:
     - **CRITICAL: Story IDs MUST be sprint-unique using format: US-{sprintNumber}-001, US-{sprintNumber}-002, etc.**
       Example for Sprint 32: US-32-001, US-32-002, US-32-003...
     - Clear acceptance criteria (4-6 items each)
     - passes: false for all stories
     - Action verbs in titles (Add, Create, Update, Fix, etc.)

$(if ($Context) { "Context from user: $Context" } else { "" })
$explorationSection
Start by reading the config and prompt files to get the current sprintNumber, then generate the PRD with sprint-prefixed story IDs.
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
