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
        Invoke agent subprocess with infinite retry on API timeout
    .DESCRIPTION
        Keeps retrying forever on API timeout (not just 3 times) with exponential backoff.
        Has a 30-minute max per story to prevent infinite loops.
        Properly handles non-timeout exceptions.
    .PARAMETER ClaudePath
        Path to agent executable
    .PARAMETER ClaudeArgs
        Arguments for agent CLI
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
    .PARAMETER PromptMethod
        How to deliver the prompt: "stdin" (pipe to stdin) or "arg" (already in args).
        Default: "stdin" for backward compatibility.
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
        [string]$FocusArea = "",
        [string]$PromptMethod = "stdin"
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
                ExecutionStart = $StoryStartTime
                ExecutionEnd = Get-Date
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
                -FocusArea $FocusArea `
                -StoryId $StoryId `
                -PromptMethod $PromptMethod

            # Success or non-timeout failure - convert to hashtable if needed
            $safeResult = if ($result -is [hashtable]) { $result }
                          elseif ($result -is [array] -and $result.Length -gt 0 -and $result[0] -is [hashtable]) { $result[0] }
                          else {
                              # Try to extract hashtable from array
                              $ht = @{}
                              foreach ($item in $result) {
                                  if ($item -is [hashtable]) {
                                      foreach ($key in $item.Keys) { $ht[$key] = $item[$key] }
                                  }
                              }
                              if ($ht.Count -gt 0) { $ht } else { $result }
                          }
            if (-not $safeResult.TimedOut) {
                # Return result as-is - Attempts is already set in Invoke-ClaudeSubprocess
                return ,$safeResult
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
            Write-Host "  [ERROR] Exception details: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "  [ERROR] Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
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
                ExecutionStart = $StoryStartTime
                ExecutionEnd = Get-Date
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
        Spawn agent CLI process with async output capture and
        activity-based timeout monitoring. Kills process on timeout.
        Supports both stdin and arg-based prompt delivery.
    .PARAMETER ClaudePath
        Full path to agent executable
    .PARAMETER ClaudeArgs
        Arguments array for agent CLI
    .PARAMETER Prompt
        Prompt text to pipe via stdin (ignored when PromptMethod is "arg")
    .PARAMETER OutFile
        Path to write stdout log
    .PARAMETER ErrFile
        Path to write stderr log
    .PARAMETER StoryId
        Story ID to monitor for early exit when marked passes:true in prd.json
    .PARAMETER PromptMethod
        How to deliver the prompt: "stdin" (pipe to stdin) or "arg" (already in args).
        Default: "stdin" for backward compatibility.
    .RETURNS
        Hashtable: Exited, ExitCode, Output, ResourceSamples, ExecutionStart, ExecutionEnd, TimedOut
    #>
    param(
        [Parameter(Mandatory)][string]$ClaudePath,
        [Parameter(Mandatory)][string[]]$ClaudeArgs,
        [Parameter(Mandatory)][string]$Prompt,
        [Parameter(Mandatory)][string]$OutFile,
        [Parameter(Mandatory)][string]$ErrFile,
        [string]$FocusArea = "",
        [string]$StoryId = "",
        [string]$PromptMethod = "stdin"
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
    $psi.RedirectStandardInput = ($PromptMethod -eq "stdin")
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi

    # Remove CLAUDECODE env var to allow nested Claude Code sessions
    $psi.Environment.Remove("CLAUDECODE")

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

        # Deliver prompt via stdin (for claude) or skip (for arg-based providers like codex)
        if ($PromptMethod -eq "stdin") {
            $process.StandardInput.Write($Prompt)
            $process.StandardInput.Close()
        }

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

        # Subprocess detection config (6th signal - pytest/test runner)
        $subprocConfig = $script:Config.stallDetection.subprocessDetection
        $subprocEnabled = if ($subprocConfig -and $null -ne $subprocConfig.enabled) { $subprocConfig.enabled } else { $true }
        $pytestTimeoutMultiplier = if ($subprocConfig -and $subprocConfig.pytestTimeoutMultiplier) { $subprocConfig.pytestTimeoutMultiplier } else { 3.0 }
        $lastChildCpuTime = 0
        $testRunnerDetected = $false
        $testRunnerAnnounced = $false

        # Early exit: track when story is marked done in prd.json
        $earlyExitConfig = $script:Config.stallDetection.storyCompletionEarlyExit
        $earlyExitEnabled = if ($earlyExitConfig -and $null -ne $earlyExitConfig.enabled) { $earlyExitConfig.enabled } else { $true }
        $storyCompletionDetected = $false
        $storyCompletionTime = $null
        $storyCompletionGraceSec = if ($earlyExitConfig -and $earlyExitConfig.gracePeriodSeconds) { $earlyExitConfig.gracePeriodSeconds } else { 15 }
        $earlyExitRecheckIntervalSec = 30  # Periodic re-check interval to catch missed file writes
        $lastEarlyExitRecheck = 0

        # Quick Edit Mode safeguard: re-check periodically in case user clicks console
        $quickEditRecheckIntervalSec = 60
        $lastQuickEditRecheck = 0

        while (-not $process.HasExited -and $timeSinceProgress -lt $timeout -and $totalElapsed -lt ($maxTotalMinutes * 60)) {
            Start-Sleep -Seconds $checkIntervalSec
            $timeSinceProgress += $checkIntervalSec
            $totalElapsed += $checkIntervalSec

            # Safeguard: re-disable Quick Edit Mode periodically in case user clicks console
            if ($totalElapsed - $lastQuickEditRecheck -ge $quickEditRecheckIntervalSec) {
                $lastQuickEditRecheck = $totalElapsed
                $qeDisabled = $false
                try {
                    $qeDisabled = Disable-QuickEditMode
                } catch {}
                # Silent re-disable - no output to avoid cluttering logs
            }

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

                # Subprocess detection (6th signal - pytest/test runner)
                $testRunnerActive = $false
                $childCpuDelta = 0
                if ($subprocEnabled) {
                    $childActivity = Get-ChildProcessActivity -ParentProcessId $process.Id
                    if ($childActivity.hasTestRunner) {
                        $testRunnerDetected = $true
                        $childCpuDelta = $childActivity.totalChildCpu - $lastChildCpuTime
                        $testRunnerActive = $childCpuDelta -gt $cpuActivityThreshold
                        $lastChildCpuTime = $childActivity.totalChildCpu

                        # Announce test runner detection once
                        if (-not $testRunnerAnnounced) {
                            $extendedKill = [math]::Round($stallThresholds.killThreshold * $pytestTimeoutMultiplier / 60, 1)
                            Write-Host "  [INFO] Test runner detected ($($childActivity.testRunnerName)) - extending kill threshold to ${extendedKill}min" -ForegroundColor Cyan
                            $testRunnerAnnounced = $true
                        }
                    }
                }

                # Early exit: check if story was marked done in prd.json
                # Trigger on: (a) prd.json file write detected, or (b) periodic re-check interval
                # The periodic re-check catches cases where passes:true was written in the same
                # polling interval as a prior write, causing $prdUpdated to miss it
                $periodicRecheck = $earlyExitEnabled -and $StoryId -and -not $storyCompletionDetected -and ($totalElapsed - $lastEarlyExitRecheck) -ge $earlyExitRecheckIntervalSec
                if ($earlyExitEnabled -and ($prdUpdated -or $periodicRecheck) -and $StoryId -and -not $storyCompletionDetected) {
                    if ($periodicRecheck) { $lastEarlyExitRecheck = $totalElapsed }
                    try {
                        $prdJson = Get-Content $script:PrdFile -Raw -ErrorAction Stop | ConvertFrom-Json
                        $story = $prdJson.userStories | Where-Object { $_.id -eq $StoryId }
                        if ($story -and $story.passes -eq $true) {
                            $storyCompletionDetected = $true
                            $storyCompletionTime = Get-Date
                            $storyCompletionGitHead = (git rev-parse HEAD 2>$null)
                            $detectMethod = if ($prdUpdated) { "file change" } else { "periodic recheck" }
                            Write-Host "  [$mins min] Story $StoryId marked DONE ($detectMethod) - grace period ${storyCompletionGraceSec}s for commit..." -ForegroundColor Green
                        }
                    } catch {
                        # PRD read failed (file locked, etc.) - skip this check
                    }
                }

                # Early exit: immediate termination if git commit detected after story done
                if ($storyCompletionDetected -and $storyCompletionGitHead) {
                    $currentHead = (git rev-parse HEAD 2>$null)
                    if ($currentHead -and $currentHead -ne $storyCompletionGitHead) {
                        Write-Host "  [$mins min] Git commit after story done - terminating" -ForegroundColor Green
                        try { taskkill /T /F /PID $process.Id 2>$null | Out-Null } catch {}
                        if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                        Start-Sleep -Milliseconds 500
                        break
                    }
                }

                # Early exit: kill Claude after grace period expires
                if ($storyCompletionDetected -and $storyCompletionTime) {
                    $sinceCompletion = ((Get-Date) - $storyCompletionTime).TotalSeconds
                    if ($sinceCompletion -ge $storyCompletionGraceSec) {
                        Write-Host "  [$mins min] Grace period expired - terminating Claude (story already done)" -ForegroundColor Yellow
                        try { taskkill /T /F /PID $process.Id 2>$null | Out-Null } catch {}
                        if (-not $process.HasExited) {
                            try { $process.Kill() } catch {}
                        }
                        Start-Sleep -Milliseconds 500
                        break
                    }
                }

                if ($prdUpdated -or $progressUpdated -or $gitChanged -or $bufferGrowing -or $cpuActive -or $testRunnerActive) {
                    $reason = if ($prdUpdated -and $storyCompletionDetected) { "prd.json (story DONE, grace period)" } `
                        elseif ($prdUpdated) { "prd.json" } `
                        elseif ($progressUpdated) { "progress.txt" } `
                        elseif ($gitChanged) { "git changes" } `
                        elseif ($cpuActive) { "CPU active (+$([math]::Round($cpuDelta, 1))s)" } `
                        elseif ($testRunnerActive) { "pytest running (+$([math]::Round($childCpuDelta, 1))s CPU)" } `
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
                    # Extend thresholds if test runner is detected
                    $effectiveStallThreshold = $stallThresholds.stallThreshold
                    $effectiveKillThreshold = $stallThresholds.killThreshold
                    if ($testRunnerDetected) {
                        $effectiveStallThreshold = [math]::Round($stallThresholds.stallThreshold * $pytestTimeoutMultiplier)
                        $effectiveKillThreshold = [math]::Round($stallThresholds.killThreshold * $pytestTimeoutMultiplier)
                    }
                    $stallCheck = Test-ClaudeStall -WaitingSeconds $totalElapsed -TimeSinceActivity $timeSinceProgress `
                        -StallThreshold $effectiveStallThreshold -KillThreshold $effectiveKillThreshold
                    if ($stallCheck.ShouldKill) {
                        $thresholdInfo = if ($testRunnerDetected) { " (extended for pytest)" } else { "" }
                        Write-Host "  [HEALING] Stall threshold exceeded (${timeSinceProgress}s without activity${thresholdInfo}) - killing process" -ForegroundColor Red
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
        # Handle case where process hasn't exited yet (shouldn't happen but defensive)
        if (-not $exited) {
            Write-Host "  Warning: Process still running after loop exit, waiting..." -ForegroundColor Yellow
            try {
                $process.WaitForExit(5000)  # Wait up to 5 seconds
                $exited = $process.HasExited
            } catch {
                $exited = $false
            }
        }

        if ($exited) {
            # Do NOT call parameterless WaitForExit() — it deadlocks on .NET Framework
            # when child processes hold stdout/stderr pipe handles open.
            # HasExited is already true, so just cancel async readers and read exit code.
            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
            Start-Sleep -Milliseconds 500  # Give async event handlers time to process final chunks
            try { $exitCode = $process.ExitCode } catch { $exitCode = $null }
        }

        # Ensure exitCode is never null - default to 1 for killed/terminated processes
        if ($null -eq $exitCode) {
            $exitCode = 1
        }

        # ULTRA-DEBUG: Log exit code state immediately
        Write-Host "  [DEBUG] Before override: exitCode=$exitCode, storyCompletionDetected=$storyCompletionDetected" -ForegroundColor Magenta

        # Early exit override: taskkill produces exit code 1, but story actually succeeded
        # Also handle null/empty exitCode from failed kill attempts
        # Re-validate passes in prd.json before overriding — Claude may have reverted
        # passes:true→false during the grace period (e.g., test failure rollback)
        if ($storyCompletionDetected) {
            $exitCodeInt = if ($exitCode -is [int]) { $exitCode } elseif ($exitCode -match '^\d+$') { [int]$exitCode } else { -1 }
            if ($exitCodeInt -ne 0) {
                $stillPasses = $false
                if ($StoryId -and $script:PrdFile -and (Test-Path $script:PrdFile)) {
                    try {
                        $freshPrd = Get-Content $script:PrdFile -Raw -ErrorAction Stop | ConvertFrom-Json
                        $freshStory = $freshPrd.userStories | Where-Object { $_.id -eq $StoryId } | Select-Object -First 1
                        $stillPasses = $freshStory -and $freshStory.passes -eq $true
                    } catch {
                        # If we can't read, trust the original detection
                        $stillPasses = $true
                    }
                } else {
                    $stillPasses = $true
                }

                if ($stillPasses) {
                    Write-Host "  [DEBUG] OVERRIDING exitCode from $exitCodeInt to 0" -ForegroundColor Magenta
                    $exitCode = 0
                } else {
                    $storyCompletionDetected = $false
                }
            }
        }
    }
    finally {
        Write-Host "  [FINALLY-DEBUG] Starting finally block" -ForegroundColor Yellow
        # Timeout-protected event cleanup — Remove-Job can deadlock when child processes
        # (e.g. Node.js subagents) inherit stdout/stderr pipe handles and hold them open
        # after the main Claude process exits. Same .NET pipe-handle issue as WaitForExit().
        Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
        Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        Stop-Job -Job $outEvent -ErrorAction SilentlyContinue
        Stop-Job -Job $errEvent -ErrorAction SilentlyContinue
        # Timeout-protected Remove-Job using [powershell]::Create() for proper runspace.
        # Raw [System.Threading.Thread] with PowerShell cmdlets causes unhandled exceptions
        # that crash the host process (.NET Framework terminates on unhandled thread exceptions).
        $ps = [powershell]::Create()
        $ps.AddScript({
            param($outJob, $errJob)
            Remove-Job -Job $outJob -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errJob -Force -ErrorAction SilentlyContinue
        }).AddArgument($outEvent).AddArgument($errEvent) | Out-Null
        $asyncResult = $ps.BeginInvoke()
        if (-not $asyncResult.AsyncWaitHandle.WaitOne(5000)) {
            Write-Host "  Warning: Event cleanup timed out (pipe handle deadlock avoided)" -ForegroundColor Yellow
        }
        $ps.Dispose()

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
            if ($OutFile) {
                $outBuilder.ToString() | Set-Content $OutFile -ErrorAction Stop
            }
        } catch {
            Write-Host "  Warning: Failed to write Claude output to $OutFile : $_" -ForegroundColor Yellow
        }
        try {
            if ($ErrFile) {
                $errBuilder.ToString() | Set-Content $ErrFile -ErrorAction Stop
            }
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

    Write-Host "  [DEBUG] RETURNING: exitCode=$exitCode, timedOut=$timedOut" -ForegroundColor Magenta

    # Explicitly create a new hashtable to avoid PowerShell return value quirks
    $returnHashtable = @{}
    $returnHashtable['Exited'] = $exited
    $returnHashtable['ExitCode'] = $exitCode
    $returnHashtable['Output'] = $outBuilder.ToString() + $errBuilder.ToString()
    $returnHashtable['ResourceSamples'] = $resourceSamples
    $returnHashtable['ExecutionStart'] = $executionStart
    $returnHashtable['ExecutionEnd'] = $executionEnd
    $returnHashtable['TimedOut'] = $timedOut
    $returnHashtable['Timeout'] = $timeout
    $returnHashtable['ProcessId'] = $processId
    $returnHashtable['Attempts'] = 1
    Write-Host "  [DEBUG] Return hashtable ExitCode: $($returnHashtable.ExitCode)" -ForegroundColor Magenta
    Write-Host "  [DEBUG] Return hashtable type: $($returnHashtable.GetType().Name)" -ForegroundColor Magenta
    return ,$returnHashtable  # Note: comma prefix prevents PowerShell unrolling
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

    # DEBUG: Log what we received
    Write-Host "  [RESOLVE-DEBUG] TimedOut=$($SubResult.TimedOut), ExitCode=$($SubResult.ExitCode)" -ForegroundColor Cyan

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
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms `
            -Role $(if ($Ctx.StoryObj) { Get-StoryRole -Story $Ctx.StoryObj } else { "" })

        if ($Ctx.StoryObj) { $null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }
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
        $diffOutput = git diff HEAD~1 -- ":(exclude)scripts/ralph/state/" ":(exclude)scripts/ralph/session/" ":(exclude)scripts/ralph/archive/" 2>$null
        $diffQuality = Get-DiffQualityScore -DiffOutput $diffOutput
        if ($diffQuality.warnings.Count -gt 0) {
            foreach ($warn in $diffQuality.warnings) {
                Write-Host "  Quality: $warn" -ForegroundColor Yellow
            }
        }

        # Test regression detection (oracle-based when regressionGuard enabled)
        $regressionResult = $null
        if ($Ctx.TestResults) {
            $baselineOverride = if ($script:State -and $script:State.PreStoryBaseline) { $script:State.PreStoryBaseline } else { $null }
            $regressionResult = Compare-TestBaseline -CurrentResults $Ctx.TestResults -BaselineOverride $baselineOverride
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
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms `
            -Role $(if ($Ctx.StoryObj) { Get-StoryRole -Story $Ctx.StoryObj } else { "" })

        if ($Ctx.StoryObj) {
            $evidenceResult = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $true -ClaudeOutput $Ctx.ClaudeOutput -DiffOutput $diffOutput

            # Evidence threshold gate: reject stories with insufficient criteria verification
            $evidenceConfig = $script:Config.stallDetection.storyCompletionEarlyExit
            $evidenceMinPct = if ($evidenceConfig -and $null -ne $evidenceConfig.evidenceThresholdPercent) { $evidenceConfig.evidenceThresholdPercent } else { 90 }
            $belowThreshold = $evidenceResult -and $evidenceResult.criteriaTotal -gt 0 -and $evidenceResult.percentage -lt $evidenceMinPct
            $keywordFallbackUsed = $evidenceResult -and $evidenceResult.usedKeywordFallback

            if ($belowThreshold -and -not $keywordFallbackUsed) {
                # LLM-based evidence is below threshold -- reject
                Write-Host "  Evidence below threshold ($($evidenceResult.percentage)% < $($evidenceMinPct)%) - rejecting story" -ForegroundColor Red
                [Console]::Out.Flush()
                $null = Update-StoryStatus -StoryId $Ctx.StoryId -Passes $false -Notes "Evidence gate: $($evidenceResult.criteriaMet)/$($evidenceResult.criteriaTotal) criteria verified ($($evidenceResult.percentage)%). Minimum: $($evidenceMinPct)%."
                $success = $false
                $iterationStatus = "evidence_rejected"
                $script:State.ConsecutiveFailures++
                Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $false; reason = "evidence_below_threshold"; evidence = "$($evidenceResult.criteriaMet)/$($evidenceResult.criteriaTotal)" }
            } elseif ($belowThreshold -and $keywordFallbackUsed) {
                # Keyword fallback is too weak to override Claude's successful completion
                Write-Host "  Evidence: keyword fallback $($evidenceResult.percentage)% < $($evidenceMinPct)% -- accepting (keyword matching too weak to reject)" -ForegroundColor DarkYellow
                [Console]::Out.Flush()
                Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $true; reason = "keyword_fallback_accepted"; evidence = "$($evidenceResult.criteriaMet)/$($evidenceResult.criteriaTotal)" }
            } else {
                Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $true }
            }

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

            # Update baseline after successful story (skip if evidence-rejected)
            if ($success -and $Ctx.TestResults) {
                $null = Update-TestBaseline -TestResults $Ctx.TestResults
            }
        }

        # Reset consecutive failures on success
        $script:State.ConsecutiveFailures = 0
    }
    else {
        # === FAILURE (non-zero exit code) ===
        $iterationStatus = "failed"
        $errorCategory = Get-ErrorCategory -Output $Ctx.ClaudeOutput -TimedOut $false

        Write-Host "  Iteration failed (exit code: $exitCode)" -ForegroundColor Red

        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Exit code: $exitCode" -Iteration $script:State.IterationCount
        Log-StateTransition -From "running" -To "failed" -Reason "Exit code: $exitCode" -Context $Ctx.TransitionContext

        Record-Metric -StoryId $Ctx.Identifier -Mode $script:State.CurrentMode `
            -DurationMin ([math]::Round($Ctx.IterationDuration.TotalMinutes, 0)) `
            -Success $false -Timeout $false -TokensUsed $Ctx.TokensUsed `
            -ErrorCategory $errorCategory -TestResults $Ctx.TestResults `
            -RetryCount $script:State.CurrentRetryCount `
            -LinesAdded 0 -LinesDeleted 0 `
            -PhaseReadMs $Ctx.PhaseTimings.read_ms -PhaseAnalyzeMs $Ctx.PhaseTimings.analyze_ms `
            -PhaseImplementMs $Ctx.PhaseTimings.implement_ms -PhaseTestMs $Ctx.PhaseTimings.test_ms `
            -PhaseCommitMs $Ctx.PhaseTimings.commit_ms `
            -Role $(if ($Ctx.StoryObj) { Get-StoryRole -Story $Ctx.StoryObj } else { "" })

        if ($Ctx.StoryObj) { $null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }
        $script:State.ConsecutiveFailures++
    }

    return @{
        Success = $success
        IterationStatus = $iterationStatus
    }
}

# ============================================================================
# INVESTIGATION / EXPLORATION MODE (Phase 2)
# ============================================================================

function Invoke-ClaudeExploration {
    <#
    .SYNOPSIS
        Run Claude in a lightweight investigation mode (no retries, no 30-min limit).
        Used for exploration, debugging, and one-shot questions.
    .PARAMETER Prompt
        The prompt to send to Claude
    .PARAMETER PromptType
        Type of prompt for logging
    .PARAMETER Identifier
        The focus area or story ID
    .PARAMETER FocusArea
        The focus area context (optional)
    .RETURNS
        Hashtable with: Output, ExitCode, TimedOut
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,
        [Parameter(Mandatory=$true)]
        [string]$PromptType,
        [Parameter(Mandatory=$true)]
        [string]$Identifier,
        [string]$FocusArea = ""
    )

    $script:State.IterationCount++
    $iterationStart = Get-Date

    Write-IterationBanner -Iteration $script:State.IterationCount -FocusArea $FocusArea -StoryId ""

    # Build Claude command
    $provider = Get-AgentProvider
    $model = Get-AgentModel -Provider $provider
    $claudePath = Get-AgentExecutable -Provider $provider
    $agentCommand = Build-AgentCommand -Provider $provider -Model $model -Prompt $Prompt -Options @{}
    $claudeArgs = $agentCommand.Args
    $promptMethod = $agentCommand.PromptMethod

    Write-Host "  Invoking Claude (exploration mode)..." -ForegroundColor Cyan

    # Setup temp files
    $outFile = Join-Path $script:Paths.SessionDir "exploration_out.txt"
    $errFile = Join-Path $script:Paths.SessionDir "exploration_err.txt"

    # Simple invocation (no retry logic for exploration)
    $result = Invoke-ClaudeSubprocess `
        -ClaudePath $claudePath `
        -ClaudeArgs $claudeArgs `
        -Prompt $Prompt `
        -OutFile $outFile `
        -ErrFile $errFile `
        -FocusArea $FocusArea `
        -PromptMethod $promptMethod

    # Return just what callers need
    return @{
        Output = $result.Output
        ExitCode = $result.ExitCode
        TimedOut = $result.TimedOut
    }
}
