# scripts/ralph/lib/loops.ps1
# Main loop implementations: interview queue, TrueAuto, standard, Ralph's Choice modes

function Start-InterviewQueueLoop {
    <#
    .SYNOPSIS
        Process focus areas from interview queue
    #>

    $script:State.CurrentMode = "Interview"
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
                try {
                    Save-SprintArchive -Reason "complete"
                    Update-QueueProgress -AreaId $areaId
                }
                catch {
                    Write-Host "  Warning: Error during sprint archive: $_" -ForegroundColor Yellow
                }
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
                    Write-SessionLog -Event "story_failed" -Message "Story $($status.nextStory.id) failed"
                }
                else {
                    Write-SessionLog -Event "story_success" -Message "Story $($status.nextStory.id) completed"
                }

                # After first story (US-001 generates full PRD), update context
                if (-not $contextRefreshed) {
                    try {
                        Update-ContextFromPRD
                    } catch {
                        Write-Host "  Warning: Failed to update context from PRD: $_" -ForegroundColor Yellow
                    }
                    $contextRefreshed = $true
                }
            }
            else {
                Write-Host "  No stories found in PRD for $areaId" -ForegroundColor Yellow
                break
            }

            # Log between-iteration pause
            Write-Heartbeat -Phase "between_iterations" -Details @{ lastStory = $status.nextStory.id }
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
    .PARAMETER FocusArea
        Optional focus area to use instead of Ralph's Choice scoring
    #>
    param([string]$FocusArea = "")

    $script:State.CurrentMode = "TrueAuto"
    Write-Host "  TrueAuto mode: Continuous improvement" -ForegroundColor Magenta
    Write-Host ""

    # Pre-flight: batch check for stories already committed in git
    $preFlightCompleted = Invoke-BatchPreFlight

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget)) {
        $status = Get-SprintStatus

        if ($status.complete) {
            Write-Host ""
            Write-Host "  Sprint complete!" -ForegroundColor Green

            # Entire sprint transition wrapped in try-catch to prevent silent exits
            try {
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
                    if (-not $scores -or $scores.Count -eq 0) {
                        Write-Host "  Error: No focus area scores returned - cannot select next area" -ForegroundColor Red
                        break
                    }
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
            catch {
                Write-Host ""
                Write-Host "  ERROR: Failed to start next sprint: $_" -ForegroundColor Red
                Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkGray
                Write-Host ""
                break
            }
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
    Write-Host "  Iterations: $($script:State.IterationCount)" -ForegroundColor DarkGray
}

function Start-StandardLoop {
    <#
    .SYNOPSIS
        Standard mode - work through stories until sprint complete
        After sprint complete, checks queue.json for pending focus areas and advances
    #>

    $script:State.CurrentMode = "Standard"

    # Check if we need to generate a new PRD for the queued focus area
    # Read queue directly (inline) for reliability - Get-NextQueuedFocusArea has been unreliable
    $queuedArea = $null
    $inlineQueue = Get-Queue
    if ($inlineQueue) {
        if ($inlineQueue.focusAreas) {
            $inlineIncomplete = @($inlineQueue.focusAreas | Where-Object { -not $_.completed })
            Write-Host "  Queue: $($inlineQueue.focusAreas.Count) areas, $($inlineIncomplete.Count) incomplete" -ForegroundColor DarkGray
            if ($inlineIncomplete.Count -gt 0) {
                $queuedArea = $inlineIncomplete[0].id
            }
        }
    }
    Write-Host "  Queue next area: '$queuedArea'" -ForegroundColor DarkGray
    if ($queuedArea) {
        # Check current PRD focus area
        $currentPrdFocus = ""
        $currentPrd = Get-Sprint
        if ($currentPrd) {
            $currentPrdFocus = if ($currentPrd.focusArea) { $currentPrd.focusArea } else { "" }
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
                Update-QueueProgress -AreaId $status.focusArea -Silent
            }

            # Check for pending queue items (inline for reliability)
            $nextArea = $null
            $loopQueue = Get-Queue
            if ($loopQueue -and $loopQueue.focusAreas) {
                $loopIncomplete = @($loopQueue.focusAreas | Where-Object { -not $_.completed })
                if ($loopIncomplete.Count -gt 0) {
                    $nextArea = $loopIncomplete[0].id
                }
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
                # Queue area already marked complete at line 371 above
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

            # Log story result
            if ($success) {
                Write-SessionLog -Event "story_success" -Message "Story $($status.nextStory.id) completed"
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
            else {
                Write-SessionLog -Event "story_failed" -Message "Story $($status.nextStory.id) failed"
            }
        }
        else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        # Log between-iteration pause
        Write-Heartbeat -Phase "between_iterations" -Details @{ lastStory = $status.nextStory.id }
        Start-Sleep -Seconds 2
    }
}

function Start-RalphsChoiceLoop {
    <#
    .SYNOPSIS
        Ralph's Choice mode - Ralph decides focus areas, user confirms each
    #>

    $script:State.CurrentMode = "RalphsChoice"
    Write-Host "  Ralph's Choice mode: Ralph decides, you confirm" -ForegroundColor Magenta
    Write-Host ""

    # Reconcile queue with sprint history (catch up on any missed completions)
    Sync-QueueFromHistory

    $sprintCount = 0
    $config = Get-RalphConfig

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget)) {
        $status = Get-SprintStatus

        if ($status.complete -or $sprintCount -eq 0) {
            if ($sprintCount -gt 0) {
                # Entire sprint transition wrapped in try-catch to prevent silent exits
                try {
                    Write-Host ""
                    Write-Host "=====================================================" -ForegroundColor Green
                    Write-Host "   SPRINT $sprintCount COMPLETE!" -ForegroundColor Green
                    Write-Host "=====================================================" -ForegroundColor Green
                    Write-Host ""

                    # Archive the completed sprint
                    Save-SprintArchive -Reason "complete"

                    # Update queue if completed area is tracked
                    if ($status.focusArea) {
                        Update-QueueProgress -AreaId $status.focusArea -Silent
                    }

                    # Check for graceful stop
                    if (Test-GracefulStopRequested) {
                        Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                        Clear-GracefulStopSignal
                        break
                    }

                    Write-Host "  Scoring focus areas for next sprint..." -ForegroundColor DarkGray

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
                }
                catch {
                    Write-Host ""
                    Write-Host "  ERROR: Failed to start next sprint: $_" -ForegroundColor Red
                    Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkGray
                    Write-Host ""
                    break
                }
            } else {
                # First sprint - just pick
                # Wrapped in try-catch to prevent silent crash (see session 2026-01-27_231506)
                try {
                    $scores = Get-AllFocusAreaScores
                    if (-not $scores -or $scores.Count -eq 0) {
                        Write-Host "  Error: No focus area scores returned" -ForegroundColor Red
                        break
                    }
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
                catch {
                    Write-Host ""
                    Write-Host "  ERROR: Failed to start next sprint: $_" -ForegroundColor Red
                    Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkGray
                    Write-Host ""
                    break
                }
            }

            Write-Host ""
            Write-Host "  Selected focus area: $selectedArea" -ForegroundColor Green
            Write-Host ""

            # Generate PRD for selected area
            try {
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
            catch {
                Write-Host ""
                Write-Host "  ERROR: Failed to generate PRD: $_" -ForegroundColor Red
                Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkGray
                Write-Host ""
                break
            }
        }

        # Work on current story
        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Log story result
            if ($success) {
                Write-SessionLog -Event "story_success" -Message "Story $($status.nextStory.id) completed"
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
            else {
                Write-SessionLog -Event "story_failed" -Message "Story $($status.nextStory.id) failed"
            }
        } else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        # Log between-iteration pause
        Write-Heartbeat -Phase "between_iterations" -Details @{ lastStory = $status.nextStory.id }
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

    $script:State.CurrentMode = "RalphsChoiceAuto"
    Write-Host "  Ralph's Choice Auto: Fully autonomous" -ForegroundColor Magenta
    Write-Host ""

    # Reconcile queue with sprint history (catch up on any missed completions)
    Sync-QueueFromHistory -Silent

    $sprintCount = 0
    $config = Get-RalphConfig
    $continueDelay = if ($config.ralphsChoice.autoMode.continueDelaySeconds) { $config.ralphsChoice.autoMode.continueDelaySeconds } else { 10 }
    $maxSprints = if ($config.ralphsChoice.autoMode.maxConsecutiveSprints) { $config.ralphsChoice.autoMode.maxConsecutiveSprints } else { 20 }

    while (-not (Test-MaxIterations) -and -not (Test-TokenBudget) -and $sprintCount -lt $maxSprints) {
        $status = Get-SprintStatus

        if ($status.complete -or $sprintCount -eq 0) {
            # Entire sprint transition wrapped in try-catch to prevent silent exits
            # (session 2026-01-27_231506 crashed silently after Save-SprintArchive)
            try {
                if ($sprintCount -gt 0) {
                    Write-Host ""
                    Write-Host "=====================================================" -ForegroundColor Green
                    Write-Host "   SPRINT $sprintCount COMPLETE!" -ForegroundColor Green
                    Write-Host "=====================================================" -ForegroundColor Green
                    Write-Host ""

                    # Archive the completed sprint
                    Save-SprintArchive -Reason "complete"

                    # Update queue if completed area is tracked
                    if ($status.focusArea) {
                        Update-QueueProgress -AreaId $status.focusArea -Silent
                    }

                    # Check for graceful stop
                    if (Test-GracefulStopRequested) {
                        Write-Host "  Honoring graceful stop request." -ForegroundColor Cyan
                        Clear-GracefulStopSignal
                        break
                    }

                    Write-Host "  Scoring focus areas for next sprint..." -ForegroundColor DarkGray
                }

                # Score, select next area, and generate new sprint PRD
                $scores = Get-AllFocusAreaScores
                if (-not $scores -or $scores.Count -eq 0) {
                    Write-Host "  Error: No focus area scores returned - cannot select next area" -ForegroundColor Red
                    break
                }
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

                    $quitRequested = $false
                    switch -Regex ($pauseChoice) {
                        "^[Qq]$" {
                            Write-Host "  Exiting Ralph's Choice Auto" -ForegroundColor Yellow
                            $quitRequested = $true
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

                    if ($quitRequested) {
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
            catch {
                Write-Host ""
                Write-Host "  ERROR: Failed to start next sprint: $_" -ForegroundColor Red
                Write-Host "  $($_.ScriptStackTrace)" -ForegroundColor DarkGray
                Write-Host ""
                break
            }
        }

        # Work on current story
        if ($status.nextStory) {
            $success = Invoke-ClaudeForStory -StoryId $status.nextStory.id

            if (Test-ShouldAbort) {
                break
            }

            # Log story result
            if ($success) {
                Write-SessionLog -Event "story_success" -Message "Story $($status.nextStory.id) completed"
                Invoke-PeriodicExplorationIfNeeded -FocusArea $status.focusArea | Out-Null
            }
            else {
                Write-SessionLog -Event "story_failed" -Message "Story $($status.nextStory.id) failed"
            }
        } else {
            Write-Host "  No stories found in PRD" -ForegroundColor Yellow
            break
        }

        # Log between-iteration pause
        Write-Heartbeat -Phase "between_iterations" -Details @{ lastStory = $status.nextStory.id }
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
