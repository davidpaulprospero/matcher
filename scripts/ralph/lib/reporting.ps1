# Ralph Reporting System
# Graceful stop reports, session summaries, recommendations

# ============================================================================
# SESSION DATA COLLECTION
# ============================================================================

function Get-SessionSummary {
    <#
    .SYNOPSIS
        Collect summary data for the current session
    .RETURNS
        Hashtable with session summary data
    #>

    $summary = @{
        startTime = $null
        endTime = (Get-Date).ToString("o")
        duration = ""
        totalIterations = 0
        mode = ""
        focusAreas = @()
    }

    # Get from state if available
    if ($script:State) {
        if ($script:State.SessionStartTime) {
            $summary.startTime = $script:State.SessionStartTime
            $start = [datetime]$script:State.SessionStartTime
            $end = Get-Date
            $duration = $end - $start
            $summary.duration = "{0:D2}:{1:D2}:{2:D2}" -f $duration.Hours, $duration.Minutes, $duration.Seconds
        }
        if ($script:State.TotalIterations) {
            $summary.totalIterations = $script:State.TotalIterations
        }
        if ($script:State.CurrentMode) {
            $summary.mode = $script:State.CurrentMode
        }
    }

    return $summary
}

function Get-SprintsSummary {
    <#
    .SYNOPSIS
        Get summary of sprints completed this session
    .RETURNS
        Array of sprint summary objects
    #>

    $sprints = @()

    # Try to get from sprint history
    try {
        $history = Get-SprintHistory
        if ($history.sprints) {
            # Get sprints from today
            $today = (Get-Date).Date
            $todaySprints = $history.sprints | Where-Object {
                if ($_.completedAt) {
                    ([datetime]$_.completedAt).Date -eq $today
                }
                else { $false }
            }

            foreach ($sprint in $todaySprints) {
                $sprints += @{
                    number = $sprint.sprintNumber
                    focusArea = $sprint.focusArea
                    status = "complete"
                    storiesCompleted = $sprint.storiesCompleted
                    storiesTotal = $sprint.storiesTotal
                    archiveFile = $sprint.archiveFile
                }
            }
        }
    }
    catch {}

    # Add current sprint if in progress
    try {
        $status = Get-SprintStatus
        if ($status -and -not $status.complete) {
            $sprints += @{
                number = "current"
                focusArea = $status.focusArea
                status = "partial"
                storiesCompleted = $status.passedCount
                storiesTotal = $status.stories.Count
                archiveFile = $null
            }
        }
    }
    catch {}

    return $sprints
}

function Get-HardStoriesThisSession {
    <#
    .SYNOPSIS
        Get hard stories marked during this session
    .RETURNS
        Array of hard story summaries
    #>

    $hardStories = @()

    try {
        $archive = Get-HardStoriesArchive
        if ($archive.stories) {
            $today = (Get-Date).Date
            $todayStories = $archive.stories | Where-Object {
                if ($_.lastFailedAt) {
                    ([datetime]$_.lastFailedAt).Date -eq $today
                }
                else { $false }
            }

            foreach ($hs in $todayStories) {
                $hardStories += @{
                    id = $hs.originalId
                    title = $hs.originalTitle
                    focusArea = $hs.focusArea
                    reason = $hs.lastReason
                    failureCount = $hs.failureCount
                    status = $hs.finalStatus
                }
            }
        }
    }
    catch {}

    return $hardStories
}

function Get-SessionErrors {
    <#
    .SYNOPSIS
        Get errors logged during this session
    .RETURNS
        Array of error summaries
    #>

    $errors = @()

    # Check session log
    $logFile = if ($script:SessionLogFile) { $script:SessionLogFile }
               else { Join-Path $script:RalphDir "logs\session.jsonl" }

    if (-not (Test-Path $logFile)) {
        return $errors
    }

    try {
        $today = (Get-Date).Date
        $lines = Get-Content $logFile

        foreach ($line in $lines) {
            try {
                $entry = $line | ConvertFrom-Json
                if ($entry.event -in @("error", "crash", "healing_failed", "story_failed")) {
                    if ($entry.timestamp) {
                        $entryDate = ([datetime]$entry.timestamp).Date
                        if ($entryDate -eq $today) {
                            $errors += @{
                                event = $entry.event
                                message = $entry.message
                                timestamp = $entry.timestamp
                            }
                        }
                    }
                }
            }
            catch {}
        }
    }
    catch {}

    return $errors
}

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

function Get-SessionRecommendations {
    <#
    .SYNOPSIS
        Generate recommendations based on session data
    .PARAMETER Sprints
        Sprint summary array
    .PARAMETER HardStories
        Hard stories array
    .PARAMETER Errors
        Errors array
    .RETURNS
        Array of recommendation strings
    #>
    param(
        [array]$Sprints = @(),
        [array]$HardStories = @(),
        [array]$Errors = @()
    )

    $recommendations = @()

    # Check for many hard stories
    if ($HardStories.Count -ge 3) {
        $recommendations += "Multiple hard stories ($($HardStories.Count)) - consider reviewing story complexity and scope"
    }

    # Check for timeout-related hard stories
    $timeoutStories = $HardStories | Where-Object { $_.reason -eq "30_minute_limit" }
    if ($timeoutStories.Count -gt 0) {
        $recommendations += "Stories hitting 30-minute limit - break down into smaller tasks"
    }

    # Check for recurring errors
    $importErrors = $Errors | Where-Object { $_.message -match "import|ImportError|ModuleNotFound" }
    if ($importErrors.Count -gt 2) {
        $recommendations += "Recurring import errors - verify requirements.txt and virtual environment"
    }

    $testFailures = $Errors | Where-Object { $_.event -eq "story_failed" }
    if ($testFailures.Count -gt 5) {
        $recommendations += "Many story failures - consider simpler acceptance criteria"
    }

    # Check sprint completion rate
    $completeSprints = $Sprints | Where-Object { $_.status -eq "complete" }
    if ($Sprints.Count -gt 0 -and $completeSprints.Count / $Sprints.Count -lt 0.5) {
        $recommendations += "Low sprint completion rate - some focus areas may need attention"
    }

    # No issues
    if ($recommendations.Count -eq 0) {
        $recommendations += "Session completed with no notable issues"
    }

    return $recommendations
}

# ============================================================================
# REPORT GENERATION
# ============================================================================

function Generate-GracefulStopReport {
    <#
    .SYNOPSIS
        Generate a comprehensive report when graceful stop is triggered
    .PARAMETER TriggerReason
        Why the stop was triggered (e.g., "user_graceful_stop", "lastsprint")
    .RETURNS
        Report hashtable
    #>
    param(
        [string]$TriggerReason = "user_graceful_stop"
    )

    Write-Host ""
    Write-Host "  Generating graceful stop report..." -ForegroundColor Cyan

    # Collect data
    $sessionSummary = Get-SessionSummary
    $sprints = Get-SprintsSummary
    $hardStories = Get-HardStoriesThisSession
    $errors = Get-SessionErrors
    $recommendations = Get-SessionRecommendations -Sprints $sprints -HardStories $hardStories -Errors $errors

    # Calculate totals
    $totalStoriesCompleted = ($sprints | Measure-Object -Property storiesCompleted -Sum).Sum
    $totalStoriesTotal = ($sprints | Measure-Object -Property storiesTotal -Sum).Sum

    # Build report
    $report = @{
        generatedAt = (Get-Date).ToString("o")
        triggeredBy = $TriggerReason

        summary = @{
            duration = $sessionSummary.duration
            mode = $sessionSummary.mode
            totalSprints = $sprints.Count
            totalStories = $totalStoriesTotal
            storiesCompleted = $totalStoriesCompleted
            storiesDecomposed = $hardStories.Count
        }

        currentState = @{
            focusAreas = $sessionSummary.focusAreas
            currentSprint = if ($script:State.CurrentSprintNumber) { $script:State.CurrentSprintNumber } else { 0 }
            currentStory = if ($script:State.CurrentStoryId) { $script:State.CurrentStoryId } else { "" }
        }

        sprints = $sprints
        hardStories = $hardStories
        errors = @($errors | Select-Object -First 10)  # Limit errors
        recommendations = $recommendations
    }

    # Save JSON report
    $reportDir = if ($script:ArchiveDir) { $script:ArchiveDir }
                 else { Join-Path $script:RalphDir "archive" }

    if (-not (Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $jsonPath = Join-Path $reportDir "overnight_report_$timestamp.json"
    $mdPath = Join-Path $reportDir "overnight_report_$timestamp.md"

    try {
        $report | ConvertTo-Json -Depth 10 | Set-Content -Path $jsonPath -Encoding UTF8
        Write-Host "  JSON report: $jsonPath" -ForegroundColor Green
    }
    catch {
        Write-Host "  Warning: Could not save JSON report: $_" -ForegroundColor Yellow
    }

    # Generate and save markdown report
    $markdown = ConvertTo-MarkdownReport -Report $report
    try {
        $markdown | Set-Content -Path $mdPath -Encoding UTF8
        Write-Host "  Markdown report: $mdPath" -ForegroundColor Green
    }
    catch {
        Write-Host "  Warning: Could not save markdown report: $_" -ForegroundColor Yellow
    }

    # Also write to root for easy access
    $rootMdPath = Join-Path $script:RalphDir "overnight_report.md"
    try {
        $markdown | Set-Content -Path $rootMdPath -Encoding UTF8
    }
    catch {}

    Write-Host ""

    return $report
}

function ConvertTo-MarkdownReport {
    <#
    .SYNOPSIS
        Convert report hashtable to markdown format
    .PARAMETER Report
        Report hashtable from Generate-GracefulStopReport
    .RETURNS
        Markdown string
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Report
    )

    $md = @()

    $md += "# Ralph Overnight Report"
    $md += ""
    $md += "Generated: $($Report.generatedAt)"
    $md += "Triggered by: $($Report.triggeredBy)"
    $md += ""

    $md += "## Summary"
    $md += ""
    $md += "| Metric | Value |"
    $md += "|--------|-------|"
    $md += "| Duration | $($Report.summary.duration) |"
    $md += "| Mode | $($Report.summary.mode) |"
    $md += "| Total Sprints | $($Report.summary.totalSprints) |"
    $md += "| Stories Completed | $($Report.summary.storiesCompleted)/$($Report.summary.totalStories) |"
    $md += "| Stories Decomposed | $($Report.summary.storiesDecomposed) |"
    $md += ""

    if ($Report.sprints -and $Report.sprints.Count -gt 0) {
        $md += "## Sprints"
        $md += ""
        $md += "| # | Focus Area | Status | Stories |"
        $md += "|---|------------|--------|---------|"
        foreach ($sprint in $Report.sprints) {
            $stories = "$($sprint.storiesCompleted)/$($sprint.storiesTotal)"
            $md += "| $($sprint.number) | $($sprint.focusArea) | $($sprint.status) | $stories |"
        }
        $md += ""
    }

    if ($Report.hardStories -and $Report.hardStories.Count -gt 0) {
        $md += "## Hard Stories"
        $md += ""
        foreach ($hs in $Report.hardStories) {
            $md += "### $($hs.id)"
            $md += "- **Title:** $($hs.title)"
            $md += "- **Focus Area:** $($hs.focusArea)"
            $md += "- **Reason:** $($hs.reason)"
            $md += "- **Failures:** $($hs.failureCount)"
            $md += "- **Status:** $($hs.status)"
            $md += ""
        }
    }

    if ($Report.errors -and $Report.errors.Count -gt 0) {
        $md += "## Errors"
        $md += ""
        foreach ($err in $Report.errors) {
            $md += "- [$($err.event)] $($err.message)"
        }
        $md += ""
    }

    $md += "## Recommendations"
    $md += ""
    foreach ($rec in $Report.recommendations) {
        $md += "- $rec"
    }
    $md += ""

    $md += "---"
    $md += ""
    $md += "*Report generated by Ralph Loop*"

    return $md -join "`n"
}

# ============================================================================
# CRASH RECOVERY (Phase 6 - P2)
# ============================================================================

function Get-CrashRecoveryFile {
    <#
    .SYNOPSIS
        Get the path to the crash recovery file
    #>
    if ($script:Paths) {
        return $script:Paths.CrashRecoveryFile
    } else {
        return (Join-Path $script:RalphDir "state\crash_recovery.json")
    }
}

function Save-CrashRecoveryState {
    <#
    .SYNOPSIS
        Save current state for crash recovery
    .DESCRIPTION
        Called periodically and before operations that might crash.
        Allows auto-resume from last known good state.
    #>

    $crashFile = Get-CrashRecoveryFile

    $recoveryState = @{
        savedAt = (Get-Date).ToString("o")
        sessionId = $script:State.SessionId
        lastSprint = if ($script:State.CurrentSprintNumber) { $script:State.CurrentSprintNumber } else { 0 }
        lastStory = if ($script:State.CurrentStoryId) { $script:State.CurrentStoryId } else { "" }
        lastFocusArea = if ($script:State.CurrentFocusArea) { $script:State.CurrentFocusArea } else { "" }
        iterationCount = $script:State.IterationCount
        mode = $script:State.CurrentMode
        nextStory = ""
    }

    # Get next story
    try {
        $status = Get-SprintStatus
        if ($status -and $status.nextStory) {
            $recoveryState.nextStory = $status.nextStory.id
        }
    }
    catch {}

    try {
        $recoveryState | ConvertTo-Json -Depth 5 | Set-Content $crashFile -Encoding UTF8
    }
    catch {
        Write-Host "  Warning: Could not save crash recovery state: $_" -ForegroundColor Yellow
    }
}

function Test-CrashRecovery {
    <#
    .SYNOPSIS
        Check if crash recovery is available
    .RETURNS
        Recovery state object or $null
    #>

    $crashFile = Get-CrashRecoveryFile

    if (-not (Test-Path $crashFile)) {
        return $null
    }

    try {
        $recovery = Get-Content $crashFile -Raw | ConvertFrom-Json

        # Check if recovery is recent (within 24 hours)
        $savedAt = [datetime]$recovery.savedAt
        $age = (Get-Date) - $savedAt

        if ($age.TotalHours -gt 24) {
            # Stale recovery file
            Remove-Item $crashFile -Force -ErrorAction SilentlyContinue
            return $null
        }

        return $recovery
    }
    catch {
        return $null
    }
}

function Invoke-CrashRecovery {
    <#
    .SYNOPSIS
        Recover from a crash and resume
    .RETURNS
        Hashtable with Resume ($true/$false) and recovery details
    #>

    $recovery = Test-CrashRecovery

    if (-not $recovery) {
        return @{ Resume = $false }
    }

    Write-Host ""
    Write-Host "  *** CRASH RECOVERY DETECTED ***" -ForegroundColor Red
    Write-Host "  Previous session crashed at: $($recovery.savedAt)" -ForegroundColor Yellow
    Write-Host "  Last completed: Sprint $($recovery.lastSprint), Story $($recovery.lastStory)" -ForegroundColor Yellow
    Write-Host "  Mode: $($recovery.mode)" -ForegroundColor DarkGray
    Write-Host ""

    # Generate partial report for the crashed session
    Generate-PartialReport -Recovery $recovery

    # Log the crash recovery
    Write-SessionLog -Event "crash_recovery" -Message "Recovering from previous session crash" -Data @{
        previousSession = $recovery.sessionId
        lastSprint = $recovery.lastSprint
        lastStory = $recovery.lastStory
    }

    Write-Host "  Auto-resuming from last known good state..." -ForegroundColor Green
    Write-Host ""

    return @{
        Resume = $true
        FromSprint = $recovery.lastSprint
        FromStory = $recovery.nextStory
        Mode = $recovery.mode
        PreviousSession = $recovery.sessionId
    }
}

function Generate-PartialReport {
    <#
    .SYNOPSIS
        Generate a partial report for an incomplete/crashed session
    .PARAMETER Recovery
        Recovery state object
    .PARAMETER Emergency
        If true, this is an emergency report during crash
    #>
    param(
        [object]$Recovery = $null,
        [switch]$Emergency
    )

    $reportDir = if ($script:ArchiveDir) { $script:ArchiveDir }
                 else { Join-Path $script:RalphDir "archive" }

    if (-not (Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $reportPath = Join-Path $reportDir "partial_report_$timestamp.md"

    $md = @()
    $md += "# Ralph Partial Report (Crash Recovery)"
    $md += ""
    $md += "**Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    $md += "**Type:** $(if ($Emergency) { 'Emergency (mid-crash)' } else { 'Recovery analysis' })"
    $md += ""

    if ($Recovery) {
        $md += "## Previous Session State"
        $md += ""
        $md += "| Field | Value |"
        $md += "|-------|-------|"
        $md += "| Session ID | $($Recovery.sessionId) |"
        $md += "| Crashed At | $($Recovery.savedAt) |"
        $md += "| Mode | $($Recovery.mode) |"
        $md += "| Last Sprint | $($Recovery.lastSprint) |"
        $md += "| Last Story | $($Recovery.lastStory) |"
        $md += "| Next Story | $($Recovery.nextStory) |"
        $md += "| Iterations | $($Recovery.iterationCount) |"
        $md += ""
    }

    # Try to get sprint status
    try {
        $status = Get-SprintStatus
        if ($status) {
            $md += "## Current Sprint Status"
            $md += ""
            $md += "- **Focus Area:** $($status.focusArea)"
            $md += "- **Stories Passed:** $($status.passed)"
            $md += "- **Stories Remaining:** $($status.failed)"
            $md += "- **Complete:** $($status.complete)"
            $md += ""
        }
    }
    catch {}

    # Get hard stories
    try {
        $hardStories = Get-HardStoriesThisSession
        if ($hardStories -and $hardStories.Count -gt 0) {
            $md += "## Hard Stories (This Session)"
            $md += ""
            foreach ($hs in $hardStories) {
                $md += "- **$($hs.id):** $($hs.title) - $($hs.reason)"
            }
            $md += ""
        }
    }
    catch {}

    $md += "---"
    $md += ""
    $md += "*Partial report generated due to session interruption*"

    try {
        ($md -join "`n") | Set-Content $reportPath -Encoding UTF8
        Write-Host "  Partial report saved: $reportPath" -ForegroundColor DarkGray
    }
    catch {
        Write-Host "  Warning: Could not save partial report: $_" -ForegroundColor Yellow
    }

    return $reportPath
}

function Clear-CrashRecovery {
    <#
    .SYNOPSIS
        Clear crash recovery file after successful session
    #>
    $crashFile = Get-CrashRecoveryFile
    if (Test-Path $crashFile) {
        Remove-Item $crashFile -Force -ErrorAction SilentlyContinue
    }
}
