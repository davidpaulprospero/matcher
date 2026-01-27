# scripts/ralph/lib/sprint.ps1
# Sprint lifecycle: PRD generation, archive, history, learning, dependencies

function Save-StateFile {
    <#
    .SYNOPSIS
        Atomic write: data -> temp file -> rename to target path
    .PARAMETER Path
        Target file path
    .PARAMETER Data
        Object to serialize as JSON
    .PARAMETER Depth
        ConvertTo-Json depth (default 10)
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][object]$Data,
        [int]$Depth = 10
    )
    $tempPath = "$Path.tmp"
    $Data | ConvertTo-Json -Depth $Depth | Set-Content -Path $tempPath -Encoding UTF8
    Move-Item -Path $tempPath -Destination $Path -Force
}

function Get-Sprint {
    <#
    .SYNOPSIS
        Load sprint/PRD data from JSON file
    #>
    param([string]$Path = $script:PrdFile)
    if (-not (Test-Path $Path)) { return $null }
    try {
        Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Warning "Failed to parse sprint file: $Path"
        return $null
    }
}

function Save-Sprint {
    <#
    .SYNOPSIS
        Save sprint/PRD data atomically
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint,
        [string]$Path = $script:PrdFile
    )
    Save-StateFile -Path $Path -Data $Sprint
}

function Update-StoryStatus {
    <#
    .SYNOPSIS
        Update a story's pass/fail status and notes in the sprint file
    #>
    param(
        [Parameter(Mandatory)][string]$StoryId,
        [bool]$Passes = $false,
        [string]$Notes = '',
        [string]$Path = $script:PrdFile
    )
    $sprint = Get-Sprint -Path $Path
    if (-not $sprint) { return }
    foreach ($story in $sprint.userStories) {
        if ($story.id -eq $StoryId) {
            $story.passes = $Passes
            if ($Notes) { $story.notes = $Notes }
        }
    }
    Save-Sprint -Sprint $sprint -Path $Path
}

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

    # Atomic write: temp file -> rename (no BOM via .NET)
    $tempPath = "$Path.tmp"
    [System.IO.File]::WriteAllText($tempPath, $Content)
    Move-Item -Path $tempPath -Destination $Path -Force
}

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
