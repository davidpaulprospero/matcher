# scripts/ralph/lib/scoring.ps1
# Ralph's Choice: focus area scoring, git activity analysis, story optimization

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
