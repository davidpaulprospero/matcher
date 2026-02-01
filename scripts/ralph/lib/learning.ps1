# Ralph Learning System
# Comprehensive learning capture and auto-application

# ============================================================================
# ERROR PATTERN ANALYSIS
# ============================================================================

function Get-RecurringErrors {
    <#
    .SYNOPSIS
        Analyze error patterns from a sprint's failed stories
    .PARAMETER Sprint
        Sprint object with userStories array
    .RETURNS
        Array of error pattern objects with type, count, examples
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint
    )

    $errorPatterns = @{}

    foreach ($story in $Sprint.userStories) {
        if ($story.passes -eq $false -and $story.notes) {
            # Categorize error type from notes
            $errorType = "unknown"
            $notes = $story.notes

            if ($notes -match "import|ImportError|ModuleNotFound") {
                $errorType = "ImportError"
            }
            elseif ($notes -match "timeout|TimedOut|exceeded") {
                $errorType = "Timeout"
            }
            elseif ($notes -match "test|pytest|assertion|AssertionError") {
                $errorType = "TestFailure"
            }
            elseif ($notes -match "syntax|SyntaxError|parse") {
                $errorType = "SyntaxError"
            }
            elseif ($notes -match "health.check|T1|T2|T3") {
                $errorType = "HealthCheckFailure"
            }
            elseif ($notes -match "exit.code") {
                $errorType = "ExitCodeError"
            }

            if (-not $errorPatterns[$errorType]) {
                $errorPatterns[$errorType] = @{
                    type = $errorType
                    count = 0
                    examples = @()
                    storyIds = @()
                }
            }

            $errorPatterns[$errorType].count++
            $errorPatterns[$errorType].storyIds += $story.id

            # Keep first 3 examples
            if ($errorPatterns[$errorType].examples.Count -lt 3) {
                $errorPatterns[$errorType].examples += $notes.Substring(0, [Math]::Min(100, $notes.Length))
            }
        }
    }

    return @($errorPatterns.Values)
}

# ============================================================================
# PROMPT EFFECTIVENESS
# ============================================================================

function Get-PromptMetrics {
    <#
    .SYNOPSIS
        Analyze prompt effectiveness metrics from sprint
    .PARAMETER Sprint
        Sprint object with userStories array
    .RETURNS
        Hashtable with prompt effectiveness metrics
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint
    )

    $metrics = @{
        totalStories = $Sprint.userStories.Count
        passedStories = 0
        failedStories = 0
        avgTitleLength = 0
        avgCriteriaCount = 0
        verbUsageSuccess = @{}
    }

    # Common action verbs in story titles
    $actionVerbs = @("Add", "Fix", "Create", "Update", "Implement", "Remove", "Refactor", "Improve", "Enable", "Configure")

    $titleLengths = @()
    $criteriaCounts = @()

    foreach ($story in $Sprint.userStories) {
        if ($story.passes -eq $true) {
            $metrics.passedStories++
        }
        else {
            $metrics.failedStories++
        }

        # Title length
        if ($story.title) {
            $titleLengths += $story.title.Length
        }

        # Criteria count
        if ($story.acceptanceCriteria) {
            $criteriaCounts += @($story.acceptanceCriteria).Count
        }

        # Verb usage analysis
        foreach ($verb in $actionVerbs) {
            if ($story.title -match "^$verb\s") {
                if (-not $metrics.verbUsageSuccess[$verb]) {
                    $metrics.verbUsageSuccess[$verb] = @{ total = 0; passed = 0 }
                }
                $metrics.verbUsageSuccess[$verb].total++
                if ($story.passes -eq $true) {
                    $metrics.verbUsageSuccess[$verb].passed++
                }
            }
        }
    }

    if ($titleLengths.Count -gt 0) {
        $metrics.avgTitleLength = [math]::Round(($titleLengths | Measure-Object -Average).Average, 1)
    }

    if ($criteriaCounts.Count -gt 0) {
        $metrics.avgCriteriaCount = [math]::Round(($criteriaCounts | Measure-Object -Average).Average, 1)
    }

    $metrics.successRate = if ($metrics.totalStories -gt 0) {
        [math]::Round($metrics.passedStories / $metrics.totalStories, 2)
    } else { 0 }

    return $metrics
}

# ============================================================================
# FILE CO-CHANGE ANALYSIS
# ============================================================================

function Get-CoChangedFiles {
    <#
    .SYNOPSIS
        Analyze which files are often changed together in successful stories
    .PARAMETER Sprint
        Sprint object with userStories array
    .PARAMETER GitLogDays
        Number of days of git history to analyze (default: 7)
    .RETURNS
        Array of file group objects
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint,
        [int]$GitLogDays = 7
    )

    $fileGroups = @()

    try {
        # Get recent commits
        $since = (Get-Date).AddDays(-$GitLogDays).ToString("yyyy-MM-dd")
        $commits = git log --since="$since" --name-only --pretty=format:"COMMIT:%H" 2>$null

        if (-not $commits) {
            return @()
        }

        $commitFiles = @{}
        $currentCommit = ""

        foreach ($line in $commits) {
            if ($line -match "^COMMIT:(.+)$") {
                $currentCommit = $Matches[1]
                $commitFiles[$currentCommit] = @()
            }
            elseif (-not [string]::IsNullOrWhiteSpace($line) -and $currentCommit) {
                $commitFiles[$currentCommit] += $line
            }
        }

        # Find files that appear together frequently
        $filePairs = @{}

        foreach ($commit in $commitFiles.Keys) {
            $files = $commitFiles[$commit]
            for ($i = 0; $i -lt $files.Count - 1; $i++) {
                for ($j = $i + 1; $j -lt $files.Count; $j++) {
                    $pair = @($files[$i], $files[$j]) | Sort-Object
                    $pairKey = $pair -join "|"
                    if (-not $filePairs[$pairKey]) {
                        $filePairs[$pairKey] = 0
                    }
                    $filePairs[$pairKey]++
                }
            }
        }

        # Return top 5 most common pairs
        $topPairs = $filePairs.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 5

        foreach ($pair in $topPairs) {
            $files = $pair.Key -split "\|"
            $fileGroups += @{
                files = $files
                coChangeCount = $pair.Value
            }
        }
    }
    catch {
        # Git analysis failed, return empty
    }

    return $fileGroups
}

# ============================================================================
# STORY ANALYSIS
# ============================================================================

function Get-StoryAnalysis {
    <#
    .SYNOPSIS
        Analyze story success patterns by various factors
    .PARAMETER Sprint
        Sprint object with userStories array
    .RETURNS
        Hashtable with story analysis metrics
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint
    )

    $analysis = @{
        avgAcceptanceCriteria = 0
        successByCriteriaCount = @{
            "1-3" = 0
            "4-6" = 0
            "7+" = 0
        }
        failureByCriteriaCount = @{
            "1-3" = 0
            "4-6" = 0
            "7+" = 0
        }
        titlesWithActionVerbs = @{
            passed = 0
            failed = 0
        }
    }

    $criteriaCountsAll = @()

    foreach ($story in $Sprint.userStories) {
        $criteriaCount = if ($story.acceptanceCriteria) { @($story.acceptanceCriteria).Count } else { 0 }
        $criteriaCountsAll += $criteriaCount

        # Categorize by criteria count
        $bucket = if ($criteriaCount -le 3) { "1-3" }
                  elseif ($criteriaCount -le 6) { "4-6" }
                  else { "7+" }

        if ($story.passes -eq $true) {
            $analysis.successByCriteriaCount[$bucket]++
        }
        else {
            $analysis.failureByCriteriaCount[$bucket]++
        }

        # Check for action verbs in title
        if ($story.title -match "^(Add|Fix|Create|Update|Implement|Remove|Refactor)\s") {
            if ($story.passes -eq $true) {
                $analysis.titlesWithActionVerbs.passed++
            }
            else {
                $analysis.titlesWithActionVerbs.failed++
            }
        }
    }

    if ($criteriaCountsAll.Count -gt 0) {
        $analysis.avgAcceptanceCriteria = [math]::Round(($criteriaCountsAll | Measure-Object -Average).Average, 1)
    }

    return $analysis
}

# ============================================================================
# AUTO-APPLY LEARNINGS
# ============================================================================

function Get-LearningAdjustments {
    <#
    .SYNOPSIS
        Determine config adjustments based on learning data
    .PARAMETER StoryAnalysis
        Story analysis metrics
    .PARAMETER ErrorPatterns
        Error pattern analysis
    .PARAMETER PromptMetrics
        Prompt effectiveness metrics
    .RETURNS
        Array of adjustment description strings
    #>
    param(
        [hashtable]$StoryAnalysis,
        [array]$ErrorPatterns,
        [hashtable]$PromptMetrics
    )

    $adjustments = @()

    # Check criteria count effectiveness
    if ($StoryAnalysis) {
        $success13 = $StoryAnalysis.successByCriteriaCount["1-3"]
        $success46 = $StoryAnalysis.successByCriteriaCount["4-6"]
        $success7plus = $StoryAnalysis.successByCriteriaCount["7+"]

        if ($success13 -gt $success46 -and $success13 -gt $success7plus) {
            $adjustments += "Stories with 1-3 acceptance criteria have highest success rate - recommend keeping stories simple"
        }
        elseif ($success46 -gt $success13 -and $success46 -gt $success7plus) {
            $adjustments += "Stories with 4-6 acceptance criteria have highest success rate - medium complexity works well"
        }
    }

    # Check error patterns
    if ($ErrorPatterns) {
        $importErrors = $ErrorPatterns | Where-Object { $_.type -eq "ImportError" }
        if ($importErrors -and $importErrors.count -gt 2) {
            $adjustments += "Recurring import errors ($($importErrors.count)) - consider adding stricter pre-flight import validation"
        }

        $timeoutErrors = $ErrorPatterns | Where-Object { $_.type -eq "Timeout" }
        if ($timeoutErrors -and $timeoutErrors.count -gt 2) {
            $adjustments += "Recurring timeouts ($($timeoutErrors.count)) - stories may be too complex"
        }

        $testFailures = $ErrorPatterns | Where-Object { $_.type -eq "TestFailure" }
        if ($testFailures -and $testFailures.count -gt 2) {
            $adjustments += "Recurring test failures ($($testFailures.count)) - consider more focused acceptance criteria"
        }
    }

    # Check prompt effectiveness
    if ($PromptMetrics -and $PromptMetrics.successRate -lt 0.5) {
        $adjustments += "Low overall success rate ($([math]::Round($PromptMetrics.successRate * 100))%) - consider reviewing story generation prompts"
    }

    return $adjustments
}

# ============================================================================
# MAIN LEARNING CAPTURE
# ============================================================================

function Save-SprintLearning {
    <#
    .SYNOPSIS
        Capture comprehensive learning from a completed sprint
    .DESCRIPTION
        Analyzes sprint results and captures 4 types of learning:
        1. Story types that succeed/fail
        2. Error patterns
        3. Prompt structures that worked
        4. Files often changed together
        Also auto-applies adjustments with logging.
    .PARAMETER Sprint
        Completed sprint object
    .PARAMETER FocusArea
        Focus area of the sprint
    .PARAMETER SprintNumber
        Sprint number (optional)
    #>
    param(
        [Parameter(Mandatory)][object]$Sprint,
        [string]$FocusArea = "",
        [int]$SprintNumber = 0
    )

    if (-not $Sprint -or -not $Sprint.userStories -or $Sprint.userStories.Count -eq 0) {
        Write-Host "  No stories to analyze for learning" -ForegroundColor DarkGray
        return
    }

    Write-Host ""
    Write-Host "  Capturing learning from sprint..." -ForegroundColor Cyan

    # 1. Story analysis
    $storyAnalysis = Get-StoryAnalysis -Sprint $Sprint

    # 2. Error patterns
    $errorPatterns = Get-RecurringErrors -Sprint $Sprint

    # 3. Prompt effectiveness
    $promptMetrics = Get-PromptMetrics -Sprint $Sprint

    # 4. Co-changed files
    $fileGroups = Get-CoChangedFiles -Sprint $Sprint

    # Build learning entry
    $learning = @{
        timestamp = (Get-Date).ToString("o")
        focusArea = $FocusArea
        sprintNumber = $SprintNumber
        storyAnalysis = $storyAnalysis
        errorPatterns = $errorPatterns
        promptEffectiveness = $promptMetrics
        fileGroups = $fileGroups
        adjustmentsApplied = @()
    }

    # Get and apply adjustments
    $adjustments = Get-LearningAdjustments -StoryAnalysis $storyAnalysis -ErrorPatterns $errorPatterns -PromptMetrics $promptMetrics
    $learning.adjustmentsApplied = $adjustments

    # Log what was learned
    if ($adjustments.Count -gt 0) {
        Write-Host "  Learning captured for $FocusArea :" -ForegroundColor Cyan
        foreach ($adj in $adjustments) {
            Write-Host "    - $adj" -ForegroundColor Green
        }
    }
    else {
        Write-Host "  Sprint completed with no notable patterns to learn" -ForegroundColor DarkGray
    }

    # Save to learning database
    Update-LearningDb -Entry $learning

    # Write to session log if available
    if (Get-Command Write-SessionLog -ErrorAction SilentlyContinue) {
        Write-SessionLog -Event "learning_captured" -Message "Sprint learning saved" -Data @{
            focusArea = $FocusArea
            adjustmentsCount = $adjustments.Count
            errorPatternsCount = $errorPatterns.Count
        }
    }

    Write-Host ""
}

function Get-LearningInsights {
    <#
    .SYNOPSIS
        Get aggregated insights from learning database
    .PARAMETER FocusArea
        Optional focus area to filter by
    .PARAMETER LastN
        Number of recent entries to analyze (default: 20)
    .RETURNS
        Hashtable with aggregated insights
    #>
    param(
        [string]$FocusArea = "",
        [int]$LastN = 20
    )

    $dbFile = if ($script:Paths) { $script:Paths.LearningDbFile } else { Join-Path $script:RalphDir "state\learning_db.json" }
    $db = Read-JsonFile -Path $dbFile
    if (-not $db -or -not $db.entries -or $db.entries.Count -eq 0) {
        return @{ hasData = $false }
    }

    # Filter and get recent entries
    $entries = $db.entries
    if ($FocusArea) {
        $entries = $entries | Where-Object { $_.focusArea -eq $FocusArea }
    }
    $entries = @($entries | Select-Object -Last $LastN)

    if ($entries.Count -eq 0) {
        return @{ hasData = $false }
    }

    $insights = @{
        hasData = $true
        entriesAnalyzed = $entries.Count
        focusArea = $FocusArea
        commonErrorTypes = @{}
        avgSuccessRate = 0
        successRates = @()
        topAdjustments = @{}
    }

    foreach ($entry in $entries) {
        # Aggregate error types
        if ($entry.errorPatterns) {
            foreach ($pattern in $entry.errorPatterns) {
                if ($pattern.type) {
                    if (-not $insights.commonErrorTypes[$pattern.type]) {
                        $insights.commonErrorTypes[$pattern.type] = 0
                    }
                    $insights.commonErrorTypes[$pattern.type] += $pattern.count
                }
            }
        }

        # Track success rates
        if ($entry.promptEffectiveness -and $entry.promptEffectiveness.successRate) {
            $insights.successRates += $entry.promptEffectiveness.successRate
        }

        # Count adjustments
        if ($entry.adjustmentsApplied) {
            foreach ($adj in $entry.adjustmentsApplied) {
                # Simplify adjustment text for grouping
                $key = if ($adj -match "^([^-]+)") { $Matches[1].Trim() } else { $adj }
                if (-not $insights.topAdjustments[$key]) {
                    $insights.topAdjustments[$key] = 0
                }
                $insights.topAdjustments[$key]++
            }
        }
    }

    # Calculate average success rate
    if ($insights.successRates.Count -gt 0) {
        $insights.avgSuccessRate = [math]::Round(($insights.successRates | Measure-Object -Average).Average, 2)
    }

    return $insights
}
