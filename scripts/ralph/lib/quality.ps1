# scripts/ralph/lib/quality.ps1
# Quality gates: review, baselines, regression, feedback, rollback

function Invoke-QualityReview {
    <#
    .SYNOPSIS
        Invoke a separate Claude session to review story quality (Story 1.1)
    .DESCRIPTION
        When llmAsJudgeQuality flag is enabled, spawns a read-only Claude session
        that reviews the git diff against acceptance criteria and returns a score.
    .PARAMETER StoryId
        The story ID being reviewed
    .PARAMETER Story
        The story object from PRD (with acceptanceCriteria)
    .PARAMETER DiffOutput
        The git diff to review
    .RETURNS
        Hashtable with score, passed, and review details; or $null if disabled/failed
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$DiffOutput
    )

    $config = Get-RalphConfig

    # Check if review is enabled via flag AND/OR config
    $flagEnabled = $config.flags.llmAsJudgeQuality -eq $true
    $reviewEnabled = $config.review -and $config.review.enabled -eq $true
    if (-not $flagEnabled -and -not $reviewEnabled) {
        return $null
    }

    Write-Host "  Quality review starting..." -ForegroundColor Cyan

    $reviewPromptPath = Join-Path $script:RalphDir "review-prompt.md"
    if (-not (Test-Path $reviewPromptPath)) {
        Write-Host "  Warning: review-prompt.md not found, skipping review" -ForegroundColor Yellow
        return $null
    }

    $reviewInstructions = Get-Content $reviewPromptPath -Raw

    # Build criteria list
    $criteriaText = "No acceptance criteria found."
    if ($Story -and $Story.acceptanceCriteria) {
        $criteriaText = ($Story.acceptanceCriteria | ForEach-Object { "- $_" }) -join "`n"
    }

    # Truncate diff if too large (keep first 3000 lines)
    $diffLines = ($DiffOutput -split "`n")
    $truncated = ""
    if ($diffLines.Count -gt 3000) {
        $DiffOutput = ($diffLines[0..2999] -join "`n")
        $truncated = "`n[TRUNCATED: showing first 3000 of $($diffLines.Count) lines]"
    }

    # Build the review request
    $reviewRequest = @"
$reviewInstructions

## Story Being Reviewed

**ID:** $StoryId
**Title:** $(if ($Story) { $Story.title } else { "Unknown" })

### Acceptance Criteria
$criteriaText

### Git Diff
$DiffOutput$truncated

Please output your review as JSON.
"@

    # Save review prompt for debugging
    $tempPromptFile = Join-Path $script:SessionLogDir "review_prompt_$StoryId.txt"
    $reviewRequest | Set-Content $tempPromptFile -Encoding UTF8

    # Get review settings
    $timeout = if ($config.review -and $config.review.timeout) { $config.review.timeout } else { 180 }
    $minScore = if ($config.review -and $config.review.minScoreToPass) { $config.review.minScoreToPass } else { 6 }
    $model = if ($config.review -and $config.review.model) { $config.review.model } else { "sonnet" }

    try {
        $claudePath = Get-ClaudePath

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model $model -p `"$reviewRequest`""
        $psi.RedirectStandardInput = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WorkingDirectory = $script:ProjectRoot

        $process = [System.Diagnostics.Process]::Start($psi)
        $exited = $process.WaitForExit($timeout * 1000)
        $output = $process.StandardOutput.ReadToEnd()

        if (-not $exited) {
            $process.Kill()
            Write-Host "  Review timed out after ${timeout}s" -ForegroundColor Yellow
            return $null
        }

        # Try to parse JSON from output
        $jsonMatch = [regex]::Match($output, '\{[\s\S]*?"overallScore"[\s\S]*?\}')
        if ($jsonMatch.Success) {
            try {
                $review = $jsonMatch.Value | ConvertFrom-Json
            }
            catch {
                Write-Host "  Could not parse review JSON" -ForegroundColor Yellow
                $rawFile = Join-Path $script:SessionLogDir "review_${StoryId}_raw.txt"
                $output | Set-Content $rawFile -Encoding UTF8
                return $null
            }

            $score = [int]$review.overallScore
            $passed = $score -ge $minScore
            $recommendation = if ($review.recommendation) { $review.recommendation } else { if ($passed) { "pass" } else { "revise" } }

            # Log review
            $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"
            $reviewData = @{
                storyId = $StoryId
                reviewedAt = (Get-Date).ToString("o")
                overallScore = $score
                scores = $review.scores
                criteriaResults = $review.criteriaResults
                issues = $review.issues
                recommendation = $recommendation
                passed = $passed
                minScoreRequired = $minScore
            }
            Write-JsonNoBom -Path $reviewFile -Content ($reviewData | ConvertTo-Json -Depth 5)

            $scoreColor = if ($passed) { "Green" } elseif ($score -ge 4) { "Yellow" } else { "Red" }
            Write-Host "  Review score: $score/10 ($recommendation)" -ForegroundColor $scoreColor

            if ($review.issues) {
                $errorCount = @($review.issues | Where-Object { $_.severity -eq "error" }).Count
                $warnCount = @($review.issues | Where-Object { $_.severity -eq "warning" }).Count
                if ($errorCount -gt 0) { Write-Host "    Errors: $errorCount" -ForegroundColor Red }
                if ($warnCount -gt 0) { Write-Host "    Warnings: $warnCount" -ForegroundColor Yellow }
            }

            Append-SessionTimeline -Event "quality_review" -Data @{
                storyId = $StoryId
                score = $score
                passed = $passed
                recommendation = $recommendation
            }

            return @{
                score = $score
                passed = $passed
                recommendation = $recommendation
                review = $reviewData
            }
        }
        else {
            Write-Host "  Could not find review JSON in output" -ForegroundColor Yellow
            $rawFile = Join-Path $script:SessionLogDir "review_${StoryId}_raw.txt"
            $output | Set-Content $rawFile -Encoding UTF8
            return $null
        }
    }
    catch {
        Write-Host "  Review failed: $_" -ForegroundColor Yellow
        return $null
    }
}

function Get-StoryFailureContext {
    <#
    .SYNOPSIS
        Build failure context for retry prompts (Story 1.3)
    .DESCRIPTION
        When a story is being retried, extracts the error category and
        last N lines from the previous attempt's output to help Claude
        understand what went wrong.
    .PARAMETER StoryId
        The story being retried
    .PARAMETER RetryCount
        Current retry attempt number
    .RETURNS
        String with failure context to prepend to prompt, or empty string
    #>
    param(
        [string]$StoryId,
        [int]$RetryCount
    )

    if ($RetryCount -le 1) {
        return ""
    }

    $context = @()
    $context += "RETRY CONTEXT (attempt $RetryCount):"
    $context += "This story has been attempted before and failed."

    # Check for last output file
    $prevIteration = $script:IterationCount  # Current iteration (we're building prompt for next)
    $outFile = Join-Path $script:SessionLogDir "claude_stdout_$prevIteration.txt"
    $errFile = Join-Path $script:SessionLogDir "claude_stderr_$prevIteration.txt"

    $lastOutput = ""
    if (Test-Path $outFile) {
        $lastOutput = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
    }
    if (Test-Path $errFile) {
        $lastOutput += "`n" + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
    }

    # Get last 50 lines of previous output
    if ($lastOutput) {
        $lines = ($lastOutput -split "`n") | Where-Object { $_.Trim() }
        $tailLines = if ($lines.Count -gt 50) { $lines[-50..-1] } else { $lines }
        $tailText = ($tailLines -join "`n").Trim()
        if ($tailText) {
            $context += ""
            $context += "Last 50 lines from previous attempt:"
            $context += "---"
            $context += $tailText
            $context += "---"
        }
    }

    # Check for previous error category in metrics
    if (Test-Path $script:MetricsFile) {
        try {
            $metrics = Import-Csv $script:MetricsFile -ErrorAction SilentlyContinue
            $storyMetrics = $metrics | Where-Object { $_.story_id -eq $StoryId -and $_.success -eq "false" } | Select-Object -Last 1
            if ($storyMetrics -and $storyMetrics.error_category) {
                $context += "Previous error category: $($storyMetrics.error_category)"
            }
        }
        catch {}
    }

    # Check for previous verification logs
    $prevVerificationFile = Join-Path $script:SessionLogDir "story_${StoryId}_verification.json"
    if (Test-Path $prevVerificationFile) {
        try {
            $prevVerification = Get-Content $prevVerificationFile -Raw | ConvertFrom-Json
            if ($prevVerification.acceptanceCriteria) {
                $unmet = @($prevVerification.acceptanceCriteria | Where-Object { -not $_.verified })
                if ($unmet.Count -gt 0) {
                    $context += ""
                    $context += "Unmet criteria from previous attempt:"
                    foreach ($c in $unmet) {
                        $context += "  - $($c.criterion)"
                    }
                }
            }
        }
        catch {}
    }

    # Check for review feedback
    $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"
    if (Test-Path $reviewFile) {
        try {
            $review = Get-Content $reviewFile -Raw | ConvertFrom-Json
            if ($review.issues) {
                $context += ""
                $context += "Issues from quality review (score: $($review.overallScore)/10):"
                foreach ($issue in $review.issues) {
                    $context += "  [$($issue.severity)] $($issue.description)"
                }
            }
        }
        catch {}
    }

    $context += ""
    $context += "Please address these issues in this attempt."
    $context += ""

    return ($context -join "`n")
}

function Get-DiffQualityScore {
    <#
    .SYNOPSIS
        Compute quality metrics from git diff (Story 1.4)
    .DESCRIPTION
        Analyzes the git diff to compute testRatio, churnRisk, and sizeAppropriate.
    .PARAMETER DiffOutput
        Raw git diff output
    .RETURNS
        Hashtable with quality metrics
    #>
    param(
        [string]$DiffOutput
    )

    $result = @{
        testRatio = 0.0
        churnRisk = "low"
        sizeAppropriate = $true
        totalLinesChanged = 0
        testLinesChanged = 0
        implLinesChanged = 0
        filesChanged = 0
        warnings = @()
    }

    if (-not $DiffOutput) {
        return $result
    }

    $config = Get-RalphConfig
    $maxDiffLines = if ($config.quality -and $config.quality.maxDiffLines) { $config.quality.maxDiffLines } else { 800 }
    $minTestRatio = if ($config.quality -and $config.quality.minTestRatio) { $config.quality.minTestRatio } else { 0.2 }

    # Parse diff to count lines by category
    $testLines = 0
    $implLines = 0
    $currentFile = ""
    $filesChanged = 0

    foreach ($line in ($DiffOutput -split "`n")) {
        if ($line -match '^diff --git a/(.+) b/') {
            $currentFile = $Matches[1]
            $filesChanged++
        }
        elseif ($line -match '^\+[^+]' -or $line -match '^\-[^-]') {
            $isTest = $currentFile -match '(test_|\.Tests\.|_test\.|tests/|spec/)'
            if ($isTest) {
                $testLines++
            }
            else {
                $implLines++
            }
        }
    }

    $totalLines = $testLines + $implLines
    $result.totalLinesChanged = $totalLines
    $result.testLinesChanged = $testLines
    $result.implLinesChanged = $implLines
    $result.filesChanged = $filesChanged

    # Test ratio
    if ($implLines -gt 0) {
        $result.testRatio = [math]::Round($testLines / $implLines, 2)
    }
    elseif ($testLines -gt 0) {
        $result.testRatio = 1.0  # Pure test changes
    }

    # Size appropriateness
    if ($totalLines -gt $maxDiffLines) {
        $result.sizeAppropriate = $false
        $result.warnings += "Diff size ($totalLines lines) exceeds maximum ($maxDiffLines)"
    }

    # Churn risk
    if ($filesChanged -gt 10) {
        $result.churnRisk = "high"
        $result.warnings += "High file count ($filesChanged files changed)"
    }
    elseif ($filesChanged -gt 5) {
        $result.churnRisk = "medium"
    }

    # Test ratio warning
    if ($implLines -gt 20 -and $result.testRatio -lt $minTestRatio) {
        $result.warnings += "Low test ratio ($($result.testRatio)) for $implLines impl lines (minimum: $minTestRatio)"
    }

    return $result
}

function Get-TestBaseline {
    <#
    .SYNOPSIS
        Capture current test suite baseline (Story 1.5)
    .DESCRIPTION
        Runs pytest --collect-only and a quick test run to establish baseline.
        Saves to test_baseline.json.
    .RETURNS
        Hashtable with test counts, or $null on failure
    #>

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    try {
        # Quick test collection count
        $collectOutput = & python -m pytest tests/ --collect-only -q 2>&1
        $collectText = $collectOutput -join "`n"

        $totalTests = 0
        if ($collectText -match '(\d+)\s+tests?\s+collected') {
            $totalTests = [int]$Matches[1]
        }
        elseif ($collectText -match '(\d+)\s+item') {
            $totalTests = [int]$Matches[1]
        }

        # Quick test run for pass/fail baseline
        $testOutput = & python -m pytest tests/ -v --tb=no -q 2>&1
        $testText = $testOutput -join "`n"

        $passed = 0
        $failed = 0
        if ($testText -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }
        if ($testText -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }

        $baseline = @{
            capturedAt = (Get-Date).ToString("o")
            totalTests = $totalTests
            passed = $passed
            failed = $failed
            errors = 0
        }

        if ($testText -match '(\d+)\s+error') {
            $baseline.errors = [int]$Matches[1]
        }

        Write-JsonNoBom -Path $baselineFile -Content ($baseline | ConvertTo-Json -Depth 3)
        Write-Host "  Test baseline: $passed passed, $failed failed of $totalTests tests" -ForegroundColor DarkGray

        return $baseline
    }
    catch {
        Write-Host "  Could not capture test baseline: $_" -ForegroundColor Yellow
        return $null
    }
}

function Compare-TestBaseline {
    <#
    .SYNOPSIS
        Compare current test results against baseline (Story 1.5)
    .PARAMETER CurrentResults
        String like "41/41 pass" from Get-TestResults
    .RETURNS
        Hashtable with regression info, or $null if no baseline
    #>
    param(
        [string]$CurrentResults
    )

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    if (-not (Test-Path $baselineFile)) {
        return $null
    }

    $config = Get-RalphConfig
    $regressionEnabled = -not $config.regression -or $config.regression.enabled -ne $false

    if (-not $regressionEnabled) {
        return $null
    }

    try {
        $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }

    # Parse current results
    $currentPassed = 0
    $currentFailed = 0
    if ($CurrentResults -match '(\d+)/(\d+)') {
        $currentPassed = [int]$Matches[1]
    }
    if ($CurrentResults -match '(\d+)\s+fail') {
        $currentFailed = [int]$Matches[1]
    }

    $baselinePassed = [int]$baseline.passed
    $baselineFailed = [int]$baseline.failed

    $passedDelta = $currentPassed - $baselinePassed
    $failedDelta = $currentFailed - $baselineFailed

    $hasRegression = ($failedDelta -gt 0) -or ($passedDelta -lt 0 -and $baselinePassed -gt 0)

    $comparison = @{
        baselinePassed = $baselinePassed
        baselineFailed = $baselineFailed
        currentPassed = $currentPassed
        currentFailed = $currentFailed
        passedDelta = $passedDelta
        failedDelta = $failedDelta
        hasRegression = $hasRegression
        capturedAt = $baseline.capturedAt
    }

    if ($hasRegression) {
        Write-Host "  REGRESSION DETECTED: $baselinePassed -> $currentPassed passed, $baselineFailed -> $currentFailed failed" -ForegroundColor Red
    }
    else {
        $deltaStr = if ($passedDelta -gt 0) { " (+$passedDelta)" } else { "" }
        Write-Host "  Test baseline OK: $currentPassed passed$deltaStr" -ForegroundColor DarkGreen
    }

    return $comparison
}

function Update-TestBaseline {
    <#
    .SYNOPSIS
        Update test baseline after successful story (Story 1.5)
    .PARAMETER TestResults
        Current test results string from Get-TestResults
    #>
    param(
        [string]$TestResults
    )

    $baselineFile = Join-Path $script:RalphDir "test_baseline.json"

    if (-not $TestResults) { return }

    $passed = 0
    $failed = 0
    $total = 0

    if ($TestResults -match '(\d+)/(\d+)') {
        $passed = [int]$Matches[1]
        $total = [int]$Matches[2]
    }
    if ($TestResults -match '(\d+)\s+fail') {
        $failed = [int]$Matches[1]
    }

    if ($total -eq 0 -and $passed -eq 0) { return }

    $baseline = @{
        capturedAt = (Get-Date).ToString("o")
        totalTests = $total
        passed = $passed
        failed = $failed
        errors = 0
    }

    Write-JsonNoBom -Path $baselineFile -Content ($baseline | ConvertTo-Json -Depth 3)
}

function Import-HumanFeedback {
    <#
    .SYNOPSIS
        Read human feedback file (Story 1.6)
    .RETURNS
        Array of feedback entries, or empty array
    #>

    $feedbackFile = Join-Path $script:RalphDir "feedback.json"

    if (-not (Test-Path $feedbackFile)) {
        return @()
    }

    try {
        $data = Get-Content $feedbackFile -Raw | ConvertFrom-Json
        if ($data.entries) {
            return @($data.entries)
        }
        return @()
    }
    catch {
        Write-Host "  Warning: Could not parse feedback.json: $_" -ForegroundColor Yellow
        return @()
    }
}

function Get-FeedbackForStory {
    <#
    .SYNOPSIS
        Get relevant feedback entries for a specific story (Story 1.6)
    .PARAMETER StoryId
        The story ID
    .PARAMETER Story
        The story object
    .PARAMETER FocusArea
        Current focus area
    .RETURNS
        String with feedback context, or empty string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$FocusArea
    )

    $feedback = Import-HumanFeedback
    if ($feedback.Count -eq 0) { return "" }

    $relevant = @()

    foreach ($entry in $feedback) {
        $isRelevant = $false

        # Match by story ID
        if ($entry.storyId -and $entry.storyId -eq $StoryId) {
            $isRelevant = $true
        }
        # Match by focus area
        elseif ($entry.focusArea -and $entry.focusArea -eq $FocusArea) {
            $isRelevant = $true
        }
        # Match by keywords in story title
        elseif ($entry.keywords -and $Story -and $Story.title) {
            foreach ($kw in $entry.keywords) {
                if ($Story.title -match [regex]::Escape($kw)) {
                    $isRelevant = $true
                    break
                }
            }
        }
        # Global feedback (no filters)
        elseif (-not $entry.storyId -and -not $entry.focusArea -and -not $entry.keywords) {
            $isRelevant = $true
        }

        if ($isRelevant) {
            $relevant += $entry
        }
    }

    if ($relevant.Count -eq 0) { return "" }

    $context = @()
    $context += "HUMAN FEEDBACK (please incorporate):"
    foreach ($entry in $relevant) {
        $line = "- $($entry.feedback)"
        if ($entry.priority) { $line += " [Priority: $($entry.priority)]" }
        $context += $line
    }
    $context += ""

    return ($context -join "`n")
}

function Format-ReviewPrompt {
    <#
    .SYNOPSIS
        Build a comprehensive review prompt for the code review agent (Story 2.1)
    .DESCRIPTION
        Creates a structured prompt with diff, acceptance criteria, file context,
        and review instructions for independent code review.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Full story object from PRD
    .PARAMETER DiffOutput
        Git diff output string
    .PARAMETER FileOps
        File operations hashtable from Get-FileOperations
    .RETURNS
        Formatted review prompt string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$DiffOutput,
        [hashtable]$FileOps = @{}
    )

    $reviewPromptFile = Join-Path $script:RalphDir "review-prompt.md"
    $basePrompt = ""
    if (Test-Path $reviewPromptFile) {
        $basePrompt = Get-Content $reviewPromptFile -Raw
    }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine($basePrompt)
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Story Under Review")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("**Story ID:** $StoryId")
    if ($Story) {
        [void]$sb.AppendLine("**Title:** $($Story.title)")
        if ($Story.acceptanceCriteria) {
            [void]$sb.AppendLine("")
            [void]$sb.AppendLine("### Acceptance Criteria")
            $i = 1
            foreach ($criterion in $Story.acceptanceCriteria) {
                [void]$sb.AppendLine("$i. $criterion")
                $i++
            }
        }
    }

    # File context
    if ($FileOps -and $FileOps.totalFilesChanged -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Files Changed")
        if ($FileOps.filesCreated) {
            foreach ($f in $FileOps.filesCreated) {
                $path = if ($f -is [hashtable]) { $f.path } else { "$f" }
                [void]$sb.AppendLine("- **Created:** $path")
            }
        }
        if ($FileOps.filesModified) {
            foreach ($f in $FileOps.filesModified) {
                $path = if ($f -is [hashtable]) { $f.path } else { "$f" }
                [void]$sb.AppendLine("- **Modified:** $path")
            }
        }
        if ($FileOps.filesDeleted) {
            foreach ($f in $FileOps.filesDeleted) {
                [void]$sb.AppendLine("- **Deleted:** $f")
            }
        }
    }

    # Diff
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("### Git Diff")
    [void]$sb.AppendLine('```diff')
    if ($DiffOutput) {
        # Truncate very large diffs
        $maxDiffChars = 15000
        if ($DiffOutput.Length -gt $maxDiffChars) {
            [void]$sb.AppendLine($DiffOutput.Substring(0, $maxDiffChars))
            [void]$sb.AppendLine("... [TRUNCATED - diff too large] ...")
        } else {
            [void]$sb.AppendLine($DiffOutput)
        }
    } else {
        [void]$sb.AppendLine("(no diff available)")
    }
    [void]$sb.AppendLine('```')

    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Review Focus Areas")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("In addition to the standard scoring criteria, pay special attention to:")
    [void]$sb.AppendLine("1. **Architecture**: Do changes follow existing patterns in the codebase?")
    [void]$sb.AppendLine("2. **Security**: Any injection risks, hardcoded secrets, unsafe operations?")
    [void]$sb.AppendLine("3. **Style**: Consistent naming, formatting, and conventions?")
    [void]$sb.AppendLine("4. **Completeness**: Are all acceptance criteria fully addressed with evidence?")
    [void]$sb.AppendLine("5. **Regressions**: Could these changes break existing functionality?")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("Output your review as the JSON format specified above.")

    return $sb.ToString()
}

function Search-CriterionEvidence {
    <#
    .SYNOPSIS
        Search Claude output and git diff for evidence of a criterion being met (Story 2.4)
    .PARAMETER Criterion
        Acceptance criterion text
    .PARAMETER ClaudeOutput
        Full Claude stdout+stderr
    .PARAMETER DiffOutput
        Git diff output
    .RETURNS
        Hashtable with found (bool), evidence (string), confidence (float)
    #>
    param(
        [string]$Criterion,
        [string]$ClaudeOutput = "",
        [string]$DiffOutput = ""
    )

    $result = @{
        found = $false
        evidence = ""
        confidence = 0.0
    }

    if (-not $Criterion) { return $result }

    $criterionLower = $Criterion.ToLower()
    $evidenceParts = @()
    $confidenceScore = 0.0

    # Extract keywords from criterion (words > 3 chars, excluding common words)
    $stopWords = @('should', 'must', 'shall', 'that', 'when', 'with', 'from', 'have', 'this', 'been', 'were', 'they', 'each', 'which', 'their', 'will', 'would', 'could', 'into', 'more', 'also', 'than', 'only', 'other', 'does', 'such')
    $keywords = @()
    foreach ($word in ($criterionLower -split '\W+')) {
        if ($word.Length -gt 3 -and $stopWords -notcontains $word) {
            $keywords += $word
        }
    }

    if ($keywords.Count -eq 0) { return $result }

    # Search in Claude output
    if ($ClaudeOutput) {
        $outputLower = $ClaudeOutput.ToLower()
        $matchedKeywords = 0
        foreach ($kw in $keywords) {
            if ($outputLower -match [regex]::Escape($kw)) {
                $matchedKeywords++
            }
        }
        $keywordRatio = if ($keywords.Count -gt 0) { $matchedKeywords / $keywords.Count } else { 0 }

        if ($keywordRatio -ge 0.5) {
            $evidenceParts += "Keywords found in Claude output ($matchedKeywords/$($keywords.Count))"
            $confidenceScore += $keywordRatio * 0.4
        }

        # Check for explicit completion signals
        $completionPatterns = @(
            'acceptance criteria.*met',
            'criterion.*satisfied',
            'implemented.*successfully',
            'all tests pass',
            'verified.*criterion',
            'criterion.*verified',
            'completed.*requirement'
        )
        foreach ($pattern in $completionPatterns) {
            if ($outputLower -match $pattern) {
                $evidenceParts += "Completion signal: $($Matches[0])"
                $confidenceScore += 0.2
                break
            }
        }
    }

    # Search in git diff
    if ($DiffOutput) {
        $diffLower = $DiffOutput.ToLower()
        $diffMatches = 0
        foreach ($kw in $keywords) {
            if ($diffLower -match [regex]::Escape($kw)) {
                $diffMatches++
            }
        }
        $diffRatio = if ($keywords.Count -gt 0) { $diffMatches / $keywords.Count } else { 0 }

        if ($diffRatio -ge 0.3) {
            $evidenceParts += "Keywords found in diff ($diffMatches/$($keywords.Count))"
            $confidenceScore += $diffRatio * 0.3
        }

        # Check for test-related evidence
        if ($criterionLower -match 'test|spec|assert|verify|check') {
            if ($diffLower -match 'def test_|it\s*\(|describe\s*\(|test\s*\(|\.Tests\.|assert|should') {
                $evidenceParts += "Test code found in diff"
                $confidenceScore += 0.2
            }
        }

        # Check for function/class creation evidence
        if ($criterionLower -match 'creat|add|implement|build|write|function|class|method') {
            if ($DiffOutput -match '^\+\s*(def |function |class |const |export )') {
                $evidenceParts += "New code definitions found in diff"
                $confidenceScore += 0.15
            }
        }
    }

    $confidenceScore = [math]::Min($confidenceScore, 1.0)

    if ($evidenceParts.Count -gt 0) {
        $result.found = $confidenceScore -ge 0.3
        $result.evidence = $evidenceParts -join "; "
        $result.confidence = [math]::Round($confidenceScore, 2)
    }
    else {
        $result.evidence = "No evidence found in output or diff"
        $result.confidence = 0.0
    }

    return $result
}

function Test-CanRollback {
    <#
    .SYNOPSIS
        Check if a story's changes can be safely rolled back (Story 4.4)
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        Hashtable with canRollback (bool), commitHash, reason
    #>
    param(
        [string]$StoryId
    )

    $result = @{
        canRollback = $false
        commitHash = ""
        reason = ""
    }

    $config = Get-RalphConfig
    $autoRollback = $config.regression -and $config.regression.autoRollback

    if (-not $autoRollback) {
        $result.reason = "Auto-rollback disabled in config"
        return $result
    }

    # Find the commit for this story
    try {
        $commit = git log --oneline -1 --grep="$StoryId" --format="%H" 2>$null
        if (-not $commit) {
            $result.reason = "No commit found for story $StoryId"
            return $result
        }

        $result.commitHash = $commit.Trim()

        # Check if this is the most recent commit (can only revert HEAD safely)
        $headHash = (git rev-parse HEAD 2>$null).Trim()
        if ($result.commitHash -ne $headHash) {
            $result.reason = "Story commit is not HEAD - cannot safely revert"
            return $result
        }

        # Check for uncommitted changes
        $status = git status --porcelain 2>$null
        if ($status) {
            $result.reason = "Uncommitted changes present - cannot safely revert"
            return $result
        }

        $result.canRollback = $true
        $result.reason = "Can safely revert commit $($result.commitHash.Substring(0, 7))"
    }
    catch {
        $result.reason = "Error checking rollback: $_"
    }

    return $result
}

function Invoke-StoryRollback {
    <#
    .SYNOPSIS
        Rollback a story's changes via git revert (Story 4.4)
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        $true if rollback succeeded, $false otherwise
    #>
    param(
        [string]$StoryId
    )

    $canRollback = Test-CanRollback -StoryId $StoryId

    if (-not $canRollback.canRollback) {
        Write-Host "  Rollback: Cannot rollback $StoryId - $($canRollback.reason)" -ForegroundColor Yellow
        return $false
    }

    Write-Host "  Rollback: Reverting commit for $StoryId..." -ForegroundColor Yellow

    try {
        $revertOutput = git revert --no-edit HEAD 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Rollback: Successfully reverted $StoryId" -ForegroundColor Green

            # Record in learning DB
            Update-LearningDb -Entry @{
                type = 'rollback'
                storyId = $StoryId
                commitHash = $canRollback.commitHash
                reason = 'Regression detected'
            }

            return $true
        }
        else {
            Write-Host "  Rollback: git revert failed: $revertOutput" -ForegroundColor Red
            # Abort the failed revert
            git revert --abort 2>$null
            return $false
        }
    }
    catch {
        Write-Host "  Rollback: Error - $_" -ForegroundColor Red
        git revert --abort 2>$null
        return $false
    }
}
