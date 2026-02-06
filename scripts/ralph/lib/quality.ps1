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
        $psi.Arguments = "--print --dangerously-skip-permissions --model $model"
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WorkingDirectory = $script:ProjectRoot

        # Async output capture (prevents deadlock when output fills pipe buffer)
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        $outBuilder = [System.Text.StringBuilder]::new()
        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()

            # Pipe prompt via stdin to avoid CLI argument length limits
            $process.StandardInput.Write($reviewRequest)
            $process.StandardInput.Close()

            # Poll HasExited instead of WaitForExit to avoid .NET Framework deadlock
            # when child processes inherit stdout/stderr pipe handles
            $deadline = (Get-Date).AddSeconds($timeout)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $exited = $process.HasExited
            try { $process.CancelOutputRead() } catch {}
            Start-Sleep -Milliseconds 200
            $output = $outBuilder.ToString()

            if (-not $exited) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                Write-Host "  Review timed out after ${timeout}s" -ForegroundColor Yellow
                return $null
            }
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            if ($process -and -not $process.HasExited) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
            }
            if ($process) { $process.Dispose() }
        }

        # Try to parse JSON from output
        $jsonMatch = [regex]::Match($output, '\{[\s\S]*"overallScore"[\s\S]*\}')
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

    if ($RetryCount -lt 1) {
        return ""
    }

    $context = @()
    $context += "============================================================"
    $context += "RETRY CONTEXT (attempt $($RetryCount + 1))"
    $context += "============================================================"
    $context += ""
    $context += "Previous attempt failed or timed out. Focus on quick, targeted changes."
    $context += "If the previous attempt stalled during exploration, skip exploration - story details are in this prompt."

    # Check for last output file
    $prevIteration = $script:State.IterationCount  # Current iteration (we're building prompt for next)
    $outFile = Join-Path $script:SessionLogDir "claude_out_$prevIteration.log"
    $errFile = Join-Path $script:SessionLogDir "claude_err_$prevIteration.log"

    $lastOutput = ""
    if (Test-Path $outFile) {
        $lastOutput = Get-Content $outFile -Raw -ErrorAction SilentlyContinue
    }
    if (Test-Path $errFile) {
        $lastOutput += "`n" + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
    }

    # Get last 20 lines of previous output (stripped of ANSI codes)
    if ($lastOutput) {
        $cleanOutput = $lastOutput -replace '\x1b\[[0-9;]*m', ''
        $lines = ($cleanOutput -split "`n") | Where-Object { $_.Trim() }
        $tailLines = if ($lines.Count -gt 20) { $lines[-20..-1] } else { $lines }
        $tailText = ($tailLines -join "`n").Trim()
        if ($tailText) {
            $context += ""
            $context += "Last 20 lines from previous attempt:"
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
    $prevVerification = Read-JsonFile -Path $prevVerificationFile
    if ($prevVerification -and $prevVerification.acceptanceCriteria) {
        $unmet = @($prevVerification.acceptanceCriteria | Where-Object { -not $_.verified })
        if ($unmet.Count -gt 0) {
            $context += ""
            $context += "Unmet criteria from previous attempt:"
            foreach ($c in $unmet) {
                $context += "  - $($c.criterion)"
            }
        }
    }

    # Check for review feedback
    $reviewFile = Join-Path $script:SessionLogDir "review_${StoryId}.json"
    $review = Read-JsonFile -Path $reviewFile
    if ($review -and $review.issues) {
        $context += ""
        $context += "Issues from quality review (score: $($review.overallScore)/10):"
        foreach ($issue in $review.issues) {
            $context += "  [$($issue.severity)] $($issue.description)"
        }
    }

    # Regression delta from pre-story baseline (regressionGuard)
    $preBaselineFile = if ($script:Paths) { $script:Paths.PreStoryBaselineFile }
                       else { Join-Path $script:RalphDir "session\pre_story_baseline.json" }
    if (Test-Path $preBaselineFile) {
        $preBaseline = Read-JsonFile -Path $preBaselineFile
        if ($preBaseline -and $preBaseline.passed) {
            $context += ""
            $context += "Pre-story test baseline: $($preBaseline.passed) passed, $($preBaseline.failed) failed"
            $context += "Your changes must not reduce the pass count below $($preBaseline.passed)."
        }
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

    $baselineFile = if ($script:Paths) { $script:Paths.TestBaselineFile } else { Join-Path $script:RalphDir "config\test_baseline.json" }

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
    .PARAMETER BaselineOverride
        Optional hashtable with passed/failed counts from pre-story oracle baseline.
        When provided, overrides the file-based baseline for oracle-based regression detection.
    .RETURNS
        Hashtable with regression info, or $null if no baseline
    #>
    param(
        [string]$CurrentResults,
        [hashtable]$BaselineOverride = $null
    )

    $baselineFile = if ($script:Paths) { $script:Paths.TestBaselineFile } else { Join-Path $script:RalphDir "config\test_baseline.json" }

    if (-not $BaselineOverride -and -not (Test-Path $baselineFile)) {
        return $null
    }

    $config = Get-RalphConfig
    $regressionEnabled = -not $config.regression -or $config.regression.enabled -ne $false

    if (-not $regressionEnabled) {
        return $null
    }

    $baseline = if ($BaselineOverride) { $BaselineOverride } else { Read-JsonFile -Path $baselineFile }
    if (-not $baseline) { return $null }

    # Parse current results
    $currentPassed = 0
    $currentFailed = 0
    $currentTotal = 0
    if ($CurrentResults -match '(\d+)/(\d+)') {
        $currentPassed = [int]$Matches[1]
        $currentTotal = [int]$Matches[2]
    }
    if ($CurrentResults -match '(\d+)\s+fail') {
        $currentFailed = [int]$Matches[1]
    }
    # Fallback: if total wasn't parsed from X/Y format, compute it
    if ($currentTotal -eq 0) {
        $currentTotal = $currentPassed + $currentFailed
    }

    $baselinePassed = [int]$baseline.passed
    $baselineFailed = [int]$baseline.failed
    $baselineTotal = $baselinePassed + $baselineFailed

    $passedDelta = $currentPassed - $baselinePassed
    $failedDelta = $currentFailed - $baselineFailed

    # Detect subset runs: iteration ran significantly fewer tests than baseline.
    # Each Claude iteration may run only its own new tests, not the full suite.
    # A passed-count drop from a subset run is NOT a regression.
    $subsetThreshold = 0.8
    $isSubsetRun = ($baselineTotal -gt 0) -and ($currentTotal -lt [math]::Floor($baselineTotal * $subsetThreshold))

    # Regression criteria:
    # 1. New failures appeared (always a regression)
    # 2. Fewer tests passing, but ONLY when running a comparable number of tests
    #    (prevents false positives from subset runs where Claude ran only its new tests)
    $hasRegression = ($failedDelta -gt 0) -or ($passedDelta -lt 0 -and $baselinePassed -gt 0 -and -not $isSubsetRun)

    $comparison = @{
        baselinePassed = $baselinePassed
        baselineFailed = $baselineFailed
        currentPassed = $currentPassed
        currentFailed = $currentFailed
        currentTotal = $currentTotal
        baselineTotal = $baselineTotal
        passedDelta = $passedDelta
        failedDelta = $failedDelta
        hasRegression = $hasRegression
        isSubsetRun = $isSubsetRun
        capturedAt = $baseline.capturedAt
    }

    if ($hasRegression) {
        Write-Host "  REGRESSION DETECTED: $baselinePassed -> $currentPassed passed, $baselineFailed -> $currentFailed failed" -ForegroundColor Red
    }
    elseif ($isSubsetRun -and $passedDelta -lt 0) {
        Write-Host "  Subset run: $currentTotal tests run (baseline: $baselineTotal) - $currentPassed passed, no regression" -ForegroundColor DarkGray
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
    .DESCRIPTION
        Only updates baseline when current run covers at least as many tests
        as the existing baseline. Subset runs (where Claude only ran its own
        new tests) should not downgrade the baseline.
    .PARAMETER TestResults
        Current test results string from Get-TestResults
    #>
    param(
        [string]$TestResults
    )

    $baselineFile = if ($script:Paths) { $script:Paths.TestBaselineFile } else { Join-Path $script:RalphDir "config\test_baseline.json" }

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

    # Read existing baseline to check if this is a subset run
    $existing = $null
    if (Test-Path $baselineFile) {
        $existing = Read-JsonFile -Path $baselineFile
    }

    if ($existing) {
        $existingTotal = [int]$existing.passed + [int]$existing.failed
        if ($existingTotal -gt 0 -and $total -lt [math]::Floor($existingTotal * 0.8)) {
            # Subset run - don't downgrade baseline
            return
        }
    }

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

    $feedbackFile = if ($script:Paths) { $script:Paths.FeedbackFile } else { Join-Path $script:RalphDir "config\feedback.json" }

    $data = Read-JsonFile -Path $feedbackFile
    if ($data -and $data.entries) {
        return @($data.entries)
    }
    return @()
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

# ============================================================================
# CODE REVIEW & PRE-FLIGHT (moved from ralph.ps1)
# ============================================================================

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
        $claudeFullPath = Get-ClaudePath

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $claudeFullPath
        $psi.Arguments = "--print --dangerously-skip-permissions --model $model"
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WorkingDirectory = $script:ProjectRoot

        # Async output capture (prevents deadlock when output fills pipe buffer)
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        $outBuilder = [System.Text.StringBuilder]::new()
        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()

            # Pipe prompt via stdin to avoid CLI argument length limits
            $process.StandardInput.Write($reviewPrompt)
            $process.StandardInput.Close()

            # Poll HasExited instead of WaitForExit to avoid .NET Framework deadlock
            $deadline = (Get-Date).AddSeconds($timeout)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $exited = $process.HasExited
            try { $process.CancelOutputRead() } catch {}
            Start-Sleep -Milliseconds 200
            $reviewOutput = $outBuilder.ToString()

            if (-not $exited) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                Write-Host "  Code review: Timed out after ${timeout}s" -ForegroundColor Yellow
                return $null
            }
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            if ($process -and -not $process.HasExited) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
            }
            if ($process) { $process.Dispose() }
        }

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

    $prd = Get-Sprint
    if (-not $prd) {
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

    # Phase 3: Auto-complete verified matches and generate refined follow-ons
    $autoCompleted = 0
    $refinedStories = @()
    foreach ($id in $matchedIds) {
        $candidate = $candidates | Where-Object { $_.storyId -eq $id } | Select-Object -First 1
        if ($candidate) {
            Write-Host "    Pre-flight: LLM confirmed $id matches commit" -ForegroundColor Green
            Write-Host "      $($candidate.commitMsg)" -ForegroundColor DarkCyan
            Complete-StoryAutomatically -StoryId $id -Story $candidate.storyObj -Reason "git-commit-detected"
            $autoCompleted++

            # Phase 3.5: Generate refined follow-on story
            $refinedStory = Invoke-StoryRefinement `
                -CompletedStory $candidate.storyObj `
                -CommitMessage $candidate.commitMsg `
                -SprintContext $(if ($prd.projectContext) { $prd.projectContext } else { "" })

            if ($refinedStory) {
                $refinedStories += $refinedStory
            }
        }
    }

    # Add refined stories to sprint
    if ($refinedStories.Count -gt 0) {
        $prd = Get-Sprint  # Reload after auto-complete updates
        foreach ($refined in $refinedStories) {
            # Check if ID already exists (avoid duplicates)
            $exists = $prd.userStories | Where-Object { $_.id -eq $refined.id }
            if (-not $exists) {
                $storyObj = [PSCustomObject]$refined
                $prd.userStories += $storyObj
                Write-Host "    Pre-flight: Added refined story $($refined.id) to sprint" -ForegroundColor Cyan
            }
        }
        Save-Sprint -Sprint $prd
        Write-Host "  Pre-flight: Generated $($refinedStories.Count) refined follow-on stories" -ForegroundColor Cyan
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
