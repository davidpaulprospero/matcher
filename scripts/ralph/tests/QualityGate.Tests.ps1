#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Phase 1 Quality Gates & Intelligence functions
.DESCRIPTION
    Tests cover:
    - Get-DiffQualityScore (Story 1.4)
    - Get-StoryFailureContext (Story 1.3)
    - Compare-TestBaseline (Story 1.5)
    - Update-TestBaseline (Story 1.5)
    - Import-HumanFeedback (Story 1.6)
    - Get-FeedbackForStory (Story 1.6)
    - New-SprintReport (Story 1.7)
    - Get-SprintTokenBudget (Story 1.8)
    - Get-EstimatedTokens (Story 1.8)
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\qualitygate"
    $script:ConfigFile = Join-Path $script:RalphDir "config\ralph-config.json"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source all functions from ralph.ps1 and lib/*.ps1 into global scope
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

    # Helper: create mock config
    function New-MockConfig {
        param(
            [hashtable]$Overrides = @{}
        )
        $base = @{
            flags = @{
                llmAsJudgeQuality = $false
                acceptanceDrivenBackpressure = $false
                phaseTracking = $true
            }
            review = @{ enabled = $false; model = "sonnet"; timeout = 180; minScoreToPass = 6 }
            quality = @{ minTestRatio = 0.2; maxDiffLines = 800 }
            regression = @{ enabled = $true; blockOnRegression = $true; autoRollback = $false }
            budget = @{ enabled = $false; maxTokensPerSprint = 500000; warnAtPercent = 80 }
        }
        foreach ($key in $Overrides.Keys) {
            $base[$key] = $Overrides[$key]
        }
        return [PSCustomObject]$base
    }
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Get-DiffQualityScore Tests (Story 1.4)
# =============================================================================

Describe "Get-DiffQualityScore" -Tag "Unit", "QualityGate", "Phase1" {

    BeforeAll {
        # Mock Get-RalphConfig to return test config
        function global:Get-RalphConfig {
            return [PSCustomObject]@{
                quality = [PSCustomObject]@{ minTestRatio = 0.2; maxDiffLines = 800 }
            }
        }
    }

    Context "Empty diff" {
        It "Returns default metrics for empty diff" {
            $result = Get-DiffQualityScore -DiffOutput ""
            $result.totalLinesChanged | Should -Be 0
            $result.testRatio | Should -Be 0.0
            $result.churnRisk | Should -Be "low"
            $result.sizeAppropriate | Should -Be $true
        }

        It "Returns default metrics for null diff" {
            $result = Get-DiffQualityScore -DiffOutput $null
            $result.totalLinesChanged | Should -Be 0
        }
    }

    Context "Implementation-only diff" {
        It "Computes zero test ratio for implementation-only changes" {
            $diff = @"
diff --git a/src/matching.py b/src/matching.py
--- a/src/matching.py
+++ b/src/matching.py
+def new_function():
+    return True
+
+def another_function():
+    pass
"@
            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.testRatio | Should -Be 0.0
            $result.implLinesChanged | Should -BeGreaterThan 0
            $result.testLinesChanged | Should -Be 0
            $result.filesChanged | Should -Be 1
        }
    }

    Context "Mixed diff with tests" {
        It "Computes correct test ratio" {
            $diff = @"
diff --git a/src/matching.py b/src/matching.py
+def new_function():
+    return True
diff --git a/tests/test_matching.py b/tests/test_matching.py
+def test_new_function():
+    assert new_function() == True
"@
            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.testRatio | Should -BeGreaterThan 0
            $result.testLinesChanged | Should -BeGreaterThan 0
            $result.implLinesChanged | Should -BeGreaterThan 0
            $result.filesChanged | Should -Be 2
        }
    }

    Context "Pure test changes" {
        It "Returns testRatio of 1.0" {
            $diff = @"
diff --git a/tests/test_cache.py b/tests/test_cache.py
+def test_cache_hit():
+    assert cache.get('key') == 'value'
"@
            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.testRatio | Should -Be 1.0
        }
    }

    Context "Size limits" {
        It "Flags oversized diffs" {
            # Create a large diff
            $lines = @("diff --git a/src/big.py b/src/big.py")
            for ($i = 0; $i -lt 900; $i++) {
                $lines += "+line_$i = True"
            }
            $diff = $lines -join "`n"

            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.sizeAppropriate | Should -Be $false
            $result.warnings.Count | Should -BeGreaterThan 0
        }
    }

    Context "Churn risk" {
        It "Detects high churn from many files" {
            $lines = @()
            for ($i = 0; $i -lt 12; $i++) {
                $lines += "diff --git a/src/file$i.py b/src/file$i.py"
                $lines += "+change_$i = True"
            }
            $diff = $lines -join "`n"

            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.churnRisk | Should -Be "high"
            $result.filesChanged | Should -Be 12
        }

        It "Detects medium churn from moderate file count" {
            $lines = @()
            for ($i = 0; $i -lt 7; $i++) {
                $lines += "diff --git a/src/file$i.py b/src/file$i.py"
                $lines += "+change_$i = True"
            }
            $diff = $lines -join "`n"

            $result = Get-DiffQualityScore -DiffOutput $diff
            $result.churnRisk | Should -Be "medium"
        }
    }
}

# =============================================================================
# Get-StoryFailureContext Tests (Story 1.3)
# =============================================================================

Describe "Get-StoryFailureContext" -Tag "Unit", "QualityGate", "Phase1" {

    BeforeAll {
        $script:SessionLogDir = $script:TestDataDir
        $script:MetricsFile = Join-Path $script:TestDataDir "metrics.csv"
        $script:State = @{
            SessionId = 'test-session'; IterationCount = 5; ConsecutiveFailures = 0
            SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
            LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
            LastExplorationSummary = ''; LastExplorationTime = $null
            SprintExplorationContext = ''; LastExplorationCommit = ''
        }
    }

    Context "First attempt" {
        It "Returns empty string for first attempt (RetryCount=0)" {
            $result = Get-StoryFailureContext -StoryId "US-001" -RetryCount 0
            $result | Should -BeNullOrEmpty
        }
    }

    Context "Retry with context" {
        It "Returns retry context for RetryCount=1 (attempt 2)" {
            $result = Get-StoryFailureContext -StoryId "US-001" -RetryCount 1
            $result | Should -Match "RETRY CONTEXT"
            $result | Should -Match "attempt 2"
        }

        It "Returns retry context for RetryCount=2 (attempt 3)" {
            $result = Get-StoryFailureContext -StoryId "US-001" -RetryCount 2
            $result | Should -Match "RETRY CONTEXT"
            $result | Should -Match "attempt 3"
        }

        It "Includes previous verification failures" {
            $verificationFile = Join-Path $script:TestDataDir "story_US-002_verification.json"
            @{
                storyId = "US-002"
                acceptanceCriteria = @(
                    @{ criterion = "Add function X"; verified = $true; evidence = "Found" }
                    @{ criterion = "Add tests for X"; verified = $false; evidence = "Not found" }
                )
            } | ConvertTo-Json -Depth 5 | Set-Content $verificationFile -Encoding UTF8

            $result = Get-StoryFailureContext -StoryId "US-002" -RetryCount 3
            $result | Should -Match "Unmet criteria"
            $result | Should -Match "Add tests for X"

            Remove-Item $verificationFile -Force -ErrorAction SilentlyContinue
        }

        It "Includes review feedback from previous attempt" {
            $reviewFile = Join-Path $script:TestDataDir "review_US-003.json"
            @{
                overallScore = 4
                issues = @(
                    @{ severity = "error"; description = "Missing unit tests" }
                    @{ severity = "warning"; description = "Long function" }
                )
            } | ConvertTo-Json -Depth 5 | Set-Content $reviewFile -Encoding UTF8

            $result = Get-StoryFailureContext -StoryId "US-003" -RetryCount 2
            $result | Should -Match "quality review"
            $result | Should -Match "Missing unit tests"

            Remove-Item $reviewFile -Force -ErrorAction SilentlyContinue
        }
    }
}

# =============================================================================
# Compare-TestBaseline / Update-TestBaseline Tests (Story 1.5)
# =============================================================================

Describe "Test Baseline Functions" -Tag "Unit", "QualityGate", "Phase1" {

    BeforeAll {
        $script:RalphDir = $script:TestDataDir
        # Create config subdirectory (Compare-TestBaseline looks in $RalphDir/config/)
        $script:ConfigDir = Join-Path $script:TestDataDir "config"
        if (-not (Test-Path $script:ConfigDir)) {
            New-Item -ItemType Directory -Path $script:ConfigDir -Force | Out-Null
        }
        function global:Get-RalphConfig {
            return [PSCustomObject]@{
                regression = [PSCustomObject]@{ enabled = $true; blockOnRegression = $true }
            }
        }
        function global:Write-JsonNoBom {
            param([string]$Path, [string]$Content)
            $parentDir = Split-Path -Parent $Path
            if ($parentDir -and -not (Test-Path $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }
            $Content | Set-Content $Path -Encoding UTF8
        }
    }

    BeforeEach {
        $baselineFile = Join-Path $script:ConfigDir "test_baseline.json"
        if (Test-Path $baselineFile) {
            Remove-Item $baselineFile -Force
        }
    }

    Context "Compare-TestBaseline" {
        It "Returns null when no baseline exists" {
            $result = Compare-TestBaseline -CurrentResults "41/41 pass"
            $result | Should -BeNullOrEmpty
        }

        It "Detects no regression when tests improve" {
            @{ capturedAt = "2026-01-01"; passed = 40; failed = 1; totalTests = 41 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "41/41 pass"
            $result.hasRegression | Should -Be $false
            $result.passedDelta | Should -Be 1
        }

        It "Detects regression when failures increase" {
            @{ capturedAt = "2026-01-01"; passed = 41; failed = 0; totalTests = 41 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "38/41 pass, 3 fail"
            $result.hasRegression | Should -Be $true
            $result.failedDelta | Should -BeGreaterThan 0
        }

        It "Returns null when regression detection disabled" {
            function global:Get-RalphConfig {
                return [PSCustomObject]@{
                    regression = [PSCustomObject]@{ enabled = $false }
                }
            }

            @{ capturedAt = "2026-01-01"; passed = 41; failed = 0; totalTests = 41 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "38/41 pass, 3 fail"
            $result | Should -BeNullOrEmpty

            # Restore
            function global:Get-RalphConfig {
                return [PSCustomObject]@{
                    regression = [PSCustomObject]@{ enabled = $true; blockOnRegression = $true }
                }
            }
        }
    }

    Context "Compare-TestBaseline subset detection" {
        It "Does not flag regression for subset runs (fewer tests, all passing)" {
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "38/38 pass"
            $result.hasRegression | Should -Be $false
            $result.isSubsetRun | Should -Be $true
        }

        It "Flags regression for subset runs with new failures" {
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "35/38 pass, 3 fail"
            $result.hasRegression | Should -Be $true
            $result.failedDelta | Should -BeGreaterThan 0
        }

        It "Flags regression for comparable runs with fewer passing" {
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            # 48/52 is 92% of baseline - comparable run, so drop IS a regression
            $result = Compare-TestBaseline -CurrentResults "48/52 pass"
            $result.hasRegression | Should -Be $true
            $result.isSubsetRun | Should -Be $false
        }

        It "Does not flag subset when current total is at threshold" {
            @{ capturedAt = "2026-01-01"; passed = 100; failed = 0; totalTests = 100 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            # 80/80 is exactly 80% of baseline - NOT a subset (threshold is <80%)
            $result = Compare-TestBaseline -CurrentResults "80/80 pass"
            $result.isSubsetRun | Should -Be $false
        }

        It "Includes currentTotal and baselineTotal in result" {
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52 } |
                ConvertTo-Json | Set-Content (Join-Path $script:ConfigDir "test_baseline.json") -Encoding UTF8

            $result = Compare-TestBaseline -CurrentResults "38/38 pass"
            $result.currentTotal | Should -Be 38
            $result.baselineTotal | Should -Be 52
        }
    }

    Context "Update-TestBaseline" {
        It "Creates baseline file from test results" {
            Update-TestBaseline -TestResults "41/41 pass"

            $baselineFile = Join-Path $script:ConfigDir "test_baseline.json"
            Test-Path $baselineFile | Should -Be $true

            $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
            $baseline.passed | Should -Be 41
            $baseline.totalTests | Should -Be 41
        }

        It "Skips update for empty results" {
            Update-TestBaseline -TestResults ""
            $baselineFile = Join-Path $script:ConfigDir "test_baseline.json"
            Test-Path $baselineFile | Should -Be $false
        }

        It "Does not downgrade baseline from subset runs" {
            $baselineFile = Join-Path $script:ConfigDir "test_baseline.json"

            # Set initial baseline with 52 tests
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52; errors = 0 } |
                ConvertTo-Json | Set-Content $baselineFile -Encoding UTF8

            # Attempt update with subset (38 tests)
            Update-TestBaseline -TestResults "38/38 pass"

            # Baseline should NOT have been downgraded
            $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
            $baseline.passed | Should -Be 52
            $baseline.totalTests | Should -Be 52
        }

        It "Updates baseline when comparable or larger run" {
            $baselineFile = Join-Path $script:ConfigDir "test_baseline.json"

            # Set initial baseline with 52 tests
            @{ capturedAt = "2026-01-01"; passed = 52; failed = 0; totalTests = 52; errors = 0 } |
                ConvertTo-Json | Set-Content $baselineFile -Encoding UTF8

            # Update with larger run (60 tests)
            Update-TestBaseline -TestResults "60/60 pass"

            $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
            $baseline.passed | Should -Be 60
            $baseline.totalTests | Should -Be 60
        }
    }
}

# =============================================================================
# Import-HumanFeedback / Get-FeedbackForStory Tests (Story 1.6)
# =============================================================================

Describe "Human Feedback Functions" -Tag "Unit", "QualityGate", "Phase1" {

    BeforeAll {
        $script:RalphDir = $script:TestDataDir
        # Create config subdirectory (Import-HumanFeedback looks in $RalphDir/config/)
        $script:ConfigDir = Join-Path $script:TestDataDir "config"
        if (-not (Test-Path $script:ConfigDir)) {
            New-Item -ItemType Directory -Path $script:ConfigDir -Force | Out-Null
        }
    }

    BeforeEach {
        $feedbackFile = Join-Path $script:ConfigDir "feedback.json"
        if (Test-Path $feedbackFile) {
            Remove-Item $feedbackFile -Force
        }
    }

    Context "Import-HumanFeedback" {
        It "Returns empty array when no feedback file" {
            $result = Import-HumanFeedback
            $result.Count | Should -Be 0
        }

        It "Reads feedback entries from file" {
            @{
                entries = @(
                    @{ feedback = "Use more descriptive variable names"; focusArea = "quality" }
                    @{ feedback = "Always add type hints"; priority = "high" }
                )
            } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:ConfigDir "feedback.json") -Encoding UTF8

            $result = Import-HumanFeedback
            $result.Count | Should -Be 2
        }

        It "Handles malformed JSON gracefully" {
            "not json" | Set-Content (Join-Path $script:ConfigDir "feedback.json") -Encoding UTF8
            $result = Import-HumanFeedback
            $result.Count | Should -Be 0
        }
    }

    Context "Get-FeedbackForStory" {
        BeforeEach {
            @{
                entries = @(
                    @{ feedback = "Fix rate limiting"; focusArea = "rate-limiting" }
                    @{ feedback = "Always test edge cases"; storyId = "US-005" }
                    @{ feedback = "Follow PEP8"; keywords = @("style", "format") }
                    @{ feedback = "Global: keep it simple" }
                )
            } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:ConfigDir "feedback.json") -Encoding UTF8
        }

        It "Matches by story ID" {
            $story = [PSCustomObject]@{ title = "Something"; acceptanceCriteria = @() }
            $result = Get-FeedbackForStory -StoryId "US-005" -Story $story -FocusArea "testing"
            $result | Should -Match "Always test edge cases"
        }

        It "Matches by focus area" {
            $story = [PSCustomObject]@{ title = "Something"; acceptanceCriteria = @() }
            $result = Get-FeedbackForStory -StoryId "US-099" -Story $story -FocusArea "rate-limiting"
            $result | Should -Match "Fix rate limiting"
        }

        It "Includes global feedback (no filters)" {
            $story = [PSCustomObject]@{ title = "Unrelated story"; acceptanceCriteria = @() }
            $result = Get-FeedbackForStory -StoryId "US-999" -Story $story -FocusArea "otio"
            $result | Should -Match "keep it simple"
        }

        It "Returns empty when no feedback matches" {
            @{
                entries = @(
                    @{ feedback = "Only for US-001"; storyId = "US-001" }
                )
            } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:ConfigDir "feedback.json") -Encoding UTF8

            $story = [PSCustomObject]@{ title = "Something"; acceptanceCriteria = @() }
            $result = Get-FeedbackForStory -StoryId "US-999" -Story $story -FocusArea "otio"
            $result | Should -BeNullOrEmpty
        }
    }
}

# =============================================================================
# Get-EstimatedTokens Tests (Story 1.8)
# =============================================================================

Describe "Get-EstimatedTokens" -Tag "Unit", "QualityGate", "Phase1" {

    Context "Empty input" {
        It "Returns 0 for empty string" {
            Get-EstimatedTokens -Output "" | Should -Be 0
        }

        It "Returns 0 for null" {
            Get-EstimatedTokens -Output $null | Should -Be 0
        }
    }

    Context "Character-based estimation" {
        It "Estimates tokens from character count" {
            $output = "a" * 400  # 400 chars = ~100 tokens
            $result = Get-EstimatedTokens -Output $output
            $result | Should -Be 100
        }
    }

    Context "Token parsing from Claude output" {
        It "Parses total_tokens from output" {
            $output = "Some output here`ntotal_tokens: 12345`nMore output"
            $result = Get-EstimatedTokens -Output $output
            $result | Should -Be 12345
        }

        It "Parses input + output tokens" {
            $output = "input_tokens: 5000`noutput_tokens: 2000"
            $result = Get-EstimatedTokens -Output $output
            $result | Should -Be 7000
        }
    }
}

# =============================================================================
# Get-SprintTokenBudget Tests (Story 1.8)
# =============================================================================

Describe "Get-SprintTokenBudget" -Tag "Unit", "QualityGate", "Phase1" {

    Context "Budget disabled" {
        It "Returns null when budget is disabled" {
            function global:Get-RalphConfig {
                return [PSCustomObject]@{
                    budget = [PSCustomObject]@{ enabled = $false }
                }
            }

            $result = Get-SprintTokenBudget
            $result | Should -BeNullOrEmpty
        }
    }

    Context "Budget enabled" {
        BeforeAll {
            $script:MetricsFile = Join-Path $script:TestDataDir "budget_metrics.csv"
            $script:State = @{
                SessionId = 'test-session'; IterationCount = 0; ConsecutiveFailures = 0
                SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
                LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
                LastExplorationSummary = ''; LastExplorationTime = $null
                SprintExplorationContext = ''; LastExplorationCommit = ''
            }
        }

        It "Returns budget status with zero usage when no metrics" {
            function global:Get-RalphConfig {
                return [PSCustomObject]@{
                    budget = [PSCustomObject]@{ enabled = $true; maxTokensPerSprint = 100000; warnAtPercent = 80 }
                }
            }

            # Ensure no metrics file
            if (Test-Path $script:MetricsFile) {
                Remove-Item $script:MetricsFile -Force
            }

            $result = Get-SprintTokenBudget
            $result | Should -Not -BeNullOrEmpty
            $result.totalUsed | Should -Be 0
            $result.maxTokens | Should -Be 100000
            $result.exceeded | Should -Be $false
        }
    }
}

# =============================================================================
# New-SprintReport Tests (Story 1.7)
# =============================================================================

Describe "New-SprintReport" -Tag "Unit", "QualityGate", "Phase1" {

    BeforeAll {
        $script:ArchiveDir = Join-Path $script:TestDataDir "archive"
        $script:MetricsFile = Join-Path $script:TestDataDir "report_metrics.csv"
        $script:SessionLogDir = Join-Path $script:TestDataDir "logs"
        $script:RalphDir = $script:TestDataDir

        if (-not (Test-Path $script:ArchiveDir)) {
            New-Item -ItemType Directory -Path $script:ArchiveDir -Force | Out-Null
        }
        if (-not (Test-Path $script:SessionLogDir)) {
            New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null
        }

        function global:Get-RalphConfig {
            return [PSCustomObject]@{
                budget = [PSCustomObject]@{ enabled = $false }
            }
        }
    }

    AfterAll {
        if (Test-Path $script:ArchiveDir) {
            Remove-Item $script:ArchiveDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Report generation" {
        It "Returns null for null PRD" {
            $result = New-SprintReport -SprintData @{ sprintNumber = 1 } -Prd $null
            $result | Should -BeNullOrEmpty
        }

        It "Generates report for completed sprint" {
            $prd = [PSCustomObject]@{
                focusArea = "testing"
                userStories = @(
                    [PSCustomObject]@{ id = "US-001"; title = "Add tests"; passes = $true; acceptanceCriteria = @("Write unit tests", "Cover edge cases") }
                    [PSCustomObject]@{ id = "US-002"; title = "Fix bugs"; passes = $true; acceptanceCriteria = @("Fix bug A") }
                )
            }

            $result = New-SprintReport -SprintData @{ sprintNumber = 99 } -Prd $prd
            $result | Should -Not -BeNullOrEmpty
            Test-Path $result | Should -Be $true

            $content = Get-Content $result -Raw
            $content | Should -Match "Sprint 99 Report"
            $content | Should -Match "testing"
            $content | Should -Match "US-001"
            $content | Should -Match "US-002"
            $content | Should -Match "Completed Stories"

            Remove-Item $result -Force -ErrorAction SilentlyContinue
        }

        It "Includes incomplete stories section" {
            $prd = [PSCustomObject]@{
                focusArea = "quality"
                userStories = @(
                    [PSCustomObject]@{ id = "US-001"; title = "Done"; passes = $true; acceptanceCriteria = @() }
                    [PSCustomObject]@{ id = "US-002"; title = "Not done"; passes = $false; acceptanceCriteria = @("Still todo") }
                )
            }

            $result = New-SprintReport -SprintData @{ sprintNumber = 100 } -Prd $prd
            $content = Get-Content $result -Raw
            $content | Should -Match "Incomplete Stories"
            $content | Should -Match "Not done"

            Remove-Item $result -Force -ErrorAction SilentlyContinue
        }

        It "Includes healing summary when healing events exist" {
            Mock Get-HealingSummary {
                return @{
                    TotalHealingSessions = 3
                    TotalResolved        = 2
                    TotalFailed          = 1
                    TotalAttempts        = 7
                    TierBreakdown        = @{ 1 = 1; 2 = 1; 3 = 1 }
                    Events               = @()
                }
            }

            $prd = [PSCustomObject]@{
                focusArea = "testing"
                userStories = @(
                    [PSCustomObject]@{ id = "US-001"; title = "Done"; passes = $true; acceptanceCriteria = @() }
                )
            }

            $result = New-SprintReport -SprintData @{ sprintNumber = 101 } -Prd $prd
            $content = Get-Content $result -Raw
            $content | Should -Match "Self-Healing"
            $content | Should -Match "Healing sessions"
            $content | Should -Match "Tier breakdown"

            Remove-Item $result -Force -ErrorAction SilentlyContinue
        }

        It "Omits healing section when no healing events" {
            Mock Get-HealingSummary {
                return @{
                    TotalHealingSessions = 0
                    TotalResolved        = 0
                    TotalFailed          = 0
                    TotalAttempts        = 0
                    TierBreakdown        = @{ 1 = 0; 2 = 0; 3 = 0 }
                    Events               = @()
                }
            }

            $prd = [PSCustomObject]@{
                focusArea = "testing"
                userStories = @(
                    [PSCustomObject]@{ id = "US-001"; title = "Done"; passes = $true; acceptanceCriteria = @() }
                )
            }

            $result = New-SprintReport -SprintData @{ sprintNumber = 102 } -Prd $prd
            $content = Get-Content $result -Raw
            $content | Should -Not -Match "Self-Healing"

            Remove-Item $result -Force -ErrorAction SilentlyContinue
        }
    }
}
