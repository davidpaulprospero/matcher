#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for pre-flight story detection functions
.DESCRIPTION
    Tests cover:
    - Test-StoryAlreadyCommitted: detect stories committed but not marked in prd.json
    - Confirm-CommitMatchesStory: LLM verification of commit/story pairs
    - Complete-StoryAutomatically: auto-mark stories as complete
    - Invoke-BatchPreFlight: batch scan at sprint start
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\preflight"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source all functions from ralph.ps1 and lib/*.ps1 into global scope
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Describe "Confirm-CommitMatchesStory" {
    It "Returns empty array when no candidates provided" {
        $result = Confirm-CommitMatchesStory -Candidates @()
        $result | Should -BeNullOrEmpty
    }

    It "Returns empty array when Claude is not available" {
        Mock Get-ClaudePath { return $null }
        $candidates = @(
            @{ storyId = "US-001"; storyTitle = "Some story"; commitMsg = "feat: [US-001] Some commit" }
        )
        $result = Confirm-CommitMatchesStory -Candidates $candidates
        $result | Should -BeNullOrEmpty
    }
}

Describe "Test-StoryAlreadyCommitted" {
    It "Returns false for non-existent story ID" {
        $story = [PSCustomObject]@{ id = "US-999"; title = "Nonexistent feature work"; passes = $false }
        $result = Test-StoryAlreadyCommitted -StoryId "US-999" -Story $story
        $result | Should -BeFalse
    }

    It "Returns false when Story parameter is null (safety guard)" {
        $result = Test-StoryAlreadyCommitted -StoryId "US-001"
        $result | Should -BeFalse
    }

    It "Returns true when LLM confirms commit matches story" {
        $commitExists = git log --oneline --all --grep="\[US-004\]" 2>$null
        if (-not $commitExists) {
            Set-ItResult -Skipped -Because "No US-004 commits in current repo"
            return
        }

        # Mock LLM to confirm match
        Mock Confirm-CommitMatchesStory { return @("US-004") }

        $story = [PSCustomObject]@{
            id = "US-004"
            title = "Wire impersonation into audio_first.py (3 command sites)"
            passes = $false
        }
        $result = Test-StoryAlreadyCommitted -StoryId "US-004" -Story $story
        $result | Should -BeTrue
    }

    It "Returns false when LLM rejects cross-sprint story" {
        $commitExists = git log --oneline --all --grep="\[US-001\]" 2>$null
        if (-not $commitExists) {
            Set-ItResult -Skipped -Because "No US-001 commits in current repo"
            return
        }

        # Mock LLM to reject match (different sprint)
        Mock Confirm-CommitMatchesStory { return @() }

        $story = [PSCustomObject]@{
            id = "US-001"
            title = "Create EscalationManager with 3-tier bypass progression"
            passes = $false
        }
        $result = Test-StoryAlreadyCommitted -StoryId "US-001" -Story $story
        $result | Should -BeFalse
    }

    It "Handles git errors gracefully" {
        Mock git { throw "git not found" } -Verifiable
        $story = [PSCustomObject]@{ id = "US-001"; title = "Some feature"; passes = $false }
        { Test-StoryAlreadyCommitted -StoryId "US-001" -Story $story } | Should -Not -Throw
    }
}

Describe "Complete-StoryAutomatically" {
    BeforeEach {
        # Save original RalphDir so we can restore it
        $script:OrigRalphDir = $script:RalphDir

        # Create a temporary prd.json for testing
        $script:TestPrdFile = Join-Path $script:TestDataDir "prd.json"
        $script:TestProgressFile = Join-Path $script:TestDataDir "progress.txt"
        $script:TestMetricsFile = Join-Path $script:TestDataDir "metrics.csv"

        $prd = @{
            focusArea = "download"
            userStories = @(
                @{ id = "US-001"; title = "Story One"; passes = $true; notes = "done" },
                @{ id = "US-002"; title = "Story Two"; passes = $false; notes = "" }
            )
        }
        $prd | ConvertTo-Json -Depth 10 | Set-Content $script:TestPrdFile -Encoding UTF8
        "" | Set-Content $script:TestProgressFile -Encoding UTF8
        "timestamp,session,sprint,story_id,mode" | Set-Content $script:TestMetricsFile -Encoding UTF8

        # Point script variables at test directory
        $script:RalphDir = $script:TestDataDir
        $script:PrdFile = $script:TestPrdFile
        $script:State = @{
            SessionId = 'test-session'; IterationCount = 0; ConsecutiveFailures = 0
            SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
            LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
            LastExplorationSummary = ''; LastExplorationTime = $null
            SprintExplorationContext = ''; LastExplorationCommit = ''
        }
        $script:SprintNumber = 99
    }

    AfterEach {
        # Restore original RalphDir
        $script:RalphDir = $script:OrigRalphDir
        Remove-Item $script:TestPrdFile -Force -ErrorAction SilentlyContinue
        Remove-Item $script:TestProgressFile -Force -ErrorAction SilentlyContinue
        Remove-Item $script:TestMetricsFile -Force -ErrorAction SilentlyContinue
    }

    It "Updates prd.json to mark story as passes: true" {
        $story = [PSCustomObject]@{ id = "US-002"; title = "Story Two"; passes = $false }
        Complete-StoryAutomatically -StoryId "US-002" -Story $story -Reason "test"

        $prd = Get-Content $script:TestPrdFile -Raw | ConvertFrom-Json
        $updated = $prd.userStories | Where-Object { $_.id -eq "US-002" }
        $updated.passes | Should -BeTrue
    }

    It "Writes notes with reason" {
        $story = [PSCustomObject]@{ id = "US-002"; title = "Story Two"; passes = $false }
        Complete-StoryAutomatically -StoryId "US-002" -Story $story -Reason "git-commit-detected"

        $prd = Get-Content $script:TestPrdFile -Raw | ConvertFrom-Json
        $updated = $prd.userStories | Where-Object { $_.id -eq "US-002" }
        $updated.notes | Should -Match "git-commit-detected"
    }

    It "Appends to progress.txt" {
        $story = [PSCustomObject]@{ id = "US-002"; title = "Story Two"; passes = $false }
        Complete-StoryAutomatically -StoryId "US-002" -Story $story -Reason "test"

        $progress = Get-Content $script:TestProgressFile -Raw
        $progress | Should -Match "US-002"
        $progress | Should -Match "Pre-flight Auto-Complete"
    }

    It "Does not throw on missing prd file" {
        $script:PrdFile = Join-Path $script:TestDataDir "nonexistent.json"
        { Complete-StoryAutomatically -StoryId "US-002" -Reason "test" } | Should -Not -Throw
    }

    It "Does not modify already-passed stories" {
        $story = [PSCustomObject]@{ id = "US-001"; title = "Story One"; passes = $true }
        Complete-StoryAutomatically -StoryId "US-001" -Story $story -Reason "test"

        $prd = Get-Content $script:TestPrdFile -Raw | ConvertFrom-Json
        $existing = $prd.userStories | Where-Object { $_.id -eq "US-001" }
        $existing.passes | Should -BeTrue
    }
}

Describe "Invoke-BatchPreFlight" {
    BeforeEach {
        $script:OrigRalphDir = $script:RalphDir
        $script:TestPrdFile = Join-Path $script:TestDataDir "prd.json"
        $script:TestProgressFile = Join-Path $script:TestDataDir "progress.txt"
        $script:TestMetricsFile = Join-Path $script:TestDataDir "metrics.csv"

        # Point script variables at test directory
        $script:RalphDir = $script:TestDataDir
        $script:PrdFile = $script:TestPrdFile
        $script:State = @{
            SessionId = 'test-session'; IterationCount = 0; ConsecutiveFailures = 0
            SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
            LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
            LastExplorationSummary = ''; LastExplorationTime = $null
            SprintExplorationContext = ''; LastExplorationCommit = ''
        }
        $script:SprintNumber = 99
    }

    AfterEach {
        $script:RalphDir = $script:OrigRalphDir
        Remove-Item $script:TestPrdFile -Force -ErrorAction SilentlyContinue
        Remove-Item $script:TestProgressFile -Force -ErrorAction SilentlyContinue
        Remove-Item $script:TestMetricsFile -Force -ErrorAction SilentlyContinue
    }

    It "Returns 0 when prd file does not exist" {
        $script:PrdFile = Join-Path $script:TestDataDir "nonexistent.json"
        $result = Invoke-BatchPreFlight
        $result | Should -Be 0
    }

    It "Returns 0 when all stories already pass" {
        $prd = @{
            focusArea = "download"
            userStories = @(
                @{ id = "US-001"; title = "Story One"; passes = $true; notes = "done" },
                @{ id = "US-002"; title = "Story Two"; passes = $true; notes = "done" }
            )
        }
        $prd | ConvertTo-Json -Depth 10 | Set-Content $script:TestPrdFile -Encoding UTF8
        "" | Set-Content $script:TestProgressFile -Encoding UTF8
        "timestamp,session,sprint,story_id,mode" | Set-Content $script:TestMetricsFile -Encoding UTF8

        $result = Invoke-BatchPreFlight
        $result | Should -Be 0
    }

    It "Auto-completes stories when LLM confirms match" {
        $commitExists = git log --oneline --all --grep="\[US-004\]" 2>$null
        if (-not $commitExists) {
            Set-ItResult -Skipped -Because "No US-004 commits in current repo"
            return
        }

        # Mock LLM to confirm US-004 only
        Mock Confirm-CommitMatchesStory { return @("US-004") }

        $prd = @{
            focusArea = "download"
            userStories = @(
                @{ id = "US-004"; title = "Wire impersonation into audio_first.py"; passes = $false; notes = "" },
                @{ id = "US-099"; title = "Nonexistent story"; passes = $false; notes = "" }
            )
        }
        $prd | ConvertTo-Json -Depth 10 | Set-Content $script:TestPrdFile -Encoding UTF8
        "" | Set-Content $script:TestProgressFile -Encoding UTF8
        "timestamp,session,sprint,story_id,mode" | Set-Content $script:TestMetricsFile -Encoding UTF8

        $result = Invoke-BatchPreFlight
        $result | Should -Be 1

        # Verify US-004 was marked complete, US-099 was not
        $updatedPrd = Get-Content $script:TestPrdFile -Raw | ConvertFrom-Json
        ($updatedPrd.userStories | Where-Object { $_.id -eq "US-004" }).passes | Should -BeTrue
        ($updatedPrd.userStories | Where-Object { $_.id -eq "US-099" }).passes | Should -BeFalse
    }

    It "Does not auto-complete when LLM rejects cross-sprint stories" {
        $commitExists = git log --oneline --all --grep="\[US-001\]" 2>$null
        if (-not $commitExists) {
            Set-ItResult -Skipped -Because "No US-001 commits in current repo"
            return
        }

        # Mock LLM to reject all (different sprint)
        Mock Confirm-CommitMatchesStory { return @() }

        $prd = @{
            focusArea = "download"
            userStories = @(
                @{ id = "US-001"; title = "Create EscalationManager with 3-tier bypass progression"; passes = $false; notes = "" }
            )
        }
        $prd | ConvertTo-Json -Depth 10 | Set-Content $script:TestPrdFile -Encoding UTF8
        "" | Set-Content $script:TestProgressFile -Encoding UTF8
        "timestamp,session,sprint,story_id,mode" | Set-Content $script:TestMetricsFile -Encoding UTF8

        $result = Invoke-BatchPreFlight
        $result | Should -Be 0

        $updatedPrd = Get-Content $script:TestPrdFile -Raw | ConvertFrom-Json
        ($updatedPrd.userStories | Where-Object { $_.id -eq "US-001" }).passes | Should -BeFalse
    }

    It "Returns 0 when LLM is unavailable (conservative fallback)" {
        $commitExists = git log --oneline --all --grep="\[US-004\]" 2>$null
        if (-not $commitExists) {
            Set-ItResult -Skipped -Because "No US-004 commits in current repo"
            return
        }

        # Mock LLM as unavailable (returns empty)
        Mock Confirm-CommitMatchesStory { return @() }

        $prd = @{
            focusArea = "download"
            userStories = @(
                @{ id = "US-004"; title = "Wire impersonation into audio_first.py"; passes = $false; notes = "" }
            )
        }
        $prd | ConvertTo-Json -Depth 10 | Set-Content $script:TestPrdFile -Encoding UTF8
        "" | Set-Content $script:TestProgressFile -Encoding UTF8
        "timestamp,session,sprint,story_id,mode" | Set-Content $script:TestMetricsFile -Encoding UTF8

        $result = Invoke-BatchPreFlight
        $result | Should -Be 0
    }

    It "Does not throw on malformed prd.json" {
        "this is not json" | Set-Content $script:TestPrdFile -Encoding UTF8
        { Invoke-BatchPreFlight } | Should -Not -Throw
        $result = Invoke-BatchPreFlight
        $result | Should -Be 0
    }
}
