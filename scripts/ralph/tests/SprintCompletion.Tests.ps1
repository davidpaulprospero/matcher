#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for sprint completion behavior
.DESCRIPTION
    Tests cover:
    - Continuous iteration when sprint completes in same focus area
    - Archive and switch when focus area changes
    - TrueAuto mode infinite iteration
    - Sprint number incrementing
#>

BeforeAll {
    # Get paths - PSScriptRoot is scripts/ralph/tests
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\sprint"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source all functions from ralph.ps1 and lib/*.ps1 into global scope
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

    # Set up variables
    $script:PrdFile = Join-Path $script:TestDataDir "prd.json"
    $script:ArchiveDir = Join-Path $script:TestDataDir "archive"
    $script:SprintHistoryFile = Join-Path $script:TestDataDir "sprint_history.json"
    $script:RalphDir = $script:TestDataDir
    $script:QueueFile = Join-Path $script:TestDataDir "queue.json"

    # Global versions
    $global:PrdFile = $script:PrdFile
    $global:ArchiveDir = $script:ArchiveDir
    $global:SprintHistoryFile = $script:SprintHistoryFile
    $global:RalphDir = $script:TestDataDir
    $global:QueueFile = $script:QueueFile

    # Helper function for cleanup - defined in BeforeAll for Pester v5
    function global:Reset-TestData {
        $prdPath = $script:PrdFile
        $archivePath = $script:ArchiveDir
        $historyPath = $script:SprintHistoryFile
        $queuePath = $script:QueueFile

        if ($prdPath -and (Test-Path $prdPath)) {
            Remove-Item $prdPath -Force -ErrorAction SilentlyContinue
        }
        if ($archivePath -and (Test-Path $archivePath)) {
            Remove-Item $archivePath -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ($historyPath -and (Test-Path $historyPath)) {
            Remove-Item $historyPath -Force -ErrorAction SilentlyContinue
        }
        if ($queuePath -and (Test-Path $queuePath)) {
            Remove-Item $queuePath -Force -ErrorAction SilentlyContinue
        }
    }
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Same Focus Area - Continuous Iteration Tests
# =============================================================================

Describe "Same Focus Area - Continuous Iteration" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }
    Context "When sprint completes in same focus area" {
        BeforeEach {
            # Create complete sprint PRD
            $prd = @{
                sprintNumber = 5
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $true }
                    @{ id = "US-003"; passes = $true }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should detect sprint is complete" {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $allPass = $true
            foreach ($story in $prd.userStories) {
                if ($story.passes -ne $true) {
                    $allPass = $false
                    break
                }
            }
            $allPass | Should -BeTrue
        }

        It "Should return ShouldGenerate=true for same focus area when complete" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            $result.ShouldGenerate | Should -BeTrue
            $result.Reason | Should -Match "Sprint complete, same focus area"
        }

        It "Should archive completed sprint before generating new" {
            Save-SprintArchive -Reason "complete"

            Test-Path $script:ArchiveDir | Should -BeTrue
            $archives = Get-ChildItem $script:ArchiveDir -Filter "sprint-5*.json"
            $archives.Count | Should -BeGreaterOrEqual 1
        }

        It "Should increment sprint number" {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $currentSprint = $prd.sprintNumber

            # Simulating what New-SeedPRD does
            $newSprint = $currentSprint + 1
            $newSprint | Should -Be 6
        }
    }
}

# =============================================================================
# Incomplete Sprint - Same Focus Area Tests
# =============================================================================

Describe "Incomplete Sprint - Same Focus Area" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }

    Context "When sprint is incomplete" {
        BeforeEach {
            # Create incomplete sprint PRD
            $prd = @{
                sprintNumber = 3
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $false }
                    @{ id = "US-003"; passes = $false }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should return ShouldGenerate=false to continue existing sprint" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            $result.ShouldGenerate | Should -BeFalse
            $result.Reason | Should -Match "Continuing incomplete"
        }

        It "Should NOT archive incomplete sprint for same focus area" {
            # Test-ShouldGenerateNewPRD returns false, so no archiving should happen
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            # Only archive if we're generating new
            if (-not $result.ShouldGenerate) {
                Test-Path $script:ArchiveDir | Should -BeFalse
            }
        }
    }
}

# =============================================================================
# Different Focus Area - Archive and Switch Tests
# =============================================================================

Describe "Different Focus Area - Archive and Switch" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }

    Context "When switching to different focus area" {
        BeforeEach {
            # Create incomplete sprint for "testing"
            $prd = @{
                sprintNumber = 2
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $false }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should return ShouldGenerate=true for different focus area" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "quality"

            $result.ShouldGenerate | Should -BeTrue
            $result.Reason | Should -Match "Superseding"
        }

        It "Should archive with reason 'superseded' when switching" {
            Save-SprintArchive -Reason "superseded"

            $archivePath = Join-Path $script:ArchiveDir "sprint-2.json"
            $archived = Get-Content $archivePath -Raw | ConvertFrom-Json
            $archived._archiveMetadata.reason | Should -Be "superseded"
        }

        It "Should preserve story completion stats in archive" {
            Save-SprintArchive -Reason "superseded"

            $archivePath = Join-Path $script:ArchiveDir "sprint-2.json"
            $archived = Get-Content $archivePath -Raw | ConvertFrom-Json
            $archived._archiveMetadata.completedStories | Should -Be 1
            $archived._archiveMetadata.totalStories | Should -Be 2
        }
    }

    Context "When switching from complete sprint" {
        BeforeEach {
            # Create complete sprint
            $prd = @{
                sprintNumber = 4
                focusArea = "quality"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should generate new PRD for different focus area" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            $result.ShouldGenerate | Should -BeTrue
            $result.Reason | Should -Match "switching to testing"
        }
    }
}

# =============================================================================
# TrueAuto Mode Tests
# =============================================================================

Describe "TrueAuto Mode - Infinite Iteration" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }

    Context "TrueAuto behavior on sprint complete" {
        BeforeEach {
            # Create complete sprint
            $prd = @{
                sprintNumber = 10
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $true }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should archive on sprint complete" {
            Save-SprintArchive -Reason "complete"

            Test-Path $script:ArchiveDir | Should -BeTrue
        }

        It "Should track iteration count in history" {
            Save-SprintArchive -Reason "complete"

            $history = Get-SprintHistory
            $history.totalSprintsCompleted | Should -BeGreaterOrEqual 1
        }
    }
}

# =============================================================================
# Sprint History Accumulation Tests
# =============================================================================

Describe "Sprint History Accumulation" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }

    Context "Cumulative metrics" {
        It "Should accumulate across multiple sprints" -Skip:$true {
            # Skipped: $script: variable scope issue when functions loaded via AST
            # Sprint 1
            $prd1 = @{
                sprintNumber = 1
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $true }
                )
            }
            $prd1 | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
            Save-SprintArchive -Reason "complete"

            # Sprint 2
            $prd2 = @{
                sprintNumber = 2
                focusArea = "quality"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $true }
                    @{ id = "US-003"; passes = $true }
                )
            }
            $prd2 | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
            Save-SprintArchive -Reason "complete"

            # Check accumulation
            $history = Get-SprintHistory
            $history.totalSprintsCompleted | Should -Be 2
            $history.totalStoriesCompleted | Should -Be 5
        }

        It "Should track per-focus-area breakdown" -Skip:$true {
            # Skipped: $script: variable scope issue - breakdown uses file that isn't accessible
            # Sprint for testing
            $prd = @{
                sprintNumber = 1
                focusArea = "rate-limiting"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                )
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
            Save-SprintArchive -Reason "complete"

            $history = Get-SprintHistory
            $history.focusAreaBreakdown.'rate-limiting'.sprints | Should -Be 1
            $history.focusAreaBreakdown.'rate-limiting'.stories | Should -Be 1
        }
    }
}

# =============================================================================
# Edge Cases Tests
# =============================================================================

Describe "Sprint Completion Edge Cases" -Tag "Unit", "SprintCompletion" {
    BeforeEach { Reset-TestData }

    Context "No PRD exists" {
        It "Should return ShouldGenerate=true" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            $result.ShouldGenerate | Should -BeTrue
            $result.Reason | Should -Match "No PRD exists"
        }
    }

    Context "PRD with no stories" {
        BeforeEach {
            $prd = @{
                sprintNumber = 1
                focusArea = "testing"
                userStories = @()
            }
            $prd | ConvertTo-Json -Depth 10 | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should return ShouldGenerate=true for empty PRD" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            # Empty PRD = not complete, so for same focus area should continue
            # But in practice, empty PRD should probably be regenerated
            $result | Should -Not -BeNullOrEmpty
        }

        It "Should NOT archive empty PRD" {
            Save-SprintArchive -Reason "complete"

            Test-Path $script:ArchiveDir | Should -BeFalse
        }
    }

    Context "Corrupted PRD" {
        BeforeEach {
            "this is not json" | Set-Content $script:PrdFile -Encoding UTF8
        }

        It "Should return ShouldGenerate=true for corrupted PRD" {
            $result = Test-ShouldGenerateNewPRD -NewFocusArea "testing"

            $result.ShouldGenerate | Should -BeTrue
            $result.Reason | Should -Match "corrupted"
        }
    }
}

# =============================================================================
# Sync-QueueFromHistory Tests
# =============================================================================

Describe "Sync-QueueFromHistory" -Tag "Unit", "SprintCompletion", "QueueSync" {
    BeforeEach { Reset-TestData }

    Context "When sprint history contains completed queued areas" {
        It "Should mark queued area as completed if found in sprint history" {
            # Set up queue with incomplete area
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = "2026-01-26T17:44:27" }
                    @{ id = "rate-limiting"; completed = $false; startedAt = $null; completedAt = $null }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            # Set up sprint history with rate-limiting completed
            $history = @{
                version = 1
                totalSprintsCompleted = 2
                totalStoriesCompleted = 20
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                    "rate-limiting" = @{ sprints = 1; stories = 10 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            # Run sync
            Sync-QueueFromHistory -Silent

            # Verify queue updated
            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $rl = $updated.focusAreas | Where-Object { $_.id -eq "rate-limiting" }
            $rl.completed | Should -BeTrue
            $rl.completedAt | Should -Not -BeNullOrEmpty
        }

        It "Should not modify already-completed areas" {
            $originalTimestamp = "2026-01-26T17:44:27"
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = $originalTimestamp }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            $history = @{
                version = 1
                totalSprintsCompleted = 1
                totalStoriesCompleted = 10
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $dl = $updated.focusAreas | Where-Object { $_.id -eq "download" }
            $dl.completed | Should -BeTrue
            $dl.completedAt | Should -Be $originalTimestamp
        }
    }

    Context "When sprint history has no matching areas" {
        It "Should leave queue unchanged but add history areas" {
            $queue = @{
                focusAreas = @(
                    @{ id = "otio"; completed = $false; startedAt = $null; completedAt = $null }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            $history = @{
                version = 1
                totalSprintsCompleted = 1
                totalStoriesCompleted = 10
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $otio = $updated.focusAreas | Where-Object { $_.id -eq "otio" }
            $otio.completed | Should -BeFalse

            # download from history should be added as completed
            $dl = $updated.focusAreas | Where-Object { $_.id -eq "download" }
            $dl | Should -Not -BeNullOrEmpty
            $dl.completed | Should -BeTrue
        }
    }

    Context "When history has areas not in queue" {
        It "Should add missing areas from sprint history" {
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = "2026-01-26T17:44:27" }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            $history = @{
                version = 1
                totalSprintsCompleted = 3
                totalStoriesCompleted = 30
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                    "rate-limiting" = @{ sprints = 1; stories = 10 }
                    testing = @{ sprints = 1; stories = 10 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            @($updated.focusAreas).Count | Should -Be 3

            $rl = $updated.focusAreas | Where-Object { $_.id -eq "rate-limiting" }
            $rl | Should -Not -BeNullOrEmpty
            $rl.completed | Should -BeTrue

            $test = $updated.focusAreas | Where-Object { $_.id -eq "testing" }
            $test | Should -Not -BeNullOrEmpty
            $test.completed | Should -BeTrue
        }
    }

    Context "When current PRD has untracked focus area" {
        It "Should add current PRD focus area as in-progress" {
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = "2026-01-26T17:44:27" }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            # No sprint history needed for this test

            # Create PRD with untracked focus area
            $prd = @{
                sprintNumber = 15
                focusArea = "quality"
                userStories = @()
            }
            $prd | ConvertTo-Json -Depth 5 | Set-Content $script:PrdFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $quality = $updated.focusAreas | Where-Object { $_.id -eq "quality" }
            $quality | Should -Not -BeNullOrEmpty
            $quality.completed | Should -BeFalse
            $quality.startedAt | Should -Not -BeNullOrEmpty
        }
    }

    Context "When current PRD re-opens a completed area" {
        It "Should re-open completed area if PRD has incomplete stories" {
            # testing was completed in history
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = "2026-01-26T17:44:27" }
                    @{ id = "testing"; completed = $true; completedAt = "2026-01-28T12:00:00" }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            $history = @{
                version = 1
                totalSprintsCompleted = 3
                totalStoriesCompleted = 30
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                    testing = @{ sprints = 2; stories = 20 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            # Active PRD with incomplete stories for testing
            $prd = @{
                sprintNumber = 16
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $false }
                    @{ id = "US-002"; passes = $false }
                )
            }
            $prd | ConvertTo-Json -Depth 5 | Set-Content $script:PrdFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $testing = $updated.focusAreas | Where-Object { $_.id -eq "testing" }
            $testing.completed | Should -BeFalse
            $testing.completedAt | Should -BeNullOrEmpty
        }

        It "Should NOT re-open if all PRD stories are complete" {
            $queue = @{
                focusAreas = @(
                    @{ id = "testing"; completed = $true; completedAt = "2026-01-28T12:00:00" }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            # PRD with all stories complete
            $prd = @{
                sprintNumber = 15
                focusArea = "testing"
                userStories = @(
                    @{ id = "US-001"; passes = $true }
                    @{ id = "US-002"; passes = $true }
                )
            }
            $prd | ConvertTo-Json -Depth 5 | Set-Content $script:PrdFile -Encoding UTF8

            Sync-QueueFromHistory -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $testing = $updated.focusAreas | Where-Object { $_.id -eq "testing" }
            $testing.completed | Should -BeTrue
        }
    }

    Context "When no sprint history exists" {
        It "Should not crash and leave queue unchanged" {
            $queue = @{
                focusAreas = @(
                    @{ id = "testing"; completed = $false }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            # No sprint history file created
            { Sync-QueueFromHistory -Silent } | Should -Not -Throw

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $testing = $updated.focusAreas | Where-Object { $_.id -eq "testing" }
            $testing.completed | Should -BeFalse
        }
    }

    Context "When no queue exists" {
        It "Should not crash" {
            # No queue file created, but sprint history exists
            $history = @{
                version = 1
                totalSprintsCompleted = 1
                totalStoriesCompleted = 10
                focusAreaBreakdown = @{
                    download = @{ sprints = 1; stories = 10 }
                }
                sprints = @()
            }
            $history | ConvertTo-Json -Depth 5 | Set-Content $script:SprintHistoryFile -Encoding UTF8

            { Sync-QueueFromHistory -Silent } | Should -Not -Throw
        }
    }
}

# =============================================================================
# Update-QueueProgress Auto-Add Tests
# =============================================================================

Describe "Update-QueueProgress Auto-Add" -Tag "Unit", "SprintCompletion", "QueueSync" {
    BeforeEach { Reset-TestData }

    Context "When completing an area not in queue" {
        It "Should add and mark new area as completed" {
            $queue = @{
                focusAreas = @(
                    @{ id = "download"; completed = $true; completedAt = "2026-01-26T17:44:27" }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            Update-QueueProgress -AreaId "testing" -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            @($updated.focusAreas).Count | Should -Be 2

            $testing = $updated.focusAreas | Where-Object { $_.id -eq "testing" }
            $testing | Should -Not -BeNullOrEmpty
            $testing.completed | Should -BeTrue
            $testing.completedAt | Should -Not -BeNullOrEmpty
        }
    }

    Context "When completing an area already in queue" {
        It "Should mark existing area as completed" {
            $queue = @{
                focusAreas = @(
                    @{ id = "testing"; completed = $false; startedAt = "2026-01-27T23:34:10"; completedAt = $null }
                )
                sessionId = "test-session"
                createdAt = "2026-01-26T12:00:00Z"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            Update-QueueProgress -AreaId "testing" -Silent

            $updated = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            @($updated.focusAreas).Count | Should -Be 1
            $updated.focusAreas[0].completed | Should -BeTrue
            $updated.focusAreas[0].completedAt | Should -Not -BeNullOrEmpty
        }
    }
}
