#Requires -Modules Pester

<#
.SYNOPSIS
    Unit tests for Ralph Loop exploration feature
.DESCRIPTION
    Tests cover:
    - Test-ShouldExplore logic (interval and git triggers)
    - Exploration config parsing
    - Exploration tracking variables
    - Invoke-PeriodicExplorationIfNeeded logic
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata"

    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Initialize mutable state hashtable (matches production pattern)
    $script:State = @{
        SessionId = 'test-session'; IterationCount = 0; ConsecutiveFailures = 0
        SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
        LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
        LastExplorationSummary = ''; LastExplorationTime = $null
        SprintExplorationContext = ''; LastExplorationCommit = ''
    }
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# TEST-SHOULDEXPLORE TESTS
# =============================================================================

Describe "Test-ShouldExplore" -Tag "Unit", "Exploration" {
    BeforeAll {
        # Define the function inline for testing
        function Test-ShouldExplore {
            param([string]$FocusArea)

            $config = $script:Config
            $explorationConfig = $config.exploration

            if (-not $explorationConfig -or -not $explorationConfig.enabled) {
                return @{ ShouldExplore = $false; Reason = "exploration_disabled" }
            }

            if (-not $explorationConfig.periodic -or -not $explorationConfig.periodic.enabled) {
                return @{ ShouldExplore = $false; Reason = "periodic_disabled" }
            }

            $intervalStories = $explorationConfig.periodic.intervalStories
            if (-not $intervalStories) { $intervalStories = 3 }

            if ($script:State.StoriesSinceExploration -ge $intervalStories) {
                return @{ ShouldExplore = $true; Reason = "interval" }
            }

            return @{ ShouldExplore = $false; Reason = "not_needed" }
        }
    }

    Context "When exploration is disabled" {
        BeforeEach {
            $script:Config = @{
                exploration = @{
                    enabled = $false
                }
            }
            $script:State.StoriesSinceExploration = 10
        }

        It "returns ShouldExplore false with reason exploration_disabled" {
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse
            $result.Reason | Should -Be "exploration_disabled"
        }
    }

    Context "When exploration config is missing" {
        BeforeEach {
            $script:Config = @{}
            $script:State.StoriesSinceExploration = 10
        }

        It "returns ShouldExplore false with reason exploration_disabled" {
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse
            $result.Reason | Should -Be "exploration_disabled"
        }
    }

    Context "When periodic exploration is disabled" {
        BeforeEach {
            $script:Config = @{
                exploration = @{
                    enabled = $true
                    periodic = @{
                        enabled = $false
                    }
                }
            }
            $script:State.StoriesSinceExploration = 10
        }

        It "returns ShouldExplore false with reason periodic_disabled" {
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse
            $result.Reason | Should -Be "periodic_disabled"
        }
    }

    Context "When interval threshold is reached" {
        BeforeEach {
            $script:Config = @{
                exploration = @{
                    enabled = $true
                    periodic = @{
                        enabled = $true
                        intervalStories = 3
                    }
                }
            }
        }

        It "returns ShouldExplore true when stories >= interval" {
            $script:State.StoriesSinceExploration = 3
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeTrue
            $result.Reason | Should -Be "interval"
        }

        It "returns ShouldExplore true when stories > interval" {
            $script:State.StoriesSinceExploration = 5
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeTrue
            $result.Reason | Should -Be "interval"
        }

        It "returns ShouldExplore false when stories < interval" {
            $script:State.StoriesSinceExploration = 2
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse
            $result.Reason | Should -Be "not_needed"
        }
    }

    Context "When using default interval" {
        BeforeEach {
            $script:Config = @{
                exploration = @{
                    enabled = $true
                    periodic = @{
                        enabled = $true
                        # intervalStories not set, should default to 3
                    }
                }
            }
        }

        It "uses default interval of 3 stories" {
            $script:State.StoriesSinceExploration = 3
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeTrue
        }

        It "does not trigger at 2 stories with default interval" {
            $script:State.StoriesSinceExploration = 2
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse
        }
    }

    Context "With custom interval setting" {
        It "respects intervalStories = 5" {
            $script:Config = @{
                exploration = @{
                    enabled = $true
                    periodic = @{
                        enabled = $true
                        intervalStories = 5
                    }
                }
            }

            $script:State.StoriesSinceExploration = 4
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeFalse

            $script:State.StoriesSinceExploration = 5
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeTrue
        }

        It "respects intervalStories = 1 (every story)" {
            $script:Config = @{
                exploration = @{
                    enabled = $true
                    periodic = @{
                        enabled = $true
                        intervalStories = 1
                    }
                }
            }

            $script:State.StoriesSinceExploration = 1
            $result = Test-ShouldExplore -FocusArea "quality"
            $result.ShouldExplore | Should -BeTrue
        }
    }
}

# =============================================================================
# EXPLORATION CONFIG TESTS
# =============================================================================

Describe "Exploration Config Parsing" -Tag "Unit", "Exploration", "Config" {
    BeforeAll {
        $script:ConfigFile = Join-Path $script:RalphDir "config\ralph-config.json"
    }

    It "loads exploration config from ralph-config.json" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration | Should -Not -BeNullOrEmpty
    }

    It "has exploration.enabled field" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration.enabled | Should -BeIn @($true, $false)
    }

    It "has sprintStart config section" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration.sprintStart | Should -Not -BeNullOrEmpty
        $config.exploration.sprintStart.enabled | Should -BeIn @($true, $false)
    }

    It "has periodic config section" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration.periodic | Should -Not -BeNullOrEmpty
        $config.exploration.periodic.enabled | Should -BeIn @($true, $false)
    }

    It "has intervalStories as positive integer" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration.periodic.intervalStories | Should -BeGreaterThan 0
    }

    It "has outputFile for sprint start exploration" {
        $config = Get-Content $script:ConfigFile -Raw | ConvertFrom-Json
        $config.exploration.sprintStart.outputFile | Should -Not -BeNullOrEmpty
    }
}

# =============================================================================
# EXPLORATION TRACKING TESTS
# =============================================================================

Describe "Exploration Tracking Variables" -Tag "Unit", "Exploration" {
    BeforeAll {
        # Simulate the tracking variables
        function Reset-ExplorationTracking {
            $script:State.StoriesSinceExploration = 0
            $script:State.LastExplorationSummary = ""
            $script:State.LastExplorationTime = $null
            $script:State.SprintExplorationContext = ""
        }

        function Increment-StoriesCounter {
            $script:State.StoriesSinceExploration++
        }

        function Set-ExplorationResult {
            param([string]$Summary)
            $script:State.LastExplorationSummary = $Summary
            $script:State.LastExplorationTime = Get-Date
        }
    }

    BeforeEach {
        Reset-ExplorationTracking
    }

    It "initializes StoriesSinceExploration to 0" {
        $script:State.StoriesSinceExploration | Should -Be 0
    }

    It "increments StoriesSinceExploration correctly" {
        Increment-StoriesCounter
        $script:State.StoriesSinceExploration | Should -Be 1

        Increment-StoriesCounter
        $script:State.StoriesSinceExploration | Should -Be 2

        Increment-StoriesCounter
        $script:State.StoriesSinceExploration | Should -Be 3
    }

    It "resets StoriesSinceExploration after exploration" {
        $script:State.StoriesSinceExploration = 5
        Reset-ExplorationTracking
        $script:State.StoriesSinceExploration | Should -Be 0
    }

    It "stores exploration summary" {
        $summary = "Test exploration summary with findings"
        Set-ExplorationResult -Summary $summary
        $script:State.LastExplorationSummary | Should -Be $summary
    }

    It "updates exploration timestamp" {
        $before = Get-Date
        Start-Sleep -Milliseconds 10
        Set-ExplorationResult -Summary "test"
        $script:State.LastExplorationTime | Should -BeGreaterThan $before
    }

    It "initializes SprintExplorationContext as empty" {
        $script:State.SprintExplorationContext | Should -BeNullOrEmpty
    }
}

# =============================================================================
# INVOKE-PERIODICEXPLORATIONIFNEEDED LOGIC TESTS
# =============================================================================

Describe "Invoke-PeriodicExplorationIfNeeded Logic" -Tag "Unit", "Exploration" {
    BeforeAll {
        # Mock the exploration check and execution
        $script:ExplorationExecuted = $false
        $script:ExplorationFocusArea = ""
        $script:ExplorationReason = ""

        function Mock-TestShouldExplore {
            param([string]$FocusArea, [bool]$ShouldTrigger, [string]$Reason = "interval")

            if ($ShouldTrigger) {
                return @{ ShouldExplore = $true; Reason = $Reason }
            }
            return @{ ShouldExplore = $false; Reason = "not_needed" }
        }

        function Invoke-PeriodicExplorationIfNeeded {
            param(
                [string]$FocusArea,
                [bool]$MockShouldTrigger = $false,
                [string]$MockReason = "interval"
            )

            $script:State.StoriesSinceExploration++

            $check = Mock-TestShouldExplore -FocusArea $FocusArea -ShouldTrigger $MockShouldTrigger -Reason $MockReason

            if ($check.ShouldExplore) {
                $script:ExplorationExecuted = $true
                $script:ExplorationFocusArea = $FocusArea
                $script:ExplorationReason = $check.Reason
                $script:State.StoriesSinceExploration = 0
                return $true
            }

            return $false
        }
    }

    BeforeEach {
        $script:State.StoriesSinceExploration = 0
        $script:ExplorationExecuted = $false
        $script:ExplorationFocusArea = ""
        $script:ExplorationReason = ""
    }

    It "increments story counter on each call" {
        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $false
        $script:State.StoriesSinceExploration | Should -Be 1

        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $false
        $script:State.StoriesSinceExploration | Should -Be 2
    }

    It "executes exploration when triggered" {
        $result = Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $true

        $result | Should -BeTrue
        $script:ExplorationExecuted | Should -BeTrue
        $script:ExplorationFocusArea | Should -Be "quality"
    }

    It "resets counter after exploration" {
        $script:State.StoriesSinceExploration = 2
        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $true

        $script:State.StoriesSinceExploration | Should -Be 0
    }

    It "does not execute exploration when not triggered" {
        $result = Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $false

        $result | Should -BeFalse
        $script:ExplorationExecuted | Should -BeFalse
    }

    It "preserves counter when exploration not triggered" {
        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $false
        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $false

        $script:State.StoriesSinceExploration | Should -Be 2
    }

    It "captures focus area for exploration" {
        Invoke-PeriodicExplorationIfNeeded -FocusArea "agents" -MockShouldTrigger $true
        $script:ExplorationFocusArea | Should -Be "agents"

        $script:ExplorationExecuted = $false
        Invoke-PeriodicExplorationIfNeeded -FocusArea "otio" -MockShouldTrigger $true
        $script:ExplorationFocusArea | Should -Be "otio"
    }

    It "captures exploration reason" {
        Invoke-PeriodicExplorationIfNeeded -FocusArea "quality" -MockShouldTrigger $true -MockReason "git_changes"
        $script:ExplorationReason | Should -Be "git_changes"
    }
}

# =============================================================================
# SPRINT-START EXPLORATION TESTS
# =============================================================================

Describe "Sprint-Start Exploration Logic" -Tag "Unit", "Exploration" {
    BeforeAll {
        function Test-SprintStartExplorationEnabled {
            param($Config)

            $explorationConfig = $Config.exploration
            return $explorationConfig -and `
                   $explorationConfig.enabled -and `
                   $explorationConfig.sprintStart -and `
                   $explorationConfig.sprintStart.enabled
        }
    }

    It "returns true when fully enabled" {
        $config = @{
            exploration = @{
                enabled = $true
                sprintStart = @{
                    enabled = $true
                }
            }
        }
        Test-SprintStartExplorationEnabled -Config $config | Should -BeTrue
    }

    It "returns false when exploration disabled" {
        $config = @{
            exploration = @{
                enabled = $false
                sprintStart = @{
                    enabled = $true
                }
            }
        }
        Test-SprintStartExplorationEnabled -Config $config | Should -BeFalse
    }

    It "returns false when sprintStart disabled" {
        $config = @{
            exploration = @{
                enabled = $true
                sprintStart = @{
                    enabled = $false
                }
            }
        }
        Test-SprintStartExplorationEnabled -Config $config | Should -BeFalse
    }

    It "returns false when exploration config missing" {
        $config = @{}
        Test-SprintStartExplorationEnabled -Config $config | Should -BeFalse
    }

    It "returns false when sprintStart section missing" {
        $config = @{
            exploration = @{
                enabled = $true
            }
        }
        Test-SprintStartExplorationEnabled -Config $config | Should -BeFalse
    }
}

# =============================================================================
# METRICS SCHEMA TESTS
# =============================================================================

Describe "Metrics V3 Schema with Exploration" -Tag "Unit", "Exploration", "Metrics" {
    BeforeAll {
        $script:V3Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms,exploration_triggered,exploration_reason,exploration_tokens"
    }

    It "has 24 columns in v3 schema" {
        $columns = $script:V3Header -split ','
        $columns.Count | Should -Be 24
    }

    It "includes exploration_triggered column" {
        $script:V3Header | Should -Match "exploration_triggered"
    }

    It "includes exploration_reason column" {
        $script:V3Header | Should -Match "exploration_reason"
    }

    It "includes exploration_tokens column" {
        $script:V3Header | Should -Match "exploration_tokens"
    }

    It "exploration columns are at the end" {
        $columns = $script:V3Header -split ','
        $columns[21] | Should -Be "exploration_triggered"
        $columns[22] | Should -Be "exploration_reason"
        $columns[23] | Should -Be "exploration_tokens"
    }
}

# =============================================================================
# EXPLORATION CONTEXT FILE TESTS
# =============================================================================

Describe "Exploration Context File Handling" -Tag "Unit", "Exploration" {
    BeforeAll {
        $script:ExplorationContextFile = Join-Path $script:TestDataDir "exploration_context.md"
    }

    BeforeEach {
        if (Test-Path $script:ExplorationContextFile) {
            Remove-Item $script:ExplorationContextFile -Force
        }
    }

    It "can write exploration context to file" {
        $content = @"
# Exploration Context - quality

## Current State
- Matching logic in src/matching.py
- 15 test files found

## Recent Changes
- Added confidence scoring

## Recommendations
- Focus on edge cases
"@
        $content | Set-Content $script:ExplorationContextFile -Encoding UTF8

        Test-Path $script:ExplorationContextFile | Should -BeTrue
    }

    It "can read exploration context from file" {
        $originalContent = "Test exploration context"
        $originalContent | Set-Content $script:ExplorationContextFile -Encoding UTF8

        $readContent = Get-Content $script:ExplorationContextFile -Raw
        $readContent.Trim() | Should -Be $originalContent
    }

    It "handles missing exploration context file gracefully" {
        $content = Get-Content $script:ExplorationContextFile -Raw -ErrorAction SilentlyContinue
        $content | Should -BeNullOrEmpty
    }
}
