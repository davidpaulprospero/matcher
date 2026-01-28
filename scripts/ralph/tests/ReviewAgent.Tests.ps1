# Phase 2: Core Quality Gate Tests
# Tests for Stories 2.1-2.5

BeforeAll {
    # Source needed functions from ralph.ps1 and lib/*.ps1
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')

    # Source Phase 2 functions + their dependencies
    $neededFunctions = @(
        'Read-JsonFile', 'Get-RalphConfig', 'Write-JsonNoBom',
        'Format-ReviewPrompt', 'Invoke-CodeReview',
        'Get-OptimalNextStory', 'Get-StoryFileTouches', 'Test-FileConflict',
        'Search-CriterionEvidence', 'Log-StoryVerification',
        'Get-SprintRetrospective', 'Get-RetrospectiveContext',
        'Import-HumanFeedback', 'Get-FeedbackForStory',
        'Get-StoryFailureContext', 'Get-DiffQualityScore'
    )

    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions $neededFunctions

    # Set up script-scoped variables
    $script:RalphDir = $TestDrive
    $script:ConfigFile = Join-Path $TestDrive 'ralph-config.json'
    $script:MetricsFile = Join-Path $TestDrive 'metrics.csv'
    $script:State = @{
        SessionId = 'test-session'; IterationCount = 0; ConsecutiveFailures = 0
        SessionStartTime = Get-Date; CurrentMode = 'Standard'; CurrentRetryCount = 0
        LastFocusAreaId = ''; LastStoryId = ''; StoriesSinceExploration = 0
        LastExplorationSummary = ''; LastExplorationTime = $null
        SprintExplorationContext = ''; LastExplorationCommit = ''
    }
    $script:SessionLogDir = Join-Path $TestDrive 'session_logs'
    $script:ProjectRoot = $TestDrive
    New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null

    # Create minimal ralph-config.json
    @{
        flags = @{
            llmAsJudgeQuality = $false
            acceptanceDrivenBackpressure = $false
        }
        review = @{ enabled = $false; model = "sonnet"; timeout = 180; minScoreToPass = 6 }
        ordering = @{ smart = $false; preferTestsFirst = $true }
        regression = @{ enabled = $true; blockOnRegression = $true }
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
}

Describe 'Format-ReviewPrompt' -Tag 'Unit', 'Phase2' {
    It 'Includes story ID and title' {
        $story = [PSCustomObject]@{
            id = 'US-001'
            title = 'Add authentication'
            acceptanceCriteria = @('Login works', 'Logout works')
        }
        $result = Format-ReviewPrompt -StoryId 'US-001' -Story $story -DiffOutput '+some code'
        $result | Should -Match 'US-001'
        $result | Should -Match 'Add authentication'
    }

    It 'Includes acceptance criteria numbered list' {
        $story = [PSCustomObject]@{
            id = 'US-002'
            title = 'Feature'
            acceptanceCriteria = @('First criterion', 'Second criterion', 'Third criterion')
        }
        $result = Format-ReviewPrompt -StoryId 'US-002' -Story $story -DiffOutput '+code'
        $result | Should -Match '1\. First criterion'
        $result | Should -Match '2\. Second criterion'
        $result | Should -Match '3\. Third criterion'
    }

    It 'Includes file operations context' {
        $fileOps = @{
            filesCreated = @(@{ path = 'src/new.py' })
            filesModified = @(@{ path = 'src/old.py' })
            filesDeleted = @('src/removed.py')
            totalFilesChanged = 3
        }
        $result = Format-ReviewPrompt -StoryId 'US-003' -Story $null -DiffOutput '+code' -FileOps $fileOps
        $result | Should -Match 'src/new.py'
        $result | Should -Match 'src/old.py'
        $result | Should -Match 'src/removed.py'
    }

    It 'Truncates large diffs' {
        $largeDiff = 'x' * 20000
        $result = Format-ReviewPrompt -StoryId 'US-004' -Story $null -DiffOutput $largeDiff
        $result | Should -Match 'TRUNCATED'
    }

    It 'Includes review focus areas' {
        $result = Format-ReviewPrompt -StoryId 'US-005' -Story $null -DiffOutput '+code'
        $result | Should -Match 'Architecture'
        $result | Should -Match 'Security'
        $result | Should -Match 'Regressions'
    }
}

Describe 'Get-OptimalNextStory' -Tag 'Unit', 'Phase2' {
    It 'Returns first incomplete when smart ordering disabled' {
        $stories = @(
            [PSCustomObject]@{ id = 'US-001'; title = 'First'; passes = $true },
            [PSCustomObject]@{ id = 'US-002'; title = 'Second'; passes = $false },
            [PSCustomObject]@{ id = 'US-003'; title = 'Third'; passes = $false }
        )
        $result = Get-OptimalNextStory -Stories $stories
        $result.id | Should -Be 'US-002'
    }

    It 'Returns null when all stories complete' {
        $stories = @(
            [PSCustomObject]@{ id = 'US-001'; passes = $true },
            [PSCustomObject]@{ id = 'US-002'; passes = $true }
        )
        $result = Get-OptimalNextStory -Stories $stories
        $result | Should -BeNullOrEmpty
    }

    It 'Prefers test stories when smart ordering enabled' {
        # Enable smart ordering
        @{
            flags = @{ llmAsJudgeQuality = $false }
            ordering = @{ smart = $true; preferTestsFirst = $true }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $stories = @(
            [PSCustomObject]@{ id = 'US-001'; title = 'Add feature X'; passes = $false },
            [PSCustomObject]@{ id = 'US-002'; title = 'Add unit tests for feature X'; passes = $false },
            [PSCustomObject]@{ id = 'US-003'; title = 'Implement logging'; passes = $false }
        )
        $result = Get-OptimalNextStory -Stories $stories
        $result.id | Should -Be 'US-002'

        # Restore config
        @{
            flags = @{ llmAsJudgeQuality = $false }
            review = @{ enabled = $false }
            ordering = @{ smart = $false; preferTestsFirst = $true }
            regression = @{ enabled = $true }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }

    It 'Penalizes stories with failure history' {
        @{
            flags = @{ llmAsJudgeQuality = $false }
            ordering = @{ smart = $true; preferTestsFirst = $true }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $stories = @(
            [PSCustomObject]@{ id = 'US-001'; title = 'Buggy story'; passes = $false },
            [PSCustomObject]@{ id = 'US-002'; title = 'Clean story'; passes = $false }
        )

        # Mock metrics with US-001 having 3 failures
        $metrics = @(
            [PSCustomObject]@{ story_id = 'US-001'; success = 'False' },
            [PSCustomObject]@{ story_id = 'US-001'; success = 'False' },
            [PSCustomObject]@{ story_id = 'US-001'; success = 'False' },
            [PSCustomObject]@{ story_id = 'US-002'; success = 'True' }
        )
        $result = Get-OptimalNextStory -Stories $stories -Metrics $metrics
        $result.id | Should -Be 'US-002'

        # Restore config
        @{
            flags = @{ llmAsJudgeQuality = $false }
            ordering = @{ smart = $false }
            regression = @{ enabled = $true }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }
}

Describe 'Get-StoryFileTouches' -Tag 'Unit', 'Phase2' {
    It 'Returns empty hashtable with no logs' {
        $result = Get-StoryFileTouches
        $result.Count | Should -Be 0
    }

    It 'Maps files from iteration logs' {
        $logData = @{
            storyId = 'US-001'
            fileOperations = @{
                filesCreated = @(@{ path = 'src/new.py' })
                filesModified = @(@{ path = 'src/old.py' })
            }
        }
        $logData | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:SessionLogDir 'iteration_1.json')

        $result = Get-StoryFileTouches
        $result.ContainsKey('US-001') | Should -Be $true
        $result['US-001'] | Should -Contain 'src/new.py'
        $result['US-001'] | Should -Contain 'src/old.py'

        Remove-Item (Join-Path $script:SessionLogDir 'iteration_1.json') -ErrorAction SilentlyContinue
    }
}

Describe 'Test-FileConflict' -Tag 'Unit', 'Phase2' {
    It 'Returns no conflict when no logs exist' {
        $result = Test-FileConflict -StoryId 'US-002'
        $result.hasConflict | Should -Be $false
    }

    It 'Detects file overlap between stories' {
        # Create logs showing US-001 touched src/config.py
        @{
            storyId = 'US-001'
            fileOperations = @{
                filesModified = @(@{ path = 'src/config.py' })
                filesCreated = @()
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:SessionLogDir 'iteration_10.json')

        # Create logs showing US-002 also touched src/config.py
        @{
            storyId = 'US-002'
            fileOperations = @{
                filesModified = @(@{ path = 'src/config.py' })
                filesCreated = @()
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:SessionLogDir 'iteration_11.json')

        $result = Test-FileConflict -StoryId 'US-002'
        $result.hasConflict | Should -Be $true
        $result.overlappingFiles | Should -Contain 'src/config.py'

        Remove-Item (Join-Path $script:SessionLogDir 'iteration_10.json') -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $script:SessionLogDir 'iteration_11.json') -ErrorAction SilentlyContinue
    }

    It 'Uses predictive matching from story title' {
        @{
            storyId = 'US-001'
            fileOperations = @{
                filesModified = @(@{ path = 'src/matching.py' })
                filesCreated = @()
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $script:SessionLogDir 'iteration_20.json')

        $story = [PSCustomObject]@{
            title = 'Improve matching algorithm'
            acceptanceCriteria = @('Update matching logic')
        }
        $result = Test-FileConflict -StoryId 'US-002' -Story $story
        $result.hasConflict | Should -Be $true

        Remove-Item (Join-Path $script:SessionLogDir 'iteration_20.json') -ErrorAction SilentlyContinue
    }
}

Describe 'Search-CriterionEvidence' -Tag 'Unit', 'Phase2' {
    It 'Returns not found with empty inputs' {
        $result = Search-CriterionEvidence -Criterion 'Test something'
        $result.found | Should -Be $false
        $result.confidence | Should -Be 0.0
    }

    It 'Finds evidence in Claude output by keyword matching' {
        $criterion = 'Create a new authentication module with JWT support'
        $claudeOutput = 'I have created the authentication module. JWT tokens are now validated in the middleware.'
        $result = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $claudeOutput
        $result.found | Should -Be $true
        $result.confidence | Should -BeGreaterThan 0.0
        $result.evidence | Should -Match 'Keywords found in Claude output'
    }

    It 'Finds evidence in git diff' {
        $criterion = 'Add test coverage for the parser'
        $diffOutput = @"
+def test_parser_basic():
+    result = parse("hello world")
+    assert result is not None
+
+def test_parser_edge_case():
+    result = parse("")
+    assert result == []
"@
        $result = Search-CriterionEvidence -Criterion $criterion -DiffOutput $diffOutput
        $result.found | Should -Be $true
        $result.evidence | Should -Match 'diff'
    }

    It 'Detects completion signals' {
        $criterion = 'Implement user login flow'
        $claudeOutput = 'The login flow has been implemented successfully. All acceptance criteria have been met and verified.'
        $result = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $claudeOutput
        $result.found | Should -Be $true
        $result.evidence | Should -Match 'Completion signal'
    }

    It 'Returns low confidence when no keywords match' {
        $criterion = 'Quantum flux capacitor alignment'
        $claudeOutput = 'I updated the README file and fixed a typo.'
        $result = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $claudeOutput
        $result.found | Should -Be $false
        $result.confidence | Should -BeLessThan 0.3
    }
}

Describe 'Log-StoryVerification (Evidence-Based)' -Tag 'Unit', 'Phase2' {
    It 'Searches output for per-criterion evidence when passed' {
        $story = [PSCustomObject]@{
            id = 'US-010'
            title = 'Add parser tests'
            acceptanceCriteria = @(
                'Create test file for parser module',
                'Include edge case testing'
            )
        }
        $claudeOutput = 'Created test_parser.py with comprehensive tests including edge cases for empty input.'
        $diffOutput = "+def test_parser_edge_case():`n+    pass"

        Log-StoryVerification -StoryId 'US-010' -Story $story -Iteration 1 -Passed $true -ClaudeOutput $claudeOutput -DiffOutput $diffOutput

        $verFile = Join-Path $script:SessionLogDir 'story_US-010_verification.json'
        Test-Path $verFile | Should -Be $true
        $ver = Get-Content $verFile -Raw | ConvertFrom-Json
        $ver.evidenceBased | Should -Be $true
        $ver.criteriaMet | Should -BeGreaterOrEqual 1
        $ver.criteriaTotal | Should -Be 2
    }

    It 'Falls back to simple verification without output' {
        $story = [PSCustomObject]@{
            id = 'US-011'
            title = 'Simple story'
            acceptanceCriteria = @('Do the thing')
        }

        Log-StoryVerification -StoryId 'US-011' -Story $story -Iteration 1 -Passed $true

        $verFile = Join-Path $script:SessionLogDir 'story_US-011_verification.json'
        $ver = Get-Content $verFile -Raw | ConvertFrom-Json
        $ver.evidenceBased | Should -Be $false
        $ver.criteriaMet | Should -Be 1
    }

    It 'Marks all criteria as unverified when story failed' {
        $story = [PSCustomObject]@{
            id = 'US-012'
            title = 'Failed story'
            acceptanceCriteria = @('First', 'Second')
        }

        Log-StoryVerification -StoryId 'US-012' -Story $story -Iteration 1 -Passed $false -ClaudeOutput 'Some output'

        $verFile = Join-Path $script:SessionLogDir 'story_US-012_verification.json'
        $ver = Get-Content $verFile -Raw | ConvertFrom-Json
        $ver.overallVerified | Should -Be $false
        $ver.criteriaMet | Should -Be 0
    }
}

Describe 'Get-SprintRetrospective' -Tag 'Unit', 'Phase2' {
    It 'Returns empty retrospective with no metrics' {
        $metricsFile = Join-Path $TestDrive 'metrics.csv'
        Remove-Item $metricsFile -ErrorAction SilentlyContinue

        $result = Get-SprintRetrospective
        $result.failurePatterns.Count | Should -Be 0
        $result.lessons.Count | Should -Be 0
    }

    It 'Detects recurring failure patterns' {
        $metricsFile = Join-Path $TestDrive 'metrics.csv'
        @(
            [PSCustomObject]@{ story_id = 'US-001'; success = 'False'; error_category = 'test_failure'; lines_added = '10'; lines_deleted = '5'; timeout = 'False' },
            [PSCustomObject]@{ story_id = 'US-001'; success = 'False'; error_category = 'test_failure'; lines_added = '15'; lines_deleted = '3'; timeout = 'False' },
            [PSCustomObject]@{ story_id = 'US-002'; success = 'True'; error_category = ''; lines_added = '50'; lines_deleted = '10'; timeout = 'False' }
        ) | Export-Csv -Path $metricsFile -NoTypeInformation

        $result = Get-SprintRetrospective
        $result.failurePatterns.Count | Should -BeGreaterThan 0
        $result.failurePatterns[0].category | Should -Be 'test_failure'

        Remove-Item $metricsFile -ErrorAction SilentlyContinue
    }

    It 'Detects excessive retry patterns' {
        $metricsFile = Join-Path $TestDrive 'metrics.csv'
        @(
            [PSCustomObject]@{ story_id = 'US-005'; success = 'False'; error_category = 'compile'; lines_added = '0'; lines_deleted = '0'; timeout = 'False' },
            [PSCustomObject]@{ story_id = 'US-005'; success = 'False'; error_category = 'compile'; lines_added = '0'; lines_deleted = '0'; timeout = 'False' },
            [PSCustomObject]@{ story_id = 'US-005'; success = 'False'; error_category = 'compile'; lines_added = '0'; lines_deleted = '0'; timeout = 'False' },
            [PSCustomObject]@{ story_id = 'US-005'; success = 'True'; error_category = ''; lines_added = '20'; lines_deleted = '5'; timeout = 'False' }
        ) | Export-Csv -Path $metricsFile -NoTypeInformation

        $result = Get-SprintRetrospective
        $result.retryPatterns.Count | Should -BeGreaterThan 0
        $result.retryPatterns[0].storyId | Should -Be 'US-005'

        Remove-Item $metricsFile -ErrorAction SilentlyContinue
    }

    It 'Generates lessons and recommendations for low pass rate' {
        $metricsFile = Join-Path $TestDrive 'metrics.csv'
        @(
            [PSCustomObject]@{ story_id = 'US-001'; success = 'True'; error_category = ''; lines_added = '20'; lines_deleted = '5'; timeout = 'False' }
        ) | Export-Csv -Path $metricsFile -NoTypeInformation

        $prd = [PSCustomObject]@{
            userStories = @(
                [PSCustomObject]@{ id = 'US-001'; passes = $true },
                [PSCustomObject]@{ id = 'US-002'; passes = $false },
                [PSCustomObject]@{ id = 'US-003'; passes = $false },
                [PSCustomObject]@{ id = 'US-004'; passes = $false }
            )
        }

        $result = Get-SprintRetrospective -Prd $prd
        ($result.lessons | Where-Object { $_ -match '25%' }) | Should -Not -BeNullOrEmpty
        $result.recommendations.Count | Should -BeGreaterThan 0

        Remove-Item $metricsFile -ErrorAction SilentlyContinue
    }
}

Describe 'Get-RetrospectiveContext' -Tag 'Unit', 'Phase2' {
    It 'Returns empty string with no data' {
        $result = Get-RetrospectiveContext -Retro @{ lessons = @(); failurePatterns = @(); recommendations = @() }
        $result | Should -Be ""
    }

    It 'Formats lessons into markdown context' {
        $retro = @{
            lessons = @('Sprint pass rate: 75%', 'Most common failure: test_failure (3x)')
            failurePatterns = @(@{ category = 'test_failure'; count = 3; lesson = 'Fix tests first' })
            recommendations = @('Consider smaller stories')
        }
        $result = Get-RetrospectiveContext -Retro $retro
        $result | Should -Match 'Lessons from Previous Sprint'
        $result | Should -Match '75%'
        $result | Should -Match 'test_failure'
        $result | Should -Match 'smaller stories'
    }

    It 'Returns empty for null input' {
        $result = Get-RetrospectiveContext -Retro $null
        $result | Should -Be ""
    }
}
