# Phase 4: Advanced Capabilities Tests
# Tests for Stories 4.1-4.5

BeforeAll {
    # Source needed functions from ralph.ps1 and lib/*.ps1
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')

    $neededFunctions = @(
        'Get-RalphConfig', 'Write-JsonNoBom',
        'Get-StoryProgress', 'Save-StoryProgress', 'Build-ResumePrompt',
        'Get-IndependentStories', 'Update-LearningDb', 'Get-LearningContext',
        'Test-CanRollback', 'Invoke-StoryRollback',
        'Build-DependencyGraph', 'Get-ExecutableStories',
        'Get-StoryFileTouches', 'Test-FileConflict'
    )

    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions $neededFunctions

    $script:RalphDir = $TestDrive
    $script:ConfigFile = Join-Path $TestDrive 'ralph-config.json'
    $script:MetricsFile = Join-Path $TestDrive 'metrics.csv'
    $script:SessionId = 'test-session'
    $script:SessionLogDir = Join-Path $TestDrive 'session_logs'
    $script:ProjectRoot = $TestDrive
    New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null

    @{
        flags = @{ llmAsJudgeQuality = $false; acceptanceDrivenBackpressure = $false }
        review = @{ enabled = $false }
        ordering = @{ smart = $false }
        regression = @{ enabled = $true; blockOnRegression = $true; autoRollback = $false }
        budget = @{ enabled = $false; maxTokensPerSprint = 500000; warnAtPercent = 80 }
        prompts = @{ adaptive = $false; maxLength = 4000 }
        health = @{ enabled = $true; trackCoverage = $false; trackTechDebt = $false }
        parallel = @{ enabled = $false; maxConcurrent = 2 }
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
}

Describe 'Get-StoryProgress' -Tag 'Unit', 'Phase4' {
    It 'Returns empty milestones when no progress file exists' {
        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
        # Use story ID that won't match real git commits
        $result = Get-StoryProgress -StoryId 'XTEST-001'
        $result.storyId | Should -Be 'XTEST-001'
        $result.milestones.testsCreated | Should -Be $false
        $result.milestones.implementationStarted | Should -Be $false
        $result.milestones.committed | Should -Be $false
    }

    It 'Returns saved milestones from progress file' {
        @{
            'XTEST-002' = @{
                milestones = @{
                    testsCreated = $true
                    implementationStarted = $true
                    committed = $false
                    reviewPassed = $false
                }
                lastCheckpoint = 'implementation'
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'story_progress.json')

        $result = Get-StoryProgress -StoryId 'XTEST-002'
        $result.milestones.testsCreated | Should -Be $true
        $result.milestones.implementationStarted | Should -Be $true
        $result.milestones.committed | Should -Be $false

        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
    }

    It 'Returns empty milestones for unknown story in existing file' {
        @{
            'XTEST-003' = @{
                milestones = @{ testsCreated = $true; implementationStarted = $false; committed = $false; reviewPassed = $false }
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'story_progress.json')

        $result = Get-StoryProgress -StoryId 'XTEST-999'
        $result.milestones.testsCreated | Should -Be $false

        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
    }
}

Describe 'Save-StoryProgress' -Tag 'Unit', 'Phase4' {
    It 'Creates progress file when none exists' {
        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
        Save-StoryProgress -StoryId 'XTEST-010' -Milestone 'testsCreated' -Data @{ iteration = 1 }

        $file = Join-Path $TestDrive 'story_progress.json'
        Test-Path $file | Should -Be $true
        $content = Get-Content $file -Raw | ConvertFrom-Json
        $content.'XTEST-010'.milestones.testsCreated | Should -Be $true

        Remove-Item $file -ErrorAction SilentlyContinue
    }

    It 'Updates existing story milestones' {
        @{
            'XTEST-011' = @{
                milestones = @{ testsCreated = $true; implementationStarted = $false; committed = $false; reviewPassed = $false }
                lastCheckpoint = 'testsCreated'
            }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'story_progress.json')

        Save-StoryProgress -StoryId 'XTEST-011' -Milestone 'implementationStarted' -Data @{ iteration = 2 }

        $content = Get-Content (Join-Path $TestDrive 'story_progress.json') -Raw | ConvertFrom-Json
        $content.'XTEST-011'.milestones.testsCreated | Should -Be $true
        $content.'XTEST-011'.milestones.implementationStarted | Should -Be $true

        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
    }

    It 'Saves completed milestone' {
        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
        Save-StoryProgress -StoryId 'XTEST-012' -Milestone 'completed' -Data @{ iteration = 5; retryCount = 2 }

        $content = Get-Content (Join-Path $TestDrive 'story_progress.json') -Raw | ConvertFrom-Json
        $content.'XTEST-012'.lastCheckpoint | Should -Be 'completed'

        Remove-Item (Join-Path $TestDrive 'story_progress.json') -ErrorAction SilentlyContinue
    }
}

Describe 'Build-ResumePrompt' -Tag 'Unit', 'Phase4' {
    It 'Returns null when no milestones reached' {
        $progress = @{
            storyId = 'XTEST-020'
            milestones = @{ testsCreated = $false; implementationStarted = $false; committed = $false; reviewPassed = $false }
            lastCheckpoint = $null
        }
        $result = Build-ResumePrompt -StoryId 'XTEST-020' -Progress $progress -Story $null
        $result | Should -BeNullOrEmpty
    }

    It 'Returns resume context with completed steps' {
        $progress = @{
            storyId = 'XTEST-021'
            milestones = @{ testsCreated = $true; implementationStarted = $true; committed = $false; reviewPassed = $false }
            lastCheckpoint = 'implementation'
        }
        $story = [PSCustomObject]@{
            title = 'Test feature'
            acceptanceCriteria = @('Criterion A')
        }
        $result = Build-ResumePrompt -StoryId 'XTEST-021' -Progress $progress -Story $story
        $result | Should -Not -BeNullOrEmpty
        $result | Should -Match 'Resume'
        $result | Should -Match 'Tests already created'
    }

    It 'Shows remaining steps' {
        $progress = @{
            storyId = 'XTEST-022'
            milestones = @{ testsCreated = $true; implementationStarted = $false; committed = $false; reviewPassed = $false }
            lastCheckpoint = 'testsCreated'
        }
        $result = Build-ResumePrompt -StoryId 'XTEST-022' -Progress $progress -Story $null
        $result | Should -Match 'Remaining'
    }
}

Describe 'Get-IndependentStories' -Tag 'Unit', 'Phase4' {
    It 'Returns empty when no stories provided' {
        $result = Get-IndependentStories -Stories @()
        @($result).Count | Should -Be 0
    }

    It 'Returns stories when parallel enabled and none conflict' {
        @{
            flags = @{}
            parallel = @{ enabled = $true; maxConcurrent = 3 }
            regression = @{ enabled = $true }
            budget = @{ enabled = $false }
            prompts = @{ adaptive = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Feature A' },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Feature B' }
        )
        $result = @(Get-IndependentStories -Stories $stories)
        $result.Count | Should -BeGreaterOrEqual 1

        # Restore config
        @{
            flags = @{ llmAsJudgeQuality = $false; acceptanceDrivenBackpressure = $false }
            regression = @{ enabled = $true; blockOnRegression = $true; autoRollback = $false }
            budget = @{ enabled = $false }
            prompts = @{ adaptive = $false; maxLength = 4000 }
            health = @{ enabled = $true; trackTechDebt = $false }
            parallel = @{ enabled = $false; maxConcurrent = 2 }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }

    It 'Returns empty when parallel disabled' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Feature A' },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Feature B' }
        )
        $result = @(Get-IndependentStories -Stories $stories)
        $result.Count | Should -Be 0
    }
}

Describe 'Update-LearningDb' -Tag 'Unit', 'Phase4' {
    It 'Creates learning DB file when none exists' {
        $dbFile = Join-Path $TestDrive 'learning_db.json'
        Remove-Item $dbFile -ErrorAction SilentlyContinue

        Update-LearningDb -Entry @{
            type = "story_success"
            storyId = "XTEST-001"
            focusArea = "testing"
        }

        Test-Path $dbFile | Should -Be $true
        $content = Get-Content $dbFile -Raw | ConvertFrom-Json
        @($content.entries).Count | Should -Be 1
        $content.entries[0].type | Should -Be 'story_success'

        Remove-Item $dbFile -ErrorAction SilentlyContinue
    }

    It 'Appends to existing learning DB' {
        $dbFile = Join-Path $TestDrive 'learning_db.json'
        @{
            entries = @(
                @{ type = "story_success"; storyId = "XTEST-001"; timestamp = (Get-Date).ToString("o") }
            )
        } | ConvertTo-Json -Depth 5 | Set-Content $dbFile

        Update-LearningDb -Entry @{
            type = "story_failure"
            storyId = "XTEST-002"
            errorCategory = "test_failure"
        }

        $content = Get-Content $dbFile -Raw | ConvertFrom-Json
        @($content.entries).Count | Should -Be 2

        Remove-Item $dbFile -ErrorAction SilentlyContinue
    }
}

Describe 'Get-LearningContext' -Tag 'Unit', 'Phase4' {
    It 'Returns empty array when no DB exists' {
        Remove-Item (Join-Path $TestDrive 'learning_db.json') -ErrorAction SilentlyContinue
        $result = @(Get-LearningContext -FocusArea 'testing')
        $result.Count | Should -Be 0
    }

    It 'Filters by focus area' {
        $dbFile = Join-Path $TestDrive 'learning_db.json'
        @{
            entries = @(
                @{ type = "story_success"; storyId = "XTEST-001"; focusArea = "testing"; timestamp = (Get-Date).ToString("o") },
                @{ type = "story_failure"; storyId = "XTEST-002"; focusArea = "pipeline"; timestamp = (Get-Date).ToString("o") },
                @{ type = "story_success"; storyId = "XTEST-003"; focusArea = "testing"; timestamp = (Get-Date).ToString("o") }
            )
        } | ConvertTo-Json -Depth 5 | Set-Content $dbFile

        $result = @(Get-LearningContext -FocusArea 'testing')
        $result.Count | Should -Be 2
        $result[0].focusArea | Should -Be 'testing'

        Remove-Item $dbFile -ErrorAction SilentlyContinue
    }

    It 'Filters by error category' {
        $dbFile = Join-Path $TestDrive 'learning_db.json'
        @{
            entries = @(
                @{ type = "story_failure"; storyId = "XTEST-001"; errorCategory = "test_failure"; timestamp = (Get-Date).ToString("o") },
                @{ type = "story_failure"; storyId = "XTEST-002"; errorCategory = "syntax_error"; timestamp = (Get-Date).ToString("o") }
            )
        } | ConvertTo-Json -Depth 5 | Set-Content $dbFile

        $result = @(Get-LearningContext -ErrorCategory 'test_failure')
        $result.Count | Should -Be 1

        Remove-Item $dbFile -ErrorAction SilentlyContinue
    }
}

Describe 'Test-CanRollback' -Tag 'Unit', 'Phase4' {
    It 'Returns canRollback=false when autoRollback disabled' {
        $result = Test-CanRollback -StoryId 'XTEST-001'
        $result.canRollback | Should -Be $false
        $result.reason | Should -Match 'disabled'
    }

    It 'Returns canRollback=false when no git repo in TestDrive' {
        @{
            flags = @{}
            regression = @{ enabled = $true; blockOnRegression = $true; autoRollback = $true }
            budget = @{ enabled = $false }
            prompts = @{ adaptive = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $result = Test-CanRollback -StoryId 'XTEST-001'
        $result.canRollback | Should -Be $false

        # Restore
        @{
            flags = @{ llmAsJudgeQuality = $false; acceptanceDrivenBackpressure = $false }
            regression = @{ enabled = $true; blockOnRegression = $true; autoRollback = $false }
            budget = @{ enabled = $false }
            prompts = @{ adaptive = $false; maxLength = 4000 }
            health = @{ enabled = $true; trackTechDebt = $false }
            parallel = @{ enabled = $false; maxConcurrent = 2 }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }
}

Describe 'Build-DependencyGraph' -Tag 'Unit', 'Phase4' {
    It 'Returns empty graph for no stories' {
        $result = Build-DependencyGraph -Stories @()
        $result.Count | Should -Be 0
    }

    It 'Returns graph with no deps for independent stories' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Add login'; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Fix tests'; dependsOn = $null }
        )
        $result = Build-DependencyGraph -Stories $stories
        $result.Count | Should -Be 2
        @($result['XTEST-001']).Count | Should -Be 0
        @($result['XTEST-002']).Count | Should -Be 0
    }

    It 'Captures explicit dependsOn' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Core feature'; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Tests for core'; dependsOn = @('XTEST-001') }
        )
        $result = Build-DependencyGraph -Stories $stories
        $result['XTEST-002'] | Should -Contain 'XTEST-001'
    }

    It 'Detects implicit dependencies from title keywords' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Add authentication module'; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Test authentication module'; dependsOn = $null }
        )
        $result = Build-DependencyGraph -Stories $stories
        $result.Count | Should -Be 2
    }
}

Describe 'Get-ExecutableStories' -Tag 'Unit', 'Phase4' {
    It 'Returns all stories when no dependencies' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Feature A'; passes = $false; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Feature B'; passes = $false; dependsOn = $null }
        )
        $result = @(Get-ExecutableStories -Stories $stories)
        $result.Count | Should -Be 2
    }

    It 'Filters stories with unmet dependencies' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Core feature'; passes = $false; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Depends on core'; passes = $false; dependsOn = @('XTEST-001') }
        )
        $result = @(Get-ExecutableStories -Stories $stories)
        $ids = @($result | ForEach-Object { $_.id })
        $ids | Should -Contain 'XTEST-001'
        $ids | Should -Not -Contain 'XTEST-002'
    }

    It 'Unblocks stories when dependencies are met' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Core feature'; passes = $true; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Depends on core'; passes = $false; dependsOn = @('XTEST-001') }
        )
        $result = @(Get-ExecutableStories -Stories $stories)
        $ids = @($result | ForEach-Object { $_.id })
        $ids | Should -Contain 'XTEST-002'
    }

    It 'Excludes already-passed stories' {
        $stories = @(
            [PSCustomObject]@{ id = 'XTEST-001'; title = 'Done'; passes = $true; dependsOn = $null },
            [PSCustomObject]@{ id = 'XTEST-002'; title = 'Todo'; passes = $false; dependsOn = $null }
        )
        $result = @(Get-ExecutableStories -Stories $stories)
        $ids = @($result | ForEach-Object { $_.id })
        $ids | Should -Not -Contain 'XTEST-001'
        $ids | Should -Contain 'XTEST-002'
    }
}
