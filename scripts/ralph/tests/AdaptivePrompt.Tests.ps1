# Phase 3: Intelligence Layer Tests
# Tests for Stories 3.1-3.5

BeforeAll {
    # Source needed functions from ralph.ps1 and lib/*.ps1
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')

    $neededFunctions = @(
        'Get-RalphConfig', 'Write-JsonNoBom',
        'Build-StoryPrompt', 'Get-RelevantFilesForStory',
        'Measure-CodebaseHealth', 'Compare-HealthMetrics',
        'Test-TokenBudget', 'Get-PromptEffectivenessHistory', 'Get-PromptRecommendation',
        'Get-StoryFailureContext', 'Get-FeedbackForStory', 'Get-RetrospectiveContext',
        'Import-HumanFeedback', 'Test-FileConflict', 'Get-StoryFileTouches',
        'Get-SprintTokenBudget', 'Search-CriterionEvidence'
    )

    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions $neededFunctions

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

    @{
        flags = @{ llmAsJudgeQuality = $false; acceptanceDrivenBackpressure = $false }
        review = @{ enabled = $false }
        ordering = @{ smart = $false }
        regression = @{ enabled = $true }
        budget = @{ enabled = $false; maxTokensPerSprint = 500000; warnAtPercent = 80 }
        prompts = @{ adaptive = $false; maxLength = 4000 }
        health = @{ enabled = $true; trackCoverage = $false; trackTechDebt = $false }
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
}

Describe 'Build-StoryPrompt' -Tag 'Unit', 'Phase3' {
    It 'Returns basic prompt when no features enabled' {
        $result = Build-StoryPrompt -StoryId 'US-001' -Story $null
        $result | Should -Match 'Work on story US-001'
        $result | Should -Match 'prd.json'
    }

    It 'Includes acceptance criteria when backpressure enabled' {
        @{
            flags = @{ acceptanceDrivenBackpressure = $true }
            prompts = @{ adaptive = $false; maxLength = 4000 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $story = [PSCustomObject]@{
            title = 'Test feature'
            acceptanceCriteria = @('Criterion A', 'Criterion B')
        }
        $result = Build-StoryPrompt -StoryId 'US-002' -Story $story
        $result | Should -Match 'ACCEPTANCE CRITERIA'
        $result | Should -Match 'Criterion A'
        $result | Should -Match 'Criterion B'

        # Restore config
        @{
            flags = @{ acceptanceDrivenBackpressure = $false }
            prompts = @{ adaptive = $false; maxLength = 4000 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }

    It 'Truncates prompt at maxLength' {
        @{
            flags = @{ acceptanceDrivenBackpressure = $false }
            prompts = @{ adaptive = $false; maxLength = 100 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        # Create a story with many criteria to make a long prompt
        $story = [PSCustomObject]@{
            title = 'Long story'
            acceptanceCriteria = @('A very long criterion ' * 10)
        }
        $result = Build-StoryPrompt -StoryId 'US-003' -Story $story
        # Even without backpressure, the basic prompt should be present
        $result.Length | Should -BeLessOrEqual 200  # 100 chars + truncation message

        # Restore config
        @{
            flags = @{ acceptanceDrivenBackpressure = $false }
            prompts = @{ adaptive = $false; maxLength = 4000 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }
}

Describe 'Get-RelevantFilesForStory' -Tag 'Unit', 'Phase3' {
    It 'Returns empty array for null story' {
        $result = Get-RelevantFilesForStory -Story $null
        $result.Count | Should -Be 0
    }

    It 'Returns empty array when no keywords found' {
        $story = [PSCustomObject]@{
            title = 'Do'
            acceptanceCriteria = @('It')
        }
        $result = Get-RelevantFilesForStory -Story $story
        $result.Count | Should -Be 0
    }

    It 'Finds files matching story keywords' {
        # Create test files in TestDrive
        $srcDir = Join-Path $TestDrive 'src'
        New-Item -ItemType Directory -Path $srcDir -Force | Out-Null
        'def authenticate(): pass' | Set-Content (Join-Path $srcDir 'authentication.py')
        'def process(): pass' | Set-Content (Join-Path $srcDir 'processor.py')

        $story = [PSCustomObject]@{
            title = 'Fix authentication module'
            acceptanceCriteria = @('Update authentication logic')
        }
        $result = Get-RelevantFilesForStory -Story $story -MaxFiles 5
        $result.Count | Should -BeGreaterThan 0
        ($result -join ',') | Should -Match 'authentication'
    }
}

Describe 'Measure-CodebaseHealth' -Tag 'Unit', 'Phase3' {
    It 'Returns null when health tracking disabled' {
        @{
            flags = @{}
            health = @{ enabled = $false }
            prompts = @{ adaptive = $false }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        $result = Measure-CodebaseHealth
        $result | Should -BeNullOrEmpty

        # Restore
        @{
            flags = @{}
            health = @{ enabled = $true; trackTechDebt = $false }
            prompts = @{ adaptive = $false }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }

    It 'Returns health metrics structure when enabled' {
        $result = Measure-CodebaseHealth
        $result | Should -Not -BeNullOrEmpty
        $result.ContainsKey('tests') | Should -Be $true
        $result.ContainsKey('codebase') | Should -Be $true
        $result.ContainsKey('techDebt') | Should -Be $true
        $result.ContainsKey('measuredAt') | Should -Be $true
    }
}

Describe 'Compare-HealthMetrics' -Tag 'Unit', 'Phase3' {
    It 'Returns baseline when no previous data' {
        Remove-Item (Join-Path $TestDrive 'health_metrics.json') -ErrorAction SilentlyContinue
        $current = @{
            tests = @{ total = 50; passed = 48; failed = 2; passRate = 0.96 }
            codebase = @{ pyFiles = 30; ps1Files = 5; totalLines = 10000 }
            techDebt = @{ todoCount = 3; fixmeCount = 1; hackCount = 0 }
        }
        $result = Compare-HealthMetrics -Current $current
        $result.isBaseline | Should -Be $true
    }

    It 'Detects test regression' {
        # Write previous health with 100% pass rate
        @{
            tests = @{ total = 50; passed = 50; failed = 0; passRate = 1.0 }
            codebase = @{ pyFiles = 30; ps1Files = 5; totalLines = 10000 }
            techDebt = @{ todoCount = 3; fixmeCount = 1; hackCount = 0 }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'health_metrics.json')

        $current = @{
            tests = @{ total = 50; passed = 45; failed = 5; passRate = 0.9 }
            codebase = @{ pyFiles = 30; ps1Files = 5; totalLines = 10000 }
            techDebt = @{ todoCount = 3; fixmeCount = 1; hackCount = 0 }
        }
        $result = Compare-HealthMetrics -Current $current
        $result.isBaseline | Should -Be $false
        $downTrends = @($result.trends | Where-Object { $_.direction -eq 'down' })
        $downTrends.Count | Should -BeGreaterThan 0

        Remove-Item (Join-Path $TestDrive 'health_metrics.json') -ErrorAction SilentlyContinue
    }
}

Describe 'Test-TokenBudget' -Tag 'Unit', 'Phase3' {
    It 'Returns false when budget not enabled' {
        $result = Test-TokenBudget
        $result | Should -Be $false
    }

    It 'Returns false when budget enabled but not exceeded' {
        @{
            flags = @{}
            budget = @{ enabled = $true; maxTokensPerSprint = 1000000; warnAtPercent = 80 }
            prompts = @{ adaptive = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        # No metrics means 0 tokens used - not exceeded
        Remove-Item (Join-Path $TestDrive 'metrics.csv') -ErrorAction SilentlyContinue
        $result = Test-TokenBudget
        $result | Should -Be $false

        # Restore
        @{
            flags = @{}
            budget = @{ enabled = $false }
            prompts = @{ adaptive = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }
}

Describe 'Get-PromptEffectivenessHistory' -Tag 'Unit', 'Phase3' {
    It 'Returns empty array when no file exists' {
        $result = Get-PromptEffectivenessHistory
        $result.Count | Should -Be 0
    }

    It 'Reads JSONL records' {
        $effectFile = Join-Path $script:SessionLogDir 'prompt_effectiveness.jsonl'
        @(
            '{"iteration":1,"promptType":"story_work","effectiveness":0.8,"promptHash":"abc123"}',
            '{"iteration":2,"promptType":"story_work","effectiveness":0.6,"promptHash":"def456"}'
        ) | Set-Content $effectFile

        $result = Get-PromptEffectivenessHistory
        $result.Count | Should -Be 2
        $result[0].promptType | Should -Be 'story_work'

        Remove-Item $effectFile -ErrorAction SilentlyContinue
    }
}

Describe 'Get-PromptRecommendation' -Tag 'Unit', 'Phase3' {
    It 'Returns defaults when adaptive disabled' {
        $result = Get-PromptRecommendation
        $result.useFailureContext | Should -Be $true
        $result.useFeedback | Should -Be $true
        $result.useCriteria | Should -Be $true
        $result.useRetrospective | Should -Be $true
        $result.useFileHints | Should -Be $true
    }

    It 'Returns defaults with insufficient history' {
        @{
            flags = @{}
            prompts = @{ adaptive = $true; maxLength = 4000 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')

        # Only 2 records - less than minimum 5
        $effectFile = Join-Path $script:SessionLogDir 'prompt_effectiveness.jsonl'
        @(
            '{"iteration":1,"promptType":"story_work","effectiveness":0.8}',
            '{"iteration":2,"promptType":"story_work","effectiveness":0.6}'
        ) | Set-Content $effectFile

        $result = Get-PromptRecommendation
        $result.useFailureContext | Should -Be $true

        Remove-Item $effectFile -ErrorAction SilentlyContinue

        # Restore config
        @{
            flags = @{}
            prompts = @{ adaptive = $false; maxLength = 4000 }
            budget = @{ enabled = $false }
            ordering = @{ smart = $false }
            health = @{ enabled = $true; trackTechDebt = $false }
        } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $TestDrive 'ralph-config.json')
    }
}
