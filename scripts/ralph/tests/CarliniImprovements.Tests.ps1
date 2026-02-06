#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Carlini-inspired improvements:
    1. Early-exit tuning (grace period 30s, git commit detection)
    2. Structured test output (Format-TestSummary, reduced context)
    3. Randomized fast test mode (T3-fast vs T3-full branching)
    4. Living sprint progress document (Update/Get/Reset-SprintProgress)
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

# ============================================================================
# Step 1: Early-Exit Tuning (config + source inspection)
# ============================================================================

Describe 'Early-Exit Tuning' -Tag 'Unit', 'Carlini' {
    It 'config gracePeriodSeconds is 30' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds | Should -Be 30
    }

    It 'claude.ps1 captures git HEAD on story completion detection' {
        $claudePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\claude.ps1'
        $content = Get-Content $claudePath -Raw
        $content | Should -BeLike '*storyCompletionGitHead = (git rev-parse HEAD*'
    }

    It 'claude.ps1 checks for git commit after story done' {
        $claudePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\claude.ps1'
        $content = Get-Content $claudePath -Raw
        $content | Should -BeLike '*currentHead -ne $storyCompletionGitHead*'
        $content | Should -BeLike '*Git commit after story done*'
    }
}

# ============================================================================
# Step 2: Structured Test Output
# ============================================================================

Describe 'Format-TestSummary' -Tag 'Unit', 'Carlini' {
    It 'returns summary line and failures' {
        $tierResult = @{
            Summary = "2 failed, 10 passed"
            Failures = @(
                @{ Test = "tests/test_a.py::test_one"; Error = "AssertionError: 1 != 2" }
                @{ Test = "tests/test_b.py::test_two"; Error = "ValueError: bad input" }
            )
            CollectionErrors = @()
        }

        $result = Format-TestSummary -TierResult $tierResult
        $result | Should -BeLike "*2 failed, 10 passed*"
        $result | Should -BeLike "*FAIL: tests/test_a.py::test_one - AssertionError*"
        $result | Should -BeLike "*FAIL: tests/test_b.py::test_two - ValueError*"
    }

    It 'truncates errors longer than 200 chars' {
        $longError = "A" * 300
        $tierResult = @{
            Summary = "1 failed"
            Failures = @(@{ Test = "test_x.py::test_y"; Error = $longError })
            CollectionErrors = @()
        }

        $result = Format-TestSummary -TierResult $tierResult
        # Should contain truncated error with "..."
        $result | Should -BeLike "*...*"
        # Should NOT contain the full 300-char error
        $result.Length | Should -BeLessThan 350
    }

    It 'limits failures to 10' {
        $failures = 1..15 | ForEach-Object {
            @{ Test = "test_$_.py::test_func"; Error = "failed $_" }
        }
        $tierResult = @{
            Summary = "15 failed"
            Failures = $failures
            CollectionErrors = @()
        }

        $result = Format-TestSummary -TierResult $tierResult
        $result | Should -BeLike "*... and 5 more failures*"
    }

    It 'includes collection errors (max 5)' {
        $errors = 1..8 | ForEach-Object {
            @{ File = "test_$_.py"; Error = "ImportError: no module $_" }
        }
        $tierResult = @{
            Summary = "8 errors"
            Failures = @()
            CollectionErrors = $errors
        }

        $result = Format-TestSummary -TierResult $tierResult
        $result | Should -BeLike "*ERROR: test_1.py*"
        $result | Should -BeLike "*... and 3 more collection errors*"
    }

    It 'strips ANSI escape codes' {
        $tierResult = @{
            Summary = "1 failed"
            Failures = @(@{
                Test = "$([char]27)[31mtests/test_x.py::test_y$([char]27)[0m"
                Error = "$([char]27)[1mAssertionError$([char]27)[0m"
            })
            CollectionErrors = @()
        }

        $result = Format-TestSummary -TierResult $tierResult
        $result | Should -Not -Match '\x1b'
        $result | Should -BeLike "*FAIL: tests/test_x.py::test_y*"
    }

    It 'returns empty when no data' {
        $tierResult = @{
            Summary = $null
            Failures = @()
            CollectionErrors = @()
        }

        $result = Format-TestSummary -TierResult $tierResult
        $result | Should -BeNullOrEmpty
    }
}

Describe 'Build-TierDiagnostics uses structured output' -Tag 'Unit', 'Carlini' {
    It 'does NOT include raw output verbatim' {
        $tierResult = @{
            Tier = 3
            SyntaxErrors = $null; MergeConflicts = $null; MissingFiles = $null
            ConfigErrors = $null; CollectionErrors = @()
            Failures = @(@{ Test = "test_x.py::test_a"; Error = "AssertionError" })
            Summary = "1 failed"
            RawOutput = "VERY LONG RAW OUTPUT THAT SHOULD NOT APPEAR VERBATIM IN DIAGNOSTICS STRING"
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -Not -BeLike "*VERY LONG RAW OUTPUT*"
        $diag | Should -BeLike "*Test summary:*"
    }
}

Describe 'Retry context reduction' -Tag 'Unit', 'Carlini' {
    It 'quality.ps1 uses 20-line tail instead of 50' {
        $qualityPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\quality.ps1'
        $content = Get-Content $qualityPath -Raw
        $content | Should -BeLike "*Last 20 lines*"
        $content | Should -Not -BeLike "*Last 50 lines*"
    }

    It 'quality.ps1 strips ANSI escape codes from retry context' {
        $qualityPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\quality.ps1'
        $content = Get-Content $qualityPath -Raw
        $content | Should -Match 'replace.*\\x1b'
    }
}

# ============================================================================
# Step 3: Randomized Fast Test Mode
# ============================================================================

Describe 'Randomized Fast Test Mode' -Tag 'Unit', 'Carlini' {
    It 'config has fastPytestArgs' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.selfHealing.fastPytestArgs | Should -Not -BeNullOrEmpty
        $config.selfHealing.fastPytestArgs | Should -BeLike "*-x*"
        $config.selfHealing.fastPytestArgs | Should -BeLike "*randomly*"
    }

    It 'requirements-dev.txt includes pytest-randomly' {
        # PSScriptRoot = scripts/ralph/tests, project root is 3 levels up
        $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
        $reqPath = Join-Path $projectRoot 'requirements-dev.txt'
        $content = Get-Content $reqPath -Raw
        $content | Should -BeLike "*pytest-randomly*"
    }

    It 'Invoke-TieredHealthCheck runs fast Tier 3 off cadence' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Get-RalphConfig { return @{ selfHealing = @{ fastPytestArgs = "tests/ --tb=line -q -x -p randomly" } } }
        Mock Write-Host {}

        $script:State = @{ IterationCount = 4 }
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        # Off cadence (iteration 4, cadence 3) -> should still call FullHealthCheck (fast mode)
        Should -Invoke Invoke-FullHealthCheck -Times 1
        $result.HasErrors | Should -BeFalse
    }

    It 'Invoke-TieredHealthCheck runs full Tier 3 on cadence' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        $script:State = @{ IterationCount = 3 }
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }

    It 'ForceFullRun bypasses fast mode' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        $script:capturedArgs = $null
        Mock Invoke-FullHealthCheck {
            param($PytestArgs)
            $script:capturedArgs = $PytestArgs
            return @{ HasErrors = $false; Tier = 3; Skipped = $false }
        }
        Mock Write-Host {}

        $script:State = @{ IterationCount = 4 }  # Off cadence
        Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3 -ForceFullRun
        # ForceFullRun should NOT pass fast args (PytestArgs should be empty/default)
        $script:capturedArgs | Should -BeNullOrEmpty
    }
}

# ============================================================================
# Step 4: Living Sprint Progress Document
# ============================================================================

Describe 'Sprint Progress Functions' -Tag 'Unit', 'Carlini' {
    BeforeEach {
        $script:Paths = @{
            SprintProgressFile = Join-Path $TestDrive "sprint_progress.md"
        }
        if (Test-Path $script:Paths.SprintProgressFile) {
            Remove-Item $script:Paths.SprintProgressFile -Force
        }
    }

    It 'Update-SprintProgress creates file with header' {
        Update-SprintProgress -StoryId "US-69-001" -StoryTitle "Fix login" -Success $true

        $script:Paths.SprintProgressFile | Should -Exist
        $content = Get-Content $script:Paths.SprintProgressFile -Raw
        $content | Should -BeLike "*# Sprint Progress*"
        $content | Should -Match '\[DONE\] US-69-001: Fix login'
    }

    It 'Update-SprintProgress records failures' {
        Update-SprintProgress -StoryId "US-69-002" -StoryTitle "Add cache" -Success $false

        $content = Get-Content $script:Paths.SprintProgressFile -Raw
        $content | Should -Match '\[FAIL\] US-69-002: Add cache'
    }

    It 'Update-SprintProgress appends multiple entries' {
        Update-SprintProgress -StoryId "US-69-001" -StoryTitle "First" -Success $true
        Update-SprintProgress -StoryId "US-69-002" -StoryTitle "Second" -Success $false
        Update-SprintProgress -StoryId "US-69-003" -StoryTitle "Third" -Success $true

        $entries = @(Get-Content $script:Paths.SprintProgressFile | Where-Object { $_ -match '^\- \[' })
        $entries.Count | Should -Be 3
    }

    It 'Update-SprintProgress truncates to MaxEntries' {
        for ($i = 1; $i -le 25; $i++) {
            Update-SprintProgress -StoryId "US-69-$($i.ToString('000'))" -StoryTitle "Story $i" -Success $true -MaxEntries 20
        }

        $entries = @(Get-Content $script:Paths.SprintProgressFile | Where-Object { $_ -match '^\- \[' })
        $entries.Count | Should -Be 20
        # Should keep the LAST 20, not the first 20
        $entries[-1] | Should -BeLike "*US-69-025*"
        $entries[0] | Should -BeLike "*US-69-006*"
    }

    It 'Update-SprintProgress includes timestamp' {
        Update-SprintProgress -StoryId "US-69-001" -StoryTitle "Test" -Success $true

        $content = Get-Content $script:Paths.SprintProgressFile -Raw
        $content | Should -Match '\[\d{2}:\d{2}\]'
    }

    It 'Get-SprintProgressContext returns formatted context' {
        Update-SprintProgress -StoryId "US-69-001" -StoryTitle "First" -Success $true
        Update-SprintProgress -StoryId "US-69-002" -StoryTitle "Second" -Success $false

        $ctx = Get-SprintProgressContext
        $ctx | Should -BeLike "*Sprint Progress So Far*"
        $ctx | Should -BeLike "*Previous stories*"
        $ctx | Should -BeLike "*US-69-001*"
    }

    It 'Get-SprintProgressContext returns empty when no file' {
        $ctx = Get-SprintProgressContext
        $ctx | Should -BeNullOrEmpty
    }

    It 'Get-SprintProgressContext returns empty when file has no entries' {
        "# Sprint Progress" | Set-Content $script:Paths.SprintProgressFile
        $ctx = Get-SprintProgressContext
        $ctx | Should -BeNullOrEmpty
    }

    It 'Reset-SprintProgress deletes the file' {
        Update-SprintProgress -StoryId "US-69-001" -StoryTitle "Test" -Success $true
        $script:Paths.SprintProgressFile | Should -Exist

        Reset-SprintProgress
        $script:Paths.SprintProgressFile | Should -Not -Exist
    }

    It 'Reset-SprintProgress handles missing file gracefully' {
        { Reset-SprintProgress } | Should -Not -Throw
    }
}

Describe 'Sprint Progress Path Registration' -Tag 'Unit', 'Carlini' {
    It 'paths.ps1 defines SprintProgressFile' {
        $pathsContent = Get-Content (Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\paths.ps1') -Raw
        $pathsContent | Should -BeLike "*SprintProgressFile*"
        $pathsContent | Should -BeLike "*sprint_progress.md*"
    }
}

Describe 'Sprint Progress in Build-StoryPrompt' -Tag 'Unit', 'Carlini' {
    It 'prompts.ps1 calls Get-SprintProgressContext' {
        $promptsContent = Get-Content (Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1') -Raw
        $promptsContent | Should -BeLike "*Get-SprintProgressContext*"
        $promptsContent | Should -BeLike "*Section 5.5*"
    }
}

Describe 'Sprint Progress in loops.ps1' -Tag 'Unit', 'Carlini' {
    It 'all 6 loop sites call Update-SprintProgress' {
        $loopsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\loops.ps1'
        $content = Get-Content $loopsPath -Raw
        $matches = [regex]::Matches($content, 'Update-SprintProgress')
        $matches.Count | Should -Be 6
    }
}

Describe 'Sprint Progress Reset at Sprint Start' -Tag 'Unit', 'Carlini' {
    It 'sprint.ps1 calls Reset-SprintProgress in New-SeedPRD' {
        $sprintPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\sprint.ps1'
        $content = Get-Content $sprintPath -Raw
        $content | Should -BeLike "*Reset-SprintProgress*"
    }
}

# ============================================================================
# Phase 1: Oracle-Based Regression Guard + Deterministic Test Subsampling
# ============================================================================

Describe 'Oracle-Based Regression Guard' -Tag 'Unit', 'Carlini' {
    It 'Test-RegressionBaseline exists in healing.ps1' {
        $healingPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\healing.ps1'
        $content = Get-Content $healingPath -Raw
        $content | Should -BeLike "*function Test-RegressionBaseline*"
    }

    It 'Compare-TestBaseline accepts BaselineOverride param' {
        $qualityPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\quality.ps1'
        $content = Get-Content $qualityPath -Raw
        $content | Should -BeLike "*`$BaselineOverride*"
    }

    It 'Invoke-ClaudeForStory calls Test-RegressionBaseline' {
        $ralphPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'ralph.ps1'
        $content = Get-Content $ralphPath -Raw
        $content | Should -BeLike "*Test-RegressionBaseline*"
    }

    It 'Resolve-ClaudeResult passes BaselineOverride' {
        $claudePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\claude.ps1'
        $content = Get-Content $claudePath -Raw
        $content | Should -BeLike "*-BaselineOverride*"
    }

    It 'Get-StoryFailureContext references pre-story baseline' {
        $qualityPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\quality.ps1'
        $content = Get-Content $qualityPath -Raw
        $content | Should -BeLike "*Pre-story test baseline*"
        $content | Should -BeLike "*PreStoryBaselineFile*"
    }

    It 'paths.ps1 has PreStoryBaselineFile' {
        $pathsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\paths.ps1'
        $content = Get-Content $pathsPath -Raw
        $content | Should -BeLike "*PreStoryBaselineFile*"
        $content | Should -BeLike "*pre_story_baseline.json*"
    }
}

Describe 'Deterministic Test Subsampling' -Tag 'Unit', 'Carlini' {
    It 'Get-SubsampledTestCommand exists in healing.ps1' {
        $healingPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\healing.ps1'
        $content = Get-Content $healingPath -Raw
        $content | Should -BeLike "*function Get-SubsampledTestCommand*"
    }

    It 'Invoke-TieredHealthCheck accepts StoryId and FocusArea params' {
        $healingPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\healing.ps1'
        $content = Get-Content $healingPath -Raw
        $content | Should -Match 'Invoke-TieredHealthCheck[\s\S]*?\[string\]\$StoryId'
        $content | Should -Match 'Invoke-TieredHealthCheck[\s\S]*?\[string\]\$FocusArea'
    }

    It 'ralph-config.json has testSubsampling section' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.testSubsampling | Should -Not -BeNullOrEmpty
        $config.testSubsampling.focusAreaScoped | Should -Be $true
        $config.testSubsampling.alwaysFullOnCadence | Should -Be $true
    }

    It 'Invoke-PostIterationHealing forwards StoryId and FocusArea' {
        $healingPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\healing.ps1'
        $content = Get-Content $healingPath -Raw
        $content | Should -BeLike "*Invoke-TieredHealthCheck*-StoryId `$StoryId*-FocusArea `$FocusArea*"
    }
}

# ============================================================================
# Phase 2: Feedback-Driven Learning Loop + Role-Based Story Specialization
# ============================================================================

Describe 'Feedback-Driven Learning Loop' -Tag 'Unit', 'Carlini' {
    It 'Get-LearningInjection exists in learning.ps1' {
        $learningPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\learning.ps1'
        $content = Get-Content $learningPath -Raw
        $content | Should -BeLike "*function Get-LearningInjection*"
    }

    It 'Build-StoryPrompt injects learning warnings' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*Get-LearningInjection*"
        $content | Should -BeLike "*Section 5.7*"
    }

    It 'Resolve-ClaudeResult tracks injection effectiveness' {
        $claudePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\claude.ps1'
        $content = Get-Content $claudePath -Raw
        $content | Should -BeLike "*injection_result*"
    }
}

Describe 'Role-Based Story Specialization' -Tag 'Unit', 'Carlini' {
    It 'Get-StoryRole exists in sprint.ps1' {
        $sprintPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\sprint.ps1'
        $content = Get-Content $sprintPath -Raw
        $content | Should -BeLike "*function Get-StoryRole*"
    }

    It 'Build-StoryPrompt has role prefix section' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*Get-StoryRole*"
        $content | Should -BeLike "*Role: *"
    }

    It 'ralph-config.json has roles section' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.roles | Should -Not -BeNullOrEmpty
        $config.roles.bugfix | Should -Not -BeNullOrEmpty
        $config.roles.bugfix.timeout | Should -Be 420
        $config.roles.feature.maxIterations | Should -Be 5
    }

    It 'Record-Metric accepts Role param' {
        $metricsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\metrics.ps1'
        $content = Get-Content $metricsPath -Raw
        $content | Should -Match '\[string\]\$Role'
    }

    It 'Record-Metric CSV has role column' {
        $metricsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\metrics.ps1'
        $content = Get-Content $metricsPath -Raw
        $content | Should -BeLike "*,role`"*"
    }

    It 'Resolve-ClaudeResult passes Role to Record-Metric' {
        $claudePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\claude.ps1'
        $content = Get-Content $claudePath -Raw
        $content | Should -BeLike "*-Role*Get-StoryRole*"
    }
}

# ============================================================================
# Phase 3: Adaptive Context Pruning + Delta Debugging
# ============================================================================

# --- Improvement #5: Adaptive Context Pruning ---

Describe 'Adaptive Context Pruning - Source Inspection' -Tag 'Unit', 'Carlini' {
    It 'Build-BudgetedPrompt function exists in prompts.ps1' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*function Build-BudgetedPrompt*"
    }

    It 'Compress-FailureContext function exists in prompts.ps1' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*function Compress-FailureContext*"
    }

    It 'Build-StoryPrompt checks adaptivePruning flag' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*adaptivePruning*"
    }

    It 'Build-StoryPrompt has dual-path branching (budgeted vs legacy)' {
        $promptsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\prompts.ps1'
        $content = Get-Content $promptsPath -Raw
        $content | Should -BeLike "*useBudgeting*"
        $content | Should -BeLike "*BUDGETED PATH*"
        $content | Should -BeLike "*LEGACY PATH*"
    }
}

Describe 'Adaptive Context Pruning - Config' -Tag 'Unit', 'Carlini' {
    It 'ralph-config.json has budgetAllocation in prompts section' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.prompts.budgetAllocation | Should -Not -BeNullOrEmpty
        $config.prompts.budgetAllocation.storyDetails | Should -Be 0.40
        $config.prompts.budgetAllocation.failureContext | Should -Be 0.20
        $config.prompts.budgetAllocation.contextHints | Should -Be 0.20
        $config.prompts.budgetAllocation.supplementary | Should -Be 0.20
    }

    It 'adaptivePruning flag exists and is false by default' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.flags.adaptivePruning | Should -Be $false
    }
}

Describe 'Compress-FailureContext' -Tag 'Unit', 'Carlini' {
    It 'passes through short context unchanged' {
        # Use 6 distinct lines that fit under budget — passthrough should return verbatim
        # Without passthrough, Compress-FailureContext extracts headers + error category +
        # unmet criteria + last output lines, producing a DIFFERENT result
        $short = "Line 1 header`nLine 2 detail`nLine 3 more`nLine 4 extra`nLine 5 data`nLine 6 end"
        $result = Compress-FailureContext -FailureContext $short -MaxChars 2000
        $result | Should -Be $short
    }

    It 'compresses long context' {
        # Build a verbose failure context > 2000 chars
        $lines = @("## Retry Context for US-69-001")
        $lines += "Attempt 3 of 3"
        $lines += "Error Category: TestFailure"
        $lines += ""
        $lines += "  [ ] Criterion that was NOT met"
        $lines += "  [x] Criterion that was met"
        # Add lots of output lines to push over budget
        for ($i = 0; $i -lt 100; $i++) {
            $lines += "Output line $i - " + ("x" * 30)
        }
        $longContext = $lines -join "`n"

        $result = Compress-FailureContext -FailureContext $longContext -MaxChars 2000
        $result.Length | Should -BeLessOrEqual 2000
        $result | Should -BeLike "*Retry Context*"
        $result | Should -BeLike "*Error Category*"
        $result | Should -BeLike "*Unmet criteria*"
    }

    It 'handles null input' {
        $result = Compress-FailureContext -FailureContext $null
        $result | Should -BeNullOrEmpty
    }

    It 'handles empty input' {
        $result = Compress-FailureContext -FailureContext ""
        $result | Should -BeNullOrEmpty
    }
}

Describe 'Build-BudgetedPrompt' -Tag 'Unit', 'Carlini' {
    It 'passes through sections under budget' {
        $sections = @(
            @{ name = "details"; content = "Story details here"; priority = 1; displayOrder = 1 }
            @{ name = "hints"; content = "File hints here"; priority = 3; displayOrder = 0 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 10000
        $result | Should -BeLike "*File hints here*"
        $result | Should -BeLike "*Story details here*"
    }

    It 'preserves display order' {
        $sections = @(
            @{ name = "second"; content = "SECOND"; priority = 1; displayOrder = 1 }
            @{ name = "first"; content = "FIRST"; priority = 3; displayOrder = 0 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 10000
        $firstIdx = $result.IndexOf("FIRST")
        $secondIdx = $result.IndexOf("SECOND")
        $firstIdx | Should -BeLessThan $secondIdx
    }

    It 'drops P4 sections first when over budget' {
        $sections = @(
            @{ name = "critical"; content = ("A" * 500); priority = 1; displayOrder = 0 }
            @{ name = "optional"; content = ("B" * 500); priority = 4; displayOrder = 1 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 600
        $result | Should -BeLike "*AAAA*"
        $result | Should -Not -BeLike "*BBBB*"
    }

    It 'never trims P1 sections' {
        $p1Content = "A" * 800
        $sections = @(
            @{ name = "critical"; content = $p1Content; priority = 1; displayOrder = 0 }
            @{ name = "p3"; content = ("B" * 500); priority = 3; displayOrder = 1 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 900
        # P1 must be fully present
        $result | Should -BeLike "*$p1Content*"
    }

    It 'filters empty sections' {
        $sections = @(
            @{ name = "empty"; content = ""; priority = 1; displayOrder = 0 }
            @{ name = "whitespace"; content = "   "; priority = 1; displayOrder = 1 }
            @{ name = "valid"; content = "Valid content"; priority = 1; displayOrder = 2 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 10000
        $result | Should -Be "Valid content"
    }

    It 'returns empty for no valid sections' {
        $sections = @(
            @{ name = "empty"; content = ""; priority = 1; displayOrder = 0 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 10000
        $result | Should -BeNullOrEmpty
    }

    It 'compresses P2 sections before trimming P3' {
        $longP2 = "## Retry Context`nAttempt 3`nError Category: TestFailure`n" + ("output line`n" * 200)
        $sections = @(
            @{ name = "p1"; content = ("A" * 200); priority = 1; displayOrder = 0 }
            @{ name = "p2"; content = $longP2; priority = 2; displayOrder = 1 }
            @{ name = "p3"; content = ("C" * 200); priority = 3; displayOrder = 2 }
        )
        $result = Build-BudgetedPrompt -Sections $sections -MaxLength 1000
        # P2 should be compressed, not dropped entirely
        $result | Should -BeLike "*Retry Context*"
        $result.Length | Should -BeLessOrEqual 1000
    }
}

# --- Improvement #6: Delta Debugging for Stuck Stories ---

Describe 'Delta Debugging - Source Inspection' -Tag 'Unit', 'Carlini' {
    It 'Split-StuckStory exists in sprint.ps1' {
        $sprintPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\sprint.ps1'
        $content = Get-Content $sprintPath -Raw
        $content | Should -BeLike "*function Split-StuckStory*"
    }

    It 'Complete-DeltaSplitParents exists in sprint.ps1' {
        $sprintPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\sprint.ps1'
        $content = Get-Content $sprintPath -Raw
        $content | Should -BeLike "*function Complete-DeltaSplitParents*"
    }

    It 'ralph.ps1 references Split-StuckStory' {
        $ralphPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'ralph.ps1'
        $content = Get-Content $ralphPath -Raw
        $content | Should -BeLike "*Split-StuckStory*"
    }

    It 'ralph.ps1 references Complete-DeltaSplitParents' {
        $ralphPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'ralph.ps1'
        $content = Get-Content $ralphPath -Raw
        $content | Should -BeLike "*Complete-DeltaSplitParents*"
    }

    It 'scoring.ps1 references deltaSplitParentIds' {
        $scoringPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'lib\scoring.ps1'
        $content = Get-Content $scoringPath -Raw
        $content | Should -BeLike "*deltaSplitParentIds*"
    }
}

Describe 'Delta Debugging - Config' -Tag 'Unit', 'Carlini' {
    It 'ralph-config.json has healing.deltaDebugging section' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.healing.deltaDebugging | Should -Not -BeNullOrEmpty
        $config.healing.deltaDebugging.triggerAfterFailures | Should -Be 2
        $config.healing.deltaDebugging.minCriteriaToSplit | Should -Be 2
    }

    It 'deltaDebugging flag exists and is false by default' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        $config.flags.deltaDebugging | Should -Be $false
    }
}

Describe 'Split-StuckStory' -Tag 'Unit', 'Carlini' {
    It 'splits multi-criteria story into per-criterion sub-stories' {
        $story = @{
            title = "Fix authentication flow"
            acceptanceCriteria = @("Login works", "Logout works", "Session persists")
            priority = "high"
            notes = ""
        }
        $result = Split-StuckStory -StoryId "US-69-003" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 3
        $result[0].acceptanceCriteria.Count | Should -Be 1
        $result[0].acceptanceCriteria[0] | Should -Be "Login works"
        $result[1].acceptanceCriteria[0] | Should -Be "Logout works"
        $result[2].acceptanceCriteria[0] | Should -Be "Session persists"
    }

    It 'returns empty for stories with < 2 criteria' {
        $story = @{
            title = "Simple fix"
            acceptanceCriteria = @("Just one criterion")
            priority = "medium"
        }
        $result = Split-StuckStory -StoryId "US-69-001" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 0
    }

    It 'returns empty for stories with no criteria' {
        $story = @{
            title = "No criteria"
            priority = "low"
        }
        $result = Split-StuckStory -StoryId "US-69-001" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 0
    }

    It 'returns empty for already-decomposed stories (nestingLevel > 0)' {
        $story = @{
            title = "Sub-task"
            acceptanceCriteria = @("A", "B", "C")
            nestingLevel = 1
            priority = "high"
        }
        $result = Split-StuckStory -StoryId "US-69-D003-01" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 0
    }

    It 'returns empty for already delta-split stories' {
        $story = @{
            title = "Already split"
            acceptanceCriteria = @("A", "B")
            notes = "Delta-split into 2 sub-stories"
            priority = "high"
        }
        $result = Split-StuckStory -StoryId "US-69-003" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 0
    }

    It 'returns empty for seed stories' {
        $story = @{
            title = "Generate sprint stories for testing"
            acceptanceCriteria = @("A", "B", "C")
            priority = "high"
        }
        $result = Split-StuckStory -StoryId "US-69-001" -Story ([PSCustomObject]$story)
        $result.Count | Should -Be 0
    }

    It 'preserves parent priority in sub-stories' {
        $story = @{
            title = "Fix stuff"
            acceptanceCriteria = @("A", "B")
            priority = "low"
        }
        $result = Split-StuckStory -StoryId "US-69-003" -Story ([PSCustomObject]$story)
        $result[0].priority | Should -Be "low"
        $result[1].priority | Should -Be "low"
    }

    It 'sets deltaDebugged marker on sub-stories' {
        $story = @{
            title = "Fix stuff"
            acceptanceCriteria = @("A", "B")
            priority = "high"
        }
        $result = Split-StuckStory -StoryId "US-69-003" -Story ([PSCustomObject]$story)
        $result[0].deltaDebugged | Should -Be $true
        $result[1].deltaDebugged | Should -Be $true
    }

    It 'sets decomposedFrom to parent ID' {
        $story = @{
            title = "Fix stuff"
            acceptanceCriteria = @("A", "B")
            priority = "high"
        }
        $result = Split-StuckStory -StoryId "US-69-003" -Story ([PSCustomObject]$story)
        $result[0].decomposedFrom | Should -Be "US-69-003"
        $result[1].decomposedFrom | Should -Be "US-69-003"
    }
}

Describe 'Complete-DeltaSplitParents' -Tag 'Unit', 'Carlini' {
    It 'marks parent as passed when all children pass' {
        $stories = @(
            [PSCustomObject]@{ id = "US-69-003"; passes = $false; notes = "Delta-split into 2 sub-stories" }
            [PSCustomObject]@{ id = "US-69-D003-01"; passes = $true; decomposedFrom = "US-69-003" }
            [PSCustomObject]@{ id = "US-69-D003-02"; passes = $true; decomposedFrom = "US-69-003" }
        )
        $result = Complete-DeltaSplitParents -Stories $stories
        $result | Should -Be $true
        $stories[0].passes | Should -Be $true
    }

    It 'does not mark parent when some children fail' {
        $stories = @(
            [PSCustomObject]@{ id = "US-69-003"; passes = $false; notes = "Delta-split into 2 sub-stories" }
            [PSCustomObject]@{ id = "US-69-D003-01"; passes = $true; decomposedFrom = "US-69-003" }
            [PSCustomObject]@{ id = "US-69-D003-02"; passes = $false; decomposedFrom = "US-69-003" }
        )
        $result = Complete-DeltaSplitParents -Stories $stories
        $result | Should -Be $false
        $stories[0].passes | Should -Be $false
    }

    It 'ignores non-delta stories' {
        $stories = @(
            [PSCustomObject]@{ id = "US-69-003"; passes = $false; notes = "Regular failed story" }
            [PSCustomObject]@{ id = "US-69-004"; passes = $true; notes = "" }
        )
        $result = Complete-DeltaSplitParents -Stories $stories
        $result | Should -Be $false
    }

    It 'returns false for empty input' {
        $result = Complete-DeltaSplitParents -Stories @()
        $result | Should -Be $false
    }

    It 'skips already-passed parents' {
        $stories = @(
            [PSCustomObject]@{ id = "US-69-003"; passes = $true; notes = "Delta-split into 2 sub-stories" }
            [PSCustomObject]@{ id = "US-69-D003-01"; passes = $true; decomposedFrom = "US-69-003" }
        )
        $result = Complete-DeltaSplitParents -Stories $stories
        $result | Should -Be $false
    }
}
