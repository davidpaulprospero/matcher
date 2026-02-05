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
