#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Self-Healing tiered health check system
.DESCRIPTION
    Tests cover:
    - Invoke-FastHealthCheck (Tier 1: syntax, conflicts, critical files)
    - Invoke-CollectionHealthCheck (Tier 2: pytest --collect-only)
    - Invoke-FullHealthCheck (Tier 3: full pytest)
    - Invoke-TieredHealthCheck (orchestrator)
    - Build-TierDiagnostics
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
}

Describe 'Invoke-FastHealthCheck (Tier 1)' -Tag 'Unit', 'Healing' {
    BeforeEach {
        $script:ProjectRoot = $TestDrive
    }

    It 'returns clean when no changed files' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*py_compile*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @()
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 1
    }

    It 'detects syntax errors via py_compile' {
        Mock Invoke-Expression {
            $global:LASTEXITCODE = 1
            return "SyntaxError: invalid syntax (src/config.py, line 42)"
        } -ParameterFilter { $Command -like '*py_compile*' }
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        # Create the file so Test-Path passes
        New-Item -Path (Join-Path $TestDrive "src") -ItemType Directory -Force | Out-Null
        New-Item -Path (Join-Path $TestDrive "src/config.py") -ItemType File -Force | Out-Null

        $result = Invoke-FastHealthCheck -ChangedFiles @("src/config.py") -CriticalFiles @()
        $result.HasErrors | Should -BeTrue
        $result.SyntaxErrors.Count | Should -BeGreaterThan 0
    }

    It 'detects missing critical files' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @("main.py")
        $result.HasErrors | Should -BeTrue
        $result.MissingFiles.Count | Should -Be 1
    }

    It 'passes when critical files exist' {
        New-Item -Path (Join-Path $TestDrive "main.py") -ItemType File -Force | Out-Null
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @("main.py")
        $result.MissingFiles.Count | Should -Be 0
    }

    It 'tracks checks run' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @()
        $result.ChecksRun | Should -Contain "merge_conflicts"
        $result.ChecksRun | Should -Contain "critical_files"
        $result.ChecksRun | Should -Contain "config_validation"
    }
}

Describe 'Invoke-CollectionHealthCheck (Tier 2)' -Tag 'Unit', 'Healing' {
    It 'returns clean when collect-only succeeds' {
        Mock Invoke-Expression {
            return "50 tests collected in 2.1s"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 2
    }

    It 'detects import errors from collect-only' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/test_budget.py - ImportError: cannot import name 'BudgetManager' from 'src.downloader'",
                "ERROR tests/test_cache.py - SyntaxError: invalid syntax",
                "2 errors in 1.3s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeTrue
        $result.ErrorCount | Should -Be 2
        $result.CollectionErrors.Count | Should -Be 2
    }

    It 'handles pytest unavailable gracefully' {
        Mock Invoke-Expression {
            throw "pytest: command not found"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.Skipped | Should -BeTrue
    }

    It 'captures raw output' {
        Mock Invoke-Expression {
            return "50 tests collected in 2.1s"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.RawOutput | Should -BeLike "*50 tests collected*"
    }
}

Describe 'Invoke-FullHealthCheck (Tier 3)' -Tag 'Unit', 'Healing' {
    It 'returns clean when all tests pass' {
        Mock Invoke-Expression {
            return "50 passed in 12.3s"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.HasErrors | Should -BeFalse
        $result.Tier | Should -Be 3
    }

    It 'detects test failures' {
        Mock Invoke-Expression {
            return @(
                "FAILED tests/test_cache.py::test_hit_rate - AssertionError",
                "FAILED tests/test_config.py::test_load - ImportError",
                "2 failed, 10 passed in 3.45s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.HasErrors | Should -BeTrue
        $result.FailureCount | Should -Be 2
        $result.Failures.Count | Should -Be 2
    }

    It 'detects collection errors from full run' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/test_budget.py - SyntaxError: invalid syntax",
                "1 error in 0.54s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.HasErrors | Should -BeTrue
        $result.ErrorCount | Should -Be 1
    }

    It 'returns clean when no tests found' {
        Mock Invoke-Expression {
            return "no tests ran in 0.01s"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.HasErrors | Should -BeFalse
    }

    It 'captures summary line' {
        Mock Invoke-Expression {
            return @(
                "FAILED tests/test_cache.py::test_hit_rate - AssertionError",
                "1 failed, 49 passed in 12.3s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.Summary | Should -BeLike "*failed*passed*"
    }
}

Describe 'Invoke-TieredHealthCheck (orchestrator)' -Tag 'Unit', 'Healing' {
    BeforeEach {
        $script:ProjectRoot = $TestDrive
        $script:State = @{ IterationCount = 5; SessionId = "test" }
    }

    It 'stops at Tier 1 when syntax error found' {
        Mock Invoke-FastHealthCheck {
            return @{
                HasErrors = $true; Tier = 1
                SyntaxErrors = @(@{File="x.py";Error="bad"})
                MergeConflicts = @(); MissingFiles = @(); ConfigErrors = @()
                RawOutput = $null
            }
        }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Write-Host {}

        $result = Invoke-TieredHealthCheck -ChangedFiles @("x.py")
        $result.HasErrors | Should -BeTrue
        $result.FailedTier | Should -Be 1
        Should -Not -Invoke Invoke-CollectionHealthCheck
    }

    It 'escalates to Tier 2 when Tier 1 clean' {
        Mock Invoke-FastHealthCheck {
            return @{ HasErrors = $false; Tier = 1 }
        }
        Mock Invoke-CollectionHealthCheck {
            return @{ HasErrors = $false; Tier = 2; Skipped = $false }
        }
        Mock Write-Host {}

        $result = Invoke-TieredHealthCheck -ChangedFiles @()
        $result.HasErrors | Should -BeFalse
        Should -Invoke Invoke-CollectionHealthCheck -Times 1
    }

    It 'runs Tier 3 on cadence (every Nth iteration)' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        # Iteration 3 = cadence hit (every 3)
        $script:State.IterationCount = 3
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }

    It 'skips Tier 3 off cadence' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3 } }
        Mock Write-Host {}

        # Iteration 4 = not on cadence
        $script:State.IterationCount = 4
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        Should -Invoke Invoke-FullHealthCheck -Times 0
    }

    It 'always runs Tier 3 on first iteration' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        $script:State.IterationCount = 1
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 5
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }

    It 'stops at Tier 2 when collection errors found' {
        Mock Invoke-FastHealthCheck {
            return @{ HasErrors = $false; Tier = 1 }
        }
        Mock Invoke-CollectionHealthCheck {
            return @{
                HasErrors = $true; Tier = 2; Skipped = $false
                ErrorCount = 1
                CollectionErrors = @(@{File="tests/test_x.py";Error="ImportError"})
                RawOutput = "ERROR tests/test_x.py - ImportError"
            }
        }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        $result = Invoke-TieredHealthCheck -ChangedFiles @()
        $result.HasErrors | Should -BeTrue
        $result.FailedTier | Should -Be 2
        Should -Not -Invoke Invoke-FullHealthCheck
    }

    It 'forces Tier 3 with ForceFullRun switch' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        # Iteration 4 = off cadence, but forced
        $script:State.IterationCount = 4
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3 -ForceFullRun
        Should -Invoke Invoke-FullHealthCheck -Times 1
    }
}

Describe 'Build-TierDiagnostics' -Tag 'Unit', 'Healing' {
    It 'formats syntax errors' {
        $tierResult = @{
            Tier = 1
            SyntaxErrors = @(@{ File = "src/config.py"; Error = "SyntaxError: invalid syntax" })
            MergeConflicts = @()
            MissingFiles = @()
            ConfigErrors = @()
            CollectionErrors = $null
            Failures = $null
            RawOutput = $null
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*SYNTAX: src/config.py*"
    }

    It 'formats merge conflicts' {
        $tierResult = @{
            Tier = 1
            SyntaxErrors = @()
            MergeConflicts = @("src/pipeline.py")
            MissingFiles = @()
            ConfigErrors = @()
            CollectionErrors = $null
            Failures = $null
            RawOutput = $null
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*CONFLICT: src/pipeline.py*"
    }

    It 'formats missing files' {
        $tierResult = @{
            Tier = 1
            SyntaxErrors = @()
            MergeConflicts = @()
            MissingFiles = @("main.py", "config.yaml")
            ConfigErrors = @()
            CollectionErrors = $null
            Failures = $null
            RawOutput = $null
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*MISSING: main.py*"
        $diag | Should -BeLike "*MISSING: config.yaml*"
    }

    It 'formats collection errors' {
        $tierResult = @{
            Tier = 2
            SyntaxErrors = $null
            MergeConflicts = $null
            MissingFiles = $null
            ConfigErrors = $null
            CollectionErrors = @(@{ File = "tests/test_cache.py"; Error = "ImportError" })
            Failures = $null
            RawOutput = "ERROR tests/test_cache.py - ImportError"
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*IMPORT: tests/test_cache.py*"
    }

    It 'formats test failures' {
        $tierResult = @{
            Tier = 3
            SyntaxErrors = $null
            MergeConflicts = $null
            MissingFiles = $null
            ConfigErrors = $null
            CollectionErrors = @()
            Failures = @(@{ Test = "tests/test_x.py::test_fail"; Error = "AssertionError" })
            RawOutput = "FAILED tests/test_x.py::test_fail - AssertionError"
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*FAIL: tests/test_x.py::test_fail*"
    }

    It 'includes raw output when present' {
        $tierResult = @{
            Tier = 3
            SyntaxErrors = $null
            MergeConflicts = $null
            MissingFiles = $null
            ConfigErrors = $null
            CollectionErrors = $null
            Failures = $null
            RawOutput = "some raw output here"
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*Raw output:*some raw output here*"
    }
}

Describe 'Log-HealingEvent' {
    BeforeEach {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
    }

    It 'creates healing log file on first write' {
        Log-HealingEvent -Event "healing_started" -Data @{
            trigger = "test_failure"
            failedTier = 2
            failureCount = 2
            errors = @("test_cache.py::test_hit_rate", "test_config.py::test_load")
        }

        $script:HealingLogFile | Should -Exist
        $lines = @(Get-Content $script:HealingLogFile)
        $lines.Count | Should -Be 1
        $entry = $lines[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_started"
        $entry.data.failedTier | Should -Be 2
        $entry.timestamp | Should -Not -BeNullOrEmpty
    }

    It 'appends multiple events to log' {
        Log-HealingEvent -Event "healing_started" -Data @{ trigger = "test_failure" }
        Log-HealingEvent -Event "healing_attempt" -Data @{ attempt = 1; prompt = "Fix import error" }
        Log-HealingEvent -Event "healing_resolved" -Data @{ attempt = 1; fix = "Added missing import" }

        $lines = Get-Content $script:HealingLogFile
        $lines.Count | Should -Be 3
    }

    It 'includes thought process in healing_resolved events' {
        Log-HealingEvent -Event "healing_resolved" -Data @{
            attempt = 1
            errorsFixed = @("ImportError in test_cache.py")
            fix = "Added 'from src.cache import BaseCache' to test file"
            thoughtProcess = "The test file was importing BaseCache but the module was refactored. Updated import path."
            filesChanged = @("tests/test_cache.py")
            failedTier = 2
        }

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.data.thoughtProcess | Should -Not -BeNullOrEmpty
        $entry.data.filesChanged.Count | Should -Be 1
        $entry.data.failedTier | Should -Be 2
    }

    It 'records session context in every event' {
        $script:State = @{ SessionId = "test-session-123"; IterationCount = 5 }
        Log-HealingEvent -Event "healing_started" -Data @{ trigger = "test" }

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.sessionId | Should -Be "test-session-123"
        $entry.iteration | Should -Be 5
    }
}

Describe 'Suspend-SprintForHealing' {
    BeforeEach {
        $script:HealingStateFile = Join-Path $TestDrive "healing_state.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_suspend.jsonl"
        if (Test-Path $script:HealingStateFile) { Remove-Item $script:HealingStateFile -Force }
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
        $script:State = @{
            SessionId = "test-session"; IterationCount = 7;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }
    }

    It 'creates healing_state.json with tier and sprint context' {
        $healthResult = @{
            HasErrors = $true; FailedTier = 2
            RawDiagnostics = "IMPORT: test_cache.py -> ImportError"
            TierResults = @(
                @{ Tier = 1; HasErrors = $false },
                @{ Tier = 2; HasErrors = $true; ErrorCount = 1;
                   CollectionErrors = @(@{ File = "test_cache.py"; Error = "ImportError" }) }
            )
        }

        Mock Write-Host {}
        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "US-005" -FocusArea "testing"

        $script:HealingStateFile | Should -Exist
        $state = Get-Content $script:HealingStateFile -Raw | ConvertFrom-Json
        $state.paused | Should -BeTrue
        $state.storyId | Should -Be "US-005"
        $state.focusArea | Should -Be "testing"
        $state.failedTier | Should -Be 2
    }

    It 'logs healing_started event with tier info' {
        $healthResult = @{
            HasErrors = $true; FailedTier = 1
            RawDiagnostics = "SYNTAX: config.py -> SyntaxError"
            TierResults = @(@{ Tier = 1; HasErrors = $true; SyntaxErrors = @(@{File="config.py";Error="bad"}) })
        }

        Mock Write-Host {}
        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "US-003" -FocusArea "pipeline"

        $script:HealingLogFile | Should -Exist
        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_started"
        $entry.data.failedTier | Should -Be 1
    }
}

Describe 'Resume-SprintFromHealing' {
    BeforeEach {
        $script:HealingStateFile = Join-Path $TestDrive "healing_state_resume.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_resume.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
        $script:State = @{ SessionId = "s1"; IterationCount = 8; ConsecutiveFailures = 0 }
    }

    It 'clears healing_state.json on resume' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Mock Write-Host {}
        Resume-SprintFromHealing -Success $true -AttemptCount 1
        $script:HealingStateFile | Should -Not -Exist
    }

    It 'logs healing_resolved on successful fix' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Mock Write-Host {}
        Resume-SprintFromHealing -Success $true -AttemptCount 2 -FixSummary "Fixed import path"

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_resolved"
        $entry.data.attempts | Should -Be 2
    }

    It 'logs healing_failed when fix unsuccessful' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Mock Write-Host {}
        Resume-SprintFromHealing -Success $false -AttemptCount 3

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_failed"
    }
}

Describe 'Test-HealingInProgress' {
    BeforeEach {
        $script:HealingStateFile = Join-Path $TestDrive "healing_state_check.json"
        if (Test-Path $script:HealingStateFile) { Remove-Item $script:HealingStateFile -Force }
    }

    It 'returns false when no healing state file' {
        Test-HealingInProgress | Should -BeFalse
    }

    It 'returns true when healing is paused' {
        @{ paused = $true; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Test-HealingInProgress | Should -BeTrue
    }

    It 'returns false when file exists but paused is false' {
        @{ paused = $false; storyId = "US-005" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        Test-HealingInProgress | Should -BeFalse
    }
}

Describe 'Build-HealingPrompt' {
    It 'includes tier-specific diagnostics' {
        $state = @{
            failedTier = 2
            rawDiagnostics = "IMPORT: test_cache.py -> ImportError"
            focusArea = "testing"
            storyId = "US-005"
        }

        $prompt = Build-HealingPrompt -HealingState $state -Attempt 1
        $prompt | Should -BeLike "*Tier 2*"
        $prompt | Should -BeLike "*Import/Collection*"
        $prompt | Should -BeLike "*IMPORT: test_cache.py*"
        $prompt | Should -BeLike "*US-005*"
    }

    It 'includes previous attempt context on retry' {
        $state = @{
            failedTier = 1
            rawDiagnostics = "SYNTAX error"
            focusArea = "pipeline"
            storyId = "US-003"
        }

        $prompt = Build-HealingPrompt -HealingState $state -Attempt 2 -PreviousOutput "Tried fixing import but failed"
        $prompt | Should -BeLike "*Previous Attempt Output*"
        $prompt | Should -BeLike "*DIFFERENT approach*"
    }

    It 'omits previous output on first attempt' {
        $state = @{ failedTier = 1; rawDiagnostics = "err"; focusArea = "x"; storyId = "y" }

        $prompt = Build-HealingPrompt -HealingState $state -Attempt 1
        $prompt | Should -Not -BeLike "*Previous Attempt*"
    }
}

Describe 'Invoke-HealingSession' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_session.jsonl"
        $script:HealingStateFile = Join-Path $TestDrive "healing_state_session.json"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
        $script:State = @{
            SessionId = "heal-test"; IterationCount = 10;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }

        @{
            paused = $true; storyId = "US-005"; focusArea = "testing"
            failedTier = 2; rawDiagnostics = "IMPORT: test_cache.py -> ImportError"
        } | ConvertTo-Json -Depth 10 | Set-Content $script:HealingStateFile
    }

    It 'attempts healing and succeeds on first try' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed the import in test_cache.py by updating path"
                ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }

        # Validation re-check passes
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0; TierResults = @(); RawDiagnostics = "" }
        }
        Mock Write-Host {}

        $result = Invoke-HealingSession
        $result.Success | Should -BeTrue
        $result.AttemptsUsed | Should -Be 1
    }

    It 'fails after max attempts exceeded' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Attempted fix"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }

        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $true; FailedTier = 2; RawDiagnostics = "still broken"; TierResults = @() }
        }
        Mock Write-Host {}

        $result = Invoke-HealingSession -MaxAttempts 2
        $result.Success | Should -BeFalse
        $result.AttemptsUsed | Should -Be 2
    }

    It 'logs each attempt to healing_log.jsonl' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed it"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0 }
        }
        Mock Write-Host {}

        Invoke-HealingSession -MaxAttempts 3

        $lines = Get-Content $script:HealingLogFile
        $lines.Count | Should -BeGreaterOrEqual 1
        $hasAttempt = $lines | Where-Object { ($_ | ConvertFrom-Json).event -eq "healing_attempt" }
        $hasAttempt | Should -Not -BeNullOrEmpty
    }

    It 'returns failure when no healing state exists' {
        Remove-Item $script:HealingStateFile -Force
        $result = Invoke-HealingSession
        $result.Success | Should -BeFalse
        $result.AttemptsUsed | Should -Be 0
    }

    It 'handles Claude subprocess timeout gracefully' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $false; ExitCode = -1; TimedOut = $true
                Output = ""; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $true; FailedTier = 2 } }
        Mock Write-Host {}

        $result = Invoke-HealingSession -MaxAttempts 1
        $result.Success | Should -BeFalse
        $result.AttemptsUsed | Should -Be 1
    }
}
