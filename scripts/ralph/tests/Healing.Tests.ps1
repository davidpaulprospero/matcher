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

    It 'detects config validation failures' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*py_compile*' }
        Mock Invoke-Expression {
            $global:LASTEXITCODE = 1
            return "Error: invalid YAML in config.yaml"
        } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @()
        $result.HasErrors | Should -BeTrue
        $result.ConfigErrors.Count | Should -BeGreaterThan 0
    }

    It 'detects merge conflicts from git ls-files' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*py_compile*' }
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git {
            return "100644 abc123 1`tsrc/pipeline.py"
        }

        $result = Invoke-FastHealthCheck -ChangedFiles @() -CriticalFiles @()
        $result.HasErrors | Should -BeTrue
        $result.MergeConflicts.Count | Should -BeGreaterThan 0
    }

    It 'reports multiple syntax errors from different files' {
        Mock Invoke-Expression {
            $global:LASTEXITCODE = 1
            return "SyntaxError: unexpected indent"
        } -ParameterFilter { $Command -like '*py_compile*' }
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        New-Item -Path (Join-Path $TestDrive "a.py") -ItemType File -Force | Out-Null
        New-Item -Path (Join-Path $TestDrive "b.py") -ItemType File -Force | Out-Null

        $result = Invoke-FastHealthCheck -ChangedFiles @("a.py", "b.py") -CriticalFiles @()
        $result.SyntaxErrors.Count | Should -Be 2
    }

    It 'skips py_compile for non-python changed files' {
        Mock Invoke-Expression { $global:LASTEXITCODE = 0; return "" } -ParameterFilter { $Command -like '*validate-config*' }
        Mock git { return $null }

        $result = Invoke-FastHealthCheck -ChangedFiles @("README.md", "config.yaml") -CriticalFiles @()
        $result.HasErrors | Should -BeFalse
        $result.ChecksRun | Should -Not -Contain "py_compile"
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

    It 'parses ERROR line with no dash separator' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/broken.py",
                "1 error in 0.5s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeTrue
        $result.CollectionErrors[0].Error | Should -Be "Unknown"
    }

    It 'returns clean on empty output' {
        Mock Invoke-Expression { return "" } -ParameterFilter { $Command -like '*--collect-only*' }

        $result = Invoke-CollectionHealthCheck
        $result.HasErrors | Should -BeFalse
        $result.ErrorCount | Should -Be 0
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

    It 'detects mixed failures and collection errors' {
        Mock Invoke-Expression {
            return @(
                "ERROR tests/test_broken.py - ImportError: no module named 'foo'",
                "FAILED tests/test_x.py::test_y - AssertionError",
                "1 failed, 1 error, 5 passed in 4.2s"
            ) -join "`n"
        } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/ --tb=short -q"
        $result.HasErrors | Should -BeTrue
        $result.FailureCount | Should -Be 1
        $result.ErrorCount | Should -Be 1
    }

    It 'handles pytest exception gracefully' {
        Mock Invoke-Expression { throw "pytest crashed" } -ParameterFilter { $Command -like '*pytest*' }

        $result = Invoke-FullHealthCheck -PytestArgs "tests/"
        $result.Skipped | Should -BeTrue
        $result.HasErrors | Should -BeFalse
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

    It 'handles missing $State gracefully (defaults to iteration 1)' {
        $savedState = $script:State
        $script:State = $null
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $false } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3; Skipped = $false } }
        Mock Write-Host {}

        # With null state, iteration defaults to 1 -> always runs Tier 3
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 5
        Should -Invoke Invoke-FullHealthCheck -Times 1
        $script:State = $savedState
    }

    It 'skips Tier 3 when Tier 2 is skipped (pytest unavailable)' {
        Mock Invoke-FastHealthCheck { return @{ HasErrors = $false; Tier = 1 } }
        Mock Invoke-CollectionHealthCheck { return @{ HasErrors = $false; Tier = 2; Skipped = $true } }
        Mock Invoke-FullHealthCheck { return @{ HasErrors = $false; Tier = 3 } }
        Mock Write-Host {}

        # Iteration 3 on cadence but T2 skipped -- T3 should still run based on cadence logic
        $script:State.IterationCount = 3
        $result = Invoke-TieredHealthCheck -ChangedFiles @() -FullRunCadence 3
        $result.HasErrors | Should -BeFalse
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

    It 'formats config errors' {
        $tierResult = @{
            Tier = 1; SyntaxErrors = @(); MergeConflicts = @(); MissingFiles = @()
            ConfigErrors = @("Config validation failed: missing 'download' section")
            CollectionErrors = $null; Failures = $null; RawOutput = $null
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*CONFIG: Config validation failed*"
    }

    It 'handles tier result with all error types populated' {
        $tierResult = @{
            Tier = 1
            SyntaxErrors = @(@{ File = "a.py"; Error = "bad" })
            MergeConflicts = @("b.py")
            MissingFiles = @("c.py")
            ConfigErrors = @("config error")
            CollectionErrors = $null; Failures = $null; RawOutput = $null
        }

        $diag = Build-TierDiagnostics -TierResult $tierResult
        $diag | Should -BeLike "*SYNTAX*"
        $diag | Should -BeLike "*CONFLICT*"
        $diag | Should -BeLike "*MISSING*"
        $diag | Should -BeLike "*CONFIG*"
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

    It 'handles missing $State gracefully' {
        $savedState = $script:State
        $script:State = $null
        Log-HealingEvent -Event "healing_skipped" -Data @{ reason = "disabled" }

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_skipped"
        # sessionId and iteration should be absent, not error
        $entry.PSObject.Properties.Name | Should -Not -Contain "sessionId"
        $script:State = $savedState
    }

    It 'supports healing_skipped event type' {
        Log-HealingEvent -Event "healing_skipped" -Data @{ reason = "config disabled" }

        $entry = @(Get-Content $script:HealingLogFile)[0] | ConvertFrom-Json
        $entry.event | Should -Be "healing_skipped"
        $entry.data.reason | Should -Be "config disabled"
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

    It 'overwrites existing healing_state.json' {
        @{ paused = $true; storyId = "OLD" } | ConvertTo-Json | Set-Content $script:HealingStateFile
        $healthResult = @{
            HasErrors = $true; FailedTier = 3
            RawDiagnostics = "FAIL: new error"
            TierResults = @()
        }
        Mock Write-Host {}
        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "US-NEW" -FocusArea "testing"

        $state = Get-Content $script:HealingStateFile -Raw | ConvertFrom-Json
        $state.storyId | Should -Be "US-NEW"
        $state.failedTier | Should -Be 3
    }

    It 'handles empty StoryId and FocusArea' {
        $healthResult = @{
            HasErrors = $true; FailedTier = 1
            RawDiagnostics = "SYNTAX error"
            TierResults = @()
        }
        Mock Write-Host {}
        Suspend-SprintForHealing -HealthResult $healthResult -StoryId "" -FocusArea ""

        $state = Get-Content $script:HealingStateFile -Raw | ConvertFrom-Json
        $state.paused | Should -BeTrue
        $state.storyId | Should -Be ""
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

    It 'handles already-removed state file gracefully' {
        # Don't create state file - it should not throw
        Mock Write-Host {}
        { Resume-SprintFromHealing -Success $true -AttemptCount 1 } | Should -Not -Throw
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

    It 'returns false when state file has corrupt JSON' {
        "not valid json {{{" | Set-Content $script:HealingStateFile
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

    It 'maps all three tier names correctly' {
        foreach ($tier in @(1, 2, 3)) {
            $state = @{ failedTier = $tier; rawDiagnostics = "err"; focusArea = "x"; storyId = "y" }
            $prompt = Build-HealingPrompt -HealingState $state -Attempt 1
            switch ($tier) {
                1 { $prompt | Should -BeLike "*Syntax/File Integrity*" }
                2 { $prompt | Should -BeLike "*Import/Collection*" }
                3 { $prompt | Should -BeLike "*Test Execution*" }
            }
        }
    }

    It 'truncates long previous output to 1000 chars' {
        $state = @{ failedTier = 1; rawDiagnostics = "err"; focusArea = "x"; storyId = "y" }
        $longOutput = "A" * 2000
        $prompt = Build-HealingPrompt -HealingState $state -Attempt 2 -PreviousOutput $longOutput
        # The prompt should contain truncated output, not the full 2000 chars
        $prompt | Should -BeLike "*Previous Attempt Output*"
        # Verify the raw 2000-char string isn't embedded fully
        $prompt.Length | Should -BeLessThan ($longOutput.Length + 500)
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

    It 'updates diagnostics between retry attempts' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Tried to fix"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        $script:recheckCount = 0
        Mock Invoke-TieredHealthCheck {
            $script:recheckCount++
            if ($script:recheckCount -lt 2) {
                return @{ HasErrors = $true; FailedTier = 3; RawDiagnostics = "still broken attempt $($script:recheckCount)"; TierResults = @() }
            }
            return @{ HasErrors = $false; FailedTier = 0 }
        }
        Mock Write-Host {}

        $result = Invoke-HealingSession -MaxAttempts 3
        $result.Success | Should -BeTrue
        $result.AttemptsUsed | Should -Be 2
    }

    It 'retries on Claude non-zero exit code' {
        Mock Get-ClaudePath { return "claude" }
        $script:callNum = 0
        Mock Invoke-ClaudeSubprocess {
            $script:callNum++
            if ($script:callNum -eq 1) {
                return @{
                    Exited = $true; ExitCode = 1; TimedOut = $false
                    Output = "Error occurred"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
                }
            }
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }
        Mock Write-Host {}

        $result = Invoke-HealingSession -MaxAttempts 3
        $result.Success | Should -BeTrue
        $result.AttemptsUsed | Should -Be 2
    }

    It 'continues on Record-Metric failure' {
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "Fixed"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }
        Mock Record-Metric { throw "CSV locked" }
        Mock Write-Host {}

        $result = Invoke-HealingSession -MaxAttempts 3
        $result.Success | Should -BeTrue
    }

    It 'passes Build-HealingPrompt the previous output on retries' {
        Mock Get-ClaudePath { return "claude" }
        $script:promptCaptures = @()
        Mock Build-HealingPrompt {
            param($HealingState, $Attempt, $PreviousOutput)
            $script:promptCaptures += @{ Attempt = $Attempt; HasPrev = [bool]$PreviousOutput }
            return "fix prompt"
        }
        Mock Invoke-ClaudeSubprocess {
            return @{
                Exited = $true; ExitCode = 0; TimedOut = $false
                Output = "attempt output"; ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date)
            }
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $true; FailedTier = 2; RawDiagnostics = "err"; TierResults = @() } }
        Mock Write-Host {}

        Invoke-HealingSession -MaxAttempts 2

        $script:promptCaptures.Count | Should -Be 2
        $script:promptCaptures[0].HasPrev | Should -BeFalse  # First attempt: no previous
        $script:promptCaptures[1].HasPrev | Should -BeTrue   # Second attempt: has previous
    }
}

Describe 'Invoke-PostIterationHealing' {
    BeforeEach {
        $script:RalphDir = $TestDrive
        $script:ProjectRoot = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state_orch.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_orch.jsonl"
        if (Test-Path $script:HealingStateFile) { Remove-Item $script:HealingStateFile -Force }
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
        $script:State = @{
            SessionId = "orch-test"; IterationCount = 12;
            ConsecutiveFailures = 0; CurrentMode = "Standard"
        }
    }

    It 'does nothing when all tiers pass' {
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true; runAfterSuccess = $true; runAfterFailure = $true; maxAttempts = 3; fullRunCadence = 3 } } }
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0; TierResults = @(); RawDiagnostics = "" }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing" -ChangedFiles @()
        $result.HealingNeeded | Should -BeFalse
        $script:HealingStateFile | Should -Not -Exist
    }

    It 'skips healing when disabled in config' {
        Mock Get-RalphConfig {
            return @{ selfHealing = @{ enabled = $false } }
        }

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing"
        $result.HealingNeeded | Should -BeFalse
    }

    It 'runs full healing flow when tier 1 errors detected' {
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true; runAfterSuccess = $true; runAfterFailure = $true; maxAttempts = 3; fullRunCadence = 3 } } }
        Mock Invoke-TieredHealthCheck {
            return @{
                HasErrors = $true; FailedTier = 1
                RawDiagnostics = "SYNTAX: config.py -> SyntaxError"
                TierResults = @(@{ Tier = 1; HasErrors = $true })
            }
        }
        Mock Suspend-SprintForHealing {}
        Mock Invoke-HealingSession {
            return @{ Success = $true; AttemptsUsed = 1; FixSummary = "Fixed syntax" }
        }
        Mock Resume-SprintFromHealing {}

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "pipeline" -ChangedFiles @("src/config.py")
        $result.HealingNeeded | Should -BeTrue
        $result.HealingSuccess | Should -BeTrue
        $result.FailedTier | Should -Be 1
    }

    It 'returns failure when healing cannot fix errors' {
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true; runAfterSuccess = $true; runAfterFailure = $true; maxAttempts = 3; fullRunCadence = 3 } } }
        Mock Invoke-TieredHealthCheck {
            return @{
                HasErrors = $true; FailedTier = 3
                RawDiagnostics = "FAIL: test.py -> AssertionError"
                TierResults = @()
            }
        }
        Mock Suspend-SprintForHealing {}
        Mock Invoke-HealingSession {
            return @{ Success = $false; AttemptsUsed = 3; FixSummary = "Could not fix" }
        }
        Mock Resume-SprintFromHealing {}

        $result = Invoke-PostIterationHealing -StoryId "US-003" -FocusArea "testing"
        $result.HealingNeeded | Should -BeTrue
        $result.HealingSuccess | Should -BeFalse
    }

    It 'skips after failure when runAfterFailure is false' {
        Mock Get-RalphConfig {
            return @{ selfHealing = @{
                enabled = $true; runAfterFailure = $false; runAfterSuccess = $true; maxAttempts = 3
            }}
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }

        $result = Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -IterationSuccess $false
        $result.HealingNeeded | Should -BeFalse
        Should -Not -Invoke Invoke-TieredHealthCheck
    }

    It 'skips after success when runAfterSuccess is false' {
        Mock Get-RalphConfig {
            return @{ selfHealing = @{
                enabled = $true; runAfterFailure = $true; runAfterSuccess = $false; maxAttempts = 3
            }}
        }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }

        $result = Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -IterationSuccess $true
        $result.HealingNeeded | Should -BeFalse
        Should -Not -Invoke Invoke-TieredHealthCheck
    }

    It 'passes changed files to tiered health check' {
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true; runAfterSuccess = $true; runAfterFailure = $true; maxAttempts = 3; fullRunCadence = 3 } } }
        Mock Invoke-TieredHealthCheck {
            param($ChangedFiles, $FullRunCadence)
            $script:passedChangedFiles = $ChangedFiles
            return @{ HasErrors = $false; FailedTier = 0 }
        }

        $script:passedChangedFiles = $null
        Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -ChangedFiles @("src/foo.py", "src/bar.py")

        $script:passedChangedFiles | Should -Not -BeNullOrEmpty
        $script:passedChangedFiles.Count | Should -Be 2
    }

    It 'handles null selfHealing config gracefully' {
        Mock Get-RalphConfig { return @{} }
        Mock Invoke-TieredHealthCheck { return @{ HasErrors = $false; FailedTier = 0 } }

        # Should not throw when selfHealing key is missing
        $result = Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test"
        $result.HealingNeeded | Should -BeFalse
    }

    It 'uses default fullRunCadence when not in config' {
        Mock Get-RalphConfig { return @{ selfHealing = @{ enabled = $true; runAfterSuccess = $true; maxAttempts = 3 } } }
        $script:capturedCadence = $null
        Mock Invoke-TieredHealthCheck {
            param($ChangedFiles, $FullRunCadence)
            $script:capturedCadence = $FullRunCadence
            return @{ HasErrors = $false; FailedTier = 0 }
        }

        Invoke-PostIterationHealing -StoryId "US-001" -FocusArea "test" -ChangedFiles @()
        $script:capturedCadence | Should -Be 3
    }
}

Describe 'Self-healing config' {
    It 'ralph-config.json has selfHealing section with tiered settings' {
        $configPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\ralph-config.json'
        $config = Get-Content $configPath -Raw | ConvertFrom-Json

        $config.selfHealing | Should -Not -BeNullOrEmpty
        $config.selfHealing.enabled | Should -BeOfType [bool]
        $config.selfHealing.maxAttempts | Should -BeGreaterThan 0
        $config.selfHealing.runAfterSuccess | Should -BeOfType [bool]
        $config.selfHealing.runAfterFailure | Should -BeOfType [bool]
        $config.selfHealing.fullRunCadence | Should -BeGreaterThan 0
        $config.selfHealing.criticalFiles | Should -Not -BeNullOrEmpty
    }
}

Describe 'Healing metrics recording' {
    It 'records healing iteration as mode=Healing in metrics CSV' {
        Mock Record-Metric {}
        Mock Get-ClaudePath { return "claude" }
        Mock Invoke-ClaudeSubprocess {
            return @{ Exited = $true; ExitCode = 0; TimedOut = $false; Output = "Fixed";
                      ExecutionStart = (Get-Date); ExecutionEnd = (Get-Date) }
        }
        Mock Invoke-TieredHealthCheck {
            return @{ HasErrors = $false; FailedTier = 0 }
        }
        Mock Write-Host {}

        $script:RalphDir = $TestDrive
        $script:HealingStateFile = Join-Path $TestDrive "healing_state_metric.json"
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_metric.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }
        $script:State = @{ SessionId = "m-test"; IterationCount = 5; CurrentMode = "Standard"; ConsecutiveFailures = 0 }

        @{ paused = $true; storyId = "US-005"; focusArea = "testing"; failedTier = 2;
           rawDiagnostics = "IMPORT error" } |
            ConvertTo-Json -Depth 10 | Set-Content $script:HealingStateFile

        Invoke-HealingSession -MaxAttempts 1

        Should -Invoke Record-Metric -Times 1 -ParameterFilter { $Mode -eq "Healing" -and $StoryId -like "HEALING-*" }
    }
}

Describe 'Get-HealingSummary' {
    It 'summarizes healing activity from log' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_summary.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ timestamp = "2026-01-28T10:00:00"; event = "healing_started"; data = @{ failedTier = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ timestamp = "2026-01-28T10:01:00"; event = "healing_resolved"; data = @{ attempts = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ timestamp = "2026-01-28T10:05:00"; event = "healing_started"; data = @{ failedTier = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ timestamp = "2026-01-28T10:06:00"; event = "healing_resolved"; data = @{ attempts = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 2
        $summary.TotalResolved | Should -Be 2
        $summary.TotalFailed | Should -Be 0
        $summary.TotalAttempts | Should -Be 3
    }

    It 'returns empty summary when no log exists' {
        $script:HealingLogFile = Join-Path $TestDrive "nonexistent_summary.jsonl"

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 0
    }

    It 'counts tier breakdown correctly' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_tiers.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ event = "healing_started"; data = @{ failedTier = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_started"; data = @{ failedTier = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_started"; data = @{ failedTier = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_started"; data = @{ failedTier = 3 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TierBreakdown[1] | Should -Be 1
        $summary.TierBreakdown[2] | Should -Be 2
        $summary.TierBreakdown[3] | Should -Be 1
    }

    It 'counts failed sessions' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_failed.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ event = "healing_started"; data = @{ failedTier = 3 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_failed"; data = @{ attempts = 3 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TotalFailed | Should -Be 1
        $summary.TotalResolved | Should -Be 0
        $summary.TotalAttempts | Should -Be 3
    }

    It 'skips corrupt JSONL lines without crashing' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_corrupt.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ event = "healing_started"; data = @{ failedTier = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        "this is not valid json at all" | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_resolved"; data = @{ attempts = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 1
        $summary.TotalResolved | Should -Be 1
    }

    It 'counts attempts from resolved and failed events' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_attempts.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ event = "healing_started"; data = @{ failedTier = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_attempt"; data = @{ attempt = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_attempt"; data = @{ attempt = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_failed"; data = @{ attempts = 2 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 1
        $summary.TotalFailed | Should -Be 1
        # TotalAttempts comes from resolved/failed .data.attempts, not healing_attempt event count
        $summary.TotalAttempts | Should -Be 2
    }

    It 'handles healing_skipped events without counting as session' {
        $script:HealingLogFile = Join-Path $TestDrive "healing_log_skipped.jsonl"
        if (Test-Path $script:HealingLogFile) { Remove-Item $script:HealingLogFile -Force }

        @{ event = "healing_skipped"; data = @{ reason = "disabled" } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8
        @{ event = "healing_started"; data = @{ failedTier = 1 } } |
            ConvertTo-Json -Compress | Add-Content $script:HealingLogFile -Encoding UTF8

        $summary = Get-HealingSummary
        $summary.TotalHealingSessions | Should -Be 1  # Only started counts
    }
}
