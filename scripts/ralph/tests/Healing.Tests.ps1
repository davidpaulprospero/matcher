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
