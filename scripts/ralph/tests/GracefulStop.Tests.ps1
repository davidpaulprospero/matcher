#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Graceful Stop functionality
.DESCRIPTION
    Tests cover:
    - Signal file creation and deletion
    - Test-GracefulStopRequested function
    - Clear-GracefulStopSignal function
    - Request-GracefulStop function
    - Startup cleanup behavior
    - graceful-stop.ps1 utility script
#>

BeforeAll {
    # Get paths - PSScriptRoot is scripts/ralph/tests
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:GracefulStopScript = Join-Path $script:RalphDir "graceful-stop.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\gracefulstop"
    $script:SignalFile = Join-Path $script:TestDataDir "graceful_stop.signal"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source only the graceful stop functions from ralph.ps1 and lib/*.ps1
    . (Join-Path $PSScriptRoot 'test-helper.ps1')

    $gracefulStopFunctions = @(
        'Read-JsonFile',
        'Test-GracefulStopRequested',
        'Clear-GracefulStopSignal',
        'Request-GracefulStop'
    )

    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope -OnlyFunctions $gracefulStopFunctions

    # Override RalphDir for testing
    $global:RalphDir = $script:TestDataDir
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Signal File Tests
# =============================================================================

Describe "Graceful Stop Signal File" -Tag "Unit", "GracefulStop" {
    BeforeEach {
        # Clean up any existing signal file
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    AfterEach {
        # Clean up after each test
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Signal file creation" {
        It "Should create signal file with valid JSON" {
            # Create signal file manually
            @{
                requestedAt = (Get-Date).ToString("o")
                reason = "Test signal"
            } | ConvertTo-Json | Set-Content $script:SignalFile -Encoding UTF8

            Test-Path $script:SignalFile | Should -BeTrue
            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            $content.reason | Should -Be "Test signal"
            $content.requestedAt | Should -Not -BeNullOrEmpty
        }

        It "Should store ISO 8601 timestamp" {
            @{
                requestedAt = (Get-Date).ToString("o")
                reason = "Timestamp test"
            } | ConvertTo-Json | Set-Content $script:SignalFile -Encoding UTF8

            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            # ISO 8601 format should parse as datetime
            { [datetime]::Parse($content.requestedAt) } | Should -Not -Throw
        }
    }

    Context "Signal file detection" {
        It "Should detect when signal file exists" {
            "test" | Set-Content $script:SignalFile -Encoding UTF8
            Test-Path $script:SignalFile | Should -BeTrue
        }

        It "Should detect when signal file does not exist" {
            Test-Path $script:SignalFile | Should -BeFalse
        }
    }
}

# =============================================================================
# Test-GracefulStopRequested Tests
# =============================================================================

Describe "Test-GracefulStopRequested" -Tag "Unit", "GracefulStop" {
    BeforeEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
        # Set the RalphDir to test directory
        $script:RalphDir = $script:TestDataDir
    }

    AfterEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "When signal file exists" {
        It "Should return true" {
            @{
                requestedAt = (Get-Date).ToString("o")
                reason = "Test"
            } | ConvertTo-Json | Set-Content $script:SignalFile -Encoding UTF8

            Test-GracefulStopRequested | Should -BeTrue
        }

        It "Should handle malformed JSON gracefully" {
            "not valid json" | Set-Content $script:SignalFile -Encoding UTF8
            # Should still return true (file exists)
            Test-GracefulStopRequested | Should -BeTrue
        }
    }

    Context "When signal file does not exist" {
        It "Should return false" {
            Test-GracefulStopRequested | Should -BeFalse
        }
    }
}

# =============================================================================
# Clear-GracefulStopSignal Tests
# =============================================================================

Describe "Clear-GracefulStopSignal" -Tag "Unit", "GracefulStop" {
    BeforeEach {
        $script:RalphDir = $script:TestDataDir
    }

    AfterEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "When signal file exists" {
        It "Should remove the signal file" {
            @{
                requestedAt = (Get-Date).ToString("o")
                reason = "Test"
            } | ConvertTo-Json | Set-Content $script:SignalFile -Encoding UTF8

            Test-Path $script:SignalFile | Should -BeTrue
            Clear-GracefulStopSignal
            Test-Path $script:SignalFile | Should -BeFalse
        }
    }

    Context "When signal file does not exist" {
        It "Should not throw an error" {
            { Clear-GracefulStopSignal } | Should -Not -Throw
        }
    }
}

# =============================================================================
# Request-GracefulStop Tests
# =============================================================================

Describe "Request-GracefulStop" -Tag "Unit", "GracefulStop" {
    BeforeEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
        $script:RalphDir = $script:TestDataDir
    }

    AfterEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Signal creation" {
        It "Should create signal file" {
            Request-GracefulStop
            Test-Path $script:SignalFile | Should -BeTrue
        }

        It "Should use default reason when not provided" {
            Request-GracefulStop
            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            $content.reason | Should -Be "User requested graceful stop"
        }

        It "Should use custom reason when provided" {
            Request-GracefulStop -Reason "Custom test reason"
            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            $content.reason | Should -Be "Custom test reason"
        }

        It "Should include timestamp" {
            Request-GracefulStop
            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            $content.requestedAt | Should -Not -BeNullOrEmpty
        }
    }
}

# =============================================================================
# graceful-stop.ps1 Utility Script Tests
# =============================================================================

Describe "graceful-stop.ps1 Utility Script" -Tag "Unit", "GracefulStop" {
    BeforeAll {
        $script:OriginalLocation = Get-Location
    }

    BeforeEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    AfterEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
        Set-Location $script:OriginalLocation
    }

    Context "Script file" {
        It "Should exist" {
            Test-Path $script:GracefulStopScript | Should -BeTrue
        }

        It "Should be valid PowerShell" {
            $errors = $null
            $null = [System.Management.Automation.Language.Parser]::ParseFile(
                $script:GracefulStopScript,
                [ref]$null,
                [ref]$errors
            )
            $errors.Count | Should -Be 0
        }
    }

    Context "Status check" {
        It "Should not throw error when checking status with no pending stop" {
            # The script uses $PSScriptRoot to find signal file, so we need to
            # verify its output. We run in the actual ralph directory.
            $actualRalphDir = Split-Path -Parent $PSScriptRoot
            $actualSignalFile = Join-Path $actualRalphDir "graceful_stop.signal"

            # Ensure no signal exists
            if (Test-Path $actualSignalFile) {
                Remove-Item $actualSignalFile -Force -ErrorAction SilentlyContinue
            }

            # Script uses Write-Host which doesn't go to pipeline, so we just
            # verify it runs without error and doesn't create a signal file
            { & $script:GracefulStopScript -Status } | Should -Not -Throw
            Test-Path $actualSignalFile | Should -BeFalse
        }
    }
}

# =============================================================================
# Integration Tests
# =============================================================================

Describe "Graceful Stop Integration" -Tag "Integration", "GracefulStop" {
    BeforeEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
        $script:RalphDir = $script:TestDataDir
    }

    AfterEach {
        if (Test-Path $script:SignalFile) {
            Remove-Item -Path $script:SignalFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Full workflow" {
        It "Should complete request -> check -> clear cycle" {
            # 1. Request graceful stop
            Request-GracefulStop -Reason "Integration test"

            # 2. Check it was requested
            Test-GracefulStopRequested | Should -BeTrue

            # 3. Clear the signal
            Clear-GracefulStopSignal

            # 4. Verify cleared
            Test-GracefulStopRequested | Should -BeFalse
        }

        It "Should handle multiple requests" {
            Request-GracefulStop -Reason "First request"
            Request-GracefulStop -Reason "Second request"

            # Should have the latest reason
            $content = Get-Content $script:SignalFile -Raw | ConvertFrom-Json
            $content.reason | Should -Be "Second request"
        }
    }
}
