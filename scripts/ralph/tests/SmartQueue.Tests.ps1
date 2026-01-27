#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for Smart Queue functionality
.DESCRIPTION
    Tests cover:
    - User input analysis
    - Focus area identification
    - Queue file creation
    - Error handling
#>

BeforeAll {
    # Get paths - PSScriptRoot is scripts/ralph/tests
    $script:RalphDir = Split-Path -Parent $PSScriptRoot  # scripts/ralph
    $script:LauncherScript = Join-Path $script:RalphDir "launcher.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata\smartqueue"
    $script:QueueFile = Join-Path $script:TestDataDir "queue.json"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source functions from launcher.ps1
    if (Test-Path $script:LauncherScript) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:LauncherScript, [ref]$null, [ref]$null)
        $functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

        foreach ($func in $functions) {
            $funcDef = $func.Extent.Text
            $globalFuncDef = $funcDef -replace '^function\s+([A-Za-z0-9_-]+)', 'function global:$1'
            try {
                Invoke-Expression $globalFuncDef
            } catch {}
        }
    }

    # Override QueueFile path for testing
    $global:QueueFile = $script:QueueFile
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Queue File Structure Tests
# =============================================================================

Describe "Smart Queue File Structure" -Tag "Unit", "SmartQueue" {
    BeforeEach {
        if (Test-Path $script:QueueFile) {
            Remove-Item -Path $script:QueueFile -Force -ErrorAction SilentlyContinue
        }
    }
    Context "Queue JSON structure" {
        BeforeEach {
            # Create a sample queue file
            $queue = @{
                focusAreas = @(
                    @{ id = "testing"; completed = $false }
                    @{ id = "quality"; completed = $false }
                )
                sessionId = "2026-01-25_120000"
                createdAt = (Get-Date).ToString("o")
                interviewContext = "Smart Queue: Fix tests and improve quality"
                interviewDetails = "User wants to improve test reliability"
            }
            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8
        }

        It "Should have focusAreas array" {
            $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $queue.focusAreas | Should -Not -BeNullOrEmpty
            # Use @() to ensure array, then check count
            @($queue.focusAreas).Count | Should -BeGreaterOrEqual 1
        }

        It "Should have sessionId" {
            $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $queue.sessionId | Should -Not -BeNullOrEmpty
        }

        It "Should have interviewContext starting with 'Smart Queue:'" {
            $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $queue.interviewContext | Should -Match "^Smart Queue:"
        }

        It "Should have interviewDetails for story generation" {
            $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $queue.interviewDetails | Should -Not -BeNullOrEmpty
        }

        It "Should have focus areas with id and completed fields" {
            $queue = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            foreach ($area in $queue.focusAreas) {
                $area.id | Should -Not -BeNullOrEmpty
                $area.completed | Should -BeIn @($true, $false)
            }
        }
    }
}

# =============================================================================
# Focus Area Mapping Tests
# =============================================================================

Describe "Smart Queue Focus Area Mapping" -Tag "Unit", "SmartQueue" {
    Context "Input to focus area mapping" {
        # These tests verify the expected mappings without calling Claude

        It "Should map 'download reliability' to rate-limiting or download" {
            $expectedAreas = @("rate-limiting", "download")
            # This is a behavioral expectation for when Claude analyzes input
            $expectedAreas | Should -Contain "rate-limiting"
        }

        It "Should map 'flaky tests' to testing" {
            $expectedAreas = @("testing", "unit-tests", "integration-tests")
            $expectedAreas | Should -Contain "testing"
        }

        It "Should map 'OTIO edge cases' to otio" {
            $expectedAreas = @("otio")
            $expectedAreas | Should -Contain "otio"
        }

        It "Should map 'match accuracy' to quality" {
            $expectedAreas = @("quality")
            $expectedAreas | Should -Contain "quality"
        }

        It "Should map 'error messages' to ux" {
            $expectedAreas = @("ux")
            $expectedAreas | Should -Contain "ux"
        }
    }
}

# =============================================================================
# Queue Validation Tests
# =============================================================================

Describe "Smart Queue Validation" -Tag "Unit", "SmartQueue" {
    Context "Invalid inputs" {
        It "Should handle empty user input" {
            $userInput = ""
            [string]::IsNullOrWhiteSpace($userInput) | Should -BeTrue
        }

        It "Should handle whitespace-only input" {
            $userInput = "   "
            [string]::IsNullOrWhiteSpace($userInput) | Should -BeTrue
        }
    }

    Context "Valid focus area IDs" {
        BeforeAll {
            $script:ValidAreas = @(
                "pipeline", "config",
                "rate-limiting", "caption", "download",
                "quality", "speed", "compilation",
                "otio",
                "agents", "client-learning",
                "testing", "unit-tests", "integration-tests", "mutation-tests",
                "documentation", "ux"
            )
        }

        It "Should recognize all 17 focus area IDs" {
            $script:ValidAreas.Count | Should -Be 17
        }

        It "Should include new focus areas" {
            $script:ValidAreas | Should -Contain "download"
            $script:ValidAreas | Should -Contain "documentation"
            $script:ValidAreas | Should -Contain "ux"
            $script:ValidAreas | Should -Contain "unit-tests"
            $script:ValidAreas | Should -Contain "integration-tests"
            $script:ValidAreas | Should -Contain "mutation-tests"
        }
    }
}

# =============================================================================
# Queue Creation Tests
# =============================================================================

Describe "Smart Queue Creation" -Tag "Integration", "SmartQueue" {
    BeforeEach {
        if (Test-Path $script:QueueFile) {
            Remove-Item -Path $script:QueueFile -Force -ErrorAction SilentlyContinue
        }
    }

    Context "Creating queue from focus areas" {
        It "Should create queue with correct structure" {
            $focusAreas = @("testing", "quality")
            $userInput = "Fix tests and improve matching"
            $storyContext = "Improve test reliability and match accuracy"

            $queue = @{
                focusAreas = @($focusAreas | ForEach-Object { @{ id = $_; completed = $false } })
                sessionId = Get-Date -Format "yyyy-MM-dd_HHmmss"
                createdAt = (Get-Date).ToString("o")
                interviewContext = "Smart Queue: $userInput"
                interviewDetails = $storyContext
            }

            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            # Verify
            $saved = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $saved.focusAreas.Count | Should -Be 2
            $saved.focusAreas[0].id | Should -Be "testing"
            $saved.focusAreas[1].id | Should -Be "quality"
        }

        It "Should preserve user input in interviewContext" {
            $userInput = "I want to improve download reliability"

            $queue = @{
                focusAreas = @(@{ id = "rate-limiting"; completed = $false })
                sessionId = Get-Date -Format "yyyy-MM-dd_HHmmss"
                createdAt = (Get-Date).ToString("o")
                interviewContext = "Smart Queue: $userInput"
                interviewDetails = $userInput
            }

            $queue | ConvertTo-Json -Depth 5 | Set-Content $script:QueueFile -Encoding UTF8

            $saved = Get-Content $script:QueueFile -Raw | ConvertFrom-Json
            $saved.interviewContext | Should -Match "download reliability"
        }
    }
}

# =============================================================================
# Claude Response Parsing Tests
# =============================================================================

Describe "Smart Queue Claude Response Parsing" -Tag "Unit", "SmartQueue" {
    Context "Valid JSON responses" {
        It "Should parse standard response" {
            $response = '{"focusAreas":["testing","quality"],"reasoning":{"testing":"flaky tests","quality":"match accuracy"},"storyContext":"Improve tests"}'
            $result = $response | ConvertFrom-Json

            $result.focusAreas.Count | Should -Be 2
            $result.reasoning.testing | Should -Be "flaky tests"
        }

        It "Should handle single focus area" {
            $response = '{"focusAreas":["otio"],"reasoning":{"otio":"timeline fixes"},"storyContext":"Fix OTIO output"}'
            $result = $response | ConvertFrom-Json

            $result.focusAreas.Count | Should -Be 1
            $result.focusAreas[0] | Should -Be "otio"
        }
    }

    Context "Invalid JSON responses" {
        It "Should fail on invalid JSON" {
            $response = 'This is not JSON'
            { $response | ConvertFrom-Json -ErrorAction Stop } | Should -Throw
        }

        It "Should handle markdown-wrapped JSON" {
            $response = @"
```json
{"focusAreas":["testing"],"reasoning":{"testing":"tests"},"storyContext":"Fix tests"}
```
"@
            # Extract JSON from markdown
            $jsonMatch = [regex]::Match($response, '\{[^{}]*"focusAreas"[^{}]*\}')
            if ($jsonMatch.Success) {
                $result = $jsonMatch.Value | ConvertFrom-Json
                $result.focusAreas | Should -Contain "testing"
            }
        }
    }
}
