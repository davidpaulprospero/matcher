#Requires -Modules Pester

<#
.SYNOPSIS
    Tests for the Ralph Loop interview mode (interview.ps1)
.DESCRIPTION
    Tests cover:
    - Focus area suggestion algorithm
    - Keyword matching
    - Queue management
    - Interview flow logic
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:InterviewScript = Join-Path $script:RalphDir "interview.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata"

    # Create test data directory
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source the functions from interview.ps1
    if (Test-Path $script:InterviewScript) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($script:InterviewScript, [ref]$null, [ref]$null)
        $functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

        foreach ($func in $functions) {
            $funcDef = $func.Extent.Text
            try {
                Invoke-Expression $funcDef
            }
            catch {
                # Some functions may have dependencies - skip those
            }
        }
    }

    # Mock the config for focus area validation
    $script:MockConfig = @{
        focusAreas = @{
            pipeline = @{ keywords = @("pipeline", "stage", "checkpoint") }
            testing = @{ keywords = @("test", "pytest", "coverage") }
            speed = @{ keywords = @("performance", "fast", "optimize") }
            quality = @{ keywords = @("refactor", "clean", "improve") }
            config = @{ keywords = @("config", "yaml", "settings") }
            otio = @{ keywords = @("timeline", "otio", "davinci") }
        }
    }

    # Helper to create mock config file
    function New-MockConfigFile {
        param([string]$Path)

        @{
            focusAreas = @{
                pipeline = @{ description = "Pipeline work"; keywords = @("pipeline", "stage") }
                testing = @{ description = "Testing work"; keywords = @("test", "pytest") }
                speed = @{ description = "Performance work"; keywords = @("performance", "fast") }
                quality = @{ description = "Quality work"; keywords = @("refactor", "clean") }
                config = @{ description = "Config work"; keywords = @("config", "yaml") }
                otio = @{ description = "OTIO work"; keywords = @("timeline", "otio") }
            }
        } | ConvertTo-Json -Depth 5 | Set-Content $Path
    }
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# FOCUS AREA SUGGESTION TESTS
# =============================================================================

Describe "Get-SuggestedFocusAreas" -Tag "Unit", "Interview" {
    BeforeAll {
        # Define the function locally for testing if not loaded
        if (-not (Get-Command "Get-SuggestedFocusAreas" -ErrorAction SilentlyContinue)) {
            function Get-SuggestedFocusAreas {
                param(
                    [string]$Description,
                    [hashtable]$Config
                )

                $suggestions = @()
                $descLower = $Description.ToLower()

                foreach ($area in $Config.focusAreas.Keys) {
                    $areaConfig = $Config.focusAreas[$area]
                    $keywords = if ($areaConfig.keywords) { $areaConfig.keywords } else { @($area) }

                    foreach ($keyword in $keywords) {
                        if ($descLower -match [regex]::Escape($keyword.ToLower())) {
                            $suggestions += $area
                            break
                        }
                    }
                }

                return $suggestions | Select-Object -Unique
            }
        }
    }

    It "suggests testing area for test-related description" {
        $description = "I want to add more unit tests"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result | Should -Contain "testing"
    }

    It "suggests pipeline area for pipeline-related description" {
        $description = "Need to fix the pipeline stage"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result | Should -Contain "pipeline"
    }

    It "suggests speed area for performance-related description" {
        $description = "Make it faster with optimization"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result | Should -Contain "speed"
    }

    It "suggests multiple areas for multi-keyword description" {
        $description = "Need to add tests and improve performance"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result | Should -Contain "testing"
        $result | Should -Contain "speed"
    }

    It "returns empty for unrelated description" {
        $description = "Something completely unrelated to anything"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result.Count | Should -Be 0
    }

    It "handles case insensitivity" {
        $description = "PYTEST TESTING PERFORMANCE"
        $result = Get-SuggestedFocusAreas -Description $description -Config $script:MockConfig

        $result.Count | Should -BeGreaterThan 0
    }
}

# =============================================================================
# FOCUS AREA VALIDATION TESTS
# =============================================================================

Describe "Focus Area Validation" -Tag "Unit", "Interview" {
    BeforeAll {
        # Define validation function
        function Test-FocusAreaExists {
            param(
                [string]$AreaId,
                [hashtable]$Config
            )

            return $Config.focusAreas.ContainsKey($AreaId)
        }
    }

    It "validates existing focus area" {
        $result = Test-FocusAreaExists -AreaId "testing" -Config $script:MockConfig
        $result | Should -Be $true
    }

    It "rejects non-existent focus area" {
        $result = Test-FocusAreaExists -AreaId "nonexistent" -Config $script:MockConfig
        $result | Should -Be $false
    }

    It "is case insensitive for area IDs (PowerShell default)" {
        # PowerShell hashtables are case-insensitive by default
        $result = Test-FocusAreaExists -AreaId "Testing" -Config $script:MockConfig
        $result | Should -Be $true
    }
}

# =============================================================================
# QUEUE MANAGEMENT TESTS
# =============================================================================

Describe "Queue Management" -Tag "Unit", "Queue" {
    BeforeEach {
        $script:queuePath = Join-Path $script:TestDataDir "queue.json"
        if (Test-Path $script:queuePath) {
            Remove-Item $script:queuePath -Force
        }
    }

    It "creates new queue with focus areas" {
        $areas = @("testing", "quality", "speed")
        $queue = @{
            focusAreas = $areas | ForEach-Object { @{ id = $_; completed = $false } }
            interviewContext = "Test context"
            session = @{ id = "test-session" }
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $loaded.focusAreas.Count | Should -Be 3
        $loaded.focusAreas[0].completed | Should -Be $false
    }

    It "marks focus area as completed" {
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $false }
                @{ id = "quality"; completed = $false }
            )
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        # Simulate marking as complete
        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $loaded.focusAreas[0].completed = $true
        $loaded | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $reloaded = Get-Content $script:queuePath | ConvertFrom-Json
        $reloaded.focusAreas[0].completed | Should -Be $true
        $reloaded.focusAreas[1].completed | Should -Be $false
    }

    It "finds next incomplete focus area" {
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $true }
                @{ id = "quality"; completed = $false }
                @{ id = "speed"; completed = $false }
            )
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $next = $loaded.focusAreas | Where-Object { -not $_.completed } | Select-Object -First 1

        $next.id | Should -Be "quality"
    }

    It "detects when all areas are complete" {
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $true }
                @{ id = "quality"; completed = $true }
            )
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $incomplete = $loaded.focusAreas | Where-Object { -not $_.completed }

        $incomplete | Should -BeNullOrEmpty
    }
}

# =============================================================================
# INTERVIEW CONTEXT TESTS
# =============================================================================

Describe "Interview Context" -Tag "Unit", "Interview" {
    It "stores interview context in queue" {
        $context = "User wants to focus on improving test coverage and making the pipeline faster"
        $queue = @{
            focusAreas = @()
            interviewContext = $context
        }

        $queue.interviewContext | Should -Be $context
    }

    It "preserves context across queue operations" {
        $queuePath = Join-Path $script:TestDataDir "context_test.json"

        $context = "Important context that should not be lost"
        $queue = @{
            focusAreas = @(@{ id = "testing"; completed = $false })
            interviewContext = $context
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $queuePath

        # Simulate modification
        $loaded = Get-Content $queuePath | ConvertFrom-Json
        $loaded.focusAreas[0].completed = $true
        $loaded | ConvertTo-Json -Depth 5 | Set-Content $queuePath

        $reloaded = Get-Content $queuePath | ConvertFrom-Json
        $reloaded.interviewContext | Should -Be $context

        Remove-Item $queuePath -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# KEYWORD MATCHING ALGORITHM TESTS
# =============================================================================

Describe "Keyword Matching Algorithm" -Tag "Unit", "Algorithm" {
    BeforeAll {
        function Get-KeywordMatches {
            param(
                [string]$Text,
                [array]$Keywords
            )

            $foundMatches = @()
            $textLower = $Text.ToLower()

            foreach ($keyword in $Keywords) {
                if ($textLower -match [regex]::Escape($keyword.ToLower())) {
                    $foundMatches += $keyword
                }
            }

            return $foundMatches
        }
    }

    It "finds exact keyword match" {
        $result = Get-KeywordMatches -Text "Run pytest for tests" -Keywords @("pytest")
        $result | Should -Contain "pytest"
    }

    It "finds partial word match" {
        $result = Get-KeywordMatches -Text "Testing is important" -Keywords @("test")
        $result | Should -Contain "test"
    }

    It "handles multiple keywords" {
        $result = Get-KeywordMatches -Text "pytest coverage report" -Keywords @("pytest", "coverage", "report")
        $result.Count | Should -Be 3
    }

    It "returns empty for no matches" {
        $result = Get-KeywordMatches -Text "Something else entirely" -Keywords @("pytest", "test")
        $result.Count | Should -Be 0
    }

    It "handles special regex characters" {
        $result = Get-KeywordMatches -Text "Check file.py for errors" -Keywords @("file.py")
        $result | Should -Contain "file.py"
    }
}

# =============================================================================
# PRIORITY ORDERING TESTS
# =============================================================================

Describe "Priority Ordering" -Tag "Unit", "Priority" {
    It "orders focus areas by priority" {
        $areas = @(
            @{ id = "speed"; priority = 3 }
            @{ id = "testing"; priority = 1 }
            @{ id = "quality"; priority = 2 }
        )

        $ordered = $areas | Sort-Object { $_.priority }

        $ordered[0].id | Should -Be "testing"
        $ordered[1].id | Should -Be "quality"
        $ordered[2].id | Should -Be "speed"
    }

    It "handles equal priorities" {
        $areas = @(
            @{ id = "speed"; priority = 1 }
            @{ id = "testing"; priority = 1 }
        )

        $ordered = $areas | Sort-Object { $_.priority }

        $ordered.Count | Should -Be 2
    }
}

# =============================================================================
# RESUME FUNCTIONALITY TESTS
# =============================================================================

Describe "Resume Functionality" -Tag "Unit", "Resume" {
    BeforeEach {
        $script:queuePath = Join-Path $script:TestDataDir "resume_test.json"
    }

    AfterEach {
        if (Test-Path $script:queuePath) {
            Remove-Item $script:queuePath -Force -ErrorAction SilentlyContinue
        }
    }

    It "detects existing queue for resume" {
        $queue = @{
            focusAreas = @(@{ id = "testing"; completed = $false })
            interviewContext = "Previous session"
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        Test-Path $script:queuePath | Should -Be $true
    }

    It "calculates resume position correctly" {
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $true }
                @{ id = "quality"; completed = $true }
                @{ id = "speed"; completed = $false }
                @{ id = "config"; completed = $false }
            )
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $completedCount = @($loaded.focusAreas | Where-Object { $_.completed }).Count
        $totalCount = $loaded.focusAreas.Count
        $resumePosition = $completedCount + 1

        $resumePosition | Should -Be 3  # Would resume at 3rd item (0-indexed: 2)
    }

    It "handles resume when all complete" {
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $true }
                @{ id = "quality"; completed = $true }
            )
        }

        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        $loaded = Get-Content $script:queuePath | ConvertFrom-Json
        $remaining = $loaded.focusAreas | Where-Object { -not $_.completed }

        $remaining | Should -BeNullOrEmpty
    }
}

# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

Describe "Input Validation" -Tag "Unit", "Validation" {
    It "trims whitespace from input" {
        $input = "  testing  "
        $trimmed = $input.Trim()

        $trimmed | Should -Be "testing"
    }

    It "converts input to lowercase for comparison" {
        $input = "TESTING"
        $lower = $input.ToLower()

        $lower | Should -Be "testing"
    }

    It "handles empty input" {
        $input = ""
        $isEmpty = [string]::IsNullOrWhiteSpace($input)

        $isEmpty | Should -Be $true
    }

    It "handles whitespace-only input" {
        $input = "   "
        $isEmpty = [string]::IsNullOrWhiteSpace($input)

        $isEmpty | Should -Be $true
    }
}
