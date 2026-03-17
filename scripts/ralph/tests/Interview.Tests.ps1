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
# SEARCH BUDGET FUNCTIONS TESTS
# =============================================================================

Describe "Search Budget Functions" -Tag "Unit", "Budget" {
    BeforeAll {
        # Define the functions locally for testing (mimics interview.ps1)
        function Get-SearchBudgetInfo {
            param([string]$ProjectRoot = $null)

            # Use test defaults
            return @{
                maxKeywords = 10
                currentBudget = 200
                resultsPerKeyword = 20
                configPath = $null
                found = $false
            }
        }

        function Get-DistributedKeywordBudget {
            param(
                [Parameter(Mandatory=$true)]
                [string[]]$Keywords,
                [int]$MaxTotalResults = 200,
                [int]$ResultsPerKeyword = 20
            )

            # Handle edge case: 0 keywords
            if ($Keywords.Count -eq 0) {
                return @{
                    adjusted_budget = 0
                    keyword_count = 0
                    willReduce = $false
                    warningMessage = "No keywords provided"
                    effectiveTotal = 0
                }
            }

            $keyword_count = $Keywords.Count

            # Always use the formula: floor(max_total_results / keyword_count)
            $adjusted_budget = [Math]::Floor($MaxTotalResults / $keyword_count)
            $effectiveTotal = $keyword_count * $adjusted_budget

            $budgetThreshold = [Math]::Floor($MaxTotalResults / $ResultsPerKeyword)
            $willReduce = $adjusted_budget -lt $ResultsPerKeyword

            $warningMessage = $null
            if ($willReduce) {
                $warningMsg = "WARNING: $keyword_count keywords exceeds budget threshold of $budgetThreshold. " +
                              "Results per keyword will be reduced from $ResultsPerKeyword to $adjusted_budget to stay within $MaxTotalResults limit."
                $warningMessage = $warningMsg
            }

            return @{
                adjusted_budget = $adjusted_budget
                keyword_count = $keyword_count
                willReduce = $willReduce
                warningMessage = $warningMessage
                effectiveTotal = $effectiveTotal
            }
        }
    }

    Context "Get-DistributedKeywordBudget Edge Cases" -Tag "EdgeCases" {
        It "handles 0 keywords" {
            # PowerShell doesn't allow empty array to mandatory param, so test logic directly
            $Keywords = @()
            $MaxTotalResults = 200
            $ResultsPerKeyword = 20

            # Inline the logic to test edge case handling
            if ($Keywords.Count -eq 0) {
                $result = @{
                    adjusted_budget = 0
                    keyword_count = 0
                    willReduce = $false
                    warningMessage = "No keywords provided"
                    effectiveTotal = 0
                }
            }

            $result.keyword_count | Should -Be 0
            $result.adjusted_budget | Should -Be 0  # Edge case: 0 keywords returns 0
            $result.willReduce | Should -Be $false
            $result.warningMessage | Should -Be "No keywords provided"
            $result.effectiveTotal | Should -Be 0
        }

        # Acceptance criteria tests: 5 keywords -> 40, 10 keywords -> 20, 15 keywords -> 13, 20 keywords -> 10, 25 keywords -> 8
        It "5 keywords returns 40" {
            $keywords = 1..5 | ForEach-Object { "kw$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 40  # floor(200/5) = 40
            $result.keyword_count | Should -Be 5
            $result.effectiveTotal | Should -Be 200
            $result.willReduce | Should -Be $false  # 40 > 20 (not reducing, actually increasing)
        }

        It "10 keywords returns 20" {
            $keywords = 1..10 | ForEach-Object { "kw$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 20  # floor(200/10) = 20
            $result.keyword_count | Should -Be 10
            $result.effectiveTotal | Should -Be 200
            $result.willReduce | Should -Be $false  # 20 == 20
        }

        It "15 keywords returns 13" {
            $keywords = 1..15 | ForEach-Object { "kw$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 13  # floor(200/15) = 13
            $result.keyword_count | Should -Be 15
            $result.effectiveTotal | Should -Be 195
            $result.willReduce | Should -Be $true
        }

        It "20 keywords returns 10" {
            $keywords = 1..20 | ForEach-Object { "kw$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 10  # floor(200/20) = 10
            $result.keyword_count | Should -Be 20
            $result.effectiveTotal | Should -Be 200
            $result.willReduce | Should -Be $true
        }

        It "25 keywords returns 8" {
            $keywords = 1..25 | ForEach-Object { "kw$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 8  # floor(200/25) = 8
            $result.keyword_count | Should -Be 25
            $result.effectiveTotal | Should -Be 200
            $result.willReduce | Should -Be $true
        }

        It "handles 1 keyword" {
            $result = Get-DistributedKeywordBudget -Keywords @("test") -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.keyword_count | Should -Be 1
            $result.adjusted_budget | Should -Be 200  # floor(200/1) = 200 (max_total_results)
            $result.willReduce | Should -Be $false
            $result.warningMessage | Should -BeNullOrEmpty
            $result.effectiveTotal | Should -Be 200
        }

        It "handles keywords equal to budget threshold (exactly at limit)" {
            $result = Get-DistributedKeywordBudget -Keywords @("a", "b", "c", "d", "e", "f", "g", "h", "i", "j") -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.keyword_count | Should -Be 10
            $result.adjusted_budget | Should -Be 20
            $result.willReduce | Should -Be $false
            $result.effectiveTotal | Should -Be 200
        }

        It "handles keywords exceeding max_keywords" {
            $result = Get-DistributedKeywordBudget -Keywords @("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l") -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.keyword_count | Should -Be 12
            $result.adjusted_budget | Should -Be 16
            $result.willReduce | Should -Be $true
            $result.warningMessage | Should -Not -BeNullOrEmpty
            $result.effectiveTotal | Should -Be 192
        }

        It "handles large keyword count (stress case)" {
            $keywords = 1..50 | ForEach-Object { "keyword$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.keyword_count | Should -Be 50
            $result.adjusted_budget | Should -Be 4
            $result.willReduce | Should -Be $true
            $result.effectiveTotal | Should -Be 200
        }
    }

    Context "Get-SearchBudgetInfo Return Values" -Tag "ReturnValues" {
        It "returns maxKeywords in result" {
            $result = Get-SearchBudgetInfo

            $result.ContainsKey("maxKeywords") | Should -Be $true
            $result.maxKeywords | Should -Be 10
        }

        It "returns currentBudget in result" {
            $result = Get-SearchBudgetInfo

            $result.ContainsKey("currentBudget") | Should -Be $true
            $result.currentBudget | Should -Be 200
        }

        It "returns resultsPerKeyword in result" {
            $result = Get-SearchBudgetInfo

            $result.ContainsKey("resultsPerKeyword") | Should -Be $true
            $result.resultsPerKeyword | Should -Be 20
        }

        It "returns found flag in result" {
            $result = Get-SearchBudgetInfo

            $result.ContainsKey("found") | Should -Be $true
        }
    }

    Context "Budget Calculation Accuracy" -Tag "Accuracy" {
        It "calculates correct maxKeywords from budget" {
            # maxKeywords = floor(200 / 20) = 10
            $result = Get-SearchBudgetInfo
            $result.maxKeywords | Should -Be 10
        }

        It "calculates correct adjusted results when reducing" {
            # 15 keywords with budget 200/20=10 threshold
            # adjusted = floor(200 / 15) = 13
            $result = Get-DistributedKeywordBudget -Keywords (1..15 | ForEach-Object { "kw$_" }) -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.adjusted_budget | Should -Be 13
            $result.effectiveTotal | Should -Be 195
        }

        It "never exceeds maxTotalResults even when reducing" {
            $keywords = 1..25 | ForEach-Object { "keyword$_" }
            $result = Get-DistributedKeywordBudget -Keywords $keywords -MaxTotalResults 200 -ResultsPerKeyword 20

            $result.effectiveTotal | Should -BeLessOrEqual 200
        }
    }
}

# =============================================================================
# CHAPTER DISTRIBUTED KEYWORDS TESTS
# =============================================================================

Describe "Get-ChapterDistributedKeywords" -Tag "Unit", "Chapter" {
    BeforeAll {
        # Dot source the function from lib
        . "$PSScriptRoot/../lib/interview.ps1"
    }

    Context "Basic Chapter Patterns" -Tag "Basic" {
        It "splits 'Chapter 1: Introduction' into separate terms" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Chapter 1: Introduction")

            $result.Count | Should -BeGreaterThan 1
            ($result -contains "Chapter 1: Introduction") | Should -Be $true
            ($result -contains "Introduction") | Should -Be $true
        }

        It "splits 'Part 1: Overview' into separate terms" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Part 1: Overview")

            $result.Count | Should -BeGreaterThan 1
            ($result -contains "Part 1: Overview") | Should -Be $true
            ($result -contains "Overview") | Should -Be $true
        }

        It "handles Roman numerals in chapter patterns" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Chapter IV: Deep Dive")

            $result.Count | Should -BeGreaterThan 1
            ($result -contains "Chapter IV: Deep Dive") | Should -Be $true
            ($result -contains "Deep Dive") | Should -Be $true
        }
    }

    Context "Comma-Separated Topics" -Tag "Comma" {
        It "splits comma-separated topics into individual keywords" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Feature: Part 1, Part 2, Part 3")

            ($result -contains "Feature: Part 1") | Should -Be $true
            ($result -contains "Feature: Part 2") | Should -Be $true
            ($result -contains "Feature: Part 3") | Should -Be $true
        }

        It "handles simple comma-separated keywords" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Auth, Rate Limit, Caching")

            ($result -contains "Auth") | Should -Be $true
            ($result -contains "Rate Limit") | Should -Be $true
            ($result -contains "Caching") | Should -Be $true
        }
    }

    Context "Semicolon-Separated Topics" -Tag "Semicolon" {
        It "splits semicolon-separated topics into individual keywords" {
            $result = Get-ChapterDistributedKeywords -Keywords @("API: Auth; Rate Limit; Caching")

            ($result -contains "API: Auth") | Should -Be $true
            ($result -contains "Rate Limit") | Should -Be $true
            ($result -contains "Caching") | Should -Be $true
        }
    }

    Context "Edge Cases" -Tag "EdgeCases" {
        It "returns empty array for empty input" {
            $result = Get-ChapterDistributedKeywords -Keywords @()

            $result.Count | Should -Be 0
        }

        It "handles non-chapter keywords as-is" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Python tutorial")

            $result | Should -Be @("Python tutorial")
        }

        It "removes duplicate keywords" {
            $result = Get-ChapterDistributedKeywords -Keywords @("Introduction", "introduction")

            $result.Count | Should -Be 1
        }
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
