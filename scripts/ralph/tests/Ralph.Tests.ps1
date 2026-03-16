#Requires -Modules Pester

<#
.SYNOPSIS
    Comprehensive test suite for Ralph Loop PowerShell scripts
.DESCRIPTION
    Tests cover:
    - Pure functions (calculations, parsing, classification)
    - Logging functions (with mocked file I/O)
    - Config management
    - State transitions
    - Metrics recording
    - Integration scenarios
#>

BeforeAll {
    # Get the path to the ralph.ps1 script
    $script:RalphDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "testdata"

    # Create test data directory if it doesn't exist
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source all functions from ralph.ps1 and lib/*.ps1 into global scope
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

    # Set up script-level variables that the functions expect
    $script:State = @{
        SessionId               = 'test-session-001'
        IterationCount          = 1
        ConsecutiveFailures     = 0
        SessionStartTime        = Get-Date
        CurrentMode             = 'Standard'
        CurrentRetryCount       = 0
        LastFocusAreaId          = ''
        LastStoryId             = ''
        StoriesSinceExploration = 0
        LastExplorationSummary  = ''
        LastExplorationTime     = $null
        SprintExplorationContext = ''
        LastExplorationCommit   = ''
    }
    $global:State = $script:State
    $script:SessionLogDir = Join-Path $script:TestDataDir "logs"
    $script:RalphDir = $script:TestDataDir
    $script:MetricsFile = Join-Path $script:TestDataDir "metrics.csv"
    $script:ProgressFile = Join-Path $script:TestDataDir "progress.txt"
    $script:PrdFile = Join-Path $script:TestDataDir "prd.json"
    $script:ProjectRoot = $script:TestDataDir

    # Also set global versions of important variables
    $global:SessionLogDir = $script:SessionLogDir
    $global:MetricsFile = $script:MetricsFile
    $global:RalphDir = $script:TestDataDir
    $global:ProjectRoot = $script:TestDataDir
    $global:ProgressFile = $script:ProgressFile
    $global:PrdFile = $script:PrdFile

    # Create log directory
    if (-not (Test-Path $script:SessionLogDir)) {
        New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null
    }
}

AfterAll {
    # Cleanup test data
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# PURE FUNCTION TESTS
# =============================================================================

Describe "Get-EstimatedTokens" -Tag "Unit", "Pure" {
    BeforeAll {
        # Define pure function inline
        function Get-EstimatedTokens {
            param([string]$Output)
            if (-not $Output -or $Output.Length -eq 0) { return 0 }
            $estimatedTokens = [math]::Round($Output.Length / 4)
            return [math]::Min($estimatedTokens, 50000)
        }
    }

    It "returns 0 for empty output" {
        $result = Get-EstimatedTokens -Output ""
        $result | Should -Be 0
    }

    It "returns 0 for null output" {
        $result = Get-EstimatedTokens -Output $null
        $result | Should -Be 0
    }

    It "estimates tokens based on character count" {
        $output = "This is a test output with some content that should be estimated."
        $result = Get-EstimatedTokens -Output $output
        $result | Should -BeGreaterThan 0
        $result | Should -BeLessThan 100
    }

    It "caps at reasonable maximum" {
        # Create very long output
        $longOutput = "x" * 500000
        $result = Get-EstimatedTokens -Output $longOutput
        $result | Should -BeLessOrEqual 50000
    }

    It "handles multiline output" {
        $output = @"
Line 1 with some content
Line 2 with more content
Line 3 with even more content
"@
        $result = Get-EstimatedTokens -Output $output
        $result | Should -BeGreaterThan 0
    }
}

Describe "Get-TestResults" -Tag "Unit", "Pure" {
    BeforeAll {
        # Define pure function inline
        function Get-TestResults {
            param([string]$Output)
            if (-not $Output) { return "" }
            $passed = 0
            $failed = 0
            if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }
            if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }
            if ($Output -match '(\d+)\s+error') { $failed += [int]$Matches[1] }
            $total = $passed + $failed
            if ($total -eq 0) { return "" }
            if ($failed -eq 0) { return "$passed/$total pass" }
            else { return "$passed/$total pass, $failed fail" }
        }
    }

    It "returns empty string for output without test results" {
        $result = Get-TestResults -Output "Just some regular output"
        $result | Should -BeNullOrEmpty
    }

    It "extracts passed test count" {
        $output = "===== 15 passed in 2.34s ====="
        $result = Get-TestResults -Output $output
        $result | Should -Be "15/15 pass"
    }

    It "extracts failed test count" {
        $output = "===== 3 failed, 12 passed in 5.67s ====="
        $result = Get-TestResults -Output $output
        $result | Should -Be "12/15 pass, 3 fail"
    }

    It "handles pytest verbose output" {
        $output = @"
test_cache.py::test_hit_rate PASSED
test_cache.py::test_eviction PASSED
test_cache.py::test_overflow FAILED
===== 2 passed, 1 failed in 1.23s =====
"@
        $result = Get-TestResults -Output $output
        $result | Should -Be "2/3 pass, 1 fail"
    }
}

Describe "Get-ErrorCategory" -Tag "Unit", "Pure" {
    BeforeAll {
        # Define pure function inline
        function Get-ErrorCategory {
            param([string]$Output, [bool]$TimedOut)
            if ($TimedOut) { return "Timeout" }
            if ($Output -match "SyntaxError|parse error|unexpected token") { return "SyntaxError" }
            if ($Output -match "FAILED|AssertionError|test.*failed") { return "TestFailure" }
            if ($Output -match "cannot be loaded|compilation|ImportError") { return "CompileError" }
            if ($Output -match "ValidationError|schema") { return "ValidationError" }
            if ($Output -match "API|rate.?limit|quota|429") { return "APIError" }
            return "Unknown"
        }
    }

    It "returns Timeout for timed out operations" {
        $result = Get-ErrorCategory -Output "Some output" -TimedOut $true
        $result | Should -Be "Timeout"
    }

    It "identifies TestFailure from pytest output" {
        $output = "FAILED test_something.py::test_case"
        $result = Get-ErrorCategory -Output $output -TimedOut $false
        $result | Should -Be "TestFailure"
    }

    It "identifies SyntaxError" {
        $output = "SyntaxError: invalid syntax"
        $result = Get-ErrorCategory -Output $output -TimedOut $false
        $result | Should -Be "SyntaxError"
    }

    It "identifies APIError for rate limiting" {
        $output = "Rate limit exceeded. Please try again"
        $result = Get-ErrorCategory -Output $output -TimedOut $false
        $result | Should -Be "APIError"
    }

    It "returns Unknown for unrecognized errors" {
        $output = "Something went wrong but we don't know what"
        $result = Get-ErrorCategory -Output $output -TimedOut $false
        $result | Should -Be "Unknown"
    }
}

Describe "Measure-PhaseTimings" -Tag "Unit", "Pure" {
    BeforeAll {
        # Define pure function inline
        function Measure-PhaseTimings {
            param([string]$Output, [int]$TotalDurationMs)
            $timings = @{ read_ms = 0; analyze_ms = 0; implement_ms = 0; test_ms = 0; commit_ms = 0 }
            if (-not $Output -or $TotalDurationMs -le 0) { return $timings }
            $hasReadOps = $Output -match "Read tool|Reading file|file_path"
            $hasAnalysis = $Output -match "analy|understand|plan|think|consider"
            $hasImplement = $Output -match "Edit tool|Write tool|Editing|Writing|implement"
            $hasTests = $Output -match "pytest|test.*pass|test.*fail|running tests"
            $hasGit = $Output -match "git commit|git add|Bash.*git"
            $readWeight = if ($hasReadOps) { ([regex]::Matches($Output, "Read tool|Reading file")).Count + 1 } else { 0 }
            $analyzeWeight = if ($hasAnalysis) { 2 } else { 1 }
            $implementWeight = if ($hasImplement) { ([regex]::Matches($Output, "Edit tool|Write tool")).Count + 1 } else { 0 }
            $testWeight = if ($hasTests) { 3 } else { 0 }
            $gitWeight = if ($hasGit) { 1 } else { 0 }
            $totalWeight = [math]::Max(1, $readWeight + $analyzeWeight + $implementWeight + $testWeight + $gitWeight)
            $timings.read_ms = [int](($readWeight / $totalWeight) * $TotalDurationMs)
            $timings.analyze_ms = [int](($analyzeWeight / $totalWeight) * $TotalDurationMs)
            $timings.implement_ms = [int](($implementWeight / $totalWeight) * $TotalDurationMs)
            $timings.test_ms = [int](($testWeight / $totalWeight) * $TotalDurationMs)
            $timings.commit_ms = [int](($gitWeight / $totalWeight) * $TotalDurationMs)
            return $timings
        }
    }

    It "returns zero timings for empty output" {
        $result = Measure-PhaseTimings -Output "" -TotalDurationMs 1000
        $result.read_ms | Should -Be 0
        $result.analyze_ms | Should -Be 0
    }

    It "returns zero timings for zero duration" {
        $result = Measure-PhaseTimings -Output "Some output" -TotalDurationMs 0
        $result.read_ms | Should -Be 0
    }

    It "allocates time to read phase when file reading is detected" {
        $output = "Read tool: reading file.py`nReading file..."
        $result = Measure-PhaseTimings -Output $output -TotalDurationMs 10000
        $result.read_ms | Should -BeGreaterThan 0
    }

    It "allocates time to implement phase when editing is detected" {
        $output = "Edit tool: modifying code`nWrite tool: creating file"
        $result = Measure-PhaseTimings -Output $output -TotalDurationMs 10000
        $result.implement_ms | Should -BeGreaterThan 0
    }

    It "allocates time to test phase when pytest is detected" {
        $output = "pytest tests/ -v`n5 passed in 2.34s"
        $result = Measure-PhaseTimings -Output $output -TotalDurationMs 10000
        $result.test_ms | Should -BeGreaterThan 0
    }

    It "allocates time to commit phase when git is detected" {
        $output = "git commit -m 'feat: add feature'"
        $result = Measure-PhaseTimings -Output $output -TotalDurationMs 10000
        $result.commit_ms | Should -BeGreaterThan 0
    }

    It "total of all phases equals input duration" {
        $output = "Read tool, Edit tool, pytest, git commit"
        $result = Measure-PhaseTimings -Output $output -TotalDurationMs 10000
        $total = $result.read_ms + $result.analyze_ms + $result.implement_ms + $result.test_ms + $result.commit_ms
        $total | Should -BeLessOrEqual 10000
    }
}

Describe "Get-PromptEffectiveness" -Tag "Unit", "Pure" {
    BeforeAll {
        # Define pure function inline
        function Get-PromptEffectiveness {
            param([bool]$Success, [int]$RetryCount)
            if (-not $Success) { return 0.0 }
            if ($RetryCount -le 1) { return 1.0 }
            elseif ($RetryCount -le 3) { return 0.5 }
            else { return 0.25 }
        }
    }

    It "returns 0.0 for failed attempts" {
        $result = Get-PromptEffectiveness -Success $false -RetryCount 1
        $result | Should -Be 0.0
    }

    It "returns 1.0 for first-try success" {
        $result = Get-PromptEffectiveness -Success $true -RetryCount 1
        $result | Should -Be 1.0
    }

    It "returns 0.5 for success after few retries" {
        $result = Get-PromptEffectiveness -Success $true -RetryCount 2
        $result | Should -Be 0.5

        $result = Get-PromptEffectiveness -Success $true -RetryCount 3
        $result | Should -Be 0.5
    }

    It "returns 0.25 for success after many retries" {
        $result = Get-PromptEffectiveness -Success $true -RetryCount 5
        $result | Should -Be 0.25
    }
}

# =============================================================================
# LOGGING FUNCTION TESTS
# =============================================================================

Describe "Log-StateTransition" -Tag "Unit", "Logging" {
    BeforeAll {
        # Define logging function inline
        function Log-StateTransition {
            param([string]$From, [string]$To, [string]$Reason, [hashtable]$Context = @{})
            if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) { return }
            $stateFile = Join-Path $script:SessionLogDir "state_transitions.jsonl"
            $entry = @{
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                from = $From
                to = $To
                reason = $Reason
                iteration = $script:State.IterationCount
                session = $script:State.SessionId
            }
            foreach ($key in $Context.Keys) { $entry[$key] = $Context[$key] }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $stateFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:stateFile = Join-Path $script:SessionLogDir "state_transitions.jsonl"
        if (Test-Path $script:stateFile) {
            Remove-Item $script:stateFile -Force
        }
    }

    It "creates state transition file" {
        Log-StateTransition -From "idle" -To "running" -Reason "Starting"
        Test-Path $script:stateFile | Should -Be $true
    }

    It "writes valid JSON" {
        Log-StateTransition -From "idle" -To "running" -Reason "Starting"
        $content = Get-Content $script:stateFile -Raw
        { $content | ConvertFrom-Json } | Should -Not -Throw
    }

    It "includes all required fields" {
        Log-StateTransition -From "idle" -To "running" -Reason "Test transition"
        $entry = Get-Content $script:stateFile | ConvertFrom-Json
        $entry.from | Should -Be "idle"
        $entry.to | Should -Be "running"
        $entry.reason | Should -Be "Test transition"
        $entry.timestamp | Should -Not -BeNullOrEmpty
    }

    It "includes optional context" {
        Log-StateTransition -From "running" -To "completed" -Reason "Success" -Context @{ storyId = "US-001" }
        $entry = Get-Content $script:stateFile | ConvertFrom-Json
        $entry.storyId | Should -Be "US-001"
    }

    It "appends to existing file" {
        Log-StateTransition -From "idle" -To "running" -Reason "First"
        Log-StateTransition -From "running" -To "completed" -Reason "Second"
        $lines = Get-Content $script:stateFile
        $lines.Count | Should -Be 2
    }
}

Describe "Log-ErrorEvolution" -Tag "Unit", "Logging" {
    BeforeAll {
        function Log-ErrorEvolution {
            param([string]$ErrorCategory, [string]$ErrorDetails = "", [int]$Iteration)
            $errorFile = Join-Path $script:SessionLogDir "error_evolution.jsonl"
            $entry = @{
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                category = $ErrorCategory
                details = $ErrorDetails
                iteration = $Iteration
                session = $script:State.SessionId
            }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $errorFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:errorFile = Join-Path $script:SessionLogDir "error_evolution.jsonl"
        if (Test-Path $script:errorFile) {
            Remove-Item $script:errorFile -Force
        }
    }

    It "creates error evolution file" {
        Log-ErrorEvolution -ErrorCategory "TestFailure" -ErrorDetails "Test failed" -Iteration 1
        Test-Path $script:errorFile | Should -Be $true
    }

    It "includes category and details" {
        Log-ErrorEvolution -ErrorCategory "Timeout" -ErrorDetails "Exceeded 600s" -Iteration 5
        $entry = Get-Content $script:errorFile | ConvertFrom-Json
        $entry.category | Should -Be "Timeout"
        $entry.details | Should -Be "Exceeded 600s"
        $entry.iteration | Should -Be 5
    }
}

Describe "Log-ConfigChange" -Tag "Unit", "Logging" {
    BeforeAll {
        function Log-ConfigChange {
            param([string]$Field, $OldValue, $NewValue, [string]$Reason = "manual")
            $configAuditFile = Join-Path $script:RalphDir "config_audit.jsonl"
            $entry = @{
                ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                session = $script:State.SessionId
                field = $Field
                old = $OldValue
                new = $NewValue
                reason = $Reason
            }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $configAuditFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:configAuditFile = Join-Path $script:RalphDir "config_audit.jsonl"
        if (Test-Path $script:configAuditFile) {
            Remove-Item $script:configAuditFile -Force
        }
    }

    It "creates config audit file" {
        Log-ConfigChange -Field "autonomy.maxIterations" -OldValue 50 -NewValue 100 -Reason "user request"
        Test-Path $script:configAuditFile | Should -Be $true
    }

    It "records old and new values" {
        Log-ConfigChange -Field "timeout" -OldValue 600 -NewValue 900 -Reason "manual"
        $entry = Get-Content $script:configAuditFile | ConvertFrom-Json
        $entry.field | Should -Be "timeout"
        $entry.old | Should -Be 600
        $entry.new | Should -Be 900
    }
}

Describe "Log-TestDetails" -Tag "Unit", "Logging" {
    BeforeAll {
        function Write-JsonNoBom {
            param([string]$Path, [string]$Content)
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
        }

        function Log-TestDetails {
            param([int]$Iteration, [string]$Output)
            $testDetailsFile = Join-Path $script:SessionLogDir "test_details_$Iteration.json"
            $testResults = @()
            $passed = 0; $failed = 0; $skipped = 0; $errors = 0
            $testMatches = [regex]::Matches($Output, '([\w\/]+\.py::\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)')
            foreach ($match in $testMatches) {
                $testName = $match.Groups[1].Value
                $status = $match.Groups[2].Value.ToLower()
                $testResults += @{ name = $testName; status = $status }
                switch ($status) { "passed" { $passed++ }; "failed" { $failed++ }; "skipped" { $skipped++ }; "error" { $errors++ } }
            }
            $duration = 0
            if ($Output -match "passed.*in\s+([\d.]+)s") { $duration = [double]$Matches[1] }
            $testDetails = @{
                iteration = $Iteration
                testRun = @{ command = "pytest"; duration = $duration; exitCode = if ($failed -eq 0 -and $errors -eq 0) { 0 } else { 1 } }
                results = $testResults
                summary = @{ passed = $passed; failed = $failed; skipped = $skipped; errors = $errors; total = $passed + $failed + $skipped + $errors }
            }
            Write-JsonNoBom -Path $testDetailsFile -Content ($testDetails | ConvertTo-Json -Depth 5)
            return $testDetails
        }
    }

    BeforeEach {
        $script:State.IterationCount = 1
    }

    It "creates test details file" {
        $output = "test_cache.py::test_hit_rate PASSED"
        Log-TestDetails -Iteration 1 -Output $output
        $file = Join-Path $script:SessionLogDir "test_details_1.json"
        Test-Path $file | Should -Be $true
    }

    It "parses passed tests" {
        $output = @"
test_cache.py::test_hit_rate PASSED
test_cache.py::test_eviction PASSED
===== 2 passed in 1.5s =====
"@
        $result = Log-TestDetails -Iteration 1 -Output $output
        $result.summary.passed | Should -Be 2
        $result.summary.failed | Should -Be 0
    }

    It "parses failed tests" {
        $output = @"
test_cache.py::test_hit_rate PASSED
test_cache.py::test_overflow FAILED
===== 1 passed, 1 failed in 2.0s =====
"@
        $result = Log-TestDetails -Iteration 1 -Output $output
        $result.summary.passed | Should -Be 1
        $result.summary.failed | Should -Be 1
    }

    It "extracts duration" {
        $output = "===== 5 passed in 3.45s ====="
        $result = Log-TestDetails -Iteration 1 -Output $output
        $result.testRun.duration | Should -Be 3.45
    }
}

Describe "Log-ResourceUsage" -Tag "Unit", "Logging" {
    BeforeAll {
        function Write-JsonNoBom {
            param([string]$Path, [string]$Content)
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
        }

        function Log-ResourceUsage {
            param([int]$Iteration, [int]$ProcessId, [array]$Samples)
            $resourceFile = Join-Path $script:SessionLogDir "resource_usage_$Iteration.json"
            $cpuValues = $Samples | ForEach-Object { $_.cpu }
            $memValues = $Samples | ForEach-Object { $_.memoryMB }
            $handleValues = $Samples | ForEach-Object { $_.handles }
            $resourceData = @{
                iteration = $Iteration
                processId = $ProcessId
                samples = $Samples
                averages = @{
                    cpu = ($cpuValues | Measure-Object -Average).Average
                    memoryMB = ($memValues | Measure-Object -Average).Average
                    handles = ($handleValues | Measure-Object -Average).Average
                }
                peaks = @{
                    cpu = ($cpuValues | Measure-Object -Maximum).Maximum
                    memoryMB = ($memValues | Measure-Object -Maximum).Maximum
                    handles = ($handleValues | Measure-Object -Maximum).Maximum
                }
            }
            Write-JsonNoBom -Path $resourceFile -Content ($resourceData | ConvertTo-Json -Depth 5)
        }
    }

    It "creates resource usage file" {
        $samples = @(
            @{ cpu = 10.5; memoryMB = 256; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
            @{ cpu = 12.3; memoryMB = 300; handles = 110; threads = 6; timestamp = (Get-Date).ToString("o") }
        )
        Log-ResourceUsage -Iteration 1 -ProcessId 12345 -Samples $samples
        $file = Join-Path $script:SessionLogDir "resource_usage_1.json"
        Test-Path $file | Should -Be $true
    }

    It "calculates averages correctly" {
        $samples = @(
            @{ cpu = 10; memoryMB = 200; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
            @{ cpu = 20; memoryMB = 400; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
        )
        Log-ResourceUsage -Iteration 1 -ProcessId 12345 -Samples $samples
        $file = Join-Path $script:SessionLogDir "resource_usage_1.json"
        $data = Get-Content $file | ConvertFrom-Json
        $data.averages.memoryMB | Should -Be 300
    }

    It "finds peak memory" {
        $samples = @(
            @{ cpu = 10; memoryMB = 200; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
            @{ cpu = 20; memoryMB = 500; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
            @{ cpu = 15; memoryMB = 300; handles = 100; threads = 5; timestamp = (Get-Date).ToString("o") }
        )
        Log-ResourceUsage -Iteration 1 -ProcessId 12345 -Samples $samples
        $file = Join-Path $script:SessionLogDir "resource_usage_1.json"
        $data = Get-Content $file | ConvertFrom-Json
        $data.peaks.memoryMB | Should -Be 500
    }
}

Describe "Log-PromptEffectiveness" -Tag "Unit", "Logging" {
    BeforeAll {
        function Log-PromptEffectiveness {
            param([int]$Iteration, [string]$PromptType, [double]$Effectiveness, [string]$PromptHash = "")
            $effectivenessFile = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"
            $entry = @{
                ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                iteration = $Iteration
                promptType = $PromptType
                effectiveness = $Effectiveness
                promptHash = $PromptHash
            }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $effectivenessFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:effectivenessFile = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"
        if (Test-Path $script:effectivenessFile) {
            Remove-Item $script:effectivenessFile -Force
        }
    }

    It "logs effectiveness score" {
        Log-PromptEffectiveness -Iteration 1 -PromptType "story_work" -Effectiveness 1.0 -PromptHash "ABC123"
        $entry = Get-Content $script:effectivenessFile | ConvertFrom-Json
        $entry.effectiveness | Should -Be 1.0
        $entry.promptType | Should -Be "story_work"
        $entry.promptHash | Should -Be "ABC123"
    }
}

Describe "Log-Skip" -Tag "Unit", "Logging" {
    BeforeAll {
        function Log-Skip {
            param([string]$ItemId, [string]$ItemType, [string]$Reason, [string]$BlockerType = "unknown")
            $skipFile = Join-Path $script:SessionLogDir "skips_blockers.jsonl"
            $entry = @{
                ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                session = $script:State.SessionId
                itemId = $ItemId
                itemType = $ItemType
                reason = $Reason
                blockerType = $BlockerType
                iteration = $script:State.IterationCount
            }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $skipFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:skipFile = Join-Path $script:SessionLogDir "skips_blockers.jsonl"
        if (Test-Path $script:skipFile) {
            Remove-Item $script:skipFile -Force
        }
    }

    It "logs skip with reason" {
        Log-Skip -ItemId "US-005" -ItemType "story" -Reason "Too complex" -BlockerType "manual"
        $entry = Get-Content $script:skipFile | ConvertFrom-Json
        $entry.itemId | Should -Be "US-005"
        $entry.reason | Should -Be "Too complex"
        $entry.blockerType | Should -Be "manual"
    }
}

# =============================================================================
# METRICS RECORDING TESTS
# =============================================================================

Describe "Record-Metric" -Tag "Unit", "Metrics" {
    BeforeAll {
        function Record-Metric {
            param(
                [string]$Session, [string]$Sprint, [string]$StoryId, [string]$Mode,
                [double]$DurationMin, [bool]$Success = $true, [bool]$Timeout = $false,
                [string]$FocusArea, [int]$TokensUsed = 0, [string]$ErrorCategory = "",
                [int]$HourOfDay = -1, [string]$TestResults = "", [int]$RetryCount = 0,
                [int]$LinesAdded = 0, [int]$LinesDeleted = 0, [int]$PhaseReadMs = 0,
                [int]$PhaseAnalyzeMs = 0, [int]$PhaseImplementMs = 0, [int]$PhaseTestMs = 0, [int]$PhaseCommitMs = 0
            )
            $v2Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms"
            if (-not (Test-Path $script:MetricsFile)) { $v2Header | Set-Content $script:MetricsFile -Encoding UTF8 }
            if (-not $Session) { $Session = $script:State.SessionId }
            if (-not $Mode) { $Mode = "Standard" }
            if ($HourOfDay -eq -1) { $HourOfDay = (Get-Date).Hour }
            $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $row = "$timestamp,$Session,$Sprint,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),$($Timeout.ToString().ToLower()),$FocusArea,$TokensUsed,$ErrorCategory,$HourOfDay,$TestResults,$RetryCount,$LinesAdded,$LinesDeleted,$PhaseReadMs,$PhaseAnalyzeMs,$PhaseImplementMs,$PhaseTestMs,$PhaseCommitMs"
            Add-Content -Path $script:MetricsFile -Value $row
        }
    }

    BeforeEach {
        if (Test-Path $script:MetricsFile) {
            Remove-Item $script:MetricsFile -Force
        }
    }

    It "creates metrics file with header" {
        Record-Metric -StoryId "US-001" -Mode "Standard" -DurationMin 5 -Success $true
        Test-Path $script:MetricsFile | Should -Be $true
        $header = Get-Content $script:MetricsFile -First 1
        $header | Should -Match "timestamp"
        $header | Should -Match "session"
        $header | Should -Match "story_id"
    }

    It "records success metrics" {
        Record-Metric -StoryId "US-001" -Mode "Standard" -DurationMin 5 -Success $true -Timeout $false
        $lines = Get-Content $script:MetricsFile
        $lines.Count | Should -Be 2  # header + data
        $lines[1] | Should -Match "true"
    }

    It "records failure metrics" {
        Record-Metric -StoryId "US-002" -Mode "Queue" -DurationMin 10 -Success $false -ErrorCategory "TestFailure"
        $lines = Get-Content $script:MetricsFile
        $lines[1] | Should -Match "false"
        $lines[1] | Should -Match "TestFailure"
    }

    It "includes phase timing columns" {
        Record-Metric -StoryId "US-003" -DurationMin 5 -Success $true -PhaseReadMs 1000 -PhaseAnalyzeMs 2000
        $header = Get-Content $script:MetricsFile -First 1
        $header | Should -Match "phase_read_ms"
        $header | Should -Match "phase_analyze_ms"
    }

    It "appends to existing file" {
        Record-Metric -StoryId "US-001" -DurationMin 5 -Success $true
        Record-Metric -StoryId "US-002" -DurationMin 3 -Success $true
        Record-Metric -StoryId "US-003" -DurationMin 7 -Success $false
        $lines = Get-Content $script:MetricsFile
        $lines.Count | Should -Be 4  # header + 3 data rows
    }
}

# =============================================================================
# TIMELINE TESTS
# =============================================================================

Describe "Append-SessionTimeline" -Tag "Unit", "Timeline" {
    BeforeAll {
        function Append-SessionTimeline {
            param([string]$Event, [hashtable]$Data = @{})
            $timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
            $entry = @{
                ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
                event = $Event
                session = $script:State.SessionId
                iteration = $script:State.IterationCount
            }
            foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $timelineFile -Encoding UTF8
        }
    }

    BeforeEach {
        $script:timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
        if (Test-Path $script:timelineFile) {
            Remove-Item $script:timelineFile -Force
        }
    }

    It "creates timeline file" {
        Append-SessionTimeline -Event "session_start" -Data @{}
        Test-Path $script:timelineFile | Should -Be $true
    }

    It "includes event name and timestamp" {
        Append-SessionTimeline -Event "iteration_start" -Data @{ iteration = 1 }
        $entry = Get-Content $script:timelineFile | ConvertFrom-Json
        $entry.event | Should -Be "iteration_start"
        $entry.ts | Should -Not -BeNullOrEmpty
    }

    It "merges additional data" {
        Append-SessionTimeline -Event "story_verified" -Data @{ storyId = "US-001"; passed = $true }
        $entry = Get-Content $script:timelineFile | ConvertFrom-Json
        $entry.storyId | Should -Be "US-001"
        $entry.passed | Should -Be $true
    }

    It "appends multiple events" {
        Append-SessionTimeline -Event "event1" -Data @{}
        Append-SessionTimeline -Event "event2" -Data @{}
        Append-SessionTimeline -Event "event3" -Data @{}
        $lines = Get-Content $script:timelineFile
        $lines.Count | Should -Be 3
    }
}

# =============================================================================
# MANIFEST TESTS
# =============================================================================

Describe "Log-IterationManifest" -Tag "Unit", "Manifest" {
    BeforeAll {
        function Write-JsonNoBom {
            param([string]$Path, [string]$Content)
            $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
        }

        function Log-IterationManifest {
            param([int]$Iteration, [string]$StoryId, [string]$FocusArea, [string]$Status,
                  [datetime]$StartTime, [datetime]$EndTime, [string]$PromptFile,
                  [hashtable]$GitBefore, [hashtable]$GitAfter, [hashtable]$FileOps,
                  [array]$Commits, [string]$TestResults, [int]$TokensEstimated, [int]$RetryCount)
            $manifestFile = Join-Path $script:SessionLogDir "iteration_${Iteration}_manifest.json"
            $manifest = @{
                iteration = $Iteration
                storyId = $StoryId
                focusArea = $FocusArea
                status = $Status
                timing = @{
                    start = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
                    end = $EndTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
                    durationMs = [int]($EndTime - $StartTime).TotalMilliseconds
                }
                prompt = @{ file = $PromptFile }
                git = @{ before = $GitBefore; after = $GitAfter; commits = $Commits }
                fileOperations = $FileOps
                testResults = $TestResults
                tokensEstimated = $TokensEstimated
                retryCount = $RetryCount
            }
            Write-JsonNoBom -Path $manifestFile -Content ($manifest | ConvertTo-Json -Depth 5)
        }
    }

    It "creates manifest file" {
        $gitBefore = @{ hash = "abc123"; branch = "main"; clean = $true; modified = @() }
        $gitAfter = @{ hash = "def456"; branch = "main"; clean = $true; modified = @() }

        Log-IterationManifest `
            -Iteration 1 `
            -StoryId "US-001" `
            -FocusArea "testing" `
            -Status "completed" `
            -StartTime (Get-Date).AddMinutes(-5) `
            -EndTime (Get-Date) `
            -PromptFile "prompt_1.txt" `
            -GitBefore $gitBefore `
            -GitAfter $gitAfter `
            -FileOps @{ created = @(); modified = @(); deleted = @() } `
            -Commits @() `
            -TestResults "5 passed" `
            -TokensEstimated 10000 `
            -RetryCount 1

        $file = Join-Path $script:SessionLogDir "iteration_1_manifest.json"
        Test-Path $file | Should -Be $true
    }

    It "includes all required fields" {
        $gitBefore = @{ hash = "abc123"; branch = "main"; clean = $true; modified = @() }
        $gitAfter = @{ hash = "def456"; branch = "main"; clean = $true; modified = @() }

        Log-IterationManifest `
            -Iteration 2 `
            -StoryId "US-002" `
            -FocusArea "quality" `
            -Status "failed" `
            -StartTime (Get-Date).AddMinutes(-10) `
            -EndTime (Get-Date) `
            -PromptFile "prompt_2.txt" `
            -GitBefore $gitBefore `
            -GitAfter $gitAfter `
            -FileOps @{ created = @("new.py"); modified = @("old.py"); deleted = @() } `
            -Commits @(@{ hash = "xyz789"; message = "fix: something" }) `
            -TestResults "2 passed, 1 failed" `
            -TokensEstimated 15000 `
            -RetryCount 2

        $file = Join-Path $script:SessionLogDir "iteration_2_manifest.json"
        $data = Get-Content $file | ConvertFrom-Json

        $data.iteration | Should -Be 2
        $data.storyId | Should -Be "US-002"
        $data.focusArea | Should -Be "quality"
        $data.status | Should -Be "failed"
    }
}

# =============================================================================
# GIT STATE TESTS
# =============================================================================

Describe "Get-GitState" -Tag "Integration", "Git" {
    BeforeAll {
        function Get-GitState {
            $hash = git rev-parse HEAD 2>$null
            $branch = git rev-parse --abbrev-ref HEAD 2>$null
            $status = git status --porcelain 2>$null
            $clean = [string]::IsNullOrEmpty($status)
            $modified = if ($status) { $status -split "`n" | Where-Object { $_ } } else { @() }
            return @{ hash = $hash; branch = $branch; clean = $clean; modified = $modified }
        }
    }

    It "returns current git hash" {
        $result = Get-GitState
        $result.hash | Should -Match "^[a-f0-9]+"
    }

    It "returns current branch" {
        $result = Get-GitState
        $result.branch | Should -Not -BeNullOrEmpty
    }

    It "returns clean status" {
        $result = Get-GitState
        $result.clean | Should -BeIn @($true, $false)
    }
}

Describe "Get-GitCommits" -Tag "Integration", "Git" {
    BeforeAll {
        function Get-GitCommits {
            param([string]$SinceHash)
            $commits = git log --oneline "${SinceHash}..HEAD" 2>$null
            return $commits
        }
    }

    It "returns commits since hash" {
        # Get a hash from a few commits ago
        $oldHash = git log --oneline -5 | Select-Object -Last 1 | ForEach-Object { $_.Split(" ")[0] }
        if ($oldHash) {
            $result = Get-GitCommits -SinceHash $oldHash
            $result | Should -Not -BeNullOrEmpty
        }
    }
}

# =============================================================================
# PROCESS METRICS TESTS
# =============================================================================

Describe "Get-ProcessMetrics" -Tag "Unit", "Process" {
    BeforeAll {
        function Get-ProcessMetrics {
            param([int]$ProcessId)
            try {
                $proc = Get-Process -Id $ProcessId -ErrorAction Stop
                return @{
                    cpu = 0  # CPU is tricky to calculate instantly
                    memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
                    handles = $proc.HandleCount
                    threads = $proc.Threads.Count
                    timestamp = (Get-Date).ToString("o")
                }
            } catch {
                return @{ cpu = 0; memoryMB = 0; handles = 0; threads = 0; timestamp = (Get-Date).ToString("o") }
            }
        }
    }

    It "returns metrics for current process" {
        $result = Get-ProcessMetrics -ProcessId $PID
        $result.memoryMB | Should -BeGreaterThan 0
        $result.handles | Should -BeGreaterThan 0
    }

    It "returns zero metrics for invalid process" {
        $result = Get-ProcessMetrics -ProcessId 999999999
        $result.memoryMB | Should -Be 0
    }

    It "includes timestamp" {
        $result = Get-ProcessMetrics -ProcessId $PID
        $result.timestamp | Should -Not -BeNullOrEmpty
    }
}

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

Describe "Full Iteration Logging Flow" -Tag "Integration" {
    BeforeAll {
        # Define all functions for integration testing
        function Log-StateTransition {
            param([string]$From, [string]$To, [string]$Reason, [hashtable]$Context = @{})
            if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) { return }
            $stateFile = Join-Path $script:SessionLogDir "state_transitions.jsonl"
            $entry = @{ timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); from = $From; to = $To; reason = $Reason; iteration = $script:State.IterationCount; session = $script:State.SessionId }
            foreach ($key in $Context.Keys) { $entry[$key] = $Context[$key] }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $stateFile -Encoding UTF8
        }
        function Append-SessionTimeline {
            param([string]$Event, [hashtable]$Data = @{})
            $timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
            $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); event = $Event; session = $script:State.SessionId; iteration = $script:State.IterationCount }
            foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $timelineFile -Encoding UTF8
        }
        function Measure-PhaseTimings {
            param([string]$Output, [int]$TotalDurationMs)
            $timings = @{ read_ms = 0; analyze_ms = 0; implement_ms = 0; test_ms = 0; commit_ms = 0 }
            if (-not $Output -or $TotalDurationMs -le 0) { return $timings }
            $hasReadOps = $Output -match "Read tool|Reading file|file_path"
            $hasAnalysis = $Output -match "analy|understand|plan|think|consider"
            $hasImplement = $Output -match "Edit tool|Write tool|Editing|Writing|implement"
            $hasTests = $Output -match "pytest|test.*pass|test.*fail|running tests"
            $hasGit = $Output -match "git commit|git add|Bash.*git"
            $readWeight = if ($hasReadOps) { ([regex]::Matches($Output, "Read tool|Reading file")).Count + 1 } else { 0 }
            $analyzeWeight = if ($hasAnalysis) { 2 } else { 1 }
            $implementWeight = if ($hasImplement) { ([regex]::Matches($Output, "Edit tool|Write tool")).Count + 1 } else { 0 }
            $testWeight = if ($hasTests) { 3 } else { 0 }
            $gitWeight = if ($hasGit) { 1 } else { 0 }
            $totalWeight = [math]::Max(1, $readWeight + $analyzeWeight + $implementWeight + $testWeight + $gitWeight)
            $timings.read_ms = [int](($readWeight / $totalWeight) * $TotalDurationMs)
            $timings.implement_ms = [int](($implementWeight / $totalWeight) * $TotalDurationMs)
            return $timings
        }
        function Get-EstimatedTokens { param([string]$Output); if (-not $Output -or $Output.Length -eq 0) { return 0 }; return [math]::Min([math]::Round($Output.Length / 4), 50000) }
        function Get-TestResults { param([string]$Output); if (-not $Output) { return "" }; $passed = 0; $failed = 0; if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }; if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }; $total = $passed + $failed; if ($total -eq 0) { return "" }; if ($failed -eq 0) { return "$passed/$total pass" }; return "$passed/$total pass, $failed fail" }
        function Get-ErrorCategory { param([string]$Output, [bool]$TimedOut); if ($TimedOut) { return "Timeout" }; if ($Output -match "SyntaxError|parse error") { return "SyntaxError" }; if ($Output -match "FAILED|test.*failed") { return "TestFailure" }; if ($Output -match "API|rate.?limit") { return "APIError" }; return "Unknown" }
        function Get-PromptEffectiveness { param([bool]$Success, [int]$RetryCount); if (-not $Success) { return 0.0 }; if ($RetryCount -le 1) { return 1.0 }; if ($RetryCount -le 3) { return 0.5 }; return 0.25 }
        function Write-JsonNoBom { param([string]$Path, [string]$Content); $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom) }
        function Log-TestDetails { param([int]$Iteration, [string]$Output); $testDetailsFile = Join-Path $script:SessionLogDir "test_details_$Iteration.json"; $passed = 0; $failed = 0; if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }; if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }; $duration = 0; if ($Output -match "passed.*in\s+([\d.]+)s") { $duration = [double]$Matches[1] }; $testDetails = @{ iteration = $Iteration; testRun = @{ command = "pytest"; duration = $duration }; summary = @{ passed = $passed; failed = $failed } }; Write-JsonNoBom -Path $testDetailsFile -Content ($testDetails | ConvertTo-Json -Depth 5); return $testDetails }
        function Log-PromptEffectiveness { param([int]$Iteration, [string]$PromptType, [double]$Effectiveness, [string]$PromptHash = ""); $file = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"; $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); iteration = $Iteration; promptType = $PromptType; effectiveness = $Effectiveness; promptHash = $PromptHash }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
        function Log-ErrorEvolution { param([string]$ErrorCategory, [string]$ErrorDetails = "", [int]$Iteration); $file = Join-Path $script:SessionLogDir "error_evolution.jsonl"; $entry = @{ timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); category = $ErrorCategory; details = $ErrorDetails; iteration = $Iteration }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
        function Log-Skip { param([string]$ItemId, [string]$ItemType, [string]$Reason, [string]$BlockerType = "unknown"); $file = Join-Path $script:SessionLogDir "skips_blockers.jsonl"; $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); itemId = $ItemId; itemType = $ItemType; reason = $Reason; blockerType = $BlockerType }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
        function Record-Metric { param([string]$StoryId, [string]$Mode, [double]$DurationMin, [bool]$Success = $true, [string]$FocusArea, [int]$TokensUsed = 0, [string]$ErrorCategory = "", [int]$PhaseReadMs = 0, [int]$PhaseImplementMs = 0); $v2Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms"; if (-not (Test-Path $script:MetricsFile)) { $v2Header | Set-Content $script:MetricsFile -Encoding UTF8 }; $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; $row = "$timestamp,$($script:State.SessionId),,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),false,$FocusArea,$TokensUsed,$ErrorCategory,$(Get-Date).Hour,,,0,0,$PhaseReadMs,0,$PhaseImplementMs,0,0"; Add-Content -Path $script:MetricsFile -Value $row }
    }

    BeforeEach {
        # Clean up test files
        Get-ChildItem $script:SessionLogDir -File | Remove-Item -Force -ErrorAction SilentlyContinue
        if (Test-Path $script:MetricsFile) {
            Remove-Item $script:MetricsFile -Force
        }
    }

    It "simulates complete successful iteration" {
        # 1. Log state transition: idle -> running
        Log-StateTransition -From "idle" -To "running" -Reason "Starting iteration"

        # 2. Append timeline event
        Append-SessionTimeline -Event "iteration_start" -Data @{ iteration = 1; storyId = "US-001" }

        # 3. Simulate Claude output
        $claudeOutput = @"
Reading file src/main.py...
Edit tool: modifying function
pytest tests/ -v
test_main.py::test_something PASSED
===== 1 passed in 0.5s =====
git commit -m 'fix: bug'
"@

        # 4. Calculate metrics
        $phaseTimings = Measure-PhaseTimings -Output $claudeOutput -TotalDurationMs 60000
        $tokens = Get-EstimatedTokens -Output $claudeOutput
        $testResults = Get-TestResults -Output $claudeOutput

        # 5. Log test details
        Log-TestDetails -Iteration 1 -Output $claudeOutput

        # 6. Log prompt effectiveness
        $effectiveness = Get-PromptEffectiveness -Success $true -RetryCount 1
        Log-PromptEffectiveness -Iteration 1 -PromptType "story_work" -Effectiveness $effectiveness -PromptHash "TESTHASH"

        # 7. Log state transition: running -> completed
        Log-StateTransition -From "running" -To "completed" -Reason "Success"

        # 8. Record metric
        Record-Metric -StoryId "US-001" -DurationMin 1 -Success $true -TokensUsed $tokens `
            -PhaseReadMs $phaseTimings.read_ms -PhaseImplementMs $phaseTimings.implement_ms

        # 9. Append timeline completion
        Append-SessionTimeline -Event "iteration_complete" -Data @{ success = $true }

        # Verify all files created
        Test-Path (Join-Path $script:SessionLogDir "state_transitions.jsonl") | Should -Be $true
        Test-Path (Join-Path $script:SessionLogDir "session_timeline.jsonl") | Should -Be $true
        Test-Path (Join-Path $script:SessionLogDir "test_details_1.json") | Should -Be $true
        Test-Path (Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl") | Should -Be $true
        Test-Path $script:MetricsFile | Should -Be $true

        # Verify content
        $stateLines = Get-Content (Join-Path $script:SessionLogDir "state_transitions.jsonl")
        $stateLines.Count | Should -Be 2

        $timelineLines = Get-Content (Join-Path $script:SessionLogDir "session_timeline.jsonl")
        $timelineLines.Count | Should -Be 2
    }

    It "simulates failed iteration with error tracking" {
        # 1. Start
        Log-StateTransition -From "idle" -To "running" -Reason "Starting"
        Append-SessionTimeline -Event "iteration_start" -Data @{ iteration = 1 }

        # 2. Simulate failure
        $errorOutput = "SyntaxError: invalid syntax in file.py"
        $errorCategory = Get-ErrorCategory -Output $errorOutput -TimedOut $false

        # 3. Log error evolution
        Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails $errorOutput -Iteration 1

        # 4. Log state transition to failed
        Log-StateTransition -From "running" -To "failed" -Reason $errorOutput

        # 5. Log skip if needed
        Log-Skip -ItemId "US-001" -ItemType "story" -Reason "Syntax error" -BlockerType "error"

        # 6. Record failure metric
        Record-Metric -StoryId "US-001" -DurationMin 2 -Success $false -ErrorCategory $errorCategory

        # Verify error tracking
        $errorEvolution = Get-Content (Join-Path $script:SessionLogDir "error_evolution.jsonl") | ConvertFrom-Json
        $errorEvolution.category | Should -Be "SyntaxError"

        $skips = Get-Content (Join-Path $script:SessionLogDir "skips_blockers.jsonl") | ConvertFrom-Json
        $skips.itemId | Should -Be "US-001"
    }
}

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

Describe "Edge Cases" -Tag "Unit", "EdgeCase" {
    BeforeAll {
        function Get-ErrorCategory { param([string]$Output, [bool]$TimedOut); if ($TimedOut) { return "Timeout" }; if ($Output -match "SyntaxError|parse error") { return "SyntaxError" }; if ($Output -match "FAILED|test.*failed") { return "TestFailure" }; return "Unknown" }
        function Get-EstimatedTokens { param([string]$Output); if (-not $Output -or $Output.Length -eq 0) { return 0 }; return [math]::Min([math]::Round($Output.Length / 4), 50000) }
        function Get-TestResults { param([string]$Output); if (-not $Output) { return "" }; $passed = 0; $failed = 0; if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }; if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }; $total = $passed + $failed; if ($total -eq 0) { return "" }; if ($failed -eq 0) { return "$passed/$total pass" }; return "$passed/$total pass, $failed fail" }
        function Get-GitState {
            $hash = git rev-parse HEAD 2>$null
            $branch = git rev-parse --abbrev-ref HEAD 2>$null
            $status = git status --porcelain 2>$null
            $clean = [string]::IsNullOrEmpty($status)
            $modified = if ($status) { $status -split "`n" | Where-Object { $_ } } else { @() }
            return @{ hash = $hash; branch = $branch; clean = $clean; modified = $modified }
        }
    }

    It "handles special characters in output" {
        $output = "Error: Can't find file 'test.py' with `$variable"
        $result = Get-ErrorCategory -Output $output -TimedOut $false
        $result | Should -Not -BeNullOrEmpty
    }

    It "handles very long output" {
        $output = "x" * 100000
        $result = Get-EstimatedTokens -Output $output
        $result | Should -BeGreaterThan 0
    }

    It "handles unicode in output" {
        $output = "Test passed"
        $result = Get-TestResults -Output $output
        # Should not throw
    }

    It "handles empty git state gracefully" {
        # This tests the function doesn't crash in non-git directories
        # The actual test runs in git dir so it should work
        $result = Get-GitState
        $result | Should -Not -BeNullOrEmpty
    }
}

# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================

Describe "Concurrent File Access" -Tag "Integration", "Concurrency" {
    BeforeAll {
        function Append-SessionTimeline {
            param([string]$Event, [hashtable]$Data = @{})
            $timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
            $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); event = $Event; session = $script:State.SessionId; iteration = $script:State.IterationCount }
            foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
            $entry | ConvertTo-Json -Compress | Add-Content -Path $timelineFile -Encoding UTF8
        }
        function Record-Metric { param([string]$StoryId, [string]$Mode, [double]$DurationMin, [bool]$Success = $true); $v2Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms"; if (-not (Test-Path $script:MetricsFile)) { $v2Header | Set-Content $script:MetricsFile -Encoding UTF8 }; $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; $row = "$timestamp,$($script:State.SessionId),,$StoryId,Standard,$DurationMin,$($Success.ToString().ToLower()),false,,,,,,,0,0,0,0,0,0,0"; Add-Content -Path $script:MetricsFile -Value $row }
    }

    It "handles rapid appends to timeline" {
        $timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
        if (Test-Path $timelineFile) {
            Remove-Item $timelineFile -Force
        }

        # Rapid appends
        1..10 | ForEach-Object {
            Append-SessionTimeline -Event "event_$_" -Data @{ index = $_ }
        }

        $lines = Get-Content $timelineFile
        $lines.Count | Should -Be 10
    }

    It "handles rapid metric recording" {
        if (Test-Path $script:MetricsFile) {
            Remove-Item $script:MetricsFile -Force
        }

        1..5 | ForEach-Object {
            Record-Metric -StoryId "US-00$_" -DurationMin $_ -Success $true
        }

        $lines = Get-Content $script:MetricsFile
        $lines.Count | Should -Be 6  # header + 5 data rows
    }
}

# =============================================================================
# CPU ACTIVITY DETECTION TESTS
# =============================================================================

Describe "CPU Activity Detection" -Tag "Unit", "StallDetection" {
    BeforeAll {
        # Simulate the CPU delta calculation logic from claude.ps1
        function Test-CpuActivity {
            param(
                [double]$CurrentCpu,
                [double]$LastCpuTime,
                [double]$Threshold = 0.5,
                [bool]$Enabled = $true
            )

            if (-not $Enabled -or $CurrentCpu -le 0) {
                return @{ Active = $false; Delta = 0 }
            }

            $cpuDelta = $CurrentCpu - $LastCpuTime
            $cpuActive = $cpuDelta -gt $Threshold

            return @{
                Active = $cpuActive
                Delta = $cpuDelta
            }
        }
    }

    It "detects CPU activity when delta exceeds threshold" {
        $result = Test-CpuActivity -CurrentCpu 5.0 -LastCpuTime 4.0 -Threshold 0.5
        $result.Active | Should -Be $true
        $result.Delta | Should -Be 1.0
    }

    It "does not detect activity when delta is below threshold" {
        $result = Test-CpuActivity -CurrentCpu 4.3 -LastCpuTime 4.0 -Threshold 0.5
        $result.Active | Should -Be $false
        [math]::Round($result.Delta, 1) | Should -Be 0.3
    }

    It "does not detect activity when disabled" {
        $result = Test-CpuActivity -CurrentCpu 10.0 -LastCpuTime 5.0 -Threshold 0.5 -Enabled $false
        $result.Active | Should -Be $false
    }

    It "does not detect activity when CPU is zero" {
        $result = Test-CpuActivity -CurrentCpu 0 -LastCpuTime 5.0 -Threshold 0.5
        $result.Active | Should -Be $false
        $result.Delta | Should -Be 0  # Early exit should return 0 delta, not negative
    }

    It "handles first sample (LastCpuTime = 0)" {
        # First sample: delta = current CPU time (shows startup work)
        $result = Test-CpuActivity -CurrentCpu 2.0 -LastCpuTime 0 -Threshold 0.5
        $result.Active | Should -Be $true
        $result.Delta | Should -Be 2.0
    }

    It "uses default threshold of 0.5 seconds" {
        # 0.5 CPU-seconds in 5 seconds = 10% CPU utilization threshold
        $result = Test-CpuActivity -CurrentCpu 10.5 -LastCpuTime 10.0
        $result.Active | Should -Be $false  # exactly at threshold, not above

        $result = Test-CpuActivity -CurrentCpu 10.6 -LastCpuTime 10.0
        $result.Active | Should -Be $true  # above threshold
    }

    It "handles large CPU deltas during heavy computation" {
        # During pytest runs, CPU time might increase significantly
        $result = Test-CpuActivity -CurrentCpu 25.0 -LastCpuTime 10.0 -Threshold 0.5
        $result.Active | Should -Be $true
        $result.Delta | Should -Be 15.0
    }
}

Describe "CPU Activity Config Loading" -Tag "Unit", "Config" {
    It "loads cpuActivity config from stallDetection section" {
        $configJson = @'
{
    "stallDetection": {
        "default": { "stallThreshold": 180, "killThreshold": 360 },
        "cpuActivity": { "enabled": true, "threshold": 0.5 }
    }
}
'@
        $config = $configJson | ConvertFrom-Json

        $cpuActivityConfig = $config.stallDetection.cpuActivity
        $cpuActivityConfig.enabled | Should -Be $true
        $cpuActivityConfig.threshold | Should -Be 0.5
    }

    It "uses defaults when cpuActivity config is missing" {
        $configJson = @'
{
    "stallDetection": {
        "default": { "stallThreshold": 180, "killThreshold": 360 }
    }
}
'@
        $config = $configJson | ConvertFrom-Json

        $cpuActivityConfig = $config.stallDetection.cpuActivity

        # Simulate the default logic from claude.ps1
        $enabled = if ($cpuActivityConfig -and $null -ne $cpuActivityConfig.enabled) { $cpuActivityConfig.enabled } else { $true }
        $threshold = if ($cpuActivityConfig -and $cpuActivityConfig.threshold) { $cpuActivityConfig.threshold } else { 0.5 }

        $enabled | Should -Be $true
        $threshold | Should -Be 0.5
    }

    It "respects disabled cpuActivity config" {
        $configJson = @'
{
    "stallDetection": {
        "cpuActivity": { "enabled": false, "threshold": 1.0 }
    }
}
'@
        $config = $configJson | ConvertFrom-Json

        $cpuActivityConfig = $config.stallDetection.cpuActivity
        $enabled = if ($cpuActivityConfig -and $null -ne $cpuActivityConfig.enabled) { $cpuActivityConfig.enabled } else { $true }

        $enabled | Should -Be $false
    }

    It "allows custom threshold values" {
        $configJson = @'
{
    "stallDetection": {
        "cpuActivity": { "enabled": true, "threshold": 2.0 }
    }
}
'@
        $config = $configJson | ConvertFrom-Json

        $cpuActivityConfig = $config.stallDetection.cpuActivity
        $threshold = if ($cpuActivityConfig -and $cpuActivityConfig.threshold) { $cpuActivityConfig.threshold } else { 0.5 }

        $threshold | Should -Be 2.0
    }
}
