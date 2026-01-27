#Requires -Modules Pester

<#
.SYNOPSIS
    End-to-end tests for Ralph Loop
.DESCRIPTION
    Tests complete workflows and scenarios:
    - Full sprint simulation
    - Queue mode progression
    - Error recovery
    - Metrics aggregation
#>

BeforeAll {
    $script:RalphDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $script:RalphScript = Join-Path $script:RalphDir "ralph.ps1"
    $script:TestDataDir = Join-Path $PSScriptRoot "e2e-testdata"

    # Create isolated test environment
    if (-not (Test-Path $script:TestDataDir)) {
        New-Item -ItemType Directory -Path $script:TestDataDir -Force | Out-Null
    }

    # Source all functions from ralph.ps1 and lib/*.ps1 into global scope
    . (Join-Path $PSScriptRoot 'test-helper.ps1')
    Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

    # Set up script-level variables
    $script:SessionId = "e2e-test-session"
    $script:SessionLogDir = Join-Path $script:TestDataDir "logs"
    $script:RalphDir = $script:TestDataDir
    $script:IterationCount = 0
    $script:CurrentRetryCount = 0
    $script:ConsecutiveFailures = 0
    $script:MetricsFile = Join-Path $script:TestDataDir "metrics.csv"
    $script:ProgressFile = Join-Path $script:TestDataDir "progress.txt"
    $script:PrdFile = Join-Path $script:TestDataDir "prd.json"
    $script:ProjectRoot = $script:TestDataDir

    # Also set global versions of important variables for functions
    $global:SessionLogDir = $script:SessionLogDir
    $global:MetricsFile = $script:MetricsFile
    $global:RalphDir = $script:TestDataDir
    $global:SessionId = $script:SessionId
    $global:IterationCount = 0
    $global:CurrentRetryCount = 0
    $global:ConsecutiveFailures = 0
    $global:ProjectRoot = $script:TestDataDir
    $global:ProgressFile = $script:ProgressFile
    $global:PrdFile = $script:PrdFile

    # Create log directory
    if (-not (Test-Path $script:SessionLogDir)) {
        New-Item -ItemType Directory -Path $script:SessionLogDir -Force | Out-Null
    }

    # Define all functions needed for simulation
    function Log-StateTransition {
        param([string]$From, [string]$To, [string]$Reason, [hashtable]$Context = @{})
        if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) { return }
        $stateFile = Join-Path $script:SessionLogDir "state_transitions.jsonl"
        $entry = @{ timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); from = $From; to = $To; reason = $Reason; iteration = $script:IterationCount; session = $script:SessionId }
        foreach ($key in $Context.Keys) { $entry[$key] = $Context[$key] }
        $entry | ConvertTo-Json -Compress | Add-Content -Path $stateFile -Encoding UTF8
    }
    function Append-SessionTimeline {
        param([string]$Event, [hashtable]$Data = @{})
        $timelineFile = Join-Path $script:SessionLogDir "session_timeline.jsonl"
        $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); event = $Event; session = $script:SessionId; iteration = $script:IterationCount }
        foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
        $entry | ConvertTo-Json -Compress | Add-Content -Path $timelineFile -Encoding UTF8
    }
    function Measure-PhaseTimings {
        param([string]$Output, [int]$TotalDurationMs)
        $timings = @{ read_ms = 0; analyze_ms = 0; implement_ms = 0; test_ms = 0; commit_ms = 0 }
        if (-not $Output -or $TotalDurationMs -le 0) { return $timings }
        $hasReadOps = $Output -match "Read tool|Reading file"
        $hasImplement = $Output -match "Edit tool|Write tool"
        $hasTests = $Output -match "pytest|test.*pass"
        $hasGit = $Output -match "git commit"
        $readWeight = if ($hasReadOps) { 2 } else { 0 }
        $implementWeight = if ($hasImplement) { 2 } else { 0 }
        $testWeight = if ($hasTests) { 3 } else { 0 }
        $gitWeight = if ($hasGit) { 1 } else { 0 }
        $totalWeight = [math]::Max(1, $readWeight + $implementWeight + $testWeight + $gitWeight + 1)
        $timings.read_ms = [int](($readWeight / $totalWeight) * $TotalDurationMs)
        $timings.implement_ms = [int](($implementWeight / $totalWeight) * $TotalDurationMs)
        return $timings
    }
    function Get-TestResults { param([string]$Output); if (-not $Output) { return "" }; $passed = 0; $failed = 0; if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }; if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }; $total = $passed + $failed; if ($total -eq 0) { return "" }; if ($failed -eq 0) { return "$passed/$total pass" }; return "$passed/$total pass, $failed fail" }
    function Get-ErrorCategory { param([string]$Output, [bool]$TimedOut); if ($TimedOut) { return "Timeout" }; if ($Output -match "SyntaxError") { return "SyntaxError" }; if ($Output -match "FAILED|test.*failed") { return "TestFailure" }; return "Unknown" }
    function Get-PromptEffectiveness { param([bool]$Success, [int]$RetryCount); if (-not $Success) { return 0.0 }; if ($RetryCount -le 1) { return 1.0 }; if ($RetryCount -le 3) { return 0.5 }; return 0.25 }
    function Write-JsonNoBom { param([string]$Path, [string]$Content); $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom) }
    function Log-TestDetails { param([int]$Iteration, [string]$Output); $file = Join-Path $script:SessionLogDir "test_details_$Iteration.json"; $passed = 0; $failed = 0; if ($Output -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }; if ($Output -match '(\d+)\s+failed') { $failed = [int]$Matches[1] }; $data = @{ iteration = $Iteration; summary = @{ passed = $passed; failed = $failed } }; Write-JsonNoBom -Path $file -Content ($data | ConvertTo-Json -Depth 5); return $data }
    function Log-PromptEffectiveness { param([int]$Iteration, [string]$PromptType, [double]$Effectiveness, [string]$PromptHash = ""); $file = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"; $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); iteration = $Iteration; promptType = $PromptType; effectiveness = $Effectiveness; promptHash = $PromptHash }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
    function Log-ErrorEvolution { param([string]$ErrorCategory, [string]$ErrorDetails = "", [int]$Iteration); $file = Join-Path $script:SessionLogDir "error_evolution.jsonl"; $entry = @{ timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); category = $ErrorCategory; details = $ErrorDetails; iteration = $Iteration }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
    function Log-Skip { param([string]$ItemId, [string]$ItemType, [string]$Reason, [string]$BlockerType = "unknown"); $file = Join-Path $script:SessionLogDir "skips_blockers.jsonl"; $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); itemId = $ItemId; itemType = $ItemType; reason = $Reason; blockerType = $BlockerType }; $entry | ConvertTo-Json -Compress | Add-Content -Path $file -Encoding UTF8 }
    function Record-Metric { param([string]$StoryId, [string]$Mode, [double]$DurationMin, [bool]$Success = $true, [string]$FocusArea, [int]$TokensUsed = 0, [string]$ErrorCategory = "", [int]$PhaseReadMs = 0, [int]$PhaseImplementMs = 0); $v2Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms"; if (-not (Test-Path $script:MetricsFile)) { $v2Header | Set-Content $script:MetricsFile -Encoding UTF8 }; $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; $row = "$timestamp,$script:SessionId,,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),false,$FocusArea,$TokensUsed,$ErrorCategory,$(Get-Date).Hour,,,0,0,$PhaseReadMs,0,$PhaseImplementMs,0,0"; Add-Content -Path $script:MetricsFile -Value $row }

    # Helper function to simulate a complete iteration
    function Invoke-SimulatedIteration {
        param(
            [string]$StoryId,
            [string]$FocusArea,
            [bool]$Success = $true,
            [int]$DurationMin = 5,
            [int]$TokensUsed = 10000
        )

        $script:IterationCount++

        # Simulate Claude output
        $output = if ($Success) {
            @"
Reading file src/main.py...
Edit tool: modifying function
pytest tests/ -v
test_main.py::test_something PASSED
===== 1 passed in 0.5s =====
git commit -m 'fix: $StoryId'
"@
        }
        else {
            @"
Reading file src/main.py...
Edit tool: modifying function
pytest tests/ -v
test_main.py::test_something FAILED
SyntaxError: invalid syntax
"@
        }

        # Log state transition
        Log-StateTransition -From "idle" -To "running" -Reason "Starting $StoryId"

        # Append timeline
        Append-SessionTimeline -Event "iteration_start" -Data @{
            iteration = $script:IterationCount
            storyId = $StoryId
            focusArea = $FocusArea
        }

        # Calculate metrics
        $phaseTimings = Measure-PhaseTimings -Output $output -TotalDurationMs ($DurationMin * 60000)
        $testResults = Get-TestResults -Output $output

        if ($Success) {
            # Log test details
            $null = Log-TestDetails -Iteration $script:IterationCount -Output $output

            # Log effectiveness
            $effectiveness = Get-PromptEffectiveness -Success $true -RetryCount $script:CurrentRetryCount
            Log-PromptEffectiveness -Iteration $script:IterationCount -PromptType "story_work" -Effectiveness $effectiveness -PromptHash "SIMHASH"

            # State transition to completed
            Log-StateTransition -From "running" -To "completed" -Reason "Success"

            # Record success metric
            Record-Metric -StoryId $StoryId -Mode "Standard" -DurationMin $DurationMin -Success $true `
                -FocusArea $FocusArea -TokensUsed $TokensUsed `
                -PhaseReadMs $phaseTimings.read_ms -PhaseImplementMs $phaseTimings.implement_ms

            $script:ConsecutiveFailures = 0
        }
        else {
            $errorCategory = Get-ErrorCategory -Output $output -TimedOut $false

            # Log error evolution
            Log-ErrorEvolution -ErrorCategory $errorCategory -ErrorDetails "Simulated failure" -Iteration $script:IterationCount

            # State transition to failed
            Log-StateTransition -From "running" -To "failed" -Reason $errorCategory

            # Record failure metric
            Record-Metric -StoryId $StoryId -Mode "Standard" -DurationMin $DurationMin -Success $false `
                -FocusArea $FocusArea -TokensUsed $TokensUsed -ErrorCategory $errorCategory

            $script:ConsecutiveFailures++
        }

        # Complete timeline
        Append-SessionTimeline -Event "iteration_complete" -Data @{
            iteration = $script:IterationCount
            storyId = $StoryId
            success = $Success
        }

        return $Success
    }

    # Helper to create PRD
    function New-SimulatedPrd {
        param(
            [string]$FocusArea,
            [int]$StoryCount = 5
        )

        $stories = 1..$StoryCount | ForEach-Object {
            @{
                id = "US-$('{0:D3}' -f $_)"
                title = "Story $_ for $FocusArea"
                passes = $false
                acceptanceCriteria = @("Criteria 1", "Criteria 2")
            }
        }

        $prd = @{
            sprintNumber = 1
            branchName = "ralph/sprint-1"
            focusArea = $FocusArea
            userStories = $stories
        }

        $prd | ConvertTo-Json -Depth 5 | Set-Content $script:PrdFile
        return $prd
    }
}

AfterAll {
    if (Test-Path $script:TestDataDir) {
        Remove-Item -Path $script:TestDataDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# COMPLETE SPRINT SIMULATION
# =============================================================================

Describe "Complete Sprint Simulation" -Tag "E2E", "Sprint" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "simulates a successful sprint with 5 stories" {
        # Create PRD
        $prd = New-SimulatedPrd -FocusArea "testing" -StoryCount 5

        # Run all stories
        foreach ($story in $prd.userStories) {
            $result = Invoke-SimulatedIteration -StoryId $story.id -FocusArea "testing" -Success $true
            $result | Should -Be $true
        }

        # Verify all iterations logged
        $metrics = Import-Csv $script:MetricsFile
        $metrics.Count | Should -Be 5

        # Verify all succeeded
        @($metrics | Where-Object { $_.success -eq 'true' }).Count | Should -Be 5

        # Verify timeline has all events
        $timeline = Get-Content (Join-Path $script:SessionLogDir "session_timeline.jsonl")
        $timeline.Count | Should -Be 10  # 5 starts + 5 completes
    }

    It "simulates a sprint with mixed success/failure" {
        $prd = New-SimulatedPrd -FocusArea "quality" -StoryCount 4

        # Story 1: Success
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "quality" -Success $true
        # Story 2: Failure
        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "quality" -Success $false
        # Story 3: Success
        Invoke-SimulatedIteration -StoryId "US-003" -FocusArea "quality" -Success $true
        # Story 4: Success
        Invoke-SimulatedIteration -StoryId "US-004" -FocusArea "quality" -Success $true

        $metrics = Import-Csv $script:MetricsFile
        @($metrics | Where-Object { $_.success -eq 'true' }).Count | Should -Be 3
        @($metrics | Where-Object { $_.success -eq 'false' }).Count | Should -Be 1

        # Verify error evolution logged
        $errors = Get-Content (Join-Path $script:SessionLogDir "error_evolution.jsonl") -ErrorAction SilentlyContinue
        $errors.Count | Should -Be 1
    }

    It "tracks consecutive failures correctly" {
        # Three consecutive failures
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $false
        $script:ConsecutiveFailures | Should -Be 1

        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "testing" -Success $false
        $script:ConsecutiveFailures | Should -Be 2

        Invoke-SimulatedIteration -StoryId "US-003" -FocusArea "testing" -Success $false
        $script:ConsecutiveFailures | Should -Be 3

        # Success resets counter
        Invoke-SimulatedIteration -StoryId "US-004" -FocusArea "testing" -Success $true
        $script:ConsecutiveFailures | Should -Be 0
    }
}

# =============================================================================
# QUEUE MODE SIMULATION
# =============================================================================

Describe "Queue Mode Progression" -Tag "E2E", "Queue" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }

        $script:queuePath = Join-Path $script:TestDataDir "queue.json"
    }

    It "simulates queue progression through multiple focus areas" {
        # Create queue
        $queue = @{
            focusAreas = @(
                @{ id = "testing"; completed = $false }
                @{ id = "quality"; completed = $false }
                @{ id = "speed"; completed = $false }
            )
            interviewContext = "E2E test"
            session = @{ id = $script:SessionId }
        }
        $queue | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath

        # Process each focus area
        foreach ($area in $queue.focusAreas) {
            # Create PRD for this area
            New-SimulatedPrd -FocusArea $area.id -StoryCount 2

            # Run stories
            Invoke-SimulatedIteration -StoryId "US-001" -FocusArea $area.id -Success $true
            Invoke-SimulatedIteration -StoryId "US-002" -FocusArea $area.id -Success $true

            # Mark area complete
            $q = Get-Content $script:queuePath | ConvertFrom-Json
            ($q.focusAreas | Where-Object { $_.id -eq $area.id }).completed = $true
            $q | ConvertTo-Json -Depth 5 | Set-Content $script:queuePath
        }

        # Verify all areas completed
        $finalQueue = Get-Content $script:queuePath | ConvertFrom-Json
        @($finalQueue.focusAreas | Where-Object { $_.completed }).Count | Should -Be 3

        # Verify metrics for all areas
        $metrics = Import-Csv $script:MetricsFile
        $metrics.Count | Should -Be 6
        ($metrics | Group-Object focus_area).Count | Should -Be 3
    }
}

# =============================================================================
# ERROR RECOVERY SIMULATION
# =============================================================================

Describe "Error Recovery Scenarios" -Tag "E2E", "ErrorRecovery" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "recovers from transient failures with retry" {
        # Fail twice, then succeed
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $false
        $script:CurrentRetryCount++

        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $false
        $script:CurrentRetryCount++

        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true

        $metrics = Import-Csv $script:MetricsFile
        $metrics.Count | Should -Be 3
        @($metrics | Where-Object { $_.success -eq 'true' }).Count | Should -Be 1
    }

    It "logs skip when story is abandoned" {
        # Fail multiple times
        1..3 | ForEach-Object {
            Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $false
        }

        # Log skip
        Log-Skip -ItemId "US-001" -ItemType "story" -Reason "Exceeded retry limit" -BlockerType "error"

        # Verify skip logged
        $skips = Get-Content (Join-Path $script:SessionLogDir "skips_blockers.jsonl") | ConvertFrom-Json
        $skips.itemId | Should -Be "US-001"
        $skips.blockerType | Should -Be "error"
    }
}

# =============================================================================
# METRICS AGGREGATION TESTS
# =============================================================================

Describe "Metrics Aggregation" -Tag "E2E", "Metrics" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "aggregates metrics by focus area" {
        # Run iterations for different areas
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true -TokensUsed 10000
        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "testing" -Success $true -TokensUsed 15000
        Invoke-SimulatedIteration -StoryId "US-003" -FocusArea "quality" -Success $true -TokensUsed 20000

        $metrics = Import-Csv $script:MetricsFile

        # Group by focus area
        $byArea = $metrics | Group-Object focus_area

        # Testing area
        $testingMetrics = ($byArea | Where-Object { $_.Name -eq "testing" }).Group
        $testingMetrics.Count | Should -Be 2
        $testingTokens = ($testingMetrics | ForEach-Object { [int]$_.tokens_used } | Measure-Object -Sum).Sum
        $testingTokens | Should -Be 25000

        # Quality area
        $qualityMetrics = ($byArea | Where-Object { $_.Name -eq "quality" }).Group
        $qualityMetrics.Count | Should -Be 1
    }

    It "calculates correct success rate per area" {
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true
        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "testing" -Success $true
        Invoke-SimulatedIteration -StoryId "US-003" -FocusArea "testing" -Success $false
        Invoke-SimulatedIteration -StoryId "US-004" -FocusArea "quality" -Success $true
        Invoke-SimulatedIteration -StoryId "US-005" -FocusArea "quality" -Success $true

        $metrics = Import-Csv $script:MetricsFile
        $byArea = $metrics | Group-Object focus_area

        # Testing: 2/3 = 66.67%
        $testing = ($byArea | Where-Object { $_.Name -eq "testing" }).Group
        $testingSuccess = @($testing | Where-Object { $_.success -eq 'true' }).Count
        $testingRate = [math]::Round(($testingSuccess / $testing.Count) * 100)
        $testingRate | Should -Be 67  # Rounded

        # Quality: 2/2 = 100%
        $quality = ($byArea | Where-Object { $_.Name -eq "quality" }).Group
        $qualitySuccess = @($quality | Where-Object { $_.success -eq 'true' }).Count
        $qualityRate = [math]::Round(($qualitySuccess / $quality.Count) * 100)
        $qualityRate | Should -Be 100
    }
}

# =============================================================================
# STATE MACHINE VERIFICATION
# =============================================================================

Describe "State Machine Verification" -Tag "E2E", "StateMachine" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "logs correct state transitions for success" {
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true

        $states = Get-Content (Join-Path $script:SessionLogDir "state_transitions.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }

        $states.Count | Should -Be 2
        $states[0].from | Should -Be "idle"
        $states[0].to | Should -Be "running"
        $states[1].from | Should -Be "running"
        $states[1].to | Should -Be "completed"
    }

    It "logs correct state transitions for failure" {
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $false

        $states = Get-Content (Join-Path $script:SessionLogDir "state_transitions.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }

        $states.Count | Should -Be 2
        $states[0].to | Should -Be "running"
        $states[1].to | Should -Be "failed"
    }
}

# =============================================================================
# PROMPT EFFECTIVENESS TRACKING
# =============================================================================

Describe "Prompt Effectiveness Tracking" -Tag "E2E", "Effectiveness" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "calculates effectiveness scores correctly" {
        # First try success
        $script:CurrentRetryCount = 1
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true

        # Success after retries
        $script:CurrentRetryCount = 3
        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "testing" -Success $true

        $effectiveness = Get-Content (Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }

        $effectiveness[0].effectiveness | Should -Be 1.0
        $effectiveness[1].effectiveness | Should -Be 0.5
    }
}

# =============================================================================
# TIMELINE INTEGRITY
# =============================================================================

Describe "Timeline Integrity" -Tag "E2E", "Timeline" {
    BeforeEach {
        # Reset state
        $script:IterationCount = 0
        $script:CurrentRetryCount = 0
        $script:ConsecutiveFailures = 0

        # Clean files
        Get-ChildItem $script:SessionLogDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path $script:MetricsFile) { Remove-Item $script:MetricsFile -Force }
        if (Test-Path $script:PrdFile) { Remove-Item $script:PrdFile -Force }
    }

    It "maintains chronological order" {
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true
        Start-Sleep -Milliseconds 100
        Invoke-SimulatedIteration -StoryId "US-002" -FocusArea "testing" -Success $true

        $timeline = Get-Content (Join-Path $script:SessionLogDir "session_timeline.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }

        # Verify timestamps are increasing
        for ($i = 1; $i -lt $timeline.Count; $i++) {
            $prev = [datetime]$timeline[$i - 1].ts
            $curr = [datetime]$timeline[$i].ts
            $curr | Should -BeGreaterOrEqual $prev
        }
    }

    It "includes all expected events" {
        Invoke-SimulatedIteration -StoryId "US-001" -FocusArea "testing" -Success $true

        $timeline = Get-Content (Join-Path $script:SessionLogDir "session_timeline.jsonl") | ForEach-Object { $_ | ConvertFrom-Json }
        $eventTypes = $timeline | ForEach-Object { $_.event } | Select-Object -Unique

        $eventTypes | Should -Contain "iteration_start"
        $eventTypes | Should -Contain "iteration_complete"
    }
}
