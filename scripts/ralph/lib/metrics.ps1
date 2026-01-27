# scripts/ralph/lib/metrics.ps1
# Metrics, logging, instrumentation: CSV tracking, git state, health, tokens

# ============================================================================
# COMPREHENSIVE LOGGING FUNCTIONS (Phase 1)
# ============================================================================

function Get-GitState {
    <#
    .SYNOPSIS
        Capture current git state (hash, branch, clean status)
    .RETURNS
        Hashtable with git state info
    #>
    $state = @{
        hash = ""
        branch = ""
        clean = $true
        modifiedFiles = @()
    }

    try {
        $state.hash = (git rev-parse HEAD 2>$null)
        $state.branch = (git rev-parse --abbrev-ref HEAD 2>$null)
        $status = git status --porcelain 2>$null
        if ($status) {
            $state.clean = $false
            $state.modifiedFiles = @($status | ForEach-Object { $_.Substring(3) })
        }
    }
    catch {}

    return $state
}

function Get-FileOperations {
    <#
    .SYNOPSIS
        Get file operations between two git states
    .PARAMETER BeforeHash
        Git commit hash before the operation
    .RETURNS
        Hashtable with files created, modified, deleted
    #>
    param([string]$BeforeHash)

    $ops = @{
        filesCreated = @()
        filesModified = @()
        filesDeleted = @()
        totalFilesChanged = 0
    }

    try {
        # Get diff stats
        $diffOutput = git diff --name-status $BeforeHash HEAD 2>$null
        if ($diffOutput) {
            foreach ($line in $diffOutput) {
                if ($line -match "^([AMDRC])\s+(.+)$") {
                    $status = $Matches[1]
                    $file = $Matches[2]
                    switch ($status) {
                        "A" { $ops.filesCreated += @{ path = $file; size = (Get-Item $file -ErrorAction SilentlyContinue).Length } }
                        "M" { $ops.filesModified += @{ path = $file } }
                        "D" { $ops.filesDeleted += $file }
                    }
                }
            }
        }

        # Also check unstaged changes
        $statusOutput = git status --porcelain 2>$null
        if ($statusOutput) {
            foreach ($line in $statusOutput) {
                if ($line -match "^\?\?\s+(.+)$") {
                    $file = $Matches[1]
                    $ops.filesCreated += @{ path = $file; size = (Get-Item $file -ErrorAction SilentlyContinue).Length }
                }
            }
        }

        $ops.totalFilesChanged = $ops.filesCreated.Count + $ops.filesModified.Count + $ops.filesDeleted.Count
    }
    catch {}

    return $ops
}

function Get-GitCommits {
    <#
    .SYNOPSIS
        Get commits made since a specific hash
    .PARAMETER SinceHash
        Git commit hash to start from
    .RETURNS
        Array of commit objects
    #>
    param([string]$SinceHash)

    $commits = @()

    try {
        $logOutput = git log --format="%H|%s|%ai|%an" "$SinceHash..HEAD" 2>$null
        if ($logOutput) {
            foreach ($line in $logOutput) {
                $parts = $line -split '\|'
                if ($parts.Count -ge 4) {
                    # Get diff stats for this commit
                    $stats = git diff --shortstat "$($parts[0])^" $parts[0] 2>$null
                    $insertions = 0
                    $deletions = 0
                    $filesChanged = 0
                    if ($stats -match "(\d+) files? changed") { $filesChanged = [int]$Matches[1] }
                    if ($stats -match "(\d+) insertions?") { $insertions = [int]$Matches[1] }
                    if ($stats -match "(\d+) deletions?") { $deletions = [int]$Matches[1] }

                    $commits += @{
                        hash = $parts[0]
                        message = $parts[1]
                        timestamp = $parts[2]
                        author = $parts[3]
                        filesChanged = $filesChanged
                        insertions = $insertions
                        deletions = $deletions
                    }
                }
            }
        }
    }
    catch {}

    return $commits
}

# Helper for JSONL append operations

# Helper for JSONL append operations
function Append-Jsonl {
    <#
    .SYNOPSIS
        Append an entry to a JSONL file with standard fields (timestamp, session)
    .PARAMETER File
        Path to the JSONL file
    .PARAMETER Data
        Hashtable of data to write
    .PARAMETER SkipSessionCheck
        If set, skip session directory check
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$File,
        [Parameter(Mandatory=$true)]
        [hashtable]$Data,
        [switch]$SkipSessionCheck
    )

    if (-not $SkipSessionCheck -and (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir))) {
        return
    }

    $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); session = $script:SessionId }
    foreach ($key in $Data.Keys) { $entry[$key] = $Data[$key] }
    $entry | ConvertTo-Json -Compress | Add-Content -Path $File -Encoding UTF8
}

function Log-ClaudeInvocation {
    <#
    .SYNOPSIS
        Log detailed Claude CLI invocation information
    .PARAMETER Iteration
        Iteration number
    .PARAMETER ClaudePath
        Path to Claude executable
    .PARAMETER Arguments
        Array of CLI arguments
    .PARAMETER PromptFile
        Path to prompt file
    .PARAMETER PromptType
        Type of prompt (prd_generation, story_work)
    .PARAMETER ProcessId
        Process ID of Claude process
    .PARAMETER StartTime
        When execution started
    .PARAMETER EndTime
        When execution ended
    .PARAMETER ExitCode
        Process exit code
    .PARAMETER TimedOut
        Whether the process timed out
    #>
    param(
        [int]$Iteration,
        [string]$ClaudePath,
        [array]$Arguments,
        [string]$PromptFile,
        [string]$PromptType,
        [int]$ProcessId,
        [datetime]$StartTime,
        [datetime]$EndTime,
        [int]$ExitCode,
        [bool]$TimedOut
    )

    # Input validation
    if (-not $script:SessionLogDir) {
        Write-Warning "Log-ClaudeInvocation: SessionLogDir not set, skipping"
        return
    }

    $invocationFile = Join-Path $script:SessionLogDir "claude_invocation_$Iteration.json"

    # Read prompt metadata
    $promptContent = ""
    $promptLines = 0
    $promptChars = 0
    if (Test-Path $PromptFile) {
        $promptContent = Get-Content $PromptFile -Raw -ErrorAction SilentlyContinue
        if ($promptContent) {
            $promptLines = ($promptContent -split "`n").Count
            $promptChars = $promptContent.Length
        }
    }

    # Get hash of prompt for tracking
    $promptHash = ""
    if ($promptContent) {
        $md5 = [System.Security.Cryptography.MD5]::Create()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($promptContent)
        $hashBytes = $md5.ComputeHash($bytes)
        $promptHash = [BitConverter]::ToString($hashBytes) -replace '-', ''
    }

    $invocation = @{
        iteration = $Iteration
        timestamp = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
        command = @{
            executable = $ClaudePath
            resolvedPath = (Resolve-Path $ClaudePath -ErrorAction SilentlyContinue).Path
            arguments = $Arguments
            workingDirectory = $script:ProjectRoot
        }
        prompt = @{
            file = (Split-Path $PromptFile -Leaf)
            type = $PromptType
            lineCount = $promptLines
            charCount = $promptChars
            hash = $promptHash.Substring(0, [Math]::Min(16, $promptHash.Length))
        }
        execution = @{
            processId = $ProcessId
            startedAt = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            endedAt = $EndTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            durationMs = [int](($EndTime - $StartTime).TotalMilliseconds)
            exitCode = $ExitCode
            timedOut = $TimedOut
        }
    }

    Write-JsonNoBom -Path $invocationFile -Content ($invocation | ConvertTo-Json -Depth 5)
}

function Log-IterationManifest {
    <#
    .SYNOPSIS
        Create structured iteration manifest
    .PARAMETER Iteration
        Iteration number
    .PARAMETER StoryId
        Story ID being worked on
    .PARAMETER FocusArea
        Focus area
    .PARAMETER Status
        Iteration status (completed, failed, timeout)
    .PARAMETER StartTime
        When iteration started
    .PARAMETER EndTime
        When iteration ended
    .PARAMETER PromptFile
        Path to prompt file
    .PARAMETER GitBefore
        Git state before iteration
    .PARAMETER GitAfter
        Git state after iteration
    .PARAMETER FileOps
        File operations during iteration
    .PARAMETER Commits
        Git commits made during iteration
    .PARAMETER TestResults
        Test results string
    .PARAMETER TokensEstimated
        Estimated token count
    .PARAMETER RetryCount
        Retry count
    #>
    param(
        [int]$Iteration,
        [string]$StoryId,
        [string]$FocusArea,
        [string]$Status,
        [datetime]$StartTime,
        [datetime]$EndTime,
        [string]$PromptFile,
        [hashtable]$GitBefore,
        [hashtable]$GitAfter,
        [hashtable]$FileOps,
        [array]$Commits,
        [string]$TestResults,
        [int]$TokensEstimated,
        [int]$RetryCount
    )

    # Input validation
    if (-not $script:SessionLogDir) {
        Write-Warning "Log-IterationManifest: SessionLogDir not set, skipping"
        return
    }
    if ($Iteration -lt 1) {
        Write-Warning "Log-IterationManifest: Invalid iteration number: $Iteration"
        $Iteration = 1
    }

    $manifestFile = Join-Path $script:SessionLogDir "iteration_${Iteration}_manifest.json"

    # Parse test results
    $testsPassed = 0
    $testsFailed = 0
    $testsSkipped = 0
    if ($TestResults -match "(\d+)\s*passed") { $testsPassed = [int]$Matches[1] }
    if ($TestResults -match "(\d+)\s*failed") { $testsFailed = [int]$Matches[1] }
    if ($TestResults -match "(\d+)\s*skipped") { $testsSkipped = [int]$Matches[1] }

    # Get PRD info
    $sprint = 0
    $prdBranch = ""
    if (Test-Path $script:PrdFile) {
        try {
            $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
            $sprint = $prd.sprintNumber
            $prdBranch = $prd.branchName
        }
        catch {}
    }

    # Use git branch as source of truth, fall back to PRD branch
    $actualBranch = if ($GitAfter -and $GitAfter.branch) { $GitAfter.branch } else { $prdBranch }

    # Calculate lines from commits
    $linesAdded = 0
    $linesDeleted = 0
    foreach ($commit in $Commits) {
        $linesAdded += $commit.insertions
        $linesDeleted += $commit.deletions
    }

    # Determine if this is focus area work vs story work
    # StoryId should be null/empty for focus area work (PRD generation)
    $isStoryWork = $StoryId -and $StoryId -match "^US-\d+"
    $effectiveStoryId = if ($isStoryWork) { $StoryId } else { $null }

    $manifest = @{
        iteration = $Iteration
        storyId = $effectiveStoryId
        focusArea = $FocusArea
        sprint = $sprint
        branch = $actualBranch
        status = $Status
        timestamps = @{
            started = $StartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            completed = $EndTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
            durationSec = [int](($EndTime - $StartTime).TotalSeconds)
        }
        prompt = @{
            file = (Split-Path $PromptFile -Leaf)
        }
        output = @{
            stdout = "claude_out_$Iteration.log"
            stderr = "claude_err_$Iteration.log"
        }
        git = @{
            beforeCommit = if ($GitBefore) { $GitBefore.hash } else { "" }
            afterCommit = if ($GitAfter) { $GitAfter.hash } else { "" }
            branch = $actualBranch
            filesCreated = if ($FileOps -and $FileOps.filesCreated) { $FileOps.filesCreated.Count } else { 0 }
            filesModified = if ($FileOps -and $FileOps.filesModified) { $FileOps.filesModified.Count } else { 0 }
            filesDeleted = if ($FileOps -and $FileOps.filesDeleted) { $FileOps.filesDeleted.Count } else { 0 }
            linesAdded = $linesAdded
            linesDeleted = $linesDeleted
            commits = $Commits
        }
        tests = @{
            passed = $testsPassed
            failed = $testsFailed
            skipped = $testsSkipped
            raw = $TestResults
        }
        metrics = @{
            tokensEstimated = $TokensEstimated
            retryCount = $RetryCount
        }
    }

    Write-JsonNoBom -Path $manifestFile -Content ($manifest | ConvertTo-Json -Depth 10)
}

function Log-FileOperations {
    <#
    .SYNOPSIS
        Log file operations to a dedicated file
    .PARAMETER Iteration
        Iteration number
    .PARAMETER FileOps
        File operations hashtable
    #>
    param(
        [int]$Iteration,
        [hashtable]$FileOps
    )

    $fileOpsFile = Join-Path $script:SessionLogDir "file_operations_$Iteration.json"

    Write-JsonNoBom -Path $fileOpsFile -Content ($FileOps | ConvertTo-Json -Depth 5)
}

function Log-GitOperations {
    <#
    .SYNOPSIS
        Log git operations to a dedicated file
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Branch
        Current branch
    .PARAMETER Commits
        Array of commits
    .PARAMETER BeforeState
        Git state before
    .PARAMETER AfterState
        Git state after
    #>
    param(
        [int]$Iteration,
        [string]$Branch,
        [array]$Commits,
        [hashtable]$BeforeState,
        [hashtable]$AfterState
    )

    $gitOpsFile = Join-Path $script:SessionLogDir "git_operations_$Iteration.json"

    $gitOps = @{
        iteration = $Iteration
        branch = $Branch
        commits = $Commits
        beforeState = @{
            hash = $BeforeState.hash
            clean = $BeforeState.clean
        }
        afterState = @{
            hash = $AfterState.hash
            clean = $AfterState.clean
        }
        totalCommits = $Commits.Count
    }

    Write-JsonNoBom -Path $gitOpsFile -Content ($gitOps | ConvertTo-Json -Depth 5)
}

function Log-StoryVerification {
    <#
    .SYNOPSIS
        Log story verification with evidence-based acceptance criteria checking (Story 2.4)
    .DESCRIPTION
        Searches Claude output and git diff for per-criterion evidence instead of
        rubber-stamping all criteria as passed. Uses Search-CriterionEvidence for
        keyword matching, completion signal detection, and diff analysis.
    .PARAMETER StoryId
        Story ID
    .PARAMETER Story
        Full story object from PRD
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Passed
        Whether story passed (exit code 0)
    .PARAMETER ClaudeOutput
        Full Claude stdout+stderr for evidence searching
    .PARAMETER DiffOutput
        Git diff output for evidence searching
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [int]$Iteration,
        [bool]$Passed,
        [string]$ClaudeOutput = "",
        [string]$DiffOutput = ""
    )

    $verificationFile = Join-Path $script:SessionLogDir "story_${StoryId}_verification.json"

    # Build acceptance criteria verification with evidence search
    $criteriaVerification = @()
    $criteriaMetCount = 0
    $criteriaTotalCount = 0

    if ($Story -and $Story.acceptanceCriteria) {
        foreach ($criterion in $Story.acceptanceCriteria) {
            $criteriaTotalCount++

            if ($Passed -and ($ClaudeOutput -or $DiffOutput)) {
                # Story 2.4: Evidence-based verification
                $evidenceResult = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $ClaudeOutput -DiffOutput $DiffOutput

                $verified = $evidenceResult.found
                $evidence = $evidenceResult.evidence
                $confidence = $evidenceResult.confidence

                if ($verified) { $criteriaMetCount++ }
            }
            elseif ($Passed) {
                # Fallback: passed but no output to search
                $verified = $true
                $evidence = "Story passed (exit code 0) but no output available for evidence search"
                $confidence = 0.5
                $criteriaMetCount++
            }
            else {
                $verified = $false
                $evidence = "Story not yet complete"
                $confidence = 0.0
            }

            $criteriaVerification += @{
                criterion = $criterion
                verified = $verified
                evidence = $evidence
                confidence = $confidence
            }
        }
    }

    $verification = @{
        storyId = $StoryId
        title = if ($Story) { $Story.title } else { "" }
        verifiedAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        iteration = $Iteration
        acceptanceCriteria = $criteriaVerification
        criteriaMet = $criteriaMetCount
        criteriaTotal = $criteriaTotalCount
        overallVerified = $Passed
        evidenceBased = ($ClaudeOutput -ne "" -or $DiffOutput -ne "")
    }

    Write-JsonNoBom -Path $verificationFile -Content ($verification | ConvertTo-Json -Depth 5)

    if ($Passed -and $criteriaTotalCount -gt 0) {
        $pct = [math]::Round(($criteriaMetCount / $criteriaTotalCount) * 100, 0)
        $color = if ($pct -ge 80) { "Green" } elseif ($pct -ge 50) { "Yellow" } else { "Red" }
        Write-Host "  Evidence: $criteriaMetCount/$criteriaTotalCount criteria verified ($pct%)" -ForegroundColor $color
    }
}

function Append-SessionTimeline {
    <#
    .SYNOPSIS
        Append an event to the session timeline
    #>
    param([string]$Event, [hashtable]$Data = @{})
    $Data.event = $Event
    Append-Jsonl -File (Join-Path $script:SessionLogDir "session_timeline.jsonl") -Data $Data
}

# ============================================================================
# QUALITY GATES & INTELLIGENCE (Phase 1)
# ============================================================================

function Get-SprintTokenBudget {
    <#
    .SYNOPSIS
        Calculate token budget status for current sprint (Story 1.8)
    .RETURNS
        Hashtable with budget status, or $null if budget tracking disabled
    #>

    $config = Get-RalphConfig

    if (-not $config.budget -or $config.budget.enabled -ne $true) {
        return $null
    }

    $maxTokens = if ($config.budget.maxTokensPerSprint) { $config.budget.maxTokensPerSprint } else { 500000 }
    $warnPercent = if ($config.budget.warnAtPercent) { $config.budget.warnAtPercent } else { 80 }

    # Sum tokens from current session metrics
    $totalUsed = 0
    if (Test-Path $script:MetricsFile) {
        try {
            $metrics = @(Import-Csv $script:MetricsFile -ErrorAction SilentlyContinue)
            $sessionMetrics = @($metrics | Where-Object { $_.session -eq $script:SessionId })
            $sumResult = ($sessionMetrics | Measure-Object -Property tokens_used -Sum).Sum
            if ($sumResult) { $totalUsed = [int]$sumResult }
        }
        catch {
            $totalUsed = 0
        }
    }

    $percentUsed = if ($maxTokens -gt 0) { [math]::Round($totalUsed / $maxTokens * 100, 1) } else { 0 }
    $remaining = [math]::Max(0, $maxTokens - $totalUsed)
    $budgetExceeded = $totalUsed -ge $maxTokens
    $budgetWarning = $percentUsed -ge $warnPercent

    $status = @{
        totalUsed = $totalUsed
        maxTokens = $maxTokens
        remaining = $remaining
        percentUsed = $percentUsed
        exceeded = $budgetExceeded
        warning = $budgetWarning
    }

    if ($budgetWarning) {
        $warnColor = if ($budgetExceeded) { "Red" } else { "Yellow" }
        $warnMsg = if ($budgetExceeded) { "TOKEN BUDGET EXCEEDED" } else { "Token budget warning" }
        Write-Host "  ${warnMsg}: $totalUsed / $maxTokens ($percentUsed%)" -ForegroundColor $warnColor
    }

    return $status
}

# ============================================================================
# CORE QUALITY GATE (Phase 2)
# ============================================================================

function Measure-CodebaseHealth {
    <#
    .SYNOPSIS
        Track codebase health metrics (Story 3.2)
    .DESCRIPTION
        Collects test count, pass rate, file count, complexity indicators,
        and tech debt markers. Saves to health_metrics.json.
    .RETURNS
        Hashtable with health metrics
    #>

    $config = Get-RalphConfig
    $healthEnabled = -not $config.health -or $config.health.enabled -ne $false

    if (-not $healthEnabled) { return $null }

    $health = @{
        measuredAt = (Get-Date).ToString("o")
        tests = @{ total = 0; passed = 0; failed = 0; passRate = 0.0 }
        codebase = @{ pyFiles = 0; ps1Files = 0; totalLines = 0 }
        techDebt = @{ todoCount = 0; fixmeCount = 0; hackCount = 0; complexFunctions = 0 }
    }

    # Test metrics
    try {
        $testsDir = Join-Path $script:ProjectRoot 'tests'
        $testOutput = & python -m pytest $testsDir --tb=no -q 2>&1
        $testText = $testOutput -join "`n"
        if ($testText -match '(\d+)\s+passed') { $health.tests.passed = [int]$Matches[1] }
        if ($testText -match '(\d+)\s+failed') { $health.tests.failed = [int]$Matches[1] }
        $health.tests.total = $health.tests.passed + $health.tests.failed
        $health.tests.passRate = if ($health.tests.total -gt 0) {
            [math]::Round($health.tests.passed / $health.tests.total, 2)
        } else { 0.0 }
    }
    catch {}

    # File metrics
    try {
        $pyFiles = @(Get-ChildItem -Path (Join-Path $script:ProjectRoot 'src') -Filter '*.py' -Recurse -ErrorAction SilentlyContinue)
        $ps1Files = @(Get-ChildItem -Path (Join-Path $script:ProjectRoot 'scripts/ralph') -Filter '*.ps1' -ErrorAction SilentlyContinue)
        $health.codebase.pyFiles = $pyFiles.Count
        $health.codebase.ps1Files = $ps1Files.Count

        $totalLines = 0
        foreach ($f in ($pyFiles + $ps1Files)) {
            try {
                $lineCount = (Get-Content $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
                $totalLines += $lineCount
            }
            catch {}
        }
        $health.codebase.totalLines = $totalLines
    }
    catch {}

    # Tech debt indicators
    $trackTechDebt = -not $config.health -or $config.health.trackTechDebt -ne $false
    if ($trackTechDebt) {
        try {
            $srcDir = Join-Path $script:ProjectRoot 'src'
            if (Test-Path $srcDir) {
                Push-Location $script:ProjectRoot
                try {
                    $todoCount = (git grep -c 'TODO' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $fixmeCount = (git grep -c 'FIXME' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $hackCount = (git grep -c 'HACK\|WORKAROUND' -- 'src/*.py' 2>$null | Measure-Object).Count
                    $health.techDebt.todoCount = $todoCount
                    $health.techDebt.fixmeCount = $fixmeCount
                    $health.techDebt.hackCount = $hackCount
                }
                finally {
                    Pop-Location
                }
            }
        }
        catch {}
    }

    # Save metrics
    $healthFile = Join-Path $script:RalphDir "health_metrics.json"
    Write-JsonNoBom -Path $healthFile -Content ($health | ConvertTo-Json -Depth 5)

    return $health
}

function Compare-HealthMetrics {
    <#
    .SYNOPSIS
        Compare current health against previous snapshot (Story 3.2)
    .PARAMETER Current
        Current health metrics hashtable
    .RETURNS
        Hashtable with deltas and trend indicators
    #>
    param(
        [hashtable]$Current
    )

    if (-not $Current) { return $null }

    $healthFile = Join-Path $script:RalphDir "health_metrics.json"
    if (-not (Test-Path $healthFile)) {
        return @{ isBaseline = $true; trends = @() }
    }

    try {
        $previous = Get-Content $healthFile -Raw | ConvertFrom-Json
    }
    catch {
        return @{ isBaseline = $true; trends = @() }
    }

    $trends = @()

    # Test pass rate trend
    $prevPassRate = if ($previous.tests -and $previous.tests.passRate) { $previous.tests.passRate } else { 0 }
    $currPassRate = $Current.tests.passRate
    if ($currPassRate -lt $prevPassRate) {
        $trends += @{ metric = "testPassRate"; direction = "down"; previous = $prevPassRate; current = $currPassRate }
    }
    elseif ($currPassRate -gt $prevPassRate) {
        $trends += @{ metric = "testPassRate"; direction = "up"; previous = $prevPassRate; current = $currPassRate }
    }

    # Test count trend
    $prevTotal = if ($previous.tests -and $previous.tests.total) { $previous.tests.total } else { 0 }
    $currTotal = $Current.tests.total
    if ($currTotal -lt $prevTotal) {
        $trends += @{ metric = "testCount"; direction = "down"; previous = $prevTotal; current = $currTotal }
    }
    elseif ($currTotal -gt $prevTotal) {
        $trends += @{ metric = "testCount"; direction = "up"; previous = $prevTotal; current = $currTotal }
    }

    # Tech debt trend
    $prevDebt = if ($previous.techDebt) {
        ($previous.techDebt.todoCount + $previous.techDebt.fixmeCount + $previous.techDebt.hackCount)
    } else { 0 }
    $currDebt = $Current.techDebt.todoCount + $Current.techDebt.fixmeCount + $Current.techDebt.hackCount
    if ($currDebt -gt $prevDebt) {
        $trends += @{ metric = "techDebt"; direction = "up"; previous = $prevDebt; current = $currDebt }
    }

    return @{ isBaseline = $false; trends = $trends }
}

function Test-TokenBudget {
    <#
    .SYNOPSIS
        Check if token budget allows continuing (Story 3.4)
    .DESCRIPTION
        Blocks story start if sprint token budget exceeded.
        Used alongside Test-MaxIterations in loop conditions.
    .RETURNS
        $true if budget exceeded (should stop), $false if OK to continue
    #>

    $config = Get-RalphConfig
    $budgetEnabled = $config.budget -and $config.budget.enabled

    if (-not $budgetEnabled) {
        return $false  # No budget enforcement
    }

    $budgetStatus = Get-SprintTokenBudget

    if (-not $budgetStatus) {
        return $false
    }

    if ($budgetStatus.exceeded) {
        Write-Host ""
        Write-Host "  TOKEN BUDGET EXCEEDED: $($budgetStatus.totalUsed) / $($budgetStatus.maxTokens) tokens" -ForegroundColor Red
        Write-Host "  Sprint budget depleted. Stopping execution." -ForegroundColor Red
        Write-Host ""
        return $true
    }

    return $false
}

# ============================================================================
# ADVANCED CAPABILITIES (Phase 4)
# ============================================================================

function Get-StoryProgress {
    <#
    .SYNOPSIS
        Get current progress milestones for a story (Story 4.1)
    .DESCRIPTION
        Checks git history and session logs to determine what milestones
        have been reached: tests created, implementation started, committed.
    .PARAMETER StoryId
        Story identifier
    .RETURNS
        Hashtable with milestone states and last checkpoint data
    #>
    param(
        [string]$StoryId
    )

    $progressFile = Join-Path $script:RalphDir "story_progress.json"
    $progress = @{
        storyId = $StoryId
        milestones = @{
            testsCreated = $false
            implementationStarted = $false
            committed = $false
            reviewPassed = $false
        }
        lastCheckpoint = $null
        lastAttemptOutput = ""
    }

    # Check saved progress
    if (Test-Path $progressFile) {
        try {
            $saved = Get-Content $progressFile -Raw | ConvertFrom-Json
            if ($saved.$StoryId) {
                $storyData = $saved.$StoryId
                if ($storyData.milestones) {
                    if ($storyData.milestones.testsCreated) { $progress.milestones.testsCreated = $true }
                    if ($storyData.milestones.implementationStarted) { $progress.milestones.implementationStarted = $true }
                    if ($storyData.milestones.committed) { $progress.milestones.committed = $true }
                    if ($storyData.milestones.reviewPassed) { $progress.milestones.reviewPassed = $true }
                }
                if ($storyData.lastCheckpoint) { $progress.lastCheckpoint = $storyData.lastCheckpoint }
            }
        }
        catch {}
    }

    # Check git log for recent commits mentioning this story
    try {
        $recentCommits = git -C $script:ProjectRoot log --oneline -10 --grep="$StoryId" 2>$null
        if ($recentCommits) {
            $progress.milestones.committed = $true
            $commitText = ($recentCommits -join " ").ToLower()
            if ($commitText -match 'test') { $progress.milestones.testsCreated = $true }
            if ($commitText -match 'implement|add|create|feat') { $progress.milestones.implementationStarted = $true }
        }
    }
    catch {}

    return $progress
}

function Save-StoryProgress {
    <#
    .SYNOPSIS
        Save story progress milestones (Story 4.1)
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Milestone
        Milestone name (testsCreated, implementationStarted, committed, completed)
    .PARAMETER Data
        Optional hashtable of additional data
    #>
    param(
        [string]$StoryId,
        [string]$Milestone,
        [hashtable]$Data = @{}
    )

    $progressFile = Join-Path $script:RalphDir "story_progress.json"
    $allProgress = @{}

    if (Test-Path $progressFile) {
        try {
            $existing = Get-Content $progressFile -Raw | ConvertFrom-Json
            # Convert PSCustomObject to hashtable
            foreach ($prop in $existing.PSObject.Properties) {
                $allProgress[$prop.Name] = $prop.Value
            }
        }
        catch {}
    }

    # Get or create story entry
    $storyEntry = $null
    if ($allProgress.ContainsKey($StoryId) -and $allProgress[$StoryId]) {
        $storyEntry = $allProgress[$StoryId]
    }

    $milestones = @{
        testsCreated = $false
        implementationStarted = $false
        committed = $false
        reviewPassed = $false
    }

    # Preserve existing milestones
    if ($storyEntry -and $storyEntry.milestones) {
        $m = $storyEntry.milestones
        if ($m.testsCreated) { $milestones.testsCreated = $true }
        if ($m.implementationStarted) { $milestones.implementationStarted = $true }
        if ($m.committed) { $milestones.committed = $true }
        if ($m.reviewPassed) { $milestones.reviewPassed = $true }
    }

    # Set the new milestone
    if ($Milestone -eq 'completed') {
        $milestones.testsCreated = $true
        $milestones.implementationStarted = $true
        $milestones.committed = $true
        $milestones.reviewPassed = $true
    }
    elseif ($milestones.ContainsKey($Milestone)) {
        $milestones[$Milestone] = $true
    }

    $allProgress[$StoryId] = @{
        milestones = $milestones
        lastCheckpoint = $Milestone
        data = $Data
    }

    Write-JsonNoBom -Path $progressFile -Content ($allProgress | ConvertTo-Json -Depth 5)
}

# ============================================================================
# STATE MACHINE LOGGING (Phase 2 - Task 2.5)
# ============================================================================

function Log-StateTransition {
    <#
    .SYNOPSIS
        Log state machine transitions for debugging
    #>
    param([string]$From, [string]$To, [string]$Reason, [hashtable]$Context = @{})
    $Context.from = $From; $Context.to = $To; $Context.reason = $Reason; $Context.iteration = $script:IterationCount
    Append-Jsonl -File (Join-Path $script:SessionLogDir "state_transitions.jsonl") -Data $Context
}

# ============================================================================
# PHASE TIMING (Phase 2 - Task 2.1)
# ============================================================================

# ============================================================================
# PHASE TIMING (Phase 2 - Task 2.1)
# ============================================================================

function Measure-PhaseTimings {
    <#
    .SYNOPSIS
        Estimate phase timings from Claude output
    .PARAMETER Output
        Claude's output text
    .PARAMETER TotalDurationMs
        Total execution time in milliseconds
    .RETURNS
        Hashtable with estimated phase timings
    #>
    param(
        [string]$Output,
        [int]$TotalDurationMs
    )

    $timings = @{
        read_ms = 0
        analyze_ms = 0
        implement_ms = 0
        test_ms = 0
        commit_ms = 0
    }

    if (-not $Output -or $TotalDurationMs -le 0) {
        return $timings
    }

    # Estimate based on output content patterns
    $hasReadOps = $Output -match "Read tool|Reading file|file_path"
    $hasAnalysis = $Output -match "analy|understand|plan|think|consider"
    $hasImplement = $Output -match "Edit tool|Write tool|Editing|Writing|implement"
    $hasTests = $Output -match "pytest|test.*pass|test.*fail|running tests"
    $hasGit = $Output -match "git commit|git add|Bash.*git"

    # Count pattern occurrences to weight phases
    $readWeight = if ($hasReadOps) { ([regex]::Matches($Output, "Read tool|Reading file")).Count + 1 } else { 0 }
    $analyzeWeight = if ($hasAnalysis) { 2 } else { 1 }  # Analysis always happens
    $implementWeight = if ($hasImplement) { ([regex]::Matches($Output, "Edit tool|Write tool")).Count + 1 } else { 0 }
    $testWeight = if ($hasTests) { 3 } else { 0 }  # Tests take significant time
    $gitWeight = if ($hasGit) { 1 } else { 0 }

    $totalWeight = [math]::Max(1, $readWeight + $analyzeWeight + $implementWeight + $testWeight + $gitWeight)

    # Distribute time based on weights
    $timings.read_ms = [int](($readWeight / $totalWeight) * $TotalDurationMs)
    $timings.analyze_ms = [int](($analyzeWeight / $totalWeight) * $TotalDurationMs)
    $timings.implement_ms = [int](($implementWeight / $totalWeight) * $TotalDurationMs)
    $timings.test_ms = [int](($testWeight / $totalWeight) * $TotalDurationMs)
    $timings.commit_ms = [int](($gitWeight / $totalWeight) * $TotalDurationMs)

    return $timings
}

# ============================================================================
# ERROR EVOLUTION (Phase 2 - Task 2.4)
# ============================================================================

# ============================================================================
# ERROR EVOLUTION (Phase 2 - Task 2.4)
# ============================================================================

function Log-ErrorEvolution {
    <#
    .SYNOPSIS
        Track error patterns over time
    #>
    param([string]$ErrorCategory, [string]$ErrorDetails = "", [int]$Iteration)
    Append-Jsonl -File (Join-Path $script:SessionLogDir "error_evolution.jsonl") -Data @{
        category = $ErrorCategory; details = $ErrorDetails; iteration = $Iteration
    }
}

# ============================================================================
# PHASE 3: DETAILED TRACKING
# ============================================================================

# Task 3.1: Configuration Change Audit Trail

# Task 3.1: Configuration Change Audit Trail
function Log-ConfigChange {
    <#
    .SYNOPSIS
        Log configuration changes for audit trail
    #>
    param([string]$Field, $OldValue, $NewValue, [string]$Reason = "manual")
    Append-Jsonl -File (Join-Path $script:RalphDir "config_audit.jsonl") -SkipSessionCheck -Data @{
        field = $Field; old = $OldValue; new = $NewValue; reason = $Reason
    }
}

# Task 3.2: Test Failure Detail Logging

# Task 3.2: Test Failure Detail Logging
function Log-TestDetails {
    <#
    .SYNOPSIS
        Parse and log detailed test results from Claude output
    .PARAMETER Iteration
        Iteration number
    .PARAMETER Output
        Claude's output containing test results
    #>
    param(
        [int]$Iteration,
        [string]$Output
    )

    $testDetailsFile = Join-Path $script:SessionLogDir "test_details_$Iteration.json"

    # Parse pytest output for individual test results
    $testResults = @()
    $passed = 0
    $failed = 0
    $skipped = 0
    $errors = 0

    # Match pytest verbose output: test_file.py::test_name PASSED/FAILED
    $testMatches = [regex]::Matches($Output, '([\w\/]+\.py::\w+)\s+(PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[\s*(\d+)%\])?')
    foreach ($match in $testMatches) {
        $testName = $match.Groups[1].Value
        $status = $match.Groups[2].Value.ToLower()

        $testResults += @{
            name = $testName
            status = $status
        }

        switch ($status) {
            "passed" { $passed++ }
            "failed" { $failed++ }
            "skipped" { $skipped++ }
            "error" { $errors++ }
        }
    }

    # Extract duration if present
    $duration = 0
    if ($Output -match "passed.*in\s+([\d.]+)s") {
        $duration = [double]$Matches[1]
    }

    $testDetails = @{
        iteration = $Iteration
        testRun = @{
            command = "pytest"
            duration = $duration
            exitCode = if ($failed -eq 0 -and $errors -eq 0) { 0 } else { 1 }
        }
        results = $testResults
        summary = @{
            passed = $passed
            failed = $failed
            skipped = $skipped
            errors = $errors
            total = $passed + $failed + $skipped + $errors
        }
    }

    Write-JsonNoBom -Path $testDetailsFile -Content ($testDetails | ConvertTo-Json -Depth 5)
    return $testDetails
}

# Task 3.3: Resource Usage Monitoring

# Task 3.3: Resource Usage Monitoring
function Get-ProcessMetrics {
    <#
    .SYNOPSIS
        Get resource usage metrics for a process
    .PARAMETER ProcessId
        Process ID to monitor
    .RETURNS
        Hashtable with CPU, memory, handles, threads
    #>
    param([int]$ProcessId)

    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            return @{
                cpu = [math]::Round($proc.CPU, 2)
                memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
                handles = $proc.HandleCount
                threads = $proc.Threads.Count
                timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
            }
        }
    }
    catch {}

    return @{ cpu = 0; memoryMB = 0; handles = 0; threads = 0; timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ") }
}

function Log-ResourceUsage {
    <#
    .SYNOPSIS
        Log resource usage for an iteration
    .PARAMETER Iteration
        Iteration number
    .PARAMETER ProcessId
        Process ID that was monitored
    .PARAMETER Samples
        Array of resource samples
    #>
    param(
        [int]$Iteration,
        [int]$ProcessId,
        [array]$Samples
    )

    $resourceFile = Join-Path $script:SessionLogDir "resource_usage_$Iteration.json"

    # Calculate averages and peaks
    $avgCpu = if ($Samples.Count -gt 0) { [math]::Round(($Samples | ForEach-Object { $_.cpu } | Measure-Object -Average).Average, 2) } else { 0 }
    $avgMem = if ($Samples.Count -gt 0) { [math]::Round(($Samples | ForEach-Object { $_.memoryMB } | Measure-Object -Average).Average, 2) } else { 0 }
    $peakMem = if ($Samples.Count -gt 0) { ($Samples | ForEach-Object { $_.memoryMB } | Measure-Object -Maximum).Maximum } else { 0 }

    $usage = @{
        iteration = $Iteration
        processId = $ProcessId
        sampleCount = $Samples.Count
        averages = @{
            cpuSeconds = $avgCpu
            memoryMB = $avgMem
        }
        peaks = @{
            memoryMB = $peakMem
        }
        samples = $Samples
    }

    Write-JsonNoBom -Path $resourceFile -Content ($usage | ConvertTo-Json -Depth 5)
}

# Task 3.4: Prompt Effectiveness Scoring

# Task 3.4: Prompt Effectiveness Scoring
function Get-PromptEffectiveness {
    <#
    .SYNOPSIS
        Calculate prompt effectiveness score
    .PARAMETER Success
        Whether iteration succeeded
    .PARAMETER RetryCount
        Number of retries attempted
    .RETURNS
        Effectiveness score (0.0 to 1.0)
    #>
    param(
        [bool]$Success,
        [int]$RetryCount
    )

    if (-not $Success) {
        return 0.0
    }

    if ($RetryCount -le 1) {
        return 1.0  # First try success
    }
    elseif ($RetryCount -le 3) {
        return 0.5  # Success after few retries
    }
    else {
        return 0.25  # Success after many retries
    }
}

function Log-PromptEffectiveness {
    <#
    .SYNOPSIS
        Log prompt effectiveness for analysis
    #>
    param([int]$Iteration, [string]$PromptType, [double]$Effectiveness, [string]$PromptHash)
    Append-Jsonl -File (Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl") -Data @{
        iteration = $Iteration; promptType = $PromptType; effectiveness = $Effectiveness; promptHash = $PromptHash
    }
}

# Task 3.5: Skip/Blocker Tracking

# Task 3.5: Skip/Blocker Tracking
function Log-Skip {
    <#
    .SYNOPSIS
        Log when a story/focus area is skipped
    #>
    param([string]$ItemId, [string]$ItemType = "story", [string]$Reason, [string]$BlockerType = "manual")
    Append-Jsonl -File (Join-Path $script:SessionLogDir "skips_blockers.jsonl") -Data @{
        itemId = $ItemId; itemType = $ItemType; reason = $Reason; blockerType = $BlockerType; resolved = $false
    }
    Append-SessionTimeline -Event "item_skipped" -Data @{ itemId = $ItemId; reason = $Reason }
}

# ============================================================================
# CLAUDE INVOCATION
# ============================================================================

function Record-Metric {
    param(
        [string]$Session,
        [string]$Sprint,
        [string]$StoryId,
        [string]$Mode,
        [double]$DurationMin,
        [bool]$Success = $true,
        [bool]$Timeout = $false,
        [string]$FocusArea,
        [int]$TokensUsed = 0,
        [string]$ErrorCategory = "",
        [int]$HourOfDay = -1,
        [string]$TestResults = "",
        [int]$RetryCount = 0,
        [int]$LinesAdded = 0,
        [int]$LinesDeleted = 0,
        # Phase 2 - Task 2.1: Phase timing breakdown
        [int]$PhaseReadMs = 0,
        [int]$PhaseAnalyzeMs = 0,
        [int]$PhaseImplementMs = 0,
        [int]$PhaseTestMs = 0,
        [int]$PhaseCommitMs = 0,
        # Exploration tracking
        [bool]$ExplorationTriggered = $false,
        [string]$ExplorationReason = "",
        [int]$ExplorationTokens = 0
    )

    # V3 schema: 24 columns including exploration tracking
    $v3Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms,exploration_triggered,exploration_reason,exploration_tokens"

    # Ensure metrics file exists with v3 header
    if (-not (Test-Path $script:MetricsFile)) {
        $v3Header | Set-Content $script:MetricsFile -Encoding UTF8
    }
    else {
        # Check if we need to migrate schema
        $header = Get-Content $script:MetricsFile -First 1
        $headerCols = ($header -split ',').Count

        if ($headerCols -lt 24) {
            # Migration: rewrite with v3 header and pad old rows
            $lines = Get-Content $script:MetricsFile
            $lines[0] = $v3Header
            # Add empty values to existing rows (pad to 24 columns)
            for ($i = 1; $i -lt $lines.Count; $i++) {
                $rowCols = ($lines[$i] -split ',').Count
                $padding = 24 - $rowCols
                if ($padding -gt 0) {
                    # Pad with appropriate defaults: false,, 0 for new exploration columns
                    if ($rowCols -eq 21) {
                        # Coming from v2, add 3 new exploration columns
                        $lines[$i] = $lines[$i] + ",false,,0"
                    } else {
                        # Coming from older version, pad with zeros
                        $lines[$i] = $lines[$i] + (',' + '0' * $padding -replace '0', ',0').Substring(1)
                    }
                }
            }
            $lines | Set-Content $script:MetricsFile -Encoding UTF8
            Write-Host "  Migrated metrics.csv to v3 schema (24 columns with exploration)" -ForegroundColor DarkGray
        }
    }

    # Use defaults from script variables if not provided
    if (-not $Session) { $Session = $script:SessionId }
    if (-not $Mode) { $Mode = $script:CurrentMode }
    if (-not $FocusArea -or -not $Sprint) {
        if (Test-Path $script:PrdFile) {
            try {
                $prd = Get-Content $script:PrdFile -Raw | ConvertFrom-Json
                if (-not $FocusArea) { $FocusArea = $prd.focusArea }
                if (-not $Sprint) { $Sprint = "sprint-$($prd.sprintNumber)" }
            }
            catch {}
        }
    }

    # Calculate hour_of_day if not provided
    if ($HourOfDay -eq -1) {
        $HourOfDay = (Get-Date).Hour
    }

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $row = "$timestamp,$Session,$Sprint,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),$($Timeout.ToString().ToLower()),$FocusArea,$TokensUsed,$ErrorCategory,$HourOfDay,$TestResults,$RetryCount,$LinesAdded,$LinesDeleted,$PhaseReadMs,$PhaseAnalyzeMs,$PhaseImplementMs,$PhaseTestMs,$PhaseCommitMs,$($ExplorationTriggered.ToString().ToLower()),$ExplorationReason,$ExplorationTokens"
    Add-Content -Path $script:MetricsFile -Value $row
}

# ============================================================================
# FAST-FAIL DETECTION
# ============================================================================
