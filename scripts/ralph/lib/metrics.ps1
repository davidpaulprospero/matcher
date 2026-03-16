# scripts/ralph/lib/metrics.ps1
# Metrics, logging, instrumentation: CSV tracking, git state, health, tokens

# ============================================================================
# NON-LOCKING CSV READER
# ============================================================================

function Write-MetricsRow {
    <#
    .SYNOPSIS
        Append a row to a CSV file using non-locking FileStream.
        Retries up to 5 times with exponential backoff on IOException.
    #>
    param(
        [string]$Path,
        [string]$Row
    )
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            $stream = [System.IO.FileStream]::new(
                $Path,
                [System.IO.FileMode]::Append,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::ReadWrite
            )
            $writer = [System.IO.StreamWriter]::new($stream, [System.Text.Encoding]::UTF8)
            try {
                $writer.WriteLine($Row)
            } finally {
                $writer.Close()
                $stream.Close()
            }
            return
        } catch {
            if ($attempt -lt 5) {
                Start-Sleep -Milliseconds (100 * $attempt)
            } else {
                Write-Host "  Warning: Failed to write metrics row after 5 attempts: $_" -ForegroundColor Yellow
            }
        }
    }
}

function Import-CsvNonLocking {
    <#
    .SYNOPSIS
        Read a CSV file without holding an exclusive lock, preventing IOException
        when another process (Add-Content/FileStream) writes concurrently.
    .PARAMETER Path
        Path to the CSV file
    .OUTPUTS
        Array of PSCustomObjects (same as Import-Csv)
    #>
    param([string]$Path)
    if (-not (Test-Path $Path)) { return @() }
    try {
        $stream = [System.IO.FileStream]::new(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
        try {
            $content = $reader.ReadToEnd()
        } finally {
            $reader.Close()
            $stream.Close()
        }
        if ([string]::IsNullOrWhiteSpace($content)) { return @() }
        return @($content | ConvertFrom-Csv)
    } catch {
        return @()
    }
}

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

    $entry = @{ ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ"); session = $script:State.SessionId }
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
    $prd = Get-Sprint
    if ($prd) {
        $sprint = $prd.sprintNumber
        $prdBranch = $prd.branchName
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

function Confirm-CriteriaEvidence {
    <#
    .SYNOPSIS
        LLM-verify acceptance criteria against git diff and Claude output (replaces keyword matching)
    .DESCRIPTION
        Sends a single LLM call (haiku) with all acceptance criteria + truncated evidence.
        Returns per-criterion MET/NOT_MET results. Falls back to Search-CriterionEvidence
        keyword matching if Claude CLI is unavailable or LLM call fails.
    .PARAMETER Criteria
        Array of acceptance criterion strings
    .PARAMETER ClaudeOutput
        Full Claude stdout+stderr for evidence searching
    .PARAMETER DiffOutput
        Git diff output for evidence searching
    .RETURNS
        Array of hashtables: @{ index; met (bool); evidence (string); confidence (float) }
    #>
    param(
        [array]$Criteria = @(),
        [string]$ClaudeOutput = "",
        [string]$DiffOutput = ""
    )

    if (-not $Criteria -or $Criteria.Count -eq 0) { return @() }

    $claudePath = Get-ClaudePath
    if (-not $claudePath) {
        Write-Host "  Evidence: Claude not available, falling back to keyword matching" -ForegroundColor DarkYellow
        return $null  # Signal caller to use fallback
    }

    # Build numbered criteria list
    $criteriaList = ""
    for ($i = 0; $i -lt $Criteria.Count; $i++) {
        $criteriaList += "$($i + 1). $($Criteria[$i])`n"
    }

    # Truncate evidence: diff = first 8000 chars, output = last 8000 chars
    $truncatedDiff = if ($DiffOutput.Length -gt 8000) { $DiffOutput.Substring(0, 8000) + "`n[...truncated]" } else { $DiffOutput }
    $truncatedOutput = if ($ClaudeOutput.Length -gt 8000) { $ClaudeOutput.Substring($ClaudeOutput.Length - 8000) } else { $ClaudeOutput }

    $prompt = @"
You are verifying whether acceptance criteria for a user story were actually implemented.
Review the git diff (code changes) and implementation output below, then determine which criteria are MET.

A criterion is MET if the code changes or output show clear evidence it was implemented.
A criterion is NOT_MET only if there is no evidence at all in the diff or output.

ACCEPTANCE CRITERIA:
$criteriaList
GIT DIFF (code changes):
$truncatedDiff

IMPLEMENTATION OUTPUT:
$truncatedOutput

For each criterion, respond with ONLY MET or NOT_MET followed by the number. Nothing else.
Example:
MET 1
NOT_MET 2
MET 3
"@

    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model haiku"
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        # Async output capture (prevents pipe buffer deadlock)
        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            $process.StandardInput.Write($prompt)
            $process.StandardInput.Close()

            # 60 second timeout for simple classification
            $deadline = (Get-Date).AddSeconds(60)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $completed = $process.HasExited

            if (-not $completed) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                Write-Host "  Evidence: LLM verification timed out, falling back to keyword matching" -ForegroundColor DarkYellow
                return $null  # Signal caller to use fallback
            }

            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()

            # Parse response: MET <n> or NOT_MET <n> (tolerant of various LLM formats)
            $results = @()
            for ($i = 0; $i -lt $Criteria.Count; $i++) {
                # Default to NOT_MET if LLM didn't mention this criterion
                $results += @{ index = $i; met = $false; evidence = "No LLM response for criterion"; confidence = 0.0 }
            }

            $parsedCount = 0
            foreach ($line in ($output -split "`n")) {
                $trimmed = $line.Trim()
                # Tolerant parsing: accept "NOT_MET 1", "1. NOT_MET", "NOT_MET: 1", "1 NOT_MET", "1 - NOT_MET"
                if ($trimmed -match '(?:^|\b)NOT[_\s-]?MET\s*[:\.\-]?\s*(\d+)' -or $trimmed -match '(\d+)\s*[:\.\-]?\s*NOT[_\s-]?MET') {
                    $num = [int]$Matches[1] - 1  # Convert 1-based to 0-based
                    if ($num -ge 0 -and $num -lt $Criteria.Count) {
                        $results[$num] = @{ index = $num; met = $false; evidence = "LLM: NOT_MET"; confidence = 0.8 }
                        $parsedCount++
                    }
                }
                # Tolerant parsing: accept "MET 1", "1. MET", "MET: 1", "1 MET", "1 - MET"
                # NOT_MET already matched above, so this elseif only fires for non-NOT_MET lines
                elseif ($trimmed -match '(?:^|\b)MET\s*[:\.\-]?\s*(\d+)' -or $trimmed -match '(\d+)\s*[:\.\-]?\s*MET(?:\b|$)') {
                    $num = [int]$Matches[1] - 1
                    if ($num -ge 0 -and $num -lt $Criteria.Count) {
                        $results[$num] = @{ index = $num; met = $true; evidence = "LLM: MET"; confidence = 0.9 }
                        $parsedCount++
                    }
                }
            }

            # If LLM ran but we couldn't parse ANY lines, return $null to trigger keyword fallback
            if ($parsedCount -eq 0) {
                Write-Host "  Evidence: LLM response unparseable, falling back to keyword matching" -ForegroundColor DarkYellow
                return $null
            }

            return $results
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
            if ($process) { $process.Dispose() }
        }
    }
    catch {
        Write-Host "  Evidence: LLM verification failed: $_, falling back to keyword matching" -ForegroundColor DarkYellow
        return $null  # Signal caller to use fallback
    }
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
    $usedKeywordFallback = $false

    if ($Story -and $Story.acceptanceCriteria) {
        $criteriaTotalCount = $Story.acceptanceCriteria.Count

        if ($Passed -and ($ClaudeOutput -or $DiffOutput)) {
            # Try LLM-based verification first (single batch call)
            $llmResults = Confirm-CriteriaEvidence -Criteria $Story.acceptanceCriteria -ClaudeOutput $ClaudeOutput -DiffOutput $DiffOutput

            if ($null -ne $llmResults) {
                # LLM verification succeeded -- map results and check for all-false safety net
                $llmMetCount = 0
                for ($i = 0; $i -lt $Story.acceptanceCriteria.Count; $i++) {
                    $r = $llmResults[$i]
                    if ($r.met) { $llmMetCount++ }
                }

                # Safety net: if LLM says 0/N criteria met, cross-check with keyword matching
                if ($llmMetCount -eq 0 -and $Story.acceptanceCriteria.Count -gt 0) {
                    $criteriaCount = $Story.acceptanceCriteria.Count
                    Write-Host "  Evidence: LLM returned 0/$criteriaCount met, cross-checking with keyword matching" -ForegroundColor DarkYellow
                    $keywordMetCount = 0
                    foreach ($criterion in $Story.acceptanceCriteria) {
                        $kwResult = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $ClaudeOutput -DiffOutput $DiffOutput
                        if ($kwResult.found) { $keywordMetCount++ }
                    }
                    if ($keywordMetCount -gt $llmMetCount) {
                        Write-Host "  Evidence: Keyword matching found $keywordMetCount/$criteriaCount -- using keyword results" -ForegroundColor DarkYellow
                        $llmResults = $null
                    }
                }

                if ($null -ne $llmResults) {
                    for ($i = 0; $i -lt $Story.acceptanceCriteria.Count; $i++) {
                        $r = $llmResults[$i]
                        if ($r.met) { $criteriaMetCount++ }
                        $criteriaVerification += @{
                            criterion  = $Story.acceptanceCriteria[$i]
                            verified   = $r.met
                            evidence   = $r.evidence
                            confidence = $r.confidence
                        }
                    }
                }
            }

            if ($null -eq $llmResults) {
                # Fallback: LLM unavailable or returned all-false, use keyword matching
                $usedKeywordFallback = $true
                foreach ($criterion in $Story.acceptanceCriteria) {
                    $evidenceResult = Search-CriterionEvidence -Criterion $criterion -ClaudeOutput $ClaudeOutput -DiffOutput $DiffOutput
                    if ($evidenceResult.found) { $criteriaMetCount++ }
                    $criteriaVerification += @{
                        criterion  = $criterion
                        verified   = $evidenceResult.found
                        evidence   = $evidenceResult.evidence
                        confidence = $evidenceResult.confidence
                    }
                }
            }
        }
        elseif ($Passed) {
            # Passed but no output to search
            foreach ($criterion in $Story.acceptanceCriteria) {
                $criteriaMetCount++
                $criteriaVerification += @{
                    criterion  = $criterion
                    verified   = $true
                    evidence   = "Story passed (exit code 0) but no output available for evidence search"
                    confidence = 0.5
                }
            }
        }
        else {
            # Story not yet complete
            foreach ($criterion in $Story.acceptanceCriteria) {
                $criteriaVerification += @{
                    criterion  = $criterion
                    verified   = $false
                    evidence   = "Story not yet complete"
                    confidence = 0.0
                }
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

    return @{
        criteriaMet         = $criteriaMetCount
        criteriaTotal       = $criteriaTotalCount
        percentage          = if ($criteriaTotalCount -gt 0) { [math]::Round(($criteriaMetCount / $criteriaTotalCount) * 100, 0) } else { 100 }
        usedKeywordFallback = $usedKeywordFallback
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
            $metrics = @(Import-CsvNonLocking $script:MetricsFile)
            $sessionMetrics = @($metrics | Where-Object { $_.session -eq $script:State.SessionId })
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

    # Test metrics (with timeout to prevent hanging the loop)
    try {
        $testsDir = Join-Path $script:ProjectRoot 'tests'
        $healthTimeoutSec = 120  # Max 2 minutes for health check pytest run
        $outTmp = [System.IO.Path]::GetTempFileName()
        $errTmp = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath "python" -ArgumentList "-m pytest `"$testsDir`" --tb=no -q -x --timeout=30" `
                -NoNewWindow -RedirectStandardOutput $outTmp -RedirectStandardError $errTmp -PassThru -WorkingDirectory $script:ProjectRoot
            $completed = $proc.WaitForExit($healthTimeoutSec * 1000)
            if (-not $completed) {
                try { $proc.Kill() } catch {}
                Write-Host "  Health: pytest timed out after ${healthTimeoutSec}s (killed)" -ForegroundColor Yellow
            }
            $testText = Get-Content $outTmp -Raw -ErrorAction SilentlyContinue
            if ($testText) {
                if ($testText -match '(\d+)\s+passed') { $health.tests.passed = [int]$Matches[1] }
                if ($testText -match '(\d+)\s+failed') { $health.tests.failed = [int]$Matches[1] }
            }
        }
        finally {
            Remove-Item $outTmp -Force -ErrorAction SilentlyContinue
            Remove-Item $errTmp -Force -ErrorAction SilentlyContinue
        }
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
    $healthFile = if ($script:Paths) { $script:Paths.HealthMetricsFile } else { Join-Path $script:RalphDir "state\health_metrics.json" }
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

    $healthFile = if ($script:Paths) { $script:Paths.HealthMetricsFile } else { Join-Path $script:RalphDir "state\health_metrics.json" }
    $previous = Read-JsonFile -Path $healthFile
    if (-not $previous) {
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

    $progressFile = if ($script:Paths) { $script:Paths.StoryProgressFile } else { Join-Path $script:RalphDir "state\story_progress.json" }
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
    $saved = Read-JsonFile -Path $progressFile
    if ($saved -and $saved.$StoryId) {
        $storyData = $saved.$StoryId
        if ($storyData.milestones) {
            if ($storyData.milestones.testsCreated) { $progress.milestones.testsCreated = $true }
            if ($storyData.milestones.implementationStarted) { $progress.milestones.implementationStarted = $true }
            if ($storyData.milestones.committed) { $progress.milestones.committed = $true }
            if ($storyData.milestones.reviewPassed) { $progress.milestones.reviewPassed = $true }
        }
        if ($storyData.lastCheckpoint) { $progress.lastCheckpoint = $storyData.lastCheckpoint }
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

    $progressFile = if ($script:Paths) { $script:Paths.StoryProgressFile } else { Join-Path $script:RalphDir "state\story_progress.json" }
    $allProgress = @{}

    $existing = Read-JsonFile -Path $progressFile
    if ($existing) {
        # Convert PSCustomObject to hashtable
        foreach ($prop in $existing.PSObject.Properties) {
            $allProgress[$prop.Name] = $prop.Value
        }
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
    $Context.from = $From; $Context.to = $To; $Context.reason = $Reason; $Context.iteration = $script:State.IterationCount
    Append-Jsonl -File (Join-Path $script:SessionLogDir "state_transitions.jsonl") -Data $Context
}

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

function Get-ChildProcessActivity {
    <#
    .SYNOPSIS
        Detect child processes (like pytest) and their CPU activity.
        Used to extend stall timeout when tests are running.
    .PARAMETER ParentProcessId
        Parent process ID (typically Claude)
    .RETURNS
        Hashtable with: hasTestRunner, testRunnerName, totalChildCpu, childCount
    #>
    param([int]$ParentProcessId)

    $result = @{
        hasTestRunner = $false
        testRunnerName = $null
        totalChildCpu = 0
        childCount = 0
        childProcesses = @()
    }

    try {
        # Get child processes using WMI (more reliable for process tree)
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentProcessId" -ErrorAction SilentlyContinue

        if ($children) {
            $result.childCount = @($children).Count

            foreach ($child in $children) {
                $childName = $child.Name.ToLower()
                $result.childProcesses += $childName

                # Check for test runners
                if ($childName -match 'pytest|python|py\.exe|node') {
                    # Get the process to check CPU and command line
                    $proc = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
                    if ($proc) {
                        $result.totalChildCpu += $proc.CPU

                        # Check command line for pytest indicators
                        $cmdLine = $child.CommandLine
                        if ($cmdLine -and ($cmdLine -match 'pytest|test_|tests/' -or $childName -eq 'pytest.exe')) {
                            $result.hasTestRunner = $true
                            $result.testRunnerName = if ($cmdLine -match 'pytest') { 'pytest' } else { $childName }
                        }
                    }
                }

                # Recursively check grandchildren
                $grandchildren = Get-ChildProcessActivity -ParentProcessId $child.ProcessId
                if ($grandchildren.hasTestRunner) {
                    $result.hasTestRunner = $true
                    $result.testRunnerName = $grandchildren.testRunnerName
                }
                $result.totalChildCpu += $grandchildren.totalChildCpu
                $result.childCount += $grandchildren.childCount
            }
        }
    }
    catch {
        # Ignore errors - process may have exited
    }

    return $result
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
        [int]$ExplorationTokens = 0,
        # Role tracking
        [string]$Role = ""
    )

    # V4 schema: 25 columns including role tracking
    $v4Header = "timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms,exploration_triggered,exploration_reason,exploration_tokens,role"

    # Ensure metrics file exists with v4 header
    if (-not (Test-Path $script:MetricsFile)) {
        $v4Header | Set-Content $script:MetricsFile -Encoding UTF8
    }
    else {
        # Check if we need to migrate schema
        $header = Get-Content $script:MetricsFile -First 1
        $headerCols = ($header -split ',').Count

        if ($headerCols -lt 25) {
            # Migration: rewrite with v4 header and pad old rows
            $lines = Get-Content $script:MetricsFile
            $lines[0] = $v4Header
            # Add empty values to existing rows (pad to 25 columns)
            for ($i = 1; $i -lt $lines.Count; $i++) {
                $rowCols = ($lines[$i] -split ',').Count
                if ($rowCols -eq 24) {
                    # Coming from v3, add 1 new role column
                    $lines[$i] = $lines[$i] + ","
                } elseif ($rowCols -eq 21) {
                    # Coming from v2, add 3 exploration columns + role
                    $lines[$i] = $lines[$i] + ",false,,0,"
                } else {
                    # Coming from older version, pad to 25
                    $padding = 25 - $rowCols
                    if ($padding -gt 0) {
                        $phasePadding = [math]::Max(0, 21 - $rowCols)
                        if ($phasePadding -gt 0) {
                            $lines[$i] = $lines[$i] + (',0' * $phasePadding)
                        }
                        $lines[$i] = $lines[$i] + ",false,,0,"
                    }
                }
            }
            $lines | Set-Content $script:MetricsFile -Encoding UTF8
            Write-Host "  Migrated metrics.csv to v4 schema (25 columns with role)" -ForegroundColor DarkGray
        }
    }

    # Use defaults from script variables if not provided
    if (-not $Session) { $Session = $script:State.SessionId }
    if (-not $Mode) { $Mode = $script:State.CurrentMode }
    if (-not $FocusArea -or -not $Sprint) {
        $prd = Get-Sprint
        if ($prd) {
            if (-not $FocusArea) { $FocusArea = $prd.focusArea }
            if (-not $Sprint) { $Sprint = "sprint-$($prd.sprintNumber)" }
        }
    }

    # Calculate hour_of_day if not provided
    if ($HourOfDay -eq -1) {
        $HourOfDay = (Get-Date).Hour
    }

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    # Quote CSV fields that may contain commas to prevent corruption
    $safeError = if ($ErrorCategory -match '[,"\r\n]') { "`"$(($ErrorCategory -replace '"', '""') -replace '[\r\n]+', ' ')`"" } else { $ErrorCategory }
    $safeResults = if ($TestResults -match '[,"\r\n]') { "`"$(($TestResults -replace '"', '""') -replace '[\r\n]+', ' ')`"" } else { $TestResults }
    $safeReason = if ($ExplorationReason -match '[,"\r\n]') { "`"$(($ExplorationReason -replace '"', '""') -replace '[\r\n]+', ' ')`"" } else { $ExplorationReason }
    $row = "$timestamp,$Session,$Sprint,$StoryId,$Mode,$DurationMin,$($Success.ToString().ToLower()),$($Timeout.ToString().ToLower()),$FocusArea,$TokensUsed,$safeError,$HourOfDay,$safeResults,$RetryCount,$LinesAdded,$LinesDeleted,$PhaseReadMs,$PhaseAnalyzeMs,$PhaseImplementMs,$PhaseTestMs,$PhaseCommitMs,$($ExplorationTriggered.ToString().ToLower()),$safeReason,$ExplorationTokens,$Role"
    Write-MetricsRow -Path $script:MetricsFile -Row $row
}

# ============================================================================
# SPRINT DIAGNOSTICS
# ============================================================================

function Get-SprintDiagnostics {
    <#
    .SYNOPSIS
        Compute sprint diagnostics: flag state, derived metrics, 3-sprint comparison.
        Writes session/diagnostics.json and returns the diagnostics hashtable.
    .PARAMETER SprintNumber
        Sprint number to analyze (0 = current sprint from PRD)
    .PARAMETER Prd
        Optional PRD object (loaded if not provided)
    .RETURNS
        Hashtable with active_flags, metrics, comparison, flag_impact
    #>
    param(
        [int]$SprintNumber = 0,
        [object]$Prd = $null
    )

    # Load PRD if not provided
    if (-not $Prd) {
        $Prd = Get-Sprint
    }
    if (-not $Prd) { return $null }

    if ($SprintNumber -eq 0 -and $Prd.sprintNumber) {
        $SprintNumber = $Prd.sprintNumber
    }

    # 1. Read config flags
    $config = Get-RalphConfig
    $activeFlags = @{}
    if ($config.flags) {
        $config.flags.PSObject.Properties | ForEach-Object {
            $activeFlags[$_.Name] = $_.Value
        }
    }

    # 2. Read metrics CSV
    $metricsFile = if ($script:MetricsFile) { $script:MetricsFile } else {
        $sessionDir = if ($script:Paths) { $script:Paths.SessionDir } else { Join-Path $script:RalphDir "session" }
        Join-Path $sessionDir "metrics.csv"
    }
    $allMetrics = @()
    if (Test-Path $metricsFile) {
        $allMetrics = @(Import-CsvNonLocking $metricsFile)
    }

    # 3. Filter to current sprint rows
    $sprintLabel = "sprint-$SprintNumber"
    $sprintRows = @($allMetrics | Where-Object { $_.sprint -eq $sprintLabel })

    # 4. Compute derived metrics
    $storyRows = @($sprintRows | Where-Object { $_.story_id -and $_.story_id -notlike 'HEALING-*' })
    $distinctStories = @($storyRows | Select-Object -ExpandProperty story_id -Unique)
    $storiesAttempted = $distinctStories.Count

    $storiesPassedFirstTry = 0
    $storiesPassedWithRetry = 0
    $storiesFailed = 0

    foreach ($sid in $distinctStories) {
        $storyMetrics = @($storyRows | Where-Object { $_.story_id -eq $sid } | Sort-Object timestamp)
        if ($storyMetrics.Count -eq 0) { continue }

        $firstRow = $storyMetrics[0]
        $anySuccess = @($storyMetrics | Where-Object { $_.success -eq 'true' -or $_.success -eq 'True' })

        if ($firstRow.success -eq 'true' -or $firstRow.success -eq 'True') {
            $retryCount = if ($firstRow.retry_count) { [int]$firstRow.retry_count } else { 0 }
            if ($retryCount -eq 0) {
                $storiesPassedFirstTry++
            } else {
                $storiesPassedWithRetry++
            }
        } elseif ($anySuccess.Count -gt 0) {
            $storiesPassedWithRetry++
        } else {
            $storiesFailed++
        }
    }

    $firstAttemptSuccessRate = if ($storiesAttempted -gt 0) { [math]::Round($storiesPassedFirstTry / $storiesAttempted, 4) } else { 0 }
    $overallSuccessRate = if ($storiesAttempted -gt 0) { [math]::Round(($storiesPassedFirstTry + $storiesPassedWithRetry) / $storiesAttempted, 4) } else { 0 }

    $avgDurationMinutes = 0
    $durationValues = @($sprintRows | Where-Object { $_.duration_min } | ForEach-Object { [double]$_.duration_min })
    if ($durationValues.Count -gt 0) {
        $avgDurationMinutes = [math]::Round(($durationValues | Measure-Object -Average).Average, 2)
    }

    $healingSessions = @($sprintRows | Where-Object { $_.story_id -like 'HEALING-*' }).Count

    # Count regressions from healing log
    $regressionsCaught = 0
    $healingLogFile = if ($script:Paths) { Join-Path $script:Paths.SessionDir "healing_log.jsonl" } else { Join-Path $script:RalphDir "session\healing_log.jsonl" }
    if (Test-Path $healingLogFile) {
        try {
            $healingLines = Get-Content $healingLogFile -ErrorAction SilentlyContinue
            foreach ($line in $healingLines) {
                if ($line -match '"regression_detected"') {
                    $regressionsCaught++
                }
            }
        } catch {}
    }

    # 5. 3-sprint comparison
    $comparison = @{ vs_last_3_sprints = $null }
    try {
        $history = Get-SprintHistory
        if ($history -and $history.sprints -and $history.sprints.Count -gt 0) {
            $recentSprints = @($history.sprints | Select-Object -Last 3)
            if ($recentSprints.Count -gt 0) {
                $avgHistoricCompleted = [math]::Round(($recentSprints | ForEach-Object { $_.storiesCompleted } | Measure-Object -Average).Average, 2)
                $avgHistoricTotal = [math]::Round(($recentSprints | ForEach-Object { $_.storiesTotal } | Measure-Object -Average).Average, 2)
                $avgHistoricRate = if ($avgHistoricTotal -gt 0) { [math]::Round($avgHistoricCompleted / $avgHistoricTotal, 4) } else { 0 }

                $firstAttemptDelta = $null
                if ($avgHistoricRate -gt 0) {
                    $delta = $firstAttemptSuccessRate - $avgHistoricRate
                    $sign = if ($delta -ge 0) { "+" } else { "" }
                    $firstAttemptDelta = "$sign$([math]::Round($delta * 100, 1))%"
                }

                $comparison.vs_last_3_sprints = @{
                    avg_stories_completed = $avgHistoricCompleted
                    avg_stories_total = $avgHistoricTotal
                    avg_success_rate = $avgHistoricRate
                    first_attempt_rate_delta = $firstAttemptDelta
                    sprints_compared = $recentSprints.Count
                }
            }
        }
    } catch {}

    # 6. Flag impact (stubbed, regressionGuard only for now)
    $flagImpact = @{}
    if ($activeFlags['regressionGuard'] -eq $true) {
        $flagImpact['regressionGuard'] = @{
            enabled_since = "current_sprint"
            regressions_caught = $regressionsCaught
        }
    }

    # 7. Build diagnostics object
    $diagnostics = @{
        sprint_number = $SprintNumber
        generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        active_flags = $activeFlags
        metrics = @{
            stories_attempted = $storiesAttempted
            stories_passed_first_try = $storiesPassedFirstTry
            stories_passed_with_retry = $storiesPassedWithRetry
            stories_failed = $storiesFailed
            first_attempt_success_rate = $firstAttemptSuccessRate
            overall_success_rate = $overallSuccessRate
            avg_duration_minutes = $avgDurationMinutes
            healing_sessions = $healingSessions
            regressions_caught = $regressionsCaught
        }
        comparison = $comparison
        flag_impact = $flagImpact
    }

    # 8. Write to session/diagnostics.json
    try {
        $diagFile = if ($script:Paths) { Join-Path $script:Paths.SessionDir "diagnostics.json" } else { Join-Path $script:RalphDir "session\diagnostics.json" }
        Write-JsonNoBom -Path $diagFile -Content ($diagnostics | ConvertTo-Json -Depth 5)
    } catch {
        Write-Host "  Warning: Could not write diagnostics.json: $_" -ForegroundColor Yellow
    }

    return $diagnostics
}

# ============================================================================
# FAST-FAIL DETECTION
# ============================================================================

# ============================================================================
# GIT/ANALYSIS FUNCTIONS (moved from ralph.ps1)
# ============================================================================

function Confirm-CommitMatchesStory {
    <#
    .SYNOPSIS
        LLM-verify that git commits semantically match their user stories
    .DESCRIPTION
        Sends a single LLM call with all candidate story/commit pairs.
        The LLM determines if each commit actually implements the described story
        (not just a coincidental ID match from a different sprint).
    .PARAMETER Candidates
        Array of hashtables with: storyId, storyTitle, commitMsg
    .RETURNS
        Array of story IDs that the LLM confirms as genuine matches
    #>
    param(
        [array]$Candidates = @()
    )

    if (-not $Candidates -or $Candidates.Count -eq 0) { return @() }

    $claudePath = Get-ClaudePath
    if (-not $claudePath) {
        Write-Host "    Pre-flight: Claude not available, skipping LLM verification" -ForegroundColor DarkYellow
        return @()
    }

    # Build verification prompt
    $pairsList = ""
    foreach ($c in $Candidates) {
        $pairsList += "- $($c.storyId): Story=`"$($c.storyTitle)`" | Commit=`"$($c.commitMsg)`"`n"
    }

    $prompt = @"
You are verifying if git commits implement specific user stories.
IMPORTANT: Different sprints reuse story IDs (US-001, US-002, etc.), so the commit might be from a DIFFERENT sprint with COMPLETELY DIFFERENT work despite having the same ID.

A commit matches a story ONLY if the commit message describes the SAME WORK as the story title.
Example MATCH: Story="Add retry logic to downloader" | Commit="feat: [US-003] Add retry logic to downloader"
Example NO MATCH: Story="Add retry logic to downloader" | Commit="feat: [US-003] Wire impersonation into core.py"

For each pair, reply MATCH or NO_MATCH followed by the story ID. Nothing else.

$pairsList
"@

    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model haiku"
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        # Async output capture (prevents pipe buffer deadlock)
        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            $process.StandardInput.Write($prompt)
            $process.StandardInput.Close()

            # 60 second timeout for simple classification
            # NOTE: Do NOT use $process.WaitForExit($ms) — deadlocks on .NET Framework
            # when child processes inherit stdout/stderr pipe handles. Poll HasExited instead.
            $deadline = (Get-Date).AddSeconds(60)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $completed = $process.HasExited

            if (-not $completed) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                Write-Host "    Pre-flight: LLM verification timed out" -ForegroundColor DarkYellow
                return @()
            }

            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()

            # Parse response for MATCH lines (exclude NO_MATCH)
            $matchedIds = @()
            $candidateIds = @($Candidates | ForEach-Object { $_.storyId })
            foreach ($line in ($output -split "`n")) {
                $trimmed = $line.Trim()
                # Skip NO_MATCH lines, then check for MATCH
                if ($trimmed -match '^NO_MATCH') { continue }
                if ($trimmed -match 'MATCH\s+(US-\d+)') {
                    $id = $Matches[1]
                    # Only accept IDs that are actual candidates (safety)
                    if ($id -in $candidateIds) {
                        $matchedIds += $id
                    }
                }
            }

            return $matchedIds
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
            if ($process) { $process.Dispose() }
        }
    }
    catch {
        Write-Host "    Pre-flight: LLM verification failed: $_" -ForegroundColor DarkYellow
        return @()
    }
}

function Invoke-StoryRefinement {
    <#
    .SYNOPSIS
        Generate a more specific follow-on story after preflight auto-completes a generic one
    .DESCRIPTION
        When preflight detects a story was already completed via git commit, this function
        uses LLM to analyze what was done and generate a more specific, refined follow-on
        story to replace the completed slot in the sprint.
    .PARAMETER CompletedStory
        The story object that was auto-completed
    .PARAMETER CommitMessage
        The git commit message that completed the story
    .PARAMETER SprintContext
        The sprint's projectContext for additional context
    .RETURNS
        Hashtable with new story fields (id, title, priority, acceptanceCriteria) or $null if refinement fails
    #>
    param(
        [Parameter(Mandatory)][object]$CompletedStory,
        [Parameter(Mandatory)][string]$CommitMessage,
        [string]$SprintContext = ""
    )

    $claudePath = Get-ClaudePath
    if (-not $claudePath) {
        Write-Host "    Refinement: Claude not available, skipping" -ForegroundColor DarkYellow
        return $null
    }

    # Generate suffixed ID for follow-on story
    $baseId = $CompletedStory.id
    $newId = "$baseId-A"

    # Build refinement prompt
    $acList = if ($CompletedStory.acceptanceCriteria) {
        ($CompletedStory.acceptanceCriteria | ForEach-Object { "- $_" }) -join "`n"
    } else { "- (none specified)" }

    $prompt = @"
You are a sprint planning assistant. A user story was auto-completed by detecting an existing git commit.
Generate a MORE SPECIFIC follow-on story that builds on what was done.

COMPLETED STORY:
- ID: $($CompletedStory.id)
- Title: $($CompletedStory.title)
- Acceptance Criteria:
$acList

GIT COMMIT THAT COMPLETED IT:
$CommitMessage

SPRINT CONTEXT:
$SprintContext

Generate a follow-on story that:
1. Is more specific than the original (deeper implementation, tests, integration, etc.)
2. Builds naturally on what was completed
3. Has 3-5 concrete acceptance criteria
4. Is achievable in a single sprint iteration

Reply in EXACTLY this format (nothing else):
TITLE: <concise title for the follow-on story>
AC1: <first acceptance criterion>
AC2: <second acceptance criterion>
AC3: <third acceptance criterion>
AC4: <optional fourth criterion>
AC5: <optional fifth criterion>
"@

    try {
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = "--print --dangerously-skip-permissions --model haiku"
        $psi.WorkingDirectory = $script:ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi

        $outBuilder = [System.Text.StringBuilder]::new()
        $errBuilder = [System.Text.StringBuilder]::new()

        $outHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }
        $errHandler = { if (-not [string]::IsNullOrEmpty($EventArgs.Data)) { $Event.MessageData.AppendLine($EventArgs.Data) } }

        $outEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outHandler -MessageData $outBuilder
        $errEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errHandler -MessageData $errBuilder

        try {
            $process.Start() | Out-Null
            $process.BeginOutputReadLine()
            $process.BeginErrorReadLine()

            $process.StandardInput.Write($prompt)
            $process.StandardInput.Close()

            # 90 second timeout for story generation
            $deadline = (Get-Date).AddSeconds(90)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $completed = $process.HasExited

            if (-not $completed) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
                Write-Host "    Refinement: LLM timed out" -ForegroundColor DarkYellow
                return $null
            }

            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()

            # Parse response
            $title = $null
            $criteria = @()
            foreach ($line in ($output -split "`n")) {
                $trimmed = $line.Trim()
                if ($trimmed -match '^TITLE:\s*(.+)$') {
                    $title = $Matches[1].Trim()
                }
                elseif ($trimmed -match '^AC\d+:\s*(.+)$') {
                    $ac = $Matches[1].Trim()
                    if ($ac -and $ac -ne "(optional)" -and $ac -notmatch '^\(optional') {
                        $criteria += $ac
                    }
                }
            }

            if (-not $title -or $criteria.Count -lt 2) {
                Write-Host "    Refinement: Could not parse LLM response" -ForegroundColor DarkYellow
                return $null
            }

            # Build new story object
            $newStory = @{
                id = $newId
                title = $title
                priority = $CompletedStory.priority
                passes = $false
                notes = "Auto-generated follow-on from $($CompletedStory.id)"
                acceptanceCriteria = $criteria
            }

            Write-Host "    Refinement: Generated follow-on $newId" -ForegroundColor Cyan
            Write-Host "      Title: $title" -ForegroundColor DarkCyan
            return $newStory
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
            if ($process) { $process.Dispose() }
        }
    }
    catch {
        Write-Host "    Refinement: Failed - $_" -ForegroundColor DarkYellow
        return $null
    }
}

function Test-StoryAlreadyCommitted {
    <#
    .SYNOPSIS
        Per-story pre-flight check (used as guard in Invoke-ClaudeForStory)
    .DESCRIPTION
        Checks if a story was already committed by finding the commit via ID match,
        then LLM-verifying the commit semantically matches the story title.
        Falls back conservatively (returns false) if LLM is unavailable.
    .PARAMETER StoryId
        Story identifier (e.g., US-005)
    .PARAMETER Story
        Story object from PRD (required for verification)
    .RETURNS
        $true if story is confirmed already committed, $false otherwise
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$StoryId,
        [object]$Story = $null
    )

    if (-not $Story -or -not $Story.title) {
        return $false
    }

    try {
        $gitLog = git log --oneline --all --grep="\[$StoryId\]" 2>$null
        if (-not $gitLog) {
            $gitLog = git log --oneline --all --grep="($StoryId)" 2>$null
        }
        if (-not $gitLog) {
            return $false
        }

        $commitLine = ($gitLog -split "`n" | Where-Object { $_ } | Select-Object -First 1)

        # Fast path: if commit is a "mark as complete" chore commit, skip LLM verification
        if ($commitLine -match 'Mark as complete|pre-implemented') {
            Write-Host "    Pre-flight: $StoryId has pre-implemented commit - auto-completing" -ForegroundColor Green
            Write-Host "      $commitLine" -ForegroundColor DarkCyan
            return $true
        }

        # LLM-verify the match
        $matchedIds = Confirm-CommitMatchesStory -Candidates @(
            @{
                storyId    = $StoryId
                storyTitle = $Story.title
                commitMsg  = $commitLine
            }
        )

        if ($StoryId -in $matchedIds) {
            Write-Host "    Pre-flight: LLM confirmed $StoryId already committed" -ForegroundColor Green
            Write-Host "      $commitLine" -ForegroundColor DarkCyan
            return $true
        }
        else {
            Write-Host "    Pre-flight: commit for $StoryId is from a different sprint - skipping" -ForegroundColor DarkYellow
        }
    }
    catch {
        Write-Host "    Pre-flight: git check failed, proceeding normally" -ForegroundColor DarkYellow
    }

    return $false
}

function Get-GitDiffStats {
    <#
    .SYNOPSIS
    Gets lines added/deleted since last commit using git diff
    #>

    try {
        # Get diff stats for staged and unstaged changes
        $diffOutput = git diff --numstat HEAD~1 2>$null

        if (-not $diffOutput) {
            return @{ Added = 0; Deleted = 0 }
        }

        $totalAdded = 0
        $totalDeleted = 0

        foreach ($line in $diffOutput -split "`n") {
            if ($line -match '^(\d+)\s+(\d+)\s+') {
                $totalAdded += [int]$Matches[1]
                $totalDeleted += [int]$Matches[2]
            }
        }

        return @{ Added = $totalAdded; Deleted = $totalDeleted }
    }
    catch {
        return @{ Added = 0; Deleted = 0 }
    }
}

function Get-TestResults {
    <#
    .SYNOPSIS
        Parse pytest output to extract pass/fail counts
    .PARAMETER Output
        The output text to analyze
    .RETURNS
        String like "41/41 pass" or "38/41 pass, 3 fail", or empty string if no test results found
    #>
    param(
        [string]$Output
    )

    if (-not $Output) {
        return ""
    }

    $passed = 0
    $failed = 0

    # Pattern: "X passed" or "X passed,"
    if ($Output -match '(\d+)\s+passed') {
        $passed = [int]$Matches[1]
    }

    # Pattern: "X failed"
    if ($Output -match '(\d+)\s+failed') {
        $failed = [int]$Matches[1]
    }

    # Pattern: "X error" (collection errors)
    if ($Output -match '(\d+)\s+error') {
        $failed += [int]$Matches[1]
    }

    $total = $passed + $failed

    if ($total -eq 0) {
        return ""  # No test results found
    }

    if ($failed -eq 0) {
        return "$passed/$total pass"
    }
    else {
        return "$passed/$total pass, $failed fail"
    }
}

function Get-ErrorCategory {
    <#
    .SYNOPSIS
        Categorize error type from Claude output
    .PARAMETER Output
        The output text to analyze
    .PARAMETER TimedOut
        Whether the iteration timed out
    .RETURNS
        Error category string
    #>
    param(
        [string]$Output,
        [bool]$TimedOut
    )

    if ($TimedOut) { return "Timeout" }
    if ($Output -match "SyntaxError|parse error|unexpected token") { return "SyntaxError" }
    if ($Output -match "FAILED|AssertionError|test.*failed") { return "TestFailure" }
    if ($Output -match "cannot be loaded|compilation|ImportError") { return "CompileError" }
    if ($Output -match "ValidationError|schema") { return "ValidationError" }
    if ($Output -match "API|rate.?limit|quota|429") { return "APIError" }
    return "Unknown"
}

function Get-EstimatedTokens {
    <#
    .SYNOPSIS
        Estimate token count from output length (Story 1.8: improved estimation)
    .DESCRIPTION
        First tries to parse actual token usage from Claude's stderr output.
        Falls back to character-based estimation (~4 chars per token).
    .PARAMETER Output
        The output text to estimate tokens from
    .RETURNS
        Estimated token count (capped at 200000)
    #>
    param(
        [string]$Output
    )

    if (-not $Output -or $Output.Length -eq 0) {
        return 0
    }

    # Try to parse actual token usage from Claude output
    # Claude CLI may output usage info like "Input tokens: 1234" or "Total tokens: 5678"
    if ($Output -match 'total[_\s]?tokens[:\s]+(\d+)') {
        return [int]$Matches[1]
    }
    if ($Output -match 'input[_\s]?tokens[:\s]+(\d+)') {
        $inputTokens = [int]$Matches[1]
        # Also check for output tokens
        $outputTokens = 0
        if ($Output -match 'output[_\s]?tokens[:\s]+(\d+)') {
            $outputTokens = [int]$Matches[1]
        }
        return $inputTokens + $outputTokens
    }

    # Fallback: estimate from output length (~4 chars per token)
    $estimatedTokens = [math]::Round($Output.Length / 4)
    # Cap at reasonable max for a single interaction
    return [math]::Min($estimatedTokens, 200000)
}
