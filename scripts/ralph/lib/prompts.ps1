# scripts/ralph/lib/prompts.ps1
# Prompt building: story prompts, exploration, context injection, effectiveness

function Get-RelevantFilesForStory {
    <#
    .SYNOPSIS
        Find files relevant to a story based on keywords (Story 3.3)
    .DESCRIPTION
        Parses story title and acceptance criteria for keywords, searches the
        codebase for matching files. Returns file paths as "start here" hints.
    .PARAMETER Story
        Story object from PRD
    .PARAMETER MaxFiles
        Maximum files to return (default 10)
    .RETURNS
        Array of file paths relevant to the story
    #>
    param(
        [object]$Story,
        [int]$MaxFiles = 10
    )

    if (-not $Story) { return @() }

    # Extract keywords from story
    $storyText = ""
    if ($Story.title) { $storyText += $Story.title + " " }
    if ($Story.acceptanceCriteria) { $storyText += ($Story.acceptanceCriteria -join " ") }

    $stopWords = @('should', 'must', 'shall', 'that', 'when', 'with', 'from', 'have', 'this', 'been', 'were', 'they', 'each', 'which', 'their', 'will', 'would', 'could', 'into', 'more', 'also', 'than', 'only', 'other', 'does', 'such', 'make', 'ensure', 'test', 'create', 'update', 'implement', 'write', 'read', 'file', 'code', 'function')
    $keywords = @()
    foreach ($word in ($storyText.ToLower() -split '\W+')) {
        if ($word.Length -gt 3 -and $stopWords -notcontains $word) {
            $keywords += $word
        }
    }

    if ($keywords.Count -eq 0) { return @() }

    $relevantFiles = @{}

    # Search Python and PowerShell source files
    $extensions = @('*.py', '*.ps1', '*.yaml', '*.json')
    $searchDirs = @('src', 'tests', 'scripts/ralph', '.')

    foreach ($dir in $searchDirs) {
        $fullDir = Join-Path $script:ProjectRoot $dir
        if (-not (Test-Path $fullDir)) { continue }

        foreach ($ext in $extensions) {
            $files = Get-ChildItem -Path $fullDir -Filter $ext -Recurse -ErrorAction SilentlyContinue -Depth 3 |
                Where-Object { $_.FullName -notmatch '[\\/](\.git|node_modules|__pycache__|\.cache|\.venv|venv)[\\/]' }
            foreach ($file in $files) {
                $relPath = $file.FullName.Replace($script:ProjectRoot, '').TrimStart('\', '/')
                $score = 0

                # Check filename against keywords
                $fileName = $file.BaseName.ToLower()
                foreach ($kw in $keywords) {
                    if ($fileName -match [regex]::Escape($kw)) {
                        $score += 3
                    }
                }

                # Quick content scan (first 50 lines) for keyword matches
                if ($score -gt 0 -or $keywords.Count -le 5) {
                    try {
                        $content = Get-Content $file.FullName -TotalCount 50 -ErrorAction SilentlyContinue
                        if ($content) {
                            $contentText = ($content -join " ").ToLower()
                            foreach ($kw in $keywords) {
                                if ($contentText -match [regex]::Escape($kw)) {
                                    $score += 1
                                }
                            }
                        }
                    }
                    catch {}
                }

                if ($score -gt 0) {
                    $relevantFiles[$relPath] = $score
                }
            }
        }
    }

    # Sort by score descending, return top N
    $sorted = $relevantFiles.GetEnumerator() | Sort-Object { $_.Value } -Descending | Select-Object -First $MaxFiles
    return @($sorted | ForEach-Object { $_.Key })
}

function Get-PromptEffectivenessHistory {
    <#
    .SYNOPSIS
        Read prompt effectiveness data from session logs (Story 3.5)
    .DESCRIPTION
        Reads prompt_effectiveness.jsonl to identify which prompt patterns
        correlate with successful outcomes.
    .RETURNS
        Array of effectiveness records
    #>

    $records = @()

    if (-not $script:SessionLogDir -or -not (Test-Path $script:SessionLogDir)) {
        return $records
    }

    $effectivenessFile = Join-Path $script:SessionLogDir "prompt_effectiveness.jsonl"
    if (-not (Test-Path $effectivenessFile)) {
        return $records
    }

    try {
        $lines = Get-Content $effectivenessFile -ErrorAction SilentlyContinue
        foreach ($line in $lines) {
            if ($line.Trim()) {
                try {
                    $record = $line | ConvertFrom-Json
                    $records += $record
                }
                catch {}
            }
        }
    }
    catch {}

    return $records
}

function Get-PromptRecommendation {
    <#
    .SYNOPSIS
        Recommend prompt sections based on effectiveness history (Story 3.5)
    .DESCRIPTION
        Analyzes prompt_effectiveness.jsonl to determine which sections
        (failure context, feedback, criteria, retrospective) correlate
        with success and should be emphasized.
    .RETURNS
        Hashtable with section weights and recommendations
    #>

    $config = Get-RalphConfig
    $adaptiveEnabled = $config.prompts -and $config.prompts.adaptive

    if (-not $adaptiveEnabled) {
        return @{
            useFailureContext = $true
            useFeedback = $true
            useCriteria = $true
            useRetrospective = $true
            useFileHints = $true
            maxLength = if ($config.prompts -and $config.prompts.maxLength) { $config.prompts.maxLength } else { 4000 }
        }
    }

    $history = Get-PromptEffectivenessHistory

    $recommendation = @{
        useFailureContext = $true
        useFeedback = $true
        useCriteria = $true
        useRetrospective = $true
        useFileHints = $true
        maxLength = if ($config.prompts -and $config.prompts.maxLength) { $config.prompts.maxLength } else { 4000 }
    }

    if ($history.Count -lt 5) {
        # Not enough data - use defaults
        return $recommendation
    }

    # Analyze effectiveness by prompt type
    $typeStats = @{}
    foreach ($record in $history) {
        $type = if ($record.promptType) { $record.promptType } else { "unknown" }
        if (-not $typeStats.ContainsKey($type)) {
            $typeStats[$type] = @{ total = 0; totalEffectiveness = 0.0 }
        }
        $typeStats[$type].total++
        $effectiveness = if ($record.effectiveness) { [double]$record.effectiveness } else { 0.5 }
        $typeStats[$type].totalEffectiveness += $effectiveness
    }

    # Calculate average effectiveness per type
    foreach ($type in $typeStats.Keys) {
        $avg = $typeStats[$type].totalEffectiveness / $typeStats[$type].total
        if ($avg -lt 0.3 -and $typeStats[$type].total -ge 3) {
            # This prompt type is consistently ineffective
            Write-Host "  Adaptive: '$type' prompts averaging $([math]::Round($avg, 2)) effectiveness" -ForegroundColor DarkYellow
        }
    }

    return $recommendation
}

function Build-StoryPrompt {
    <#
    .SYNOPSIS
        Adaptive prompt builder for stories (Story 3.1)
    .DESCRIPTION
        Replaces static one-liner prompt with multi-section prompt builder.
        Combines: failure context, review feedback, retrospective, file hints,
        human feedback, conflict warnings, acceptance criteria.
        Respects prompt length limits and adaptive settings.
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Story object from PRD
    .PARAMETER FocusArea
        Focus area name
    .PARAMETER RetryCount
        Current retry count
    .RETURNS
        Formatted prompt string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [string]$FocusArea = "",
        [int]$RetryCount = 0
    )

    $config = Get-RalphConfig
    $recommendation = Get-PromptRecommendation
    $maxLength = $recommendation.maxLength
    $promptParts = @()

    # Section 1: Failure context (on retries)
    if ($recommendation.useFailureContext -and $RetryCount -gt 0) {
        $failureContext = Get-StoryFailureContext -StoryId $StoryId -RetryCount $RetryCount
        if ($failureContext) {
            $promptParts += $failureContext
        }
    }

    # Section 2: Human feedback
    if ($recommendation.useFeedback) {
        $feedbackContext = Get-FeedbackForStory -StoryId $StoryId -Story $Story -FocusArea $FocusArea
        if ($feedbackContext) {
            $promptParts += $feedbackContext
        }
    }

    # Section 3: Retrospective context
    if ($recommendation.useRetrospective) {
        $retroFile = Join-Path $script:RalphDir "last_retrospective.json"
        $retro = Read-JsonFile -Path $retroFile
        if ($retro) {
            $retroContext = Get-RetrospectiveContext -Retro @{
                lessons = @($retro.lessons)
                failurePatterns = @($retro.failurePatterns)
                recommendations = @($retro.recommendations)
            }
            if ($retroContext) {
                $promptParts += $retroContext
            }
        }
    }

    # Section 4: Conflict warnings
    $conflictResult = Test-FileConflict -StoryId $StoryId -Story $Story
    if ($conflictResult -and $conflictResult.hasConflict) {
        $promptParts += ""
        $promptParts += "## File Conflict Warning"
        $promptParts += "The following files were recently modified by other stories: $($conflictResult.overlappingFiles -join ', ')"
        $promptParts += "Take extra care when modifying these files to avoid regressions."
    }

    # Section 5: Relevant file hints
    if ($recommendation.useFileHints -and $Story) {
        $relevantFiles = Get-RelevantFilesForStory -Story $Story -MaxFiles 8
        if ($relevantFiles.Count -gt 0) {
            $promptParts += ""
            $promptParts += "## Relevant Files (start here)"
            foreach ($f in $relevantFiles) {
                $promptParts += "- $f"
            }
        }
    }

    # Section 6: Resume context from checkpoints (Phase 4, Story 4.1)
    if ($RetryCount -gt 0 -and $StoryId) {
        try {
            $storyProgress = Get-StoryProgress -StoryId $StoryId
            if ($storyProgress -and ($storyProgress.milestones.testsCreated -or $storyProgress.milestones.implementationStarted -or $storyProgress.milestones.committed)) {
                $resumeContext = Build-ResumePrompt -StoryId $StoryId -Progress $storyProgress -Story $Story
                if ($resumeContext) {
                    $promptParts += $resumeContext
                }
            }
        }
        catch {}
    }

    # Section 7: Acceptance criteria (when backpressure enabled)
    if ($recommendation.useCriteria -and $config.flags.acceptanceDrivenBackpressure -and $Story) {
        $promptParts += ""
        $promptParts += "STORY: $StoryId - $($Story.title)"
        $promptParts += ""
        if ($Story.acceptanceCriteria) {
            $promptParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
            foreach ($criterion in $Story.acceptanceCriteria) {
                $promptParts += "  [ ] $criterion"
            }
            $promptParts += ""
        }
        $promptParts += "Work on this story from scripts/ralph/prd.json. Read scripts/ralph/prompt.md for instructions."
        $promptParts += "Before marking the story as passed, verify EACH acceptance criterion above is met."
    }
    else {
        $promptParts += "Work on story $StoryId from scripts/ralph/prd.json. Read scripts/ralph/prompt.md for instructions."
    }

    $prompt = $promptParts -join "`n"

    # Enforce max length
    if ($maxLength -gt 0 -and $prompt.Length -gt $maxLength) {
        $prompt = $prompt.Substring(0, $maxLength - 50) + "`n`n[Prompt truncated to $maxLength chars]"
    }

    return $prompt
}

function Build-ResumePrompt {
    <#
    .SYNOPSIS
        Build a resume-aware prompt for partially completed stories (Story 4.1)
    .PARAMETER StoryId
        Story identifier
    .PARAMETER Story
        Story object
    .PARAMETER Progress
        Story progress from Get-StoryProgress
    .RETURNS
        Additional prompt context for resuming, or empty string
    #>
    param(
        [string]$StoryId,
        [object]$Story,
        [hashtable]$Progress
    )

    if (-not $Progress -or -not $Progress.milestones) { return "" }

    $milestones = $Progress.milestones
    $completedSteps = @()
    $remainingSteps = @()

    if ($milestones.testsCreated) { $completedSteps += "Tests already created" }
    else { $remainingSteps += "Create tests" }

    if ($milestones.implementationStarted) { $completedSteps += "Implementation started" }
    else { $remainingSteps += "Implement the feature" }

    if ($milestones.committed) { $completedSteps += "Changes committed" }
    else { $remainingSteps += "Commit changes" }

    if ($completedSteps.Count -eq 0) { return "" }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## Resume Context")
    [void]$sb.AppendLine("This story was partially completed in a previous attempt.")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("### Completed Steps")
    foreach ($step in $completedSteps) {
        [void]$sb.AppendLine("- [x] $step")
    }
    if ($remainingSteps.Count -gt 0) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("### Remaining Steps")
        foreach ($step in $remainingSteps) {
            [void]$sb.AppendLine("- [ ] $step")
        }
    }
    if ($Progress.lastCheckpoint) {
        [void]$sb.AppendLine("")
        [void]$sb.AppendLine("Last checkpoint: $($Progress.lastCheckpoint)")
    }

    return $sb.ToString()
}


function Invoke-ClaudeExploration {
    <#
    .SYNOPSIS
        Invoke Claude for exploration with read-only or full tools
    .PARAMETER Prompt
        The exploration prompt to send
    .PARAMETER FullExplore
        If set, uses full exploration tools (Task, Read, Glob, Grep, Bash, Write)
        Otherwise uses lightweight read-only tools (Read, Glob, Grep)
    .RETURNS
        The exploration output string
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,
        [switch]$FullExplore
    )

    $claudePath = Get-ClaudePath
    $tempFile = [System.IO.Path]::GetTempFileName()

    try {
        $Prompt | Out-File -FilePath $tempFile -Encoding UTF8 -NoNewline

        $claudeArgs = @("--print", "--dangerously-skip-permissions")

        if ($FullExplore) {
            # Full exploration with Task/Explore agent access
            $claudeArgs += "--allowedTools=Task,Read,Glob,Grep,Bash,Write"
        } else {
            # Light exploration - read-only, fast
            # Use haiku for speed/cost efficiency if configured
            $explorationConfig = $script:Config.exploration
            if ($explorationConfig -and $explorationConfig.periodic -and $explorationConfig.periodic.useHaiku) {
                $claudeArgs += "--model"
                $claudeArgs += "haiku"
            }
            $claudeArgs += "--allowedTools=Read,Glob,Grep"
        }

        # Create process
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $claudePath
        $psi.Arguments = $claudeArgs -join ' '
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

            # Send prompt via stdin
            $process.StandardInput.Write($Prompt)
            $process.StandardInput.Close()

            # Wait with timeout (5 minutes for exploration)
            $timeoutMs = 300000
            $completed = $process.WaitForExit($timeoutMs)

            if (-not $completed) {
                $process.Kill()
                Write-Host "  Exploration timed out after 5 minutes" -ForegroundColor Yellow
            }

            # Small delay to let async handlers flush
            Start-Sleep -Milliseconds 200

            $output = $outBuilder.ToString()
            $stderr = $errBuilder.ToString()

            if ($stderr) {
                $output += "`n$stderr"
            }

            return $output
        }
        finally {
            Unregister-Event -SourceIdentifier $outEvent.Name -ErrorAction SilentlyContinue
            Unregister-Event -SourceIdentifier $errEvent.Name -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-Host "  Exploration failed: $_" -ForegroundColor Red
        return ""
    }
    finally {
        Remove-Item $tempFile -ErrorAction SilentlyContinue
    }
}

function Invoke-FocusAreaExploration {
    <#
    .SYNOPSIS
        Perform exploration for a focus area and cache results
    .PARAMETER FocusArea
        The focus area ID to explore
    .PARAMETER Reason
        Why exploration is being triggered: "interval", "git_changes", "sprint_start"
    .PARAMETER FullExplore
        If set, uses full exploration (for sprint start)
    .RETURNS
        The exploration summary string
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$FocusArea,
        [Parameter(Mandatory=$true)]
        [string]$Reason,
        [switch]$FullExplore
    )

    $config = $script:Config

    # Get relevant files for this focus area from config
    $relevantFiles = @()
    if ($config.relevantFiles -and $config.relevantFiles.$FocusArea) {
        $relevantFiles = $config.relevantFiles.$FocusArea
    }

    # Get test patterns for this focus area
    $testPatterns = @()
    if ($config.testPatterns -and $config.testPatterns.$FocusArea) {
        $testPatterns = $config.testPatterns.$FocusArea
    }

    # Get recent git changes in these files
    $recentChanges = ""
    try {
        $filePatterns = $relevantFiles -join " "
        if ($filePatterns) {
            $recentChanges = git diff --stat HEAD~5 -- $relevantFiles 2>$null
            if (-not $recentChanges) { $recentChanges = "(no recent changes)" }
        }
    }
    catch {
        $recentChanges = "(could not retrieve git changes)"
    }

    # Build exploration prompt
    if ($FullExplore) {
        # Full sprint-start exploration
        $explorationPrompt = @"
MANDATORY EXPLORATION - Focus Area: $FocusArea
Reason: $Reason

You MUST explore this focus area before generating stories.

## Step 1: Explore relevant files
Use the Task tool with subagent_type=Explore to scan these files/patterns:
$($relevantFiles -join "`n")

## Step 2: Check recent changes
Run: git log --oneline -10 -- $($relevantFiles -join " ")

Recent diff stats:
$recentChanges

## Step 3: Run relevant tests to see current state
$(if ($testPatterns.Count -gt 0) { "Run: pytest tests/$($testPatterns[0]) -v --tb=short 2>&1 | head -50" } else { "No specific test patterns configured for this focus area" })

## Step 4: Summarize findings
After exploration, write a summary to: scripts/ralph/exploration_context.md

Include:
- Current implementation state (what exists, what's missing)
- Recent changes and their impact
- Test status (passing/failing, coverage gaps)
- Technical debt or issues noticed
- Patterns to follow when implementing stories

This exploration context will be used for PRD generation.
Keep the summary focused and actionable (under 1000 words).
"@
    }
    else {
        # Lightweight periodic exploration
        $explorationPrompt = @"
EXPLORATION PHASE - Focus Area: $FocusArea
Reason: $Reason

Quickly scan these files for the current state:
$($relevantFiles -join "`n")

Recent changes:
$recentChanges

Summarize in 3-5 bullet points:
- Current implementation state
- Any issues or TODOs noticed
- Key patterns to follow
- Anything that might affect upcoming stories

Keep response under 500 words. This is context-gathering, not implementation.
"@
    }

    Write-Host "  Running exploration ($Reason)..." -ForegroundColor Cyan

    # Invoke Claude
    $result = Invoke-ClaudeExploration -Prompt $explorationPrompt -FullExplore:$FullExplore

    # Cache the summary for use in story prompts
    $script:State.LastExplorationSummary = $result
    $script:State.LastExplorationTime = Get-Date

    # For sprint-start, also update the context file
    if ($FullExplore -and $result) {
        # Try to read what Claude wrote to the file
        if (Test-Path $script:ExplorationContextFile) {
            $script:State.SprintExplorationContext = Get-Content $script:ExplorationContextFile -Raw -ErrorAction SilentlyContinue
        }
        if (-not $script:State.SprintExplorationContext) {
            # Fall back to the output if file wasn't written
            $script:State.SprintExplorationContext = $result
        }
    }

    Write-Host "  Exploration complete" -ForegroundColor Green

    return $result
}

function Test-ShouldExplore {
    <#
    .SYNOPSIS
        Check if periodic exploration should be triggered
    .PARAMETER FocusArea
        The current focus area to check for git changes
    .RETURNS
        Hashtable with ShouldExplore (bool) and Reason (string)
    #>
    param(
        [string]$FocusArea
    )

    $config = $script:Config
    $explorationConfig = $config.exploration

    # Check if exploration is enabled
    if (-not $explorationConfig -or -not $explorationConfig.enabled) {
        return @{ ShouldExplore = $false; Reason = "exploration_disabled" }
    }

    if (-not $explorationConfig.periodic -or -not $explorationConfig.periodic.enabled) {
        return @{ ShouldExplore = $false; Reason = "periodic_disabled" }
    }

    # Trigger 1: Interval-based (every N stories)
    $intervalStories = $explorationConfig.periodic.intervalStories
    if (-not $intervalStories) { $intervalStories = 3 }

    if ($script:State.StoriesSinceExploration -ge $intervalStories) {
        return @{ ShouldExplore = $true; Reason = "interval" }
    }

    # Trigger 2: Git changes in focus area files (since last exploration)
    if ($explorationConfig.periodic.triggerOnGitChanges -and $FocusArea) {
        $relevantFiles = @()
        if ($config.relevantFiles -and $config.relevantFiles.$FocusArea) {
            $relevantFiles = $config.relevantFiles.$FocusArea
        }

        if ($relevantFiles.Count -gt 0 -and $script:State.LastExplorationCommit) {
            try {
                # Compare against the commit from last exploration, not HEAD~1
                # This prevents triggering on every iteration since Ralph commits after each story
                $changedFiles = git diff --name-only $script:State.LastExplorationCommit HEAD 2>$null
                if ($changedFiles) {
                    foreach ($changed in ($changedFiles -split "`n")) {
                        foreach ($pattern in $relevantFiles) {
                            if ($changed -like $pattern -or $changed -like "*$pattern*") {
                                return @{ ShouldExplore = $true; Reason = "git_changes" }
                            }
                        }
                    }
                }
            }
            catch {
                # Ignore git errors
            }
        }
    }

    return @{ ShouldExplore = $false; Reason = "not_needed" }
}

function Invoke-PeriodicExplorationIfNeeded {
    <#
    .SYNOPSIS
        Check and perform periodic exploration after story completion
    .PARAMETER FocusArea
        The current focus area
    .RETURNS
        $true if exploration was performed, $false otherwise
    #>
    param(
        [string]$FocusArea
    )

    # Increment story counter
    $script:State.StoriesSinceExploration++

    # Check if exploration needed
    $check = Test-ShouldExplore -FocusArea $FocusArea

    if ($check.ShouldExplore) {
        Write-Host ""
        Write-Host ">>> Periodic Exploration triggered ($($check.Reason))" -ForegroundColor Cyan
        Write-Host ""

        # Perform lightweight exploration
        $result = Invoke-FocusAreaExploration -FocusArea $FocusArea -Reason $check.Reason

        # Reset counter and update git baseline
        $script:State.StoriesSinceExploration = 0
        $script:State.LastExplorationCommit = (git rev-parse HEAD 2>$null)

        return $true
    }

    return $false
}
