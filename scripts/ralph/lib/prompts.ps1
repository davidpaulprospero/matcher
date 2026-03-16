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

function Compress-FailureContext {
    <#
    .SYNOPSIS
        Compress verbose failure context into compact format under budget pressure
    .DESCRIPTION
        Extracts key information from Get-StoryFailureContext output:
        retry header (3 lines), error category, unmet criteria (single line),
        baseline info, last 5 output lines. Passes through unchanged if already
        under MaxChars budget.
    .PARAMETER FailureContext
        The failure context string from Get-StoryFailureContext
    .PARAMETER MaxChars
        Maximum character budget for the compressed output
    .RETURNS
        Compressed failure context string
    #>
    param(
        [string]$FailureContext,
        [int]$MaxChars = 2000
    )

    if (-not $FailureContext) { return "" }
    if ($FailureContext.Length -le $MaxChars) { return $FailureContext }

    $lines = $FailureContext -split "`n"
    $compressed = @()

    # Extract retry header (first 3 non-empty lines)
    $headerCount = 0
    foreach ($line in $lines) {
        if ($line.Trim() -and $headerCount -lt 3) {
            $compressed += $line
            $headerCount++
        }
        if ($headerCount -ge 3) { break }
    }

    # Extract error category line
    foreach ($line in $lines) {
        if ($line -match 'error.?category|Error Category|error_category') {
            $compressed += $line
            break
        }
    }

    # Extract unmet criteria as single line
    $unmetCriteria = @()
    foreach ($line in $lines) {
        if ($line -match '^\s*\[ \]' -or $line -match 'unmet|NOT met|failed criterion') {
            $unmetCriteria += ($line.Trim() -replace '^\s*\[ \]\s*', '')
        }
    }
    if ($unmetCriteria.Count -gt 0) {
        $compressed += "Unmet criteria: $($unmetCriteria -join '; ')"
    }

    # Extract baseline info
    foreach ($line in $lines) {
        if ($line -match 'baseline|pre-story|test.*pass') {
            $compressed += $line
            break
        }
    }

    # Last 5 lines of output (tail section)
    $outputLines = @($lines | Where-Object { $_.Trim() })
    if ($outputLines.Count -gt 5) {
        $compressed += ""
        $compressed += "Last output:"
        $compressed += $outputLines[-5..-1]
    }

    $result = $compressed -join "`n"

    # Final safety truncation
    if ($result.Length -gt $MaxChars) {
        $result = $result.Substring(0, $MaxChars - 30) + "`n[Compressed context truncated]"
    }

    return $result
}

function Build-BudgetedPrompt {
    <#
    .SYNOPSIS
        Assemble prompt sections with priority-based pruning
    .DESCRIPTION
        Each section is a hashtable with name, content, priority (1-4), displayOrder.
        Three-phase pruning when over budget:
        1. Drop P4 sections entirely
        2. Compress P2 sections (failure context via Compress-FailureContext)
        3. Trim P3 sections proportionally
        P1 is never touched.
    .PARAMETER Sections
        Array of section hashtables: @{ name; content; priority; displayOrder }
    .PARAMETER MaxLength
        Maximum prompt length in characters
    .PARAMETER BudgetAllocation
        Budget allocation hashtable (optional, for future use)
    .RETURNS
        Assembled prompt string within budget
    #>
    param(
        [array]$Sections,
        [int]$MaxLength = 10000,
        [hashtable]$BudgetAllocation = @{}
    )

    # Filter empty sections
    $validSections = @($Sections | Where-Object { $_.content -and $_.content.Trim() })

    if ($validSections.Count -eq 0) { return "" }

    # Calculate total length
    $totalLength = ($validSections | ForEach-Object { $_.content.Length } | Measure-Object -Sum).Sum

    # If under budget, join in displayOrder and return
    if ($totalLength -le $MaxLength) {
        $ordered = $validSections | Sort-Object { $_.displayOrder }
        return ($ordered | ForEach-Object { $_.content }) -join "`n"
    }

    # Phase 1: Drop P4 sections
    $remaining = @($validSections | Where-Object { $_.priority -le 3 })
    $totalLength = ($remaining | ForEach-Object { $_.content.Length } | Measure-Object -Sum).Sum

    if ($totalLength -le $MaxLength) {
        $ordered = $remaining | Sort-Object { $_.displayOrder }
        return ($ordered | ForEach-Object { $_.content }) -join "`n"
    }

    # Phase 2: Compress P2 sections (failure context)
    foreach ($section in $remaining) {
        if ($section.priority -eq 2) {
            $section.content = Compress-FailureContext -FailureContext $section.content -MaxChars 2000
        }
    }
    $totalLength = ($remaining | ForEach-Object { $_.content.Length } | Measure-Object -Sum).Sum

    if ($totalLength -le $MaxLength) {
        $ordered = $remaining | Sort-Object { $_.displayOrder }
        return ($ordered | ForEach-Object { $_.content }) -join "`n"
    }

    # Phase 3: Trim P3 sections proportionally
    $p1Length = ($remaining | Where-Object { $_.priority -eq 1 } | ForEach-Object { $_.content.Length } | Measure-Object -Sum).Sum
    $p2Length = ($remaining | Where-Object { $_.priority -eq 2 } | ForEach-Object { $_.content.Length } | Measure-Object -Sum).Sum
    $p3Sections = @($remaining | Where-Object { $_.priority -eq 3 })

    $availableForP3 = $MaxLength - $p1Length - $p2Length
    if ($availableForP3 -lt 0) { $availableForP3 = 0 }

    if ($p3Sections.Count -gt 0 -and $availableForP3 -gt 0) {
        $budgetPerSection = [math]::Floor($availableForP3 / $p3Sections.Count)
        foreach ($section in $p3Sections) {
            if ($section.content.Length -gt $budgetPerSection -and $budgetPerSection -gt 50) {
                $section.content = $section.content.Substring(0, $budgetPerSection - 30) + "`n[Section trimmed]"
            }
        }
    }
    elseif ($availableForP3 -le 0) {
        # No space for P3 — drop them
        $remaining = @($remaining | Where-Object { $_.priority -lt 3 })
    }

    # Assemble in display order
    $ordered = $remaining | Sort-Object { $_.displayOrder }
    $result = ($ordered | ForEach-Object { $_.content }) -join "`n"

    # Final safety truncation fallback
    if ($result.Length -gt $MaxLength) {
        $result = $result.Substring(0, $MaxLength - 50) + "`n`n[Prompt truncated to $MaxLength chars]"
    }

    return $result
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

    # Provider-specific preamble (e.g., Codex instructions)
    $provider = Get-AgentProvider
    $providerInstructions = Get-ProviderInstructions -Provider $provider

    # Check for adaptive pruning flag
    $useBudgeting = $config.flags -and $config.flags.adaptivePruning

    if ($useBudgeting) {
        # === BUDGETED PATH: Build tagged sections, then assemble with priority-based pruning ===
        $sections = @()

        # Section: Provider instructions — P1 (never trimmed), displayOrder -1
        if ($providerInstructions) {
            $sections += @{ name = "providerInstructions"; content = $providerInstructions; priority = 1; displayOrder = -1 }
        }

        # Section: Failure context (on retries) — P2, displayOrder 0
        if ($recommendation.useFailureContext -and $RetryCount -gt 0) {
            $failureContext = Get-StoryFailureContext -StoryId $StoryId -RetryCount $RetryCount
            if ($failureContext) {
                $sections += @{ name = "failureContext"; content = $failureContext; priority = 2; displayOrder = 0 }
            }
        }

        # Section: Human feedback — P3, displayOrder 1
        if ($recommendation.useFeedback) {
            $feedbackContext = Get-FeedbackForStory -StoryId $StoryId -Story $Story -FocusArea $FocusArea
            if ($feedbackContext) {
                $sections += @{ name = "humanFeedback"; content = $feedbackContext; priority = 3; displayOrder = 1 }
            }
        }

        # Section: Retrospective context — P4, displayOrder 2
        if ($recommendation.useRetrospective) {
            $retroFile = if ($script:Paths) { $script:Paths.LastRetrospectiveFile } else { Join-Path $script:RalphDir "state\last_retrospective.json" }
            $retro = Read-JsonFile -Path $retroFile
            if ($retro) {
                $retroContext = Get-RetrospectiveContext -Retro @{
                    lessons = @($retro.lessons)
                    failurePatterns = @($retro.failurePatterns)
                    recommendations = @($retro.recommendations)
                }
                if ($retroContext) {
                    $sections += @{ name = "retrospective"; content = $retroContext; priority = 4; displayOrder = 2 }
                }
            }
        }

        # Section: Conflict warnings — P4, displayOrder 3
        $conflictResult = Test-FileConflict -StoryId $StoryId -Story $Story
        if ($conflictResult -and $conflictResult.hasConflict) {
            $conflictContent = @("", "## File Conflict Warning",
                "The following files were recently modified by other stories: $($conflictResult.overlappingFiles -join ', ')",
                "Take extra care when modifying these files to avoid regressions.") -join "`n"
            $sections += @{ name = "conflictWarnings"; content = $conflictContent; priority = 4; displayOrder = 3 }
        }

        # Section: Relevant file hints — P3, displayOrder 4
        if ($recommendation.useFileHints -and $Story) {
            $relevantFiles = Get-RelevantFilesForStory -Story $Story -MaxFiles 8
            if ($relevantFiles.Count -gt 0) {
                $fileHintsContent = @("", "## Relevant Files (start here)") + ($relevantFiles | ForEach-Object { "- $_" })
                $sections += @{ name = "fileHints"; content = ($fileHintsContent -join "`n"); priority = 3; displayOrder = 4 }
            }
        }

        # Section: Sprint progress — P3, displayOrder 5
        $sprintProgress = Get-SprintProgressContext
        if ($sprintProgress) {
            $sections += @{ name = "sprintProgress"; content = $sprintProgress; priority = 3; displayOrder = 5 }
        }

        # Section: Learning injection — P3, displayOrder 6
        $learningWarnings = Get-LearningInjection -FocusArea $FocusArea
        if ($learningWarnings) {
            $sections += @{ name = "learningInjection"; content = $learningWarnings; priority = 3; displayOrder = 6 }
        }

        # Section: Resume context — P4, displayOrder 7
        if ($RetryCount -gt 0 -and $StoryId) {
            try {
                $storyProgress = Get-StoryProgress -StoryId $StoryId
                if ($storyProgress -and ($storyProgress.milestones.testsCreated -or $storyProgress.milestones.implementationStarted -or $storyProgress.milestones.committed)) {
                    $resumeContext = Build-ResumePrompt -StoryId $StoryId -Progress $storyProgress -Story $Story
                    if ($resumeContext) {
                        $sections += @{ name = "resumeContext"; content = $resumeContext; priority = 4; displayOrder = 7 }
                    }
                }
            }
            catch {}
        }

        # Section: Role specialization — P3, displayOrder 8
        $storyRole = Get-StoryRole -Story $Story
        if ($storyRole -ne 'feature') {
            $rolePrefix = switch ($storyRole) {
                'bugfix'      { "You are a debugging specialist. Reproduce the bug first with a failing test, then fix. Verify the fix resolves the issue and doesn't break existing tests." }
                'refactor'    { "You are a refactoring specialist. Preserve ALL existing behavior. Run the full test suite BEFORE and AFTER changes. If any test fails after your change, revert immediately." }
                'test'        { "You are a test engineer. Focus on edge cases, error paths, and mutation-resistant assertions. Tests should fail for the right reasons when code is broken." }
                'docs'        { "You are a documentation specialist. Be concise. Update only what's changed. Don't add boilerplate." }
                'performance' { "You are a performance specialist. Measure before and after. Include benchmark numbers in your commit message. No premature optimization." }
            }
            if ($rolePrefix) {
                $roleContent = "`n## Role: $($storyRole.ToUpper())`n$rolePrefix"
                $sections += @{ name = "roleSpecialization"; content = $roleContent; priority = 3; displayOrder = 8 }
            }
        }

        # Section: Story details — P1 (never trimmed), displayOrder 9
        $isSeedStory = $Story.title -and $Story.title -match 'Generate sprint stories'
        $storyParts = @()
        $storyParts += ""
        $storyParts += "============================================================"
        $storyParts += "STORY: $StoryId - $($Story.title)"
        $storyParts += "============================================================"
        $storyParts += ""

        if ($isSeedStory) {
            $storyParts += "INSTRUCTIONS:"
            $storyParts += "1. This is a SEED STORY that generates work items - do NOT implement code fixes"
            $storyParts += "2. Read scripts/ralph/state/prd.json first, then the config/context files in acceptance criteria"
            $storyParts += "3. Analyze the codebase and generate 8-12 stories as specified"
            $storyParts += "4. Write the COMPLETE updated prd.json with all new stories AND set this story's passes: true"
            $storyParts += "5. CRITICAL: The file must be actually written - verify by reading it back after writing"
            $storyParts += ""
            if ($Story.acceptanceCriteria) {
                $storyParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
                foreach ($criterion in $Story.acceptanceCriteria) { $storyParts += "  [ ] $criterion" }
                $storyParts += ""
            }
            if ($Story.notes) {
                $storyParts += "USER INTERVIEW NOTES (background context for generating stories - DO NOT implement these directly):"
                $storyParts += $Story.notes
                $storyParts += ""
            }
        }
        else {
            if ($Story.notes) {
                $storyParts += "CONTEXT: $($Story.notes)"
                $storyParts += ""
            }
            if ($Story.acceptanceCriteria) {
                $storyParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
                foreach ($criterion in $Story.acceptanceCriteria) { $storyParts += "  [ ] $criterion" }
                $storyParts += ""
            }
            $storyParts += "INSTRUCTIONS:"
            $storyParts += "1. All story details are above - DO NOT read prd.json (saves time)"
            $storyParts += "2. Read scripts/ralph/session/prompt.md for project-level instructions"
            $storyParts += "3. Implement the story, verify all acceptance criteria"
            $storyParts += "4. Update scripts/ralph/state/prd.json to set passes: true when complete"
        }

        $sections += @{ name = "storyDetails"; content = ($storyParts -join "`n"); priority = 1; displayOrder = 9 }

        # Assemble with budget allocation
        $budgetAlloc = @{}
        if ($config.prompts -and $config.prompts.budgetAllocation) {
            $ba = $config.prompts.budgetAllocation
            $budgetAlloc = @{
                storyDetails = if ($ba.storyDetails) { $ba.storyDetails } else { 0.40 }
                failureContext = if ($ba.failureContext) { $ba.failureContext } else { 0.20 }
                contextHints = if ($ba.contextHints) { $ba.contextHints } else { 0.20 }
                supplementary = if ($ba.supplementary) { $ba.supplementary } else { 0.20 }
            }
        }

        $prompt = Build-BudgetedPrompt -Sections $sections -MaxLength $maxLength -BudgetAllocation $budgetAlloc
        return $prompt
    }
    else {
        # === LEGACY PATH: Flat $promptParts array with naive truncation (existing behavior) ===
        $promptParts = @()

        # Section 0: Provider instructions (e.g., Codex-specific preamble)
        if ($providerInstructions) {
            $promptParts += $providerInstructions
        }

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
            $retroFile = if ($script:Paths) { $script:Paths.LastRetrospectiveFile } else { Join-Path $script:RalphDir "state\last_retrospective.json" }
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

        # Section 5.5: Sprint progress context (what previous sessions accomplished)
        $sprintProgress = Get-SprintProgressContext
        if ($sprintProgress) {
            $promptParts += $sprintProgress
        }

        # Section 5.7: Learning injection (known issues from past sprints)
        $learningWarnings = Get-LearningInjection -FocusArea $FocusArea
        if ($learningWarnings) {
            $promptParts += $learningWarnings
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

        # Detect seed stories early (affects how notes are presented)
        $isSeedStory = $Story.title -and $Story.title -match 'Generate sprint stories'

        # Role-based story specialization
        $storyRole = Get-StoryRole -Story $Story
        if ($storyRole -ne 'feature') {
            $rolePrefix = switch ($storyRole) {
                'bugfix'      { "You are a debugging specialist. Reproduce the bug first with a failing test, then fix. Verify the fix resolves the issue and doesn't break existing tests." }
                'refactor'    { "You are a refactoring specialist. Preserve ALL existing behavior. Run the full test suite BEFORE and AFTER changes. If any test fails after your change, revert immediately." }
                'test'        { "You are a test engineer. Focus on edge cases, error paths, and mutation-resistant assertions. Tests should fail for the right reasons when code is broken." }
                'docs'        { "You are a documentation specialist. Be concise. Update only what's changed. Don't add boilerplate." }
                'performance' { "You are a performance specialist. Measure before and after. Include benchmark numbers in your commit message. No premature optimization." }
            }
            if ($rolePrefix) {
                $promptParts += ""
                $promptParts += "## Role: $($storyRole.ToUpper())"
                $promptParts += $rolePrefix
            }
        }

        # Section 7: ALWAYS include full story details in prompt (prevents exploration stalls)
        $promptParts += ""
        $promptParts += "============================================================"
        $promptParts += "STORY: $StoryId - $($Story.title)"
        $promptParts += "============================================================"
        $promptParts += ""

        # For seed stories: instructions and criteria FIRST (truncation-safe),
        # notes last (background context that can be safely trimmed).
        # For regular stories: notes first (implementation context), then criteria, then instructions.
        if ($isSeedStory) {
            # Instructions first - these MUST survive truncation
            $promptParts += "INSTRUCTIONS:"
            $promptParts += "1. This is a SEED STORY that generates work items - do NOT implement code fixes"
            $promptParts += "2. Read scripts/ralph/state/prd.json first, then the config/context files in acceptance criteria"
            $promptParts += "3. Analyze the codebase and generate 8-12 stories as specified"
            $promptParts += "4. Write the COMPLETE updated prd.json with all new stories AND set this story's passes: true"
            $promptParts += "5. CRITICAL: The file must be actually written - verify by reading it back after writing"
            $promptParts += ""

            if ($Story.acceptanceCriteria) {
                $promptParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
                foreach ($criterion in $Story.acceptanceCriteria) {
                    $promptParts += "  [ ] $criterion"
                }
                $promptParts += ""
            }

            # Notes last - background context, safe to truncate
            if ($Story.notes) {
                $promptParts += "USER INTERVIEW NOTES (background context for generating stories - DO NOT implement these directly):"
                $promptParts += $Story.notes
                $promptParts += ""
            }
        }
        else {
            # Regular stories: notes as implementation context, then criteria, then instructions
            if ($Story.notes) {
                $promptParts += "CONTEXT: $($Story.notes)"
                $promptParts += ""
            }

            if ($Story.acceptanceCriteria) {
                $promptParts += "ACCEPTANCE CRITERIA (verify each before marking complete):"
                foreach ($criterion in $Story.acceptanceCriteria) {
                    $promptParts += "  [ ] $criterion"
                }
                $promptParts += ""
            }

            $promptParts += "INSTRUCTIONS:"
            $promptParts += "1. All story details are above - DO NOT read prd.json (saves time)"
            $promptParts += "2. Read scripts/ralph/session/prompt.md for project-level instructions"
            $promptParts += "3. Implement the story, verify all acceptance criteria"
            $promptParts += "4. Update scripts/ralph/state/prd.json to set passes: true when complete"
        }

        $prompt = $promptParts -join "`n"

        # Enforce max length
        if ($maxLength -gt 0 -and $prompt.Length -gt $maxLength) {
            $prompt = $prompt.Substring(0, $maxLength - 50) + "`n`n[Prompt truncated to $maxLength chars]"
        }

        return $prompt
    }
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

        $model = if ($script:Config.model) { $script:Config.model } else { "opus" }
        $claudeArgs = @("--print", "--dangerously-skip-permissions", "--model", $model)

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
            # NOTE: Do NOT use $process.WaitForExit($ms) — it deadlocks on .NET Framework
            # when child processes (Task/Explore node subagents) inherit stdout/stderr pipe
            # handles. Poll HasExited instead.
            $timeoutMs = 300000
            $deadline = (Get-Date).AddMilliseconds($timeoutMs)
            while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
                Start-Sleep -Milliseconds 500
            }
            $completed = $process.HasExited

            if (-not $completed) {
                # Kill entire process tree — $process.Kill() only kills parent,
                # leaving orphaned node subagents that hold pipes open forever
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) {
                    try { $process.Kill() } catch {}
                }
                Write-Host "  Exploration timed out after 5 minutes" -ForegroundColor Yellow
            }

            # Stop async readers and let final events flush
            try { $process.CancelOutputRead() } catch {}
            try { $process.CancelErrorRead() } catch {}
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
            Remove-Job -Job $outEvent -Force -ErrorAction SilentlyContinue
            Remove-Job -Job $errEvent -Force -ErrorAction SilentlyContinue
            if ($process -and -not $process.HasExited) {
                $treePid = $process.Id
                try { taskkill /T /F /PID $treePid 2>$null | Out-Null } catch {}
                if (-not $process.HasExited) { try { $process.Kill() } catch {} }
            }
            if ($process) { $process.Dispose() }
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

    # Invoke Claude - use claude.ps1 signature with required PromptType and Identifier
    # Note: FullExplore is handled via PromptType (full_exploration vs periodic_exploration)
    if ($FullExplore) {
        $result = Invoke-ClaudeExploration -Prompt $explorationPrompt -PromptType "full_exploration" -Identifier $FocusArea
    } else {
        $result = Invoke-ClaudeExploration -Prompt $explorationPrompt -PromptType "periodic_exploration" -Identifier $FocusArea
    }

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

function Update-SprintProgress {
    <#
    .SYNOPSIS
        Append a one-line entry to sprint_progress.md after each story.
        Keeps only the last $MaxEntries entries to prevent bloat.
    #>
    param(
        [Parameter(Mandatory)][string]$StoryId,
        [string]$StoryTitle = "",
        [Parameter(Mandatory)][bool]$Success,
        [string]$Summary = "",
        [int]$MaxEntries = 20
    )

    $progressFile = if ($script:Paths -and $script:Paths.SprintProgressFile) {
        $script:Paths.SprintProgressFile
    } else {
        Join-Path $script:RalphDir "session\sprint_progress.md"
    }

    $tag = if ($Success) { "DONE" } else { "FAIL" }
    $time = Get-Date -Format "HH:mm"
    $titlePart = if ($StoryTitle) { ": $StoryTitle" } else { "" }
    $summaryPart = if ($Summary -and $Summary -ne "completed" -and $Summary -ne "failed") { " ($Summary)" } else { "" }
    $entry = "- [$tag] $StoryId$titlePart$summaryPart [$time]"

    # Read existing entries
    $entries = @()
    if (Test-Path $progressFile) {
        $entries = @(Get-Content $progressFile -ErrorAction SilentlyContinue | Where-Object { $_.Trim() -match '^- \[' })
    }

    $entries += $entry

    # Truncate to last MaxEntries
    if ($entries.Count -gt $MaxEntries) {
        $entries = $entries[($entries.Count - $MaxEntries)..($entries.Count - 1)]
    }

    # Write file with header
    $content = @("# Sprint Progress", "") + $entries
    try {
        $content -join "`n" | Set-Content $progressFile -Encoding UTF8
    } catch {}
}

function Get-SprintProgressContext {
    <#
    .SYNOPSIS
        Return formatted sprint progress for prompt injection.
        Returns empty string if file doesn't exist or has no entries.
    #>
    $progressFile = if ($script:Paths -and $script:Paths.SprintProgressFile) {
        $script:Paths.SprintProgressFile
    } else {
        Join-Path $script:RalphDir "session\sprint_progress.md"
    }

    if (-not (Test-Path $progressFile)) { return "" }

    $content = Get-Content $progressFile -Raw -ErrorAction SilentlyContinue
    if (-not $content -or $content.Trim().Length -lt 20) { return "" }

    $entries = @($content -split "`n" | Where-Object { $_.Trim() -match '^- \[' })
    if ($entries.Count -eq 0) { return "" }

    $lines = @()
    $lines += ""
    $lines += "## Sprint Progress So Far"
    $lines += "Previous stories in this sprint:"
    $lines += $entries
    return ($lines -join "`n")
}

function Reset-SprintProgress {
    <#
    .SYNOPSIS
        Delete sprint_progress.md. Called at sprint start.
    #>
    $progressFile = if ($script:Paths -and $script:Paths.SprintProgressFile) {
        $script:Paths.SprintProgressFile
    } else {
        Join-Path $script:RalphDir "session\sprint_progress.md"
    }

    if (Test-Path $progressFile) {
        Remove-Item $progressFile -Force -ErrorAction SilentlyContinue
    }
}
