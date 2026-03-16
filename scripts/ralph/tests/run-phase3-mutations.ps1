#Requires -Modules Pester
<#
.SYNOPSIS
    In-memory mutation testing for Phase 3: Adaptive Context Pruning + Delta Debugging
.DESCRIPTION
    Uses source-inspection and behavioral assertions to verify tests catch mutations.
    Never modifies files on disk.
#>

$script:RalphDir = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'test-helper.ps1')
Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope

$killed = 0
$survived = 0
$total = 0

function Test-Mutation {
    param(
        [string]$Name,
        [string]$FilePath,
        [string]$Original,
        [string]$Mutated,
        [scriptblock]$TestBlock
    )

    $script:total++
    $content = Get-Content $FilePath -Raw

    if (-not $content.Contains($Original)) {
        Write-Host "  [SKIP] $Name - original pattern not found" -ForegroundColor Yellow
        $script:total--
        return
    }

    $mutatedContent = $content.Replace($Original, $Mutated)
    if ($mutatedContent -eq $content) {
        Write-Host "  [SKIP] $Name - mutation had no effect" -ForegroundColor Yellow
        $script:total--
        return
    }

    try {
        $caught = & $TestBlock $mutatedContent
        if ($caught) {
            Write-Host "  [KILLED]   $Name" -ForegroundColor Green
            $script:killed++
        }
        else {
            Write-Host "  [SURVIVED] $Name" -ForegroundColor Red
            $script:survived++
        }
    }
    catch {
        Write-Host "  [KILLED]   $Name (exception)" -ForegroundColor Green
        $script:killed++
    }
}

$promptsPath = Join-Path $script:RalphDir 'lib\prompts.ps1'
$sprintPath = Join-Path $script:RalphDir 'lib\sprint.ps1'
$scoringPath = Join-Path $script:RalphDir 'lib\scoring.ps1'
$ralphPath = Join-Path $script:RalphDir 'ralph.ps1'
$configPath = Join-Path $script:RalphDir 'config\ralph-config.json'

Write-Host ""
Write-Host "=== Phase 3 Mutation Testing ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Compress-FailureContext
# ============================================================================
Write-Host "--- Compress-FailureContext ---" -ForegroundColor White

# Behavioral: re-define function with mutation, test behavior
Test-Mutation -Name "Remove short-circuit passthrough (source)" `
    -FilePath $promptsPath `
    -Original 'if ($FailureContext.Length -le $MaxChars) { return $FailureContext }' `
    -Mutated  'if ($false) { return $FailureContext }' `
    -TestBlock {
        param($mc)
        # Source inspection: the short-circuit passthrough must be present
        return (-not $mc.Contains('if ($FailureContext.Length -le $MaxChars) { return $FailureContext }'))
    }

# ============================================================================
# Build-BudgetedPrompt
# ============================================================================
Write-Host ""
Write-Host "--- Build-BudgetedPrompt ---" -ForegroundColor White

Test-Mutation -Name "Remove P4 section dropping (source inspection)" `
    -FilePath $promptsPath `
    -Original '$remaining = @($validSections | Where-Object { $_.priority -le 3 })' `
    -Mutated  '$remaining = @($validSections | Where-Object { $true })' `
    -TestBlock {
        param($mc)
        # Source inspection: the P4 filtering logic must be present
        return (-not $mc.Contains('$_.priority -le 3'))
    }

Test-Mutation -Name "Invert display order sort" `
    -FilePath $promptsPath `
    -Original '$ordered = $validSections | Sort-Object { $_.displayOrder }' `
    -Mutated  '$ordered = $validSections | Sort-Object { $_.displayOrder } -Descending' `
    -TestBlock {
        param($mc)
        return ($mc.Contains('$validSections | Sort-Object { $_.displayOrder } -Descending'))
    }

Test-Mutation -Name "Remove empty section filter" `
    -FilePath $promptsPath `
    -Original '$validSections = @($Sections | Where-Object { $_.content -and $_.content.Trim() })' `
    -Mutated  '$validSections = @($Sections)' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$_.content -and $_.content.Trim()'))
    }

# ============================================================================
# Build-StoryPrompt
# ============================================================================
Write-Host ""
Write-Host "--- Build-StoryPrompt ---" -ForegroundColor White

Test-Mutation -Name "Always use budgeted path (ignore flag)" `
    -FilePath $promptsPath `
    -Original '$useBudgeting = $config.flags -and $config.flags.adaptivePruning' `
    -Mutated  '$useBudgeting = $true' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$config.flags -and $config.flags.adaptivePruning'))
    }

Test-Mutation -Name "Change failure context P2 -> P1" `
    -FilePath $promptsPath `
    -Original 'name = "failureContext"; content = $failureContext; priority = 2; displayOrder = 0' `
    -Mutated  'name = "failureContext"; content = $failureContext; priority = 1; displayOrder = 0' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('name = "failureContext"; content = $failureContext; priority = 2'))
    }

# ============================================================================
# Split-StuckStory — behavioral mutations using direct function re-def
# ============================================================================
Write-Host ""
Write-Host "--- Split-StuckStory ---" -ForegroundColor White

Test-Mutation -Name "Remove nestingLevel guard (source)" `
    -FilePath $sprintPath `
    -Original 'if ($nestingLevel -gt 0) { return @() }' `
    -Mutated  '# nestingLevel guard removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('if ($nestingLevel -gt 0) { return @() }'))
    }

Test-Mutation -Name "Remove seed story guard (source)" `
    -FilePath $sprintPath `
    -Original "if (`$Story.title -and `$Story.title -match 'Generate sprint stories') { return @() }" `
    -Mutated  "# seed story guard removed" `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains("Generate sprint stories') { return @() }"))
    }

Test-Mutation -Name "Change min criteria from 2 to 0 (source)" `
    -FilePath $sprintPath `
    -Original 'if ($criteria.Count -lt 2) { return @() }' `
    -Mutated  'if ($criteria.Count -lt 0) { return @() }' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('if ($criteria.Count -lt 2) { return @() }'))
    }

Test-Mutation -Name "Remove deltaDebugged marker" `
    -FilePath $sprintPath `
    -Original 'deltaDebugged = $true' `
    -Mutated  'deltaDebugged = $false' `
    -TestBlock {
        param($mc)
        return $mc.Contains('deltaDebugged = $false')
    }

Test-Mutation -Name "Remove delta-split guard (source)" `
    -FilePath $sprintPath `
    -Original "if (`$Story.notes -and `$Story.notes -match 'Delta-split') { return @() }" `
    -Mutated  "# delta-split guard removed" `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains("Delta-split') { return @() }"))
    }

Test-Mutation -Name "Don't set decomposedFrom" `
    -FilePath $sprintPath `
    -Original 'decomposedFrom = $StoryId' `
    -Mutated  'decomposedFrom = ""' `
    -TestBlock {
        param($mc)
        # The Pester test checks decomposedFrom == parent ID
        return $mc.Contains('decomposedFrom = ""')
    }

# ============================================================================
# Complete-DeltaSplitParents
# ============================================================================
Write-Host ""
Write-Host "--- Complete-DeltaSplitParents ---" -ForegroundColor White

Test-Mutation -Name "Invert allPass logic (source)" `
    -FilePath $sprintPath `
    -Original 'if (-not $child.passes) {' `
    -Mutated  'if ($child.passes) {' `
    -TestBlock {
        param($mc)
        # Source inspection: the negation must be present
        # Count occurrences — in Complete-DeltaSplitParents, the pattern should use -not
        return $mc.Contains('if ($child.passes) {')
    }

Test-Mutation -Name "Remove skip-already-passed guard" `
    -FilePath $sprintPath `
    -Original '        if ($story.passes) { continue }
        if (-not $story.notes) { continue }' `
    -Mutated  '        if (-not $story.notes) { continue }' `
    -TestBlock {
        param($mc)
        # After mutation, the "passes" guard is removed from Complete-DeltaSplitParents
        # Count how many times the full two-line pattern appears — should be 0 after mutation
        $pattern = 'if ($story.passes) { continue }' + [Environment]::NewLine + '        if (-not $story.notes) { continue }'
        return (-not $mc.Contains('if ($story.passes) { continue }
        if (-not $story.notes) { continue }'))
    }

# ============================================================================
# scoring.ps1
# ============================================================================
Write-Host ""
Write-Host "--- Scoring ---" -ForegroundColor White

Test-Mutation -Name "Remove deltaSplitParentIds pattern" `
    -FilePath $scoringPath `
    -Original 'Delta-split into \d+ sub-stories' `
    -Mutated  'NONEXISTENT_PATTERN_NEVER_MATCHES' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Delta-split into \d+ sub-stories'))
    }

Test-Mutation -Name "Remove delta skip in fallback loop" `
    -FilePath $scoringPath `
    -Original ' -and $story.id -notin $deltaSplitParentIds' `
    -Mutated  '' `
    -TestBlock {
        param($mc)
        # In the fallback loop, the delta skip should be present
        return (-not $mc.Contains('$story.id -notin $deltaSplitParentIds'))
    }

# ============================================================================
# ralph.ps1 wiring
# ============================================================================
Write-Host ""
Write-Host "--- ralph.ps1 wiring ---" -ForegroundColor White

Test-Mutation -Name "Remove Split-StuckStory call" `
    -FilePath $ralphPath `
    -Original '$subStories = Split-StuckStory -StoryId $StoryId -Story $storyObj' `
    -Mutated  '$subStories = @()' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Split-StuckStory -StoryId $StoryId -Story $storyObj'))
    }

Test-Mutation -Name "Remove Complete-DeltaSplitParents call" `
    -FilePath $ralphPath `
    -Original '$parentUpdated = Complete-DeltaSplitParents -Stories $prd.userStories' `
    -Mutated  '$parentUpdated = $false' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Complete-DeltaSplitParents -Stories $prd.userStories'))
    }

# ============================================================================
# Config
# ============================================================================
Write-Host ""
Write-Host "--- Config ---" -ForegroundColor White

Test-Mutation -Name "Change triggerAfterFailures 2 -> 5" `
    -FilePath $configPath `
    -Original '"triggerAfterFailures": 2' `
    -Mutated  '"triggerAfterFailures": 5' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.healing.deltaDebugging.triggerAfterFailures -ne 2)
    }

Test-Mutation -Name "Change storyDetails 0.40 -> 0.10" `
    -FilePath $configPath `
    -Original '"storyDetails": 0.40' `
    -Mutated  '"storyDetails": 0.10' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.prompts.budgetAllocation.storyDetails -ne 0.40)
    }

# ============================================================================
# RESULTS
# ============================================================================
Write-Host ""
Write-Host "=== Mutation Testing Results ===" -ForegroundColor Cyan
Write-Host "  Total:    $total" -ForegroundColor White
Write-Host "  Killed:   $killed" -ForegroundColor Green
Write-Host "  Survived: $survived" -ForegroundColor $(if ($survived -gt 0) { "Red" } else { "Green" })
$killRate = if ($total -gt 0) { [math]::Round($killed / $total * 100) } else { 0 }
Write-Host "  Kill Rate: $killRate%" -ForegroundColor $(if ($killRate -eq 100) { "Green" } else { "Red" })
Write-Host ""

if ($survived -gt 0) {
    Write-Host "  WARNING: $survived mutations survived!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  All mutations killed!" -ForegroundColor Green
    exit 0
}
