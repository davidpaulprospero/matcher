#Requires -Modules Pester
<#
.SYNOPSIS
    In-memory mutation testing for Phases 0-2: Carlini-inspired improvements
    Phase 0: Sprint Diagnostics, Early-Exit Tuning, Structured Output, Fast Tests, Sprint Progress
    Phase 1: Oracle-Based Regression Guard + Deterministic Test Subsampling
    Phase 2: Feedback-Driven Learning Loop + Role-Based Story Specialization
.DESCRIPTION
    Uses source-inspection assertions to verify tests catch mutations.
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

$claudePath = Join-Path $script:RalphDir 'lib\claude.ps1'
$healingPath = Join-Path $script:RalphDir 'lib\healing.ps1'
$learningPath = Join-Path $script:RalphDir 'lib\learning.ps1'
$metricsPath = Join-Path $script:RalphDir 'lib\metrics.ps1'
$promptsPath = Join-Path $script:RalphDir 'lib\prompts.ps1'
$qualityPath = Join-Path $script:RalphDir 'lib\quality.ps1'
$sprintPath = Join-Path $script:RalphDir 'lib\sprint.ps1'
$pathsPath = Join-Path $script:RalphDir 'lib\paths.ps1'
$ralphPath = Join-Path $script:RalphDir 'ralph.ps1'
$configPath = Join-Path $script:RalphDir 'config\ralph-config.json'

Write-Host ""
Write-Host "=== Phase 0-2 Mutation Testing ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Phase 0: Early-Exit Tuning
# ============================================================================
Write-Host "--- Early-Exit Tuning ---" -ForegroundColor White

Test-Mutation -Name "Change grace period 30 -> 60 (config)" `
    -FilePath $configPath `
    -Original '"gracePeriodSeconds": 30' `
    -Mutated  '"gracePeriodSeconds": 60' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds -ne 30)
    }

Test-Mutation -Name "Remove git HEAD capture on story completion" `
    -FilePath $claudePath `
    -Original '$storyCompletionGitHead = (git rev-parse HEAD' `
    -Mutated  '$storyCompletionGitHead = $null  # MUTATED' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$storyCompletionGitHead = (git rev-parse HEAD'))
    }

Test-Mutation -Name "Remove git commit detection after story done" `
    -FilePath $claudePath `
    -Original '$currentHead -ne $storyCompletionGitHead' `
    -Mutated  '$false  # MUTATED' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$currentHead -ne $storyCompletionGitHead'))
    }

# ============================================================================
# Phase 0: Structured Test Output (Format-TestSummary)
# ============================================================================
Write-Host ""
Write-Host "--- Format-TestSummary ---" -ForegroundColor White

Test-Mutation -Name "Change failure limit 10 -> 100" `
    -FilePath $healingPath `
    -Original 'if ($failCount -ge 10)' `
    -Mutated  'if ($failCount -ge 100)' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('if ($failCount -ge 10)'))
    }

Test-Mutation -Name "Change error truncation 200 -> 2000" `
    -FilePath $healingPath `
    -Original 'if ($errorMsg.Length -gt 200) { $errorMsg = $errorMsg.Substring(0, 200)' `
    -Mutated  'if ($errorMsg.Length -gt 2000) { $errorMsg = $errorMsg.Substring(0, 2000)' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('if ($errorMsg.Length -gt 200)'))
    }

Test-Mutation -Name "Change collection error limit 5 -> 50" `
    -FilePath $healingPath `
    -Original 'if ($errCount -ge 5)' `
    -Mutated  'if ($errCount -ge 50)' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('if ($errCount -ge 5)'))
    }

Test-Mutation -Name "Remove ANSI stripping in Format-TestSummary" `
    -FilePath $healingPath `
    -Original ('-replace ''\x1b\[[0-9;]*m'', ''''') `
    -Mutated  ('#-replace ANSI removed') `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains("-replace '\x1b\[[0-9;]*m', ''"))
    }

# ============================================================================
# Phase 0: Retry Context Reduction (quality.ps1)
# ============================================================================
Write-Host ""
Write-Host "--- Retry Context Reduction ---" -ForegroundColor White

Test-Mutation -Name "Change output tail from 20 to 50 lines" `
    -FilePath $qualityPath `
    -Original '$tailLines = if ($lines.Count -gt 20) { $lines[-20..-1] } else { $lines }' `
    -Mutated  '$tailLines = if ($lines.Count -gt 50) { $lines[-50..-1] } else { $lines }' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$lines.Count -gt 20'))
    }

Test-Mutation -Name "Remove ANSI stripping from retry context" `
    -FilePath $qualityPath `
    -Original "`$cleanOutput = `$lastOutput -replace '\x1b\[[0-9;]*m', ''" `
    -Mutated  '$cleanOutput = $lastOutput' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains("`$lastOutput -replace '\x1b\[[0-9;]*m', ''"))
    }

# ============================================================================
# Phase 0: Sprint Progress Functions
# ============================================================================
Write-Host ""
Write-Host "--- Sprint Progress ---" -ForegroundColor White

Test-Mutation -Name "Change MaxEntries default from 20 to 5" `
    -FilePath $promptsPath `
    -Original '[int]$MaxEntries = 20' `
    -Mutated  '[int]$MaxEntries = 5' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('[int]$MaxEntries = 20'))
    }

Test-Mutation -Name "Remove DONE/FAIL tag in Update-SprintProgress" `
    -FilePath $promptsPath `
    -Original '$tag = if ($Success) { "DONE" } else { "FAIL" }' `
    -Mutated  '$tag = "LOG"' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$tag = if ($Success) { "DONE" } else { "FAIL" }'))
    }

Test-Mutation -Name "Remove entry filtering in Get-SprintProgressContext" `
    -FilePath $promptsPath `
    -Original '$entries = @($content -split "`n" | Where-Object { $_.Trim() -match ''^- \['' })' `
    -Mutated  '$entries = @($content -split "`n")' `
    -TestBlock {
        param($mc)
        # In Get-SprintProgressContext, the entry filtering must be present
        return (-not $mc.Contains("`$entries = @(`$content -split `"`n`" | Where-Object { `$_.Trim() -match '^- \[' })"))
    }

Test-Mutation -Name "Remove sprint progress section header" `
    -FilePath $promptsPath `
    -Original '## Sprint Progress So Far' `
    -Mutated  '## Progress' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('## Sprint Progress So Far'))
    }

Test-Mutation -Name "paths.ps1 PreStoryBaselineFile reference" `
    -FilePath $pathsPath `
    -Original 'PreStoryBaselineFile' `
    -Mutated  'PreBaselineFile' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('PreStoryBaselineFile'))
    }

# ============================================================================
# Phase 1: Oracle-Based Regression Guard
# ============================================================================
Write-Host ""
Write-Host "--- Oracle-Based Regression Guard ---" -ForegroundColor White

Test-Mutation -Name "Remove regressionGuard flag check" `
    -FilePath $healingPath `
    -Original 'if (-not $config.flags -or -not $config.flags.regressionGuard) { return $null }' `
    -Mutated  '# flag check removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$config.flags.regressionGuard'))
    }

Test-Mutation -Name "Remove pre-story baseline save" `
    -FilePath $healingPath `
    -Original 'Write-JsonNoBom -Path $baselineFile -Content ($baseline | ConvertTo-Json -Depth 3)' `
    -Mutated  '# baseline save removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Write-JsonNoBom -Path $baselineFile'))
    }

Test-Mutation -Name "Remove BaselineOverride param in Compare-TestBaseline" `
    -FilePath $qualityPath `
    -Original '$BaselineOverride' `
    -Mutated  '$BslnOvrd' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$BaselineOverride'))
    }

Test-Mutation -Name "Remove Test-RegressionBaseline call in ralph.ps1" `
    -FilePath $ralphPath `
    -Original 'Test-RegressionBaseline' `
    -Mutated  'Capture-PreStoryBsln' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Test-RegressionBaseline'))
    }

Test-Mutation -Name "Remove -BaselineOverride forwarding in claude.ps1" `
    -FilePath $claudePath `
    -Original '-BaselineOverride' `
    -Mutated  '-BaseOverrideRemoved' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('-BaselineOverride'))
    }

# ============================================================================
# Phase 1: Deterministic Test Subsampling
# ============================================================================
Write-Host ""
Write-Host "--- Deterministic Test Subsampling ---" -ForegroundColor White

Test-Mutation -Name "Remove testSubsampling flag check" `
    -FilePath $healingPath `
    -Original 'if (-not $config.flags -or -not $config.flags.testSubsampling) { return $null }' `
    -Mutated  '# flag check removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$config.flags.testSubsampling'))
    }

Test-Mutation -Name "Change seed hash algorithm (31 -> 37)" `
    -FilePath $healingPath `
    -Original 'foreach ($c in $StoryId.ToCharArray()) { $seed = ($seed * 31 + [int]$c) -band 0x7FFFFFFF }' `
    -Mutated  'foreach ($c in $StoryId.ToCharArray()) { $seed = ($seed * 37 + [int]$c) -band 0x7FFFFFFF }' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$seed * 31'))
    }

Test-Mutation -Name "Remove focusAreaScoped config check" `
    -FilePath $configPath `
    -Original '"focusAreaScoped": true' `
    -Mutated  '"focusAreaScoped": false' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.testSubsampling.focusAreaScoped -ne $true)
    }

Test-Mutation -Name "Remove StoryId/FocusArea forwarding in healing" `
    -FilePath $healingPath `
    -Original 'Invoke-TieredHealthCheck' `
    -Mutated  'Run-TieredHC' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Invoke-TieredHealthCheck'))
    }

# ============================================================================
# Phase 2: Feedback-Driven Learning Loop
# ============================================================================
Write-Host ""
Write-Host "--- Feedback-Driven Learning Loop ---" -ForegroundColor White

Test-Mutation -Name "Remove learningInjection flag check" `
    -FilePath $learningPath `
    -Original 'if (-not $config.flags -or -not $config.flags.learningInjection) { return "" }' `
    -Mutated  '# flag check removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$config.flags.learningInjection'))
    }

Test-Mutation -Name "Change min occurrence threshold 3 -> 1" `
    -FilePath $learningPath `
    -Original 'Where-Object { $_.Value -ge 3 }' `
    -Mutated  'Where-Object { $_.Value -ge 1 }' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$_.Value -ge 3'))
    }

Test-Mutation -Name "Remove learning section injection in prompts" `
    -FilePath $promptsPath `
    -Original 'Get-LearningInjection' `
    -Mutated  'Fetch-LearnWarn' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Get-LearningInjection'))
    }

Test-Mutation -Name "Remove injection effectiveness tracking" `
    -FilePath $claudePath `
    -Original 'injection_result' `
    -Mutated  'inj_tracking_off' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('injection_result'))
    }

# ============================================================================
# Phase 2: Role-Based Story Specialization
# ============================================================================
Write-Host ""
Write-Host "--- Role-Based Story Specialization ---" -ForegroundColor White

Test-Mutation -Name "Remove roleSpecialization flag check" `
    -FilePath $sprintPath `
    -Original "if (-not `$config.flags -or -not `$config.flags.roleSpecialization) { return 'feature' }" `
    -Mutated  "# flag check removed; return 'feature'" `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('$config.flags.roleSpecialization'))
    }

Test-Mutation -Name "Remove bugfix keyword match" `
    -FilePath $sprintPath `
    -Original "if (`$combined -match 'fix|bug|error|broken|crash|regression') { return 'bugfix' }" `
    -Mutated  "# bugfix match removed" `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains("return 'bugfix'"))
    }

Test-Mutation -Name "Change bugfix timeout 420 -> 600" `
    -FilePath $configPath `
    -Original '"bugfix":      { "timeout": 420' `
    -Mutated  '"bugfix":      { "timeout": 600' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.roles.bugfix.timeout -ne 420)
    }

Test-Mutation -Name "Change feature maxIterations 5 -> 10" `
    -FilePath $configPath `
    -Original '"feature":     { "timeout": 600,  "maxIterations": 5' `
    -Mutated  '"feature":     { "timeout": 600,  "maxIterations": 10' `
    -TestBlock {
        param($mc)
        $config = $mc | ConvertFrom-Json
        return ($config.roles.feature.maxIterations -ne 5)
    }

Test-Mutation -Name "Remove Role param from Record-Metric" `
    -FilePath $metricsPath `
    -Original '[string]$Role = ""' `
    -Mutated  '# Role param removed' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('[string]$Role'))
    }

Test-Mutation -Name "Remove role column from CSV header" `
    -FilePath $metricsPath `
    -Original ',role"' `
    -Mutated  '"' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains(',role"'))
    }

Test-Mutation -Name "Remove Role forwarding in Resolve-ClaudeResult" `
    -FilePath $claudePath `
    -Original 'Get-StoryRole' `
    -Mutated  'Classify-Role' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('Get-StoryRole'))
    }

# ============================================================================
# Phase 0: Get-SprintDiagnostics
# ============================================================================
Write-Host ""
Write-Host "--- Sprint Diagnostics ---" -ForegroundColor White

Test-Mutation -Name "Remove function Get-SprintDiagnostics" `
    -FilePath $metricsPath `
    -Original 'function Get-SprintDiagnostics' `
    -Mutated  'function Compute-DiagReport' `
    -TestBlock {
        param($mc)
        return (-not $mc.Contains('function Get-SprintDiagnostics'))
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
