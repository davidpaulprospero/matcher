# Mutation testing for [bool] cast pipeline leak prevention
# Applies mutations IN MEMORY and runs assertions against mutated strings
# Does not modify any files on disk.

$ralphDir = Split-Path $PSScriptRoot -Parent
$loopsFile = Join-Path $ralphDir 'lib\loops.ps1'
$claudeFile = Join-Path $ralphDir 'lib\claude.ps1'
$ralphFile = Join-Path $ralphDir 'ralph.ps1'

$loopsOrig = Get-Content $loopsFile -Raw
$claudeOrig = Get-Content $claudeFile -Raw
$ralphOrig = Get-Content $ralphFile -Raw

$mutations = @(
    # --- loops.ps1 mutations: Remove [bool] cast ---
    @{
        Name = "M1: Remove [bool] cast from first Invoke-ClaudeForStory in loops.ps1"
        Source = "loops"
        Find = '[bool](Invoke-ClaudeForStory -StoryId $status.nextStory.id | Select-Object -Last 1)'
        Replace = 'Invoke-ClaudeForStory -StoryId $status.nextStory.id'
        Test = {
            param($src)
            # All $success assignments from Invoke-ClaudeForStory must have [bool]
            $assignments = [regex]::Matches($src, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeForStory.*$')
            $allCast = $true
            foreach ($m in $assignments) { if ($m.Value -notmatch '\[bool\]') { $allCast = $false } }
            $allCast
        }
    },
    @{
        Name = "M2: Remove Select-Object -Last 1 from Invoke-ClaudeForFocusArea in loops.ps1"
        Source = "loops"
        Find = '[bool](Invoke-ClaudeForFocusArea -FocusAreaId $selectedArea -Context $context -GeneratePRD | Select-Object -Last 1)'
        Replace = '[bool](Invoke-ClaudeForFocusArea -FocusAreaId $selectedArea -Context $context -GeneratePRD)'
        Test = {
            param($src)
            $assignments = [regex]::Matches($src, '(?m)^\s*\$prdGenerated\s*=\s*.*Invoke-ClaudeForFocusArea.*$')
            $allPiped = $true
            foreach ($m in $assignments) { if ($m.Value -notmatch 'Select-Object\s+-Last\s+1') { $allPiped = $false } }
            $allPiped
        }
    },
    # --- claude.ps1 mutations: Remove $null suppression ---
    @{
        Name = "M3: Remove $null from Update-TestBaseline"
        Source = "claude"
        Find = '$null = Update-TestBaseline -TestResults $Ctx.TestResults'
        Replace = 'Update-TestBaseline -TestResults $Ctx.TestResults'
        Test = {
            param($src)
            $resolveBody = [regex]::Match($src, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
            $resolveBody -match '\$null\s*=\s*Update-TestBaseline'
        }
    },
    @{
        Name = "M4: Remove $null from Save-StoryProgress"
        Source = "claude"
        Find = '$null = Save-StoryProgress'
        Replace = 'Save-StoryProgress'
        Test = {
            param($src)
            $resolveBody = [regex]::Match($src, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
            $resolveBody -match '\$null\s*=\s*Save-StoryProgress'
        }
    },
    @{
        Name = "M5: Remove $null from first Update-LearningDb (success path)"
        Source = "claude"
        Find = '$null = Update-LearningDb -Entry @{
                    type = "story_success"'
        Replace = 'Update-LearningDb -Entry @{
                    type = "story_success"'
        Test = {
            param($src)
            $resolveBody = [regex]::Match($src, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
            $calls = [regex]::Matches($resolveBody, '(?m)^.*Update-LearningDb.*$')
            $allSuppressed = $true
            foreach ($c in $calls) {
                $line = $c.Value.Trim()
                if ($line -match '^\s*#') { continue }
                if ($line -notmatch '\$null\s*=\s*Update-LearningDb') { $allSuppressed = $false }
            }
            $allSuppressed
        }
    },
    @{
        Name = "M6: Remove $null from failure path Update-LearningDb"
        Source = "claude"
        Find = '$null = Update-LearningDb -Entry @{
                    type = "story_failure"'
        Replace = 'Update-LearningDb -Entry @{
                    type = "story_failure"'
        Test = {
            param($src)
            $resolveBody = [regex]::Match($src, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
            $calls = [regex]::Matches($resolveBody, '(?m)^.*Update-LearningDb.*$')
            $allSuppressed = $true
            foreach ($c in $calls) {
                $line = $c.Value.Trim()
                if ($line -match '^\s*#') { continue }
                if ($line -notmatch '\$null\s*=\s*Update-LearningDb') { $allSuppressed = $false }
            }
            $allSuppressed
        }
    },
    # --- ralph.ps1 mutation: Remove [bool] cast from Invoke-ClaudeProcess ---
    @{
        Name = "M7: Remove [bool] cast from Invoke-ClaudeProcess in ralph.ps1"
        Source = "ralph"
        Find = '[bool](Invoke-ClaudeProcess -Prompt $prompt -PromptType "story_work" -Identifier $StoryId -FocusArea $focusArea -StoryObj $storyObj | Select-Object -Last 1)'
        Replace = 'Invoke-ClaudeProcess -Prompt $prompt -PromptType "story_work" -Identifier $StoryId -FocusArea $focusArea -StoryObj $storyObj'
        Test = {
            param($src)
            $funcBody = [regex]::Match($src, 'function Invoke-ClaudeForStory[\s\S]*?(?=\nfunction\s|\z)').Value
            $processCall = [regex]::Match($funcBody, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeProcess.*$').Value
            $processCall -match '\[bool\]'
        }
    },
    # --- claude.ps1 mutation: Remove [bool] cast from Invoke-ClaudeForFocusArea ---
    @{
        Name = "M8: Remove [bool] cast from Invoke-ClaudeProcess in claude.ps1"
        Source = "claude"
        Find = '[bool](Invoke-ClaudeProcess -Prompt $prompt -PromptType $promptType -Identifier $FocusAreaId -AllowedTools:$useTools | Select-Object -Last 1)'
        Replace = 'Invoke-ClaudeProcess -Prompt $prompt -PromptType $promptType -Identifier $FocusAreaId -AllowedTools:$useTools'
        Test = {
            param($src)
            $funcBody = [regex]::Match($src, 'function Invoke-ClaudeForFocusArea[\s\S]*?(?=\nfunction\s|\z)').Value
            $processCall = [regex]::Match($funcBody, '(?m)^\s*\$result\s*=\s*.*Invoke-ClaudeProcess.*$').Value
            $processCall -match '\[bool\]'
        }
    },
    # --- Logic mutation: Change -Last 1 to -First 1 ---
    @{
        Name = "M9: Change Select-Object -Last 1 to -First 1 (would pick leaked value instead of bool)"
        Source = "loops"
        Find = 'Select-Object -Last 1)'
        Replace = 'Select-Object -First 1)'
        Test = {
            param($src)
            $assignments = [regex]::Matches($src, '(?m)^\s*\$success\s*=\s*.*Invoke-ClaudeForStory.*$')
            $allLast = $true
            foreach ($m in $assignments) { if ($m.Value -notmatch 'Select-Object\s+-Last\s+1') { $allLast = $false } }
            $allLast
        }
    },
    # --- Inversion mutation: Remove Out-Null from Get-SprintTokenBudget ---
    @{
        Name = "M10: Remove Out-Null from Get-SprintTokenBudget"
        Source = "claude"
        Find = 'Get-SprintTokenBudget | Out-Null'
        Replace = 'Get-SprintTokenBudget'
        Test = {
            param($src)
            $resolveBody = [regex]::Match($src, 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)').Value
            $resolveBody -match 'Get-SprintTokenBudget\s*\|\s*Out-Null'
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $orig = switch ($m.Source) {
        "loops" { $loopsOrig }
        "claude" { $claudeOrig }
        "ralph" { $ralphOrig }
    }

    $mutated = $orig.Replace($m.Find, $m.Replace)

    if ($mutated -eq $orig) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in source" }
        continue
    }

    $testResult = & $m.Test $mutated

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== MUTATION TESTING: Bool Cast Pipeline Leak ===" -ForegroundColor Cyan
Write-Host ""
foreach ($r in $results) {
    $color = switch ($r.Status) {
        "KILLED" { "Green" }
        "SURVIVED" { "Red" }
        default { "Yellow" }
    }
    Write-Host "$($r.Name): $($r.Status)" -ForegroundColor $color
    if ($r.Detail) { Write-Host "  -> $($r.Detail)" -ForegroundColor DarkGray }
}

$killed = @($results | Where-Object { $_.Status -eq "KILLED" }).Count
$survived = @($results | Where-Object { $_.Status -eq "SURVIVED" }).Count
$skipped = @($results | Where-Object { $_.Status -eq "SKIP" }).Count
$total = $killed + $survived
Write-Host ""
if ($total -gt 0) {
    $pct = [math]::Round($killed / $total * 100)
    $color = if ($survived -eq 0) { "Green" } else { "Red" }
    Write-Host "Score: $killed/$total mutations killed ($pct%) | $skipped skipped" -ForegroundColor $color
} else {
    Write-Host "No mutations applied" -ForegroundColor Yellow
}
