# Mutation testing for StoryCompletionEarlyExit
# Applies mutations IN MEMORY and runs source-inspection assertions against mutated strings
# Does not modify any files on disk.

$claudeFile = Join-Path $PSScriptRoot "..\lib\claude.ps1"
$configFile = Join-Path $PSScriptRoot "..\config\ralph-config.json"
$original = Get-Content $claudeFile -Raw
$configOriginal = Get-Content $configFile -Raw

$funcPattern = 'function Invoke-ClaudeSubprocess\s*\{([\s\S]*?)^\}'

function Get-FuncBody($source) {
    [regex]::Match($source, $funcPattern, 'Multiline').Groups[1].Value
}

$mutations = @(
    @{
        Name = "M1: Change grace period default from 15 to 30"
        Find = "} else { 15 }"
        Replace = "} else { 30 }"
        Target = "source"
        Test = {
            param($funcBody)
            # Should fail: default is no longer 15
            $funcBody -match 'else\s*\{\s*15\s*\}' -and
            $funcBody -notmatch 'else\s*\{\s*30\s*\}'
        }
    },
    @{
        Name = "M2: Change enabled default from true to false"
        Find = '} else { $true }'
        Replace = '} else { $false }'
        Target = "source"
        Test = {
            param($funcBody)
            $enabledLine = ($funcBody -split "`n" | Where-Object { $_ -match 'earlyExitEnabled' -and $_ -match 'else' })
            ($enabledLine -match '\$true') -and ($enabledLine -notmatch '\$false')
        }
    },
    @{
        Name = "M3: Check passes -eq false instead of true"
        Find = '$story.passes -eq $true'
        Replace = '$story.passes -eq $false'
        Target = "source"
        Test = {
            param($funcBody)
            $passesLine = ($funcBody -split "`n" | Where-Object { $_ -match 'story\.passes' })
            ($passesLine -match '-eq\s+\$true') -and ($passesLine -notmatch '-eq\s+\$false')
        }
    },
    @{
        Name = "M4: Change -ge to -gt in grace period comparison"
        Find = '$sinceCompletion -ge $storyCompletionGraceSec'
        Replace = '$sinceCompletion -gt $storyCompletionGraceSec'
        Target = "source"
        Test = {
            param($funcBody)
            $compareLine = ($funcBody -split "`n" | Where-Object { $_ -match 'sinceCompletion' -and $_ -match 'storyCompletionGraceSec' })
            ($compareLine -match '-ge') -and ($compareLine -notmatch '-gt\s')
        }
    },
    @{
        Name = "M5: Remove StoryId from passthrough"
        Find = '-StoryId $StoryId'
        Replace = ''
        Target = "fullsource"
        Test = {
            param($source)
            $source -match 'Invoke-ClaudeSubprocess\s[^}]*-StoryId\s+\$StoryId'
        }
    },
    @{
        Name = "M6: Remove earlyExitEnabled guard"
        Find = '$earlyExitEnabled -and $prdUpdated -and $StoryId'
        Replace = '$prdUpdated -and $StoryId'
        Target = "source"
        Test = {
            param($funcBody)
            $guardLine = ($funcBody -split "`n" | Where-Object { $_ -match 'earlyExitEnabled.*prdUpdated.*StoryId' })
            $null -ne $guardLine -and $guardLine -ne ''
        }
    },
    @{
        Name = "M7: Remove try/catch around PRD read (use if instead)"
        Find = '                    try {'
        Replace = '                    if ($true) {'
        Target = "source"
        Test = {
            param($funcBody)
            $tryPattern = 'try\s*\{[^}]*Get-Content\s+\$script:PrdFile'
            $funcBody -match $tryPattern
        }
    },
    @{
        Name = "M8: Remove /T flag from taskkill (no tree kill)"
        Find = 'taskkill /T /F /PID $process.Id'
        Replace = 'taskkill /F /PID $process.Id'
        Target = "source"
        Test = {
            param($funcBody)
            $graceBlock = [regex]::Match($funcBody, 'Grace period expired[\s\S]{0,500}?break').Value
            $graceBlock -match 'taskkill /T /F /PID'
        }
    },
    @{
        Name = "M9: Config grace period changed to 60"
        Find = '"gracePeriodSeconds": 15'
        Replace = '"gracePeriodSeconds": 60'
        Target = "config"
        Test = {
            param($configJson)
            $config = $configJson | ConvertFrom-Json
            $config.stallDetection.storyCompletionEarlyExit.gracePeriodSeconds -eq 15
        }
    },
    @{
        Name = "M10: Config enabled changed to false"
        Find = '"enabled": true'
        Replace = '"enabled": false'
        Target = "config_early_exit"
        Test = {
            param($configJson)
            $config = $configJson | ConvertFrom-Json
            $config.stallDetection.storyCompletionEarlyExit.enabled -eq $true
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $targetSource = if ($m.Target -eq "config" -or $m.Target -eq "config_early_exit") { $configOriginal } else { $original }
    $mutated = $targetSource.Replace($m.Find, $m.Replace)

    if ($mutated -eq $targetSource) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in target" }
        continue
    }

    # Run assertion against mutated content
    $testResult = $false
    switch ($m.Target) {
        "source" {
            $funcBody = Get-FuncBody $mutated
            $testResult = & $m.Test $funcBody
        }
        "fullsource" {
            $testResult = & $m.Test $mutated
        }
        "config" {
            $testResult = & $m.Test $mutated
        }
        "config_early_exit" {
            # Need to only change in the storyCompletionEarlyExit section
            # The simple replace might change other "enabled": true - let's handle specifically
            $section = [regex]::Match($targetSource, '"storyCompletionEarlyExit":\s*\{[^}]+\}').Value
            $mutatedSection = $section.Replace($m.Find, $m.Replace)
            $mutatedConfig = $targetSource.Replace($section, $mutatedSection)
            $testResult = & $m.Test $mutatedConfig
        }
    }

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== MUTATION TESTING RESULTS ===" -ForegroundColor Cyan
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
