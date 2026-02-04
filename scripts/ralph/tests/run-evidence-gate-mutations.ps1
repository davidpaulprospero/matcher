# Mutation testing for EvidenceGate
# Applies mutations IN MEMORY and runs source-inspection assertions against mutated strings
# Does not modify any files on disk.

$claudeFile = Join-Path $PSScriptRoot "..\lib\claude.ps1"
$metricsFile = Join-Path $PSScriptRoot "..\lib\metrics.ps1"
$configFile = Join-Path $PSScriptRoot "..\config\ralph-config.json"
$claudeOriginal = Get-Content $claudeFile -Raw
$metricsOriginal = Get-Content $metricsFile -Raw
$configOriginal = Get-Content $configFile -Raw

$mutations = @(
    @{
        Name = "M1: Change default threshold from 90 to 50"
        Find = "} else { 90 }"
        Replace = "} else { 50 }"
        Target = "claude"
        Test = {
            param($source)
            $source -match 'else\s*\{\s*90\s*\}'
        }
    },
    @{
        Name = "M2: Change -lt to -le (90% would fail instead of pass)"
        Find = 'percentage -lt $evidenceMinPct'
        Replace = 'percentage -le $evidenceMinPct'
        Target = "claude"
        Test = {
            param($source)
            $source -match 'percentage.*-lt.*evidenceMinPct'
        }
    },
    @{
        Name = "M3: Remove Update-StoryStatus rejection call"
        Find = 'Update-StoryStatus -StoryId $Ctx.StoryId -Passes $false -Notes "Evidence gate'
        Replace = '# REMOVED rejection call'
        Target = "claude"
        Test = {
            param($source)
            $source -match 'Update-StoryStatus\s+-StoryId.*-Passes\s+\$false.*Evidence gate'
        }
    },
    @{
        Name = "M4: Remove success=false on rejection (story accepted despite low evidence)"
        Find = '$success = $false
                $iterationStatus = "evidence_rejected"'
        Replace = '# REMOVED success flip'
        Target = "claude"
        Test = {
            param($source)
            $gateBlock = [regex]::Match($source, 'Evidence below threshold[\s\S]{0,500}?Append-SessionTimeline').Value
            $gateBlock -match '\$success\s*=\s*\$false' -and $gateBlock -match 'evidence_rejected'
        }
    },
    @{
        Name = "M5: Remove ConsecutiveFailures increment"
        Find = '$script:State.ConsecutiveFailures++
                Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $false'
        Replace = 'Append-SessionTimeline -Event "story_verified" -Data @{ storyId = $Ctx.StoryId; passed = $false'
        Target = "claude"
        Test = {
            param($source)
            $gateBlock = [regex]::Match($source, 'evidence_rejected[\s\S]{0,300}?Append-SessionTimeline').Value
            $gateBlock -match 'ConsecutiveFailures\+\+'
        }
    },
    @{
        Name = "M6: Remove $success guard on Update-TestBaseline"
        Find = 'if ($success -and $Ctx.TestResults) {'
        Replace = 'if ($Ctx.TestResults) {'
        Target = "claude"
        Test = {
            param($source)
            $source -match 'if\s*\(\$success\s+-and\s+\$Ctx\.TestResults\)'
        }
    },
    @{
        Name = "M7: Remove $success guard on Save-StoryProgress"
        Find = '            if ($success) {
                try {
                    Save-StoryProgress'
        Replace = '            try {
                    Save-StoryProgress'
        Target = "claude"
        Test = {
            param($source)
            $source -match 'if\s*\(\$success\)\s*\{[^}]*Save-StoryProgress'
        }
    },
    @{
        Name = "M8: Remove return statement from Log-StoryVerification"
        Find = '    return @{
        criteriaMet   = $criteriaMetCount
        criteriaTotal = $criteriaTotalCount'
        Replace = '    # REMOVED return'
        Target = "metrics"
        Test = {
            param($source)
            $funcPattern = 'function Log-StoryVerification\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
            $funcBody = [regex]::Match($source, $funcPattern).Value
            $funcBody -match 'return\s+@\{[^}]*criteriaMet'
        }
    },
    @{
        Name = "M9: Config threshold changed to 50"
        Find = '"evidenceThresholdPercent": 90'
        Replace = '"evidenceThresholdPercent": 50'
        Target = "config"
        Test = {
            param($configJson)
            $config = $configJson | ConvertFrom-Json
            $config.stallDetection.storyCompletionEarlyExit.evidenceThresholdPercent -eq 90
        }
    },
    @{
        Name = "M10: Remove evidenceThresholdPercent from config entirely"
        Find = ",`n            `"evidenceThresholdPercent`": 90"
        Replace = ""
        Target = "config"
        Test = {
            param($configJson)
            $config = $configJson | ConvertFrom-Json
            $null -ne $config.stallDetection.storyCompletionEarlyExit.evidenceThresholdPercent
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $targetSource = switch ($m.Target) {
        "claude" { $claudeOriginal }
        "metrics" { $metricsOriginal }
        "config" { $configOriginal }
    }

    $mutated = $targetSource.Replace($m.Find, $m.Replace)

    if ($mutated -eq $targetSource) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in target" }
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
Write-Host "=== EVIDENCE GATE MUTATION TESTING RESULTS ===" -ForegroundColor Cyan
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
