# Mutation testing for LLM-based criteria evidence verification
# Applies mutations IN MEMORY and runs source-inspection assertions against mutated strings
# Does not modify any files on disk.

$metricsFile = Join-Path $PSScriptRoot "..\lib\metrics.ps1"
$original = Get-Content $metricsFile -Raw

# Extract Confirm-CriteriaEvidence function body
$confirmPattern = 'function Confirm-CriteriaEvidence\s*\{([\s\S]*?)(?=\nfunction\s)'
function Get-ConfirmBody($source) {
    [regex]::Match($source, $confirmPattern).Value
}

# Extract Log-StoryVerification function body
$logVerifyPattern = 'function Log-StoryVerification\s*\{([\s\S]*?)(?=\nfunction\s|\z)'
function Get-LogVerifyBody($source) {
    [regex]::Match($source, $logVerifyPattern).Value
}

$mutations = @(
    @{
        Name = "M1: Change model from haiku to sonnet"
        Find = "--model haiku"
        Replace = "--model sonnet"
        Target = "confirm"
        Test = {
            param($funcBody)
            $funcBody -match '--model haiku' -and $funcBody -notmatch '--model sonnet'
        }
    },
    @{
        Name = "M2: Change truncation limit from 8000 to 4000"
        Find = "-gt 8000"
        Replace = "-gt 4000"
        Target = "confirm"
        Test = {
            param($funcBody)
            $funcBody -match '-gt 8000' -and $funcBody -notmatch '-gt 4000'
        }
    },
    @{
        Name = "M3: Remove null fallback return on Claude unavailable"
        Find = "return `$null  # Signal caller to use fallback`r`n    }`r`n`r`n    # Build numbered criteria list"
        Replace = "return @()  # Return empty instead`r`n    }`r`n`r`n    # Build numbered criteria list"
        Target = "confirm"
        Test = {
            param($funcBody)
            # The function must return $null (not @()) when Claude is unavailable
            $claudeBlock = [regex]::Match($funcBody, 'if \(-not \$claudePath\)[\s\S]{0,300}').Value
            $claudeBlock -match 'return \$null'
        }
    },
    @{
        Name = "M4: Change timeout from 60 to 120 seconds"
        Find = "AddSeconds(60)"
        Replace = "AddSeconds(120)"
        Target = "confirm"
        Test = {
            param($funcBody)
            $funcBody -match 'AddSeconds\(60\)' -and $funcBody -notmatch 'AddSeconds\(120\)'
        }
    },
    @{
        Name = "M5: Swap MET confidence from 0.9 to 0.5"
        Find = 'evidence = "LLM: MET"; confidence = 0.9'
        Replace = 'evidence = "LLM: MET"; confidence = 0.5'
        Target = "confirm"
        Test = {
            param($funcBody)
            $metLine = ($funcBody -split "`n" | Where-Object { $_ -match 'evidence = "LLM: MET"' -and $_ -match 'confidence' })
            $metLine -match '0\.9' -and $metLine -notmatch '0\.5'
        }
    },
    @{
        Name = "M6: Remove NOT_MET parsing (only MET recognized)"
        Find = "'^NOT_MET\s+(\d+)'"
        Replace = "'^NEVER_MATCH_THIS\s+(\d+)'"
        Target = "confirm"
        Test = {
            param($funcBody)
            $funcBody -match "\'\^NOT_MET\\s\+\(\\d\+\)\'"
        }
    },
    @{
        Name = "M7: Invert met=true to met=false in MET parse"
        Find = '@{ index = $num; met = $true; evidence = "LLM: MET"'
        Replace = '@{ index = $num; met = $false; evidence = "LLM: MET"'
        Target = "confirm"
        Test = {
            param($funcBody)
            $metBlock = [regex]::Match($funcBody, "elseif.*'\^MET[\s\S]{0,200}").Value
            $metBlock -match 'met = \$true'
        }
    },
    @{
        Name = "M8: Remove Confirm-CriteriaEvidence call from Log-StoryVerification"
        Find = "Confirm-CriteriaEvidence -Criteria"
        Replace = "# REMOVED LLM call"
        Target = "logverify"
        Test = {
            param($funcBody)
            $funcBody -match 'Confirm-CriteriaEvidence'
        }
    },
    @{
        Name = "M9: Remove Search-CriterionEvidence fallback from Log-StoryVerification"
        Find = "Search-CriterionEvidence -Criterion"
        Replace = "# REMOVED fallback"
        Target = "logverify"
        Test = {
            param($funcBody)
            # Must match actual function call, not just docstring mentions
            $funcBody -match 'Search-CriterionEvidence\s+-Criterion'
        }
    },
    @{
        Name = "M10: Remove Get-ClaudePath check (skip availability test)"
        Find = '$claudePath = Get-ClaudePath'
        Replace = '$claudePath = "always-available"'
        Target = "confirm"
        Test = {
            param($funcBody)
            $funcBody -match 'Get-ClaudePath' -and $funcBody -notmatch '"always-available"'
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $mutated = $original.Replace($m.Find, $m.Replace)

    if ($mutated -eq $original) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in target" }
        continue
    }

    # Run assertion against mutated content
    $testResult = $false
    switch ($m.Target) {
        "confirm" {
            $funcBody = Get-ConfirmBody $mutated
            $testResult = & $m.Test $funcBody
        }
        "logverify" {
            $funcBody = Get-LogVerifyBody $mutated
            $testResult = & $m.Test $funcBody
        }
    }

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== MUTATION TESTING: LLM Evidence Verification ===" -ForegroundColor Cyan
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
