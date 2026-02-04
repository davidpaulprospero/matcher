# Mutation testing for pipeline leak suppression in Resolve-ClaudeResult
# Applies mutations IN MEMORY and runs assertions against mutated strings
# Does not modify any files on disk.

$claudeFile = Join-Path $PSScriptRoot "..\lib\claude.ps1"
$original = Get-Content $claudeFile -Raw

# Extract Resolve-ClaudeResult function body
$resolvePattern = 'function Resolve-ClaudeResult\s*\{([\s\S]*?)(?=\nfunction\s)'
function Get-ResolveBody($source) {
    [regex]::Match($source, $resolvePattern).Value
}

$mutations = @(
    @{
        Name = "M1: Remove $null from timeout path Log-StoryVerification"
        Find = '$null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false'
        Replace = 'Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false'
        Test = {
            param($resolveBody)
            # All Log-StoryVerification calls must be captured or suppressed
            $calls = [regex]::Matches($resolveBody, '(?m)^.*Log-StoryVerification.*$')
            $allSuppressed = $true
            foreach ($call in $calls) {
                if ($call.Value.Trim() -notmatch '(\$null|\$\w+)\s*=\s*Log-StoryVerification') {
                    $allSuppressed = $false
                }
            }
            $allSuppressed
        }
    },
    @{
        Name = "M2: Replace $null with void cast on timeout path"
        Find = '$null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false'
        Replace = '[void](Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false)'
        Test = {
            param($resolveBody)
            # Verify $null = pattern specifically (not [void] cast)
            $timeoutBlock = [regex]::Match($resolveBody, 'TIMEOUT[\s\S]*?(?=elseif.*ExitCode\s+-eq\s+0)').Value
            $timeoutBlock -match '\$null\s*=\s*Log-StoryVerification'
        }
    },
    @{
        Name = "M3: Remove entire timeout Log-StoryVerification call"
        Find = 'if ($Ctx.StoryObj) { $null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }'
        Replace = '# REMOVED timeout verification'
        Test = {
            param($resolveBody)
            $timeoutBlock = [regex]::Match($resolveBody, 'TIMEOUT[\s\S]*?(?=elseif.*ExitCode\s+-eq\s+0)').Value
            $timeoutBlock -match 'Log-StoryVerification'
        }
    },
    @{
        Name = "M4: Remove entire failure Log-StoryVerification call"
        Find = 'if ($Ctx.StoryObj) { $null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false }'
        Replace = '# REMOVED failure verification'
        Test = {
            param($resolveBody)
            $failureBlock = [regex]::Match($resolveBody, 'FAILURE[\s\S]*?Update learning database').Value
            $failureBlock -match 'Log-StoryVerification'
        }
    },
    @{
        Name = "M5: Change -Passed `$false to -Passed `$true on failure path"
        Find = '$null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $false'
        Replace = '$null = Log-StoryVerification -StoryId $Ctx.StoryId -Story $Ctx.StoryObj -Iteration $script:State.IterationCount -Passed $true'
        Test = {
            param($resolveBody)
            # Timeout and failure paths should have -Passed $false, not $true
            $failureBlock = [regex]::Match($resolveBody, 'FAILURE[\s\S]*?Update learning database').Value
            $failureBlock -match 'Passed \$false'
        }
    },
    @{
        Name = "M6: Remove $evidenceResult capture on success path"
        Find = '$evidenceResult = Log-StoryVerification'
        Replace = 'Log-StoryVerification'
        Test = {
            param($resolveBody)
            $resolveBody -match '\$evidenceResult\s*=\s*Log-StoryVerification'
        }
    }
)

$results = @()
foreach ($m in $mutations) {
    $mutated = $original.Replace($m.Find, $m.Replace)

    if ($mutated -eq $original) {
        $results += @{ Name = $m.Name; Status = "SKIP"; Detail = "pattern not found in source" }
        continue
    }

    $resolveBody = Get-ResolveBody $mutated
    $testResult = & $m.Test $resolveBody

    if (-not $testResult) {
        $results += @{ Name = $m.Name; Status = "KILLED"; Detail = "assertion correctly failed on mutated code" }
    } else {
        $results += @{ Name = $m.Name; Status = "SURVIVED"; Detail = "assertion still passed on mutated code!" }
    }
}

Write-Host ""
Write-Host "=== MUTATION TESTING: Pipeline Leak Suppression ===" -ForegroundColor Cyan
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
