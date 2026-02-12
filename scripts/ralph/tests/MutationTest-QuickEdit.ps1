#Requires -Modules Pester

<#
.SYNOPSIS
    Mutation testing for Quick Edit Mode disable fix.
.DESCRIPTION
    Applies mutations to display.ps1 and ralph.ps1, then verifies the
    QuickEdit.Tests.ps1 tests catch them. Each mutation should cause at
    least one test to fail, proving the tests are effective.
#>

param(
    [switch]$Verbose
)

$RalphDir = Split-Path -Parent $PSScriptRoot
$DisplayFile = Join-Path $RalphDir 'lib\display.ps1'
$ClaudeFile = Join-Path $RalphDir 'lib\claude.ps1'
$RalphFile = Join-Path $RalphDir 'ralph.ps1'
$TestFile = Join-Path $PSScriptRoot 'QuickEdit.Tests.ps1'

# Read originals
$displayOriginal = Get-Content $DisplayFile -Raw
$claudeOriginal = Get-Content $ClaudeFile -Raw
$ralphOriginal = Get-Content $RalphFile -Raw

# Define mutations
$mutations = @(
    # --- display.ps1 mutations ---
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Wrong constant: ENABLE_QUICK_EDIT = 0x0040 -> 0x0004"
        Pattern = '$ENABLE_QUICK_EDIT = 0x0040'
        Replacement = '$ENABLE_QUICK_EDIT = 0x0004'
        ExpectedFailures = @('ENABLE_QUICK_EDIT.*0x0040')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Wrong handle: STD_INPUT_HANDLE = -10 -> -11 (stdout)"
        Pattern = '$STD_INPUT_HANDLE = -10'
        Replacement = '$STD_INPUT_HANDLE = -11'
        ExpectedFailures = @('STD_INPUT_HANDLE.*-10')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Use -bor instead of -band (set bit instead of clear)"
        Pattern = '$newMode = $mode -band (-bnot $ENABLE_QUICK_EDIT)'
        Replacement = '$newMode = $mode -bor $ENABLE_QUICK_EDIT'
        ExpectedFailures = @('-band.*-bnot')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Remove OS check (function runs on non-Windows)"
        Pattern = "if (`$env:OS -ne 'Windows_NT') { return `$false }"
        Replacement = "# OS check removed"
        ExpectedFailures = @('OS.*Windows_NT')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Return wrong boolean on already-disabled"
        Pattern = '# Quick Edit was already disabled
            return $true'
        Replacement = '# Quick Edit was already disabled
            return $false'
        ExpectedFailures = @('already disabled.*true')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Skip GetConsoleMode check (always proceed)"
        Pattern = 'if (-not $gotMode) { return $false }'
        Replacement = '# if (-not $gotMode) { return $false }'
        ExpectedFailures = @('GetConsoleMode.*fail.*false')
    },
    @{
        File = "display.ps1"
        FilePath = $DisplayFile
        Original = $displayOriginal
        Name = "Remove entire function body (return nothing)"
        Pattern = 'function Disable-QuickEditMode {'
        Replacement = 'function Disable-QuickEditMode { return $null
function _Disable-QuickEditMode_Dead {'
        ExpectedFailures = @('Returns a boolean')
    },
    # --- ralph.ps1 mutations ---
    @{
        File = "ralph.ps1"
        FilePath = $RalphFile
        Original = $ralphOriginal
        Name = "Remove Disable-QuickEditMode call from startup"
        Pattern = '$quickEditDisabled = Disable-QuickEditMode'
        Replacement = '$quickEditDisabled = $false  # MUTANT: call removed'
        ExpectedFailures = @('Disable-QuickEditMode.*ralph\.ps1')
    },
    # --- claude.ps1 periodic safeguard mutations ---
    @{
        File = "claude.ps1"
        FilePath = $ClaudeFile
        Original = $claudeOriginal
        Name = "Change quickEditRecheckIntervalSec from 60 to 5 seconds"
        Pattern = 'quickEditRecheckIntervalSec = 60'
        Replacement = 'quickEditRecheckIntervalSec = 5'
        ExpectedFailures = @('quickEditRecheckIntervalSec.*60')
    },
    @{
        File = "claude.ps1"
        FilePath = $ClaudeFile
        Original = $claudeOriginal
        Name = "Remove try block (will crash on console error)"
        Pattern = 'try \{'
        Replacement = '# try removed'
        ExpectedFailures = @('try.*catch')
    },
    @{
        File = "claude.ps1"
        FilePath = $ClaudeFile
        Original = $claudeOriginal
        Name = "Change interval check from ge to gt (never triggers)"
        Pattern = '-ge $quickEditRecheckIntervalSec'
        Replacement = '-gt $quickEditRecheckIntervalSec'
        ExpectedFailures = @('quickEditRecheckIntervalSec')
    },
    @{
        File = "claude.ps1"
        FilePath = $ClaudeFile
        Name = "Add Write-Host output in safeguard (verbose mode)"
        Pattern = '# Silent re-disable'
        Replacement = 'Write-Host "QuickEdit re-check"'
        ExpectedFailures = @('Does not output messages')
    }
)

$results = @()

foreach ($mutation in $mutations) {
    Write-Host "`n=====================================================" -ForegroundColor Cyan
    Write-Host "Testing mutation: $($mutation.Name)" -ForegroundColor Cyan
    Write-Host "  File: $($mutation.File)" -ForegroundColor DarkGray
    Write-Host "=====================================================" -ForegroundColor Cyan

    # Apply mutation
    $mutated = $mutation.Original -replace [regex]::Escape($mutation.Pattern), $mutation.Replacement

    if ($mutated -eq $mutation.Original) {
        Write-Host "  [SKIP] Pattern not found in source" -ForegroundColor Yellow
        $results += @{
            Mutation = $mutation.Name
            Status = "SKIP"
            Reason = "Pattern not found"
        }
        continue
    }

    # Write mutated file
    $mutated | Set-Content $mutation.FilePath -NoNewline

    try {
        # Run tests (expecting failures)
        $pesterResult = Invoke-Pester -Path $TestFile -PassThru -Output None

        $failedTests = @($pesterResult.Failed | ForEach-Object { $_.Name })

        if ($failedTests.Count -eq 0) {
            Write-Host "  [FAIL] Mutation SURVIVED - no tests caught it!" -ForegroundColor Red
            $results += @{
                Mutation = $mutation.Name
                Status = "SURVIVED"
                Reason = "No tests failed"
            }
        }
        else {
            Write-Host "  [PASS] Mutation KILLED by $($failedTests.Count) test(s)" -ForegroundColor Green
            if ($Verbose) {
                foreach ($f in $failedTests) {
                    Write-Host "         - $f" -ForegroundColor DarkGreen
                }
            }
            $results += @{
                Mutation = $mutation.Name
                Status = "KILLED"
                FailedTests = $failedTests.Count
            }
        }
    }
    finally {
        # Restore original
        $mutation.Original | Set-Content $mutation.FilePath -NoNewline
    }
}

# Summary
Write-Host "`n=====================================================" -ForegroundColor Cyan
Write-Host "MUTATION TESTING SUMMARY" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

$killed = @($results | Where-Object { $_.Status -eq "KILLED" })
$survived = @($results | Where-Object { $_.Status -eq "SURVIVED" })
$skipped = @($results | Where-Object { $_.Status -eq "SKIP" })

Write-Host "Total mutations: $($mutations.Count)" -ForegroundColor White
Write-Host "Killed:          $($killed.Count)" -ForegroundColor Green
Write-Host "Survived:        $($survived.Count)" -ForegroundColor $(if ($survived.Count -gt 0) { "Red" } else { "Green" })
Write-Host "Skipped:         $($skipped.Count)" -ForegroundColor Yellow

$score = if (($killed.Count + $survived.Count) -gt 0) {
    [math]::Round($killed.Count / ($killed.Count + $survived.Count) * 100)
} else { 0 }
Write-Host "Score:           $score%" -ForegroundColor $(if ($score -ge 80) { "Green" } elseif ($score -ge 50) { "Yellow" } else { "Red" })

if ($survived.Count -gt 0) {
    Write-Host "`nSurviving mutations (TEST GAPS):" -ForegroundColor Red
    foreach ($s in $survived) {
        Write-Host "  - $($s.Mutation)" -ForegroundColor Red
    }
    exit 1
}
else {
    Write-Host "`nAll mutations were killed - tests are effective!" -ForegroundColor Green
    exit 0
}
