#Requires -Modules Pester

<#
.SYNOPSIS
    Mutation testing for race condition fix
.DESCRIPTION
    Applies mutations to claude.ps1 and verifies tests catch them.
    Each mutation should cause at least one test to fail.
#>

param(
    [switch]$Verbose
)

$RalphDir = Split-Path -Parent $PSScriptRoot
$ClaudeFile = Join-Path $RalphDir 'lib\claude.ps1'
$TestFile = Join-Path $PSScriptRoot 'ClaudeRaceCondition.Tests.ps1'

# Read original content
$original = Get-Content $ClaudeFile -Raw

# Define mutations to test
$mutations = @(
    @{
        Name = "Change 500ms to 200ms (post-cancel sleep)"
        Pattern = 'Start-Sleep -Milliseconds 500  # Give async event handlers'
        Replacement = 'Start-Sleep -Milliseconds 200  # Give async event handlers'
        ExpectedFailures = @('post-cancel sleep is exactly 500ms', 'no longer uses 200ms post-cancel')
    },
    @{
        Name = "Change 500ms to 100ms (post-cancel sleep)"
        Pattern = 'Start-Sleep -Milliseconds 500  # Give async event handlers'
        Replacement = 'Start-Sleep -Milliseconds 100  # Give async event handlers'
        ExpectedFailures = @('post-cancel sleep is exactly 500ms')
    },
    @{
        Name = "Change maxWait from 2000 to 1000"
        Pattern = '$maxWait = 2000'
        Replacement = '$maxWait = 1000'
        ExpectedFailures = @('buffer poll max wait is exactly 2000ms')
    },
    @{
        Name = "Change poll interval from 200ms to 100ms"
        Pattern = 'Start-Sleep -Milliseconds 200'
        Replacement = 'Start-Sleep -Milliseconds 100'
        ExpectedFailures = @('buffer poll interval is exactly 200ms', 'waited increment matches sleep duration')
    },
    @{
        Name = "Remove Console.Out.Flush()"
        Pattern = '[Console]::Out.Flush()  # Force immediate display'
        Replacement = '# [Console]::Out.Flush()  # Force immediate display (REMOVED)'
        ExpectedFailures = @('flushes console after success message', 'Console flush uses Out stream')
    },
    @{
        Name = "Change -eq to -ne in break condition"
        Pattern = 'if ($currentLen -eq $lastLen) { break }  # Buffer stable'
        Replacement = 'if ($currentLen -ne $lastLen) { break }  # Buffer stable'
        ExpectedFailures = @('break condition uses -eq')
    },
    @{
        Name = "Change -lt to -le in while loop"
        Pattern = 'while ($waited -lt $maxWait)'
        Replacement = 'while ($waited -le $maxWait)'
        ExpectedFailures = @('while loop uses -lt')
    },
    @{
        Name = "Use errBuilder instead of outBuilder"
        Pattern = '$lastLen = $outBuilder.Length'
        Replacement = '$lastLen = $errBuilder.Length'
        ExpectedFailures = @('uses outBuilder.Length not errBuilder.Length')
    }
)

$results = @()

foreach ($mutation in $mutations) {
    Write-Host "`n=====================================================" -ForegroundColor Cyan
    Write-Host "Testing mutation: $($mutation.Name)" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan

    # Apply mutation
    $mutated = $original -replace [regex]::Escape($mutation.Pattern), $mutation.Replacement

    if ($mutated -eq $original) {
        Write-Host "  [SKIP] Pattern not found in source - mutation may already be invalid" -ForegroundColor Yellow
        $results += @{
            Mutation = $mutation.Name
            Status = "SKIP"
            Reason = "Pattern not found"
        }
        continue
    }

    # Write mutated file
    $mutated | Set-Content $ClaudeFile -NoNewline

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
            # Check if expected tests failed
            $expectedCaught = $false
            foreach ($expected in $mutation.ExpectedFailures) {
                foreach ($failed in $failedTests) {
                    if ($failed -like "*$expected*") {
                        $expectedCaught = $true
                        break
                    }
                }
                if ($expectedCaught) { break }
            }

            if ($expectedCaught) {
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
            else {
                Write-Host "  [WARN] Tests failed but not the expected ones" -ForegroundColor Yellow
                Write-Host "         Expected: $($mutation.ExpectedFailures -join ', ')" -ForegroundColor Yellow
                Write-Host "         Got: $($failedTests -join ', ')" -ForegroundColor Yellow
                $results += @{
                    Mutation = $mutation.Name
                    Status = "KILLED_UNEXPECTED"
                    FailedTests = $failedTests
                }
            }
        }
    }
    finally {
        # Restore original
        $original | Set-Content $ClaudeFile -NoNewline
    }
}

# Summary
Write-Host "`n=====================================================" -ForegroundColor Cyan
Write-Host "MUTATION TESTING SUMMARY" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

$killed = @($results | Where-Object { $_.Status -eq "KILLED" -or $_.Status -eq "KILLED_UNEXPECTED" })
$survived = @($results | Where-Object { $_.Status -eq "SURVIVED" })
$skipped = @($results | Where-Object { $_.Status -eq "SKIP" })

Write-Host "Total mutations: $($mutations.Count)" -ForegroundColor White
Write-Host "Killed:          $($killed.Count)" -ForegroundColor Green
Write-Host "Survived:        $($survived.Count)" -ForegroundColor $(if ($survived.Count -gt 0) { "Red" } else { "Green" })
Write-Host "Skipped:         $($skipped.Count)" -ForegroundColor Yellow

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
